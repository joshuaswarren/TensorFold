//! Private probe for the R-output re-batching of qmm_q4_pk: qmm_q4_pk_r.comp
//! at OUT_PER_WG = 2, 4, 8 carries R outputs per 64-lane workgroup through the
//! SAME per-output arithmetic with the barriers shared, so every output word
//! must be bit-identical to the one-output kernel. Per served shape this probe
//! checks the R=1 kernel against the pk contract's f64 bound, checks every R
//! variant's full output BIT-EXACT against R=1's (a re-batching, not a
//! rewrite — any difference is a bug), and only then times R = 1, 2, 4, 8 at
//! N = 0, 1, 4, 8 dependent dispatches, printing per-dispatch p50, effective
//! weight GB/s and the ratio to R=1.
const std = @import("std");
const vk = @import("vulkan");
const contract = @import("contract");
const shaders = @import("multi_shaders");
const qf = @import("qf");
const c = contract.spec;
const dispatch = vk.dispatch;

const usage = "usage: tf-qmm-multi [--vendor ID] [--timeout-ms MS] [--add-floor-us US] | --help\n" ++
    "  --help            show this text and exit without opening a loader or device\n" ++
    "  --vendor ID       refuse a device whose Vulkan vendorID differs (decimal or 0x hex; no default)\n" ++
    "  --timeout-ms MS   bound on every fence wait, 1 to 600000 (default 10000); a timeout ends the process\n" ++
    "  --add-floor-us US this chip's own measured per-dispatch floor from tf-dispatch-floor; only then is a ratio printed\n";

const warmups = 3;
const sample_count = 63;
const staging_bytes = 1 << 20;
const variants = [_]u32{ 1, 2, 4, 8 };
const deps = [_]usize{ 0, 1, 4, 8 };

comptime {
    if (variants[0] != 1 or deps[0] != 0 or deps[deps.len - 1] != 8) @compileError("the variant/depth tables drifted: R=1 and the N=0 floor anchor the ratios");
}

/// ceil(a/b), refusing a zero divisor. Pure and host-tested.
pub fn ceilDiv(a: u64, b: u64) error{ZeroDiv}!u64 {
    if (b == 0) return error.ZeroDiv;
    return (a + b - 1) / b;
}

/// Workgroups for one R-output dispatch: ceil(total/R), capped at the device's
/// maxComputeWorkGroupCount[0] exactly as ops_q4.pkGroups caps the R=1 grid
/// (the kernel grid-strides the rest, so the cap never drops work). Zero
/// outputs or a zero R are refused, not dispatched as zero groups. Pure and
/// host-tested.
pub fn wgGroups(total: u64, out_per_wg: u32, max_groups: u32) error{ Invalid, ZeroDiv }!u32 {
    if (total == 0 or out_per_wg == 0) return error.Invalid;
    const groups = try ceilDiv(total, out_per_wg);
    return @intCast(@min(groups, @as(u64, max_groups)));
}

pub const WordDiff = struct { index: usize, got: u16, want: u16 };

/// The first differing output word of two same-length bf16-word slices, with
/// both words. Null when equal. Pure and host-tested: the bit-exact gate's
/// FAIL line names the index and both words.
pub fn firstDiff16(a: []const u16, b: []const u16) ?WordDiff {
    if (a.len != b.len) return .{ .index = 0, .got = if (a.len == 0) 0 else a[0], .want = if (b.len == 0) 0 else b[0] };
    for (a, b, 0..) |x, y, i| if (x != y) return .{ .index = i, .got = x, .want = y };
    return null;
}

fn checkSpec() c.Spec {
    return qf.pkSpec();
}

fn us(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1000.0;
}

fn printStats(n: usize, stats: qf.Stats) void {
    std.debug.print("    N={d: >3}: min={d:.1} us p50={d:.1} us p95={d:.1} us max={d:.1} us\n", .{
        n, us(stats.min_ns), us(stats.p50_ns), us(stats.p95_ns), us(stats.max_ns),
    });
}

fn sampleBatch(io: std.Io, cmds: *vk.Commands, staging: vk.Buffer, pipe: *dispatch.Pipeline, bindings: []const dispatch.Binding, push: []const u8, groups: [3]u32, n: usize, timeout: u64) !u64 {
    try vk.orExit(cmds.beginBatch(staging));
    errdefer cmds.abortBatch();
    const t0 = std.Io.Clock.awake.now(io);
    for (0..n) |_| {
        try vk.orExit(pipe.bind(bindings));
        try vk.orExit(pipe.record(try cmds.begin(), push, groups));
    }
    try vk.orExit(cmds.endBatch(timeout));
    const t1 = std.Io.Clock.awake.now(io);
    return @intCast(t1.nanoseconds - t0.nanoseconds);
}

fn runCase(io: std.Io, cmds: *vk.Commands, staging: vk.Buffer, pipe: *dispatch.Pipeline, bindings: []const dispatch.Binding, push: []const u8, groups: [3]u32, n: usize, timeout: u64, out: *qf.Stats) !void {
    var samples: [sample_count]u64 = undefined;
    for (0..warmups) |_| _ = try sampleBatch(io, cmds, staging, pipe, bindings, push, groups, n, timeout);
    for (0..sample_count) |i| samples[i] = try sampleBatch(io, cmds, staging, pipe, bindings, push, groups, n, timeout);
    var scratch: [sample_count]u64 = undefined;
    out.* = qf.statsOf(&samples, &scratch).?; // fixed sizes: cannot fail
}

fn runShape(io: std.Io, a: std.mem.Allocator, dev: *vk.Device, cmds: *vk.Commands, staging: vk.Buffer, pipes: []const *dispatch.Pipeline, n: usize, k: usize, timeout: u64, add_floor_us: ?f64, special: bool) !bool {
    const words_count = n * k / 8;
    const param_count = n * (k / 64);
    const x_bytes = k * @sizeOf(u16);
    const weight_bytes = words_count * @sizeOf(u32);
    const param_bytes = param_count * @sizeOf(u16);
    const output_bytes = n * @sizeOf(u16);
    const weights = try a.alloc(u32, words_count);
    defer a.free(weights);
    const scales = try a.alloc(u16, param_count);
    defer a.free(scales);
    const biases = try a.alloc(u16, param_count);
    defer a.free(biases);
    const x = try a.alloc(u16, k);
    defer a.free(x);
    const ref_out = try a.alloc(u16, n);
    defer a.free(ref_out);
    const r_out = try a.alloc(u16, n);
    defer a.free(r_out);
    const want = try a.alloc(f64, n);
    defer a.free(want);
    const magnitude = try a.alloc(f64, n);
    defer a.free(magnitude);
    const allowances = try a.alloc(f64, n);
    defer a.free(allowances);
    qf.fillInputs(weights, scales, biases, x, 0x716d6d00 ^ @as(u64, n));
    if (special) {
        x[0] = 0x7FC0; // NaN
        x[1] = 0x7F80; // +inf
        x[2] = 0xFF80; // -inf
        x[3] = 0x0000; // zero
        scales[0] = 0x7F7F; // max-magnitude scale
        scales[1] = 0x0000; // zero scale
        biases[0] = 0xFF7F; // max-magnitude negative bias
        @memset(weights[0..16], 0); // zero nibble rows
        @memset(weights[100..116], 0xFFFF_FFFF); // all-15 nibbles
    }
    const shape: c.Shape = .{ .rows = 1, .n = n, .k = k };
    const spec = checkSpec();
    for (0..n) |column| {
        const ref = qf.referenceOne(weights, scales, biases, k, column);
        want[column] = ref[0];
        magnitude[column] = ref[1];
        allowances[column] = c.bound(spec, shape, ref[0], ref[1]);
    }
    var x_buf = try vk.Buffer.init(dev, .device, x_bytes);
    defer x_buf.deinit();
    var w_buf = try vk.Buffer.init(dev, .device, weight_bytes);
    defer w_buf.deinit();
    var scale_buf = try vk.Buffer.init(dev, .device, param_bytes);
    defer scale_buf.deinit();
    var bias_buf = try vk.Buffer.init(dev, .device, param_bytes);
    defer bias_buf.deinit();
    var out_buf = try vk.Buffer.init(dev, .device, output_bytes);
    defer out_buf.deinit();
    try vk.orExit(vk.upload(cmds, x_buf, 0, staging, std.mem.sliceAsBytes(x), timeout));
    try vk.orExit(vk.upload(cmds, w_buf, 0, staging, std.mem.sliceAsBytes(weights), timeout));
    try vk.orExit(vk.upload(cmds, scale_buf, 0, staging, std.mem.sliceAsBytes(scales), timeout));
    try vk.orExit(vk.upload(cmds, bias_buf, 0, staging, std.mem.sliceAsBytes(biases), timeout));
    @memset(ref_out, 0x7fc5);
    try vk.orExit(vk.upload(cmds, out_buf, 0, staging, std.mem.sliceAsBytes(ref_out), timeout));
    const bindings = [_]dispatch.Binding{
        .{ .buffer = &x_buf, .len = x_buf.len },
        .{ .buffer = &w_buf, .len = w_buf.len },
        .{ .buffer = &scale_buf, .len = scale_buf.len },
        .{ .buffer = &bias_buf, .len = bias_buf.len },
        .{ .buffer = &out_buf, .len = out_buf.len },
    };
    const push = qf.Params{ .count = @intCast(n), .output_size = @intCast(n), .matrix_m = 1, .matrix_n = @intCast(n), .matrix_k = @intCast(k), .flags = vk.ops_q4.pk_variant };
    const push_bytes = std.mem.asBytes(&push)[0..@sizeOf(qf.Params)];
    const max_wg = dev.properties.limits.maxComputeWorkGroupCount[0];

    // ---- R=1: the unchanged kernel, checked against the pk contract bound
    const groups1 = try vk.ops_q4.pkGroups(n, max_wg);
    try vk.orExit(pipes[0].bind(&bindings));
    try vk.orExit(pipes[0].run(cmds, push_bytes, .{ groups1, 1, 1 }, timeout));
    try vk.orExit(vk.download(cmds, out_buf, 0, staging, std.mem.sliceAsBytes(ref_out), timeout));
    var worst: f64 = 0;
    var failed: ?usize = null;
    if (!special) {
        for (ref_out, 0..) |bits, column| {
            const got = c.fromBf16(bits);
            const ratio = if (got == want[column]) 0 else @abs(got - want[column]) / allowances[column];
            worst = @max(worst, ratio);
            if (!(ratio <= 1) and failed == null) failed = column;
        }
        if (failed) |column| {
            std.debug.print("FAIL check: qmm_q4_pk R=1 n={d} k={d} output={d} got={e:.6} want={e:.6} bound-ratio={d:.3}\n", .{ n, k, column, c.fromBf16(ref_out[column]), want[column], worst });
            return false;
        }
    }
    std.debug.print("MILESTONE checks-passed R=1 n={d} k={d} outputs={d} worst={d:.3} of pk-contract bound{s}\n", .{ n, k, n, worst, if (special) " (bound check skipped: special inputs)" else "" });

    // ---- bit-exact gate: every R variant's full output equals R=1's, word for word
    for (variants[1..], 1..) |r, vi| {
        const groups_r = try wgGroups(n, r, max_wg);
        try vk.orExit(pipes[vi].bind(&bindings));
        try vk.orExit(pipes[vi].run(cmds, push_bytes, .{ groups_r, 1, 1 }, timeout));
        try vk.orExit(vk.download(cmds, out_buf, 0, staging, std.mem.sliceAsBytes(r_out), timeout));
        if (firstDiff16(r_out, ref_out)) |d| {
            std.debug.print("FAIL check: R={d} output differs from R=1 at index {d}: got=0x{x:0>4} want=0x{x:0>4}\n", .{ r, d.index, d.got, d.want });
            return false;
        }
        std.debug.print("MILESTONE bit-exact R={d} vs R=1: {d} outputs equal\n", .{ r, n });
    }
    std.debug.print("MILESTONE checks-passed n={d} k={d} bit-exact R=2/4/8 vs R=1; timing begins\n", .{ n, k });

    // ---- timing: R in {1,2,4,8} x N in {0,1,4,8} dependent dispatches
    var stats: [variants.len][deps.len]qf.Stats = undefined;
    for (variants, 0..) |r, vi| {
        const groups_r = if (r == 1) groups1 else try wgGroups(n, r, max_wg);
        for (deps, 0..) |dep, di| {
            try runCase(io, cmds, staging, pipes[vi], &bindings, push_bytes, .{ groups_r, 1, 1 }, dep, timeout, &stats[vi][di]);
            printStats(dep, stats[vi][di]);
        }
    }
    const bytes = qf.weightBytesFor(n, k);
    std.debug.print("    weight bytes/dispatch={d}; p50 slope (N=8 minus N=0)/8, effective weight GB/s, ratio to R=1:\n", .{bytes});
    var base_gbps: f64 = 1;
    for (variants, 0..) |r, vi| {
        const per = try qf.perDispatchNs(stats[vi][3].p50_ns, stats[vi][0].p50_ns, deps[deps.len - 1]);
        const gbps = qf.effectiveGbps(bytes, @intFromFloat(@max(per, 1.0)));
        if (vi == 0) base_gbps = gbps;
        std.debug.print("    R={d}: p50 {d:.3} us/dispatch, {d:.2} GB/s, {d:.2}x R=1\n", .{
            r, per / 1000.0, gbps, gbps / base_gbps,
        });
    }
    if (add_floor_us) |f| {
        const per1 = try qf.perDispatchNs(stats[0][3].p50_ns, stats[0][0].p50_ns, deps[deps.len - 1]);
        std.debug.print("    R=1 p50/add-floor={d:.2}x (add floor {d:.3} us given on the command line)\n", .{ (per1 / 1000.0) / f, f });
    }
    return true;
}

pub fn main(init: std.process.Init) !u8 {
    const io = init.io;
    const a = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    for (args[1..]) |arg| if (std.mem.eql(u8, arg, "--help")) {
        std.debug.print(usage, .{});
        return 0;
    };
    const opts = qf.parseArgs(args[1..]) orelse {
        std.debug.print(usage, .{});
        return 2;
    };
    const timeout = opts.timeout_ms * std.time.ns_per_ms;
    // PREFLIGHT ONLY, NOT A RESULT: this throwaway build admits a CPU (lavapipe)
    // device under BW_PREFLIGHT_CPU, divides the served shape sizes by
    // BW_PREFLIGHT_SCALE, and plants non-finite/extreme inputs under
    // BW_PREFLIGHT_SPECIAL. None of the knobs exists in the real branch.
    var scale: u64 = 1;
    if (std.c.getenv("BW_PREFLIGHT_SCALE")) |v| scale = std.fmt.parseInt(u64, std.mem.span(v), 10) catch 1;
    if (scale == 0) scale = 1;
    const preflight_cpu = std.c.getenv("BW_PREFLIGHT_CPU") != null;
    const special = std.c.getenv("BW_PREFLIGHT_SPECIAL") != null;
    std.debug.print("PREFLIGHT ONLY, NOT A RESULT: cpu-allowed={} scale={d} special={}\n", .{ preflight_cpu, scale, special });
    var rt = try vk.Runtime.open();
    defer rt.close();
    var dev = try vk.Device.open(&rt, .{});
    defer dev.close();
    const pr = dev.properties;
    const name = std.mem.sliceTo(&pr.deviceName, 0);
    std.debug.print("MILESTONE device-found name=\"{s}\" vendor=0x{x:0>4} device=0x{x:0>4} timeout={d}ms\n", .{ name, pr.vendorID, pr.deviceID, opts.timeout_ms });
    if (pr.deviceType == vk.c.VK_PHYSICAL_DEVICE_TYPE_CPU and !preflight_cpu) {
        std.debug.print("FAIL device: a CPU (software) Vulkan device is never a GPU result\n", .{});
        return 1;
    }
    if (opts.vendor) |want| if (pr.vendorID != want) {
        std.debug.print("FAIL device: vendor 0x{x:0>4} is not the requested 0x{x:0>4}\n", .{ pr.vendorID, want });
        return 1;
    };
    var cmds = try vk.Commands.init(&dev);
    defer cmds.deinit();
    var staging = try vk.Buffer.init(&dev, .staging, staging_bytes);
    defer staging.deinit();
    const layout: dispatch.Layout = .{ .bindings = 5, .push_bytes = @sizeOf(qf.Params), .writes = 1 << 4 };
    var pipe1 = try dispatch.Pipeline.init(&dev, &shaders.qmm_q4_pk, layout);
    defer pipe1.deinit();
    var pipe2 = try dispatch.Pipeline.init(&dev, &shaders.qmm_q4_pk_r2, layout);
    defer pipe2.deinit();
    var pipe4 = try dispatch.Pipeline.init(&dev, &shaders.qmm_q4_pk_r4, layout);
    defer pipe4.deinit();
    var pipe8 = try dispatch.Pipeline.init(&dev, &shaders.qmm_q4_pk_r8, layout);
    defer pipe8.deinit();
    if (!std.mem.eql(u32, &pipe1.local, &.{ 64, 1, 1 })) {
        std.debug.print("FAIL device: qmm_q4_pk local size is {any}, expected 64x1x1\n", .{pipe1.local});
        return 1;
    }
    if (!std.mem.eql(u32, &pipe8.local, &.{ 64, 1, 1 })) {
        std.debug.print("FAIL device: qmm_q4_pk_r8 local size is {any}, expected 64x1x1\n", .{pipe8.local});
        return 1;
    }
    const pipes = [_]*dispatch.Pipeline{ &pipe1, &pipe2, &pipe4, &pipe8 };
    const plan = qf.planCases(pr.limits.maxStorageBufferRange);
    if (!plan.head_fits) std.debug.print("SKIP served head n=248320 k=2048: packed Q4 weights exceed maxStorageBufferRange={d}; using largest listed fallback n={d} k={d}\n", .{ pr.limits.maxStorageBufferRange, plan.shapes[2][0], plan.shapes[2][1] });
    var completed: usize = 0;
    for (plan.shapes) |shape| {
        const n_scaled = shape[0] / scale;
        std.debug.print("MILESTONE case n={d} k={d} weight-bytes-per-dispatch={d}\n", .{ n_scaled, shape[1], qf.weightBytesFor(n_scaled, shape[1]) });
        if (!try runShape(io, a, &dev, &cmds, staging, &pipes, n_scaled, shape[1], timeout, opts.add_floor_us, special)) return 1;
        completed += 1;
    }
    std.debug.print("RESULT qmm-multi on \"{s}\": cases={d} ok\n", .{ name, completed });
    return 0;
}

const testing = std.testing;

test "ceilDiv rounds up and refuses a zero divisor" {
    try testing.expectError(error.ZeroDiv, ceilDiv(1, 0));
    try testing.expectEqual(@as(u64, 0), try ceilDiv(0, 8));
    try testing.expectEqual(@as(u64, 1), try ceilDiv(1, 8));
    try testing.expectEqual(@as(u64, 768), try ceilDiv(6144, 8));
    try testing.expectEqual(@as(u64, 2), try ceilDiv(9, 8));
    try testing.expectEqual(@as(u64, 31040), try ceilDiv(248320, 8));
}

test "wgGroups mirrors pkGroups' cap and refuses empty work" {
    try testing.expectError(error.Invalid, wgGroups(0, 8, 1024));
    try testing.expectError(error.Invalid, wgGroups(64, 0, 1024));
    try testing.expectEqual(@as(u32, 768), try wgGroups(6144, 8, 65535));
    try testing.expectEqual(@as(u32, 65535), try wgGroups(248320, 2, 65535)); // capped; the kernel grid-strides the rest
    try testing.expectEqual(@as(u32, 2), try wgGroups(9, 8, 65535)); // 1 full + 1 tail workgroup
}

test "firstDiff16 names the first differing word or null when equal" {
    try testing.expect(firstDiff16(&.{ 1, 2 }, &.{ 1, 2 }) == null);
    try testing.expect(firstDiff16(&.{}, &.{}) == null);
    const d = firstDiff16(&.{ 0x1234, 0x5678, 0x9abc }, &.{ 0x1234, 0xffff, 0x9abc }).?;
    try testing.expectEqual(@as(usize, 1), d.index);
    try testing.expectEqual(@as(u16, 0x5678), d.got);
    try testing.expectEqual(@as(u16, 0xffff), d.want);
}
