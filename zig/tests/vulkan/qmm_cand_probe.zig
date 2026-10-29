//! Private candidate probe for qmm_q4_pk's per-output cost. The ablation
//! showed loads (5.66 ms) and arithmetic (9.67 ms) nearly additive inside A0's
//! 16.7 ms, and the compensation itself about 9 ms. Three candidates:
//!   C1 the same per-output arithmetic with one-deep software-pipelined loads
//!      (must be BIT-IDENTICAL to A0 — checked before timing);
//!   C2 pack-level compensation: 8 plain f32 adds inside a packed word, then
//!      ONE TwoSum fold per pack (bits change: gated on the f64 bound with
//!      packCompDepth's recomputed chain);
//!   C3 C1's pipelined loads on top of C2 (gated like C2);
//! with A1 (plain accumulation) and A3 (cache-resident) as reference rows.
//! Timing: N = 0, 1, 4, 8 dependent dispatches, slopes from N=8 (an N=4 slope
//! still carries submit overhead at the small shapes). Every variant's GB/s
//! uses the shape's REAL weight bytes, so ratios are direct.
const std = @import("std");
const vk = @import("vulkan");
const contract = @import("contract");
const shaders = @import("cand_shaders");
const qf = @import("qf");
const c = contract.spec;
const dispatch = vk.dispatch;

const usage = "usage: tf-qmm-cand [--vendor ID] [--timeout-ms MS] [--add-floor-us US] | --help\n" ++
    "  --help            show this text and exit without opening a loader or device\n" ++
    "  --vendor ID       refuse a device whose Vulkan vendorID differs (decimal or 0x hex; no default)\n" ++
    "  --timeout-ms MS   bound on every fence wait, 1 to 600000 (default 10000); a timeout ends the process\n" ++
    "  --add-floor-us US this chip's own measured per-dispatch floor from tf-dispatch-floor; only then is a ratio printed\n";

const warmups = 3;
const sample_count = 63;
const staging_bytes = 1 << 20;
const deps = [_]usize{ 0, 1, 4, 8 };

comptime {
    if (deps[0] != 0 or deps[deps.len - 1] != 8) @compileError("the depth table drifted: N=0 is the floor, N=8 the slope the ratios use");
}

/// The rounding-chain depth for pack-level compensation at L lanes: per pack
/// the 8 terms take 7 plain left-associated adds, and the pack sum folds into
/// the lane's (hi, lo) with one TwoSum — so 8 roundings a pack — then the
/// fixed 6-level tree, the final high+low add, and the contract's +10 slack
/// for the first term's scale*q, +bias, *x roundings and the second-order lo
/// chain. Pure and host-tested.
pub fn packCompDepth(lanes: u32, k: usize) error{BadLanes}!usize {
    if (lanes < 64 or (lanes & (lanes - 1)) != 0) return error.BadLanes;
    const packs_lane = (k / 8 + lanes - 1) / lanes;
    return 7 * packs_lane + packs_lane + 6 + 1 + 10;
}

/// Spec.depth for the 64-lane pack-compensated candidates.
pub fn packDepth(s: c.Shape) usize {
    return packCompDepth(64, s.k) catch unreachable;
}

pub const WordDiff = struct { index: usize, got: u16, want: u16 };

/// bf16 NaN: exponent all ones, mantissa nonzero. Pure and host-tested: C1's
/// bit-exact gate treats NaN-vs-NaN payload differences separately, because
/// the sign of an invalid-op NaN (inf + -inf, 0 * inf) is not pinned by the
/// source's `precise` order.
pub fn isNan16(bits: u16) bool {
    return (bits & 0x7f80) == 0x7f80 and (bits & 0x7f) != 0;
}

/// The first differing output word of two same-length bf16-word slices, with
/// both words. Null when equal. Pure and host-tested: C1's bit-exact gate's
/// FAIL line names the index and both words.
pub fn firstDiff16(a: []const u16, b: []const u16) ?WordDiff {
    if (a.len != b.len) return .{ .index = 0, .got = if (a.len == 0) 0 else a[0], .want = if (b.len == 0) 0 else b[0] };
    for (a, b, 0..) |x, y, i| if (x != y) return .{ .index = i, .got = x, .want = y };
    return null;
}

fn candSpec(comptime name: []const u8) c.Spec {
    return .{ .name = name, .accum = .f32, .depth = &packDepth, .out = .bf16, .served = &.{} };
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

/// One bound gate against the f64 reference, skipping (and counting)
/// non-finite expectations — a NaN bound ratio is false even for identical bits.
fn runGate(cmds: *vk.Commands, staging: vk.Buffer, pipe: *dispatch.Pipeline, bindings: []const dispatch.Binding, push: []const u8, groups: [3]u32, timeout: u64, out16: []u16, want: []const f64, allowances: []const f64, spec: c.Spec, shape: c.Shape, label: []const u8) !bool {
    @memset(out16, 0x7fc5);
    try vk.orExit(vk.upload(cmds, bindings[4].buffer.*, 0, staging, std.mem.sliceAsBytes(out16), timeout));
    try pipe.bind(bindings);
    try vk.orExit(pipe.run(cmds, push, groups, timeout));
    try vk.orExit(vk.download(cmds, bindings[4].buffer.*, 0, staging, std.mem.sliceAsBytes(out16), timeout));
    var worst: f64 = 0;
    var failed: ?usize = null;
    var skipped: usize = 0;
    for (out16, 0..) |bits, column| {
        if (!std.math.isFinite(want[column])) {
            skipped += 1;
            continue;
        }
        const got = c.fromBf16(bits);
        const ratio = if (got == want[column]) 0 else @abs(got - want[column]) / allowances[column];
        worst = @max(worst, ratio);
        if (!(ratio <= 1) and failed == null) failed = column;
    }
    if (failed) |column| {
        std.debug.print("FAIL check: {s} n={d} k={d} output={d} got={e:.6} want={e:.6} bound-ratio={d:.3}\n", .{ label, shape.n, shape.k, column, c.fromBf16(out16[column]), want[column], worst });
        return false;
    }
    var skip_buf: [64]u8 = undefined;
    const skip_note = if (skipped > 0) std.fmt.bufPrint(&skip_buf, " ({d} non-finite expectations skipped)", .{skipped}) catch "" else "";
    std.debug.print("MILESTONE checks-passed {s} n={d} k={d} outputs={d} worst={d:.3} of {s} bound{s}\n", .{ label, shape.n, shape.k, shape.n, worst, spec.name, skip_note });
    return true;
}

fn runShape(io: std.Io, a: std.mem.Allocator, dev: *vk.Device, cmds: *vk.Commands, staging: vk.Buffer, pipes: []const *dispatch.Pipeline, n: usize, k: usize, timeout: u64, add_floor_us: ?f64) !bool {
    const words_count = n * k / 8;
    const param_count = n * (k / 64);
    const x_bytes = k * @sizeOf(u16);
    const weight_bytes = words_count * @sizeOf(u32);
    const param_bytes = param_count * @sizeOf(u16);
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
    const c_out = try a.alloc(u16, n);
    defer a.free(c_out);
    const want = try a.alloc(f64, n);
    defer a.free(want);
    const magnitude = try a.alloc(f64, n);
    defer a.free(magnitude);
    const allowances = try a.alloc(f64, n);
    defer a.free(allowances);
    const allowances_pack = try a.alloc(f64, n);
    defer a.free(allowances_pack);
    qf.fillInputs(weights, scales, biases, x, 0x716d6d00 ^ @as(u64, n));
    const shape: c.Shape = .{ .rows = 1, .n = n, .k = k };
    const spec_a0 = qf.pkSpec();
    const spec_pack = candSpec("qmm-cand pack-compensated");
    for (0..n) |column| {
        const ref = qf.referenceOne(weights, scales, biases, x, k, column);
        want[column] = ref[0];
        magnitude[column] = ref[1];
        allowances[column] = c.bound(spec_a0, shape, ref[0], ref[1]);
        allowances_pack[column] = c.bound(spec_pack, shape, ref[0], ref[1]);
    }
    var x_buf = try vk.Buffer.init(dev, .device, x_bytes);
    defer x_buf.deinit();
    var w_buf = try vk.Buffer.init(dev, .device, weight_bytes);
    defer w_buf.deinit();
    var scale_buf = try vk.Buffer.init(dev, .device, param_bytes);
    defer scale_buf.deinit();
    var bias_buf = try vk.Buffer.init(dev, .device, param_bytes);
    defer bias_buf.deinit();
    var out16_buf = try vk.Buffer.init(dev, .device, n * @sizeOf(u16));
    defer out16_buf.deinit();
    try vk.orExit(vk.upload(cmds, x_buf, 0, staging, std.mem.sliceAsBytes(x), timeout));
    try vk.orExit(vk.upload(cmds, w_buf, 0, staging, std.mem.sliceAsBytes(weights), timeout));
    try vk.orExit(vk.upload(cmds, scale_buf, 0, staging, std.mem.sliceAsBytes(scales), timeout));
    try vk.orExit(vk.upload(cmds, bias_buf, 0, staging, std.mem.sliceAsBytes(biases), timeout));
    const bindings = [_]dispatch.Binding{
        .{ .buffer = &x_buf, .len = x_buf.len },
        .{ .buffer = &w_buf, .len = w_buf.len },
        .{ .buffer = &scale_buf, .len = scale_buf.len },
        .{ .buffer = &bias_buf, .len = bias_buf.len },
        .{ .buffer = &out16_buf, .len = out16_buf.len },
    };
    const push = qf.Params{ .count = @intCast(n), .output_size = @intCast(n), .matrix_m = 1, .matrix_n = @intCast(n), .matrix_k = @intCast(k), .flags = vk.ops_q4.pk_variant };
    const push_bytes = std.mem.asBytes(&push)[0..@sizeOf(qf.Params)];
    const groups = try vk.ops_q4.pkGroups(n, dev.properties.limits.maxComputeWorkGroupCount[0]);
    const groups_vec = [3]u32{ groups, 1, 1 };
    const labels = [_][]const u8{ "A0 real", "C1 pipelined", "C2 pack-comp", "C3 C1+C2", "C1v2 precise accumulators", "A3 cached" };
    const timing_only = [_]bool{ false, false, false, false, false, true };

    // ---- correctness gates: A0's f64 bound, C1 bit-exact vs A0, C2/C3's pack bound
    if (!try runGate(cmds, staging, pipes[0], &bindings, push_bytes, groups_vec, timeout, ref_out, want, allowances, spec_a0, shape, "A0 real")) return false;
    // ---- DIAGNOSTIC, report-only: C1 and four suspect variants against A0 (real activations); nothing here is a result
    const diag_labels = [_][]const u8{ "C1 pipelined", "C1v1 dq precise", "C1v2 precise accumulators", "C1v3 x as u32 words", "C1v4 all three", "A0p the OLD kernel with precise accumulators" };
    for ([_]usize{ 1, 6, 7, 8, 9, 10 }, 0..) |vi, di| {
        @memset(c_out, 0x7fc5);
        try pipes[vi].bind(&bindings);
        try vk.orExit(pipes[vi].run(cmds, push_bytes, groups_vec, timeout));
        try vk.orExit(vk.download(cmds, out16_buf, 0, staging, std.mem.sliceAsBytes(c_out), timeout));
        var finite_diff: usize = 0;
        var nan_diff: usize = 0;
        var shown: usize = 0;
        var dbase: usize = 0;
        while (dbase < c_out.len) {
            const d = firstDiff16(c_out[dbase..], ref_out[dbase..]) orelse break;
            const idx = dbase + d.index;
            if (isNan16(d.got) and isNan16(d.want)) {
                nan_diff += 1;
            } else {
                finite_diff += 1;
                if (shown < 3) {
                    std.debug.print("DIAG {s} vs A0 first differences: index {d} got=0x{x:0>4} want=0x{x:0>4}\n", .{ diag_labels[di], idx, d.got, d.want });
                    shown += 1;
                }
            }
            dbase = idx + 1;
        }
        std.debug.print("DIAG {s} vs A0 (report-only): {d} of {d} outputs differ ({d} NaN payload only)\n", .{ diag_labels[di], finite_diff, n, nan_diff });
    }
    for ([_]struct { vi: usize, label: []const u8 }{ .{ .vi = 1, .label = "C1 pipelined" }, .{ .vi = 2, .label = "C2 pack-comp" }, .{ .vi = 3, .label = "C3 C1+C2" } }) |gate| {
        @memset(c_out, 0x7fc5);
        try pipes[gate.vi].bind(&bindings);
        try vk.orExit(pipes[gate.vi].run(cmds, push_bytes, groups_vec, timeout));
        try vk.orExit(vk.download(cmds, out16_buf, 0, staging, std.mem.sliceAsBytes(c_out), timeout));
        if (gate.vi == 1) {
            // Bit-exact vs A0, NaN payloads treated separately: the sign of an
            // invalid-op NaN (inf + -inf, 0 * inf) is not pinned by the
            // source's precise order, so a NaN-vs-NaN payload difference is
            // counted and reported, while any finite-word difference is fatal.
            var nan_diff: usize = 0;
            var fatal: ?WordDiff = null;
            var base: usize = 0;
            while (firstDiff16(c_out[base..], ref_out[base..])) |d| {
                const idx = base + d.index;
                if (isNan16(d.got) and isNan16(d.want)) {
                    nan_diff += 1;
                    base = idx + 1;
                } else {
                    fatal = .{ .index = idx, .got = d.got, .want = d.want };
                    break;
                }
            }
            if (fatal) |d| {
                // DIAGNOSTIC BUILD: report and go on, so the other shapes run too (the product probe exits here)
                std.debug.print("NOTE (diagnostic build, not fatal): {s} output differs from A0 at index {d}: got=0x{x:0>4} want=0x{x:0>4}\n", .{ gate.label, d.index, d.got, d.want });
            }
            var nan_buf: [64]u8 = undefined;
            const nan_note = if (nan_diff > 0) std.fmt.bufPrint(&nan_buf, "; {d} NaN payload signs differ (not pinned by precise; all finite outputs identical)", .{nan_diff}) catch "" else "";
            std.debug.print("MILESTONE bit-exact {s} vs A0: {d} outputs equal{s}\n", .{ gate.label, n, nan_note });
        } else {
            var worst: f64 = 0;
            var failed: ?usize = null;
            var skipped: usize = 0;
            for (c_out, 0..) |bits, column| {
                if (!std.math.isFinite(want[column])) {
                    skipped += 1;
                    continue;
                }
                const got = c.fromBf16(bits);
                const ratio = if (got == want[column]) 0 else @abs(got - want[column]) / allowances_pack[column];
                worst = @max(worst, ratio);
                if (!(ratio <= 1) and failed == null) failed = column;
            }
            if (failed) |column| {
                std.debug.print("FAIL check: {s} n={d} k={d} output={d} got={e:.6} want={e:.6} pack-bound-ratio={d:.3}\n", .{ gate.label, n, k, column, c.fromBf16(c_out[column]), want[column], worst });
                return false;
            }
            var skip_buf: [64]u8 = undefined;
            const skip_note = if (skipped > 0) std.fmt.bufPrint(&skip_buf, " ({d} non-finite expectations skipped)", .{skipped}) catch "" else "";
            std.debug.print("MILESTONE checks-passed {s} n={d} k={d} outputs={d} worst={d:.3} of pack-comp bound{s}\n", .{ gate.label, n, k, n, worst, skip_note });
        }
    }
    std.debug.print("MILESTONE checks-passed n={d} k={d} C1 bit-exact, C2/C3 within pack-comp bound; timing begins\n", .{ n, k });

    // ---- timing: all six variants, N in {0,1,4,8} dependent dispatches
    var stats: [6][deps.len]qf.Stats = undefined;
    for (pipes[0..6], 0..) |pipe, vi| {
        for (deps, 0..) |dep, di| {
            try runCase(io, cmds, staging, pipe, &bindings, push_bytes, groups_vec, dep, timeout, &stats[vi][di]);
            printStats(dep, stats[vi][di]);
        }
    }
    const bytes = qf.weightBytesFor(n, k);
    std.debug.print("    weight bytes/dispatch={d} for EVERY variant; p50 slope (N=8 minus N=0)/8, effective weight GB/s, ratio to A0:\n", .{bytes});
    var base_gbps: f64 = 1;
    for (labels, 0..) |label, vi| {
        const per = try qf.perDispatchNs(stats[vi][3].p50_ns, stats[vi][0].p50_ns, deps[deps.len - 1]);
        const gbps = qf.effectiveGbps(bytes, @intFromFloat(@max(per, 1.0)));
        if (vi == 0) base_gbps = gbps;
        std.debug.print("    {s}: p50 {d:.3} us/dispatch, {d:.2} GB/s, {d:.2}x A0{s}\n", .{
            label, per / 1000.0, gbps, gbps / base_gbps,
            if (timing_only[vi]) " (timing-only, not a product-kernel result)" else if (vi == 1) " (bit-identical to A0)" else if (vi <= 3) " (new bits, pack-comp bound)" else "",
        });
    }
    if (add_floor_us) |f| {
        const per0 = try qf.perDispatchNs(stats[0][3].p50_ns, stats[0][0].p50_ns, deps[deps.len - 1]);
        std.debug.print("    A0 p50/add-floor={d:.2}x (add floor {d:.3} us given on the command line)\n", .{ (per0 / 1000.0) / f, f });
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
    var rt = try vk.Runtime.open();
    defer rt.close();
    var dev = try vk.Device.open(&rt, .{});
    defer dev.close();
    const pr = dev.properties;
    const name = std.mem.sliceTo(&pr.deviceName, 0);
    std.debug.print("MILESTONE device-found name=\"{s}\" vendor=0x{x:0>4} device=0x{x:0>4} timeout={d}ms\n", .{ name, pr.vendorID, pr.deviceID, opts.timeout_ms });
    if (pr.deviceType == vk.c.VK_PHYSICAL_DEVICE_TYPE_CPU) {
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
    var pipe_a0 = try dispatch.Pipeline.init(&dev, &shaders.qmm_q4_pk, layout);
    defer pipe_a0.deinit();
    var pipe_c1 = try dispatch.Pipeline.init(&dev, &shaders.qmm_cand_c1, layout);
    defer pipe_c1.deinit();
    var pipe_c2 = try dispatch.Pipeline.init(&dev, &shaders.qmm_cand_c2, layout);
    defer pipe_c2.deinit();
    var pipe_c3 = try dispatch.Pipeline.init(&dev, &shaders.qmm_cand_c3, layout);
    defer pipe_c3.deinit();
    var pipe_a1 = try dispatch.Pipeline.init(&dev, &shaders.qmm_ablate_a1, layout);
    defer pipe_a1.deinit();
    var pipe_a3 = try dispatch.Pipeline.init(&dev, &shaders.qmm_ablate_a3, layout);
    defer pipe_a3.deinit();
    var pipe_v1 = try dispatch.Pipeline.init(&dev, &shaders.qmm_cand_c1v1, layout);
    defer pipe_v1.deinit();
    var pipe_v2 = try dispatch.Pipeline.init(&dev, &shaders.qmm_cand_c1v2, layout);
    defer pipe_v2.deinit();
    var pipe_v3 = try dispatch.Pipeline.init(&dev, &shaders.qmm_cand_c1v3, layout);
    defer pipe_v3.deinit();
    var pipe_v4 = try dispatch.Pipeline.init(&dev, &shaders.qmm_cand_c1v4, layout);
    defer pipe_v4.deinit();
    var pipe_a0p = try dispatch.Pipeline.init(&dev, &shaders.qmm_cand_a0p, layout);
    defer pipe_a0p.deinit();
    if (!std.mem.eql(u32, &pipe_a0.local, &.{ 64, 1, 1 })) {
        std.debug.print("FAIL device: qmm_q4_pk local size is {any}, expected 64x1x1\n", .{pipe_a0.local});
        return 1;
    }
    const pipes = [_]*dispatch.Pipeline{ &pipe_a0, &pipe_c1, &pipe_c2, &pipe_c3, &pipe_v2, &pipe_a3, &pipe_v1, &pipe_v2, &pipe_v3, &pipe_v4, &pipe_a0p };
    const plan = qf.planCases(pr.limits.maxStorageBufferRange);
    if (!plan.head_fits) std.debug.print("SKIP served head n=248320 k=2048: packed Q4 weights exceed maxStorageBufferRange={d}; using largest listed fallback n={d} k={d}\n", .{ pr.limits.maxStorageBufferRange, plan.shapes[2][0], plan.shapes[2][1] });
    var completed: usize = 0;
    // DIAGNOSTIC shape list: the decode token's own shapes (n, k), the head last (it may not fit a small device)
    const token_shapes = [_][2]usize{ .{ 2048, 6144 }, .{ 6144, 2048 }, .{ 2048, 2048 }, .{ 4096, 2048 }, .{ 512, 2048 }, plan.shapes[2] };
    for (token_shapes) |shape| {
        std.debug.print("MILESTONE case n={d} k={d} weight-bytes-per-dispatch={d}\n", .{ shape[0], shape[1], qf.weightBytesFor(shape[0], shape[1]) });
        if (!try runShape(io, a, &dev, &cmds, staging, &pipes, shape[0], shape[1], timeout, opts.add_floor_us)) return 1;
        completed += 1;
    }
    std.debug.print("RESULT qmm-cand on \"{s}\": cases={d} ok\n", .{ name, completed });
    return 0;
}

const testing = std.testing;

test "packCompDepth counts 8 roundings a pack plus tree, final and slack" {
    try testing.expectError(error.BadLanes, packCompDepth(48, 2048));
    try testing.expectError(error.BadLanes, packCompDepth(32, 2048));
    // K=2048, 64 lanes: 4 packs a lane -> 7*4 plain adds + 4 folds + 6 tree + 1 final + 10 slack
    try testing.expectEqual(@as(usize, 49), try packCompDepth(64, 2048));
    // K=6144: 12 packs a lane
    try testing.expectEqual(@as(usize, 113), try packCompDepth(64, 6144));
    // the named spec depth agrees with the shape-typed contract function
    try testing.expectEqual(@as(usize, 49), packDepth(.{ .rows = 1, .n = 8, .k = 2048 }));
}

test "isNan16 classifies bf16 NaNs only" {
    try testing.expect(!isNan16(0x0000)); // +0
    try testing.expect(!isNan16(0x8000)); // -0
    try testing.expect(!isNan16(0x3f80)); // 1.0
    try testing.expect(!isNan16(0x7f80)); // +inf
    try testing.expect(!isNan16(0xff80)); // -inf
    try testing.expect(isNan16(0x7fc0)); // quiet NaN
    try testing.expect(isNan16(0xffc0)); // NaN, sign set
    try testing.expect(isNan16(0x7f81)); // signalling NaN
}

test "firstDiff16 names the first differing word or null when equal" {
    try testing.expect(firstDiff16(&.{ 1, 2 }, &.{ 1, 2 }) == null);
    try testing.expect(firstDiff16(&.{}, &.{}) == null);
    const d = firstDiff16(&.{ 0x1234, 0x5678 }, &.{ 0x1234, 0x4321 }).?;
    try testing.expectEqual(@as(usize, 1), d.index);
    try testing.expectEqual(@as(u16, 0x5678), d.got);
    try testing.expectEqual(@as(u16, 0x4321), d.want);
}
