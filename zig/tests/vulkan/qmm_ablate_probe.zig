//! Private ablation probe for qmm_q4_pk: what keeps the served kernel at 28
//! percent of the chip's streaming bandwidth? Six variants per served shape:
//!   A0 the real kernel (the reference, gated on the pk contract's f64 bound);
//!   A1 plain f32 accumulation, no TwoSum (timing-only);
//!   A2 loads-only xor fold, no dequant, no float (timing-only);
//!   A3 the full compensated arithmetic reading one shared cache-resident
//!      64-word window (timing-only);
//!   B1/B2 the same arithmetic at 128/256 lanes per output (bits change, so
//!      they are gated on the f64 bound recomputed for the wider chain).
//! Every variant is timed at N = 0, 1, 4 dependent dispatches with the same
//! stats, the shape's REAL weight bytes for GB/s (comparable across variants),
//! and the ratio to A0. A1/A2/A3 lines say timing-only: they are not results
//! of the product kernel.
const std = @import("std");
const vk = @import("vulkan");
const contract = @import("contract");
const shaders = @import("ablate_shaders");
const qf = @import("qf");
const c = contract.spec;
const dispatch = vk.dispatch;

const usage = "usage: tf-qmm-ablate [--vendor ID] [--timeout-ms MS] [--add-floor-us US] | --help\n" ++
    "  --help            show this text and exit without opening a loader or device\n" ++
    "  --vendor ID       refuse a device whose Vulkan vendorID differs (decimal or 0x hex; no default)\n" ++
    "  --timeout-ms MS   bound on every fence wait, 1 to 600000 (default 10000); a timeout ends the process\n" ++
    "  --add-floor-us US this chip's own measured per-dispatch floor from tf-dispatch-floor; only then is a ratio printed\n";

const warmups = 3;
const sample_count = 63;
const staging_bytes = 1 << 20;
const deps = [_]usize{ 0, 1, 4 };

comptime {
    if (deps[0] != 0 or deps[deps.len - 1] != 4) @compileError("the depth table drifted: N=0 is the floor the slopes subtract");
}

/// fxpk's chain-length rule for an L-lane workgroup: 8 elements a pack,
/// ceil(k/8/L) packs a lane, the same +10 for the bias/compensation terms,
/// plus the extra tree levels past 64's six. Only powers of two from 64 up.
/// Pure and host-tested.
pub fn wideDepthFor(lanes: u32, k: usize) error{BadLanes}!usize {
    if (lanes < 64 or (lanes & (lanes - 1)) != 0) return error.BadLanes;
    const extra_levels = std.math.log2_int(u32, lanes) - 6;
    return 8 * ((k / 8 + lanes - 1) / lanes) + 10 + extra_levels;
}

/// Spec.depth functions for the two wide variants (Spec.depth is a plain
/// function pointer, so the lanes are baked into each).
pub fn depth128(s: c.Shape) usize {
    return wideDepthFor(128, s.k) catch unreachable;
}

pub fn depth256(s: c.Shape) usize {
    return wideDepthFor(256, s.k) catch unreachable;
}

/// How many of the f64 expectations are non-finite: bound ratios against them
/// are meaningless (NaN <= allowance is false even for identical bits), so the
/// gate skips and counts them. Pure and host-tested.
pub fn countNonFinite(want: []const f64) usize {
    var n: usize = 0;
    for (want) |v| {
        if (!std.math.isFinite(v)) {
            n += 1;
        }
    }
    return n;
}

fn ablateSpec(comptime name: []const u8, comptime depth_fn: *const fn (c.Shape) usize) c.Spec {
    return .{ .name = name, .accum = .f32, .depth = depth_fn, .out = .bf16, .served = &.{} };
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

/// One bound gate: the device output against the f64 reference with `spec`'s
/// bound, skipping (and counting) non-finite expectations — a NaN bound ratio
/// is false even for identical bits.
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
    const out16 = try a.alloc(u16, n);
    defer a.free(out16);
    const want = try a.alloc(f64, n);
    defer a.free(want);
    const magnitude = try a.alloc(f64, n);
    defer a.free(magnitude);
    const allowances = try a.alloc(f64, n);
    defer a.free(allowances);
    const allowances_b1 = try a.alloc(f64, n);
    defer a.free(allowances_b1);
    const allowances_b2 = try a.alloc(f64, n);
    defer a.free(allowances_b2);
    qf.fillInputs(weights, scales, biases, x, 0x716d6d00 ^ @as(u64, n));
    const shape: c.Shape = .{ .rows = 1, .n = n, .k = k };
    const spec_a0 = qf.pkSpec();
    const spec_b1 = ablateSpec("qmm-ablate B1 128-lane compensated", &depth128);
    const spec_b2 = ablateSpec("qmm-ablate B2 256-lane compensated", &depth256);
    for (0..n) |column| {
        const ref = qf.referenceOne(weights, scales, biases, k, column);
        want[column] = ref[0];
        magnitude[column] = ref[1];
        allowances[column] = c.bound(spec_a0, shape, ref[0], ref[1]);
        allowances_b1[column] = c.bound(spec_b1, shape, ref[0], ref[1]);
        allowances_b2[column] = c.bound(spec_b2, shape, ref[0], ref[1]);
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
    var out32_buf = try vk.Buffer.init(dev, .device, n * @sizeOf(u32)); // A2 stores one u32 word per output
    defer out32_buf.deinit();
    try vk.orExit(vk.upload(cmds, x_buf, 0, staging, std.mem.sliceAsBytes(x), timeout));
    try vk.orExit(vk.upload(cmds, w_buf, 0, staging, std.mem.sliceAsBytes(weights), timeout));
    try vk.orExit(vk.upload(cmds, scale_buf, 0, staging, std.mem.sliceAsBytes(scales), timeout));
    try vk.orExit(vk.upload(cmds, bias_buf, 0, staging, std.mem.sliceAsBytes(biases), timeout));
    const bindings16 = [_]dispatch.Binding{
        .{ .buffer = &x_buf, .len = x_buf.len },
        .{ .buffer = &w_buf, .len = w_buf.len },
        .{ .buffer = &scale_buf, .len = scale_buf.len },
        .{ .buffer = &bias_buf, .len = bias_buf.len },
        .{ .buffer = &out16_buf, .len = out16_buf.len },
    };
    const bindings32 = [_]dispatch.Binding{
        .{ .buffer = &x_buf, .len = x_buf.len },
        .{ .buffer = &w_buf, .len = w_buf.len },
        .{ .buffer = &scale_buf, .len = scale_buf.len },
        .{ .buffer = &bias_buf, .len = bias_buf.len },
        .{ .buffer = &out32_buf, .len = out32_buf.len },
    };
    const push = qf.Params{ .count = @intCast(n), .output_size = @intCast(n), .matrix_m = 1, .matrix_n = @intCast(n), .matrix_k = @intCast(k), .flags = vk.ops_q4.pk_variant };
    const push_bytes = std.mem.asBytes(&push)[0..@sizeOf(qf.Params)];
    const groups = try vk.ops_q4.pkGroups(n, dev.properties.limits.maxComputeWorkGroupCount[0]);
    const groups_vec = [3]u32{ groups, 1, 1 };
    const labels = [_][]const u8{ "A0 real", "A1 plain", "A2 loads", "A3 cached", "B1 128-lane", "B2 256-lane" };
    const timing_only = [_]bool{ false, true, true, true, false, false };

    // ---- correctness gates: A0, B1, B2 against their f64 bounds
    if (!try runGate(cmds, staging, pipes[0], &bindings16, push_bytes, groups_vec, timeout, out16, want, allowances, spec_a0, shape, "A0 real")) return false;
    if (!try runGate(cmds, staging, pipes[4], &bindings16, push_bytes, groups_vec, timeout, out16, want, allowances_b1, spec_b1, shape, "B1 128-lane")) return false;
    if (!try runGate(cmds, staging, pipes[5], &bindings16, push_bytes, groups_vec, timeout, out16, want, allowances_b2, spec_b2, shape, "B2 256-lane")) return false;
    std.debug.print("MILESTONE checks-passed n={d} k={d} A0/B1/B2 within bound; A1/A2/A3 timing-only, timing begins\n", .{ n, k });

    // ---- timing: all six variants, N in {0,1,4} dependent dispatches
    var stats: [6][deps.len]qf.Stats = undefined;
    for (pipes, 0..) |pipe, vi| {
        const bindings: []const dispatch.Binding = if (vi == 2) &bindings32 else &bindings16;
        for (deps, 0..) |dep, di| {
            try runCase(io, cmds, staging, pipe, bindings, push_bytes, groups_vec, dep, timeout, &stats[vi][di]);
            printStats(dep, stats[vi][di]);
        }
    }
    const bytes = qf.weightBytesFor(n, k);
    std.debug.print("    weight bytes/dispatch={d} for EVERY variant; p50 slope (N=4 minus N=0)/4, effective weight GB/s, ratio to A0:\n", .{bytes});
    var base_gbps: f64 = 1;
    for (labels, 0..) |label, vi| {
        const per = try qf.perDispatchNs(stats[vi][2].p50_ns, stats[vi][0].p50_ns, deps[deps.len - 1]);
        const gbps = qf.effectiveGbps(bytes, @intFromFloat(@max(per, 1.0)));
        if (vi == 0) base_gbps = gbps;
        std.debug.print("    {s}: p50 {d:.3} us/dispatch, {d:.2} GB/s, {d:.2}x A0{s}\n", .{
            label, per / 1000.0, gbps, gbps / base_gbps,
            if (timing_only[vi]) " (timing-only, not a product-kernel result)" else "",
        });
    }
    if (add_floor_us) |f| {
        const per0 = try qf.perDispatchNs(stats[0][2].p50_ns, stats[0][0].p50_ns, deps[deps.len - 1]);
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
    var pipe_a1 = try dispatch.Pipeline.init(&dev, &shaders.qmm_ablate_a1, layout);
    defer pipe_a1.deinit();
    var pipe_a2 = try dispatch.Pipeline.init(&dev, &shaders.qmm_ablate_a2, layout);
    defer pipe_a2.deinit();
    var pipe_a3 = try dispatch.Pipeline.init(&dev, &shaders.qmm_ablate_a3, layout);
    defer pipe_a3.deinit();
    var pipe_b1 = try dispatch.Pipeline.init(&dev, &shaders.qmm_ablate_wide128, layout);
    defer pipe_b1.deinit();
    var pipe_b2 = try dispatch.Pipeline.init(&dev, &shaders.qmm_ablate_wide256, layout);
    defer pipe_b2.deinit();
    if (!std.mem.eql(u32, &pipe_a0.local, &.{ 64, 1, 1 })) {
        std.debug.print("FAIL device: qmm_q4_pk local size is {any}, expected 64x1x1\n", .{pipe_a0.local});
        return 1;
    }
    if (!std.mem.eql(u32, &pipe_b1.local, &.{ 128, 1, 1 }) or !std.mem.eql(u32, &pipe_b2.local, &.{ 256, 1, 1 })) {
        std.debug.print("FAIL device: wide local sizes are {any} and {any}, expected 128x1x1 and 256x1x1\n", .{ pipe_b1.local, pipe_b2.local });
        return 1;
    }
    const pipes = [_]*dispatch.Pipeline{ &pipe_a0, &pipe_a1, &pipe_a2, &pipe_a3, &pipe_b1, &pipe_b2 };
    const plan = qf.planCases(pr.limits.maxStorageBufferRange);
    if (!plan.head_fits) std.debug.print("SKIP served head n=248320 k=2048: packed Q4 weights exceed maxStorageBufferRange={d}; using largest listed fallback n={d} k={d}\n", .{ pr.limits.maxStorageBufferRange, plan.shapes[2][0], plan.shapes[2][1] });
    var completed: usize = 0;
    for (plan.shapes) |shape| {
        std.debug.print("MILESTONE case n={d} k={d} weight-bytes-per-dispatch={d}\n", .{ shape[0], shape[1], qf.weightBytesFor(shape[0], shape[1]) });
        if (!try runShape(io, a, &dev, &cmds, staging, &pipes, shape[0], shape[1], timeout, opts.add_floor_us)) return 1;
        completed += 1;
    }
    std.debug.print("RESULT qmm-ablate on \"{s}\": cases={d} ok\n", .{ name, completed });
    return 0;
}

const testing = std.testing;

test "wideDepthFor extends fxpk's chain rule and refuses non-power-of-two lanes" {
    try testing.expectError(error.BadLanes, wideDepthFor(48, 2048));
    try testing.expectError(error.BadLanes, wideDepthFor(32, 2048));
    // at 64 lanes it must reproduce fxpk.depth exactly
    try testing.expectEqual(@as(usize, 8 * ((2048 / 8 + 63) / 64) + 10), try wideDepthFor(64, 2048));
    // 128 lanes: 2 packs a lane, one extra tree level
    try testing.expectEqual(@as(usize, 8 * ((2048 / 8 + 127) / 128) + 10 + 1), try wideDepthFor(128, 2048));
    // 256 lanes: 1 pack a lane, two extra tree levels
    try testing.expectEqual(@as(usize, 8 * ((2048 / 8 + 255) / 256) + 10 + 2), try wideDepthFor(256, 2048));
    // the named spec depths agree with the shape-typed contract function
    try testing.expectEqual(@as(usize, try wideDepthFor(128, 2048)), depth128(.{ .rows = 1, .n = 8, .k = 2048 }));
    try testing.expectEqual(@as(usize, try wideDepthFor(256, 6144)), depth256(.{ .rows = 1, .n = 8, .k = 6144 }));
}

test "countNonFinite counts NaN and infinite expectations only" {
    try testing.expectEqual(@as(usize, 0), countNonFinite(&.{ 0, 1, -2.5 }));
    try testing.expectEqual(@as(usize, 3), countNonFinite(&.{ std.math.nan(f64), 1, std.math.inf(f64), -std.math.inf(f64), 0 }));
}
