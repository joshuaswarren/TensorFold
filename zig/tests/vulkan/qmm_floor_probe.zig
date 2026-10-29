//! Private timing probe for the engine's compensated split-K qmm_q4_pk dispatch.
//! It uses the same command-buffer batch path as dispatch-floor, but measures
//! real Q4 projections at served decode shapes. Every shape is checked against
//! the QMM contract's float64 affine reference and bound before timing.
const std = @import("std");
const vk = @import("vulkan");
const contract = @import("contract");
const qmm = @import("contract_qmm.zig");
const fxpk = @import("contract_qmm_pk.zig");
const shaders = @import("qmm_pk_shaders");
const c = contract.spec;
const dispatch = vk.dispatch;

const usage = "usage: tf-qmm-floor [--vendor ID] [--timeout-ms MS] [--add-floor-us US] | --help\n" ++
    "  --help            show this text and exit without opening a loader or device\n" ++
    "  --vendor ID       refuse a device whose Vulkan vendorID differs (decimal or 0x hex; no default)\n" ++
    "  --timeout-ms MS   bound on every fence wait, 1 to 600000 (default 10000); a timeout ends the process\n" ++
    "  --add-floor-us US this chip's own measured per-dispatch floor from tf-dispatch-floor; only then is a ratio printed\n";

const warmups = 3;
const sample_count = 63;
const staging_bytes = 1 << 20;
const cases = [_][2]usize{ .{ 6144, 2048 }, .{ 2048, 2048 }, .{ 248320, 2048 } };
const fallback_shapes = [_][2]usize{ .{ 4096, 2048 }, .{ 2048, 6144 }, .{ 512, 2048 } };
const dispatch_counts = [_]usize{ 0, 1, 16, 64 };
var add_floor_us: ?f64 = null; // set once in main from --add-floor-us; no built-in default, a floor belongs to one chip

pub const Options = struct { vendor: ?u32 = null, timeout_ms: u64 = 10_000, add_floor_us: ?f64 = null };

pub fn parseArgs(args: []const []const u8) ?Options {
    var o: Options = .{};
    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--vendor")) {
            if (i + 1 >= args.len) return null;
            o.vendor = std.fmt.parseInt(u32, args[i + 1], 0) catch return null;
            i += 2;
        } else if (std.mem.eql(u8, arg, "--timeout-ms")) {
            if (i + 1 >= args.len) return null;
            o.timeout_ms = std.fmt.parseInt(u64, args[i + 1], 10) catch return null;
            if (o.timeout_ms == 0 or o.timeout_ms > 600_000) return null;
            i += 2;
        } else if (std.mem.eql(u8, arg, "--add-floor-us")) {
            if (i + 1 >= args.len) return null;
            const v = std.fmt.parseFloat(f64, args[i + 1]) catch return null;
            if (!std.math.isFinite(v) or v <= 0) return null;
            o.add_floor_us = v;
            i += 2;
        } else return null;
    }
    return o;
}

pub const Stats = struct { min_ns: u64, p50_ns: u64, p95_ns: u64, max_ns: u64 };

pub fn rankIndex(p: u32, n: usize) ?usize {
    if (n == 0) return null;
    return (p * n + 99) / 100 - 1;
}

pub fn statsOf(samples: []const u64, scratch: []u64) ?Stats {
    if (samples.len == 0 or scratch.len < samples.len) return null;
    @memcpy(scratch[0..samples.len], samples);
    std.mem.sort(u64, scratch[0..samples.len], {}, std.sort.asc(u64));
    const s = scratch[0..samples.len];
    return .{ .min_ns = s[0], .p50_ns = s[rankIndex(50, s.len).?], .p95_ns = s[rankIndex(95, s.len).?], .max_ns = s[s.len - 1] };
}

pub fn perDispatchNs(high_ns: u64, low_ns: u64, n: u64) error{ZeroN}!f64 {
    if (n == 0) return error.ZeroN;
    return (@as(f64, @floatFromInt(high_ns)) - @as(f64, @floatFromInt(low_ns))) / @as(f64, @floatFromInt(n));
}

pub fn effectiveGbps(bytes: u64, ns: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(ns));
}

pub fn packedWeightBytesFor(n: usize, k: usize) u64 {
    return @intCast(n * k / 2);
}

pub fn weightBytesFor(n: usize, k: usize) u64 {
    return @intCast(n * k / 2 + 2 * (n * (k / 64) * 2));
}

fn bf16RneBits(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    if (std.math.isNan(value)) return @intCast((bits >> 16) | 0x40);
    return @intCast((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

fn bf16ToF64(bits: u16) f64 {
    const value: f32 = @bitCast(@as(u32, bits) << 16);
    return value;
}

pub fn nibbleAt(words: []const u32, element: usize) u32 {
    return (words[element / 8] >> @intCast((element % 8) * 4)) & 0xf;
}

/// Fill packed Q4 words and normal-range bf16 scales/biases; x is the required row of ones.
pub fn fillInputs(words: []u32, scales: []u16, biases: []u16, x: []u16, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    for (words) |*word| word.* = rng.int(u32);
    for (scales) |*s| s.* = bf16RneBits(rng.float(f32) * 0.02 + 0.001);
    for (biases) |*b| b.* = bf16RneBits((rng.float(f32) - 0.66) * 0.15);
    @memset(x, 0x3f80);
}

/// The pk contract's dequantized f64 reference and term magnitude for x=1.
pub fn referenceOne(words: []const u32, scales: []const u16, biases: []const u16, k: usize, column: usize) [2]f64 {
    const groups = k / 64;
    var want: f64 = 0;
    var magnitude: f64 = 0;
    for (0..groups) |group| {
        const scale = bf16ToF64(scales[column * groups + group]);
        const bias = bf16ToF64(biases[column * groups + group]);
        var sum_q: f64 = 0;
        for (0..64) |offset| {
            const q: f64 = @floatFromInt(nibbleAt(words, column * k + group * 64 + offset));
            want += scale * q + bias;
            sum_q += q;
        }
        magnitude += @abs(scale) * sum_q + @abs(bias) * 64;
    }
    return .{ want, magnitude };
}

pub fn withinBound(got: f64, want: f64, allowance: f64) bool {
    return if (got == want) true else @abs(got - want) <= allowance;
}

pub const CasePlan = struct { shapes: [3][2]usize, head_fits: bool };

pub fn planCases(max_range: u64) CasePlan {
    var result = CasePlan{ .shapes = cases, .head_fits = false };
    if (packedWeightBytesFor(cases[2][0], cases[2][1]) <= max_range) {
        result.head_fits = true;
        return result;
    }
    for (fallback_shapes) |shape| {
        if (packedWeightBytesFor(shape[0], shape[1]) <= max_range) {
            result.shapes[2] = shape;
            return result;
        }
    }
    result.shapes[2] = fallback_shapes[fallback_shapes.len - 1];
    return result;
}

fn checkSpec() c.Spec {
    return .{ .name = "qmm-floor pk self-check", .accum = .f32, .depth = fxpk.depth, .out = .bf16, .served = &.{} };
}

/// Re-exports for tf-qmm-multi: Zig forbids a file living in two modules, so
/// the multi probe takes the push-constant layout and the pk spec from here
/// instead of importing contract_qmm.zig itself. Behavior-neutral for this
/// probe's binary.
pub const Params = qmm.Params;

pub fn pkSpec() c.Spec {
    return checkSpec();
}

fn us(ns: u64) f64 { return @as(f64, @floatFromInt(ns)) / 1000.0; }

fn printStats(n: usize, stats: Stats) void {
    std.debug.print("    N={d: >3}: min={d:.1} us p50={d:.1} us p95={d:.1} us max={d:.1} us\n", .{ n, us(stats.min_ns), us(stats.p50_ns), us(stats.p95_ns), us(stats.max_ns) });
}

fn sampleBatch(io: std.Io, cmds: *vk.Commands, staging: vk.Buffer, pipes: []const *dispatch.Pipeline, bindings: []const dispatch.Binding, push: []const u8, groups: [3]u32, n: usize, timeout: u64) !u64 {
    try vk.orExit(cmds.beginBatch(staging));
    errdefer cmds.abortBatch();
    const t0 = std.Io.Clock.awake.now(io);
    for (0..n) |i| {
        const pipe = pipes[i * pipes.len / n];
        try vk.orExit(pipe.bind(bindings));
        try vk.orExit(pipe.record(try cmds.begin(), push, groups));
    }
    try vk.orExit(cmds.endBatch(timeout));
    const t1 = std.Io.Clock.awake.now(io);
    return @intCast(t1.nanoseconds - t0.nanoseconds);
}

fn runCase(io: std.Io, cmds: *vk.Commands, staging: vk.Buffer, pipes: []const *dispatch.Pipeline, bindings: []const dispatch.Binding, push: []const u8, groups: [3]u32, n: usize, timeout: u64, out: *Stats) !void {
    var samples: [sample_count]u64 = undefined;
    for (0..warmups) |_| _ = try sampleBatch(io, cmds, staging, pipes, bindings, push, groups, n, timeout);
    for (0..sample_count) |i| samples[i] = try sampleBatch(io, cmds, staging, pipes, bindings, push, groups, n, timeout);
    var scratch: [sample_count]u64 = undefined;
    out.* = statsOf(&samples, &scratch).?;
}

fn runShape(io: std.Io, a: std.mem.Allocator, dev: *vk.Device, cmds: *vk.Commands, staging: vk.Buffer, pipe0: *dispatch.Pipeline, pipe1: *dispatch.Pipeline, n: usize, k: usize, timeout: u64) !bool {
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
    const host_out = try a.alloc(u16, n);
    defer a.free(host_out);
    const want = try a.alloc(f64, n);
    defer a.free(want);
    const magnitude = try a.alloc(f64, n);
    defer a.free(magnitude);
    const allowances = try a.alloc(f64, n);
    defer a.free(allowances);
    fillInputs(weights, scales, biases, x, 0x716d6d00 ^ @as(u64, n));
    const shape: c.Shape = .{ .rows = 1, .n = n, .k = k };
    const spec = checkSpec();
    for (0..n) |column| {
        const ref = referenceOne(weights, scales, biases, k, column);
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
    @memset(host_out, 0x7fc5);
    try vk.orExit(vk.upload(cmds, out_buf, 0, staging, std.mem.sliceAsBytes(host_out), timeout));
    const bindings = [_]dispatch.Binding{
        .{ .buffer = &x_buf, .len = x_buf.len },
        .{ .buffer = &w_buf, .len = w_buf.len },
        .{ .buffer = &scale_buf, .len = scale_buf.len },
        .{ .buffer = &bias_buf, .len = bias_buf.len },
        .{ .buffer = &out_buf, .len = out_buf.len },
    };
    const push = qmm.Params{ .count = @intCast(n), .output_size = @intCast(n), .matrix_m = 1, .matrix_n = @intCast(n), .matrix_k = @intCast(k), .flags = vk.ops_q4.pk_variant };
    const groups = try vk.ops_q4.pkGroups(n, dev.properties.limits.maxComputeWorkGroupCount[0]);
    try vk.orExit(pipe0.bind(&bindings));
    try vk.orExit(pipe0.run(cmds, std.mem.asBytes(&push), .{ groups, 1, 1 }, timeout));
    try vk.orExit(vk.download(cmds, out_buf, 0, staging, std.mem.sliceAsBytes(host_out), timeout));
    var worst: f64 = 0;
    var failed: ?usize = null;
    for (host_out, 0..) |bits, column| {
        const got = c.fromBf16(bits);
        const ratio = if (got == want[column]) 0 else @abs(got - want[column]) / allowances[column];
        worst = @max(worst, ratio);
        if (!(ratio <= 1) and failed == null) failed = column;
    }
    if (failed) |column| {
        std.debug.print("FAIL check: qmm_q4_pk n={d} k={d} output={d} got={e:.6} want={e:.6} bound-ratio={d:.3}\n", .{ n, k, column, c.fromBf16(host_out[column]), want[column], worst });
        return false;
    }
    std.debug.print("MILESTONE checks-passed n={d} k={d} outputs={d} worst={d:.3} of pk-contract bound; timing begins\n", .{ n, k, n, worst });
    const pipes = [_]*dispatch.Pipeline{ pipe0, pipe1 };
    var stats: [dispatch_counts.len]Stats = undefined;
    const push_bytes = std.mem.asBytes(&push);
    for (dispatch_counts, 0..) |count, i| {
        const selected: []const *dispatch.Pipeline = if (count > dispatch.batch_cap) &pipes else pipes[0..1];
        try runCase(io, cmds, staging, selected, &bindings, push_bytes, .{ groups, 1, 1 }, count, timeout, &stats[i]);
        printStats(count, stats[i]);
    }
    const p50_64 = try perDispatchNs(stats[3].p50_ns, stats[0].p50_ns, 64);
    const p50_16 = try perDispatchNs(stats[2].p50_ns, stats[0].p50_ns, 16);
    const min_64 = try perDispatchNs(stats[3].min_ns, stats[0].min_ns, 64);
    const bytes = weightBytesFor(n, k);
    const gbps = effectiveGbps(bytes, @intFromFloat(@max(p50_64, 1.0)));
    const min_gbps = effectiveGbps(bytes, @intFromFloat(@max(min_64, 1.0)));
    std.debug.print("    p50 slope (64-0)/64={d:.3} us/dispatch; p50 slope (16-0)/16={d:.3} us/dispatch; min slope (64-0)/64={d:.3} us/dispatch\n", .{ p50_64 / 1000.0, p50_16 / 1000.0, min_64 / 1000.0 });
    std.debug.print("    weight bytes/dispatch={d} (packed={d}, scales={d}, biases={d}); effective bandwidth p50={d:.2} GB/s min={d:.2} GB/s\n", .{ bytes, weight_bytes, param_bytes, param_bytes, gbps, min_gbps });
    if (add_floor_us) |f| std.debug.print("    p50/add-floor={d:.2}x (add floor {d:.3} us given on the command line)\n", .{ (p50_64 / 1000.0) / f, f });
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
    const opts = parseArgs(args[1..]) orelse { std.debug.print("{s}", .{usage}); return 2; };
    add_floor_us = opts.add_floor_us;
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
    const layout: dispatch.Layout = .{ .bindings = 5, .push_bytes = @sizeOf(qmm.Params), .writes = 1 << 4 };
    var pipe0 = try dispatch.Pipeline.init(&dev, &shaders.qmm_q4_pk, layout);
    defer pipe0.deinit();
    var pipe1 = try dispatch.Pipeline.init(&dev, &shaders.qmm_q4_pk, layout);
    defer pipe1.deinit();
    if (!std.mem.eql(u32, &pipe0.local, &.{ 64, 1, 1 })) {
        std.debug.print("FAIL device: qmm_q4_pk local size is {any}, expected 64x1x1\n", .{pipe0.local});
        return 1;
    }
    const plan = planCases(pr.limits.maxStorageBufferRange);
    if (!plan.head_fits) std.debug.print("SKIP served head n=248320 k=2048: packed Q4 weights exceed maxStorageBufferRange={d}; using largest listed fallback n={d} k={d}\n", .{ pr.limits.maxStorageBufferRange, plan.shapes[2][0], plan.shapes[2][1] });
    var completed: usize = 0;
    for (plan.shapes) |shape| {
        const n = shape[0];
        const k = shape[1];
        std.debug.print("MILESTONE case n={d} k={d} weight-bytes-per-dispatch={d}\n", .{ n, k, weightBytesFor(n, k) });
        if (!try runShape(io, a, &dev, &cmds, staging, &pipe0, &pipe1, n, k, timeout)) return 1;
        completed += 1;
    }
    std.debug.print("RESULT qmm-floor on \"{s}\": cases={d} ok\n", .{ name, completed });
    return 0;
}

test "argument parser validates vendor and timeout options" {
    const ok = parseArgs(&.{ "--vendor", "0x106b", "--timeout-ms", "90000" }).?;
    try std.testing.expectEqual(@as(?u32, 0x106b), ok.vendor);
    try std.testing.expectEqual(@as(u64, 90000), ok.timeout_ms);
    try std.testing.expect(parseArgs(&.{ "--vendor" }) == null);
    try std.testing.expect(parseArgs(&.{ "--vendor", "x" }) == null);
    try std.testing.expect(parseArgs(&.{ "--timeout-ms", "0" }) == null);
    try std.testing.expect(parseArgs(&.{ "--timeout-ms", "600001" }) == null);
    try std.testing.expect(parseArgs(&.{"--wat"}) == null);
    try std.testing.expect(parseArgs(&.{ "/m", "/i" }) == null);
}

test "the add floor comes only from the command line and must be a positive finite number" {
    try std.testing.expectEqual(@as(?f64, null), parseArgs(&.{}).?.add_floor_us);
    try std.testing.expectEqual(@as(?f64, 5.127), parseArgs(&.{ "--add-floor-us", "5.127" }).?.add_floor_us);
    try std.testing.expect(parseArgs(&.{ "--add-floor-us", "0" }) == null);
    try std.testing.expect(parseArgs(&.{ "--add-floor-us", "-1" }) == null);
    try std.testing.expect(parseArgs(&.{ "--add-floor-us", "nan" }) == null);
    try std.testing.expect(parseArgs(&.{ "--add-floor-us", "inf" }) == null);
    try std.testing.expect(parseArgs(&.{"--add-floor-us"}) == null);
}

test "stats use nearest-rank and slopes subtract the empty-batch floor" {
    try std.testing.expectEqual(@as(?usize, null), rankIndex(50, 0));
    try std.testing.expectEqual(@as(usize, 31), rankIndex(50, 63).?);
    try std.testing.expectEqual(@as(usize, 59), rankIndex(95, 63).?);
    const samples = [_]u64{ 7, 3, 9, 1, 5 };
    var scratch: [5]u64 = undefined;
    const stats = statsOf(&samples, &scratch).?;
    try std.testing.expectEqual(@as(u64, 1), stats.min_ns);
    try std.testing.expectEqual(@as(u64, 5), stats.p50_ns);
    try std.testing.expectEqual(@as(u64, 9), stats.p95_ns);
    try std.testing.expectEqual(@as(u64, 9), stats.max_ns);
    try std.testing.expectEqual(@as(f64, 2), try perDispatchNs(1000, 0, 500));
    try std.testing.expectError(error.ZeroN, perDispatchNs(1, 0, 0));
    try std.testing.expectEqual(@as(f64, 8), effectiveGbps(8, 1));
}

test "input encoding uses packed Q4 and normal bf16 values with a unit activation" {
    var words: [32]u32 = undefined;
    var scales: [16]u16 = undefined;
    var biases: [16]u16 = undefined;
    var x: [64]u16 = undefined;
    fillInputs(&words, &scales, &biases, &x, 0x716d6d00);
    for (x) |v| try std.testing.expectEqual(@as(u16, 0x3f80), v);
    for (scales ++ biases) |v| try std.testing.expect(v & 0x7f80 != 0);
    for (0..256) |i| try std.testing.expect(nibbleAt(&words, i) <= 15);
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16RneBits(1));
}

test "the f64 reference matches the contract affine helper and uses the pk bound" {
    const shape: c.Shape = .{ .rows = 1, .n = 3, .k = 128 };
    var words: [48]u32 = undefined;
    var scales: [6]u16 = undefined;
    var biases: [6]u16 = undefined;
    var x: [128]u16 = undefined;
    fillInputs(&words, &scales, &biases, &x, 0x1234);
    var affine = try contract.affine.Affine.init(std.testing.allocator, shape, 4, 64);
    defer affine.deinit();
    affine.form = .dequantized;
    affine.off = 0;
    @memcpy(affine.x, &x);
    @memcpy(affine.scales, &scales);
    @memcpy(affine.biases, &biases);
    for (affine.codes, 0..) |*code, i| code.* = @intCast(nibbleAt(&words, i));
    var expected: [3]f64 = undefined;
    var magnitude: [3]f64 = undefined;
    try affine.reference(&expected, &magnitude);
    for (0..3) |column| {
        const ref = referenceOne(&words, &scales, &biases, 128, column);
        try std.testing.expectEqual(expected[column], ref[0]);
        try std.testing.expectEqual(magnitude[column], ref[1]);
        const allowance = c.bound(checkSpec(), shape, ref[0], ref[1]);
        try std.testing.expect(allowance > 0 and std.math.isFinite(allowance));
        try std.testing.expect(withinBound(ref[0], ref[0], allowance));
        try std.testing.expect(!withinBound(ref[0] + allowance * 2, ref[0], allowance));
    }
}

test "weight-byte accounting and head fallback follow storage binding limits" {
    try std.testing.expectEqual(@as(u64, 7_077_888), weightBytesFor(6144, 2048));
    try std.testing.expectEqual(@as(u64, 286_064_640), weightBytesFor(248320, 2048));
    const full = planCases(packedWeightBytesFor(248320, 2048));
    try std.testing.expect(full.head_fits);
    const fallback = planCases(packedWeightBytesFor(4096, 2048));
    try std.testing.expect(!fallback.head_fits);
    try std.testing.expectEqual(@as(usize, 4096), fallback.shapes[2][0]);
}
