//! The dispatch-floor probe: what one more recorded dispatch costs inside a
//! decode-token command buffer. Every sample is the engine's real batching
//! path — Commands.beginBatch, dispatch.Pipeline in batch mode (staged binds,
//! a fresh descriptor-set bank snapshot per dispatch), each record's trailing
//! end barrier, endBatch's one staging flush + one submit + one fence wait —
//! with the wall clock (std.Io.Clock.awake, the model probe's steadyToken
//! clock) read just before the first bind/record of the batch and again just
//! after endBatch returns. One command buffer, one submit, one fence wait per
//! sample; 63 samples after 3 warmups (warmups also absorb the descriptor
//! bank's first-use pool/set allocation).
//!
//! (A) N trivial dependent dispatches: the bf16 add (binary_vec.comp's default
//!     bf16 build, 2048 elements, out = a + b into a scratch distinct from the
//!     inputs), N in {0, 1, 32, 128, 256, 486, 972}; the per-dispatch cost is
//!     the slope of p50: (p50(486)-p50(0))/486 and the linearity check
//!     (p50(972)-p50(486))/486. N=0 is the empty-batch floor the slopes
//!     subtract. 972 records on ONE pipeline would overflow its bounded bank
//!     (dispatch.batch_cap = 512 snapshots per batch epoch; a real token never
//!     exceeds 489 on one pipeline), so the second 486 ride a second instance
//!     of the same image — every record still binds pipeline + set + pushes
//!     and dispatches + barriers exactly as before.
//! (B) the memory-heavy pure dispatch: copy_general.comp -DUSE_U16 copying
//!     6 MiB of bf16 (3145728 elements, one workgroup per 256 elements =
//!     12288 workgroups; u16 storage is bit-exact bf16), N in {0, 64, 128};
//!     ms per dispatch and effective GB/s with read+write counted.
//! (C) independent dispatches: skipped — dispatch.Pipeline.record always
//!     appends the pipeline's end barrier, and no barrier-free record path
//!     exists in the Pipeline API.
//!
//! Exit codes follow the other probes: --help exits 0 before any loader or
//! device; bad usage exits 2; a fence timeout exits 124 and any unknown-state
//! submit failure exits 1 (inside vk.orExit); a failed probe exits 1. The
//! host tests cover the parser, the sample statistics and the slope
//! arithmetic without a GPU.
const std = @import("std");
const vk = @import("vulkan");
const shaders = @import("floor_shaders");
const c = vk.c;
const dispatch = vk.dispatch;
const params = vk.params;

const usage = "usage: tf-dispatch-floor [--vendor ID] [--timeout-ms MS]\n" ++
    "  --help            show this text and exit without opening a loader or device\n" ++
    "  --vendor ID       refuse a device whose Vulkan vendorID differs (decimal or 0x hex; Apple is 0x10005 on the Honeykrisp devices; no default)\n" ++
    "  --timeout-ms MS   bound on every fence wait, 1 to 600000 (default 10000); a timeout ends the process\n";

const warmups = 3;
const sample_count = 63;
const add_elements = 2048; // 4-wide shader: 8 workgroups of 256
const add_push_bytes = params.cast_push_bytes; // binary_vec reads only the first nine words
const copy_bytes = 6 << 20; // 6 MiB of bf16
const copy_elements = copy_bytes / 2;
const staging_bytes = 1 << 20;
const case_a_ns = [_]usize{ 0, 1, 32, 128, 256, 486, 972 };
const case_b_ns = [_]usize{ 0, 64, 128 };

/// binary_vec.comp's push block, its first nine words; sizes in elements.
const AddPush = extern struct {
    count: u32,
    operation: u32,
    lhs_size: u32,
    rhs_size: u32,
    reduce_size: u32 = 0,
    output_size: u32 = 0,
    lhs_offset: u32 = 0,
    rhs_offset: u32 = 0,
    output_offset: u32 = 0,
};

comptime {
    if (@sizeOf(AddPush) != add_push_bytes) @compileError("AddPush drifted from binary_vec's nine-word push prefix");
}

pub const Options = struct {
    vendor: ?u32 = null,
    timeout_ms: u64 = 10_000,
};

/// Flags only, no positionals; unknown flags and bad values are usage errors.
/// Pure and host-tested (the probe's only argument validator — no duplicate).
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
        } else return null;
    }
    return o;
}

pub const Stats = struct { min_ns: u64, p50_ns: u64, p95_ns: u64, max_ns: u64 };

/// Nearest-rank p/100 over n ascending samples: the ceil(p*n/100)-th, 0-based.
/// Pure and host-tested.
pub fn rankIndex(p: u32, n: usize) ?usize {
    if (n == 0) return null;
    return (p * n + 99) / 100 - 1; // ceil(p*n/100) without floats
}

/// min/p50/p95/max of a copy of the samples, or null on empty input or a too
/// small scratch; pure and host-tested (the probe's only summarizer).
pub fn statsOf(samples: []const u64, scratch: []u64) ?Stats {
    if (samples.len == 0 or scratch.len < samples.len) return null;
    @memcpy(scratch[0..samples.len], samples);
    std.mem.sort(u64, scratch[0..samples.len], {}, std.sort.asc(u64));
    const s = scratch[0..samples.len];
    return .{
        .min_ns = s[0],
        .p50_ns = s[rankIndex(50, s.len).?],
        .p95_ns = s[rankIndex(95, s.len).?],
        .max_ns = s[s.len - 1],
    };
}

/// (p50(high) - p50(low)) / n in nanoseconds; the N=0 division refuses — the
/// empty batch's floor is subtracted, never divided by. Pure and host-tested.
pub fn perDispatchNs(high_p50_ns: u64, low_p50_ns: u64, n: u64) error{ZeroN}!f64 {
    if (n == 0) return error.ZeroN;
    return (@as(f64, @floatFromInt(high_p50_ns)) - @as(f64, @floatFromInt(low_p50_ns))) / @as(f64, @floatFromInt(n));
}

/// Bytes moved per nanosecond is exactly GB/s (10^9 bytes per second).
pub fn effectiveGbps(bytes_moved: u64, ns: u64) f64 {
    return @as(f64, @floatFromInt(bytes_moved)) / @as(f64, @floatFromInt(ns));
}

/// binary_vec.comp's bf16 store: round to nearest even, NaN keeps the quiet
/// bit — the host reference the add check compares against.
fn bf16RneBits(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    if (std.math.isNan(value)) return @intCast((bits >> 16) | 0x40);
    return @intCast((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

fn us(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1000.0;
}

fn printStats(n: usize, st: Stats) void {
    std.debug.print("    N={d: >3}: min={d:.1} us p50={d:.1} us p95={d:.1} us max={d:.1} us\n", .{
        n, us(st.min_ns), us(st.p50_ns), us(st.p95_ns), us(st.max_ns),
    });
}

/// One sample: the whole batch cycle for N dispatches of one shape, timed
/// from just before the first bind/record to just after the fence returns.
fn sampleBatch(
    io: std.Io,
    cmds: *vk.Commands,
    staging: vk.Buffer,
    pipes: []const *dispatch.Pipeline,
    binds: []const dispatch.Binding,
    push: []const u8,
    groups: [3]u32,
    n: usize,
    timeout: u64,
) !u64 {
    try vk.orExit(cmds.beginBatch(staging));
    errdefer cmds.abortBatch();
    const t0 = std.Io.Clock.awake.now(io);
    for (0..n) |i| {
        const pipe = pipes[i * pipes.len / n];
        try vk.orExit(pipe.bind(binds)); // staged while the batch is open, snapshotted at record
        try vk.orExit(pipe.record(try cmds.begin(), push, groups)); // dispatch + the end barrier
    }
    try vk.orExit(cmds.endBatch(timeout)); // flush, one submit, one fence wait, invalidate
    const t1 = std.Io.Clock.awake.now(io);
    return @intCast(t1.nanoseconds - t0.nanoseconds);
}

/// 3 warmups, then 63 timed samples of one N; the summary lands in `out`.
fn runCase(
    io: std.Io,
    cmds: *vk.Commands,
    staging: vk.Buffer,
    pipes: []const *dispatch.Pipeline,
    binds: []const dispatch.Binding,
    push: []const u8,
    groups: [3]u32,
    n: usize,
    timeout: u64,
    out: *Stats,
) !void {
    var samples: [sample_count]u64 = undefined;
    for (0..warmups) |_| _ = try sampleBatch(io, cmds, staging, pipes, binds, push, groups, n, timeout);
    for (0..sample_count) |i| samples[i] = try sampleBatch(io, cmds, staging, pipes, binds, push, groups, n, timeout);
    var scratch: [sample_count]u64 = undefined;
    out.* = statsOf(&samples, &scratch).?; // fixed sizes: cannot fail
}

pub fn main(init: std.process.Init) !u8 {
    const io = init.io;
    const a = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    for (args[1..]) |arg| if (std.mem.eql(u8, arg, "--help")) {
        std.debug.print(usage, .{});
        return 0;
    };
    const opts = parseArgs(args[1..]) orelse {
        std.debug.print(usage, .{});
        return 2;
    };
    const timeout = opts.timeout_ms * std.time.ns_per_ms;

    var rt = try vk.Runtime.open();
    defer rt.close();
    var dev = try vk.Device.open(&rt, .{}); // refuses CPU (software) devices and Vulkan below 1.3
    defer dev.close();
    const pr = dev.properties;
    const name = std.mem.sliceTo(&pr.deviceName, 0);
    const types = [_][]const u8{ "other", "integrated", "discrete", "virtual", "cpu" };
    const kind = if (pr.deviceType < types.len) types[pr.deviceType] else "unknown";
    std.debug.print("MILESTONE device-found name=\"{s}\" vendor=0x{x:0>4} device=0x{x:0>4} type={s} timeout={d}ms\n", .{ name, pr.vendorID, pr.deviceID, kind, opts.timeout_ms });
    if (pr.deviceType == c.VK_PHYSICAL_DEVICE_TYPE_CPU) {
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

    const add_layout: dispatch.Layout = .{ .bindings = 3, .push_bytes = add_push_bytes, .writes = 1 << 2 };
    var add_pipe0 = try dispatch.Pipeline.init(&dev, &shaders.add_bf16, add_layout);
    defer add_pipe0.deinit();
    // One pipeline's descriptor bank caps at dispatch.batch_cap snapshots per
    // batch; 972 records ride a second instance of the same image.
    var add_pipe1 = try dispatch.Pipeline.init(&dev, &shaders.add_bf16, add_layout);
    defer add_pipe1.deinit();
    var copy_pipe = try dispatch.Pipeline.init(&dev, &shaders.copy_u16, .{ .bindings = 4, .push_bytes = params.push_bytes, .writes = 1 << 2 });
    defer copy_pipe.deinit();

    var a_buf = try vk.Buffer.init(&dev, .device, add_elements * 2);
    defer a_buf.deinit();
    var b_buf = try vk.Buffer.init(&dev, .device, add_elements * 2);
    defer b_buf.deinit();
    var out_buf = try vk.Buffer.init(&dev, .device, add_elements * 2);
    defer out_buf.deinit();
    var copy_src = try vk.Buffer.init(&dev, .device, copy_bytes);
    defer copy_src.deinit();
    var copy_dst = try vk.Buffer.init(&dev, .device, copy_bytes);
    defer copy_dst.deinit();
    var dummy = try vk.Buffer.init(&dev, .device, 256); // copy's never-read bindings 1 and 3
    defer dummy.deinit();

    // Host inputs: exact-integer bf16 adds (every bf16 here is a normal number and the RNE round is exact below 256),
    // pseudo-random bytes for the copy.
    var add_a: [add_elements]u16 = undefined;
    var add_b: [add_elements]u16 = undefined;
    var add_want: [add_elements]u16 = undefined;
    for (&add_a, &add_b, &add_want, 0..) |*x, *y, *w, i| {
        const xv: f32 = @floatFromInt(i % 61);
        const yv: f32 = @floatFromInt((i * 7) % 23 + 1);
        // bf16 BITS of the integers; raw 0..60 would be bf16 subnormals, which the device flushes.
        x.* = bf16RneBits(xv);
        y.* = bf16RneBits(yv);
        w.* = bf16RneBits(xv + yv);
    }
    const copy_want = try a.alloc(u8, copy_bytes);
    defer a.free(copy_want);
    var prng = std.Random.DefaultPrng.init(0xd15f1000);
    prng.random().bytes(copy_want);

    try vk.orExit(vk.upload(&cmds, a_buf, 0, staging, std.mem.sliceAsBytes(&add_a), timeout));
    try vk.orExit(vk.upload(&cmds, b_buf, 0, staging, std.mem.sliceAsBytes(&add_b), timeout));
    try vk.orExit(vk.upload(&cmds, copy_src, 0, staging, copy_want, timeout));

    // Prove the timed dispatches are real work before timing anything: one
    // immediate add and one immediate copy into zeroed scratch, byte-exact
    // against the host references.
    try vk.orExit(vk.zero(&cmds, out_buf, 0, out_buf.len, timeout));
    const add_push = AddPush{ .count = add_elements, .operation = 0, .lhs_size = add_elements, .rhs_size = add_elements };
    const add_binds = [_]dispatch.Binding{
        .{ .buffer = &a_buf, .len = a_buf.len },
        .{ .buffer = &b_buf, .len = b_buf.len },
        .{ .buffer = &out_buf, .len = out_buf.len },
    };
    const add_groups = [3]u32{ add_elements / 256, 1, 1 };
    try add_pipe0.bind(&add_binds);
    try vk.orExit(add_pipe0.run(&cmds, std.mem.asBytes(&add_push)[0..add_push_bytes], add_groups, timeout));
    var add_got: [add_elements]u16 = undefined;
    try vk.orExit(vk.download(&cmds, out_buf, 0, staging, std.mem.sliceAsBytes(&add_got), timeout));
    if (!std.mem.eql(u16, &add_got, &add_want)) {
        std.debug.print("FAIL check: the bf16 add's output differs from the host RNE reference\n", .{});
        return 1;
    }
    try vk.orExit(vk.zero(&cmds, copy_dst, 0, copy_dst.len, timeout));
    const copy_binds = [_]dispatch.Binding{
        .{ .buffer = &copy_src, .len = copy_src.len },
        .{ .buffer = &dummy, .len = dummy.len },
        .{ .buffer = &copy_dst, .len = copy_dst.len },
        .{ .buffer = &dummy, .len = dummy.len },
    };
    var cp: params.Params = .{
        .count = copy_elements,
        .dims = 1,
        .shape = .{ copy_elements, 0, 0, 0 },
        .in_strides = .{ 1, 0, 0, 0 },
        .out_strides = .{ 1, 0, 0, 0 },
    };
    const copy_groups = [3]u32{ copy_elements / 256, 1, 1 };
    try copy_pipe.bind(&copy_binds);
    try vk.orExit(copy_pipe.run(&cmds, std.mem.asBytes(&cp), copy_groups, timeout));
    const copy_got = try a.alloc(u8, copy_bytes);
    defer a.free(copy_got);
    try vk.orExit(vk.download(&cmds, copy_dst, 0, staging, copy_got, timeout));
    if (!std.mem.eql(u8, copy_got, copy_want)) {
        std.debug.print("FAIL check: the 6 MiB copy's output differs from its source\n", .{});
        return 1;
    }
    std.debug.print("MILESTONE checks-passed add={d} elements copy={d} bytes, byte-exact; timing begins\n", .{ add_elements, copy_bytes });

    // ---- (A) dependent dispatches
    const add_pipes = [_]*dispatch.Pipeline{ &add_pipe0, &add_pipe1 };
    std.debug.print("(A) dependent dispatches: bf16 add out = a + b, {d} elements (8 workgroups of 256), one batch per sample; bindings re-bound and bank-snapshotted per dispatch, the pipeline's end barrier between dispatches; N=0 is the empty-batch floor\n", .{add_elements});
    var a_stats: [case_a_ns.len]Stats = undefined;
    for (case_a_ns, 0..) |n, i| {
        const pipes: []const *dispatch.Pipeline = if (n > dispatch.batch_cap) &add_pipes else add_pipes[0..1];
        try runCase(io, &cmds, staging, pipes, &add_binds, std.mem.asBytes(&add_push)[0..add_push_bytes], add_groups, n, timeout, &a_stats[i]);
        printStats(n, a_stats[i]);
    }
    // case_a_ns[5] is 486, case_a_ns[6] is 972 — guarded so a list edit fails here
    comptime {
        if (case_a_ns[5] != 486 or case_a_ns[6] != 972 or case_a_ns[0] != 0) @compileError("slope indices drifted from case_a_ns");
    }
    std.debug.print("    slope (p50(486)-p50(0))/486 = {d:.3} us per dispatch\n", .{(try perDispatchNs(a_stats[5].p50_ns, a_stats[0].p50_ns, 486)) / 1000.0});
    std.debug.print("    slope (p50(972)-p50(486))/486 = {d:.3} us per dispatch (linearity check; 972 rides two pipeline instances past the 512-snapshot bank)\n", .{(try perDispatchNs(a_stats[6].p50_ns, a_stats[5].p50_ns, 486)) / 1000.0});

    // ---- (B) memory-heavy dispatches
    std.debug.print("(B) memory-heavy dispatches: bf16-width copy of {d} MiB ({d} elements, {d} workgroups of 256), same batch path, N in {{0, 64, 128}}\n", .{ copy_bytes >> 20, copy_elements, copy_elements / 256 });
    var b_stats: [case_b_ns.len]Stats = undefined;
    const copy_pipes = [_]*dispatch.Pipeline{&copy_pipe};
    for (case_b_ns, 0..) |n, i| {
        try runCase(io, &cmds, staging, &copy_pipes, &copy_binds, std.mem.asBytes(&cp), copy_groups, n, timeout, &b_stats[i]);
        printStats(n, b_stats[i]);
    }
    comptime {
        if (case_b_ns[1] != 64 or case_b_ns[2] != 128 or case_b_ns[0] != 0) @compileError("slope indices drifted from case_b_ns");
    }
    for ([_]struct { i: usize, n: usize }{ .{ .i = 1, .n = 64 }, .{ .i = 2, .n = 128 } }) |e| {
        const per_ns = try perDispatchNs(b_stats[e.i].p50_ns, b_stats[0].p50_ns, e.n);
        // a negative slope (noise below the floor) prints, but the GB/s line clamps to 1 ns
        const gbps_ns: u64 = @intFromFloat(@max(per_ns, 1.0));
        std.debug.print("    per-dispatch (p50({d})-p50(0))/{d} = {d:.4} ms, effective {d:.1} GB/s (12 MiB read+write per dispatch)\n", .{
            e.n, e.n, per_ns / 1e6, effectiveGbps(2 * copy_bytes, gbps_ns),
        });
    }

    // ---- (C) independent dispatches: not recordable on the engine's real path
    std.debug.print("(C) skipped: dispatch.Pipeline.record always appends the pipeline's end barrier; the Pipeline API has no barrier-free record path, so independent dispatches cannot ride the engine's real recording path\n", .{});

    std.debug.print("RESULT dispatch-floor on \"{s}\": cases=2 ok\n", .{name});
    return 0;
}

const testing = std.testing;

test "parseArgs takes the two flags, bounds the timeout and refuses everything else" {
    const ok = parseArgs(&.{ "--vendor", "0x106b", "--timeout-ms", "90000" }).?;
    try testing.expectEqual(@as(?u32, 0x106b), ok.vendor);
    try testing.expectEqual(@as(u64, 90000), ok.timeout_ms);
    const plain = parseArgs(&.{}).?;
    try testing.expect(plain.vendor == null);
    try testing.expectEqual(@as(u64, 10_000), plain.timeout_ms);
    try testing.expectEqual(@as(u32, 0x10005), parseArgs(&.{ "--vendor", "65541" }).?.vendor.?); // decimal Honeykrisp Apple id
    try testing.expect(parseArgs(&.{"--vendor"}) == null); // missing value
    try testing.expect(parseArgs(&.{ "--vendor", "x" }) == null);
    try testing.expect(parseArgs(&.{ "--timeout-ms", "0" }) == null);
    try testing.expect(parseArgs(&.{ "--timeout-ms", "600001" }) == null);
    try testing.expect(parseArgs(&.{"--wat"}) == null);
    try testing.expect(parseArgs(&.{ "/m", "/i" }) == null); // no positionals accepted
}

test "rankIndex is nearest-rank and statsOf summarizes a fixed sample set" {
    try testing.expectEqual(@as(?usize, null), rankIndex(50, 0));
    try testing.expectEqual(@as(usize, 0), rankIndex(50, 1).?);
    try testing.expectEqual(@as(usize, 31), rankIndex(50, 63).?);
    try testing.expectEqual(@as(usize, 59), rankIndex(95, 63).?);
    const samples = [_]u64{ 7, 3, 9, 1, 5 };
    var scratch: [8]u64 = undefined;
    const st = statsOf(&samples, &scratch).?;
    try testing.expectEqual(@as(u64, 1), st.min_ns);
    try testing.expectEqual(@as(u64, 5), st.p50_ns); // ceil(2.5)=3rd of sorted {1,3,5,7,9}
    try testing.expectEqual(@as(u64, 9), st.p95_ns); // ceil(4.75)=5th
    try testing.expectEqual(@as(u64, 9), st.max_ns);
    try testing.expect(statsOf(&.{}, &scratch) == null); // empty refuses
    try testing.expect(statsOf(&samples, scratch[0..4]) == null); // scratch too small
}

test "per-dispatch slopes subtract the floor and refuse the N=0 division" {
    try testing.expectEqual(@as(f64, 2.0), try perDispatchNs(1000, 0, 500));
    try testing.expectEqual(@as(f64, 1000.0), try perDispatchNs(2000, 1000, 1));
    try testing.expectError(error.ZeroN, perDispatchNs(2000, 1000, 0));
    try testing.expect(try perDispatchNs(1000, 2000, 1) < 0); // noise reports negative, never refuses
}

test "effectiveGbps is bytes per nanosecond and bf16RneBits matches the shader formula" {
    try testing.expectEqual(@as(f64, 8.0), effectiveGbps(8, 1));
    try testing.expectEqual(@as(f64, 12.0), effectiveGbps(12 * 1024 * 1024, 1024 * 1024));
    try testing.expectEqual(@as(u16, 0x3f80), bf16RneBits(1.0));
    try testing.expectEqual(@as(u16, 0x4240), bf16RneBits(48.0));
    try testing.expectEqual(@as(u16, 0x4241), bf16RneBits(48.25)); // representable: identity
    try testing.expectEqual(@as(u16, 0x4240), bf16RneBits(48.125)); // tie: even kept mantissa rounds down
    try testing.expectEqual(@as(u16, 0x4242), bf16RneBits(48.375)); // tie: odd kept mantissa rounds up
}

test "the add inputs are normal bf16 values, not raw small integers" {
    // The first probe wrote i % 61 as raw bits (bf16 subnormals) and failed its own check on a device that flushes denormals.
    var i: u32 = 1;
    while (i < 61) : (i += 1) {
        const bits = bf16RneBits(@floatFromInt(i));
        try std.testing.expect((bits & 0x7f80) != 0); // exponent nonzero: normal
        const back: f32 = @bitCast(@as(u32, bits) << 16);
        try std.testing.expectEqual(@as(f32, @floatFromInt(i)), back);
    }
}
