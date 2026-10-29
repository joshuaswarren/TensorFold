//! Private bandwidth-ceiling probe: how many GB/s can this GPU stream, and
//! does that speed depend on the GPU's power state? Our matmul probe measured
//! 89 GB/s on a chip rated near 400 — but that number came from a copy shader
//! moving 16 bits per thread, so it is a lower bound. Here every invocation
//! streams one uvec4 (16 B) with a grid-stride loop, in three integer-only
//! kernels (no float arithmetic anywhere, so no denormal or NaN behaviour):
//! COPY (read a uvec4, write a uvec4), READ (xor-reduce, one word stored per
//! workgroup so the compiler cannot drop the loads) and WRITE (a pattern).
//!
//! Cases, in run order:
//!   0. a byte-exact self-check on a 1 MiB buffer (copy against host bytes,
//!      xor-reduce against the host fold, write against the host pattern) —
//!      a check failure exits 1 before any timing.
//!   1. COLD-TO-WARM RAMP, the first timed work: 32 batches of 4 dependent
//!      256 MiB copies, each batch's GB/s printed in order, so a memory-clock
//!      ramp shows as a rising series; first-batch GB/s vs the median of the
//!      last 8.
//!   2. SIZE SWEEP, warm: COPY at 4/16/64/256/512/1024 MiB with N=0 and
//!      N=16/16/8/4/2/2 dependent dispatches per batch; READ at 256/512/1024
//!      MiB (N=0 then 4/2/2); WRITE at 256/1024 MiB (N=0 then 4/2). Effective
//!      GB/s = bytes moved per dispatch / per-dispatch time, the N=0 empty
//!      batch subtracted, p50 and min. A size the device cannot bind or
//!      allocate is skipped with a clear message and the run continues.
//!   3. IDLE RE-ENTRY: 3 s of host sleep with no GPU work, then one batch of
//!      4 dependent 256 MiB copies — does the memory clock fall back?
//!
//! The timed big buffers stream device-resident bytes whose values do not
//! affect bandwidth; the correctness gate is the small-buffer self-check.
//! One src and one dst buffer sized to the largest enabled sweep case serve
//! every size (the push-constant word count bounds each dispatch); total
//! device memory is 2 GiB at the full sweep.
//!
//! Exit codes follow the other probes: --help exits 0 before any loader or
//! device; bad usage exits 2; a fence timeout exits 124 and any unknown-state
//! submit failure exits 1 (inside vk.orExit); a failed probe exits 1. The
//! host tests cover the parser, the statistics, the ramp ratio, the
//! pattern/xor helpers and the sweep plan without a GPU.
const std = @import("std");
const vk = @import("vulkan");
const shaders = @import("bw_shaders");
const c = vk.c;
const dispatch = vk.dispatch;

const usage = "usage: tf-bw-ceiling [--vendor ID] [--timeout-ms MS] | --help\n" ++
    "  --help            show this text and exit without opening a loader or device\n" ++
    "  --vendor ID       refuse a device whose Vulkan vendorID differs (decimal or 0x hex; Apple is 0x10005 on the Honeykrisp devices; no default)\n" ++
    "  --timeout-ms MS   bound on every fence wait, 1 to 600000 (default 10000); a timeout ends the process\n";

const warmups = 3;
const sample_count = 63;
const staging_bytes = 1 << 20;
const check_bytes = 1 << 20; // the byte-exact gate runs here, before any timing
const ramp_batches = 32;
const ramp_dep = 4; // dependent copies per ramp/idle batch
const big_bytes: u64 = 256 << 20; // ramp + idle batch size; the smallest allocation worth running for
const idle_sleep_ms = 3000;
const groups_x: u32 = 4096; // 4096 x 256 = 1,048,576 invocations, grid-stride
const red_words = groups_x; // the READ kernel stores one xor word per workgroup
const copy_sizes = [_]u64{ 4 << 20, 16 << 20, 64 << 20, 256 << 20, 512 << 20, 1 << 30 };
const copy_dep = [_]usize{ 16, 16, 8, 4, 2, 2 };
const read_sizes = [_]u64{ 256 << 20, 512 << 20, 1 << 30 };
const read_dep = [_]usize{ 4, 2, 2 };
const write_sizes = [_]u64{ 256 << 20, 1 << 30 };
const write_dep = [_]usize{ 4, 2 };

comptime {
    if (copy_sizes.len != copy_dep.len or read_sizes.len != read_dep.len or write_sizes.len != write_dep.len)
        @compileError("sweep size and dispatch-count tables drifted");
    for (copy_sizes ++ read_sizes ++ write_sizes) |s| if (s % 4 != 0) @compileError("sweep sizes must be whole uvec4s");
    for (copy_dep ++ read_dep ++ write_dep) |n| if (n == 0 or n > dispatch.batch_cap) @compileError("sweep dispatch counts must be nonzero and fit one pipeline's descriptor bank");
}

/// The shader's push block, one word: the dispatch's coverage in u32 words.
const Push = extern struct { count_words: u32 };

pub const Options = struct { vendor: ?u32 = null, timeout_ms: u64 = 10_000 };

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
pub fn rankIndex(p: u32, n: usize) ?usize {
    if (n == 0) return null;
    return (p * n + 99) / 100 - 1; // ceil(p*n/100) without floats
}

/// min/p50/p95/max of a copy of the samples, or null on empty input or a too
/// small scratch. Pure and host-tested (the probe's only summarizer).
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

/// (high - low) / n in nanoseconds; the N=0 division refuses — the empty
/// batch's floor is subtracted, never divided by.
pub fn perDispatchNs(high_ns: u64, low_ns: u64, n: u64) error{ZeroN}!f64 {
    if (n == 0) return error.ZeroN;
    return (@as(f64, @floatFromInt(high_ns)) - @as(f64, @floatFromInt(low_ns))) / @as(f64, @floatFromInt(n));
}

/// Bytes moved per nanosecond is exactly GB/s (10^9 bytes per second).
pub fn effectiveGbps(bytes_moved: u64, ns: u64) f64 {
    return @as(f64, @floatFromInt(bytes_moved)) / @as(f64, @floatFromInt(ns));
}

/// Warm GB/s over cold GB/s: how many times faster the GPU streams once it
/// has been busy. Pure and host-tested.
pub fn rampRatio(warm_gbps: f64, cold_gbps: f64) f64 {
    return warm_gbps / cold_gbps;
}

/// The host mirror of stream_u32x4.comp's pattern(): what STREAM_WRITE must
/// land in words 4i..4i+3. Integer ops only; u32 wrap-around on both sides.
/// Pure and host-tested.
pub fn writePattern(i: u32) [4]u32 {
    return .{ i, ~i, i ^ 0xA5A5_A5A5, i *% 7 +% 3 };
}

/// Deterministic u32 fill for the copy self-check; same seed, same bytes, and
/// different seeds disagree. Pure and host-tested.
pub fn fillPseudo(words: []u32, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    for (words) |*w| w.* = rng.int(u32);
}

/// Xor-fold of the words — the host's expected READ result.
pub fn xorWords(words: []const u32) u32 {
    var acc: u32 = 0;
    for (words) |w| acc ^= w;
    return acc;
}

pub const SweepPlan = struct {
    /// The one src/dst allocation covering every enabled size.
    len: u64,
    copy: [copy_sizes.len]bool,
    read: [read_sizes.len]bool,
    write: [write_sizes.len]bool,
};

/// Sweep sizes the device's storage-buffer limits allow; `len` never exceeds
/// 1 GiB (the full sweep's allocation). Pure and host-tested.
pub fn planSweep(max_range: u64, max_buffer: u64) SweepPlan {
    const limit = @min(max_range, max_buffer);
    var p: SweepPlan = .{ .len = @min(@as(u64, 1 << 30), limit), .copy = undefined, .read = undefined, .write = undefined };
    for (copy_sizes, 0..) |s, i| p.copy[i] = s <= limit;
    for (read_sizes, 0..) |s, i| p.read[i] = s <= limit;
    for (write_sizes, 0..) |s, i| p.write[i] = s <= limit;
    return p;
}

fn us(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1000.0;
}

fn mib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0);
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

/// N=0 floor plus N dependent dispatches, the per-dispatch p50/min derived by
/// subtraction, and effective GB/s from `bytes_moved` per dispatch.
fn sweepCase(
    io: std.Io,
    cmds: *vk.Commands,
    staging: vk.Buffer,
    pipes: []const *dispatch.Pipeline,
    binds: []const dispatch.Binding,
    push: []const u8,
    groups: [3]u32,
    n: usize,
    bytes_moved: u64,
    timeout: u64,
) !void {
    var s0: Stats = undefined;
    var s1: Stats = undefined;
    try runCase(io, cmds, staging, pipes, binds, push, groups, 0, timeout, &s0);
    try runCase(io, cmds, staging, pipes, binds, push, groups, n, timeout, &s1);
    printStats(0, s0);
    printStats(n, s1);
    const per50 = try perDispatchNs(s1.p50_ns, s0.p50_ns, @intCast(n));
    const perMin = try perDispatchNs(s1.min_ns, s0.min_ns, @intCast(n));
    const g50 = effectiveGbps(bytes_moved, @intFromFloat(@max(per50, 1.0)));
    const gMin = effectiveGbps(bytes_moved, @intFromFloat(@max(perMin, 1.0)));
    std.debug.print("    per-dispatch p50={d:.3} ms min={d:.3} ms -> {d:.1} MiB/dispatch: {d:.2} GB/s (p50), {d:.2} GB/s (min)\n", .{
        per50 / 1e6, perMin / 1e6, mib(bytes_moved), g50, gMin,
    });
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

    // Sweep plan against the device's real binding limits; sizes above them
    // are skipped with a message and the rest of the run continues.
    var plan = planSweep(pr.limits.maxStorageBufferRange, dev.v13.maxBufferSize);
    for (copy_sizes, plan.copy) |size, on| if (!on) {
        std.debug.print("SKIP sweep copy {d:.0} MiB: exceeds {s}={d}\n", .{ mib(size), if (size > pr.limits.maxStorageBufferRange) "maxStorageBufferRange" else "maxBufferSize", @min(pr.limits.maxStorageBufferRange, dev.v13.maxBufferSize) });
    };
    for (read_sizes, plan.read) |size, on| if (!on) {
        std.debug.print("SKIP sweep read {d:.0} MiB: exceeds {s}={d}\n", .{ mib(size), if (size > pr.limits.maxStorageBufferRange) "maxStorageBufferRange" else "maxBufferSize", @min(pr.limits.maxStorageBufferRange, dev.v13.maxBufferSize) });
    };
    for (write_sizes, plan.write) |size, on| if (!on) {
        std.debug.print("SKIP sweep write {d:.0} MiB: exceeds {s}={d}\n", .{ mib(size), if (size > pr.limits.maxStorageBufferRange) "maxStorageBufferRange" else "maxBufferSize", @min(pr.limits.maxStorageBufferRange, dev.v13.maxBufferSize) });
    };
    if (plan.len < big_bytes) {
        std.debug.print("FAIL device: the largest allocatable sweep buffer is {d:.0} MiB; the ramp and idle cases need {d:.0} MiB\n", .{ mib(plan.len), mib(big_bytes) });
        return 1;
    }

    // One src and one dst pair sized for the largest enabled case; an
    // allocation failure steps down and re-filters instead of ending the run.
    var src: vk.Buffer = undefined;
    var dst: vk.Buffer = undefined;
    while (true) {
        src = vk.Buffer.init(&dev, .device, plan.len) catch {
            plan.len /= 2;
            if (plan.len < big_bytes) break;
            continue;
        };
        dst = vk.Buffer.init(&dev, .device, plan.len) catch {
            src.deinit();
            plan.len /= 2;
            if (plan.len < big_bytes) break;
            continue;
        };
        break;
    }
    if (plan.len < big_bytes) {
        std.debug.print("FAIL device: could not allocate a {d:.0} MiB src/dst pair; the ramp and idle cases need it\n", .{mib(big_bytes)});
        return 1;
    }
    for (copy_sizes, 0..) |size, i| if (plan.copy[i] and size > plan.len) {
        plan.copy[i] = false;
        std.debug.print("SKIP sweep copy {d:.0} MiB: the device refused a {d:.0} MiB allocation\n", .{ mib(size), mib(size * 2) });
    };
    for (read_sizes, 0..) |size, i| if (plan.read[i] and size > plan.len) {
        plan.read[i] = false;
        std.debug.print("SKIP sweep read {d:.0} MiB: the device refused a {d:.0} MiB allocation\n", .{ mib(size), mib(size) });
    };
    for (write_sizes, 0..) |size, i| if (plan.write[i] and size > plan.len) {
        plan.write[i] = false;
        std.debug.print("SKIP sweep write {d:.0} MiB: the device refused a {d:.0} MiB allocation\n", .{ mib(size), mib(size) });
    };

    std.debug.print("MILESTONE bw-ceiling cases: (1) cold-to-warm ramp: {d} batches x {d} dependent {d:.0} MiB copies; (2) copy sweep: 4/16/64/256/512/1024 MiB x N=0 then 16/16/8/4/2/2; (3) read sweep: 256/512/1024 MiB x N=0 then 4/2/2; (4) write sweep: 256/1024 MiB x N=0 then 4/2; (5) idle re-entry: 3 s host sleep then {d} dependent {d:.0} MiB copies\n", .{ ramp_batches, ramp_dep, mib(big_bytes), ramp_dep, mib(big_bytes) });

    var cmds = try vk.Commands.init(&dev);
    defer cmds.deinit();
    var staging = try vk.Buffer.init(&dev, .staging, staging_bytes);
    defer staging.deinit();
    var red = try vk.Buffer.init(&dev, .device, red_words * 4);
    defer red.deinit();
    var check_src = try vk.Buffer.init(&dev, .device, check_bytes);
    defer check_src.deinit();
    var check_dst = try vk.Buffer.init(&dev, .device, check_bytes);
    defer check_dst.deinit();

    const layout: dispatch.Layout = .{ .bindings = 2, .push_bytes = @sizeOf(Push), .writes = 1 << 1 };
    var copy_pipe = try dispatch.Pipeline.init(&dev, &shaders.stream_copy, layout);
    defer copy_pipe.deinit();
    var read_pipe = try dispatch.Pipeline.init(&dev, &shaders.stream_read, layout);
    defer read_pipe.deinit();
    var write_pipe = try dispatch.Pipeline.init(&dev, &shaders.stream_write, layout);
    defer write_pipe.deinit();
    if (!std.mem.eql(u32, &copy_pipe.local, &.{ 256, 1, 1 })) {
        std.debug.print("FAIL device: stream_u32x4 local size is {any}, expected 256x1x1\n", .{copy_pipe.local});
        return 1;
    }

    const copy_binds = [_]dispatch.Binding{
        .{ .buffer = &src, .len = src.len },
        .{ .buffer = &dst, .len = dst.len },
    };
    const read_binds = [_]dispatch.Binding{
        .{ .buffer = &src, .len = src.len },
        .{ .buffer = &red, .len = red.len },
    };
    const write_binds = [_]dispatch.Binding{
        .{ .buffer = &src, .len = src.len }, // STREAM_WRITE never reads binding 0; the layout still binds it
        .{ .buffer = &dst, .len = dst.len },
    };
    const groups = [3]u32{ groups_x, 1, 1 };
    const copy_pipes = [_]*dispatch.Pipeline{&copy_pipe};
    const read_pipes = [_]*dispatch.Pipeline{&read_pipe};
    const write_pipes = [_]*dispatch.Pipeline{&write_pipe};

    // ---- (0) byte-exact self-check on a 1 MiB buffer, before any timing
    const check_words: u32 = check_bytes / 4;
    const small = try a.alloc(u32, check_words);
    defer a.free(small);
    const want_bytes = try a.alloc(u8, check_bytes);
    defer a.free(want_bytes);
    const got_bytes = try a.alloc(u8, check_bytes);
    defer a.free(got_bytes);
    const red_got = try a.alloc(u32, red_words);
    defer a.free(red_got);
    fillPseudo(small, 0xCE1107B5);
    try vk.orExit(vk.upload(&cmds, check_src, 0, staging, std.mem.sliceAsBytes(small), timeout));

    const check_push = Push{ .count_words = check_words };
    const check_push_bytes = std.mem.asBytes(&check_push)[0..@sizeOf(Push)];
    try vk.orExit(zeroBuf(&cmds, check_dst, timeout));
    try copy_pipe.bind(&copy_binds);
    try vk.orExit(copy_pipe.run(&cmds, check_push_bytes, groups, timeout));
    try vk.orExit(vk.download(&cmds, check_dst, 0, staging, got_bytes, timeout));
    if (!std.mem.eql(u8, got_bytes, std.mem.sliceAsBytes(small))) {
        std.debug.print("FAIL check: STREAM_COPY output differs from its source on the {d} KiB check buffer\n", .{check_bytes >> 10});
        return 1;
    }

    try read_pipe.bind(&read_binds);
    try vk.orExit(read_pipe.run(&cmds, check_push_bytes, groups, timeout));
    try vk.orExit(vk.download(&cmds, red, 0, staging, std.mem.sliceAsBytes(red_got), timeout));
    if (xorWords(red_got) != xorWords(small)) {
        std.debug.print("FAIL check: STREAM_READ xor does not match the host fold on the {d} KiB check buffer\n", .{check_bytes >> 10});
        return 1;
    }

    try write_pipe.bind(&write_binds);
    try vk.orExit(write_pipe.run(&cmds, check_push_bytes, groups, timeout));
    try vk.orExit(vk.download(&cmds, check_dst, 0, staging, got_bytes, timeout));
    {
        const want_words = std.mem.bytesAsSlice(u32, want_bytes[0..check_bytes]);
        for (0..check_words / 4) |i| {
            const p = writePattern(@intCast(i));
            @memcpy(want_words[i * 4 ..][0..4], &p);
        }
    }
    if (!std.mem.eql(u8, got_bytes, want_bytes)) {
        std.debug.print("FAIL check: STREAM_WRITE output differs from the host pattern on the {d} KiB check buffer\n", .{check_bytes >> 10});
        return 1;
    }
    std.debug.print("MILESTONE checks-passed copy/read/write byte-exact on {d} KiB; timing begins\n", .{check_bytes >> 10});

    // ---- (1) cold-to-warm ramp: the first timed GPU work after the checks
    const ramp_bytes = ramp_dep * 2 * big_bytes;
    const ramp_push = Push{ .count_words = @intCast(big_bytes / 4) };
    const ramp_push_bytes = std.mem.asBytes(&ramp_push)[0..@sizeOf(Push)];
    std.debug.print("MILESTONE ramp-start: {d} batches x {d} dependent {d:.0} MiB copies ({d:.2} GiB per batch), cold-to-warm\n", .{ ramp_batches, ramp_dep, mib(big_bytes), mib(ramp_bytes) / 1024.0 });
    var ramp_ns: [ramp_batches]u64 = undefined;
    for (0..ramp_batches) |b| {
        ramp_ns[b] = try sampleBatch(io, &cmds, staging, &copy_pipes, &copy_binds, ramp_push_bytes, groups, ramp_dep, timeout);
        std.debug.print("RAMP {d: >2}: {d:.1} us {d:.2} GB/s\n", .{ b, us(ramp_ns[b]), effectiveGbps(ramp_bytes, ramp_ns[b]) });
    }
    var scratch8: [8]u64 = undefined;
    const warm_ns = statsOf(ramp_ns[ramp_batches - 8 ..], &scratch8).?.p50_ns;
    const cold_gbps = effectiveGbps(ramp_bytes, ramp_ns[0]);
    const warm_gbps = effectiveGbps(ramp_bytes, warm_ns);
    std.debug.print("RAMP summary: first batch {d:.2} GB/s; median of last 8 batches {d:.2} GB/s; warm/cold ratio {d:.2}x\n", .{ cold_gbps, warm_gbps, rampRatio(warm_gbps, cold_gbps) });

    // ---- (2) size sweep, warm
    std.debug.print("MILESTONE sweep copy: sizes 4/16/64/256/512/1024 MiB, N=0 then N=16/16/8/4/2/2 dependent dispatches (2x size moved per dispatch)\n", .{});
    for (copy_sizes, copy_dep, 0..) |size, dep, i| {
        if (!plan.copy[i]) continue;
        std.debug.print("MILESTONE sweep copy {d:.0} MiB N={d}\n", .{ mib(size), dep });
        const push = Push{ .count_words = @intCast(size / 4) };
        try sweepCase(io, &cmds, staging, &copy_pipes, &copy_binds, std.mem.asBytes(&push)[0..@sizeOf(Push)], groups, dep, 2 * size, timeout);
    }
    std.debug.print("MILESTONE sweep read: sizes 256/512/1024 MiB, N=0 then N=4/2/2 dependent dispatches (size moved per dispatch)\n", .{});
    for (read_sizes, read_dep, 0..) |size, dep, i| {
        if (!plan.read[i]) continue;
        std.debug.print("MILESTONE sweep read {d:.0} MiB N={d}\n", .{ mib(size), dep });
        const push = Push{ .count_words = @intCast(size / 4) };
        try sweepCase(io, &cmds, staging, &read_pipes, &read_binds, std.mem.asBytes(&push)[0..@sizeOf(Push)], groups, dep, size, timeout);
    }
    std.debug.print("MILESTONE sweep write: sizes 256/1024 MiB, N=0 then N=4/2 dependent dispatches (size moved per dispatch)\n", .{});
    for (write_sizes, write_dep, 0..) |size, dep, i| {
        if (!plan.write[i]) continue;
        std.debug.print("MILESTONE sweep write {d:.0} MiB N={d}\n", .{ mib(size), dep });
        const push = Push{ .count_words = @intCast(size / 4) };
        try sweepCase(io, &cmds, staging, &write_pipes, &write_binds, std.mem.asBytes(&push)[0..@sizeOf(Push)], groups, dep, size, timeout);
    }

    // ---- (3) idle re-entry: does the memory clock fall back?
    std.debug.print("MILESTONE idle-re-entry: {d} s host sleep, then one batch of {d} dependent {d:.0} MiB copies\n", .{ idle_sleep_ms / 1000, ramp_dep, mib(big_bytes) });
    std.Io.sleep(io, .fromMilliseconds(idle_sleep_ms), .awake) catch {};
    const idle_ns = try sampleBatch(io, &cmds, staging, &copy_pipes, &copy_binds, ramp_push_bytes, groups, ramp_dep, timeout);
    std.debug.print("IDLE after {d} s: {d:.1} us {d:.2} GB/s ({d} dependent {d:.0} MiB copies per batch)\n", .{ idle_sleep_ms / 1000, us(idle_ns), effectiveGbps(ramp_bytes, idle_ns), ramp_dep, mib(big_bytes) });

    std.debug.print("RESULT bw-ceiling on \"{s}\": ramp=1 sweep-copy={d}/{d} sweep-read={d}/{d} sweep-write={d}/{d} idle=1 ok\n", .{
        name, enabledCases(&plan, .copy), copy_sizes.len, enabledCases(&plan, .read), read_sizes.len, enabledCases(&plan, .write), write_sizes.len,
    });
    return 0;
}

fn enabledCases(plan: *const SweepPlan, comptime which: enum { copy, read, write }) usize {
    var n: usize = 0;
    for (@field(plan, @tagName(which))) |on| n += @intFromBool(on);
    return n;
}

fn zeroBuf(cmds: *vk.Commands, buf: vk.Buffer, timeout: u64) !void {
    try vk.orExit(vk.zero(cmds, buf, 0, buf.len, timeout));
}

const testing = std.testing;

test "parseArgs takes the two flags, bounds the timeout and refuses everything else" {
    const ok = parseArgs(&.{ "--vendor", "0x106b", "--timeout-ms", "90000" }).?;
    try testing.expectEqual(@as(?u32, 0x106b), ok.vendor);
    try testing.expectEqual(@as(u64, 90000), ok.timeout_ms);
    const plain = parseArgs(&.{}).?;
    try testing.expect(plain.vendor == null);
    try testing.expectEqual(@as(u64, 10_000), plain.timeout_ms);
    try testing.expectEqual(@as(u32, 0x10005), parseArgs(&.{ "--vendor", "65541" }).?.vendor.?);
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
    try testing.expectEqual(@as(u64, 5), st.p50_ns);
    try testing.expectEqual(@as(u64, 9), st.p95_ns);
    try testing.expectEqual(@as(u64, 9), st.max_ns);
    try testing.expect(statsOf(&.{}, &scratch) == null);
    try testing.expect(statsOf(&samples, scratch[0..4]) == null);
}

test "slopes subtract the floor and effectiveGbps is bytes per nanosecond" {
    try testing.expectEqual(@as(f64, 2.0), try perDispatchNs(1000, 0, 500));
    try testing.expectError(error.ZeroN, perDispatchNs(2000, 1000, 0));
    try testing.expectEqual(@as(f64, 8.0), effectiveGbps(8, 1));
    // the ramp's arithmetic: 2 GiB moved in 68 ms is 31.5816 GB/s
    try testing.expectApproxEqAbs(@as(f64, 31.5816), effectiveGbps(2 << 30, 68_000_000), 0.001);
}

test "rampRatio is warm over cold" {
    try testing.expectEqual(136.3 / 32.7, rampRatio(136.3, 32.7));
    try testing.expectEqual(@as(f64, 1.0), rampRatio(89.0, 89.0));
    try testing.expect(rampRatio(50.0, 100.0) < 1.0); // a warm-up slower than cold reports honestly
}

test "writePattern mirrors the shader's integer formula" {
    try testing.expectEqual([4]u32{ 0, 0xFFFF_FFFF, 0xA5A5_A5A5, 3 }, writePattern(0));
    try testing.expectEqual([4]u32{ 5, ~@as(u32, 5), 5 ^ 0xA5A5_A5A5, 5 * 7 + 3 }, writePattern(5));
    for (0..64) |i| {
        const p = writePattern(@intCast(i));
        try testing.expectEqual(@as(u32, @intCast(i)), p[0]);
        try testing.expectEqual(@as(u32, @as(u32, @intCast(i)) *% 7 +% 3), p[3]);
    }
}

test "fillPseudo is deterministic and seed-sensitive and xorWords folds" {
    var x: [16]u32 = undefined;
    var y: [16]u32 = undefined;
    fillPseudo(&x, 7);
    fillPseudo(&y, 7);
    try testing.expectEqualSlices(u32, &x, &y);
    fillPseudo(&y, 8);
    try testing.expect(!std.mem.eql(u32, &x, &y));
    try testing.expectEqual(@as(u32, 0xa ^ 0xb ^ 0xc), xorWords(&.{ 0xa, 0xb, 0xc }));
    try testing.expectEqual(@as(u32, 0), xorWords(&.{}));
}

test "planSweep enables sizes within the binding limits and sizes the allocation" {
    const full = planSweep(1 << 30, 4 << 40);
    try testing.expectEqual(@as(u64, 1 << 30), full.len);
    for (full.copy) |on| try testing.expect(on);
    for (full.read) |on| try testing.expect(on);
    for (full.write) |on| try testing.expect(on);
    const mid = planSweep(300 << 20, 4 << 40);
    try testing.expectEqual(@as(u64, 300 << 20), mid.len);
    try testing.expect(mid.copy[3] and mid.read[0] and mid.write[0]); // 256 MiB fits
    try testing.expect(!mid.copy[4] and !mid.read[1] and !mid.write[1]); // 512 MiB and 1 GiB do not
    const tiny = planSweep(64 << 20, 64 << 20);
    try testing.expect(tiny.len < big_bytes); // the caller must refuse: the ramp cannot run
}
