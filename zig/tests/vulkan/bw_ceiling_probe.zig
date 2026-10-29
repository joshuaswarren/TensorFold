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
//!   2. SIZE SWEEP, warm: COPY/READ/WRITE at 64/256/512/1024 MiB, each at
//!      dependent depth N=0 floor plus N=1, 4 and 16 in one batched command
//!      buffer (1 GiB runs N=1, 4 and 8 instead of 16 — Main's sanctioned
//!      fallback, because 16 x 1 GiB batches can bust the two-minute cap at
//!      slow clocks; the header says so when it runs). Per-dispatch p50 for
//!      N=4 and N=16/8 comes from (p50(N)-p50(0))/N; the N=1 batch time is
//!      printed as is, submit latency included. COPY also keeps its small
//!      4/16 MiB rows at N=16. A size the device cannot bind or allocate is
//!      skipped with a clear message and the run continues. A fourth sweep
//!      runs the driver's own vkCmdCopyBuffer (device-local to device-local)
//!      over the same sizes and depths, a transfer barrier between copies.
//!      Timed samples shrink with batch cost (15..63) so the whole run holds
//!      near two minutes.
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
const copy_small_sizes = [_]u64{ 4 << 20, 16 << 20 }; // COPY keeps its small rows
const copy_small_deps = [_]usize{16};
const stream_sizes = [_]u64{ 64 << 20, 256 << 20, 512 << 20, 1 << 30 };
const stream_deps = [_]usize{ 1, 4, 16 };
const stream_1gib_deps = [_]usize{ 1, 4, 8 }; // 16 x 1 GiB batches can bust the two-minute cap; the header says so

comptime {
    for (stream_sizes[1..], 0..) |s, i| if (s <= stream_sizes[i]) @compileError("stream sizes must ascend");
    for (copy_small_deps ++ stream_deps ++ stream_1gib_deps) |n| if (n == 0 or n > dispatch.batch_cap) @compileError("sweep dispatch counts must be nonzero and fit one pipeline's descriptor bank");
    if (stream_1gib_deps.len != stream_deps.len) @compileError("the 1 GiB fallback must keep three depths");
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
    /// The 4/16 MiB small COPY rows; on whenever the ramp's allocation fits.
    small: bool,
    /// The 64/256/512/1024 MiB stream rows, all four kernels.
    stream: [stream_sizes.len]bool,
};

/// Sweep sizes the device's storage-buffer limits allow; `len` never exceeds
/// 1 GiB (the full sweep's allocation). Pure and host-tested.
pub fn planSweep(max_range: u64, max_buffer: u64) SweepPlan {
    const limit = @min(max_range, max_buffer);
    var p: SweepPlan = .{ .len = @min(@as(u64, 1 << 30), limit), .small = 16 << 20 <= limit, .stream = undefined };
    for (stream_sizes, 0..) |s, i| p.stream[i] = s <= limit;
    return p;
}

/// Timed samples for one sweep case: 63 when the batch is cheap, shrinking
/// with batch cost (a 3 s-per-case budget at 40 GB/s) so the whole run holds
/// near two minutes; never fewer than 15, always odd for a clean nearest-rank
/// p50. Pure and host-tested.
pub fn sampleCount(batch_bytes: u64) usize {
    if (batch_bytes == 0) return sample_count;
    const budget: u64 = 120_000_000_000 / batch_bytes; // 3 s of that batch at 40 GB/s
    return @intCast(@min(63, @max(15, budget | 1)));
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

/// One recorded sweep unit: a shader dispatch's bind+record, or a driver
/// vkCmdCopyBuffer of `len` bytes.
const Unit = union(enum) {
    shader: struct {
        pipes: []const *dispatch.Pipeline,
        binds: []const dispatch.Binding,
        push: []const u8,
        groups: [3]u32,
    },
    driver: struct { src: vk.Buffer, dst: vk.Buffer },
};

/// One sample: the whole batch cycle for N dependent units, timed from just
/// before the first record to just after the fence returns. Every unit
/// records into the open batch and carries the trailing barrier (dispatch
/// end barrier, or the copy's transfer barrier), so the N units are
/// dependent; one pipeline's descriptor bank holds batch_cap snapshots and
/// every sweep depth is comptime-bounded below it.
fn sampleBatch(
    io: std.Io,
    cmds: *vk.Commands,
    staging: vk.Buffer,
    unit: Unit,
    len: u64,
    n: usize,
    timeout: u64,
) !u64 {
    try vk.orExit(cmds.beginBatch(staging));
    errdefer cmds.abortBatch();
    const t0 = std.Io.Clock.awake.now(io);
    for (0..n) |_| {
        switch (unit) {
            .shader => |s| {
                const pipe = s.pipes[0];
                try vk.orExit(pipe.bind(s.binds)); // staged while the batch is open, snapshotted at record
                try vk.orExit(pipe.record(try cmds.begin(), s.push, s.groups)); // dispatch + the end barrier
            },
            .driver => |d| try vk.orExit(vk.batchCopyBuffer(cmds, d.src, d.dst, len)),
        }
    }
    try vk.orExit(cmds.endBatch(timeout)); // flush, one submit, one fence wait, invalidate
    const t1 = std.Io.Clock.awake.now(io);
    return @intCast(t1.nanoseconds - t0.nanoseconds);
}

/// 3 warmups, then `samples` timed batches of one case; the summary lands in
/// `out`.
fn runCase(
    io: std.Io,
    cmds: *vk.Commands,
    staging: vk.Buffer,
    unit: Unit,
    len: u64,
    n: usize,
    samples: usize,
    timeout: u64,
    out: *Stats,
) !void {
    var arr: [sample_count]u64 = undefined;
    var scratch: [sample_count]u64 = undefined;
    for (0..warmups) |_| _ = try sampleBatch(io, cmds, staging, unit, len, n, timeout);
    for (0..samples) |i| arr[i] = try sampleBatch(io, cmds, staging, unit, len, n, timeout);
    out.* = statsOf(arr[0..samples], &scratch).?; // samples <= sample_count: cannot fail
}

/// One kernel's sweep: for each enabled size, the N=0 empty-batch floor, then
/// each dependent depth. N>=4 gets per-dispatch p50/min from floor
/// subtraction and effective GB/s from `bytes_moved_of * size` per dispatch;
/// the N=1 batch prints as recorded, submit latency included.
fn runSweep(
    io: std.Io,
    cmds: *vk.Commands,
    staging: vk.Buffer,
    label: []const u8,
    unit: Unit,
    sizes: []const u64,
    enabled: []const bool,
    deps: []const usize,
    big_dep: ?[]const usize,
    bytes_moved_of: u64, // 2 for a copy (read + write), 1 for read- or write-only
    timeout: u64,
) !void {
    for (sizes, enabled) |size, on| {
        if (!on) continue;
        const chosen: []const usize = if (big_dep) |bd| (if (size >= (1 << 30)) bd else deps) else deps;
        const bytes_moved = bytes_moved_of * size;
        std.debug.print("MILESTONE sweep {s} {d:.0} MiB: N=0 floor then dependent N=1/4/{d}, {d:.0} MiB moved per dispatch\n", .{ label, mib(size), chosen[chosen.len - 1], mib(bytes_moved) });
        var u = unit;
        if (u == .shader) u.shader.push = std.mem.asBytes(&Push{ .count_words = @intCast(size / 4) })[0..@sizeOf(Push)];
        var s0: Stats = undefined;
        try runCase(io, cmds, staging, u, size, 0, sample_count, timeout, &s0);
        printStats(0, s0);
        for (chosen) |dep| {
            var s1: Stats = undefined;
            try runCase(io, cmds, staging, u, size, dep, sampleCount(dep * bytes_moved), timeout, &s1);
            printStats(dep, s1);
            if (dep == 1) {
                std.debug.print("    N=  1 batch p50={d:.3} ms as recorded (includes submit latency) -> {d:.2} GB/s\n", .{
                    us(s1.p50_ns) / 1000.0, effectiveGbps(bytes_moved, s1.p50_ns),
                });
            } else {
                const per50 = try perDispatchNs(s1.p50_ns, s0.p50_ns, @intCast(dep));
                const perMin = try perDispatchNs(s1.min_ns, s0.min_ns, @intCast(dep));
                std.debug.print("    per-dispatch p50={d:.3} ms min={d:.3} ms -> {d:.2} GB/s (p50), {d:.2} GB/s (min)\n", .{
                    per50 / 1e6, perMin / 1e6,
                    effectiveGbps(bytes_moved, @intFromFloat(@max(per50, 1.0))),
                    effectiveGbps(bytes_moved, @intFromFloat(@max(perMin, 1.0))),
                });
            }
        }
    }
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

    // PREFLIGHT ONLY, NOT A RESULT: this throwaway build admits a CPU
    // (lavapipe) device under BW_PREFLIGHT_CPU and divides the sweep sizes by
    // BW_PREFLIGHT_SCALE so every case line executes without minutes of CPU
    // memcpy. Neither knob exists in the real branch.
    var scale: u64 = 1;
    if (std.c.getenv("BW_PREFLIGHT_SCALE")) |v| scale = std.fmt.parseInt(u64, std.mem.span(v), 10) catch 1;
    if (scale == 0) scale = 1;
    const preflight_cpu = std.c.getenv("BW_PREFLIGHT_CPU") != null;
    std.debug.print("PREFLIGHT ONLY, NOT A RESULT: cpu-allowed={} scale={d}\n", .{ preflight_cpu, scale });
    var stream_scaled: [stream_sizes.len]u64 = undefined;
    for (stream_sizes, 0..) |s, i| stream_scaled[i] = s / scale;
    var small_scaled: [copy_small_sizes.len]u64 = undefined;
    for (copy_small_sizes, 0..) |s, i| small_scaled[i] = s / scale;
    const big = big_bytes / scale;

    var rt = try vk.Runtime.open();
    defer rt.close();
    var dev = try vk.Device.open(&rt, .{}); // refuses CPU (software) devices and Vulkan below 1.3
    defer dev.close();
    const pr = dev.properties;
    const name = std.mem.sliceTo(&pr.deviceName, 0);
    const types = [_][]const u8{ "other", "integrated", "discrete", "virtual", "cpu" };
    const kind = if (pr.deviceType < types.len) types[pr.deviceType] else "unknown";
    std.debug.print("MILESTONE device-found name=\"{s}\" vendor=0x{x:0>4} device=0x{x:0>4} type={s} timeout={d}ms\n", .{ name, pr.vendorID, pr.deviceID, kind, opts.timeout_ms });
    if (pr.deviceType == c.VK_PHYSICAL_DEVICE_TYPE_CPU and !preflight_cpu) {
        std.debug.print("FAIL device: a CPU (software) Vulkan device is never a GPU result\n", .{});
        return 1;
    }
    if (opts.vendor) |want| if (pr.vendorID != want) {
        std.debug.print("FAIL device: vendor 0x{x:0>4} is not the requested 0x{x:0>4}\n", .{ pr.vendorID, want });
        return 1;
    };

    // Sweep plan against the device's real binding limits; sizes above them
    // are skipped with a message and the rest of the run continues.
    var plan = planSweep(pr.limits.maxStorageBufferRange * scale, dev.v13.maxBufferSize * scale);
    for (stream_sizes, plan.stream) |size, on| if (!on) {
        std.debug.print("SKIP sweep {d:.0} MiB: exceeds {s}={d}\n", .{ mib(size), if (size > pr.limits.maxStorageBufferRange) "maxStorageBufferRange" else "maxBufferSize", @min(pr.limits.maxStorageBufferRange, dev.v13.maxBufferSize) });
    };
    if (plan.len / scale < big) {
        std.debug.print("FAIL device: the largest allocatable sweep buffer is {d:.0} MiB; the ramp and idle cases need {d:.0} MiB\n", .{ mib(plan.len / scale), mib(big) });
        return 1;
    }

    // One src and one dst pair sized for the largest enabled case; an
    // allocation failure steps down and re-filters instead of ending the run.
    var src: vk.Buffer = undefined;
    var dst: vk.Buffer = undefined;
    while (true) {
        src = vk.Buffer.init(&dev, .device, plan.len / scale) catch {
            plan.len /= 2;
            if (plan.len < big_bytes) break;
            continue;
        };
        dst = vk.Buffer.init(&dev, .device, plan.len / scale) catch {
            src.deinit();
            plan.len /= 2;
            if (plan.len < big_bytes) break;
            continue;
        };
        break;
    }
    if (plan.len / scale < big) {
        std.debug.print("FAIL device: could not allocate a {d:.0} MiB src/dst pair; the ramp and idle cases need it\n", .{mib(big)});
        return 1;
    }
    for (stream_sizes, 0..) |size, i| if (plan.stream[i] and size > plan.len) {
        plan.stream[i] = false;
        std.debug.print("SKIP sweep {d:.0} MiB: the device refused a {d:.0} MiB allocation\n", .{ mib(size), mib(size * 2) });
    };

    std.debug.print("MILESTONE bw-ceiling cases: (1) cold-to-warm ramp: {d} batches x {d} dependent {d:.0} MiB copies; (2) shader copy (uvec4) sweep: 4/16 MiB x N=16 then 64/256/512/1024 MiB x N=1/4/16 (1 GiB at N=1/4/8, since 16 x 1 GiB batches can bust the two-minute cap); (3) read sweep and (4) write sweep: 64/256/512/1024 MiB x N=1/4/16 (1 GiB at N=1/4/8); (5) driver copy (vkCmdCopyBuffer) sweep: 64/256/512/1024 MiB x N=1/4/16 (1 GiB at N=1/4/8), a transfer barrier between copies; (6) idle re-entry: 3 s host sleep then {d} dependent {d:.0} MiB copies; timed samples shrink with batch cost (15..63) to hold the two-minute cap\n", .{ ramp_batches, ramp_dep, mib(big_bytes), ramp_dep, mib(big_bytes) });

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
    const copy_unit: Unit = .{ .shader = .{ .pipes = &copy_pipes, .binds = &copy_binds, .push = &[_]u8{}, .groups = groups } };
    const read_unit: Unit = .{ .shader = .{ .pipes = &read_pipes, .binds = &read_binds, .push = &[_]u8{}, .groups = groups } };
    const write_unit: Unit = .{ .shader = .{ .pipes = &write_pipes, .binds = &write_binds, .push = &[_]u8{}, .groups = groups } };
    const driver_unit: Unit = .{ .driver = .{ .src = src, .dst = dst } };

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
    const ramp_bytes = ramp_dep * 2 * big;
    var ramp_unit = copy_unit;
    ramp_unit.shader.push = std.mem.asBytes(&Push{ .count_words = @intCast(big / 4) })[0..@sizeOf(Push)];
    std.debug.print("MILESTONE ramp-start: {d} batches x {d} dependent {d:.0} MiB copies ({d:.2} GiB per batch), cold-to-warm\n", .{ ramp_batches, ramp_dep, mib(big), mib(ramp_bytes) / 1024.0 });
    var ramp_ns: [ramp_batches]u64 = undefined;
    for (0..ramp_batches) |b| {
        ramp_ns[b] = try sampleBatch(io, &cmds, staging, ramp_unit, big, ramp_dep, timeout);
        std.debug.print("RAMP {d: >2}: {d:.1} us {d:.2} GB/s\n", .{ b, us(ramp_ns[b]), effectiveGbps(ramp_bytes, ramp_ns[b]) });
    }
    var scratch8: [8]u64 = undefined;
    const warm_ns = statsOf(ramp_ns[ramp_batches - 8 ..], &scratch8).?.p50_ns;
    const cold_gbps = effectiveGbps(ramp_bytes, ramp_ns[0]);
    const warm_gbps = effectiveGbps(ramp_bytes, warm_ns);
    std.debug.print("RAMP summary: first batch {d:.2} GB/s; median of last 8 batches {d:.2} GB/s; warm/cold ratio {d:.2}x\n", .{ cold_gbps, warm_gbps, rampRatio(warm_gbps, cold_gbps) });

    // ---- (2) size sweep, warm: shader kernels, then the driver copy
    const small_on = [_]bool{ plan.small, plan.small };
    std.debug.print("MILESTONE sweep shader copy (uvec4): small sizes then 64/256/512/1024 MiB at N=1/4/16 (1 GiB at N=1/4/8); 2x size moved per dispatch\n", .{});
    try runSweep(io, &cmds, staging, "shader copy (uvec4)", copy_unit, &small_scaled, &small_on, &copy_small_deps, null, 2, timeout);
    try runSweep(io, &cmds, staging, "shader copy (uvec4)", copy_unit, &stream_scaled, &plan.stream, &stream_deps, &stream_1gib_deps, 2, timeout);
    std.debug.print("MILESTONE sweep read (xor-reduce): 64/256/512/1024 MiB at N=1/4/16 (1 GiB at N=1/4/8); size moved per dispatch\n", .{});
    try runSweep(io, &cmds, staging, "read (xor-reduce)", read_unit, &stream_scaled, &plan.stream, &stream_deps, &stream_1gib_deps, 1, timeout);
    std.debug.print("MILESTONE sweep write (pattern): 64/256/512/1024 MiB at N=1/4/16 (1 GiB at N=1/4/8); size moved per dispatch\n", .{});
    try runSweep(io, &cmds, staging, "write (pattern)", write_unit, &stream_scaled, &plan.stream, &stream_deps, &stream_1gib_deps, 1, timeout);
    std.debug.print("MILESTONE sweep driver copy (vkCmdCopyBuffer): 64/256/512/1024 MiB at N=1/4/16 (1 GiB at N=1/4/8); a transfer barrier between copies; 2x size moved per dispatch\n", .{});
    try runSweep(io, &cmds, staging, "driver copy (vkCmdCopyBuffer)", driver_unit, &stream_scaled, &plan.stream, &stream_deps, &stream_1gib_deps, 2, timeout);

    // ---- (3) idle re-entry: does the memory clock fall back?
    std.debug.print("MILESTONE idle-re-entry: {d} s host sleep, then one batch of {d} dependent {d:.0} MiB copies\n", .{ idle_sleep_ms / 1000, ramp_dep, mib(big) });
    std.Io.sleep(io, .fromMilliseconds(idle_sleep_ms), .awake) catch {};
    const idle_ns = try sampleBatch(io, &cmds, staging, ramp_unit, big, ramp_dep, timeout);
    std.debug.print("IDLE after {d} s: {d:.1} us {d:.2} GB/s ({d} dependent {d:.0} MiB copies per batch)\n", .{ idle_sleep_ms / 1000, us(idle_ns), effectiveGbps(ramp_bytes, idle_ns), ramp_dep, mib(big) });

    var enabled: usize = 0;
    for (plan.stream) |on| enabled += @intFromBool(on);
    std.debug.print("RESULT bw-ceiling on \"{s}\": ramp=1 shader-copy-small={s} stream sizes {d}/{d} for shader-copy, read, write and driver-copy; idle=1 ok\n", .{
        name, if (plan.small) "on" else "off", enabled, stream_sizes.len,
    });
    return 0;
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
    try testing.expect(full.small);
    for (full.stream) |on| try testing.expect(on);
    const mid = planSweep(300 << 20, 4 << 40);
    try testing.expectEqual(@as(u64, 300 << 20), mid.len);
    try testing.expect(mid.small);
    try testing.expect(mid.stream[0] and mid.stream[1]); // 64 and 256 MiB fit
    try testing.expect(!mid.stream[2] and !mid.stream[3]); // 512 MiB and 1 GiB do not
    const tiny = planSweep(64 << 20, 64 << 20);
    try testing.expect(tiny.len < big_bytes); // the caller must refuse: the ramp cannot run
    try testing.expect(!tiny.stream[2] and !tiny.stream[3]);
}

test "sampleCount shrinks with batch cost inside [15, 63], always odd" {
    try testing.expectEqual(@as(usize, 63), sampleCount(0));
    try testing.expectEqual(@as(usize, 63), sampleCount(64 << 20));
    try testing.expectEqual(@as(usize, 63), sampleCount(1 << 30));
    try testing.expectEqual(@as(usize, 55), sampleCount(2 << 30)); // 55.87 floor, odd
    try testing.expectEqual(@as(usize, 27), sampleCount(4 << 30)); // 27.9 floor, odd
    try testing.expectEqual(@as(usize, 15), sampleCount(8 << 30)); // 13.9 floors to the 15 minimum
    try testing.expectEqual(@as(usize, 15), sampleCount(32 << 30));
}
