//! tf-contract-penalties: penalize.comp through gates 1-4 with four planted controls, plus the fp32-denormal row gate.
const std = @import("std");
const vk = @import("vulkan");
const contract = @import("contract");
const g = contract.gates;
const fx = @import("contract_penalties.zig");
const shaders = @import("penalties_contract_shaders");

const usage = "usage: tf-contract-penalties [--vendor ID] [--timeout-ms MS] | --help\n" ++
    "  --help          show this text and exit without opening a loader or device\n" ++
    "  --vendor ID     refuse a device whose Vulkan vendorID differs (decimal or 0x hex; no default)\n" ++
    "  --timeout-ms MS bound on every fence wait, 1 to 600000 (default 10000); a timeout ends the process\n";

/// Milliseconds on the awake clock since main started, printed on every milestone.
const Elapsed = struct {
    io: std.Io,
    start: std.Io.Timestamp,

    fn ms(e: Elapsed) i64 {
        return e.start.durationTo(std.Io.Clock.awake.now(e.io)).toMilliseconds();
    }
};

const seed: u64 = 0x70e5;

pub fn main(init: std.process.Init) !u8 {
    const clock: Elapsed = .{ .io = init.io, .start = std.Io.Clock.awake.now(init.io) };
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    for (args[1..]) |arg| if (std.mem.eql(u8, arg, "--help")) {
        std.debug.print(usage, .{});
        return 0;
    };
    var vendor: ?u32 = null;
    var timeout_ms: u64 = 10_000;
    var i: usize = 1;
    while (i < args.len) : (i += 2) {
        const value = if (i + 1 < args.len) args[i + 1] else "";
        if (std.mem.eql(u8, args[i], "--vendor")) {
            vendor = std.fmt.parseInt(u32, value, 0) catch null;
            if (vendor == null) break;
        } else if (std.mem.eql(u8, args[i], "--timeout-ms")) {
            timeout_ms = std.fmt.parseInt(u64, value, 10) catch 0;
            if (timeout_ms == 0 or timeout_ms > 600_000) break;
        } else break;
    } else return if (probe(init.gpa, clock, vendor, timeout_ms * std.time.ns_per_ms)) 0 else 1;
    std.debug.print(usage, .{});
    return 2;
}

fn probe(a: std.mem.Allocator, clock: Elapsed, vendor: ?u32, timeout: u64) bool {
    return body(a, clock, vendor, timeout) catch |e| {
        std.debug.print("FAIL contract-penalties case=\"device\": {s}\n", .{@errorName(e)});
        std.debug.print("RESULT contract-penalties elapsed_ms={d}: passed=0 known=0 total=6\n", .{clock.ms()});
        return false;
    };
}

fn body(a: std.mem.Allocator, clock: Elapsed, vendor: ?u32, timeout: u64) !bool {
    var rt = try vk.Runtime.open();
    defer rt.close();
    var dev = try vk.Device.open(&rt, .{ .features = .{ .storage16 = true, .int16 = true } });
    defer dev.close();
    const pr = dev.properties;
    const name = std.mem.sliceTo(&pr.deviceName, 0);
    const types = [_][]const u8{ "other", "integrated", "discrete", "virtual", "cpu" };
    const device_kind = if (pr.deviceType < types.len) types[pr.deviceType] else "unknown";
    const v = pr.apiVersion;
    const preserve32 = dev.v12.shaderDenormPreserveFloat32 != 0;
    const flush32 = dev.v12.shaderDenormFlushToZeroFloat32 != 0;
    std.debug.print("MILESTONE contract-penalties device-found elapsed_ms={d} name=\"{s}\" vendor=0x{x:0>4} device=0x{x:0>4} type={s} api={d}.{d}.{d} " ++
        "storage16={} int16={} preserve32={} flush32={} fence-timeout={d}ms\n", .{ clock.ms(), name, pr.vendorID, pr.deviceID, device_kind, v >> 22 & 0x7f, v >> 12 & 0x3ff, v & 0xfff, dev.features.storage16, dev.features.int16, preserve32, flush32, timeout / std.time.ns_per_ms });
    var refused = false;
    if (pr.deviceType == vk.c.VK_PHYSICAL_DEVICE_TYPE_CPU) {
        std.debug.print("FAIL contract-penalties device: a CPU (software) Vulkan device is never a GPU result\n", .{});
        refused = true;
    }
    if (vendor) |want| if (pr.vendorID != want) {
        std.debug.print("FAIL contract-penalties device: vendor 0x{x:0>4} is not the requested 0x{x:0>4}\n", .{ pr.vendorID, want });
        refused = true;
    };
    var t: g.Tally = .{};
    var passed: usize = 0;
    var known: usize = 0;
    var sub_verdict: fx.Verdict = .fail;
    if (!refused) {
        var cmds = try vk.Commands.init(&dev);
        defer cmds.deinit();
        // the largest contract shell (248320-word row plus guards) is under half a megabyte, so it rides staging
        var staging = try vk.Buffer.init(&dev, .staging, 1 << 20);
        defer staging.deinit();
        var real = fx.Penalize.init(a, &dev, &cmds, staging, timeout, &shaders.penalize, .kernel, .double);
        defer real.deinit();
        var ctl_double = fx.Penalize.init(a, &dev, &cmds, staging, timeout, &shaders.ctl_double, .control, .double);
        defer ctl_double.deinit();
        var ctl_bf16 = fx.Penalize.init(a, &dev, &cmds, staging, timeout, &shaders.ctl_bf16_sub, .control, .bf16_sub);
        defer ctl_bf16.deinit();
        var ctl_scribble = fx.Penalize.init(a, &dev, &cmds, staging, timeout, &shaders.ctl_scribble, .control, .scribble);
        defer ctl_scribble.deinit();
        var ctl_tail = fx.Penalize.init(a, &dev, &cmds, staging, timeout, &shaders.ctl_tail, .control, .tail);
        defer ctl_tail.deinit();
        var kernels = [_]g.Kernel{
            real.kernel(),
            ctl_double.kernel(),
            ctl_bf16.kernel(),
            ctl_scribble.kernel(),
            ctl_tail.kernel(),
        };
        // a raised gate prints that gate's FAIL and the run continues to the RESULT line
        g.runAll(a, &kernels, seed, &t) catch |e|
            std.debug.print("FAIL contract-penalties case=\"gates\": {s}\n", .{@errorName(e)});
        // the denormal row: its own gate, KNOWN only under its stated condition
        var sub = fx.Penalize.init(a, &dev, &cmds, staging, timeout, &shaders.penalize, .subnormal, .double);
        defer sub.deinit();
        sub_verdict = subnormal(&sub, preserve32, seed) catch |e| blk: {
            std.debug.print("FAIL subnormal case=\"denormal-cancel\": {s}\n", .{@errorName(e)});
            break :blk .fail;
        };
    }
    const acct = fx.accounting(t, sub_verdict, refused);
    passed = acct.passed;
    known = acct.known;
    const total = acct.total;
    std.debug.print("SUMMARY contract-penalties on \"{s}\": {d} kernel, {d} pass gates 1-4, {d} fail; " ++
        "{d} controls: {d} red as planted, {d} missed; denormal row: {t}\n", .{ name, t.kernels, t.passed, t.failed, t.controls, t.caught, t.missed, sub_verdict });
    std.debug.print("RESULT contract-penalties elapsed_ms={d} on \"{s}\": passed={d} known={d} total={d} preserve32={} flush32={}\n", .{ clock.ms(), name, passed, known, total, preserve32, flush32 });
    return acct.ok;
}

/// A raised error is that case's FAIL, never the run's death; the verdict is the caller's to count.
fn subnormal(f: *fx.Penalize, preserve32: bool, gate_seed: u64) !fx.Verdict {
    if (!try f.prepare(.{ .rows = 1, .n = fx.vocab, .k = 0 }, .random, gate_seed)) return error.Refused;
    return f.subnormalGate(preserve32);
}
