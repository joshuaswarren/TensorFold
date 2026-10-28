//! Vulkan penalties bring-up: presence/frequency penalization over one BF16
//! logits row at the fixture's fixed CUDA-formula words, through the same
//! dispatch the engine records, on a real GPU. Exit 0 pass, 1 fail, 2
//! usage, 124 fence timeout. Bring-up only: no model, no performance claim.
//! Every expected word is read back from the GPU and compared against the
//! fixture's independent CUDA-arithmetic literals (exact bf16 widening, one
//! fp32 subtract, one bf16 RNE) — never recomputed on the host, so a device
//! that diverges (the fp32-subnormal flush on some drivers, an RNE drift)
//! fails its gate honestly. The greedy gates run the production Greedy
//! after the penalize dispatch: the winner-change row must select the
//! untouched token, and the finite-guard row must refuse NonFinite. Every
//! GPU call after submit goes through vk.orExit; the expected NonFinite is
//! caught explicitly and never reported as a GPU failure.
const std = @import("std");
const vk = @import("vulkan");
const shaders = @import("penalties_shaders");
const fx = @import("penalties_fixture.zig");

const usage = "usage: tf-penalties-gpu [--vendor ID] [--timeout-ms MS] | --help\n" ++
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
    } else return if (try probe(init.gpa, clock, vendor, timeout_ms * std.time.ns_per_ms)) 0 else 1;
    std.debug.print(usage, .{});
    return 2;
}

fn probe(a: std.mem.Allocator, clock: Elapsed, vendor: ?u32, timeout: u64) !bool {
    var rt = try vk.Runtime.open();
    defer rt.close();
    var dev = try vk.Device.open(&rt, .{ .features = .{ .storage16 = true, .int16 = true } });
    defer dev.close();
    const pr = dev.properties;
    const name = std.mem.sliceTo(&pr.deviceName, 0);
    std.debug.print("MILESTONE device-found elapsed_ms={d} name=\"{s}\" vendor=0x{x:0>4} fence-timeout={d}ms\n", .{ clock.ms(), name, pr.vendorID, timeout / std.time.ns_per_ms });
    if (pr.deviceType == vk.c.VK_PHYSICAL_DEVICE_TYPE_CPU) {
        std.debug.print("FAIL device: a CPU (software) Vulkan device is never a GPU result\n", .{});
        return false;
    }
    if (vendor) |want| if (pr.vendorID != want) {
        std.debug.print("FAIL device: vendor 0x{x:0>4} is not the requested 0x{x:0>4}\n", .{ pr.vendorID, want });
        return false;
    };

    var cmds = try vk.Commands.init(&dev);
    defer cmds.deinit();
    var staging = try vk.Buffer.init(&dev, .staging, 1 << 20);
    defer staging.deinit();

    var pipe = try vk.dispatch.Pipeline.init(&dev, &shaders.penalize, .{ .bindings = 3, .push_bytes = @sizeOf(u32), .writes = 1 });
    defer pipe.deinit();

    var passed: usize = 0;
    var total: usize = 0;
    for (&fx.rows) |row| {
        total += 1;
        passed += @intFromBool(try rowGate(a, &dev, &cmds, staging, &pipe, row, timeout));
    }
    std.debug.print("RESULT penalties elapsed_ms={d} on \"{s}\": passed={d} total={d}\n", .{ clock.ms(), name, passed, total });
    return passed == total;
}

fn rowGate(a: std.mem.Allocator, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, pipe: *vk.dispatch.Pipeline, row: fx.Row, timeout: u64) !bool {
    const bits = try row.shell(a);
    defer a.free(bits);
    var buf = vk.Buffer.init(dev, .device, std.mem.sliceAsBytes(bits).len) catch |e| return gpuFail(row.name, e);
    defer buf.deinit();
    vk.orExit(vk.upload(cmds, buf, 0, staging, std.mem.sliceAsBytes(bits), timeout)) catch |e| return gpuFail(row.name, e);

    // the staged entry table: toks then amounts, exactly as the engine lays it out
    var toks: [fx.count]u32 = undefined;
    var subs: [fx.count]f32 = undefined;
    for (row.entries, 0..) |e, i| {
        toks[i] = e.tok;
        subs[i] = e.sub;
    }
    const n: u32 = @intCast(row.entries.len);
    // toks at 0, amounts at a 256-byte multiple: a storage binding's offset must be aligned
    const amounts_at: u64 = 256;
    var table = vk.Buffer.init(dev, .device, amounts_at + 4 * fx.count) catch |e| return gpuFail(row.name, e);
    defer table.deinit();
    vk.orExit(vk.upload(cmds, table, 0, staging, std.mem.sliceAsBytes(toks[0..n]), timeout)) catch |e| return gpuFail(row.name, e);
    vk.orExit(vk.upload(cmds, table, amounts_at, staging, std.mem.sliceAsBytes(subs[0..n]), timeout)) catch |e| return gpuFail(row.name, e);

    const view = vk.tensor.View.init(&buf, .bf16, fx.at * 2, fx.count) catch |e| return gpuFail(row.name, e);
    pipe.bind(&.{
        view.bind(),
        .{ .buffer = &table, .len = @as(u64, n) * 4 },
        .{ .buffer = &table, .offset = amounts_at, .len = @as(u64, n) * 4 },
    }) catch |e| return gpuFail(row.name, e);
    const cmd = cmds.begin() catch |e| return gpuFail(row.name, e);
    pipe.record(cmd, std.mem.asBytes(&n), .{ (n + 255) / 256, 1, 1 }) catch |e| return gpuFail(row.name, e);
    vk.orExit(cmds.submit(timeout)) catch |e| return gpuFail(row.name, e);

    // the row back, whole shell: every fixed word exact, poison intact
    vk.orExit(vk.download(cmds, buf, 0, staging, std.mem.sliceAsBytes(bits), timeout)) catch |e| return gpuFail(row.name, e);
    for (row.wants) |w| {
        const got = bits[fx.at + w.tok];
        if (got != w.word) {
            std.debug.print("FAIL penalties row=\"{s}\" token={d}: read back 0x{x:0>4}, the CUDA formula says 0x{x:0>4}\n", .{ row.name, w.tok, got, w.word });
            return false;
        }
    }
    var poison_ok = true;
    for (bits[fx.at + fx.count ..]) |b| poison_ok = poison_ok and b == fx.poison16;
    if (!poison_ok) {
        std.debug.print("FAIL penalties row=\"{s}\": the poison tail is not intact\n", .{row.name});
        return false;
    }
    for (bits[fx.at..][0..fx.count], 0..) |b, t| {
        var touched = false;
        for (row.entries) |e| touched = touched or e.tok == t;
        var planted: ?u16 = null;
        for (row.planted) |p| if (p.tok == t) {
            planted = p.word;
        };
        const untouched = if (planted) |w| w else @as(u16, 0);
        if (!touched and b != untouched) {
            std.debug.print("FAIL penalties row=\"{s}\" token={d}: an untouched word changed to 0x{x:0>4}\n", .{ row.name, t, b });
            return false;
        }
    }

    // the outcome gate runs the production greedy over the penalized row
    var g = try vk.Greedy.init(dev, .{
        .finite_logits = &shaders.finite_logits,
        .argreduce_chunk = &shaders.argreduce_chunk,
        .argreduce_combine = &shaders.argreduce_combine,
    }, fx.count);
    defer g.deinit();
    switch (row.outcome) {
        .winner => |want| {
            const tok = try vk.orExit(g.select(cmds, staging, view, timeout));
            if (tok != want) {
                std.debug.print("FAIL penalties row=\"{s}\": the greedy draw selected {d}, the penalized row's winner is {d}\n", .{ row.name, tok, want });
                return false;
            }
        },
        .non_finite => {
            _ = g.select(cmds, staging, view, timeout) catch |e| {
                if (e == error.NonFinite) return true;
                return gpuFail(row.name, e);
            };
            std.debug.print("FAIL penalties row=\"{s}\": the greedy draw accepted a row with a planted NaN\n", .{row.name});
            return false;
        },
    }
    return true;
}

fn gpuFail(gate: []const u8, e: anyerror) bool {
    std.debug.print("FAIL penalties row=\"{s}\": {s}\n", .{ gate, @errorName(e) });
    return false;
}
