//! tf-expinf-probe: exp(-inf) and the GDN gate chain on a real Vulkan GPU — the direct
//! driver data for the a = +inf question. One dispatch, seven inputs (bit patterns read
//! from a storage buffer, so nothing folds), nine formulas each, every output compared
//! against an f64 expectation with an exact class match and a 4-ulp f32 tolerance for
//! finite values; exp underflow accepts exact zero or denormal-flush-to-zero below 1e-37.
//! Exit 0 pass, 1 fail, 2 usage/device refusal, 124 fence timeout.
const std = @import("std");
const fx = @import("expinf_fixture.zig");
const vk = @import("vulkan");
const shaders = @import("expinf_shaders");
const dispatch = vk.dispatch;

const usage = "usage: tf-expinf-probe [--vendor ID] [--timeout-ms MS] | --help\n" ++
    "  --help          show this text and exit without opening a loader or device\n" ++
    "  --vendor ID     refuse a device whose Vulkan vendorID differs (decimal or 0x hex; Apple is 0x106b)\n" ++
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
    } else return if (try probe(init.gpa, clock, vendor, timeout_ms)) 0 else 1;
    std.debug.print(usage, .{});
    return 2;
}

fn probe(a: std.mem.Allocator, clock: Elapsed, vendor: ?u32, timeout_ms: u64) !bool {
    const timeout = timeout_ms * std.time.ns_per_ms;
    var rt = try vk.Runtime.open();
    defer rt.close();
    var dev = try vk.Device.open(&rt, .{}); // refuses CPU (software) devices and Vulkan below 1.3
    defer dev.close();
    const pr = dev.properties;
    const name = std.mem.sliceTo(&pr.deviceName, 0);
    const types = [_][]const u8{ "other", "integrated", "discrete", "virtual", "cpu" };
    const kind = if (pr.deviceType < types.len) types[pr.deviceType] else "unknown";
    std.debug.print("MILESTONE device-found elapsed_ms={d} name=\"{s}\" vendor=0x{x:0>4} device=0x{x:0>4} type={s} " ++
        "api={d}.{d}.{d} fence-timeout={d}ms\n", .{ clock.ms(), name, pr.vendorID, pr.deviceID, kind, pr.apiVersion >> 22 & 0x7f, pr.apiVersion >> 12 & 0x3ff, pr.apiVersion & 0xfff, timeout_ms });
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
    var staging = try vk.Buffer.init(&dev, .staging, 64 << 10);
    defer staging.deinit();

    var pipe = try dispatch.Pipeline.init(&dev, &shaders.expinf_probe, .{
        .bindings = 2,
        .push_bytes = @sizeOf(Params),
        .writes = 1 << 1,
    });
    defer pipe.deinit();

    var in = try vk.Buffer.init(&dev, .device, @sizeOf(u32) * fx.input_bits.len);
    defer in.deinit();
    var out = try vk.Buffer.init(&dev, .device, @sizeOf(u32) * fx.outputs);
    defer out.deinit();
    try vk.orExit(vk.upload(&cmds, in, 0, staging, std.mem.sliceAsBytes(&fx.input_bits), timeout));
    const dead: [fx.outputs]u32 = @splat(0xdeadbeef);
    try vk.orExit(vk.upload(&cmds, out, 0, staging, std.mem.sliceAsBytes(&dead), timeout));
    try pipe.bind(&.{ .{ .buffer = &in, .len = @sizeOf(u32) * fx.input_bits.len }, .{ .buffer = &out, .len = @sizeOf(u32) * fx.outputs } });
    const p = Params{ .n = fx.input_bits.len };
    try vk.orExit(pipe.run(&cmds, std.mem.asBytes(&p), .{ fx.n_inputs, 1, 1 }, timeout));
    const bits = try a.alloc(u32, fx.outputs);
    defer a.free(bits);
    try vk.orExit(vk.download(&cmds, out, 0, staging, std.mem.sliceAsBytes(bits), timeout));

    var passed: usize = 0;
    for (0..fx.n_inputs) |i| {
        for (0..fx.n_formulas) |f| {
            const got_bits = bits[i * fx.n_formulas + f];
            const got: f32 = @bitCast(got_bits);
            const v = fx.check(got_bits, i, f);
            const ok = std.meta.activeTag(v) == .ok;
            const w = fx.want(i, f);
            std.debug.print("x={s} {s}: got=0x{x:0>8} ({s}) want={d} ({s}) {s}\n", .{
                fx.input_names[i], fx.formulas[f], got_bits, fx.className(fx.classify(got)),
                w, fx.className(fx.classify(w)), if (ok) "ok" else "FAIL",
            });
            if (!ok) {
                switch (v) {
                    .class_mismatch => |m| std.debug.print("FAIL expinf x={s} {s}: class {s}, expected {s} (observed bits above are quotable)\n", .{ fx.input_names[i], fx.formulas[f], fx.className(m.got), fx.className(m.want) }),
                    .value_mismatch => |m| std.debug.print("FAIL expinf x={s} {s}: got {e}, expected {e} within {d} ulp (observed bits above are quotable)\n", .{ fx.input_names[i], fx.formulas[f], m.got, m.want, fx.ulps(i, f) }),
                    .ok => {},
                }
            } else passed += 1;
        }
    }
    std.debug.print("RESULT expinf-probe elapsed_ms={d} on \"{s}\": passed={d} total={d}\n", .{ clock.ms(), name, passed, fx.outputs });
    return passed == fx.outputs;
}

const Params = extern struct { n: u32 };
