//! Decode microbenchmark for qmm_q4_tile at m=1 over the Qwen3.5-2B decode shapes; throwaway tooling, not a contract.
const std = @import("std");
const vk = @import("vulkan");
const params_mod = @import("stage1_params");
const pinned = @import("qwen35_config");
const bench_shaders = @import("qmm_bench_shaders");
const Params = params_mod.Params;
const dispatch = vk.dispatch;
const group: usize = 64;
const warmup_dispatches: usize = 20;
const copies_bytes: usize = 256 << 20;
const copies_cap: usize = 64;
const staging_bytes: usize = 8 << 20;
const max_variants: usize = 7;
const seed: u64 = 0x766b;

const usage = "usage: tf-qmm-decode-bench [--vendor ID] [--timeout-ms MS] [--reps N] [--shader LABEL=PATH.spv]... [--shapes small]\n" ++
    "  --vendor ID       refuse a device from any other PCI vendor (Apple is 0x106b, Mesa is 0x10005)\n" ++
    "  --timeout-ms MS   bound on every fence wait, 1 to 600000 (default 10000); a timeout ends the process\n" ++
    "  --reps N          timed dispatches per shape and label (default 400)\n" ++
    "  --shader L=P.spv  load another qmm_q4_tile-compatible image as variant LABEL\n" ++
    "  --nobarrier       also time the dispatches without the per-dispatch compute barrier, both modes in one run\n" ++
    "  --shapes small    only the three smallest shapes, and reps 50 unless --reps is given\n";

/// One decode projection: n outputs, k inputs, dispatches per decode step, and the loader names it came from.
const Shape = struct { n: usize, k: usize, count: usize, source: []const u8 };

/// The per-step table from pinned config.zig and the weights loader: every tiled projection at m=1 plus the tied head.
fn shapeTable() [7]Shape {
    var linear_layers: usize = 0;
    var full_layers: usize = 0;
    for (0..pinned.layers) |i| if (pinned.linear(i)) {
        linear_layers += 1;
    } else {
        full_layers += 1;
    };
    const hidden = pinned.hidden;
    const kv = pinned.kv_heads * pinned.head_dim;
    return .{
        .{ .n = pinned.linear_heads, .k = hidden, .count = linear_layers * 2, .source = "in_proj_a,in_proj_b" },
        .{ .n = kv, .k = hidden, .count = full_layers * 2, .source = "k_proj,v_proj" },
        .{ .n = hidden, .k = hidden, .count = linear_layers * 2 + full_layers, .source = "in_proj_z,out_proj,o_proj" },
        .{ .n = hidden, .k = pinned.intermediate, .count = pinned.layers, .source = "mlp.down_proj" },
        .{ .n = 2 * pinned.query_heads * pinned.head_dim, .k = hidden, .count = full_layers, .source = "q_proj" },
        .{ .n = pinned.intermediate, .k = hidden, .count = linear_layers * 3, .source = "in_proj_qkv,mlp.gate_proj,mlp.up_proj" },
        .{ .n = pinned.vocab, .k = hidden, .count = 1, .source = "lm_head(tied embedding)" },
    };
}

/// Bytes charged per dispatch over the padded (stored) columns: 4-bit words, bf16 scales and biases, x, and out.
fn shapeBytes(s: Shape) u64 {
    const stored: u64 = vk.tile.padded(s.n);
    return stored * s.k / 2 + 2 * (stored * (s.k / group)) * 2 + s.k * 2 + s.n * 2;
}

fn bf16F32(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

/// The shader's bf16_store: round to nearest even on the f32 bits; NaNs keep their payload bit.
fn bf16Round(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @intCast((bits >> 16) | 0x40);
    return @intCast((bits +% 0x7fff +% (bits >> 16 & 1)) >> 16);
}

fn nextRandom(state: *u64) u32 {
    state.* = state.* *% 6364136223846793005 +% 1442695040888963407;
    return @truncate(state.* >> 33);
}

/// Host fixture for one shape: row-layout weights for the reference, tiled buffers for the device.
const Fixture = struct {
    x: []u16,
    w_words: []u32,
    scales: []u16,
    biases: []u16,
    w_tiled: []u8,
    scales_tiled: []u8,
    biases_tiled: []u8,

    /// Values keep every shader arithmetic step exact in fp32, so the CPU reference matches any sum order.
    fn init(a: std.mem.Allocator, s: Shape) !Fixture {
        var rand: u64 = seed +% s.n *% 31 +% s.k;
        const x = try a.alloc(u16, s.k);
        errdefer a.free(x);
        for (x) |*v| v.* = bf16Round(@as(f32, @floatFromInt(nextRandom(&rand) % 16 + 1)) * 0.015625);
        const groups = s.k / group;
        const scales = try a.alloc(u16, s.n * groups);
        errdefer a.free(scales);
        const biases = try a.alloc(u16, s.n * groups);
        errdefer a.free(biases);
        for (scales) |*v| v.* = bf16Round(if (nextRandom(&rand) & 1 == 0) 1.0 else 2.0);
        for (biases) |*v| v.* = bf16Round(@as(f32, @floatFromInt(nextRandom(&rand) % 8)) * 0.5);
        const w_words = try a.alloc(u32, s.n * s.k / 8);
        errdefer a.free(w_words);
        for (w_words) |*v| v.* = nextRandom(&rand);
        const w_tiled = try a.alloc(u8, vk.tile.padded(s.n) * s.k / 2);
        errdefer a.free(w_tiled);
        const scales_tiled = try a.alloc(u8, vk.tile.padded(s.n) * s.k / 32);
        errdefer a.free(scales_tiled);
        const biases_tiled = try a.alloc(u8, vk.tile.padded(s.n) * s.k / 32);
        errdefer a.free(biases_tiled);
        vk.tile.tileWords(w_tiled, std.mem.sliceAsBytes(w_words), s.n, s.k);
        vk.tile.tileParams(scales_tiled, std.mem.sliceAsBytes(scales), s.n, s.k);
        vk.tile.tileParams(biases_tiled, std.mem.sliceAsBytes(biases), s.n, s.k);
        return .{ .x = x, .w_words = w_words, .scales = scales, .biases = biases, .w_tiled = w_tiled, .scales_tiled = scales_tiled, .biases_tiled = biases_tiled };
    }

    fn deinit(f: *Fixture, a: std.mem.Allocator) void {
        a.free(f.x);
        a.free(f.w_words);
        a.free(f.scales);
        a.free(f.biases);
        a.free(f.w_tiled);
        a.free(f.scales_tiled);
        a.free(f.biases_tiled);
        f.* = undefined;
    }

    /// fp32 accumulate in ascending k with the shader's exact products, then the shader's bf16 rounding.
    fn reference(f: *const Fixture, s: Shape, out: []u16) void {
        const groups = s.k / group;
        for (0..s.n) |c| {
            var acc: f32 = 0;
            for (0..s.k) |j| {
                const word = f.w_words[c * (s.k / 8) + j / 8];
                const nib: f32 = @floatFromInt((word >> @as(u5, @intCast((j % 8) * 4))) & 0xf);
                const g = c * groups + j / group;
                const q = bf16F32(f.scales[g]) * nib + bf16F32(f.biases[g]);
                acc += bf16F32(f.x[j]) * q;
            }
            out[c] = bf16Round(acc);
        }
    }
};

const ArgShader = struct { label: []const u8, path: []const u8 };
const bench_modes: [2][]const u8 = .{ "barrier", "nobarrier" };

const Options = struct {
    vendor: ?u32 = null,
    timeout_ms: u64 = 10_000,
    reps: usize = 400,
    small: bool = false,
    nobarrier: bool = false,
    explicit_reps: bool = false,
    shader_args: [max_variants]ArgShader = undefined,
    shader_count: usize = 0,
};

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var o: Options = .{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const value = if (i + 1 < args.len) args[i + 1] else "";
        if (std.mem.eql(u8, args[i], "--vendor")) {
            o.vendor = std.fmt.parseInt(u32, value, 0) catch break;
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--timeout-ms")) {
            o.timeout_ms = std.fmt.parseInt(u64, value, 10) catch 0;
            if (o.timeout_ms == 0 or o.timeout_ms > 600_000) break;
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--reps")) {
            o.reps = std.fmt.parseInt(usize, value, 10) catch 0;
            if (o.reps == 0) break;
            o.explicit_reps = true;
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--shapes")) {
            if (!std.mem.eql(u8, value, "small")) break;
            o.small = true;
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--nobarrier")) {
            o.nobarrier = true;
        } else if (std.mem.eql(u8, args[i], "--shader")) {
            const split = std.mem.indexOfScalar(u8, value, '=') orelse break;
            if (split == 0 or o.shader_count == max_variants) break;
            o.shader_args[o.shader_count] = .{ .label = value[0..split], .path = value[split + 1 ..] };
            o.shader_count += 1;
            i += 1;
        } else break;
    } else return run(init, o) catch |e| {
        std.debug.print("FAIL {s}\n", .{@errorName(e)});
        return 1;
    };
    std.debug.print(usage, .{});
    return 2;
}

/// One runtime-loaded or embedded image with its own pipeline over the shared 5-binding qmm_q4_tile layout.
const Label = struct { name: []const u8, pipe: *dispatch.Pipeline, own: ?[]align(4) u8 = null };

fn run(init: std.process.Init, o_in: Options) !u8 {
    const a = init.gpa;
    var o = o_in;
    if (o.small and !o.explicit_reps) o.reps = 50;
    const timeout = o.timeout_ms * std.time.ns_per_ms;
    var rt = try vk.Runtime.open();
    defer rt.close();
    var dev = try vk.Device.open(&rt, .{ .features = .{ .storage16 = true } });
    defer dev.close();
    const pr = dev.properties;
    std.debug.print("DEVICE name=\"{s}\" vendor=0x{x:0>4} device=0x{x:0>4}\n", .{ std.mem.sliceTo(&pr.deviceName, 0), pr.vendorID, pr.deviceID });
    if (pr.deviceType == vk.c.VK_PHYSICAL_DEVICE_TYPE_CPU) {
        std.debug.print("FAIL device: a CPU (software) Vulkan device is never a GPU result\n", .{});
        return 1;
    }
    if (o.vendor) |want| if (pr.vendorID != want) {
        std.debug.print("FAIL device: vendor 0x{x:0>4} is not the requested 0x{x:0>4}\n", .{ pr.vendorID, want });
        return 1;
    };
    var cmds = try vk.Commands.init(&dev);
    defer cmds.deinit();
    var staging = try vk.Buffer.init(&dev, .staging, staging_bytes);
    defer staging.deinit();
    const table = shapeTable();
    const picked: []const Shape = if (o.small) table[0..3] else table[0..];
    var step_total: usize = 0;
    for (picked) |s| {
        std.debug.print("SHAPES n={d} k={d} count={d} {s}\n", .{ s.n, s.k, s.count, s.source });
        step_total += s.count;
    }
    std.debug.print("DEVICE dispatches-per-step={d} reps={d}\n", .{ step_total, o.reps });

    const layout: dispatch.Layout = .{ .bindings = 5, .push_bytes = @sizeOf(Params), .writes = 1 << 4 };
    var default_pipe = try dispatch.Pipeline.init(&dev, &bench_shaders.qmm_q4_tile, layout);
    defer default_pipe.deinit();
    var labels: [max_variants + 1]Label = undefined;
    var label_count: usize = 1;
    labels[0] = .{ .name = "default", .pipe = &default_pipe };
    defer for (labels[1..label_count]) |l| if (l.own) |code| {
        l.pipe.deinit();
        a.destroy(l.pipe);
        a.free(code);
    };
    for (o.shader_args[0..o.shader_count]) |arg| {
        const raw = try std.Io.Dir.cwd().readFileAlloc(init.io, arg.path, a, .limited(64 << 20));
        const code = try a.alignedAlloc(u8, .@"4", raw.len);
        @memcpy(code, raw);
        a.free(raw);
        errdefer a.free(code);
        const pipe = try a.create(dispatch.Pipeline);
        errdefer a.destroy(pipe);
        pipe.* = try dispatch.Pipeline.init(&dev, code, layout);
        labels[label_count] = .{ .name = arg.label, .pipe = pipe, .own = code };
        label_count += 1;
        std.debug.print("DEVICE variant \"{s}\" {d} bytes local={d}\n", .{ arg.label, code.len, pipe.local[0] });
    }

    var step_us = try a.alloc(f64, label_count * 2);
    defer a.free(step_us);
    @memset(step_us, 0);
    for (picked, 0..) |s, shape_index| {
        var f = try Fixture.init(a, s);
        defer f.deinit(a);
        var wbufs: [copies_cap]vk.Buffer = undefined;
        var sbufs: [copies_cap]vk.Buffer = undefined;
        var bbufs: [copies_cap]vk.Buffer = undefined;
        var outbufs: [copies_cap]vk.Buffer = undefined;
        var made: usize = 0;
        defer for (0..made) |c| {
            wbufs[c].deinit();
            sbufs[c].deinit();
            bbufs[c].deinit();
            outbufs[c].deinit();
        };
        const per_copy: u64 = @intCast(f.w_tiled.len + f.scales_tiled.len + f.biases_tiled.len);
        const copies: usize = @max(1, @min(copies_cap, (copies_bytes + per_copy - 1) / per_copy));
        for (0..copies) |_| {
            wbufs[made] = try vk.Buffer.init(&dev, .device, f.w_tiled.len);
            sbufs[made] = try vk.Buffer.init(&dev, .device, f.scales_tiled.len);
            bbufs[made] = try vk.Buffer.init(&dev, .device, f.biases_tiled.len);
            outbufs[made] = try vk.Buffer.init(&dev, .device, s.n * 2);
            made += 1;
            try vk.orExit(vk.upload(&cmds, wbufs[made - 1], 0, staging, f.w_tiled, timeout));
            try vk.orExit(vk.upload(&cmds, sbufs[made - 1], 0, staging, f.scales_tiled, timeout));
            try vk.orExit(vk.upload(&cmds, bbufs[made - 1], 0, staging, f.biases_tiled, timeout));
        }
        std.debug.print("DEVICE n={d} k={d} copies={d} copy_bytes={d}\n", .{ s.n, s.k, copies, per_copy });
        var xbuf = try vk.Buffer.init(&dev, .device, f.x.len * 2);
        defer xbuf.deinit();
        try vk.orExit(vk.upload(&cmds, xbuf, 0, staging, std.mem.sliceAsBytes(f.x), timeout));

        const p = Params{ .count = @intCast(s.n), .operation = 4, .reduce_size = @intCast(group), .output_size = @intCast(s.n), .matrix_m = 1, .matrix_n = @intCast(s.n), .matrix_k = @intCast(s.k), .shape = .{ 1, 0, 0, 0 } };
        const groups: u32 = @intCast((s.n + default_pipe.local[0] - 1) / default_pipe.local[0]);

        // correctness: every label's output must reproduce the default barrier-mode output bytes exactly, both modes
        const want = try a.alloc(u16, s.n);
        defer a.free(want);
        try dispatchOne(&labels[0], &cmds, &xbuf, &wbufs[0], &sbufs[0], &bbufs[0], &outbufs[0], &p, groups, timeout);
        try vk.orExit(vk.download(&cmds, outbufs[0], 0, staging, std.mem.sliceAsBytes(want), timeout));
        for (labels[1..label_count]) |l| {
            const got = try a.alloc(u16, s.n);
            defer a.free(got);
            try dispatchOne(&l, &cmds, &xbuf, &wbufs[0], &sbufs[0], &bbufs[0], &outbufs[0], &p, groups, timeout);
            try vk.orExit(vk.download(&cmds, outbufs[0], 0, staging, std.mem.sliceAsBytes(got), timeout));
            const same = std.mem.eql(u16, want, got);
            std.debug.print("EXACT {s} n={d} k={d} mode=barrier {s}\n", .{ l.name, s.n, s.k, if (same) "ok" else "DIFFERENT" });
            if (!same) return 1;
        }
        if (o.nobarrier) for (labels[0..label_count]) |l| {
            const got = try a.alloc(u16, s.n);
            defer a.free(got);
            l.pipe.barrier = false;
            try dispatchOne(&l, &cmds, &xbuf, &wbufs[0], &sbufs[0], &bbufs[0], &outbufs[0], &p, groups, timeout);
            l.pipe.barrier = true;
            try vk.orExit(vk.download(&cmds, outbufs[0], 0, staging, std.mem.sliceAsBytes(got), timeout));
            const same = std.mem.eql(u16, want, got);
            std.debug.print("EXACT {s} n={d} k={d} mode=nobarrier {s}\n", .{ l.name, s.n, s.k, if (same) "ok" else "DIFFERENT" });
            if (!same) return 1;
        };
        if (shape_index == 0) {
            const ref = try a.alloc(u16, s.n);
            defer a.free(ref);
            f.reference(s, ref);
            const same = std.mem.eql(u16, ref, want);
            std.debug.print("REFERENCE n={d} k={d} {s}\n", .{ s.n, s.k, if (same) "ok" else "MISMATCH" });
            if (!same) return 1;
        }

        // timing: cycle the weight and output copies to defeat caches, batched through the engine's submit path
        const bytes = shapeBytes(s);
        const clock = init.io;
        const mode_count: usize = if (o.nobarrier) 2 else 1;
        for (labels[0..label_count], 0..) |l, li| {
            for (bench_modes[0..mode_count], 0..) |mode, mi| {
                l.pipe.barrier = mi == 0;
                var c: usize = 0;
                for (0..warmup_dispatches) |_| {
                    try dispatchOne(&l, &cmds, &xbuf, &wbufs[c], &sbufs[c], &bbufs[c], &outbufs[c], &p, groups, timeout);
                    c = (c + 1) % copies;
                }
                try vk.orExit(cmds.submit(timeout));
                const t0 = std.Io.Clock.awake.now(clock);
                for (0..o.reps) |_| {
                    try dispatchOne(&l, &cmds, &xbuf, &wbufs[c], &sbufs[c], &bbufs[c], &outbufs[c], &p, groups, timeout);
                    c = (c + 1) % copies;
                }
                try vk.orExit(cmds.submit(timeout));
                const ns: u64 = @intCast(t0.durationTo(std.Io.Clock.awake.now(clock)).toNanoseconds());
                const us = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(o.reps)) / 1000.0;
                const gbps = @as(f64, @floatFromInt(bytes)) * @as(f64, @floatFromInt(o.reps)) / @as(f64, @floatFromInt(ns));
                std.debug.print("BENCH {s} n={d} k={d} count={d} us={d:.3} gbps={d:.2} mode={s}\n", .{ l.name, s.n, s.k, s.count, us, gbps, mode });
                step_us[li * 2 + mi] += us * @as(f64, @floatFromInt(s.count)) / 1000.0;
            }
            l.pipe.barrier = true;
        }
    }
    const mode_count: usize = if (o.nobarrier) 2 else 1;
    for (labels[0..label_count], 0..) |l, li| for (bench_modes[0..mode_count], 0..) |mode, mi| {
        std.debug.print("STEP {s} mode={s} step_ms={d:.3}\n", .{ l.name, mode, step_us[li * 2 + mi] });
    };
    return 0;
}

/// One dispatch: bind the given copy, record, and let the batched submission flush at half a batch.
fn dispatchOne(l: *const Label, cmds: *vk.Commands, x: *vk.Buffer, w: *vk.Buffer, s: *vk.Buffer, b: *vk.Buffer, out: *vk.Buffer, p: *const Params, groups: u32, timeout: u64) !void {
    const B = dispatch.Binding;
    const binds = [_]B{ B{ .buffer = x, .len = x.len }, B{ .buffer = w, .len = w.len }, B{ .buffer = s, .len = s.len }, B{ .buffer = b, .len = b.len }, B{ .buffer = out, .len = out.len } };
    try l.pipe.bind(&binds);
    try vk.orExit(l.pipe.run(cmds, std.mem.asBytes(p), .{ groups, 1, 1 }, timeout));
}
