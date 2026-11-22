//! The MXFP8 128x64 prompt tile bit-equal to the 64x64 kernel on every dense projection shape, with a planted control.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const Rig = @import("contract_rig.zig").Rig;
const c = core.contract.spec;
const g = core.contract.gates;
const Dense = core.contract.dense.Dense;
const q = cuda.qmmf;
const cases = @import("qmmf_prompt_mx_tests.zig");

/// Dense MXFP8 projections served unsplit (the 128-row tile's pinned sk stays 1).
const unsplit = [_]c.Shape{
    .{ .rows = 512, .n = 16480, .k = 2560 },
    .{ .rows = 513, .n = 13952, .k = 2560 },
    .{ .rows = 512, .n = 1280, .k = 640 },
    .{ .rows = 640, .n = 2560, .k = 640 },
    .{ .rows = 512, .n = 10240, .k = 320 },
    .{ .rows = 1536, .n = 96, .k = 512 },
    .{ .rows = 4218, .n = 2560, .k = 640 },
};
/// The same projections with their splitK() slices folded inside the 128-row tile's blocks.
const fused_shapes = [_]c.Shape{
    .{ .rows = 512, .n = 2560, .k = 2560 },
    .{ .rows = 513, .n = 640, .k = 2560 },
    .{ .rows = 640, .n = 512, .k = 2560 },
    .{ .rows = 1536, .n = 7296, .k = 2560 },
    .{ .rows = 4218, .n = 2560, .k = 3072 },
    .{ .rows = 512, .n = 324, .k = 10240 },
    .{ .rows = 512, .n = 2560, .k = 6144 },
    .{ .rows = 1536, .n = 1280, .k = 6144 },
    .{ .rows = 1024, .n = 10240, .k = 2560 },
    .{ .rows = 512, .n = 129, .k = 2048 },
    .{ .rows = 4218, .n = 8240, .k = 2560 },
};
const fused_fp32 = fused_shapes ++ [_]c.Shape{
    .{ .rows = 1024, .n = 2560, .k = 2560 }, // MTP fc_hidden: four streams of a 256-token piece
    .{ .rows = 1536, .n = 320, .k = 10240 }, // final mixer under .all logits; padded N is 384
};

fn slices(shape: c.Shape, fused: bool) u32 {
    return if (fused) q.splitK(shape.n, shape.k) else 1;
}

fn depth(shape: c.Shape) usize {
    return shape.k / 16 + 24;
}

pub fn spec(out: q.Out, fused: bool, control: bool) c.Spec {
    return .{
        .name = if (control) (if (out == .f32) "control: qmmf MXFP8 prompt fp32 scaled 1.5x" else "control: qmmf MXFP8 prompt bf16 scaled 1.5x") else switch (out) {
            .bf16 => if (fused) "cuda qmmf MXFP8 prompt 128x64 bf16 fused" else "cuda qmmf MXFP8 prompt 128x64 bf16 unsplit",
            .f32 => if (fused) "cuda qmmf MXFP8 prompt 128x64 fp32 fused" else "cuda qmmf MXFP8 prompt 128x64 fp32 unsplit",
        },
        .accum = .f32,
        .depth = depth,
        .ops = 1,
        .out = if (out == .f32) .f32 else .bf16,
        .replaces = "qmmf MXFP8 64x64 with identical pinned K slices",
        .tiles = .{ .{ .size = 128 }, .{ .size = 128, .min = 128 }, .{ .size = if (fused) 2048 else 512, .step = 256, .min = if (fused) 2048 else 512 } },
        .served = if (fused) (if (out == .f32) &fused_fp32 else &fused_shapes) else &unsplit,
    };
}

pub const Product = struct {
    rig: Rig,
    lane: q.Lane, // a copy with the tile loaded: other sets sharing the module keep the lane alone
    format: q.Out,
    fused: bool,
    control: bool = false,
    d: ?Dense = null,
    w: q.Weight = undefined,
    x: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,
    part: u64 = 0,

    pub fn deinit(self: *Product) void {
        if (self.d) |d| d.deinit();
        self.rig.deinit();
    }
    const vt = g.hooks(Product);
    pub fn kernel(self: *Product) g.Kernel {
        return .{ .spec = spec(self.format, self.fused, self.control), .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null };
    }
    fn width(self: *const Product) usize {
        return if (self.format == .f32) 4 else 2;
    }
    pub fn prepare(self: *Product, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        self.part = 0;
        if (shape.rows == 0 or shape.n < 64 or shape.k % 64 != 0 or shape.k < 320) return false;
        const a = self.rig.gpu.gpa;
        const d = try Dense.init(a, shape, 1);
        self.d = d;
        d.fillX(fill, seed, 16);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x8f4a);
        const random = prng.random();
        const n = shape.n;
        const k = shape.k;
        const codes = try a.alloc(u8, n * k);
        defer a.free(codes);
        const scales = try a.alloc(u8, n * k / 32);
        defer a.free(scales);
        for (scales) |*s| s.* = if (fill == .stress) 127 else 123 + random.uintLessThan(u8, 8);
        for (0..n) |j| for (0..k) |i| {
            codes[j * k + i] = if (fill == .stress) 0x38 + random.uintLessThan(u8, 24) else random.uintLessThan(u8, 0x7f);
            if (fill != .stress and random.boolean()) codes[j * k + i] |= 0x80;
        };
        if (fill == .stress) for (0..n) |j| for (k / 2..k) |i| {
            codes[j * k + i] = codes[j * k + i - k / 2] | 0x80;
        };
        for (0..n) |j| for (0..k) |i| {
            const scale = std.math.ldexp(@as(f64, 1), @as(i32, scales[j * (k / 32) + i / 32]) - 127);
            d.w[j * k + i] = core.quant.fp8.e4m3(codes[j * k + i]) * scale;
        };
        const npad = cuda.mxfp8.padded(n);
        const tiled_codes = try a.alloc(u8, npad * k);
        defer a.free(tiled_codes);
        const packed_scales = try a.alloc(u8, npad * k / 32);
        defer a.free(packed_scales);
        cuda.mxfp8.packCodes(tiled_codes, codes, n, k);
        cuda.mxfp8.packScales(packed_scales, scales, n, k);
        const codes_dev = try self.rig.input(tiled_codes);
        const scales_dev = try self.rig.input(packed_scales);
        const sk = slices(shape, self.fused);
        self.w = try q.pinned(cuda.mxfp8.weight(codes_dev, scales_dev, n, k), sk);
        self.x = try self.rig.input(std.mem.sliceAsBytes(d.x));
        const bytes = self.count() * self.width();
        self.out = try self.rig.output(bytes);
        self.base = try self.rig.scratch(bytes);
        const partials = self.lane.partBytes(shape.rows, self.w);
        self.part = if (partials == 0) 0 else try self.rig.scratch(partials);
        if (shape.rows >= q.prompt_rows and (!q.promptEligible(shape.rows, self.w) or self.lane.prompt_mx == null)) return error.PromptPathNotSelected;
        return true;
    }
    pub fn run(self: *Product) !void {
        const shape = self.d.?.shape;
        var w = self.w;
        if (self.control) w.scale = 1.5;
        if (self.lane.usesPrompt(shape.rows, w)) {
            try self.lane.matmulOut(self.rig.s, self.x, shape.k, shape.rows, w, self.out, self.format, self.part);
        } else {
            // Shrunk gates and grids under the SM gate would take the lane: force the instance instead.
            try self.forced(w, self.out);
        }
        try self.rig.s.synchronize();
    }
    fn forced(self: *Product, w: q.Weight, dest: u64) !void {
        const shape = self.d.?.shape;
        const band: usize = @max(1, @min((shape.rows + 127) / 128, (12 << 20) / (128 * shape.k * 2)));
        var args: cuda.Args = .{};
        args.add(self.x);
        args.add(w.codes);
        args.add(w.scales);
        args.add(w.scale);
        args.add(dest);
        args.add(@as(u64, 0));
        for ([_]usize{ shape.rows, shape.n, shape.k, w.sk, w.npad, shape.k, band }) |v| args.add(@as(i32, @intCast(v)));
        const fun = self.lane.prompt_mx.?[@backingInt(self.format)][@intFromBool(self.fused)];
        try cuda.launch.launch(fun, .{
            .grid = .{ .x = @intCast(((shape.rows + 127) / 128) * ((shape.n + 63) / 64)) },
            .block = .{ .x = 256 },
            .shared = q.prompt_mx_shared,
        }, self.rig.s, &args);
    }
    pub fn replaced(self: *Product) ![]const u8 {
        var lane = self.lane;
        lane.prompt_mx = null;
        const shape = self.d.?.shape;
        try lane.matmulOut(self.rig.s, self.x, shape.k, shape.rows, self.w, self.base, self.format, self.part);
        return self.rig.bytes(self.base, self.count() * self.width());
    }
    pub fn count(self: *Product) usize {
        return self.d.?.outputs();
    }
    pub fn reference(self: *Product, want: []f64, magnitude: []f64) !void {
        try self.d.?.reference(want, magnitude);
    }
    pub fn simulate(self: *Product, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.d.?.simulate(acc, order, out);
    }
};

pub const Registrations = struct {
    products: [6]Product,
    pub fn init(gpu: check.Gpu, s: cuda.Stream, lane: q.Lane) Registrations {
        var r: Registrations = undefined;
        for ([_]q.Out{ .bf16, .f32 }, 0..) |out, oi| for ([_]bool{ false, true }, 0..) |fused, fi| {
            r.products[oi * 2 + fi] = .{ .rig = .{ .gpu = gpu, .s = s }, .lane = lane, .format = out, .fused = fused };
        };
        r.products[4] = .{ .rig = .{ .gpu = gpu, .s = s }, .lane = lane, .format = .bf16, .fused = false, .control = true };
        r.products[5] = .{ .rig = .{ .gpu = gpu, .s = s }, .lane = lane, .format = .bf16, .fused = true, .control = true };
        return r;
    }
    pub fn deinit(self: *Registrations) void {
        for (&self.products) |*p| p.deinit();
    }
    pub fn append(self: *Registrations, cap: u32, a: std.mem.Allocator, kernels: *std.ArrayList(g.Kernel)) !void {
        if (cap / 10 != 12) return;
        for (&self.products) |*p| try kernels.append(a, p.kernel());
    }
};

pub fn run(gpu: check.Gpu) !void {
    const cap = try gpu.ctx.capability();
    try check.expect(cap / 10 == 12, "qmmf MXFP8 prompt contract requires sm_12x", .{});
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var lane = try q.Lane.load(gpu.d, cap / 10);
    defer lane.unload();
    try lane.loadPrompt(gpu.d);
    var regs = Registrations.init(gpu, s, lane);
    defer regs.deinit();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    try regs.append(cap, gpu.gpa, &kernels);
    var tally: g.Tally = .{};
    try g.runAll(gpu.gpa, kernels.items, 0x8f4a, &tally);
    std.debug.print("SUMMARY qmmf MXFP8 prompt gates 1-4: {d} pass, {d} fail; {d} controls red, {d} missed\n", .{ tally.passed, tally.failed, tally.caught, tally.missed });
    try check.expect(tally.problems() == 0, "qmmf MXFP8 prompt contract failed", .{});
}

test "qmmf MXFP8 prompt contract registers the 128-row tile, its controls and every projection shape" {
    const a = std.testing.allocator;
    const lane: q.Lane = undefined;
    var regs = Registrations.init(undefined, undefined, lane);
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(a);
    try regs.append(89, a, &kernels);
    try std.testing.expectEqual(@as(usize, 0), kernels.items.len);
    try regs.append(121, a, &kernels);
    try std.testing.expectEqual(@as(usize, 6), kernels.items.len);
    inline for (.{ q.Out.bf16, q.Out.f32 }, 0..) |out, oi| inline for (.{ false, true }, 0..) |fused, fi| {
        const k = kernels.items[oi * 2 + fi];
        try std.testing.expect(k.spec.replaces != null and k.vt.replaced != null);
        try std.testing.expect(k.vt.simulate != null and k.control == null);
        const symbol = q.promptMxSymbol(out, fused);
        const record = for (core.contract.registry.all_cuda) |r| {
            if (std.mem.eql(u8, r.launch.symbol, symbol)) break r;
        } else return error.MissingPromptMxRecord;
        try std.testing.expectEqual(q.prompt_mx_shared, record.launch.dynamic);
        try std.testing.expectEqual(@as(u32, 256), record.launch.threads);
        try std.testing.expect(record.launch.opt_in);
        for (k.spec.served) |shape| {
            const sk = slices(shape, fused);
            const w = try q.pinned(cuda.mxfp8.weight(0, 0, shape.n, shape.k), sk);
            try std.testing.expect(shape.rows >= q.prompt_rows and q.promptEligible(shape.rows, w));
            try std.testing.expectEqual(fused, q.planFor(shape.rows, w, true).fused);
        }
    };
    try std.testing.expectEqual(c.Gate.shapes, kernels.items[4].control.?);
    try std.testing.expectEqual(c.Gate.shapes, kernels.items[5].control.?);
    try std.testing.expect(!regs.products[4].fused and regs.products[5].fused);
}

test "qmmf MXFP8 prompt contract covers every dense MXFP8 projection shape" {
    for (cases.projections) |projection| {
        const sk = q.splitK(projection.n, projection.k);
        const w = try q.pinned(cuda.mxfp8.weight(0, 0, projection.n, projection.k), sk);
        try std.testing.expect(q.promptEligible(q.prompt_rows, w));
        const s = spec(projection.out, sk > 1, false);
        var found = false;
        for (s.served) |shape| {
            if (shape.n == projection.n and shape.k == projection.k) found = true;
        }
        try std.testing.expect(found);
    }
}
