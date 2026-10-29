//! Vulkan Q4 parallel-K fixture: the qmm_q4_pk kernel through the shared gates at the engine's
//! real calls (one-row decode and prefill chunks to the forward pass's chunk ceiling), its five
//! planted controls, and the host models that prove each control cannot pass its gate. Host
//! tests here run without a device; the gates themselves run in tf-vulkan-kernel-contract, whose
//! contract_main.zig wires this fixture's pipelines from the same SPIR-V images the build compiles.
const std = @import("std");
const vk = @import("vulkan");
const contract = @import("contract");
const cap_mod = @import("qwen35_chunk_cap");
const c = contract.spec;
const g = contract.gates;
const dispatch = vk.dispatch;
const qmm = @import("contract_qmm.zig");
const group: usize = 64;
pub const local_size_x: u32 = 64;
pub const poison: u16 = 0x7fc5;
const timeout_ns: u64 = 10 * std.time.ns_per_s;
pub const Params = qmm.Params;
const Rig = qmm.Rig;

/// The engine's per-pass forward chunk ceiling (qwen3_5 vulkan forward.zig's max_prefill_rows,
/// shared from one leaf file): a prompt is cut into passes of at most this many rows, so the
/// projection's row counts are 1 and 2..here. The contract's row envelope and its fixtures
/// derive from this constant, not from a local copy of it.
pub const max_chunk_rows: usize = cap_mod.max_prefill_rows;

/// tf-vulkan-kernel-contract's runAll seed: the host tests below check each control at the exact
/// fill the device run will use, so the literal here and the runner's must stay the same one.
pub const runner_seed: u64 = 0x766b;

fn harnessSeed(of: []const u8) u64 {
    return std.hash.Wyhash.hash(runner_seed, of);
}

/// The elements one lane sums: 8 a pack, ceil(k / 512) strided packs (64 lanes over k / 8 packs).
fn laneElements(k: usize) usize {
    return 8 * ((k / 8 + 63) / 64);
}

/// The longest chain of roundings ONE OUTPUT's sums pass through; the row count never enters,
/// because rows only repeat the same per-output chain, so the bound at a 32-row prefill chunk is
/// the bound at one row. Each term takes three (scale*q, + bias, * x, the qmm counting); the
/// lane's f32 chain adds one per further element of its share (the Neumaier compensation only
/// tightens a plain chain, and its own second-order accumulation is covered by the +1 below);
/// the fixed tree over the 64 lanes adds its six levels; the final high + low add is the last
/// one. At K 2048 that is 42 roundings against the serial kernel's 2050: gamma(42, 2^-24) is
/// about 2.5e-6 where the serial bound's gamma(2050) is about 1.2e-4, so the compensated kernel
/// states the tighter bound, and gate 4 still proves fp32 sums exact.
pub fn depth(s: c.Shape) usize {
    return laneElements(s.k) + 10;
}

/// The projection widths the forward pass calls, as (n, k): the attention k/v 512, the o
/// projection and the GDN z/a/b/out 2048, the q projection 4096, the GDN qkv and the MLP gate/up
/// 6144, all at K 2048; the MLP down at K 6144. The 248320-row head is the finish path's last-row
/// projection, so it is called at one row only.
const projection_pairs = [_][2]usize{ .{ 512, 2048 }, .{ 2048, 2048 }, .{ 4096, 2048 }, .{ 6144, 2048 }, .{ 2048, 6144 } };

const served = blk: {
    var list: []const c.Shape = &.{};
    // the fixture's small shape: the base every tile edge builds on
    list = list ++ [_]c.Shape{.{ .rows = 1, .n = 16, .k = 64 }};
    // the decode calls: every projection at one row
    for (projection_pairs) |pair| {
        list = list ++ [_]c.Shape{.{ .rows = 1, .n = pair[0], .k = pair[1] }};
    }
    // the head, past the workgroup-count clamp so the kernel grid-strides
    list = list ++ [_]c.Shape{.{ .rows = 1, .n = 248320, .k = 2048 }};
    // the prefill calls: the forward pass cuts a prompt into chunks of 2..ceiling rows, every
    // projection width, the same (n, k) as decode
    var rows: usize = 2;
    while (rows <= max_chunk_rows) : (rows += 1) {
        for (projection_pairs) |pair| {
            list = list ++ [_]c.Shape{.{ .rows = rows, .n = pair[0], .k = pair[1] }};
        }
    }
    break :blk list;
};
pub const spec: c.Spec = .{
    .name = "vulkan qmm_q4_pk affine q4 g64 transposed compensated",
    .accum = .f32,
    .depth = depth,
    .out = .bf16,
    // rows' edges sit at one row, the chunk ceiling and one under it; a chunk of ceiling+1 rows
    // is not an engine call (the pass never cuts one), so nothing past the ceiling is served.
    // n's edges sit at the smallest workgroup-count clamp Vulkan allows (65535) and one past it,
    // then at the head; k's edges at the group and the first edge past it. k is always a group
    // multiple: the kernel's packs and scale groups require it.
    .tiles = .{
        .{ .size = max_chunk_rows, .step = 1, .min = 1 },
        .{ .size = 65536, .step = 1, .min = 1 },
        .{ .size = 64, .step = 64, .min = 64 },
    },
    .served = served,
    // gate 4 keeps the harness default: fp16 and bf16 sums must miss by 100x (no claim that output rounding hides them)
};

/// The planted controls, each one fault compiled out of contract_controls_pk.comp.
pub const Kind = enum { kernel, drop_tail, fp16_accum, skip_level, write_weights, write_biases };

pub fn name(kind: Kind) []const u8 {
    return switch (kind) {
        .kernel => spec.name,
        .drop_tail => "control: qmm_q4_pk drops the row's tail pack",
        .fp16_accum => "control: qmm_q4_pk accumulates in fp16",
        .skip_level => "control: qmm_q4_pk tree skips its top level",
        .write_weights => "control: qmm_q4_pk writes its weights",
        .write_biases => "control: qmm_q4_pk writes its biases",
    };
}

fn controlGate(kind: Kind) g.Gate {
    return switch (kind) {
        .kernel => unreachable,
        .drop_tail, .skip_level => .shapes,
        .fp16_accum => .sums,
        .write_weights, .write_biases => .repeat,
    };
}

/// The shapes a control runs at. The two arithmetic faults are exercised at one row and at a
/// full prefill chunk, so the fault is proven caught in both dispatch shapes; n 16 keeps sixteen
/// independent draws sharing the fault at each row count, so one near-cancelling column cannot
/// leave a control green. The fp16 accumulator's gate replays one row's sums (the sums gate caps
/// rows at 4 anyway) and the input writes are row-independent, so one shape carries those.
fn controlShapes(kind: Kind) [2]c.Shape {
    const one = [_]c.Shape{.{ .rows = 1, .n = 16, .k = 2048 }};
    return switch (kind) {
        .drop_tail, .skip_level => .{ one[0], .{ .rows = max_chunk_rows, .n = 16, .k = 2048 } },
        else => .{ one[0], one[0] },
    };
}

fn controlSpec(f: *const Fixture) c.Spec {
    return .{
        .name = name(f.kind),
        .accum = .f32,
        .depth = depth,
        .out = .bf16,
        .served = f.served[0..f.served_n],
        .sums = if (f.kind == .fp16_accum) .{ .f16 = 0, .bf16 = 0, .why = "a planted control: it carries no sums proof, it exists to fail this gate" } else .{},
    };
}

/// Element i of output (r, j)'s sum: the kernel's per-element expression in f64, the fixture
/// family's dequantized form.
pub fn termAt(q: *const contract.affine.Affine, r: usize, j: usize, i: usize) f64 {
    const k = q.shape.k;
    const at = j * (k / q.group) + i / q.group;
    const s: f64 = c.fromBf16(q.scales[at]);
    const b: f64 = c.fromBf16(q.biases[at]);
    const x: f64 = c.fromBf16(q.x[r * k + i]);
    const code: f64 = @floatFromInt(q.codes[j * k + i]);
    return x * (s * code + b);
}

fn roundTo(acc: c.Accum, v: f64) f64 {
    return switch (acc) {
        .f32 => @as(f32, @floatCast(v)),
        .f16 => @as(f16, @floatCast(v)),
        .bf16 => c.fromBf16(c.toBf16(@floatCast(v))),
        .none => v,
    };
}

/// One output's sum in the kernel's own order, replayed in `acc`: 64 lanes over strided packs,
/// then the fixed tree (strides 32, 16, 8, 4, 2, 1). The model's lane sums are plain accumulator
/// chains; the kernel's Neumaier compensation only tightens them.
pub fn pkSum(acc: c.Accum, q: *const contract.affine.Affine, r: usize, j: usize) f64 {
    const packs = q.shape.k / 8;
    var tree: [64]f64 = @splat(0);
    for (0..64) |lane| {
        var partial: f64 = 0;
        var pack: usize = lane;
        while (pack < packs) : (pack += 64) {
            for (0..8) |e| partial = roundTo(acc, partial + roundTo(acc, termAt(q, r, j, pack * 8 + e)));
        }
        tree[lane] = partial;
    }
    var stride: usize = 32;
    while (stride > 0) : (stride /= 2) {
        for (0..stride) |i| tree[i] = roundTo(acc, tree[i] + tree[i + stride]);
    }
    return tree[0];
}

/// A control's arithmetic for output (r, j) on the host, in the shader's f32 lane and tree order,
/// fault and all: the value its kernel stores, before the bf16 store. `kind` must be a control.
pub fn controlSum(kind: Kind, q: *const contract.affine.Affine, r: usize, j: usize) f64 {
    std.debug.assert(kind != .kernel);
    const packs = q.shape.k / 8;
    var tree: [64]f32 = @splat(0);
    for (0..64) |lane| {
        var partial: f32 = 0;
        var pack: usize = lane;
        while (pack < packs) : (pack += 64) {
            if (kind == .drop_tail and lane == 63 and pack + 1 == packs) continue; // the fault
            for (0..8) |e| {
                const term: f32 = @floatCast(termAt(q, r, j, pack * 8 + e));
                if (kind == .fp16_accum) {
                    partial = @floatCast(@as(f16, @floatCast(partial)) + @as(f16, @floatCast(term)));
                } else {
                    partial += term;
                }
            }
        }
        tree[lane] = partial;
    }
    var stride: usize = if (kind == .skip_level) 16 else 32; // the fault: the top level never runs
    while (stride > 0) : (stride /= 2) {
        for (0..stride) |i| tree[i] += tree[i + stride];
    }
    return tree[0];
}

/// The control's stored output (after the bf16 store) for output (r, j).
fn controlStored(kind: Kind, q: *const contract.affine.Affine, r: usize, j: usize) f64 {
    return c.fromBf16(c.toBf16(@floatCast(controlSum(kind, q, r, j))));
}

/// Caps on what prepare() builds on the host, the qmm fixture's: the largest served shape, the
/// 248320-row head at K 2048, takes 0.5 GB of codes plus 0.25 GB of packed words. A served shape
/// over a cap would be refused and fail gate 3; the caps bound the generated edge shapes only.
const max_weights: usize = 520_000_000;
const max_outputs: usize = 300_000;

/// The gate harness's rig and fixture for the kernel and, with a control kind, its controls: the
/// same bindings, pushes and dispatch shape, only the SPIR-V image and the registered spec differ.
pub const Fixture = struct {
    a: std.mem.Allocator,
    dev: *const vk.Device,
    cmds: *vk.Commands,
    staging: vk.Buffer,
    pipeline: *dispatch.Pipeline,
    kind: Kind = .kernel,
    served: [2]c.Shape = undefined,
    served_n: usize = 0,
    shape: c.Shape = .{ .rows = 1, .n = 16, .k = 64 },
    affine: ?contract.affine.Affine = null,
    x: []u16 = &.{}, w: []u32 = &.{}, scales: []u16 = &.{}, biases: []u16 = &.{}, out: []u16 = &.{},
    want: []f64 = &.{}, magnitude: []f64 = &.{}, input_snapshot: []u8 = &.{}, rig: Rig = .{},
    buffers: [5]vk.Buffer = undefined, made: usize = 0, bindings: [5]dispatch.Binding = undefined,

    pub fn init(a: std.mem.Allocator, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, pipeline: *dispatch.Pipeline) Fixture {
        return .{ .a = a, .dev = dev, .cmds = cmds, .staging = staging, .pipeline = pipeline };
    }

    /// A planted control over the same fixture: its shapes are fixed, its gate is the one the fault breaks.
    pub fn initControl(a: std.mem.Allocator, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, pipeline: *dispatch.Pipeline, kind: Kind) Fixture {
        var f = init(a, dev, cmds, staging, pipeline);
        f.kind = kind;
        f.served = controlShapes(kind);
        f.served_n = switch (kind) {
            .drop_tail, .skip_level => 2,
            else => 1,
        };
        return f;
    }

    pub fn deinit(f: *Fixture) void {
        f.reset();
    }
    fn reset(f: *Fixture) void {
        while (f.made > 0) {
            f.made -= 1;
            f.buffers[f.made].deinit();
        }
        if (f.affine) |q| q.deinit();
        f.affine = null;
        for ([_][]u16{ f.x, f.scales, f.biases, f.out }) |b| if (b.len != 0) f.a.free(b);
        if (f.w.len != 0) f.a.free(f.w);
        for ([_][]f64{ f.want, f.magnitude }) |b| if (b.len != 0) f.a.free(b);
        if (f.input_snapshot.len != 0) f.a.free(f.input_snapshot);
        f.x = &.{}; f.w = &.{}; f.scales = &.{}; f.biases = &.{}; f.out = &.{};
        f.want = &.{}; f.magnitude = &.{}; f.input_snapshot = &.{}; f.rig = .{};
    }

    // The table is a comptime constant: `&g.hooks(Fixture)` would point at a stack temporary that dies when kernel() returns.
    const vt = g.hooks(Fixture);

    pub fn kernel(f: *Fixture) g.Kernel {
        if (f.kind == .kernel) return .{ .spec = spec, .ctx = f, .vt = &vt };
        return .{ .spec = controlSpec(f), .ctx = f, .vt = &vt, .control = controlGate(f.kind) };
    }

    fn allocPoison(a: std.mem.Allocator, data_len: usize) ![]u16 {
        const extra: usize = if (data_len % 2 == 0) 16 else 17;
        const p = try a.alloc(u16, data_len + extra);
        @memset(p, poison);
        return p;
    }

    pub fn prepare(f: *Fixture, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k % group != 0 or shape.k < group or shape.n * shape.k > max_weights or shape.rows * shape.n > max_outputs) return false;
        f.reset();
        f.shape = shape;
        var q = try contract.affine.Affine.init(f.a, shape, 4, group);
        errdefer q.deinit();
        q.form = .dequantized;
        try q.fill(switch (fill) {
            .random => .random,
            .nonfinite => .nonfinite,
            .stress => .stress,
        }, seed);
        const x = try allocPoison(f.a, shape.rows * shape.k);
        errdefer f.a.free(x);
        @memcpy(x[0..q.x.len], q.x);
        const words_len = shape.n * shape.k / 8;
        const w = try f.a.alloc(u32, words_len + 4);
        errdefer f.a.free(w);
        @memset(w, 0xffff_ffff);
        q.words(w[0..words_len]);
        const nparam = shape.n * (shape.k / group);
        const scales = try allocPoison(f.a, nparam);
        errdefer f.a.free(scales);
        @memcpy(scales[0..nparam], q.scales);
        const biases = try allocPoison(f.a, nparam);
        errdefer f.a.free(biases);
        @memcpy(biases[0..nparam], q.biases);
        const out = try allocPoison(f.a, shape.rows * shape.n);
        errdefer f.a.free(out);
        const want = try f.a.alloc(f64, shape.rows * shape.n);
        errdefer f.a.free(want);
        const mag = try f.a.alloc(f64, want.len);
        errdefer f.a.free(mag);
        try q.reference(want, mag);
        if (fill != .nonfinite) for (want, mag) |v, m| if (!std.math.isFinite(v) or !std.math.isFinite(m)) return error.NonFiniteReference;
        f.x = x; f.w = w; f.scales = scales; f.biases = biases; f.out = out; f.want = want; f.magnitude = mag; f.affine = q; q = undefined;
        const data = [_][]const u8{ std.mem.sliceAsBytes(f.x), std.mem.sliceAsBytes(f.w), std.mem.sliceAsBytes(f.scales), std.mem.sliceAsBytes(f.biases), std.mem.sliceAsBytes(f.out) };
        var input_total: usize = 0;
        for (data[0..4]) |bytes| input_total += bytes.len;
        f.input_snapshot = try f.a.alloc(u8, input_total);
        var pos: usize = 0;
        for (data[0..4]) |bytes| {
            @memcpy(f.input_snapshot[pos..][0..bytes.len], bytes);
            pos += bytes.len;
            f.buffers[f.made] = try vk.Buffer.init(f.dev, .device, bytes.len);
            const b = &f.buffers[f.made];
            f.made += 1;
            f.bindings[f.made - 1] = .{ .buffer = b, .len = bytes.len };
            try vk.orExit(vk.upload(f.cmds, b.*, 0, f.staging, bytes, timeout_ns));
        }
        f.buffers[f.made] = try vk.Buffer.init(f.dev, .device, data[4].len);
        f.bindings[4] = .{ .buffer = &f.buffers[f.made], .len = data[4].len };
        f.made += 1;
        try f.pipeline.bind(&f.bindings);
        f.rig = .{ .output = std.mem.sliceAsBytes(f.out), .inputs = f.input_snapshot, .guards_ok = true };
        return true;
    }

    pub fn run(f: *Fixture) !void {
        const s = f.shape;
        var p = Params{
            .count = @intCast(s.rows * s.n),
            .output_size = @intCast(s.rows * s.n),
            .matrix_m = @intCast(s.rows),
            .matrix_n = @intCast(s.n),
            .matrix_k = @intCast(s.k),
        };
        p.flags = vk.ops_q4.pk_variant; // strided lanes, compensated sums: the kernel as shipped
        try vk.orExit(vk.upload(f.cmds, f.buffers[4], 0, f.staging, std.mem.sliceAsBytes(f.out), timeout_ns));
        // One 64-lane workgroup per output, clamped at the device's count: the kernel grid-strides the rest.
        const groups = try vk.ops_q4.pkGroups(p.count, f.dev.properties.limits.maxComputeWorkGroupCount[0]);
        try vk.orExit(f.pipeline.run(f.cmds, std.mem.asBytes(&p), .{ groups, 1, 1 }, timeout_ns));
        try vk.orExit(vk.download(f.cmds, f.buffers[4], 0, f.staging, std.mem.sliceAsBytes(f.out), timeout_ns));
        // Gate 2 compares the inputs before and after the runs, so what it sees must be what the
        // device holds now: read the four input buffers back into the snapshot.
        var at: usize = 0;
        for (f.bindings[0..4]) |b| {
            try vk.orExit(vk.download(f.cmds, b.buffer.*, 0, f.staging, f.input_snapshot[at..][0..b.len], timeout_ns));
            at += b.len;
        }
        const data_count = s.rows * s.n;
        f.rig.guards_ok = true;
        for (f.out[data_count..]) |v| if (v != poison) {
            f.rig.guards_ok = false;
            break;
        };
        f.rig.output = std.mem.sliceAsBytes(f.out);
    }

    pub fn count(f: *Fixture) usize {
        return f.want.len;
    }
    pub fn reference(f: *Fixture, want: []f64, mag: []f64) !void {
        @memcpy(want, f.want);
        @memcpy(mag, f.magnitude);
    }

    /// The sums replayed in `acc`: the kernel's own order (64 lanes, then the tree) and, for the
    /// harness's other two orders, the same terms in one chain and in balanced halves. The stress
    /// inputs are exact in fp32 in any of the three, so the order only changes what gate 4 can
    /// prove about fp16 and bf16.
    pub fn simulate(f: *Fixture, accum: c.Accum, order: contract.rows.Order, out: []f64) !void {
        const q = &f.affine.?;
        const k = f.shape.k;
        const terms = try f.a.alloc(f64, k);
        defer f.a.free(terms);
        for (0..f.shape.rows) |r| for (0..f.shape.n) |j| {
            for (0..k) |i| terms[i] = termAt(q, r, j, i);
            out[r * f.shape.n + j] = switch (order) {
                .strided => pkSum(accum, q, r, j),
                else => contract.rows.sumIn(accum, order, terms),
            };
        };
    }
};

const testing = std.testing;

test "planted parallel-K controls bind to the intended gate" {
    inline for (.{ Kind.drop_tail, Kind.skip_level }) |kind| try testing.expectEqual(g.Gate.shapes, controlGate(kind));
    try testing.expectEqual(g.Gate.sums, controlGate(.fp16_accum));
    inline for (.{ Kind.write_weights, Kind.write_biases }) |kind| try testing.expectEqual(g.Gate.repeat, controlGate(kind));
}

test "the served shapes are the engine's calls: one-row decode and prefill chunks to the ceiling" {
    // the ceiling is the engine's forward chunking constant, not a local guess: no served row
    // count passes it, and every projection width repeats at every chunk size 2..ceiling
    var top: usize = 0;
    for (served) |s| {
        try testing.expect(s.rows == 1 or (s.rows >= 2 and s.rows <= max_chunk_rows));
        try testing.expectEqual(@as(usize, 0), s.k % group);
        try testing.expect(s.k >= group);
        top = @max(top, s.rows);
    }
    try testing.expectEqual(max_chunk_rows, top);
    // the head is the finish path's last-row projection: it appears only at one row
    for (served) |s| if (s.n == 248320) try testing.expectEqual(@as(usize, 1), s.rows);
    // the decode calls and the whole prefill range, per width
    for (projection_pairs) |pair| {
        var found_one = false;
        var rows: usize = 2;
        while (rows <= max_chunk_rows) : (rows += 1) {
            var found = false;
            for (served) |s| found = found or (s.rows == rows and s.n == pair[0] and s.k == pair[1]);
            try testing.expect(found);
        }
        for (served) |s| found_one = found_one or (s.rows == 1 and s.n == pair[0] and s.k == pair[1]);
        try testing.expect(found_one);
    }
}

test "the stated depth comes from the lane's strided packs and the six tree levels, per output" {
    // K 64: one pack a lane that has work; K 2048: four packs, 32 elements; K 6144: twelve, 96.
    try testing.expectEqual(@as(usize, 8), laneElements(64));
    try testing.expectEqual(@as(usize, 8), laneElements(128));
    try testing.expectEqual(@as(usize, 32), laneElements(2048));
    try testing.expectEqual(@as(usize, 96), laneElements(6144));
    try testing.expectEqual(@as(usize, 18), depth(.{ .rows = 1, .n = 1, .k = 64 }));
    try testing.expectEqual(@as(usize, 42), depth(.{ .rows = 1, .n = 16, .k = 2048 }));
    try testing.expectEqual(@as(usize, 106), depth(.{ .rows = 1, .n = 2048, .k = 6144 }));
    // rows repeat the per-output chain and add nothing to it: the bound at a full prefill chunk
    // is the bound at one row of the same width
    try testing.expectEqual(depth(.{ .rows = 1, .n = 6144, .k = 2048 }), depth(.{ .rows = max_chunk_rows, .n = 6144, .k = 2048 }));
    // tighter than the serial kernel's k + 2 chain at every served K
    for (served) |s| try testing.expect(depth(s) < qmm.depth(s));
}

fn hostFixture(a: std.mem.Allocator, shape: c.Shape, fill: g.Fill, seed: u64) !contract.affine.Affine {
    var q = try contract.affine.Affine.init(a, shape, 4, group);
    errdefer q.deinit();
    q.form = .dequantized;
    try q.fill(switch (fill) {
        .random => .random,
        .nonfinite => .nonfinite,
        .stress => .stress,
    }, seed);
    return q;
}

test "the parallel-K model meets the bound at the control shapes, so a green control means the fault" {
    const a = testing.allocator;
    for (controlShapes(.drop_tail)) |shape| {
        for (0..8) |seed| {
            var q = try hostFixture(a, shape, .random, 0x70 + seed);
            defer q.deinit();
            for (0..shape.rows) |r| for (0..shape.n) |j| {
                const ref = q.one(r, j);
                const got: f64 = c.fromBf16(c.toBf16(@floatCast(pkSum(.f32, &q, r, j))));
                const allow = c.bound(spec, shape, ref[0], ref[1]);
                try testing.expect(@abs(got - ref[0]) <= allow);
            };
        }
    }
}

/// Gate 3's checker on the host: true when some output leaves its bound (the gate's FAIL).
fn shapesCheckRefuses(kind: Kind, q: *const contract.affine.Affine) bool {
    const shape = q.shape;
    for (0..shape.rows) |r| for (0..shape.n) |j| {
        const ref = q.one(r, j);
        const got = controlStored(kind, q, r, j);
        const allow = c.bound(spec, shape, ref[0], ref[1]);
        if (!std.math.isFinite(got) or !(@abs(got - ref[0]) <= allow)) return true;
    };
    return false;
}

/// Gate 4's device check on the host: true when some output leaves its exact-sum bound.
fn sumsCheckRefuses(kind: Kind, q: *const contract.affine.Affine) bool {
    const shape = q.shape;
    for (0..shape.rows) |r| for (0..shape.n) |j| {
        const ref = q.one(r, j);
        const got = controlStored(kind, q, r, j);
        const allow = c.boundExact(spec, ref[0]);
        if (!std.math.isFinite(got) or !(@abs(got - ref[0]) <= allow)) return true;
    };
    return false;
}

test "every control's arithmetic is refused by its gate's checker at the runner's seeds and stays refused across seeds" {
    const a = testing.allocator;
    // the shapes gate fills shape si with seed + si, so the device run's fills are the seeds here
    inline for (.{ Kind.drop_tail, Kind.skip_level }) |kind| {
        for (controlShapes(kind), 0..) |shape, si| {
            var q = try hostFixture(a, shape, .random, harnessSeed(name(kind)) + si);
            defer q.deinit();
            try testing.expect(shapesCheckRefuses(kind, &q));
        }
        for (0..16) |seed| {
            for (controlShapes(kind)) |shape| {
                var q = try hostFixture(a, shape, .random, 0xc0ffee + seed);
                defer q.deinit();
                try testing.expect(shapesCheckRefuses(kind, &q));
            }
        }
    }
    // gate 4's control: the fp16 accumulator against the exact-sum bound on the stress fill
    {
        const shape = controlShapes(.fp16_accum)[0];
        var q = try hostFixture(a, shape, .stress, harnessSeed(name(.fp16_accum)));
        defer q.deinit();
        try testing.expect(sumsCheckRefuses(.fp16_accum, &q));
        for (0..16) |seed| {
            var s = try hostFixture(a, shape, .stress, 0x5eed + seed);
            defer s.deinit();
            try testing.expect(sumsCheckRefuses(.fp16_accum, &s));
        }
    }
}

test "the fp32 replay of the kernel's own order stays inside the exact-sum bound, fp16 and bf16 far outside" {
    const a = testing.allocator;
    const orders = [_]contract.rows.Order{ .sequential, .pairwise, .strided };
    // the control's shape, and the kernel's own gate-4 shape: the largest served K at one row
    for ([_]c.Shape{ controlShapes(.fp16_accum)[0], .{ .rows = 1, .n = 2048, .k = 6144 } }) |shape| {
        for (orders) |order| {
            var q = try hostFixture(a, shape, .stress, 0x51ce);
            defer q.deinit();
            const n = shape.rows * shape.n;
            const got = try a.alloc(f64, n);
            defer a.free(got);
            const terms = try a.alloc(f64, shape.k);
            defer a.free(terms);
            for (0..shape.rows) |r| for (0..shape.n) |j| {
                for (0..shape.k) |i| terms[i] = termAt(&q, r, j, i);
                got[r * shape.n + j] = switch (order) {
                    .strided => pkSum(.f32, &q, r, j),
                    else => contract.rows.sumIn(.f32, order, terms),
                };
            };
            for (got, 0..) |v, i| {
                const ref = q.one(i / shape.n, i % shape.n);
                try testing.expect(@abs(v - ref[0]) <= c.boundExact(spec, ref[0]));
            }
            for ([_]c.Accum{ .f16, .bf16 }) |acc| {
                var most: f64 = 0;
                for (0..shape.rows) |r| for (0..shape.n) |j| {
                    for (0..shape.k) |i| terms[i] = termAt(&q, r, j, i);
                    const v = switch (order) {
                        .strided => pkSum(acc, &q, r, j),
                        else => contract.rows.sumIn(acc, order, terms),
                    };
                    const ref = q.one(r, j);
                    const allow = c.boundExact(spec, ref[0]);
                    most = @max(most, if (std.math.isFinite(v)) @abs(v - ref[0]) / allow else std.math.inf(f64));
                };
                try testing.expect(most >= 100);
            }
        }
    }
}

/// The weights buffer's layout as prepare() uploads it: n rows of k/8 words, then four guard words.
fn weightsWords(shape: c.Shape) usize {
    return shape.n * shape.k / 8;
}

test "the input-write controls land a fixed value the read-back must see, past the data" {
    const shape = controlShapes(.write_weights)[0];
    const a = testing.allocator;
    var q = try hostFixture(a, shape, .random, harnessSeed(name(.write_weights)));
    defer q.deinit();
    var w = try a.alloc(u32, weightsWords(shape) + 4);
    defer a.free(w);
    @memset(w, 0xffff_ffff);
    q.words(w[0..weightsWords(shape)]);
    // the weights control writes 0xdeadbeef at the first guard word past the data, where no
    // workgroup reads: the bytes the read-back returns must differ from the snapshot's
    const at = weightsWords(shape);
    try testing.expect(at + 1 <= w.len);
    try testing.expect(w[at] != 0xdeadbeef);
    // the biases control writes a fixed bf16 1.0 at the first guard element past its data,
    // which the fixture fills with the 0x7fc5 poison: the bytes must differ there too
    try testing.expect(@as(u16, 0x3f80) != poison);
}

test "the parallel-K model is row-independent: a row's output alone matches its output in a batch" {
    const a = testing.allocator;
    for ([_]usize{ 2048, 6144 }) |k| {
        const n = 6;
        var alone = try hostFixture(a, .{ .rows = 1, .n = n, .k = k }, .random, 0x9e37);
        defer alone.deinit();
        var batch = try hostFixture(a, .{ .rows = 32, .n = n, .k = k }, .random, 0x9e37);
        defer batch.deinit();
        for (0..n) |j| try testing.expectEqual(pkSum(.f32, &alone, 0, j), pkSum(.f32, &batch, 0, j));
    }
}

test "the fixture keeps the qmm storage discipline: raw u16 poison, guards aligned for storage bindings" {
    const values = try Fixture.allocPoison(testing.allocator, 3);
    defer testing.allocator.free(values);
    try testing.expectEqual(@as(usize, 20), values.len);
    try testing.expectEqual(poison, values[0]);
    try testing.expectEqual(qmm.poison, poison);
    try testing.expectEqual(@as(u32, 64), local_size_x);
}
