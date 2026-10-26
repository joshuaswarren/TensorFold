//! The head-only adapter: the CUDA trunk's last rows go to an Intel Arc GPU over Level Zero, the q4g64 head
//! runs there (the existing XPU GEMM kernel), and the logits come back to CUDA before the sampler reads them.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const xpu = @import("xpu");
const arc_kernels = @import("arc_kernels");

const Geometry = @import("cuda_geometry.zig").Geometry;

/// The option: TF_Q35_ARC_HEAD=1 runs the head on an Intel Arc GPU; the CUDA trunk stays where it is.
pub const request_env = "TF_Q35_ARC_HEAD";

/// Whether the environment asks for the Arc head (TF_Q35_ARC_HEAD=1).
pub fn requested() bool {
    const v = std.c.getenv(request_env) orelse return false;
    const t = std.mem.span(v);
    return std.mem.eql(u8, t, "1") or std.mem.eql(u8, t, "true") or std.mem.eql(u8, t, "on");
}

/// The GEMM tile's columns (the XPU loader pads N to it); the padded columns read as zeros.
pub const tile = 256;
/// The rows a work-group writes (gemm.cl's w4_decode reads and writes eight rows at a time).
pub const row_tile = 8;
/// The columns a work-group writes (gemm.cl's w4_decode: SG_N 32 columns a sub-group, four sub-groups).
pub const col_tile = 128;
/// The rows a work-group takes (gemm.cl's w4_decode: SG_M 16 rows a sub-group).
pub const group_rows = 16;
/// The work-group's threads (gemm.cl's w4_decode: 16 lanes a sub-group, four sub-groups).
pub const group_threads = 64;

/// fp16's largest finite value; a bf16 row above it does not fit and is refused.
pub const fp16_max = 65504.0;

/// bf16 bits to fp16 bits; null when the value is not finite or does not fit fp16's range.
pub fn bf16ToFp16(h: u16) ?u16 {
    const v: f32 = @bitCast(@as(u32, h) << 16);
    if (!std.math.isFinite(v)) return null; // a NaN or an infinity has no place in the bridge
    if (@abs(v) > fp16_max) return null; // fp16's largest finite value
    return @bitCast(@as(f16, @floatCast(v)));
}

/// fp16 bits to bf16 bits (round to nearest even); a value is rounded, not kept: bf16 has fewer mantissa bits.
pub fn fp16ToBf16(h: u16) u16 {
    const v: f32 = @as(f32, @as(f16, @bitCast(h)));
    if (std.math.isNan(v)) return 0x7fc0; // a quiet NaN, the one bf16 keeps
    const u: u32 = @bitCast(v);
    return @intCast((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

/// The head's own tensors: the model's embedding when the head is tied to it, else the head's projection.
pub const Tensors = struct { weight: []const u8, scales: []const u8, biases: []const u8 };

pub fn validateFormat(g: Geometry) !void {
    if (g.kinds.modelopt or g.exl3 or g.bits.head != 4) return error.UnsupportedArcHeadFormat;
    if (g.tied and g.bits.embed != 4) return error.UnsupportedArcHeadFormat;
    if (g.hidden == 0 or g.hidden % 64 != 0 or g.vocab < 2) return error.UnsupportedArcHeadFormat;
}

/// The checkpoint's raw MLX affine 4/64 head: the words, the scales and the biases as stored.
pub fn headTensors(ck: *core.Checkpoint, g: Geometry) !Tensors {
    try validateFormat(g);
    var buf: [160]u8 = undefined;
    if (g.tied) {
        // the tied head is the embedding's own rows: the same selection the CUDA loader makes
        const m = "language_model.model.";
        const w = try ck.expect(try std.fmt.bufPrint(&buf, "{s}embed_tokens.weight", .{m}), .u32, &.{ g.vocab, g.hidden / 8 });
        const s = try ck.expect(try std.fmt.bufPrint(&buf, "{s}embed_tokens.scales", .{m}), .bf16, &.{ g.vocab, g.hidden / 64 });
        const b = try ck.expect(try std.fmt.bufPrint(&buf, "{s}embed_tokens.biases", .{m}), .bf16, &.{ g.vocab, g.hidden / 64 });
        return .{ .weight = w.bytes, .scales = s.bytes, .biases = b.bytes };
    }
    const p = "language_model.";
    const w = try ck.expect(try std.fmt.bufPrint(&buf, "{s}lm_head.weight", .{p}), .u32, &.{ g.vocab, g.hidden / 8 });
    const s = try ck.expect(try std.fmt.bufPrint(&buf, "{s}lm_head.scales", .{p}), .bf16, &.{ g.vocab, g.hidden / 64 });
    const b = try ck.expect(try std.fmt.bufPrint(&buf, "{s}lm_head.biases", .{p}), .bf16, &.{ g.vocab, g.hidden / 64 });
    return .{ .weight = w.bytes, .scales = s.bytes, .biases = b.bytes };
}

/// The GEMM layout's word as the MLX word it holds: the code 2i in bits 16+4i and the code 2i+1 in bits 4i.
/// (The inverse of deviceWord; zig/src/xpu/fp64_ref.zig reads the device layout back this way.)
pub fn mlxWord(o: u32) u32 {
    var m: u32 = 0;
    for (0..4) |i| {
        const sh: u5 = @intCast(4 * i);
        m |= ((o >> sh) & 0xF) << (2 * sh); // the code 2i
        m |= ((o >> (sh + 16)) & 0xF) << (2 * sh + 4); // the code 2i+1
    }
    return m;
}

/// The head in the XPU GEMM layout: the words [K/8][NT], the scales and the biases fp16 [groups][NT].
pub const Packed = struct { w: []u32, s: []u16, b: []u16 };

/// The MLX word as the GEMM layout's word: the code 2i in bits 4i and the code 2i+1 in bits 16+4i.
/// (glue_quant.cl's repack does this on the device; zig/src/xpu/fp64_ref.zig reads the layout back that way.)
pub fn deviceWord(m: u32) u32 {
    var o: u32 = 0;
    for (0..4) |i| {
        const sh: u5 = @intCast(4 * i);
        o |= ((m >> (2 * sh)) & 0xF) << sh; // the code 2i
        o |= ((m >> (2 * sh + 4)) & 0xF) << (sh + 16); // the code 2i+1
    }
    return o;
}

/// The head packed for the Arc device: the words transposed and interleaved, the tables group-major and fp16.
/// The bias is the group's own bias (the plain GEMM kernel adds it per weight); the padded columns read as zeros.
pub fn packHead(gpa: std.mem.Allocator, t: Tensors, vocab: usize, hidden: usize, nt: usize) !Packed {
    const groups = hidden / 64;
    const words = hidden / 8;
    if (hidden % 64 != 0 or vocab == 0 or nt < vocab) return error.UnsupportedArcHeadFormat;
    if (t.weight.len < vocab * words * 4 or t.scales.len < vocab * groups * 2 or t.biases.len < vocab * groups * 2) return error.UnexpectedTensor;
    const w = try gpa.alloc(u32, words * nt);
    errdefer gpa.free(w);
    const s = try gpa.alloc(u16, groups * nt);
    errdefer gpa.free(s);
    const b = try gpa.alloc(u16, groups * nt);
    errdefer gpa.free(b);
    @memset(w, 0);
    @memset(s, 0);
    @memset(b, 0);
    const raw = std.mem.bytesAsSlice(u32, t.weight[0 .. vocab * words * 4]);
    const scales = std.mem.bytesAsSlice(u16, t.scales[0 .. vocab * groups * 2]);
    const biases = std.mem.bytesAsSlice(u16, t.biases[0 .. vocab * groups * 2]);
    for (0..vocab) |n| {
        for (0..words) |wi| {
            w[wi * nt + n] = deviceWord(raw[n * words + wi]);
        }
        for (0..groups) |g| {
            const sv = bf16ToFp16(scales[n * groups + g]) orelse return error.ArcHeadTableOverflow;
            const bv = bf16ToFp16(biases[n * groups + g]) orelse return error.ArcHeadTableOverflow;
            s[g * nt + n] = sv;
            b[g * nt + n] = bv;
        }
    }
    return .{ .w = w, .s = s, .b = b };
}

/// The head adapter: the Arc device, the resident head and the buffers the bridge reuses.
pub const Adapter = struct {
    gpa: std.mem.Allocator,
    driver: xpu.Driver,
    ctx: xpu.Context,
    stream: xpu.Stream,
    rt: xpu.rt.Runtime,
    mod: xpu.rt.Module,
    kern: xpu.rt.Kernel,
    w: xpu.rt.Buffer, // the head's words
    s: xpu.rt.Buffer, // the head's scales
    b: xpu.rt.Buffer, // the head's biases
    a: xpu.rt.Buffer, // the rows, fp16
    c: xpu.rt.Buffer, // the logits, fp16
    rows: cuda.HostBuffer, // the CUDA rows, bf16
    fx: cuda.HostBuffer, // the rows, fp16
    logits: cuda.HostBuffer, // the Arc's logits, fp16
    out: cuda.HostBuffer, // the logits for CUDA, bf16
    cuda: *const cuda.Driver, // the CUDA side the bridge reads and writes
    cuda_stream: *const cuda.Stream,
    vocab: usize,
    hidden: usize,
    nt: usize,
    groups: usize,
    rows_max: usize,

    fn drain(a: *Adapter) void {
        const rc = a.rt.drv.api.zeCommandListHostSynchronize(a.rt.list, 10_000_000_000);
        if (rc != 0) {
            std.log.err("Arc head queue did not drain (0x{x}); exiting without freeing buffers still in use", .{rc});
            std.process.exit(1);
        }
        a.rt.deinit();
    }

    /// The adapter on the Arc device with the head resident: the checkpoint's raw tensors packed and uploaded.
    pub fn init(gpa: std.mem.Allocator, ck: *core.Checkpoint, g: Geometry, d: *const cuda.Driver, cuda_stream: *const cuda.Stream, rows_max: usize) !*Adapter {
        if (arc_kernels.gemm.len == 0) return error.ArcHeadNotBuilt; // the build has no Intel GPU kernels
        if (g.hidden % 64 != 0) return error.UnsupportedArcHeadFormat; // the groups are 64 wide
        const t = try headTensors(ck, g);
        const nt = (g.vocab + tile - 1) / tile * tile; // the vocabulary padded to the GEMM tile
        const head_pack = try packHead(gpa, t, g.vocab, g.hidden, nt);
        defer gpa.free(head_pack.w);
        defer gpa.free(head_pack.s);
        defer gpa.free(head_pack.b);
        const a = try gpa.create(Adapter);
        errdefer gpa.destroy(a);
        a.* = .{
            .gpa = gpa,
            .driver = undefined,
            .ctx = undefined,
            .stream = undefined,
            .rt = undefined,
            .mod = undefined,
            .kern = undefined,
            .w = undefined,
            .s = undefined,
            .b = undefined,
            .a = undefined,
            .c = undefined,
            .rows = undefined,
            .fx = undefined,
            .logits = undefined,
            .out = undefined,
            .cuda = d,
            .cuda_stream = cuda_stream,
            .vocab = g.vocab,
            .hidden = g.hidden,
            .nt = nt,
            .groups = g.hidden / 64,
            .rows_max = rows_max,
        };
        // every errdefer follows the resource it frees: a failure never frees what was never made
        a.driver = try xpu.Driver.open();
        errdefer a.driver.close();
        a.ctx = try xpu.Context.init(&a.driver, null); // the Arc card, else the first device
        errdefer a.ctx.deinit();
        const props = a.ctx.gpu.props;
        const name = a.ctx.name();
        std.debug.print("arc head device: {s}, vendor 0x{x:0>4}, device 0x{x:0>4}, module input format {d}\n", .{ name, props.vendorId, props.deviceId, 1 });
        if (props.vendorId != 0x8086 or std.mem.indexOf(u8, name, "Arc") == null) return error.UnsupportedArcHeadDevice;
        a.stream = try xpu.Stream.init(&a.ctx);
        errdefer a.stream.deinit();
        a.rt = try xpu.rt.Runtime.init(&a.ctx, a.stream);
        var rt_drained = false;
        errdefer if (!rt_drained) a.rt.deinit();
        a.mod = try a.rt.moduleNative(&arc_kernels.gemm); // the offline image (ocloc's .bin): no JIT on the device
        errdefer a.mod.deinit();
        a.kern = try a.mod.kernel("w4_decode", .{ group_threads, 1, 1 });
        errdefer a.kern.deinit();
        a.w = try a.rt.alloc(head_pack.w.len * 4);
        errdefer a.w.free();
        a.s = try a.rt.alloc(head_pack.s.len * 2);
        errdefer a.s.free();
        a.b = try a.rt.alloc(head_pack.b.len * 2);
        errdefer a.b.free();
        const padded = (rows_max + row_tile - 1) / row_tile * row_tile;
        a.a = try a.rt.alloc(padded * g.hidden * 2);
        errdefer a.a.free();
        a.c = try a.rt.alloc(padded * nt * 2);
        errdefer a.c.free();
        errdefer {
            a.drain();
            rt_drained = true;
        }
        try a.rt.upload(a.w, std.mem.sliceAsBytes(head_pack.w));
        try a.rt.upload(a.s, std.mem.sliceAsBytes(head_pack.s));
        try a.rt.upload(a.b, std.mem.sliceAsBytes(head_pack.b));
        try a.rt.sync();
        // the bridge's host buffers: the CUDA rows in, the logits out
        a.rows = try cuda.HostBuffer.alloc(d, rows_max * g.hidden * 2);
        errdefer a.rows.free();
        a.fx = try cuda.HostBuffer.alloc(d, padded * g.hidden * 2);
        errdefer a.fx.free();
        a.logits = try cuda.HostBuffer.alloc(d, padded * nt * 2);
        errdefer a.logits.free();
        a.out = try cuda.HostBuffer.alloc(d, rows_max * g.vocab * 2);
        errdefer a.out.free();
        return a;
    }

    /// The device work is drained before a buffer goes: the queue must not read freed memory.
    pub fn deinit(a: *Adapter) void {
        a.drain();
        a.c.free();
        a.a.free();
        a.b.free();
        a.s.free();
        a.w.free();
        a.kern.deinit();
        a.mod.deinit();
        cuda.completion.drain(a.cuda, a.cuda_stream.*, "Arc head's CUDA copy did not drain; its pinned buffers stay alive");
        a.out.free();
        a.logits.free();
        a.fx.free();
        a.rows.free();
        a.stream.deinit();
        a.ctx.deinit();
        a.driver.close();
        a.gpa.destroy(a);
    }

    /// The head over the trunk's last rows: the CUDA rows to the Arc, the logits back to CUDA's own buffer.
    pub fn project(a: *Adapter, e: anytype, total: usize, rows: usize) !void {
        if (rows == 0 or rows > a.rows_max or rows > total) return error.ArcHeadRows;
        const padded = (rows + row_tile - 1) / row_tile * row_tile;
        // the trunk's rows are ready before they are read
        try e.stream.synchronize();
        const src = e.x + (total - rows) * a.hidden * 2;
        try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(a.rows.bytes.ptr, src, rows * a.hidden * 2, e.stream.handle), "cuMemcpyDtoHAsync");
        try e.stream.synchronize();
        // the rows as fp16: a row that does not fit fp16's range is refused, not rounded into an infinity
        const raw = std.mem.bytesAsSlice(u16, a.rows.bytes[0 .. rows * a.hidden * 2]);
        const fx = std.mem.bytesAsSlice(u16, a.fx.bytes[0 .. padded * a.hidden * 2]);
        @memset(fx, 0);
        for (raw, fx[0..raw.len]) |h, *o| o.* = bf16ToFp16(h) orelse return error.ArcHeadRowNotFinite;
        try a.rt.upload(a.a, a.fx.bytes[0 .. padded * a.hidden * 2]);
        try a.kern.setBuffer(0, a.a);
        try a.kern.setBuffer(1, a.w);
        try a.kern.setBuffer(2, a.s);
        try a.kern.setBuffer(3, a.b);
        try a.kern.setBuffer(4, a.c);
        try a.kern.setU32(5, @intCast(padded));
        try a.kern.setU32(6, @intCast(a.nt));
        try a.kern.setU32(7, @intCast(a.hidden));
        try a.kern.launch(.{ @intCast((padded + group_rows - 1) / group_rows), @intCast(a.nt / col_tile), 1 });
        try a.rt.sync();
        try a.rt.download(a.logits.bytes[0 .. padded * a.nt * 2], a.c);
        try a.rt.sync();
        // the logits as bf16 for CUDA: the real vocabulary's columns, not the tile's padding
        const got = std.mem.bytesAsSlice(u16, a.logits.bytes[0 .. padded * a.nt * 2]);
        const out = std.mem.bytesAsSlice(u16, a.out.bytes[0 .. rows * a.vocab * 2]);
        for (0..rows) |r| {
            for (0..a.vocab) |n| {
                const v = got[r * a.nt + n];
                // a logit that is not a number would go into CUDA's sampler: it is refused here
                if (!std.math.isFinite(@as(f32, @as(f16, @bitCast(v))))) return error.ArcHeadLogitNotFinite;
                out[r * a.vocab + n] = fp16ToBf16(v);
            }
        }
        // the logits land in CUDA's own buffer, on its stream: the sampler reads them there
        try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(e.logits, a.out.bytes.ptr, rows * a.vocab * 2, e.stream.handle), "cuMemcpyHtoDAsync");
        try e.stream.synchronize();
    }
};

test "the bridge's conversions refuse what fp16 cannot hold" {
    try std.testing.expectEqual(@as(?u16, null), bf16ToFp16(0x7f80)); // an infinity
    try std.testing.expectEqual(@as(?u16, null), bf16ToFp16(0x7fc0)); // a NaN
    try std.testing.expectEqual(@as(?u16, null), bf16ToFp16(0x7b00)); // a large finite value, above fp16's range
    try std.testing.expectEqual(@as(?u16, 0x0000), bf16ToFp16(0x0000)); // zero
    try std.testing.expectEqual(@as(?u16, 0x3c00), bf16ToFp16(0x3f80)); // one
    try std.testing.expectEqual(@as(?u16, 0x3800), bf16ToFp16(0x3f00)); // a half
    // the largest bf16 that fits fp16
    try std.testing.expect(bf16ToFp16(0x477f) != null);
}

test "the bridge's conversions keep the values and the infinities" {
    try std.testing.expectEqual(@as(u16, 0x3f80), fp16ToBf16(0x3c00)); // one
    try std.testing.expectEqual(@as(u16, 0x3f00), fp16ToBf16(0x3800)); // a half
    try std.testing.expectEqual(@as(u16, 0x7f80), fp16ToBf16(0x7c00)); // an infinity
    try std.testing.expectEqual(@as(u16, 0x7fc0), fp16ToBf16(0x7e00)); // a NaN
}

test "the packed word holds the codes the device layout reads back" {
    // the MLX word's codes 0..7 in its nibbles; the device word's nibble i is the code 2i
    const m: u32 = 0x76543210;
    const o = deviceWord(m);
    try std.testing.expectEqual(@as(u32, 0x0000), (o >> 0) & 0xF); // the code 0
    try std.testing.expectEqual(@as(u32, 0x0002), (o >> 4) & 0xF); // the code 2
    try std.testing.expectEqual(@as(u32, 0x0004), (o >> 8) & 0xF); // the code 4
    try std.testing.expectEqual(@as(u32, 0x0006), (o >> 12) & 0xF); // the code 6
    try std.testing.expectEqual(@as(u32, 0x0001), (o >> 16) & 0xF); // the code 1
    try std.testing.expectEqual(@as(u32, 0x0003), (o >> 20) & 0xF); // the code 3
    try std.testing.expectEqual(@as(u32, 0x0005), (o >> 24) & 0xF); // the code 5
    try std.testing.expectEqual(@as(u32, 0x0007), (o >> 28) & 0xF); // the code 7
}

test "the packed head's rows read back as the checkpoint's own rows" {
    // two rows of two groups over a hidden of 128: the words, the scales and the biases as stored
    const gpa = std.testing.allocator;
    const vocab = 2;
    const hidden = 128;
    const nt = 256;
    var words: [vocab * hidden / 8]u32 = undefined;
    var scales: [vocab * hidden / 64]u16 = undefined;
    var biases: [vocab * hidden / 64]u16 = undefined;
    for (&words, 0..) |*w, i| w.* = @intCast(i);
    for (&scales) |*s| s.* = 0x3f80; // one
    for (&biases) |*b| b.* = 0x0000;
    const head_pack = try packHead(gpa, .{ .weight = std.mem.sliceAsBytes(&words), .scales = std.mem.sliceAsBytes(&scales), .biases = std.mem.sliceAsBytes(&biases) }, vocab, hidden, nt);
    defer gpa.free(head_pack.w);
    defer gpa.free(head_pack.s);
    defer gpa.free(head_pack.b);
    // the tables are group-major, the padded columns are zero
    try std.testing.expectEqual(@as(u16, 0x3c00), head_pack.s[0]); // the first row's first group, as fp16
    try std.testing.expectEqual(@as(u16, 0x3c00), head_pack.s[1 * nt]);
    try std.testing.expectEqual(@as(u16, 0x0000), head_pack.s[vocab]);
    try std.testing.expectEqual(@as(u16, 0x0000), head_pack.b[vocab]);
    // the words are transposed: a row's word sits at its column, a word's rows at its stride
    try std.testing.expectEqual(deviceWord(words[0]), head_pack.w[0]); // the first row's first word
    try std.testing.expectEqual(deviceWord(words[hidden / 8]), head_pack.w[1]); // the second row's first word
    try std.testing.expectEqual(deviceWord(words[1]), head_pack.w[nt]); // the first row's second word
    try std.testing.expectEqual(@as(u32, 0), head_pack.w[nt - 1]); // the padded column
}

test "the packed head projects what the checkpoint's own rows project" {
    // the contract: the head's values, read back from the packed layout, are the checkpoint's values
    const gpa = std.testing.allocator;
    const vocab = 2;
    const hidden = 128;
    const groups = hidden / 64;
    const words_per_row = hidden / 8;
    const nt = 256;
    var words: [vocab * words_per_row]u32 = undefined;
    var scales: [vocab * groups]u16 = undefined;
    var biases: [vocab * groups]u16 = undefined;
    for (&words, 0..) |*w, i| w.* = @truncate(i *% 2654435761);
    for (&scales, 0..) |*s, i| s.* = if (i % 2 == 0) 0x3f81 else 0x3f80;
    for (&biases, 0..) |*b, i| b.* = if (i % 2 == 0) 0x3e81 else 0x3e80;
    const head_pack = try packHead(gpa, .{ .weight = std.mem.sliceAsBytes(&words), .scales = std.mem.sliceAsBytes(&scales), .biases = std.mem.sliceAsBytes(&biases) }, vocab, hidden, nt);
    defer gpa.free(head_pack.w);
    defer gpa.free(head_pack.s);
    defer gpa.free(head_pack.b);
    // the packed words back as the checkpoint's own rows
    var back: [vocab * words_per_row]u32 = undefined;
    for (0..vocab) |n| {
        for (0..words_per_row) |wi| back[n * words_per_row + wi] = mlxWord(head_pack.w[wi * nt + n]);
    }
    try std.testing.expectEqualSlices(u32, &words, &back);
    // the packed tables back as the checkpoint's own rows (fp16, the group-major layout undone)
    var ps: [vocab * groups]u16 = undefined;
    var pb: [vocab * groups]u16 = undefined;
    for (0..vocab) |n| for (0..groups) |g| {
        ps[n * groups + g] = head_pack.s[g * nt + n];
        pb[n * groups + g] = head_pack.b[g * nt + n];
    };
    // the fp64 reference of both: the checkpoint's rows and the packed ones
    const raw: core.quant.mlx.Host = .{
        .words = .{ .dtype = .i32, .rank = 2, .shape = .{ vocab, hidden / 8, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(&words) },
        .scales = .{ .dtype = .bf16, .rank = 2, .shape = .{ vocab, groups, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(&scales) },
        .biases = .{ .dtype = .bf16, .rank = 2, .shape = .{ vocab, groups, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(&biases) },
        .bits = 4,
        .group = 64,
    };
    const got: core.quant.mlx.Host = .{
        .words = .{ .dtype = .i32, .rank = 2, .shape = .{ vocab, hidden / 8, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(&back) },
        .scales = .{ .dtype = .f16, .rank = 2, .shape = .{ vocab, groups, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(&ps) },
        .biases = .{ .dtype = .f16, .rank = 2, .shape = .{ vocab, groups, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(&pb) },
        .bits = 4,
        .group = 64,
    };
    const want = try gpa.alloc(f64, vocab * hidden);
    defer gpa.free(want);
    const have = try gpa.alloc(f64, vocab * hidden);
    defer gpa.free(have);
    try core.quant.mlx.reference(raw, want);
    try core.quant.mlx.reference(got, have);
    for (want, have) |w, h| try std.testing.expectApproxEqAbs(w, h, 1e-6);
    // Compare real row dot products in f64: checkpoint BF16 dequantization versus the kernel's fp16-rounded weights.
    var worst_relative_dot_error: f64 = 0;
    for (0..vocab) |n| {
        var exact_dot: f64 = 0;
        var packed_dot: f64 = 0;
        for (0..hidden) |c| {
            const code: f64 = @floatFromInt((words[n * words_per_row + c / 8] >> @intCast(4 * (c % 8))) & 0xF);
            const scale: f64 = @floatCast(@as(f32, @bitCast(@as(u32, scales[n * groups + c / 64]) << 16)));
            const bias: f64 = @floatCast(@as(f32, @bitCast(@as(u32, biases[n * groups + c / 64]) << 16)));
            const exact_weight = code * scale + bias;
            const fp16_weight: f16 = @floatCast(exact_weight); // gemm.cl's one fp16 rounding after fma
            const activation: f64 = @as(f64, @floatFromInt(c % 13 + 1)) / 13;
            exact_dot += activation * exact_weight;
            packed_dot += activation * @as(f64, @floatCast(fp16_weight));
        }
        worst_relative_dot_error = @max(worst_relative_dot_error, @abs(packed_dot - exact_dot) / @max(@abs(exact_dot), 1e-30));
    }
    try std.testing.expect(worst_relative_dot_error > 0); // this fixture exercises actual fp16 dequantization rounding
    try std.testing.expect(worst_relative_dot_error <= 1e-2); // conservative 1% relative bound on row projections
}

test "the packed head refuses unsupported formats before loading weights" {
    var g: Geometry = .{ .hidden = 128, .intermediate = 128, .layers = 1, .vocab = 2, .k_heads = 1, .v_heads = 1, .q_heads = 1, .kv_heads = 1, .eps = 1e-6, .tied = false, .context = 64, .linear = @splat(true) };
    try validateFormat(g);
    g.exl3 = true;
    try std.testing.expectError(error.UnsupportedArcHeadFormat, validateFormat(g));
    g.exl3 = false;
    g.kinds.modelopt = true;
    try std.testing.expectError(error.UnsupportedArcHeadFormat, validateFormat(g));
    g.kinds.modelopt = false;
    g.bits.head = 8;
    try std.testing.expectError(error.UnsupportedArcHeadFormat, validateFormat(g));
    g.bits.head = 4;
    g.tied = true;
    g.bits.embed = 8;
    try std.testing.expectError(error.UnsupportedArcHeadFormat, validateFormat(g));
    g.bits.embed = 4;
    try validateFormat(g);
    g.hidden = 63;
    try std.testing.expectError(error.UnsupportedArcHeadFormat, validateFormat(g));
}

test "the bridge's learning mode refuses before touching device state" {
    const E = @import("cuda_engine.zig").Engine;
    var e: E = undefined;
    var learning: @import("cuda_learn_hooks.zig").Keep = undefined;
    var adapter: Adapter = undefined;
    e.learning = &learning;
    e.arc = &adapter;
    try std.testing.expectError(error.ArcHeadNoLearning, e.admit(&.{}, &.{}, .hidden));
}
