//! The composed stage 1 closures: Q4 affine projection and Q4-table embedding over gathered, dequantized rows.
const std = @import("std");
const commands = @import("commands.zig");
const dispatch = @import("dispatch.zig");
const params = @import("stage1_params");
const tensor = @import("tensor.zig");
const arena_mod = @import("arena.zig");
const ops_mod = @import("ops.zig");
const Commands = commands.Commands;
const Arena = arena_mod.Arena;
const Ops = ops_mod.Ops;
const View = tensor.View;
const Error = dispatch.Error;

/// Stage 1 serves Q4, group 64, transposed weights only; wider coverage is later work, not a silent pass.
pub const group_size: u64 = 64;
pub const bits: u64 = 4;

/// The device numbers the chunk planner shapes itself around.
pub const Limits = struct { max_range: u64, alignment: u64 };

/// Rows of n per qmm chunk, or 0 when the whole w view fits one binding. Chunk row counts are
/// multiples of the least step keeping every chunk's w and scale start byte aligned, and each chunk's
/// w bytes stay within max_range; k must be a multiple of 64. Pure and host-tested.
pub fn chunkRows(l: Limits, k: u64, n: u64) Error!u64 {
    const row_w = k / 2; // k/8 u32 words per row
    const row_s = k / 32; // k/64 bf16 scales (or biases) per row
    if (n *| row_w <= l.max_range) return 0;
    const step_w = l.alignment / std.math.gcd(l.alignment, row_w);
    const step_s = l.alignment / std.math.gcd(l.alignment, row_s);
    const step = std.math.mul(u64, step_w / std.math.gcd(step_w, step_s), step_s) catch return error.TooLarge;
    const group_bytes = std.math.mul(u64, step, row_w) catch return error.TooLarge;
    if (group_bytes > l.max_range) return error.TooLarge;
    return (l.max_range / group_bytes) * step;
}

/// qmm_q4_pk arithmetic, read from the push flags: 0 = plain f32 partials (the frozen v0 candidate),
/// bit 0 = contiguous lane ranges, bit 1 = Neumaier-compensated sums with a TwoSum tree. The CANDIDATE
/// build is fixed at 2 (strided lanes, compensated); nothing selects it at run time.
pub var pk_variant: u32 = 2;

/// Which kernel serves a projection: the compensated parallel-K kernel for the one-row decode
/// projections it was built for, the serial qmm for a prefill chunk's rows. Pure and host-tested.
pub const Route = enum { parallel_k, serial };

pub fn route(rows: u64) Route {
    return if (rows == 1) .parallel_k else .serial;
}

const Geometry = struct { m: u64, n: u64 };

/// Workgroups for one qmm_q4_pk dispatch: one 64-lane workgroup per output, capped at the
/// device's maxComputeWorkGroupCount[0] (the kernel grid-strides the rest, so the cap never
/// drops work). Zero outputs cannot happen past the geometry checks but is refused, not
/// dispatched as zero groups. Pure and host-tested.
pub fn pkGroups(outputs: u64, max_groups: u32) Error!u32 {
    if (outputs == 0) return error.Invalid;
    const groups = std.math.cast(u32, outputs) orelse return error.TooLarge;
    return @min(groups, max_groups);
}

/// The projection's m and n from the views and k, or the named reason it cannot run; pure.
fn projectGeometry(out: View, x: View, w: View, scales: View, biases: View, k: u64) Error!Geometry {
    if (out.kind != .bf16 or x.kind != .bf16 or w.kind != .u32 or scales.kind != .bf16 or biases.kind != .bf16) return error.Invalid;
    if (k < group_size or k % group_size != 0) return error.Invalid;
    if (x.elements == 0 or x.elements % k != 0) return error.Invalid;
    const m = x.elements / k;
    if (out.elements == 0 or out.elements % m != 0) return error.Invalid;
    const n = out.elements / m;
    const per_row_w = k / 8;
    const per_row_p = k / 64;
    const want_w = std.math.mul(u64, n, per_row_w) catch return error.Invalid;
    const want_p = std.math.mul(u64, n, per_row_p) catch return error.Invalid;
    if (w.elements != want_w or scales.elements != want_p or biases.elements != want_p) return error.Invalid;
    return .{ .m = m, .n = n };
}

fn q4Push(m: u64, n: u64, k: u64) Error!params.Params {
    const count = try ops_mod.cast32(m * n);
    return .{
        .count = count,
        .operation = @intCast(bits),
        .reduce_size = @intCast(group_size),
        .output_size = count,
        .matrix_m = try ops_mod.cast32(m),
        .matrix_n = try ops_mod.cast32(n),
        .matrix_k = try ops_mod.cast32(k),
        .shape = .{ 1, 0, 0, 0 }, // one x batch, shared w and per-row params
    };
}

/// Q4 affine projection out[m, n] = x[m, k] . dequant(w, scales, biases)[n, k], group 64, transposed.
/// The one-row decode projections take the qmm_q4_pk kernel (one 64-lane workgroup per output,
/// parallel over K); a prefill chunk's rows take the serial qmm (one thread per output), the
/// kernel they ran on before the parallel-K kernel landed. Either way: one dispatch when the w
/// view fits maxStorageBufferRange; otherwise n-row chunks write packed blocks to `scratch` and
/// a copy_general scatter places each block's columns. Every range and shape is validated before
/// the first dispatch; x and out must each fit one binding.
pub fn project(o: *Ops, cmds: *Commands, out: View, x: View, w: View, scales: View, biases: View, k: u64, scratch: *Arena, timeout_ns: u64) Error!void {
    return projectAs(o, cmds, out, x, w, scales, biases, k, scratch, timeout_ns, null);
}

/// The parallel-K kernel at any row count: the engine's one-row routing, and the stage 1 probe's
/// coverage of the kernel itself. project() routes by row count instead (see route).
pub fn projectParallelK(o: *Ops, cmds: *Commands, out: View, x: View, w: View, scales: View, biases: View, k: u64, scratch: *Arena, timeout_ns: u64) Error!void {
    return projectAs(o, cmds, out, x, w, scales, biases, k, scratch, timeout_ns, true);
}

fn projectAs(o: *Ops, cmds: *Commands, out: View, x: View, w: View, scales: View, biases: View, k: u64, scratch: *Arena, timeout_ns: u64, force_pk: ?bool) Error!void {
    const g = try projectGeometry(out, x, w, scales, biases, k);
    const limits = o.dev.properties.limits;
    const lim = Limits{ .max_range = limits.maxStorageBufferRange, .alignment = limits.minStorageBufferOffsetAlignment };
    if (x.byteLen() > lim.max_range or out.byteLen() > lim.max_range) return error.TooLarge;
    const use_pk = force_pk orelse (route(g.m) == .parallel_k);
    const rows = try chunkRows(lim, k, g.n);
    if (rows == 0) {
        const pipe = if (use_pk) &o.qmm_q4_pk else &o.qmm_bf16;
        try pipe.bind(&.{ x.bind(), w.bind(), scales.bind(), biases.bind(), out.bind() });
        var p = try q4Push(g.m, g.n, k);
        const groups: [3]u32 = if (use_pk)
            .{ try pkGroups(p.count, limits.maxComputeWorkGroupCount[0]), 1, 1 }
        else
            ops_mod.dispatchGroups(p.count, pipe.local[0]);
        if (use_pk) p.flags = pk_variant;
        try pipe.run(cmds, std.mem.asBytes(&p), groups, timeout_ns);
        return;
    }
    const block = try scratch.alloc(.bf16, g.m * rows);
    if (block.byteLen() > lim.max_range) return error.TooLarge;
    const row_words = k / 8;
    const row_groups = k / 64;
    var first: u64 = 0;
    while (first < g.n) {
        const chunk = @min(rows, g.n - first);
        // chunkRows made every chunk start an aligned byte of the w and scale streams.
        const w_chunk = try w.slice(first * row_words, chunk * row_words);
        const s_chunk = try scales.slice(first * row_groups, chunk * row_groups);
        const b_chunk = try biases.slice(first * row_groups, chunk * row_groups);
        const block_chunk = try block.slice(0, g.m * chunk);
        const pipe = if (use_pk) &o.qmm_q4_pk else &o.qmm_bf16;
        try pipe.bind(&.{ x.bind(), w_chunk.bind(), s_chunk.bind(), b_chunk.bind(), block_chunk.bind() });
        var p = try q4Push(g.m, chunk, k);
        const groups: [3]u32 = if (use_pk)
            .{ try pkGroups(p.count, limits.maxComputeWorkGroupCount[0]), 1, 1 }
        else
            ops_mod.dispatchGroups(p.count, pipe.local[0]);
        if (use_pk) p.flags = pk_variant;
        try pipe.run(cmds, std.mem.asBytes(&p), groups, timeout_ns);
        // Scatter the packed block into the full row-major output; raw u16 columns keep every RNE byte.
        const scatter = &o.copy_u16;
        try scatter.bind(&.{ block_chunk.bind(), o.dummyBinding(), out.bind(), o.dummyBinding() });
        var cp: params.Params = .{
            .count = try ops_mod.cast32(g.m * chunk),
            .dims = 2,
            .shape = .{ try ops_mod.cast32(g.m), try ops_mod.cast32(chunk), 0, 0 },
            .in_strides = .{ try ops_mod.cast32(chunk), 1, 0, 0 },
            .out_strides = .{ try ops_mod.cast32(g.n), 1, 0, 0 },
            .output_offset = try ops_mod.cast32(first),
        };
        try scatter.run(cmds, std.mem.asBytes(&cp), ops_mod.dispatchGroups(cp.count, scatter.local[0]), timeout_ns);
        first += chunk;
    }
}

/// Host-side token validation: every id must index a table row; call before the ids are uploaded.
/// Device-side out-of-range ids gather to zero (the shader's documented read), never a wrong row.
pub fn checkIds(ids: []const u32, rows: u64) Error!void {
    for (ids) |id| if (id >= rows) return error.OutOfRange;
}

/// The embedding's rows and tokens from the views and k, or the named reason it cannot run; pure.
/// The gathered scale and bias rows ride the 32-bit transport, so tokens * (k / 64) must fill whole
/// words: k 64 with an odd token count is refused here (a 2B-size k gives even groups for any tokens).
fn embeddingGeometry(out: View, ids: View, table_w: View, table_s: View, table_b: View, k: u64) Error!struct { rows: u64, tokens: u64 } {
    if (out.kind != .bf16 or ids.kind != .u32 or table_w.kind != .u32 or table_s.kind != .bf16 or table_b.kind != .bf16) return error.Invalid;
    if (k < group_size or k % group_size != 0 or ids.elements == 0) return error.Invalid;
    const tokens = ids.elements;
    const want_out = std.math.mul(u64, tokens, k) catch return error.Invalid;
    if (out.elements != want_out) return error.Invalid;
    const row_words = k / 8;
    const row_groups = k / 64;
    if ((tokens * row_groups) % 2 != 0) return error.Invalid;
    if (table_w.elements == 0 or table_w.elements % row_words != 0) return error.Invalid;
    const rows = table_w.elements / row_words;
    const want_p = std.math.mul(u64, rows, row_groups) catch return error.Invalid;
    if (table_s.elements != want_p or table_b.elements != want_p) return error.Invalid;
    return .{ .rows = rows, .tokens = tokens };
}

/// Q4-table embedding: gathers each id's packed weight row and its scale and bias rows into `scratch`,
/// then dequantizes to bf16 out[tokens, k]. The three table views are single bindings, so each must fit
/// maxStorageBufferRange whole; anything larger is refused before any dispatch, never silently chunked.
pub fn embedding(o: *Ops, cmds: *Commands, out: View, ids: View, table_w: View, table_s: View, table_b: View, k: u64, scratch: *Arena, timeout_ns: u64) Error!void {
    const g = try embeddingGeometry(out, ids, table_w, table_s, table_b, k);
    const range = o.dev.properties.limits.maxStorageBufferRange;
    if (table_w.byteLen() > range or table_s.byteLen() > range or table_b.byteLen() > range) return error.TooLarge;
    const row_words = k / 8;
    const row_groups = k / 64;
    // Allocations happen before any dispatch, so a full scratch is a clean refusal, not a torn result.
    const gathered_w = try scratch.alloc(.u32, g.tokens * row_words);
    const gathered_s = try scratch.alloc(.bf16, g.tokens * row_groups);
    const gathered_b = try scratch.alloc(.bf16, g.tokens * row_groups);
    try o.gather(cmds, gathered_w, table_w, ids, g.rows, timeout_ns);
    try o.gather(cmds, gathered_s, table_s, ids, g.rows, timeout_ns);
    try o.gather(cmds, gathered_b, table_b, ids, g.rows, timeout_ns);
    try o.dequant(cmds, out, gathered_w, gathered_s, gathered_b, k, group_size, timeout_ns);
}

const testing = std.testing;

test "chunkRows returns zero when one binding fits and aligned groups when it does not" {
    const fits: Limits = .{ .max_range = 1 << 30, .alignment = 256 };
    try testing.expectEqual(@as(u64, 0), try chunkRows(fits, 4096, 1000));
    const tight: Limits = .{ .max_range = 1 << 27, .alignment = 256 };
    try testing.expectEqual(@as(u64, 65536), try chunkRows(tight, 4096, 100_000)); // 2-row steps of 4096 bytes fill 128 MiB
    const small: Limits = .{ .max_range = 1 << 16, .alignment = 256 };
    const rows = try chunkRows(small, 64, 1 << 20); // row 32B, scale row 2B: 128-row steps of 4096 bytes, 16 steps fill 64 KiB
    try testing.expectEqual(@as(u64, 2048), rows);
    try testing.expectEqual(@as(u64, 0), rows % 128);
    try testing.expect(rows * 32 <= small.max_range);
}

test "chunkRows refuses an aligned group larger than the range" {
    const absurd: Limits = .{ .max_range = 1 << 12, .alignment = 1 << 12 };
    try testing.expectError(error.TooLarge, chunkRows(absurd, 64, 1 << 20));
}

test "chunkRows plans the production head: one binding when the range allows, aligned chunks below" {
    const wide: Limits = .{ .max_range = 1 << 32, .alignment = 256 };
    try testing.expectEqual(@as(u64, 0), try chunkRows(wide, 2048, 248320));
    const mobile: Limits = .{ .max_range = 128 << 20, .alignment = 256 };
    const rows = try chunkRows(mobile, 2048, 248320); // 242.6 MiB of head w never fits one 128 MiB binding
    try testing.expect(rows > 0 and rows < 248320);
    try testing.expectEqual(@as(u64, 0), rows % 2); // whole aligned row groups per chunk
    try testing.expect(rows * 1024 <= mobile.max_range); // every chunk's w bytes stay in range
    try testing.expectEqual(@as(u64, 0), try chunkRows(mobile, 6144, 2048)); // the down proj always fits
}

test "project geometry derives m and n and refuses torn or wrong-kind views" {
    const dev: @import("device.zig").Device = undefined;
    const buf = @import("memory.zig").Buffer{ .dev = &dev, .handle = null, .memory = null, .len = 1 << 20, .kind = .device, .mapped = null, .coherent = false };
    const x = try View.init(&buf, .bf16, 0, 3 * 128);
    const out = try View.init(&buf, .bf16, 1024, 3 * 100);
    const w = try View.init(&buf, .u32, 2048, 100 * 16);
    const scales = try View.init(&buf, .bf16, 4096, 100 * 2);
    const biases = try View.init(&buf, .bf16, 4096, 100 * 2);
    const g = try projectGeometry(out, x, w, scales, biases, 128);
    try testing.expectEqual(@as(u64, 3), g.m);
    try testing.expectEqual(@as(u64, 100), g.n);
    try testing.expectError(error.Invalid, projectGeometry(out, x, w, scales, biases, 100));
    try testing.expectError(error.Invalid, projectGeometry(out, x, w, scales, biases, 256));
    const f32x = try View.init(&buf, .f32, 1024, 3 * 128);
    try testing.expectError(error.Invalid, projectGeometry(out, f32x, w, scales, biases, 128));
    const torn = try View.init(&buf, .u32, 2048, 100 * 16 + 1);
    try testing.expectError(error.Invalid, projectGeometry(out, x, torn, scales, biases, 128));
}

test "embedding geometry derives rows and tokens and refuses torn tables" {
    const dev: @import("device.zig").Device = undefined;
    const buf = @import("memory.zig").Buffer{ .dev = &dev, .handle = null, .memory = null, .len = 1 << 20, .kind = .device, .mapped = null, .coherent = false };
    const ids = try View.init(&buf, .u32, 0, 5);
    const out = try View.init(&buf, .bf16, 64, 5 * 128);
    const w = try View.init(&buf, .u32, 2048, 100 * 16);
    const scales = try View.init(&buf, .bf16, 4096, 100 * 2);
    const biases = try View.init(&buf, .bf16, 4096, 100 * 2);
    const g = try embeddingGeometry(out, ids, w, scales, biases, 128);
    try testing.expectEqual(@as(u64, 100), g.rows);
    try testing.expectEqual(@as(u64, 5), g.tokens);
    try testing.expectError(error.Invalid, embeddingGeometry(out, ids, w, scales, biases, 64)); // out rows torn
    const torn_s = try View.init(&buf, .bf16, 4096, 99 * 2);
    try testing.expectError(error.Invalid, embeddingGeometry(out, ids, w, torn_s, biases, 128));
    // k 64 gathers one scale a token: five tokens leave a ragged bf16 pair, six fill whole words.
    const k64_out5 = try View.init(&buf, .bf16, 8192, 5 * 64);
    const ids6 = try View.init(&buf, .u32, 0, 6);
    const k64_out6 = try View.init(&buf, .bf16, 8192, 6 * 64);
    const w64 = try View.init(&buf, .u32, 8704, 100 * 8);
    const s64 = try View.init(&buf, .bf16, 8704, 100);
    try testing.expectError(error.Invalid, embeddingGeometry(k64_out5, ids, w64, s64, s64, 64));
    const g64 = try embeddingGeometry(k64_out6, ids6, w64, s64, s64, 64);
    try testing.expectEqual(@as(u64, 6), g64.tokens);
    try testing.expectEqual(@as(u64, 100), g64.rows);
}

test "checkIds refuses any id past the table" {
    try checkIds(&.{ 0, 3, 99 }, 100);
    try checkIds(&.{}, 1);
    try testing.expectError(error.OutOfRange, checkIds(&.{ 0, 100 }, 100));
}

test "pkGroups plans one workgroup per output and clamps at the device limit" {
    const roomy: u32 = 0x7fff_ffff; // above every output count below: no clamp
    const smallest: u32 = 65535; // the smallest maxComputeWorkGroupCount[0] Vulkan requires
    try testing.expectEqual(@as(u32, 1), try pkGroups(1, roomy)); // M=1, N=1
    try testing.expectEqual(@as(u32, 16), try pkGroups(16, roomy)); // M=1, N=16: a group each, the lanes split K
    try testing.expectEqual(@as(u32, 512), try pkGroups(32 * 16, roomy)); // M=32, N=16
    try testing.expectEqual(@as(u32, 248320), try pkGroups(248320, roomy)); // M=1 at the head's N
    try testing.expectEqual(@as(u32, 7946240), try pkGroups(32 * 248320, roomy)); // M=32 prefill at the head
    try testing.expectEqual(@as(u32, 65535), try pkGroups(248320, smallest)); // the cap: the kernel grid-strides the rest
    try testing.expectEqual(@as(u32, 1000), try pkGroups(1 << 20, 1000));
    try testing.expectEqual(@as(u32, 1), try pkGroups(1 << 20, 1));
    try testing.expectError(error.Invalid, pkGroups(0, roomy));
    try testing.expectError(error.TooLarge, pkGroups(1 << 32, roomy));
}

test "the routing: one row to the parallel-K kernel, a prefill chunk's rows to the serial qmm" {
    try testing.expectEqual(Route.parallel_k, route(1)); // the decode projections, the head included
    for ([_]u64{ 2, 3, 31, 32 }) |rows| try testing.expectEqual(Route.serial, route(rows));
}
