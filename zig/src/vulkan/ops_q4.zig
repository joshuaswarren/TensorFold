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

// n rows per qmm chunk (0 if all of w fits one binding); k % 64 == 0; w bytes within max_range; w/scale starts aligned.
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

/// qmm_q4_pk flags define plain, contiguous, or compensated arithmetic; production fixes strided compensated mode.
pub var pk_variant: u32 = 2;

/// The device subgroup facts the decode kernel choice reads (the 1.1 properties' two fields).
pub const SubgroupCaps = struct { subgroup_size: u32, subgroup_ops: u32 };

/// The unit 11 probe's sgSubgroupOk gate: a 32-lane subgroup with arithmetic ops. Pure and host-tested.
pub fn sgDeviceOk(subgroup_size: u32, supported_ops: u32) bool {
    const arithmetic: u32 = 1 << 2; // VK_SUBGROUP_FEATURE_ARITHMETIC_BIT
    return subgroup_size == 32 and (supported_ops & arithmetic) != 0;
}

/// The decode matmul takes the subgroup qmv when one row fits the lane step and column width on an admitted device.
pub fn useSubgroupQmv(rows: u64, k: u64, n: u64, caps: SubgroupCaps) bool {
    return rows == 1 and k % 512 == 0 and n % 8 == 0 and sgDeviceOk(caps.subgroup_size, caps.subgroup_ops);
}

/// The SG decision minus the pipeline lookup: one row, shape fits, device admits, pipeline built; m past 1 keeps old.
pub fn qmmDecodeSgAdmits(m: u64, k: u64, n: u64, caps: SubgroupCaps, have_pipeline: bool) bool {
    return have_pipeline and useSubgroupQmv(m, k, n, caps);
}

/// The decode kernel the engine opened with: sg when init built the SG pipeline, else why it did not.
pub const QmmPath = enum { sg, old_env, old_device };

const Geometry = struct { m: u64, n: u64 };

/// One 64-lane workgroup per output; cap the dispatch and grid-stride the rest; refuse zero groups.
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

/// Q4 affine group-64 projection uses one parallel-K dispatch or validated chunk-and-scatter above binding limits.
pub fn project(o: *Ops, cmds: *Commands, out: View, x: View, w: View, scales: View, biases: View, k: u64, scratch: *Arena, timeout_ns: u64) Error!void {
    const g = try projectGeometry(out, x, w, scales, biases, k);
    const limits = o.dev.properties.limits;
    const lim = Limits{ .max_range = limits.maxStorageBufferRange, .alignment = limits.minStorageBufferOffsetAlignment };
    if (x.byteLen() > lim.max_range or out.byteLen() > lim.max_range) return error.TooLarge;
    const rows = try chunkRows(lim, k, g.n);
    if (rows == 0) {
        // The single-row dispatch takes the SG master when built and the shape fits; the old kernel otherwise.
        if (o.qmmDecodeSg(g.m, k, g.n)) |sg| {
            try sg.bind(&.{ x.bind(), w.bind(), scales.bind(), biases.bind(), out.bind() });
            var p = try q4Push(g.m, g.n, k);
            p.flags = pk_variant;
            try sg.run(cmds, std.mem.asBytes(&p), .{ try ops_mod.cast32((g.n + 7) / 8), 1, 1 }, timeout_ns);
            return;
        }
        const pipe = &o.qmm_q4_pk;
        try pipe.bind(&.{ x.bind(), w.bind(), scales.bind(), biases.bind(), out.bind() });
        var p = try q4Push(g.m, g.n, k);
        p.flags = pk_variant;
        const groups = try pkGroups(p.count, limits.maxComputeWorkGroupCount[0]);
        try pipe.run(cmds, std.mem.asBytes(&p), .{ groups, 1, 1 }, timeout_ns);
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
        const pipe = &o.qmm_q4_pk;
        try pipe.bind(&.{ x.bind(), w_chunk.bind(), s_chunk.bind(), b_chunk.bind(), block_chunk.bind() });
        var p = try q4Push(g.m, chunk, k);
        p.flags = pk_variant;
        const groups = try pkGroups(p.count, limits.maxComputeWorkGroupCount[0]);
        try pipe.run(cmds, std.mem.asBytes(&p), .{ groups, 1, 1 }, timeout_ns);
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

/// Candidate C: the merged gate|up projection and the ASH swiglu in ONE
/// dispatch of the -DFUSED_SWIGLU kernel: slots 0-1 compute gate rows
/// 4w..4w+3, slots 2-3 the matching up rows of the merged weight, and the
/// epilogue writes only the `act` activations (gate_rows elements). Refuses
/// when the SG decode gate does not admit the merged shape; the caller then
/// runs candidate B's path. The merged weight must already be laid out.
pub fn projectSwiglu(o: *Ops, cmds: *Commands, act: tensor.View, x: tensor.View, fused_w: tensor.View, fused_scales: tensor.View, fused_biases: tensor.View, k: u64, timeout_ns: u64) Error!void {
    const pipe = if (o.qmm_q4_sg_swiglu) |*sw| sw else return error.Invalid;
    // NOT projectGeometry: the weight carries gate+up rows (2 x act.elements)
    // against an activation-sized output, which the plain geometry refuses.
    if (act.kind != .bf16 or x.kind != .bf16 or fused_w.kind != .u32) return error.Invalid;
    if (k == 0 or k % group_size != 0) return error.Invalid;
    if (x.elements != k) return error.Invalid; // the epilogue is a one-row dispatch
    const per_row_w = k / 8;
    const per_row_p = k / 64;
    const n_merged = fused_w.elements / per_row_w;
    if (n_merged % 2 != 0) return error.Invalid;
    const gate_rows = n_merged / 2;
    if (act.elements != gate_rows) return error.Invalid;
    if (fused_scales.elements != n_merged * per_row_p or fused_biases.elements != n_merged * per_row_p) return error.Invalid;
    const limits = o.dev.properties.limits;
    const lim = Limits{ .max_range = limits.maxStorageBufferRange, .alignment = limits.minStorageBufferOffsetAlignment };
    if (x.byteLen() > lim.max_range or act.byteLen() > lim.max_range) return error.TooLarge;
    const rows = try chunkRows(lim, k, n_merged);
    if (rows != 0) return error.TooLarge; // the epilogue's one-workgroup mapping needs the whole weight
    if (!o.fuse_swiglu) return error.Invalid;
    try pipe.bind(&.{ x.bind(), fused_w.bind(), fused_scales.bind(), fused_biases.bind(), act.bind() });
    var p = try q4Push(1, n_merged, k);
    p.flags = pk_variant;
    try pipe.run(cmds, std.mem.asBytes(&p), .{ try ops_mod.cast32(gate_rows / 4), 1, 1 }, timeout_ns);
}

// Host token validation: every id must index a table row, checked before upload; out-of-range ids gather to zero.
pub fn checkIds(ids: []const u32, rows: u64) Error!void {
    for (ids) |id| if (id >= rows) return error.OutOfRange;
}

// Embedding rows/tokens or why not; scales/biases ride the 32-bit transport, so tokens*(k/64) fills whole words.
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

// Q4 embedding: rows to `scratch`, bf16 out; tables are single bindings — larger refused, never chunked.
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

test "useSubgroupQmv admits exactly the decode shapes the SG kernel serves on a 32-lane arithmetic subgroup" {
    const ok: SubgroupCaps = .{ .subgroup_size = 32, .subgroup_ops = 1 << 2 }; // VK_SUBGROUP_FEATURE_ARITHMETIC_BIT
    // the decode token's own shapes (n, k): every layer and the head
    try testing.expect(useSubgroupQmv(1, 6144, 2048, ok));
    try testing.expect(useSubgroupQmv(1, 2048, 6144, ok));
    try testing.expect(useSubgroupQmv(1, 2048, 2048, ok));
    try testing.expect(useSubgroupQmv(1, 2048, 4096, ok));
    try testing.expect(useSubgroupQmv(1, 2048, 512, ok));
    try testing.expect(useSubgroupQmv(1, 2048, 248320, ok));
    // prefill rows never take the SG path
    try testing.expect(!useSubgroupQmv(2, 2048, 2048, ok));
    try testing.expect(!useSubgroupQmv(16, 2048, 2048, ok));
    // the kernel's 512-element lane step needs whole steps; its 8-wide columns need whole columns
    try testing.expect(!useSubgroupQmv(1, 2000, 2048, ok));
    try testing.expect(!useSubgroupQmv(1, 2048, 7, ok));
    // the subgroupAdd tree is written for a 32-lane subgroup with the arithmetic operations
    try testing.expect(!useSubgroupQmv(1, 2048, 2048, .{ .subgroup_size = 16, .subgroup_ops = ok.subgroup_ops }));
    try testing.expect(!useSubgroupQmv(1, 2048, 2048, .{ .subgroup_size = 64, .subgroup_ops = ok.subgroup_ops }));
    try testing.expect(!useSubgroupQmv(1, 2048, 2048, .{ .subgroup_size = 32, .subgroup_ops = 1 })); // basic only
}

test "qmmDecodeSgAdmits refuses every multi-row batch even with the pipeline built" {
    const ok: SubgroupCaps = .{ .subgroup_size = 32, .subgroup_ops = 1 << 2 }; // VK_SUBGROUP_FEATURE_ARITHMETIC_BIT
    // Batched prefill hits the single-binding branch with several rows; the kernel computes row 0 only.
    try testing.expect(!qmmDecodeSgAdmits(2, 2048, 2048, ok, true));
    try testing.expect(!qmmDecodeSgAdmits(16, 6144, 2048, ok, true));
    try testing.expect(!qmmDecodeSgAdmits(319, 2048, 2048, ok, true));
    // one row with the pipeline built and the shape fine is the decode dispatch
    try testing.expect(qmmDecodeSgAdmits(1, 2048, 2048, ok, true));
    // a built pipeline cannot rescue a shape or device the kernel does not serve
    try testing.expect(!qmmDecodeSgAdmits(1, 2000, 2048, ok, true));
    try testing.expect(!qmmDecodeSgAdmits(1, 2048, 2048, .{ .subgroup_size = 64, .subgroup_ops = ok.subgroup_ops }, true));
    try testing.expect(!qmmDecodeSgAdmits(1, 2048, 2048, ok, false));
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
