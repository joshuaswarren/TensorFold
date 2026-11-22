//! The dense MXFP8 projections the 128x64 prompt tile replaces: every (n, k) the FP4 checkpoint's dense layers call.
const std = @import("std");
const cuda = @import("cuda");
const q = cuda.qmmf;

pub const Projection = struct { name: []const u8, n: usize, k: usize, out: q.Out };

/// Dense MXFP8 projections in the NVIDIA FP4-Local checkpoint: qkvz+ab, attention qkv, outs, shared and HC glue.
pub const projections = [_]Projection{
    .{ .name = "GDN input", .n = 16480, .k = 2560, .out = .bf16 },
    .{ .name = "attention input", .n = 13952, .k = 2560, .out = .bf16 },
    .{ .name = "mixer output", .n = 2560, .k = 6144, .out = .f32 },
    .{ .name = "shared gate/up", .n = 640, .k = 2560, .out = .bf16 },
    .{ .name = "shared down", .n = 2560, .k = 640, .out = .f32 },
    .{ .name = "HC up", .n = 10240, .k = 320, .out = .bf16 },
    .{ .name = "HC down/inject", .n = 324, .k = 10240, .out = .f32 },
    .{ .name = "PLE key", .n = 10240, .k = 2560, .out = .bf16 },
    .{ .name = "PLE value", .n = 2560, .k = 2560, .out = .f32 },
    .{ .name = "attention qkv", .n = 8240, .k = 2560, .out = .bf16 },
    .{ .name = "indexer qkv", .n = 7296, .k = 2560, .out = .bf16 },
    .{ .name = "o_proj", .n = 1280, .k = 6144, .out = .bf16 },
    .{ .name = "out_proj", .n = 1280, .k = 640, .out = .bf16 },
    .{ .name = "final HC down", .n = 320, .k = 10240, .out = .f32 },
    .{ .name = "MTP hidden", .n = 2560, .k = 2560, .out = .f32 },
};

test "qmmf MXFP8 prompt scratch bounds every projection and the eligibility edges" {
    try std.testing.expectEqual(@as(u32, 82432), q.prompt_mx_shared);
    try std.testing.expectEqual(q.sharedBytes(.mxfp8, 128), q.prompt_mx_shared);
    try std.testing.expectEqual(@as(usize, 64), q.promptCols(.mxfp8));
    try std.testing.expectEqual(@as(usize, 128), q.promptCols(.bf16));
    try std.testing.expectEqual(@as(usize, 4 * 258), q.promptBlocks(512, 16480, q.promptCols(.mxfp8)));

    for (projections) |p| {
        const w = try q.pinned(cuda.mxfp8.weight(0, 0, p.n, p.k), q.splitK(p.n, p.k));
        try std.testing.expect(q.promptEligible(q.prompt_rows, w));
        try std.testing.expect(!q.promptEligible(q.prompt_rows - 1, w));
    }

    const hc_up = try q.pinned(cuda.mxfp8.weight(0, 0, 10240, 320), 1);
    try std.testing.expect(q.promptEligible(512, hc_up));
    const small_k = try q.pinned(cuda.mxfp8.weight(0, 0, 640, 320), 1);
    try std.testing.expect(!q.promptEligible(512, small_k));
    const narrow = try q.pinned(cuda.mxfp8.weight(0, 0, 48, 512), 1);
    try std.testing.expect(!q.promptEligible(512, narrow));
}
