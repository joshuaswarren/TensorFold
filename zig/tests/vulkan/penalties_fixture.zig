//! Expected BF16 words as fixed literals, hand-derived from the CUDA tf_penalize_rows formula, never from host helpers.
const std = @import("std");

/// One penalize workgroup covers the row; the count is even for bf16 view word padding.
pub const count: usize = 16;
/// Data window offset in elements: 128 bf16 = 256 bytes, the largest minStorageBufferOffsetAlignment Vulkan allows.
pub const at: usize = 128;
pub const tail: usize = 16;
pub const poison16: u16 = 0xffff;

pub const Entry = struct { tok: u32, sub: f32 };
pub const Want = struct { tok: u32, word: u16 };

/// What the caller must observe after the penalize dispatch: a greedy winner, or a NonFinite refusal.
pub const Outcome = union(enum) {
    winner: u32,
    non_finite,
};

pub const Row = struct {
    name: []const u8,
    /// The data window's planted words, token index -> bf16 bits.
    planted: []const Want,
    entries: []const Entry,
    wants: []const Want,
    outcome: Outcome,

    /// The whole poisoned shell: guards, data window (planted over 0x0000), poison tail.
    pub fn shell(r: Row, a: std.mem.Allocator) ![]u16 {
        const bits = try a.alloc(u16, at + count + tail);
        @memset(bits, 0);
        for (bits[at + count ..]) |*b| b.* = poison16;
        for (r.planted) |p| bits[at + p.tok] = p.word;
        return bits;
    }
};

// RNE words by hand: 1.0 plus 1/256 ties to even (0x3f80), plus 3/256 ties away (0x3f82); the negative pair mirrors it.
pub const rne = Row{
    .name = "rne-ties",
    .planted = &.{
        .{ .tok = 0, .word = 0x3f80 },
        .{ .tok = 1, .word = 0x3f80 },
        .{ .tok = 2, .word = 0xbf80 },
        .{ .tok = 3, .word = 0xbf80 },
    },
    .entries = &.{
        .{ .tok = 0, .sub = -0.00390625 },
        .{ .tok = 1, .sub = -0.01171875 },
        .{ .tok = 2, .sub = 0.00390625 },
        .{ .tok = 3, .sub = 0.01171875 },
    },
    .wants = &.{
        .{ .tok = 0, .word = 0x3f80 },
        .{ .tok = 1, .word = 0x3f82 },
        .{ .tok = 2, .word = 0xbf80 },
        .{ .tok = 3, .word = 0xbf82 },
    },
    .outcome = .{ .winner = 1 }, // token 1 rounds away to 0x3f82, above token 0's 1.0: no tie
};

// CUDA subnormal edge: 0x0080 minus fp32 2^-133 is exact only as a subnormal and rounds to 0x007f; flushing fails.
pub const subnormal = Row{
    .name = "subnormal-cancel",
    .planted = &.{.{ .tok = 5, .word = 0x0080 }},
    .entries = &.{.{ .tok = 5, .sub = std.math.pow(f32, 2, -133) }},
    .wants = &.{.{ .tok = 5, .word = 0x007f }},
    .outcome = .{ .winner = 5 },
};

// Tokens 0 and 1 at 4.0 each lose exactly 2.0, so untouched token 2 at 3.0 becomes the greedy winner.
pub const winner = Row{
    .name = "adjacent-winner",
    .planted = &.{
        .{ .tok = 0, .word = 0x4080 },
        .{ .tok = 1, .word = 0x4080 },
        .{ .tok = 2, .word = 0x4040 },
    },
    .entries = &.{
        .{ .tok = 0, .sub = 2.0 },
        .{ .tok = 1, .sub = 2.0 },
    },
    .wants = &.{
        .{ .tok = 0, .word = 0x4000 },
        .{ .tok = 1, .word = 0x4000 },
        .{ .tok = 2, .word = 0x4040 },
    },
    .outcome = .{ .winner = 2 },
};

// A planted NaN the penalties never touch refuses the draw, and 1.0 minus 1.0 is POSITIVE zero (0x0000).
pub const finite = Row{
    .name = "finite-guard-signed-zero",
    .planted = &.{
        .{ .tok = 4, .word = 0x3f80 },
        .{ .tok = 15, .word = 0x7fc0 },
    },
    .entries = &.{.{ .tok = 4, .sub = 1.0 }},
    .wants = &.{.{ .tok = 4, .word = 0x0000 }},
    .outcome = .non_finite,
};

// Token 1 (0x3f81) loses 1/128 and ties token 0 at 1.0 (0x3f80); the lowest index wins, token 0.
pub const tie = Row{
    .name = "tie-lowest-index",
    .planted = &.{
        .{ .tok = 0, .word = 0x3f80 },
        .{ .tok = 1, .word = 0x3f81 },
    },
    .entries = &.{.{ .tok = 1, .sub = 0.0078125 }},
    .wants = &.{.{ .tok = 1, .word = 0x3f80 }},
    .outcome = .{ .winner = 0 },
};

pub const rows = [_]Row{ rne, subnormal, winner, finite, tie };

test "every row stages unique tokens inside the window and wants each planted token once" {
    for (&rows) |r| {
        for (r.entries) |e| try std.testing.expect(e.tok < count);
        for (r.planted) |p| try std.testing.expect(p.tok < count);
        for (r.wants) |w| try std.testing.expect(w.tok < count);
        for (r.entries, 0..) |e, i| for (r.entries[i + 1 ..]) |o| {
            try std.testing.expect(e.tok != o.tok); // unique token per entry: no halfword races
        };
        for (r.wants) |w| {
            var planted = false;
            for (r.planted) |p| planted = planted or p.tok == w.tok;
            try std.testing.expect(planted); // every expectation names a word the GPU actually changed
        }
    }
}
