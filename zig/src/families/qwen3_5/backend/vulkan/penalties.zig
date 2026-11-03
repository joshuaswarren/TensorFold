//! Presence and frequency penalties for the Vulkan engine: the host stages reply-only counts, the GPU applies them.
const std = @import("std");
const lanes = @import("lanes");
const pinned = @import("../../config.zig");
const state = @import("state.zig");
const vk = @import("vulkan");

const Allocator = std.mem.Allocator;
const Buffer = vk.Buffer;
const Commands = vk.commands.Commands;
const View = vk.tensor.View;

/// Entries one decode can stage at most: every committed token plus the pending one (ceiling in state.zig).
pub const max_entries: u32 = @intCast(state.max_capacity + 1);

/// One staging call's outcome; the engine apply surface adds only dispatch errors.
pub const StageError = error{ OutOfMemory, NonFinite, InvalidQwenPosition, InvalidQwenToken };

/// Byte offset of the amounts in a table of `cap` entries: the toks' bytes rounded up to 256, aligned for the binding.
pub fn amountsAt(cap: u32) u64 {
    return (@as(u64, cap) * 4 + 255) / 256 * 256;
}

pub const Penalties = struct {
    gpa: Allocator,
    dev: *const vk.Device,
    /// The penalize pipeline: the logits row (binding 0) and the entry table (bindings 1 and 2, read-only).
    pipe: vk.dispatch.Pipeline = undefined,
    /// The staged entries for the next apply: one per distinct reply token.
    toks: std.ArrayList(u32) = .empty,
    subs: std.ArrayList(f32) = .empty,
    counts: std.AutoArrayHashMapUnmanaged(u32, u32) = .empty,
    /// The device entry table: toks u32[cap] at 0, amounts f32[cap] at `amountsAt(cap)`; null until first use.
    entries: ?Buffer = null,
    cap: u32 = 0,

    /// Frees the staged lists and the device entry table; the pipeline's deinit is the owner's.
    pub fn deinit(p: *Penalties) void {
        p.toks.deinit(p.gpa);
        p.subs.deinit(p.gpa);
        p.counts.deinit(p.gpa);
        if (p.entries) |*b| b.deinit();
    }

    /// The staged entries for the next decode: the reply's tokens before `cache_len`, then `next` (counted once).
    pub fn stage(p: *Penalties, stream: *const lanes.Stream, cache_len: usize, next: u32, pen: lanes.Penalties) StageError!void {
        if (!pen.active()) return;
        if (!std.math.isFinite(pen.presence) or !std.math.isFinite(pen.frequency)) return error.NonFinite;
        const context = stream.context.items;
        if (cache_len < stream.prompt_len or cache_len > context.len) return error.InvalidQwenPosition;
        if (next >= pinned.vocab) return error.InvalidQwenToken;
        for (context[stream.prompt_len..cache_len]) |t| if (t >= pinned.vocab) return error.InvalidQwenToken;
        p.clear(); // refused calls above leave the previous staging untouched
        for (context[stream.prompt_len..cache_len]) |t| try p.bump(t);
        try p.bump(next);
        try p.toks.ensureUnusedCapacity(p.gpa, p.counts.count());
        try p.subs.ensureUnusedCapacity(p.gpa, p.counts.count());
        for (p.counts.keys(), p.counts.values()) |t, c| {
            p.toks.appendAssumeCapacity(t);
            p.subs.appendAssumeCapacity(pen.of(c));
        }
    }

    /// Nothing staged left; the device buffer stays for the next apply.
    pub fn clear(p: *Penalties) void {
        p.toks.clearRetainingCapacity();
        p.subs.clearRetainingCapacity();
        p.counts.clearRetainingCapacity();
    }

    /// How many entries the next apply would take.
    pub fn staged(p: *const Penalties) u32 {
        return @intCast(p.toks.items.len);
    }

    fn bump(p: *Penalties, t: u32) StageError!void {
        const gop = try p.counts.getOrPut(p.gpa, t);
        if (!gop.found_existing) gop.value_ptr.* = 0;
        gop.value_ptr.* += 1;
    }

    /// The staged entries into the device table: toks at 0, amounts after, each its own staging slice.
    fn uploadEntries(p: *Penalties, cmds: *Commands, staging: Buffer, n: u32, timeout_ns: u64) !void {
        const e = p.entries.?;
        try vk.upload(cmds, e, 0, staging, std.mem.sliceAsBytes(p.toks.items[0..n]), timeout_ns);
        try vk.upload(cmds, e, amountsAt(p.cap), staging, std.mem.sliceAsBytes(p.subs.items[0..n]), timeout_ns);
    }

    /// The device entry table for `n` entries: grown (never shrunk) to at least `n`, refused past the bound.
    fn grow(p: *Penalties, n: u32) !void {
        if (p.entries != null and p.cap >= n) return;
        if (n > max_entries) return error.Invalid;
        const new_cap: u32 = @max(n, 64);
        var b = try Buffer.init(p.dev, .device, amountsAt(new_cap) + @as(u64, new_cap) * 4);
        errdefer b.deinit();
        if (p.entries) |*old| old.deinit();
        p.entries = b;
        p.cap = new_cap;
    }

    fn bindTable(p: *Penalties, logits: View, n: u32) !void {
        const e = &p.entries.?;
        try p.pipe.bind(&.{
            logits.bind(),
            .{ .buffer = e, .len = @as(u64, n) * 4 },
            .{ .buffer = e, .offset = amountsAt(p.cap), .len = @as(u64, n) * 4 },
        });
    }

    /// The staged entries on the immediate draw path: one fenced upload and one fenced dispatch before `Greedy.select`.
    pub fn runOnce(p: *Penalties, cmds: *Commands, staging: Buffer, logits: View, timeout_ns: u64) !void {
        const n = p.staged();
        defer p.clear();
        if (n == 0) return;
        try p.grow(n);
        try p.uploadEntries(cmds, staging, n, timeout_ns);
        try p.bindTable(logits, n);
        // one fenced dispatch, ordered before the caller's select
        try p.pipe.run(cmds, std.mem.asBytes(&n), .{ (n + 255) / 256, 1, 1 }, timeout_ns);
    }
};

fn fakeStream(context: []const u32, prompt_len: usize) lanes.Stream {
    var s: lanes.Stream = undefined;
    s.context = .fromOwnedSlice(@constCast(context));
    s.prompt_len = prompt_len;
    return s;
}

const testing = std.testing;

test "a decode's entries count the reply and the pending token once each; the prompt is never counted" {
    const gpa = testing.allocator;
    var p: Penalties = .{ .gpa = gpa, .dev = undefined };
    defer p.deinit();
    // prompt 9 9 (not counted), reply 5 7 5, pending 7 at cache_len 5
    const s = fakeStream(&.{ 9, 9, 5, 7, 5, 7 }, 2);
    try p.stage(&s, 5, 7, .{ .presence = 1, .frequency = 0.5 });
    try testing.expectEqual(@as(u32, 2), p.staged());
    try testing.expectEqualSlices(u32, &.{ 5, 7 }, p.toks.items);
    try testing.expectEqualSlices(f32, &.{ 2.0, 2.0 }, p.subs.items); // presence + frequency * 2
}

test "the first decode after the prompt penalizes only the pending token" {
    const gpa = testing.allocator;
    var p: Penalties = .{ .gpa = gpa, .dev = undefined };
    defer p.deinit();
    const s = fakeStream(&.{ 1, 4 }, 1);
    try p.stage(&s, 2, 6, .{ .frequency = 1 });
    try testing.expectEqualSlices(u32, &.{ 4, 6 }, p.toks.items);
    try testing.expectEqualSlices(f32, &.{ 1, 1 }, p.subs.items);
}

test "inactive penalties stage nothing, even over a broken position" {
    const gpa = testing.allocator;
    var p: Penalties = .{ .gpa = gpa, .dev = undefined };
    defer p.deinit();
    const s = fakeStream(&.{ 1, 2, 3 }, 2);
    try p.stage(&s, 99, 0, .{}); // defaults are off: no guard fires, nothing staged
    try testing.expectEqual(@as(u32, 0), p.staged());
}

test "a position before the reply or past the committed tokens is refused" {
    const gpa = testing.allocator;
    var p: Penalties = .{ .gpa = gpa, .dev = undefined };
    defer p.deinit();
    const s = fakeStream(&.{ 1, 2, 3 }, 2);
    try testing.expectError(error.InvalidQwenPosition, p.stage(&s, 1, 0, .{ .presence = 1 }));
    try testing.expectError(error.InvalidQwenPosition, p.stage(&s, 4, 0, .{ .presence = 1 }));
    try testing.expectEqual(@as(u32, 0), p.staged());
}

test "non-finite controls and out-of-vocabulary tokens are refused before anything stages" {
    const gpa = testing.allocator;
    var p: Penalties = .{ .gpa = gpa, .dev = undefined };
    defer p.deinit();
    const s = fakeStream(&.{ 1, 2 }, 1);
    try testing.expectError(error.NonFinite, p.stage(&s, 2, 3, .{ .presence = std.math.nan(f32) }));
    try testing.expectError(error.NonFinite, p.stage(&s, 2, 3, .{ .frequency = std.math.inf(f32) }));
    const over: [2]u32 = .{ 1, pinned.vocab };
    const oov = fakeStream(&over, 1);
    try testing.expectError(error.InvalidQwenToken, p.stage(&oov, 2, 0, .{ .presence = 1 }));
    try testing.expectError(error.InvalidQwenToken, p.stage(&s, 2, pinned.vocab, .{ .presence = 1 }));
    try testing.expectEqual(@as(u32, 0), p.staged());
}

test "staging clears between decodes: only the new decode's entries ride the next apply" {
    const gpa = testing.allocator;
    var p: Penalties = .{ .gpa = gpa, .dev = undefined };
    defer p.deinit();
    const s = fakeStream(&.{ 1, 4, 8 }, 1);
    try p.stage(&s, 2, 8, .{ .frequency = 1 });
    try testing.expectEqualSlices(u32, &.{ 4, 8 }, p.toks.items);
    try p.stage(&s, 3, 4, .{ .frequency = 1 }); // 4 8 committed, 4 pending: 4:2, 8:1
    try testing.expectEqualSlices(u32, &.{ 4, 8 }, p.toks.items);
    try testing.expectEqualSlices(f32, &.{ 2, 1 }, p.subs.items);
    p.clear();
    try testing.expectEqual(@as(u32, 0), p.staged());
}

test "the amounts binding offset is a 256-byte multiple past every toks range" {
    for ([_]u32{ 1, 2, 4, 63, 64, 65, 1000, max_entries }) |cap| {
        try testing.expectEqual(@as(u64, 0), amountsAt(cap) % 256);
        try testing.expect(amountsAt(cap) >= @as(u64, cap) * 4);
        try testing.expect(amountsAt(cap) < @as(u64, cap) * 4 + 256);
    }
}
