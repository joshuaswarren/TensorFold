//! Decode group fusion: independent qmm projections of one activation run as one dispatch; pure plan.

const std = @import("std");
const ops_q4 = @import("ops_q4.zig");
const dispatch = @import("dispatch.zig");
const Error = dispatch.Error;

/// TF_VK_FUSE=0 loads exactly as before; any other value or unset fuses.
pub fn fuseFromEnv(v: ?[*:0]const u8) bool {
    const s = v orelse return true;
    return !std.mem.eql(u8, std.mem.span(s), "0");
}

pub const max_members = 4;

/// One member's place in the fused buffers: `rows` rows from `start`, chosen so its binding offsets are all aligned.
pub const MemberPlan = struct { rows: u64, start: u64 };

pub const GroupPlan = struct {
    members: [max_members]MemberPlan = undefined,
    n: u32 = 0,
    total_rows: u64 = 0,
    w_bytes: u64 = 0,
    params_bytes: u64 = 0,
};

/// The fused layout of `member_rows` members of one k: pad before each so every binding start aligns; refuses past it.
pub fn planGroup(k: u64, member_rows: []const u64, lim: ops_q4.Limits) Error!GroupPlan {
    if (member_rows.len == 0 or member_rows.len > max_members) return error.Invalid;
    if (k == 0 or k % ops_q4.group_size != 0) return error.Invalid;
    const w_row: u64 = k / 2;
    const param_row: u64 = k / 32;
    var plan: GroupPlan = .{};
    var cursor: u64 = 0;
    for (member_rows) |rows| {
        if (rows == 0) return error.Invalid;
        var pad: u64 = 0;
        while ((cursor + pad) * 2 % lim.alignment != 0 or
            (cursor + pad) * param_row % lim.alignment != 0 or
            (cursor + pad) * w_row % lim.alignment != 0)
        {
            pad += 1;
            if (pad > lim.alignment) return error.TooLarge; // a start is always found within one period
        }
        cursor += pad;
        plan.members[plan.n] = .{ .rows = rows, .start = cursor };
        plan.n += 1;
        cursor += rows;
    }
    plan.total_rows = cursor;
    plan.w_bytes = cursor * w_row;
    plan.params_bytes = cursor * param_row;
    if (plan.w_bytes > lim.max_range or plan.params_bytes > lim.max_range) return error.TooLarge;
    return plan;
}

/// Would one dispatch over the fused group take the SG decode kernel within the device ranges? Else members run apart.
pub fn groupAdmits(plan: GroupPlan, k: u64, caps: ops_q4.SubgroupCaps, have_sg: bool, lim: ops_q4.Limits, max_groups: u32) bool {
    if (plan.n == 0 or plan.w_bytes > lim.max_range or plan.params_bytes > lim.max_range) return false;
    if (!ops_q4.qmmDecodeSgAdmits(1, k, plan.total_rows, caps, have_sg)) return false;
    _ = ops_q4.pkGroups(plan.total_rows, max_groups) catch return false;
    return true;
}

/// The fused tensor's bytes: pad rows zero, member i's rows at its start; pure and host-tested.
pub fn assemble(comptime T: type, dest: []T, row_elems: u64, plan: GroupPlan, members: []const []const T) void {
    @memset(dest, 0);
    for (members, 0..) |m, i| {
        const at = plan.members[i].start * row_elems;
        @memcpy(dest[at..][0..m.len], m);
    }
}

const testing = std.testing;

const align256: ops_q4.Limits = .{ .max_range = 1 << 30, .alignment = 256 };

test "envOne's sibling: fuseFromEnv turns off only for the exact 0" {
    try testing.expect(fuseFromEnv(null));
    try testing.expect(fuseFromEnv("1"));
    try testing.expect(!fuseFromEnv("0"));
    try testing.expect(fuseFromEnv("off"));
}

test "the group planner pads the real 2B decode groups to aligned starts" {
    // GDN: qkv 6144, z 2048, a 16, b 16: only b pads, 112 rows, because a ends 32 bytes past an aligned output start
    const gdn = try planGroup(2048, &.{ 6144, 2048, 16, 16 }, align256);
    try testing.expectEqual(@as(u32, 4), gdn.n);
    try testing.expectEqual([_]u64{ 0, 6144, 8192, 8320 }, [_]u64{ gdn.members[0].start, gdn.members[1].start, gdn.members[2].start, gdn.members[3].start });
    try testing.expectEqual(@as(u64, 8336), gdn.total_rows);
    for (0..gdn.n) |i| {
        try testing.expectEqual(@as(u64, 0), gdn.members[i].start * 2 % 256); // output binding
        try testing.expectEqual(@as(u64, 0), gdn.members[i].start * 64 % 256); // scale and bias bindings
        try testing.expectEqual(@as(u64, 0), gdn.members[i].start * 1024 % 256); // weight binding
    }
    // full attention: q (the interleaved q|gate rows) 4096, k 512, v 512, no pad
    const attn = try planGroup(2048, &.{ 4096, 512, 512 }, align256);
    try testing.expectEqual([_]u64{ 0, 4096, 4608 }, [_]u64{ attn.members[0].start, attn.members[1].start, attn.members[2].start });
    try testing.expectEqual(@as(u64, 5120), attn.total_rows);
    // dense MLP: gate 6144, up 6144, no pad
    const mlp = try planGroup(2048, &.{ 6144, 6144 }, align256);
    try testing.expectEqual([_]u64{ 0, 6144 }, [_]u64{ mlp.members[0].start, mlp.members[1].start });
    try testing.expectEqual(@as(u64, 12288), mlp.total_rows);
    // the merged weight of the widest group fits the device; a tighter range refuses
    try testing.expect(gdn.w_bytes <= align256.max_range);
    try testing.expectError(error.TooLarge, planGroup(2048, &.{ 6144, 2048, 16, 16 }, .{ .max_range = 1 << 22, .alignment = 256 }));
    try testing.expectError(error.Invalid, planGroup(2048, &.{}, align256));
    try testing.expectError(error.Invalid, planGroup(2048, &.{ 6144, 2048, 16, 16, 16 }, align256));
}

test "groupAdmits rides the SG decode gate and the device ranges" {
    const gdn = try planGroup(2048, &.{ 6144, 2048, 16, 16 }, align256);
    const caps: ops_q4.SubgroupCaps = .{ .subgroup_size = 32, .subgroup_ops = 1 << 2 };
    try testing.expect(groupAdmits(gdn, 2048, caps, true, align256, 65535));
    try testing.expect(!groupAdmits(gdn, 2048, caps, false, align256, 65535)); // no SG pipeline
    try testing.expect(!groupAdmits(gdn, 2048, .{ .subgroup_size = 64, .subgroup_ops = 1 << 2 }, true, align256, 65535));
    try testing.expect(!groupAdmits(gdn, 2048, caps, true, .{ .max_range = 1 << 22, .alignment = 256 }, 65535));
}

test "assembling the fused bytes keeps member rows and zeroes pads" {
    const plan = try planGroup(2048, &.{ 4, 2, 16 }, align256); // small stand-ins, same alignment shape
    const row_words: u64 = 2048 / 8;
    const w = try testing.allocator.alloc(u32, plan.total_rows * row_words);
    defer testing.allocator.free(w);
    const m0: [1024]u32 = .{ 1, 2, 3, 4, 5, 6, 7, 8 } ++ @as([1016]u32, @splat(0));
    const m1: [512]u32 = .{ 9, 10, 11, 12, 13, 14, 15, 16 } ++ @as([504]u32, @splat(0));
    var m2: [16 * 256]u32 = undefined;
    for (&m2, 0..) |*v, i| v.* = @intCast(100 + i);
    assemble(u32, w, row_words, plan, &.{ &m0, &m1, &m2 });
    try testing.expectEqualSlices(u32, &m0, w[0..m0.len]);
    for (w[m0.len .. plan.members[1].start * row_words]) |v| try testing.expectEqual(@as(u32, 0), v); // pad rows zero
    try testing.expectEqualSlices(u32, &m1, w[plan.members[1].start * row_words ..][0..m1.len]);
    try testing.expectEqualSlices(u32, &m2, w[plan.members[2].start * row_words ..][0..m2.len]);
}

/// The host single-row qmm chain over packed words as uploaded (in-order acc += x * (s q + b)): one output per row.
fn replayRow(x: []const f32, w: []const u32, scales: []const u16, biases: []const u16, w_row_words: u64, param_row_groups: u64, row: u64, k: u64) f32 {
    var acc: f32 = 0;
    const s_at = row * param_row_groups;
    const w_at = row * w_row_words;
    for (0..k) |i| {
        const sc: f32 = @floatCast(f16bit(scales[s_at + i / 64]));
        const b: f32 = @floatCast(f16bit(biases[s_at + i / 64]));
        const word = w[w_at + i / 8];
        const q: f32 = @floatFromInt((word >> @intCast(4 * (i % 8))) & 0xf);
        acc += x[i] * (sc * q + b);
    }
    return acc;
}

fn f16bit(v: u16) f32 {
    return @as(f32, @floatFromInt(v)); // synthetic params are small integers: exact in every widening
}

test "a row replayed from the fused bytes equals the member replayed alone, bit for bit" {
    const k: u64 = 2048;
    const plan = try planGroup(k, &.{ 6144, 2048, 16, 16 }, align256);
    const row_words: u64 = k / 8;
    const row_groups: u64 = k / 64;
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const rand = prng.random();
    // synthetic members: packed nibble words and small bf16-exact params
    const w = try testing.allocator.alloc(u32, plan.total_rows * row_words);
    defer testing.allocator.free(w);
    const s = try testing.allocator.alloc(u16, plan.total_rows * row_groups);
    defer testing.allocator.free(s);
    const b = try testing.allocator.alloc(u16, plan.total_rows * row_groups);
    defer testing.allocator.free(b);
    for (w) |*v| v.* = rand.int(u32);
    for (s) |*v| v.* = @intCast(1 + rand.uintLessThan(u16, 4));
    for (b) |*v| v.* = @intCast(rand.uintLessThan(u16, 4));
    const x = try testing.allocator.alloc(f32, k);
    defer testing.allocator.free(x);
    for (x) |*v| v.* = rand.float(f32) * 2 - 1;
    var members: [4][]u32 = undefined;
    var members_s: [4][]u16 = undefined;
    var members_b: [4][]u16 = undefined;
    defer for (0..4) |i| {
        testing.allocator.free(members[i]);
        testing.allocator.free(members_s[i]);
        testing.allocator.free(members_b[i]);
    };
    const member_rows = [_]u64{ 6144, 2048, 16, 16 };
    for (0..4) |i| {
        members[i] = try testing.allocator.alloc(u32, member_rows[i] * row_words);
        members_s[i] = try testing.allocator.alloc(u16, member_rows[i] * row_groups);
        members_b[i] = try testing.allocator.alloc(u16, member_rows[i] * row_groups);
        for (members[i]) |*v| v.* = rand.int(u32);
        for (members_s[i]) |*v| v.* = @intCast(1 + rand.uintLessThan(u16, 4));
        for (members_b[i]) |*v| v.* = @intCast(rand.uintLessThan(u16, 4));
    }
    assemble(u32, w, row_words, plan, &members);
    assemble(u16, s, row_groups, plan, &members_s);
    assemble(u16, b, row_groups, plan, &members_b);
    // every member row: fused bytes replay to the member's own replay, bit for bit
    for (0..4) |i| for (0..member_rows[i]) |r| {
        const fused = replayRow(x, w, s, b, row_words, row_groups, plan.members[i].start + r, k);
        const alone = replayRow(x, members[i], members_s[i], members_b[i], row_words, row_groups, r, k);
        try testing.expect(fused == alone);
    };
}

test "the real 2B tower fuses 90 of its 490 decode dispatches away" {
    // qmm dispatches per decode token: 18 x (qkv,z,a,b,out) + 6 x (q,k,v,o) + 24 x (gate,up,down) + head = 187.
    var qmm: u64 = 1; // the head
    var saved: u64 = 0;
    for (0..24) |i| {
        if (i % 4 == 3) {
            qmm += 4; // q,k,v,o
            saved += 2; // q,k,v -> one
        } else {
            qmm += 5; // qkv,z,a,b,out
            saved += 3; // qkv,z,a,b -> one
        }
        qmm += 3; // gate,up,down
        saved += 1; // gate,up -> one
    }
    try testing.expectEqual(@as(u64, 187), qmm);
    try testing.expectEqual(@as(u64, 90), saved);
    try testing.expectEqual(@as(u64, 490 - 90), 490 - saved); // 400 remain
}
