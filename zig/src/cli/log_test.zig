//! `tensorfold log` over a written request log: list, show and pairing.
const std = @import("std");
const log = @import("log.zig");

fn fixture(a: std.mem.Allocator) ![]const u8 {
    const tmp = std.testing.tmpDir(.{});
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/requests.jsonl", .{tmp.sub_path});
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data =
        \\{"model":"m","messages":[{"role":"user","content":"hi"}],"request_id":"chatcmpl-1"}
        \\{"request_id":"chatcmpl-1","reply":{"content":"Hello","reasoning":"Hmm","finish_reason":"stop","usage":{"prompt_tokens":4,"cached_tokens":1,"completion_tokens":2,"reasoning_tokens":1},"timings":{"queued_s":0.005,"ttft_s":0.01,"elapsed_s":0.05,"tok_s":40.0}}}
        \\{"model":"m","prompt":"gone","request_id":"chatcmpl-2"}
        \\
    });
    return path;
}

test "log load pairs each reply with its request line" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const entries = try log.load(a, std.testing.io, try fixture(a));
    try std.testing.expectEqual(@as(usize, 2), entries.len);
    try std.testing.expectEqualStrings("chatcmpl-1", entries[0].id);
    try std.testing.expect(entries[0].request != null and entries[0].reply != null);
    try std.testing.expectEqualStrings("chatcmpl-2", entries[1].id);
    try std.testing.expect(entries[1].request != null and entries[1].reply == null);
}

test "log lists every request and shows one id's pair" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const path = try fixture(a);
    var out_buf: [16 << 10]u8 = undefined;
    var out: std.Io.Writer = .fixed(&out_buf);
    var err_buf: [4 << 10]u8 = undefined;
    var err: std.Io.Writer = .fixed(&err_buf);
    try std.testing.expectEqual(@as(u8, 0), try log.run(a, std.testing.io, &out, &err, null, &.{ "--file", path }));
    const listed = out.buffered();
    try std.testing.expect(std.mem.indexOf(u8, listed, "chatcmpl-1") != null);
    try std.testing.expect(std.mem.indexOf(u8, listed, "stop") != null and std.mem.indexOf(u8, listed, "Hello") != null);
    try std.testing.expect(std.mem.indexOf(u8, listed, "pending") != null); // the request with no reply yet
    out = .fixed(&out_buf);
    try std.testing.expectEqual(@as(u8, 0), try log.run(a, std.testing.io, &out, &err, null, &.{ "--file", path, "show", "chatcmpl-1" }));
    const shown = out.buffered();
    try std.testing.expect(std.mem.indexOf(u8, shown, "\"messages\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, shown, "\"finish_reason\":\"stop\"") != null or std.mem.indexOf(u8, shown, "\"finish_reason\": \"stop\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, shown, "chatcmpl-2") == null);
    out = .fixed(&out_buf);
    try std.testing.expectEqual(@as(u8, 1), try log.run(a, std.testing.io, &out, &err, null, &.{ "--file", path, "show", "chatcmpl-none" }));
}

test "log with no file named explains instead of reading nothing" {
    var err_buf: [4 << 10]u8 = undefined;
    var err: std.Io.Writer = .fixed(&err_buf);
    var out_buf: [16 << 10]u8 = undefined;
    var out: std.Io.Writer = .fixed(&out_buf);
    var env = std.process.Environ.Map.init(std.testing.allocator);
    try std.testing.expectEqual(@as(u8, 2), try log.run(std.testing.allocator, std.testing.io, &out, &err, &env, &.{}));
    try std.testing.expect(std.mem.indexOf(u8, err.buffered(), "TENSORFOLD_REQUEST_LOG") != null);
}
