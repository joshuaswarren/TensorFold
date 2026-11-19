//! TENSORFOLD_REQUEST_LOG: each chat and completion body appended as one JSON line, images redacted, for replays.
const std = @import("std");
const builtin = @import("builtin");
const platform = @import("sys");
const json = @import("json");
const meta = @import("file_meta.zig").meta;
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// The body with every image part's URL replaced, the rest kept.
fn redact(a: Allocator, v: Value) Allocator.Error!Value {
    switch (v) {
        .array => |items| {
            const out = try a.alloc(Value, items.len);
            for (items, out) |item, *slot| slot.* = try redact(a, item);
            return .{ .array = out };
        },
        .object => |o| {
            if (v.typeIs("image_url")) {
                const copy = try json.copyObject(a, o);
                const url = try json.newObject(a);
                try url.put(a, "url", .{ .string = "<redacted>" });
                try copy.put(a, "image_url", .{ .object = url });
                return .{ .object = copy };
            }
            if (v.typeIs("file")) {
                const copy = try json.copyObject(a, o);
                const file_obj = try json.newObject(a);
                if (v.get("file")) |f| if (f == .object) if (f.get("filename")) |fn_val| try file_obj.put(a, "filename", fn_val);
                try file_obj.put(a, "file_data", .{ .string = "<redacted>" });
                try copy.put(a, "file", .{ .object = file_obj });
                return .{ .object = copy };
            }
            const out = try json.newObject(a);
            for (o.keys(), o.values()) |k, item| try out.put(a, k, try redact(a, item));
            return .{ .object = out };
        },
        else => return v,
    }
}

/// Appends ``body`` to ``path`` unless it is background work (batch jobs are not client traffic).
/// ``request_id`` stamps the reply id onto the line so its reply record can sit beside it.
pub fn append(a: Allocator, path: []const u8, body: Value, request_id: ?[]const u8) void {
    if (body.strField("priority")) |p| if (std.mem.eql(u8, p, "background")) return;
    var kept = redact(a, body) catch return;
    if (request_id) |id| if (kept == .object) kept.object.put(a, "request_id", .{ .string = id }) catch return;
    put(a, path, kept);
}

/// The reply half of one request line: the answer and reasoning returned, its finish, counts and timings.
pub const ReplyRecord = struct {
    id: []const u8,
    content: []const u8,
    reasoning: ?[]const u8,
    finish: []const u8,
    prompt_tokens: usize,
    cached_tokens: usize,
    completion_tokens: usize,
    reasoning_tokens: usize,
    queued_s: f64,
    ttft_s: f64,
    elapsed_s: f64,
    tok_s: f64,
};

/// Appends a reply record for ``request_id``; the same privacy rules as the request side.
pub fn reply(a: Allocator, path: []const u8, r: ReplyRecord) void {
    const o = json.newObject(a) catch return;
    o.put(a, "request_id", .{ .string = r.id }) catch return;
    const rep = json.newObject(a) catch return;
    rep.put(a, "content", .{ .string = r.content }) catch return;
    if (r.reasoning) |text| rep.put(a, "reasoning", .{ .string = text }) catch return;
    rep.put(a, "finish_reason", .{ .string = r.finish }) catch return;
    const usage = json.newObject(a) catch return;
    usage.put(a, "prompt_tokens", json.intValue(a, r.prompt_tokens) catch return) catch return;
    usage.put(a, "cached_tokens", json.intValue(a, r.cached_tokens) catch return) catch return;
    usage.put(a, "completion_tokens", json.intValue(a, r.completion_tokens) catch return) catch return;
    usage.put(a, "reasoning_tokens", json.intValue(a, r.reasoning_tokens) catch return) catch return;
    rep.put(a, "usage", .{ .object = usage }) catch return;
    const timings = json.newObject(a) catch return;
    timings.put(a, "queued_s", .{ .float = r.queued_s }) catch return;
    timings.put(a, "ttft_s", .{ .float = r.ttft_s }) catch return;
    timings.put(a, "elapsed_s", .{ .float = r.elapsed_s }) catch return;
    timings.put(a, "tok_s", .{ .float = r.tok_s }) catch return;
    rep.put(a, "timings", .{ .object = timings }) catch return;
    o.put(a, "reply", .{ .object = rep }) catch return;
    put(a, path, .{ .object = o });
}

/// One JSON value as one line in ``path``; refusals warn once, for the request and reply sides alike.
fn put(a: Allocator, path: []const u8, v: Value) void {
    const line = json.stringify(a, v, .{}) catch return;
    const pathz = a.dupeSentinel(u8, path, 0) catch return;
    const fd = platform.c.open(pathz, .{ .ACCMODE = .WRONLY, .APPEND = true, .CREAT = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(platform.c.mode_t, 0o600));
    if (fd < 0) return warnOnce(path, "it cannot be opened, or it is a symbolic link");
    defer _ = platform.c.close(fd);
    if (!private(fd)) return warnOnce(path, "it is not a regular file only its owner can read");
    const framed = std.mem.concat(a, u8, &.{ line, "\n" }) catch return;
    _ = platform.c.write(fd, framed.ptr, framed.len);
}

/// A regular file no one but its owner can read or write: a name in a shared directory can be someone else's file.
fn private(fd: platform.c.fd_t) bool {
    if (builtin.os.tag == .windows) return true; // NTFS keeps access in ACLs, not mode bits
    const m = meta(fd, null) orelse return false;
    return m.mode & 0o170000 == 0o100000 and m.mode & 0o077 == 0;
}

var warned = std.atomic.Value(bool).init(false);

fn warnOnce(path: []const u8, why: []const u8) void {
    if (warned.swap(true, .monotonic)) return;
    std.log.warn("TENSORFOLD_REQUEST_LOG {s} gets no requests: {s}", .{ path, why });
}

test "the request log is created owner-only and keeps image URLs out" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/requests.jsonl", .{tmp.sub_path});
    const url = try json.newObject(a);
    try url.put(a, "url", .{ .string = "https://example.test/private.png" });
    const part = try json.newObject(a);
    try part.put(a, "type", .{ .string = "image_url" });
    try part.put(a, "image_url", .{ .object = url });
    const body = try json.newObject(a);
    const file = (try json.parse(a, "{\"type\": \"file\", \"file\": {\"filename\": \"a.pdf\", \"file_data\": \"JVBERi1zZWNyZXQ=\"}}")).ok;
    try body.put(a, "messages", .{ .array = try a.dupe(Value, &.{ .{ .object = part }, file }) });
    append(a, path, .{ .object = body }, null);
    if (comptime @import("builtin").os.tag != .windows) { // Windows files have no mode bits
        const st = try std.Io.Dir.cwd().statFile(std.testing.io, path, .{}); // Linux's libc has no stat for std.c
        try std.testing.expectEqual(@as(u32, 0o600), @as(u32, @intCast(st.permissions.toMode())) & 0o777);
    }
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(1 << 16));
    try std.testing.expect(std.mem.indexOf(u8, text, "<redacted>") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "example.test") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "JVBERi1zZWNyZXQ=") == null and std.mem.indexOf(u8, text, "a.pdf") != null);
}

test "appends from several threads keep every line whole" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cwd = try std.process.currentPathAlloc(std.testing.io, a);
    defer a.free(cwd);
    const path = try std.fmt.allocPrint(a, "{s}/.zig-cache/tmp/{s}/log.jsonl", .{ cwd, tmp.sub_path });
    defer a.free(path);
    const threads = 8;
    const lines = 200;
    const head = "{\"line\": \"";
    const Worker = struct {
        fn run(p: []const u8, k: u8) void {
            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            const text = arena.allocator().alloc(u8, 300) catch return;
            @memset(text, 'a' + k);
            for (0..lines) |_| {
                const o = json.newObject(arena.allocator()) catch return;
                o.put(arena.allocator(), "line", .{ .string = text }) catch return;
                append(arena.allocator(), p, .{ .object = o }, null);
            }
        }
    };
    var workers: [threads]std.Thread = undefined;
    for (&workers, 0..) |*w, k| w.* = try std.Thread.spawn(.{}, Worker.run, .{ path, @as(u8, @intCast(k)) });
    for (workers) |w| w.join();
    const got = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(1 << 24));
    defer a.free(got);
    var counts: [threads]usize = @splat(0);
    var it = std.mem.splitScalar(u8, got, '\n');
    while (it.next()) |line| {
        if (line.len == 0) continue;
        try std.testing.expectEqual(head.len + 300 + 2, line.len);
        const k = line[head.len] - 'a';
        try std.testing.expect(k < threads);
        for (line[head.len..][0..300]) |c| try std.testing.expectEqual('a' + k, c);
        counts[k] += 1;
    }
    for (counts) |n| try std.testing.expectEqual(@as(usize, lines), n);
}

test "a log others can read, or a symbolic link, gets no requests; the owner's own file still does" {
    const log_level = std.testing.log_level;
    std.testing.log_level = .err;
    defer std.testing.log_level = log_level;
    if (builtin.os.tag == .windows) return error.SkipZigTest; // refusal by mode bits and links; NTFS keeps access in ACLs
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const body = try json.newObject(a);
    try body.put(a, "model", .{ .string = "m" });
    // a name another user created first, readable by others: nothing is appended to it
    try tmp.dir.writeFile(io, .{ .sub_path = "taken.jsonl", .data = "" });
    try tmp.dir.setFilePermissions(io, "taken.jsonl", .fromMode(0o644), .{}); // set after creating: a umask clears bits
    append(a, try std.fmt.allocPrint(a, "{s}/taken.jsonl", .{dir}), .{ .object = body }, null);
    try std.testing.expectEqual(@as(usize, 0), (try tmp.dir.readFileAlloc(io, "taken.jsonl", a, .limited(1 << 10))).len);
    // a symbolic link to another file: the target is never written
    try tmp.dir.writeFile(io, .{ .sub_path = "target", .data = "", .flags = .{ .permissions = .fromMode(0o600) } });
    try tmp.dir.symLink(io, "target", "link.jsonl", .{});
    append(a, try std.fmt.allocPrint(a, "{s}/link.jsonl", .{dir}), .{ .object = body }, null);
    try std.testing.expectEqual(@as(usize, 0), (try tmp.dir.readFileAlloc(io, "target", a, .limited(1 << 10))).len);
    // the owner's own 0600 file keeps its lines
    append(a, try std.fmt.allocPrint(a, "{s}/target", .{dir}), .{ .object = body }, null);
    try std.testing.expect((try tmp.dir.readFileAlloc(io, "target", a, .limited(1 << 10))).len > 0);
}

test "a reply record is appended beside its request, owner-only, with counts and timings" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/requests.jsonl", .{tmp.sub_path});
    const body = try json.newObject(a);
    try body.put(a, "model", .{ .string = "m" });
    append(a, path, .{ .object = body }, "chatcmpl-abc");
    reply(a, path, .{
        .id = "chatcmpl-abc",
        .content = "Hi",
        .reasoning = "Plan",
        .finish = "stop",
        .prompt_tokens = 5,
        .cached_tokens = 2,
        .completion_tokens = 3,
        .reasoning_tokens = 1,
        .queued_s = 0.01,
        .ttft_s = 0.02,
        .elapsed_s = 0.05,
        .tok_s = 60.0,
    });
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(1 << 16));
    var it = std.mem.splitScalar(u8, text, '\n');
    const req_line = (try json.parse(a, it.next().?)).ok;
    try std.testing.expectEqualStrings("chatcmpl-abc", req_line.strField("request_id").?);
    try std.testing.expectEqualStrings("m", req_line.strField("model").?);
    const rep_line = (try json.parse(a, it.next().?)).ok;
    try std.testing.expectEqualStrings("chatcmpl-abc", rep_line.strField("request_id").?);
    const r = rep_line.get("reply").?;
    try std.testing.expectEqualStrings("Hi", r.strField("content").?);
    try std.testing.expectEqualStrings("Plan", r.strField("reasoning").?);
    try std.testing.expectEqualStrings("stop", r.strField("finish_reason").?);
    const usage = r.get("usage").?;
    try std.testing.expectEqual(@as(?i64, 5), usage.get("prompt_tokens").?.int64());
    try std.testing.expectEqual(@as(?i64, 2), usage.get("cached_tokens").?.int64());
    try std.testing.expectEqual(@as(?i64, 3), usage.get("completion_tokens").?.int64());
    try std.testing.expectEqual(@as(?i64, 1), usage.get("reasoning_tokens").?.int64());
    const timings = r.get("timings").?;
    try std.testing.expectApproxEqAbs(@as(f64, 0.02), timings.get("ttft_s").?.float, 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 0.01), timings.get("queued_s").?.float, 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 0.05), timings.get("elapsed_s").?.float, 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 60.0), timings.get("tok_s").?.float, 0.5);
    try std.testing.expect(it.next() == null or it.next().?.len == 0);
}

test "a reply to an unreadable log is refused like the request side" {
    const log_level = std.testing.log_level;
    std.testing.log_level = .err;
    defer std.testing.log_level = log_level;
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    try tmp.dir.writeFile(io, .{ .sub_path = "taken.jsonl", .data = "" });
    try tmp.dir.setFilePermissions(io, "taken.jsonl", .fromMode(0o644), .{});
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    reply(a, try std.fmt.allocPrint(a, "{s}/taken.jsonl", .{dir}), .{
        .id = "chatcmpl-x", .content = "Hi", .reasoning = null, .finish = "stop",
        .prompt_tokens = 1, .cached_tokens = 0, .completion_tokens = 1, .reasoning_tokens = 0,
        .queued_s = 0, .ttft_s = 0, .elapsed_s = 0, .tok_s = 0,
    });
    try std.testing.expectEqual(@as(usize, 0), (try tmp.dir.readFileAlloc(io, "taken.jsonl", a, .limited(1 << 10))).len);
}
