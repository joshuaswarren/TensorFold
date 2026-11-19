//! `tensorfold log`: read a TENSORFOLD_REQUEST_LOG file - each request beside its reply, one request, or follow.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// One request id's pair of lines; a request the reply never reached has `reply` null.
/// `raw` keeps the request line's own text for `show`.
pub const Entry = struct {
    id: []const u8,
    request: ?std.json.Value = null,
    request_raw: []const u8 = "",
    reply: ?std.json.Value = null,
    reply_raw: []const u8 = "",
};

fn field(v: std.json.Value, key: []const u8) ?std.json.Value {
    return if (v == .object) v.object.get(key) else null;
}

fn str(v: std.json.Value, key: []const u8) ?[]const u8 {
    const f = field(v, key) orelse return null;
    return if (f == .string) f.string else null;
}

/// Reads the whole file and pairs its lines by `request_id`, in file order.
pub fn load(a: Allocator, io: std.Io, path: []const u8) ![]Entry {
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 28)) catch return error.OpenFailed;
    var entries: std.ArrayList(Entry) = .empty;
    var by_id: std.StringHashMap(usize) = .init(a);
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, line, .{}) catch continue;
        const id = str(parsed, "request_id") orelse "";
        const at = by_id.get(id) orelse blk: {
            try entries.append(a, .{ .id = id });
            const i = entries.items.len - 1;
            if (id.len > 0) try by_id.put(id, i);
            break :blk i;
        };
        if (field(parsed, "reply") != null) {
            entries.items[at].reply = parsed;
            entries.items[at].reply_raw = line;
        } else {
            entries.items[at].request = parsed;
            entries.items[at].request_raw = line;
        }
    }
    return entries.items;
}

fn preview(v: std.json.Value, buf: []u8) []const u8 {
    const text = str(v, "content") orelse return "";
    if (text.len <= buf.len) return text;
    return std.fmt.bufPrint(buf, "{s}...", .{text[0 .. buf.len - 3]}) catch text[0 .. buf.len - 3];
}

/// One row per request: the id, the model asked for, and the reply's finish, counts and first words.
fn list(out: *std.Io.Writer, entries: []const Entry) !void {
    for (entries) |e| {
        const model = if (e.request) |r| str(r, "model") orelse "-" else "-";
        try out.print("{s}\t{s}", .{ if (e.id.len > 0) e.id else "-", model });
        if (e.reply) |v| {
            const r = field(v, "reply").?;
            var buf: [64]u8 = undefined;
            const counts = field(r, "usage") orelse .null;
            const timings = field(r, "timings") orelse .null;
            const ttft = if (field(timings, "ttft_s")) |t| t.float else 0;
            try out.print("\t{s}\ttokens={?}/{?}\tttft={d:.3}s\t{s}\n", .{
                str(r, "finish_reason") orelse "-",
                if (field(counts, "prompt_tokens")) |n| n.integer else null,
                if (field(counts, "completion_tokens")) |n| n.integer else null,
                ttft,
                preview(r, &buf),
            });
        } else try out.writeAll("\tpending\n");
    }
}

/// The request and reply lines of one id, as logged.
fn show(out: *std.Io.Writer, entries: []const Entry, id: []const u8) !bool {
    for (entries) |e| {
        if (!std.mem.eql(u8, e.id, id)) continue;
        if (e.request_raw.len > 0) try out.print("{s}\n", .{e.request_raw});
        if (e.reply_raw.len > 0) try out.print("{s}\n", .{e.reply_raw});
        return true;
    }
    return false;
}

const usage =
    \\usage: tensorfold log [--file PATH] [show ID] [--follow]
    \\
;

/// `tensorfold log`: PATH comes from --file or TENSORFOLD_REQUEST_LOG; `show ID` one request, `--follow` keeps printing.
pub fn run(a: Allocator, io: std.Io, out: *std.Io.Writer, err: *std.Io.Writer, env: ?*const std.process.Environ.Map, args: []const []const u8) !u8 {
    var path: ?[]const u8 = null;
    var show_id: ?[]const u8 = null;
    var follow = false;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--file")) {
            i += 1;
            if (i >= args.len) break;
            path = args[i];
        } else if (std.mem.eql(u8, arg, "show")) {
            i += 1;
            if (i >= args.len) break;
            show_id = args[i];
        } else if (std.mem.eql(u8, arg, "--follow") or std.mem.eql(u8, arg, "-f")) follow = true else {
            try err.writeAll(usage);
            return 2;
        }
    }
    if (path == null and env != null) path = env.?.get("TENSORFOLD_REQUEST_LOG");
    const file = path orelse {
        try err.writeAll("tensorfold log: no request log named; pass --file PATH or set TENSORFOLD_REQUEST_LOG\n");
        return 2;
    };
    while (true) {
        const entries = load(a, io, file) catch {
            try err.print("tensorfold log: cannot read {s}\n", .{file});
            return 2;
        };
        if (show_id) |id| {
            if (!try show(out, entries, id)) {
                try err.print("tensorfold log: no request {s} in {s}\n", .{ id, file });
                return 1;
            }
        } else try list(out, entries);
        try out.flush();
        if (!follow) return 0;
        try std.Io.sleep(io, .fromMilliseconds(250), .awake);
    }
}

test {
    _ = @import("log_test.zig");
}
