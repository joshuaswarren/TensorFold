//! The `pull` command: a Hugging Face checkpoint into the same cache 0.6.6 uses, resumable and verified.
const std = @import("std");
const Allocator = std.mem.Allocator;
const hub = @import("hub.zig");
const pull_net = @import("pull_net.zig");

pub const GiB: u64 = 1 << 30;

/// One file the hub's tree lists for the revision.
const Entry = struct { path: []const u8, size: u64, sha256: ?[32]u8 = null };

/// How pull talks to the hub: its proxy scope and the stall/resume knobs.
const Policy = struct {
    env: ?*const std.process.Environ.Map,
    host: []const u8,
    port: u16,
    stall_ms: u64,
    retries: u32,
};

fn endpointHost(endpoint: []const u8) []const u8 {
    const uri = std.Uri.parse(endpoint) catch return "";
    const h = uri.host orelse return "";
    return switch (h) {
        .raw, .percent_encoded => |s| s,
    };
}

fn endpointPort(endpoint: []const u8) u16 {
    const uri = std.Uri.parse(endpoint) catch return 443;
    if (uri.port) |p| return p;
    return if (std.mem.eql(u8, uri.scheme, "http")) 80 else 443;
}

/// Downloads ``repo[@revision]`` into the cache and prints where it landed. Returns the exit code.
pub fn run(a: Allocator, io: std.Io, out: *std.Io.Writer, err_out: *std.Io.Writer, env: ?*const std.process.Environ.Map, override: ?[]const u8, spec: []const u8) !u8 {
    const at = std.mem.indexOfScalar(u8, spec, '@');
    const repo = if (at) |i| spec[0..i] else spec;
    const revision = if (at) |i| spec[i + 1 ..] else "main";
    if (!hub.isRepoIdLike(repo)) {
        try err_out.print("{s} is not a Hugging Face repo id (owner/name)\n", .{spec});
        return 2;
    }
    const endpoint = envValue(env, "HF_ENDPOINT") orelse "https://huggingface.co";
    const root = try hub.cacheDir(a, env, override);
    const stall_ms = pull_net.envInt(env, "TENSORFOLD_PULL_STALL_MS", pull_net.default_stall_ms);
    const policy: Policy = .{
        .env = env,
        .host = endpointHost(endpoint),
        .port = endpointPort(endpoint),
        .stall_ms = stall_ms,
        .retries = @intCast(@min(pull_net.envInt(env, "TENSORFOLD_PULL_RETRIES", pull_net.default_retries), std.math.maxInt(u32))),
    };
    var client: std.http.Client = .{ .allocator = a, .io = io };
    defer client.deinit();
    pull_net.applyProxy(a, &client, env, policy.host, policy.port) catch {};

    const sha = resolveRevision(a, io, &client, out, endpoint, repo, revision, stall_ms) catch |e| switch (e) {
        error.HubStatus => return 1,
        error.PullStalled => {
            try err_out.print("{s}@{s}: no response from the hub for {d} ms; check the network and the proxy settings\n", .{ repo, revision, stall_ms });
            return 1;
        },
        else => {
            try err_out.print("{s}@{s}: the hub is unreachable ({t}); check the network or HF_ENDPOINT\n", .{ repo, revision, e });
            return 1;
        },
    };

    const entries = listTree(a, io, &client, endpoint, repo, sha, stall_ms) catch |e| switch (e) {
        error.HubStatus, error.NoConfig => return 1,
        error.PullStalled => {
            try err_out.print("{s}@{s}: no response from the hub for {d} ms; check the network and the proxy settings\n", .{ repo, revision, stall_ms });
            return 1;
        },
        else => {
            try err_out.print("{s}@{s}: the hub's file tree is unreachable ({t})\n", .{ repo, revision, e });
            return 1;
        },
    };

    // The family check runs on config.json before any weight moves.
    var config: ?Entry = null;
    for (entries) |e| if (std.mem.eql(u8, e.path, "config.json")) {
        config = e;
        break;
    };
    if (config == null) {
        try err_out.print("{s}: the hub tree has no config.json; refusing\n", .{repo});
        return 1;
    }
    const config_bytes = fetchOne(a, io, &client, endpoint, repo, sha, "config.json", stall_ms) catch |e| {
        try err_out.print("{s}: config.json is unreachable ({t})\n", .{ repo, e });
        return 1;
    };
    const model_type = modelTypeOf(a, config_bytes);
    if (hub.family(model_type)) |family| {
        try out.print("{s}: {s} ({s})\n", .{ repo, family.title, family.model_type });
    } else {
        try err_out.print("{s}: no registered Zig family serves model_type {s}; the 0.6 line may: tensorfold@0.6\n", .{ repo, model_type });
        return 1;
    }

    const repo_dir = try std.fs.path.join(a, &.{ root, try hub.repoDirName(a, repo) });
    const blobs_dir = try std.fs.path.join(a, &.{ repo_dir, "blobs" });
    const snapshot_dir = try std.fs.path.join(a, &.{ repo_dir, "snapshots", sha });
    const refs_dir = try std.fs.path.join(a, &.{ repo_dir, "refs" });
    const w = std.Io.Dir.cwd();
    for ([_][]const u8{ blobs_dir, snapshot_dir, refs_dir }) |d| try w.createDirPath(io, d);

    for (entries) |e| {
        const blob_name = try blobName(a, e);
        const blob_path = try std.fs.path.join(a, &.{ blobs_dir, blob_name });
        if (e.sha256 != null) blob_exists: {
            const size = fileStat(io, blob_path) catch break :blob_exists;
            if (size != e.size) break :blob_exists;
            try out.print("  {s}: cached\n", .{e.path});
            try linkIntoSnapshot(a, io, snapshot_dir, e.path, blob_name);
            continue;
        }
        const url = try std.fmt.allocPrint(a, "{s}/{s}/resolve/{s}/{s}", .{ endpoint, repo, sha, e.path });
        const got = download(a, io, out, url, blobs_dir, e, policy) catch |err| {
            if (err == error.PullStalled) {
                try err_out.print("  {s}: no progress for {d} ms after {d} resumes; check the network and the proxy settings\n", .{ e.path, stall_ms, policy.retries });
            } else {
                try err_out.print("  {s}: the download failed ({t})\n", .{ e.path, err });
            }
            return 1;
        };
        try linkIntoSnapshot(a, io, snapshot_dir, e.path, got);
    }
    try w.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ refs_dir, "main" }), .data = sha });
    try out.print("stored {s}@{s} at {s}\n", .{ repo, sha, snapshot_dir });
    return 0;
}

fn envValue(env: ?*const std.process.Environ.Map, key: []const u8) ?[]const u8 {
    return if (env) |m| m.get(key) else null;
}

/// The commit the hub serves for ``repo@revision``.
fn resolveRevision(a: Allocator, io: std.Io, client: *std.http.Client, out: *std.Io.Writer, endpoint: []const u8, repo: []const u8, revision: []const u8, stall_ms: u64) ![]const u8 {
    const url = try std.fmt.allocPrint(a, "{s}/api/models/{s}/revision/{s}", .{ endpoint, repo, revision });
    const body = try getJson(a, io, client, url, stall_ms);
    try out.print("[tensorfold] downloading {s}@{s} from Hugging Face\n", .{ repo, revision });
    var parsed = std.json.parseFromSlice(std.json.Value, a, body, .{}) catch return error.BadHubJson;
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadHubJson;
    const sha = parsed.value.object.get("sha") orelse return error.BadHubJson;
    if (sha != .string or sha.string.len == 0) return error.BadHubJson;
    return try a.dupe(u8, sha.string);
}

/// The revision's files: path, size and the sha256 the hub states for LFS objects.
fn listTree(a: Allocator, io: std.Io, client: *std.http.Client, endpoint: []const u8, repo: []const u8, sha: []const u8, stall_ms: u64) ![]Entry {
    const url = try std.fmt.allocPrint(a, "{s}/api/models/{s}/tree/{s}?recursive=true", .{ endpoint, repo, sha });
    const body = try getJson(a, io, client, url, stall_ms);
    var parsed = std.json.parseFromSlice(std.json.Value, a, body, .{}) catch return error.BadHubJson;
    defer parsed.deinit();
    if (parsed.value != .array) return error.BadHubJson;
    var entries: std.ArrayList(Entry) = .empty;
    for (parsed.value.array.items) |item| {
        if (item != .object) continue;
        const o = item.object;
        const kind = o.get("type") orelse continue;
        if (kind != .string or !std.mem.eql(u8, kind.string, "file")) continue;
        const path_v = o.get("path") orelse continue;
        if (path_v != .string) continue;
        const size_v = o.get("size");
        const size: u64 = if (size_v != null and size_v.? == .integer and size_v.?.integer >= 0) @intCast(size_v.?.integer) else 0;
        var sha256: ?[32]u8 = null;
        if (o.get("lfs")) |lfs| {
            if (lfs == .object) {
                if (lfs.object.get("oid")) |oid| {
                    if (oid == .string and std.mem.startsWith(u8, oid.string, "sha256:")) {
                        sha256 = parseHex32(oid.string[7..]) orelse null;
                    }
                }
            }
        }
        try entries.append(a, .{ .path = try a.dupe(u8, path_v.string), .size = size, .sha256 = sha256 });
    }
    if (entries.items.len == 0) return error.NoConfig;
    return entries.items;
}

/// config.json's model_type from raw bytes.
fn modelTypeOf(a: Allocator, bytes: []const u8) []const u8 {
    var parsed = std.json.parseFromSlice(std.json.Value, a, bytes, .{}) catch return "unknown";
    defer parsed.deinit();
    if (parsed.value != .object) return "unknown";
    const t = parsed.value.object.get("model_type") orelse return "unknown";
    return if (t == .string) t.string else "unknown";
}

fn parseHex32(text: []const u8) ?[32]u8 {
    if (text.len != 64) return null;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, text) catch return null;
    return out;
}

/// The blob's final cache name: the LFS sha256 when the hub states one, else the sha256 of the bytes.
fn blobName(a: Allocator, e: Entry) ![]const u8 {
    if (e.sha256) |digest| return try std.fmt.allocPrint(a, "{s}", .{std.fmt.bytesToHex(digest, .lower)});
    const safe = try a.dupe(u8, e.path);
    for (safe) |*ch| if (ch.* == '/') {
        ch.* = '_';
    };
    return safe;
}

/// One small file fetched whole into memory.
fn fetchOne(a: Allocator, io: std.Io, client: *std.http.Client, endpoint: []const u8, repo: []const u8, sha: []const u8, path: []const u8, stall_ms: u64) ![]u8 {
    const url = try std.fmt.allocPrint(a, "{s}/{s}/resolve/{s}/{s}", .{ endpoint, repo, sha, path });
    return getJson(a, io, client, url, stall_ms);
}

const FetchCtx = struct {
    a: Allocator,
    client: *std.http.Client,
    url: []const u8,
    body: []u8 = &.{},
};

fn fetchBody(ctx_any: *anyopaque, guard: *pull_net.Guard) anyerror!void {
    const ctx: *FetchCtx = @ptrCast(@alignCast(ctx_any));
    var body: std.Io.Writer.Allocating = .init(ctx.a);
    const result = ctx.client.fetch(.{ .location = .{ .url = ctx.url }, .response_writer = &body.writer }) catch return error.HubUnreachable;
    guard.bump();
    if (result.status != .ok) return error.HubStatus;
    ctx.body = try body.toOwnedSlice();
}

fn getJson(a: Allocator, io: std.Io, client: *std.http.Client, url: []const u8, stall_ms: u64) ![]u8 {
    var ctx: FetchCtx = .{ .a = a, .client = client, .url = url };
    try pull_net.guarded(io, stall_ms, &ctx, fetchBody);
    return ctx.body;
}

/// Downloads one file, resuming on a stall until `policy.retries` runs out, and returns the blob name.
fn download(a: Allocator, io: std.Io, out: *std.Io.Writer, url: []const u8, blobs_dir: []const u8, e: Entry, policy: Policy) ![]const u8 {
    var attempt: u32 = 0;
    while (true) {
        var ctx: DownloadCtx = .{ .a = a, .io = io, .out = out, .url = url, .blobs_dir = blobs_dir, .e = e, .policy = policy };
        pull_net.guarded(io, policy.stall_ms, &ctx, downloadBody) catch |err| {
            attempt += 1;
            if (err == error.PullStalled and attempt <= policy.retries) {
                try out.print("  {s}: no progress for {d} ms; resuming\n", .{ e.path, policy.stall_ms });
                continue;
            }
            return err;
        };
        return ctx.name;
    }
}

const DownloadCtx = struct {
    a: Allocator,
    io: std.Io,
    out: *std.Io.Writer,
    url: []const u8,
    blobs_dir: []const u8,
    e: Entry,
    policy: Policy,
    name: []const u8 = "",
};

fn downloadBody(ctx_any: *anyopaque, guard: *pull_net.Guard) anyerror!void {
    const ctx: *DownloadCtx = @ptrCast(@alignCast(ctx_any));
    // A fresh client per attempt: a stalled connection must not poison the resume.
    var client: std.http.Client = .{ .allocator = ctx.a, .io = ctx.io };
    defer client.deinit();
    pull_net.applyProxy(ctx.a, &client, ctx.policy.env, ctx.policy.host, ctx.policy.port) catch {};
    ctx.name = try downloadOnce(ctx.a, ctx.io, &client, ctx.out, ctx.url, ctx.blobs_dir, ctx.e, guard);
}

/// One attempt: HTTP range resume from the `.incomplete` file, then a size and sha256 check.
fn downloadOnce(a: Allocator, io: std.Io, client: *std.http.Client, out: *std.Io.Writer, url: []const u8, blobs_dir: []const u8, e: Entry, guard: *pull_net.Guard) ![]const u8 {
    const final_name = try blobName(a, e);
    const final_path = try std.fs.path.join(a, &.{ blobs_dir, final_name });
    const partial_path = try std.fmt.allocPrint(a, "{s}.incomplete", .{final_path});
    const w = std.Io.Dir.cwd();
    var resume_from: u64 = 0;
    if (e.sha256 != null) {
        if (fileStat(io, partial_path)) |size| resume_from = size else |_| {}
    }
    if (resume_from > e.size) resume_from = 0;

    var headers: [1]std.http.Header = undefined;
    var range_buf: [32]u8 = undefined;
    var extra: []const std.http.Header = &.{};
    if (resume_from > 0) {
        const range = try std.fmt.bufPrint(&range_buf, "bytes={d}-", .{resume_from});
        headers[0] = .{ .name = "Range", .value = range };
        extra = &headers;
    }
    const uri = std.Uri.parse(url) catch return error.BadUrl;
    var req = try client.request(.GET, uri, .{ .extra_headers = extra });
    defer req.deinit();
    try req.sendBodiless();
    var redirect_buf: [8 << 10]u8 = undefined;
    var response = try req.receiveHead(&redirect_buf);
    const status = response.head.status;
    if (status != .ok and status != .partial_content) return error.HubStatus;
    var restart = false;
    if (resume_from > 0 and status != .partial_content) {
        restart = true; // the hub ignored the range; start over
        resume_from = 0;
    }
    // Read access too: a resumed blob feeds its on-disk prefix into the same digest.
    const file = try w.createFile(io, partial_path, .{ .read = true, .truncate = restart });
    defer file.close(io);
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    if (resume_from > 0) {
        // The prefix already on disk feeds the same digest so the final check spans the whole file.
        const prefix = try a.alloc(u8, @intCast(resume_from));
        const read = file.readPositionalAll(io, prefix, 0) catch 0;
        hash.update(prefix[0..read]);
    }
    guard.bump();
    var reader_buf: [64 << 10]u8 = undefined;
    var r = response.reader(&reader_buf);
    var chunk: [32 << 10]u8 = undefined;
    var offset = resume_from;
    while (true) {
        const n = r.readSliceShort(&chunk) catch return error.ReadFailed;
        if (n == 0) break;
        guard.bump();
        hash.update(chunk[0..n]);
        try file.writePositionalAll(io, chunk[0..n], offset);
        offset += n;
    }
    const total = offset;
    if (total != e.size) return error.SizeMismatch;
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    if (e.sha256) |want| {
        if (!std.mem.eql(u8, &digest, &want)) return error.ShaMismatch;
    }
    try file.setLength(io, total);
    file.sync(io) catch {};
    try w.rename(partial_path, w, final_path, io);
    try out.print("  {s}: {d:.2} MiB\n", .{ e.path, @as(f64, @floatFromInt(total)) / (1 << 20) });
    return final_name;
}

fn fileStat(io: std.Io, path: []const u8) !u64 {
    const st = try std.Io.Dir.cwd().statFile(io, path, .{});
    return st.size;
}

/// The snapshot's path points at the blob, the way huggingface_hub's cache links them.
fn linkIntoSnapshot(a: Allocator, io: std.Io, snapshot_dir: []const u8, path: []const u8, blob_name: []const u8) !void {
    const link_path = try std.fs.path.join(a, &.{ snapshot_dir, path });
    if (std.fs.path.dirname(link_path)) |parent| try std.Io.Dir.cwd().createDirPath(io, parent);
    std.Io.Dir.cwd().deleteFile(io, link_path) catch {};
    const target = try std.fs.path.join(a, &.{ "..", "..", "blobs", blob_name });
    std.Io.Dir.cwd().symLink(io, target, link_path, .{}) catch return error.SymLinkFailed;
}
