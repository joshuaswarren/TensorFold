//! Host tests for `pull`: a fake hub on 127.0.0.1 with range support, a proxy and a stall.
const std = @import("std");
const Allocator = std.mem.Allocator;
const pull = @import("pull.zig");

/// A fake hub on 127.0.0.1: the revision and tree APIs, and resolve endpoints with range support.
const FakeHub = struct {
    io: std.Io,
    server_fd: std.posix.socket_t,
    stop: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    weight_requests: std.atomic.Value(u32) = .init(0),
    range_starts: std.atomic.Value(u64) = .init(0),
    weights: []const u8,
    config: []const u8,
    weights_sha_hex: []const u8,
    port: u16,
    stall: bool = false,
    active_handlers: std.atomic.Value(u32) = .init(0),
    log_mutex: std.atomic.Mutex = .unlocked,
    log: [16][]const u8 = @splat(""),
    log_buf: [16][512]u8 = @splat(@splat(0)),
    log_len: usize = 0,

    const weights_path = "/Org/Flash/resolve/rev1sha/weights.safetensors";
    const config_path = "/Org/Flash/resolve/rev1sha/config.json";

    fn open() !FakeHub {
        const posix = std.posix;
        const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
        if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
        const fd: posix.socket_t = @intCast(rc);
        errdefer _ = posix.system.close(fd);
        const one: c_int = 1;
        try posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.REUSEADDR, std.mem.asBytes(&one));
        var addr: posix.sockaddr.in = .{ .family = posix.AF.INET, .port = std.mem.nativeToBig(u16, 0), .addr = std.mem.nativeToBig(u32, 0x7f000001) };
        if (posix.errno(posix.system.bind(fd, @ptrCast(&addr), @sizeOf(posix.sockaddr.in))) != .SUCCESS) return error.BindFailed;
        if (posix.errno(posix.system.listen(fd, 8)) != .SUCCESS) return error.ListenFailed;
        var bound: posix.sockaddr.in = undefined;
        var len: posix.socklen_t = @sizeOf(posix.sockaddr.in);
        if (posix.errno(posix.system.getsockname(fd, @ptrCast(&bound), &len)) != .SUCCESS) return error.BindFailed;
        const port = std.mem.bigToNative(u16, bound.port);
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update(weights_body);
        var digest: [32]u8 = undefined;
        hash.final(&digest);
        return .{
            .io = std.testing.io,
            .server_fd = fd,
            .weights = weights_body,
            .config = config_body,
            .weights_sha_hex = try std.fmt.allocPrint(std.testing.allocator, "{s}", .{std.fmt.bytesToHex(digest, .lower)}),
            .port = port,
        };
    }

    const config_body = "{\"model_type\": \"qwen4_exp\", \"quantization\": {\"bits\": 6, \"group_size\": 32}, \"max_position_embeddings\": 262144}";
    var weights_body_storage: [5000]u8 = @splat(0xA5);
    const weights_body: []const u8 = &weights_body_storage;

    fn serve(fake: *FakeHub) void {
        while (!fake.stop.load(.acquire)) {
            var fds = [_]std.posix.pollfd{.{ .fd = fake.server_fd, .events = std.posix.POLL.IN, .revents = 0 }};
            const ready = std.posix.poll(&fds, 50) catch return;
            if (ready == 0) continue;
            const rc = std.posix.system.accept(fake.server_fd, null, null);
            if (std.posix.errno(rc) != .SUCCESS) return;
            const conn_fd: std.posix.socket_t = @intCast(rc);
            // Recv timeout lets detached handler threads notice `stop` while parked in read on a keep-alive connection.
            const rcvtimeo = std.posix.timeval{ .sec = 0, .usec = 250 * std.time.us_per_ms };
            _ = std.posix.setsockopt(conn_fd, std.posix.SOL.SOCKET, std.posix.SO.RCVTIMEO, std.mem.asBytes(&rcvtimeo)) catch {};
            // One thread per connection: the download opens a second connection while the first stays keep-alive.
            _ = fake.active_handlers.fetchAdd(1, .acq_rel);
            // The count goes up before the spawn so stopServer cannot free the fake under a live handler.
            const t = std.Thread.spawn(.{}, handle, .{ fake, conn_fd }) catch {
                _ = std.posix.system.close(conn_fd);
                _ = fake.active_handlers.fetchSub(1, .acq_rel);
                continue;
            };
            t.detach();
        }
    }

    fn handle(fake: *FakeHub, fd: std.posix.socket_t) void {
        defer {
            _ = std.posix.system.close(fd);
            _ = fake.active_handlers.fetchSub(1, .acq_rel);
        }
        if (fake.stall) {
            const ts: std.c.timespec = .{ .sec = 0, .nsec = 10 * std.time.ns_per_ms };
            while (!fake.stop.load(.acquire)) _ = std.c.nanosleep(&ts, null);
            return;
        }
        var buf: [8192]u8 = undefined;
        var end: usize = 0;
        while (true) {
            const head_end = std.mem.indexOf(u8, buf[0..end], "\r\n\r\n") orelse {
                if (end == buf.len) return;
                const n = std.posix.read(fd, buf[end..]) catch |e| switch (e) {
                    error.WouldBlock => if (fake.stop.load(.acquire)) return else continue,
                    else => return,
                };
                if (n == 0) return;
                end += n;
                continue;
            };
            fake.respondOne(fd, buf[0..head_end]);
            const rest = end - (head_end + 4);
            std.mem.copyForwards(u8, buf[0..rest], buf[head_end + 4 .. end]);
            end = rest;
        }
    }

    fn respondOne(fake: *FakeHub, fd: std.posix.socket_t, head: []const u8) void {
        {
            while (!fake.log_mutex.tryLock()) {}
            defer fake.log_mutex.unlock();
            if (fake.log_len < fake.log.len) {
                const line_end_i = std.mem.indexOfScalar(u8, head, '\r') orelse head.len;
                const n = @min(line_end_i, 512);
                @memcpy(fake.log_buf[fake.log_len][0..n], head[0..n]);
                fake.log[fake.log_len] = fake.log_buf[fake.log_len][0..n];
                fake.log_len += 1;
            }
        }
        const line_end = std.mem.indexOf(u8, head, "\r\n") orelse return;
        var parts = std.mem.tokenizeScalar(u8, head[0..line_end], ' ');
        const method = parts.next() orelse return;
        // A proxy tunnel: acknowledge CONNECT so the tunneled requests can follow.
        if (std.mem.eql(u8, method, "CONNECT")) {
            sendAll(fd, "HTTP/1.1 200 Connection established\r\nContent-Length: 0\r\n\r\n");
            return;
        }
        if (!std.mem.eql(u8, method, "GET")) return;
        const raw_path = parts.next() orelse return;
        var path = if (std.mem.indexOfScalar(u8, raw_path, '?')) |q| raw_path[0..q] else raw_path;
        // A plain http proxy receives the request-target as an absolute URI.
        if (std.mem.indexOf(u8, path, "://")) |scheme_end| {
            const rest = path[scheme_end + 3 ..];
            path = rest[std.mem.indexOfScalar(u8, rest, '/') orelse return ..];
        }
        var range_start: ?u64 = null;
        var it = std.mem.splitSequence(u8, head[line_end + 2 ..], "\r\n");
        while (it.next()) |h| {
            const colon = std.mem.indexOfScalar(u8, h, ':') orelse continue;
            if (std.ascii.eqlIgnoreCase(std.mem.trim(u8, h[0..colon], " "), "range")) {
                const value = std.mem.trim(u8, h[colon + 1 ..], " ");
                if (std.mem.startsWith(u8, value, "bytes=")) {
                    const spec = std.mem.trimStart(u8, value[6..], " ");
                    const dash = std.mem.indexOfScalar(u8, spec, '-') orelse spec.len;
                    range_start = std.fmt.parseInt(u64, spec[0..dash], 10) catch null;
                }
            }
        }
        if (std.mem.eql(u8, path, "/api/models/Org/Flash/revision/main")) {
            respond(fd, "{\"sha\": \"rev1sha\"}");
        } else if (std.mem.eql(u8, path, "/api/models/Org/Draft/revision/main")) {
            respond(fd, "{\"sha\": \"draftsha\"}");
        } else if (std.mem.eql(u8, path, "/api/models/Org/Draft/tree/draftsha")) {
            respond(fd, "[{\"type\": \"file\", \"path\": \"config.json\", \"size\": 27}]");
        } else if (std.mem.eql(u8, path, "/Org/Draft/resolve/draftsha/config.json")) {
            respond(fd, "{\"model_type\": \"gemma4\"}");
        } else if (std.mem.eql(u8, path, "/api/models/Org/Flash/tree/rev1sha")) {
            var tree_buf: [512]u8 = undefined;
            const tree = std.fmt.bufPrint(&tree_buf, "[{{\"type\": \"file\", \"path\": \"config.json\", \"size\": {d}}}, {{\"type\": \"file\", \"path\": \"weights.safetensors\", \"size\": {d}, \"lfs\": {{\"oid\": \"sha256:{s}\", \"size\": {d}}}}}]", .{ fake.config.len, fake.weights.len, fake.weights_sha_hex, fake.weights.len }) catch return;
            respond(fd, tree);
        } else if (std.mem.eql(u8, path, config_path)) {
            respond(fd, fake.config);
        } else if (std.mem.eql(u8, path, weights_path)) {
            _ = fake.weight_requests.fetchAdd(1, .monotonic);
            if (range_start) |from| {
                fake.range_starts.store(from, .monotonic);
                if (from >= fake.weights.len) {
                    respond(fd, "");
                    return;
                }
                const body = fake.weights[from..];
                var head_buf: [256]u8 = undefined;
                const head_text = std.fmt.bufPrint(&head_buf, "HTTP/1.1 206 Partial Content\r\nContent-Type: application/octet-stream\r\nContent-Length: {d}\r\nContent-Range: bytes {d}-{d}/{d}\r\nConnection: keep-alive\r\n\r\n", .{ body.len, from, fake.weights.len - 1, fake.weights.len }) catch return;
                sendAll(fd, head_text);
                sendAll(fd, body);
            } else {
                var head_buf: [128]u8 = undefined;
                const head_text = std.fmt.bufPrint(&head_buf, "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {d}\r\nConnection: keep-alive\r\n\r\n", .{fake.weights.len}) catch return;
                sendAll(fd, head_text);
                sendAll(fd, fake.weights);
            }
        } else {
            respond(fd, "{}");
        }
    }

    fn respond(fd: std.posix.socket_t, body: []const u8) void {
        var head_buf: [128]u8 = undefined;
        const head_text = std.fmt.bufPrint(&head_buf, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: keep-alive\r\n\r\n", .{body.len}) catch return;
        sendAll(fd, head_text);
        sendAll(fd, body);
    }

    fn sendAll(fd: std.posix.socket_t, bytes: []const u8) void {
        var sent: usize = 0;
        while (sent < bytes.len) {
            // MSG_NOSIGNAL: a client that closed mid-answer must not kill the test process with SIGPIPE.
            const n = std.posix.system.send(fd, bytes[sent..].ptr, bytes.len - sent, std.posix.MSG.NOSIGNAL);
            if (std.posix.errno(n) != .SUCCESS) return;
            sent += @intCast(n);
        }
    }

    fn start(fake: *FakeHub) void {
        fake.thread = std.Thread.spawn(.{}, serve, .{fake}) catch null;
    }

    fn endpoint(fake: *FakeHub, a: Allocator) ![]const u8 {
        return std.fmt.allocPrint(a, "http://127.0.0.1:{d}", .{fake.port});
    }

    fn stopServer(fake: *FakeHub) void {
        fake.stop.store(true, .release);
        if (fake.thread) |t| t.join();
        _ = std.posix.system.close(fake.server_fd);
        // Handler threads are detached: wait for them to leave so no thread touches this fake after stop returns.
        const ts: std.c.timespec = .{ .sec = 0, .nsec = 10 * std.time.ns_per_ms };
        while (fake.active_handlers.load(.acquire) != 0) _ = std.c.nanosleep(&ts, null);
        std.testing.allocator.free(fake.weights_sha_hex);
    }
};

fn testEnv(a: Allocator, endpoint_url: []const u8) !std.process.Environ.Map {
    var env: std.process.Environ.Map = .{ .array_hash_map = .empty, .allocator = a };
    try env.put("HF_ENDPOINT", endpoint_url);
    return env;
}

test "pull routes through the proxy environment, honours no_proxy and times out on a stall" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var fake_hub = FakeHub.open() catch {
        return error.SkipZigTest;
    };
    fake_hub.start();
    defer fake_hub.stopServer();
    const root = try std.fmt.allocPrint(a, ".tf-pull-proxy-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};

    // The endpoint is dead; only the proxy can answer. CONNECT lines for 127.0.0.1:1 prove it tunneled.
    var envt = try testEnv(a, "http://127.0.0.1:1");
    try envt.put("TENSORFOLD_PULL_STALL_MS", "2000");
    try envt.put("HTTP_PROXY", try fake_hub.endpoint(a));
    var out: std.Io.Writer.Allocating = .init(a);
    var err_out: std.Io.Writer.Allocating = .init(a);
    const rc1 = try pull.run(a, io, &out.writer, &err_out.writer, &envt, root, "Org/Flash");
    try std.testing.expectEqual(@as(u8, 0), rc1);
    try std.testing.expectEqual(@as(u32, 1), fake_hub.weight_requests.load(.monotonic));
    var saw_connect = false;
    for (0..fake_hub.log_len) |i| {
        if (std.mem.indexOf(u8, fake_hub.log[i], "CONNECT 127.0.0.1:1 ") != null) saw_connect = true;
    }
    try std.testing.expect(saw_connect);

    // no_proxy covers the endpoint: a dead proxy must not be used.
    var env2 = try testEnv(a, try fake_hub.endpoint(a));
    try env2.put("HTTP_PROXY", "http://127.0.0.1:1");
    try env2.put("NO_PROXY", "127.0.0.1");
    var out2: std.Io.Writer.Allocating = .init(a);
    err_out.writer.end = 0;
    const rc2 = try pull.run(a, io, &out2.writer, &err_out.writer, &env2, root, "Org/Flash");
    try std.testing.expectEqual(@as(u8, 0), rc2);

    // A hub that accepts and never answers fails instead of hanging.
    var stalled_hub = FakeHub.open() catch {
        return error.SkipZigTest;
    };
    stalled_hub.stall = true;
    stalled_hub.start();
    defer stalled_hub.stopServer();
    var env3 = try testEnv(a, try stalled_hub.endpoint(a));
    try env3.put("TENSORFOLD_PULL_STALL_MS", "300");
    const started = std.Io.Clock.awake.now(io);
    var err3: std.Io.Writer.Allocating = .init(a);
    var out3: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 1), try pull.run(a, io, &out3.writer, &err3.writer, &env3, root, "Org/Flash"));
    const waited_ms = started.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds();
    try std.testing.expect(waited_ms < 20_000);
    try std.testing.expect(std.mem.indexOf(u8, err3.written(), "timed out") != null or
        std.mem.indexOf(u8, err3.written(), "no response") != null or
        std.mem.indexOf(u8, err3.written(), "no progress") != null);
}

test "pull downloads, verifies, resumes and refuses a family Zig cannot serve" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var fake_hub = FakeHub.open() catch {
        return error.SkipZigTest;
    };
    fake_hub.start();
    defer fake_hub.stopServer();
    const root = try std.fmt.allocPrint(a, ".tf-pull-test-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    const env = try testEnv(a, try fake_hub.endpoint(a));

    var out: std.Io.Writer.Allocating = .init(a);
    var err_out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try pull.run(a, io, &out.writer, &err_out.writer, &env, root, "Org/Flash"));
    const snapshot = try std.fs.path.join(a, &.{ root, "models--Org--Flash/snapshots/rev1sha" });
    const config_link = try std.fs.path.join(a, &.{ snapshot, "config.json" });
    const weights_link = try std.fs.path.join(a, &.{ snapshot, "weights.safetensors" });
    const config_read = try std.Io.Dir.cwd().readFileAlloc(io, config_link, a, .limited(1 << 20));
    try std.testing.expectEqualStrings(FakeHub.config_body, config_read);
    const weights_read = try std.Io.Dir.cwd().readFileAlloc(io, weights_link, a, .limited(1 << 20));
    try std.testing.expectEqualSlices(u8, FakeHub.weights_body, weights_read);
    const ref = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ root, "models--Org--Flash/refs/main" }), a, .limited(64));
    try std.testing.expectEqualStrings("rev1sha", std.mem.trim(u8, ref, " \r\n"));
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "weights.safetensors") != null);
    try std.testing.expectEqual(@as(u32, 1), fake_hub.weight_requests.load(.monotonic));

    // A second pull skips the verified blob and touches the weights endpoint no more.
    var out2: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try pull.run(a, io, &out2.writer, &err_out.writer, &env, root, "Org/Flash"));
    try std.testing.expectEqual(@as(u32, 1), fake_hub.weight_requests.load(.monotonic));
    try std.testing.expect(std.mem.indexOf(u8, out2.written(), "weights.safetensors: cached") != null);

    // A partial blob resumes from its own length.
    const blob_path = try std.fs.path.join(a, &.{ root, "models--Org--Flash/blobs", fake_hub.weights_sha_hex });
    std.Io.Dir.cwd().deleteFile(io, blob_path) catch {};
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}.incomplete", .{blob_path}), .data = FakeHub.weights_body[0..2000] });
    var out3: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try pull.run(a, io, &out3.writer, &err_out.writer, &env, root, "Org/Flash"));
    try std.testing.expectEqual(@as(u64, 2000), fake_hub.range_starts.load(.monotonic));
    try std.testing.expectEqual(@as(u32, 2), fake_hub.weight_requests.load(.monotonic));
    const resumed = try std.Io.Dir.cwd().readFileAlloc(io, weights_link, a, .limited(1 << 20));
    try std.testing.expectEqualSlices(u8, FakeHub.weights_body, resumed);

    // A family Zig cannot serve is refused with the 0.6 line, after the config check only.
    var refused_out: std.Io.Writer.Allocating = .init(a);
    var refused_err: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 1), try pull.run(a, io, &refused_out.writer, &refused_err.writer, &env, root, "Org/Draft"));
    try std.testing.expect(std.mem.indexOf(u8, refused_err.written(), "model_type gemma4") != null);
    try std.testing.expect(std.mem.indexOf(u8, refused_err.written(), "tensorfold@0.6") != null);

    // A bad repo id is usage.
    var bad_err: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 2), try pull.run(a, io, &out.writer, &bad_err.writer, &env, root, "no-slash"));
}
