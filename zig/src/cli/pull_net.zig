//! `pull`'s network policy: the proxy environment, a stall watchdog and bounded resume knobs.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// Milliseconds without forward progress before a hub call is dropped (env: TENSORFOLD_PULL_STALL_MS; 0 disables).
pub const default_stall_ms: u64 = 30_000;
/// Resume attempts for one weight file after its download stalls (env: TENSORFOLD_PULL_RETRIES).
pub const default_retries: u32 = 3;

/// A whole number from the environment, or the default when it is unset or unreadable.
pub fn envInt(env: ?*const std.process.Environ.Map, key: []const u8, default: u64) u64 {
    const m = env orelse return default;
    const v = m.get(key) orelse return default;
    return std.fmt.parseInt(u64, std.mem.trim(u8, v, " \t"), 10) catch default;
}

/// Installs the environment's proxies on the client, unless no_proxy covers `host:port`.
pub fn applyProxy(a: Allocator, client: *std.http.Client, env: ?*const std.process.Environ.Map, host: []const u8, port: u16) !void {
    const m = env orelse return;
    if (noProxy(m, host, port)) return;
    try client.initDefaultProxies(a, m);
}

/// no_proxy/NO_PROXY: "*", or comma-separated hosts and .suffixes, each with an optional :port.
pub fn noProxy(env: *const std.process.Environ.Map, host: []const u8, port: u16) bool {
    const list = env.get("no_proxy") orelse env.get("NO_PROXY") orelse return false;
    var hbuf: [256]u8 = undefined;
    if (host.len > hbuf.len) return false;
    const h = std.ascii.lowerString(&hbuf, host);
    var it = std.mem.tokenizeScalar(u8, list, ',');
    while (it.next()) |raw| {
        var e = std.mem.trim(u8, raw, " \t");
        if (e.len == 0) continue;
        if (std.mem.eql(u8, e, "*")) return true;
        if (std.mem.lastIndexOfScalar(u8, e, ':')) |colon| {
            if (std.fmt.parseInt(u16, e[colon + 1 ..], 10)) |p| {
                if (p != port) continue;
                e = e[0..colon];
            } else |_| {}
        }
        var ebuf: [256]u8 = undefined;
        if (e.len == 0 or e.len > ebuf.len) continue;
        const suffix = std.mem.trimStart(u8, std.ascii.lowerString(&ebuf, e), ".");
        if (suffix.len == 0) continue;
        if (std.mem.eql(u8, h, suffix)) return true;
        if (h.len > suffix.len and std.mem.endsWith(u8, h, suffix) and h[h.len - suffix.len - 1] == '.') return true;
    }
    return false;
}

/// A byte moved, a header landed: forward progress the watchdog counts.
pub const Guard = struct {
    progress: std.atomic.Value(u64) = .init(0),

    pub fn bump(g: *Guard) void {
        _ = g.progress.fetchAdd(1, .monotonic);
    }
};

const Job = struct {
    context: *anyopaque,
    runFn: *const fn (context: *anyopaque, guard: *Guard) anyerror!void,
    guard: Guard = .{},
    done: std.atomic.Value(bool) = .init(false),
    err: ?anyerror = null,

    fn run(j: *Job) void {
        j.runFn(j.context, &j.guard) catch |e| {
            j.err = e;
        };
        j.done.store(true, .release);
    }
};

/// Runs `runFn(context, guard)` concurrently; `error.PullStalled` after one quiet `stall_ms` (0 disables).
pub fn guarded(io: std.Io, stall_ms: u64, context: *anyopaque, comptime runFn: fn (context: *anyopaque, guard: *Guard) anyerror!void) !void {
    var job: Job = .{ .context = context, .runFn = runFn };
    if (stall_ms == 0) return runFn(context, &job.guard);
    var work = io.concurrent(Job.run, .{&job}) catch |e| switch (e) {
        // No unit to cancel a blocked call on: run without the watchdog.
        error.ConcurrencyUnavailable => return runFn(context, &job.guard),
    };
    const stall_ns: i96 = @as(i96, @intCast(@min(stall_ms, @as(u64, 1) << 62))) * std.time.ns_per_ms;
    const poll_ms: i64 = @intCast(@min(stall_ms, 50));
    var last: u64 = 0;
    var quiet = std.Io.Clock.awake.now(io);
    while (true) {
        std.Io.sleep(io, .fromMilliseconds(poll_ms), .awake) catch |e| {
            _ = work.cancel(io);
            return e;
        };
        if (job.done.load(.acquire)) break;
        const now = std.Io.Clock.awake.now(io);
        const p = job.guard.progress.load(.monotonic);
        if (p != last) {
            last = p;
            quiet = now;
            continue;
        }
        if (quiet.durationTo(now).toNanoseconds() >= stall_ns) {
            _ = work.cancel(io);
            return error.PullStalled;
        }
    }
    _ = work.await(io);
    if (job.err) |e| return e;
}
