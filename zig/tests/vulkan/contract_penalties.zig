//! The penalize kernel's Vulkan contract: presence/frequency penalization over
//! one bf16 logits row (zig/kernels/vulkan/penalize.comp, the CUDA
//! tf_penalize_rows primitive on the single-sequence engine) through the kernel
//! contract's gates 1-4. No device code here — contract_penalties_main.zig
//! drives the real GPU; this file holds the rig, the hooks, the f64 reference,
//! the spec and the host proofs.
//!
//! Spec choices, and why:
//!  - Served shapes: rows = the staged entry count, n = the vocab row width
//!    (248320, config.zig), k unused. The engine serves ONE logits row of
//!    248320 words and stages 1 entry (the first pending token) up to
//!    max_entries = state.max_capacity + 1 = 7169 (penalties.zig). The tile
//!    edges are the dispatch-group boundary: 255, 256 and 257 entries at 256
//!    threads a group (groups = (n+255)/256). The largest served count is the
//!    engine's structural max, not a device limit — 28 groups — and the
//!    workgroup-count bound is asserted on the host against the 65535 the
//!    Vulkan spec guarantees for maxComputeWorkGroupCount[0], then enforced per
//!    device by dispatch.Pipeline.init (as in the greedy contract).
//!  - NaN rule (stated, the shader's exact arithmetic): an entry widens its
//!    logit exactly to fp32 (bits << 16), subtracts the entry's amount once in
//!    fp32 and rounds once to bf16 RNE — so an infinite logit keeps its
//!    infinity (inf - finite), a NaN logit or amount stays NaN, exactly the
//!    f64 reference's classes; every element no entry names is bit-identical
//!    to its input.
//!  - Entries rule: ONE entry per distinct reply token. penalties.zig's
//!    stage() counts tokens in a map and emits one (token, amount) pair per
//!    distinct token, so no two threads ever write the same bf16 halfword and
//!    the shader needs no atomics. The contract feeds distinct tokens (its
//!    generator draws distinct even tokens; odd neighbors are never entries),
//!    stages them the way the engine does (stageEntries below), and its
//!    checker refuses a duplicate list. Every staged amount provably moves
//!    its logit's bf16 word (informativeAmount): an entry whose penalty
//!    rounds back to its input carries no evidence, and one such entry was
//!    exactly how the tail control — whose only informative token is the last
//!    — went green on a real device. The controls run the 255/256/257
//!    dispatch-edge shapes, and their host semantics are asserted rejected at
//!    every shape, so a control that cannot go red fails at build time.
//!  - Depth 0, accum none, in_place: one exact widening, one fp32 subtract,
//!    one RNE store that rewrites the logits row — the outputs are compared
//!    bit for bit (the checker hook), never inside a rounding budget.
//!  - The fp32-denormal row (0x0080 minus 2^-133: CUDA 0x007f) is a
//!    value-level case, not a shape: like the pointwise preserve-add's
//!    subnormal rows it is its own gate in the runner, where a flush reads as
//!    a KNOWN finding exactly when every mismatch is a named token read as its
//!    unchanged input AND the device reports shaderDenormPreserveFloat32 = 0.
//!    Otherwise the gate FAILs — the reference is never bent to a flush.
const std = @import("std");
const vk = @import("vulkan");
const contract = @import("contract");
const shaders = @import("penalties_contract_shaders");

const c = contract.spec;
const g = contract.gates;

/// The engine's vocab row (qwen3_5 config.zig) and its staged-entry ceiling
/// (penalties.zig max_entries = state.zig max_capacity + 1).
pub const vocab: usize = 248320;
pub const max_entries: u32 = 7169;

/// 256 threads a workgroup; groups = ceil(entries / 256).
pub const local_size: u32 = 256;
/// Guard bytes on both sides of every rig payload: the largest
/// minStorageBufferOffsetAlignment Vulkan allows, so the data windows open on
/// a legal storage offset.
pub const guard: usize = 256;
const pattern: u8 = 0xa5;
const guard_words: usize = guard / 2;

/// The amounts' byte offset inside a `cap`-entry table: penalties.zig's own
/// pure function (toks bytes rounded up to 256).
pub fn amountsAt(cap: u32) u64 {
    return (@as(u64, cap) * 4 + 255) / 256 * 256;
}
pub fn groups(n: u32) u32 {
    return (n + local_size - 1) / local_size;
}
pub fn logitsBytes(count: usize) usize {
    return 2 * guard + 2 * count;
}
pub fn tableBytes(n: u32) usize {
    return 2 * guard + @as(usize, @intCast(amountsAt(n) + @as(u64, n) * 4));
}

// ---------------------------------------------------------------------------
// The spec
// ---------------------------------------------------------------------------

fn flat(_: c.Shape) usize {
    return 0;
}

pub const nan_rule = "an entry widens its logit exactly to fp32, subtracts the amount once and rounds once to bf16 RNE: " ++
    "an infinite logit stays infinite (inf minus finite), a NaN logit or amount stays NaN, the f64 reference's classes; " ++
    "every element no entry names is bit-identical to its input";

pub const spec: c.Spec = .{
    .name = "vulkan penalize bf16 (one fp32 subtract, one RNE store)",
    .accum = .none,
    .depth = flat,
    .out = .bf16,
    .in_place = true,
    .nan = .{ .stated = nan_rule },
    .tiles = .{ .{ .size = local_size, .step = 1, .min = 1 }, .{}, .{} },
    .served = &.{
        .{ .rows = 1, .n = vocab, .k = 0 },
        .{ .rows = max_entries, .n = vocab, .k = 0 },
    },
};

const control_served = [_]c.Shape{
    .{ .rows = 255, .n = vocab, .k = 0 },
    .{ .rows = 256, .n = vocab, .k = 0 },
    .{ .rows = 257, .n = vocab, .k = 0 },
};

/// The four planted controls and the gate each must go red at.
pub const Kind = enum { double, bf16_sub, scribble, tail };

pub fn controlName(kind: Kind) []const u8 {
    return switch (kind) {
        .double => "control: penalize applies each entry twice",
        .bf16_sub => "control: penalize subtracts in bf16",
        .scribble => "control: penalize scribbles the word beside its token",
        .tail => "control: penalize drops the last entry",
    };
}

pub fn controlGate(kind: Kind) g.Gate {
    return switch (kind) {
        .double, .bf16_sub, .tail => .shapes,
        .scribble => .repeat,
    };
}

fn controlSpec(kind: Kind) c.Spec {
    return .{
        .name = controlName(kind),
        .accum = .none,
        .depth = flat,
        .out = .bf16,
        .in_place = true,
        .nan = .{ .stated = nan_rule },
        .served = &control_served,
    };
}

// ---------------------------------------------------------------------------
// The f64 reference and the checkers
// ---------------------------------------------------------------------------

/// The CUDA formula in f64: exact bf16 widening (a shift), one fp32 subtract
/// (the exact difference rounded to f32 — @floatCast rounds RNE as the f32
/// subtract does), one bf16 RNE store.
pub fn wantBits(logit_bits: u16, amount: f32) u16 {
    const diff: f64 = @as(f64, c.fromBf16(logit_bits)) - @as(f64, amount);
    return c.toBf16(@as(f32, @floatCast(diff)));
}

/// The double control's arithmetic: each subtract rounds to f32 (the shader's
/// `(x - subs[i]) - subs[i]`), one RNE store.
pub fn doubleBits(logit_bits: u16, amount: f32) u16 {
    const x: f64 = c.fromBf16(logit_bits);
    const once: f64 = @as(f32, @floatCast(x - amount));
    return c.toBf16(@as(f32, @floatCast(once - amount)));
}

/// The bf16-subtract control's arithmetic: both operands rounded to bf16
/// before the f32 subtract (the shader's bf16_round pair).
pub fn bf16SubBits(logit_bits: u16, amount: f32) u16 {
    const x = c.toBf16(@as(f32, @floatCast(c.fromBf16(logit_bits))));
    const s = c.toBf16(amount);
    return c.toBf16(@as(f32, @floatCast(c.fromBf16(x) - c.fromBf16(s))));
}

pub const Fault = struct { token: usize, got: u16, want: u16 };

fn classWord(bits: u16) []const u8 {
    return switch (c.classOf(c.fromBf16(bits))) {
        .finite => "finite",
        .nan => "nan",
        .pos_inf => "+inf",
        .neg_inf => "-inf",
    };
}

/// The word-level verdict behind check(): a named token's finite want is
/// bit-exact, its non-finite want is class-equal (a NaN payload is the
/// hardware's); a token no entry names must be bit-identical to its input.
pub fn firstFault(got: []const u16, want: []const u16, input: []const u16, named: []const bool) ?Fault {
    for (got, want, input, named, 0..) |gv, w, iv, is_entry, t| {
        const bad = if (!is_entry)
            gv != iv
        else if (std.math.isFinite(c.fromBf16(w)))
            gv != w
        else
            c.classOf(c.fromBf16(gv)) != c.classOf(c.fromBf16(w));
        if (bad) return .{ .token = t, .got = gv, .want = w };
    }
    return null;
}

/// The denormal row's verdict: pass when the row is bit-exact; KNOWN-shaped
/// only when every mismatch is a named token read as its UNCHANGED input (the
/// flushed addend changed nothing) on a device reporting no fp32 denormal
/// preservation; anything else fails.
pub const Verdict = enum { pass, known, fail };

/// The run's accounting, pure so a host test pins it: the items are the kernel's gates (one per kernel), every
/// planted control (it counts as passed when it went red as planted) and the denormal row; `known` is the denormal
/// row's KNOWN verdict. The run passes when no kernel failed, no control was missed and every item passed or is known.
pub const Accounting = struct { passed: usize, known: usize, total: usize, ok: bool };
pub fn accounting(t: g.Tally, sub: Verdict, refused: bool) Accounting {
    const passed = t.passed + t.caught + @intFromBool(sub == .pass);
    const known = t.known + @intFromBool(sub == .known);
    const total = t.kernels + t.controls + 1;
    return .{ .passed = passed, .known = known, .total = total, .ok = !refused and t.problems() == 0 and t.caught == t.controls and passed + known == total };
}

pub fn subnormalVerdict(got: []const u16, want: []const u16, input: []const u16, preserve32: bool) Verdict {
    var clean = true;
    var flushed_only = true;
    for (got, want, input) |gv, w, iv| {
        if (gv == w) continue;
        clean = false;
        if (gv != iv) flushed_only = false;
    }
    if (clean) return .pass;
    if (flushed_only and !preserve32) return .known;
    return .fail;
}

// ---------------------------------------------------------------------------
// The staging model and the launch geometry
// ---------------------------------------------------------------------------

pub const Entry = struct { tok: u32, sub: f32 };

/// The engine's staging (penalties.zig stage/bump): one entry per DISTINCT
/// token, in ascending token order, the amount the engine's pen.of(count)
/// gives. A duplicate list is refused: the engine's map makes duplicates
/// unreachable, and the contract's fixture refuses them at the same place.
pub fn stageEntries(a: std.mem.Allocator, tokens: []const u32, presence: f32, frequency: f32) ![]Entry {
    var counts = std.AutoArrayHashMapUnmanaged(u32, u32).empty;
    defer counts.deinit(a);
    for (tokens) |t| {
        if (t >= vocab) return error.OutOfRange;
        const gop = try counts.getOrPut(a, t);
        if (!gop.found_existing) gop.value_ptr.* = 0;
        gop.value_ptr.* += 1;
    }
    const keys = try a.dupe(u32, counts.keys());
    defer a.free(keys);
    std.mem.sort(u32, keys, {}, std.sort.asc(u32));
    const out = try a.alloc(Entry, keys.len);
    for (keys, out) |t, *e| e.* = .{ .tok = t, .sub = presence + frequency * @as(f32, @floatFromInt(counts.get(t).?)) };
    return out;
}

/// The engine refuses an out-of-vocabulary token before anything stages
/// (penalties.zig's InvalidQwenToken); the contract declares the same bound.
pub fn tokenInRange(tok: u32) bool {
    return tok < vocab;
}

/// The launch geometry one prepare allocates and slices against: the pure
/// plan build() rides, so the host suite can prove every shape the spec
/// generates uploads inside its own allocation (the out-of-range-slice class
/// of bug).
pub const Plan = struct {
    shell_words: usize,
    table_bytes: usize,
    amounts_at: u64,
    toks_off: u64,
    toks_len: u64,
    subs_off: u64,
    subs_len: u64,
    groups: u32,
};

pub fn planFor(shape: c.Shape) !Plan {
    if (shape.n != vocab or shape.rows == 0 or shape.rows > max_entries) return error.Invalid;
    const n: u32 = @intCast(shape.rows);
    return .{
        .shell_words = guard_words + shape.n + guard_words,
        .table_bytes = tableBytes(n),
        .amounts_at = amountsAt(n),
        .toks_off = guard,
        .toks_len = @as(u64, n) * 4,
        .subs_off = guard + amountsAt(n),
        .subs_len = @as(u64, n) * 4,
        .groups = groups(n),
    };
}

// ---------------------------------------------------------------------------
// The input families: normal-range words, so no random row can trip a device
// denormal flush where the f64 reference does not (the denormal row is its
// own gate, not an accident of the fill)
// ---------------------------------------------------------------------------

/// A bf16 normal with |x| in [2^-60, 2^60]: two bf16 words that far apart
/// subtract to a normal fp32 or to zero, never to a subnormal an FTZ device
/// would flush.
fn randomLogit(rng: std.Random) u16 {
    const sign: u16 = rng.int(u1);
    const exp: u16 = 67 + rng.uintLessThan(u8, 121);
    return sign << 15 | exp << 7 | rng.int(u7);
}

fn randomAmount(rng: std.Random) f32 {
    const sign: f32 = if (rng.boolean()) -1 else 1;
    const exp: i32 = @as(i32, 67 + rng.uintLessThan(u8, 121)) - 127;
    return sign * std.math.pow(f32, 2, @floatFromInt(exp)) * (1 + rng.float(f32));
}

/// An amount that provably moves its logit's bf16 word. A plain random draw
/// mostly does not (an amount below half the logit's bf16 ulp rounds straight
/// back), and an entry whose penalty rounds to a no-op carries no evidence:
/// the tail control's ONLY informative token is the last entry, so one no-op
/// there silences the control (the missed control on the M2 Max run). The
/// fallback subtracts the logit's own magnitude, which always leaves the word.
fn informativeAmount(rng: std.Random, logit_bits: u16) f32 {
    var s = randomAmount(rng);
    var tries: usize = 0;
    while (wantBits(logit_bits, s) == logit_bits) {
        s = randomAmount(rng);
        tries += 1;
        if (tries == 10_000) return @abs(c.fromBf16(logit_bits));
    }
    return s;
}

/// Distinct even tokens in [0, vocab): the contract's entries never collide
/// on a halfword, and every odd neighbor is an untouched word.
fn distinctTokens(a: std.mem.Allocator, rng: std.Random, n: u32) ![]u32 {
    const out = try a.alloc(u32, n);
    var seen = std.AutoHashMap(u32, void).init(a);
    defer seen.deinit();
    var i: usize = 0;
    while (i < out.len) {
        const tok = 2 * rng.uintLessThan(u32, @intCast(vocab / 2));
        if (seen.contains(tok)) continue;
        try seen.put(tok, {});
        out[i] = tok;
        i += 1;
    }
    return out;
}

/// The nonfinite fill: NaN and both infinities at three entry tokens and in
/// three amounts, plus the three classes at odd words no entry names (they
/// must survive bit-identical).
fn plantNonfinite(toks: []const u32, subs: []f32, input: []u16) void {
    subs[0] = std.math.nan(f32);
    subs[1] = std.math.inf(f32);
    subs[2] = -std.math.inf(f32);
    input[toks[0]] = 0x7fc0;
    input[toks[1]] = 0x7f80;
    input[toks[2]] = 0xff80;
    input[input.len - 1] = 0x7fc0;
    input[input.len - 3] = 0x7f80;
    input[input.len - 5] = 0xff80;
}

/// The denormal row: the strict CUDA edge — smallest bf16 normal minus an
/// fp32-only subnormal is a bf16 SUBNORMAL (0x007f); a device that flushes
/// fp32 denormals computes the unpenalized 0x0080.
pub const subnormal_token: usize = 5;
pub const subnormal_logit: u16 = 0x0080;
pub const subnormal_amount: f32 = std.math.pow(f32, 2, -133);

// ---------------------------------------------------------------------------
// The rig and the fixture
// ---------------------------------------------------------------------------

/// The bytes the harness's hooks read: filled by prepare (the prepared table
/// bytes) and refreshed by run (what the device holds NOW — gate 2's
/// unchanged-inputs check reads the table back, as contract_qmm does).
const Rig = struct {
    output: []const u8 = &.{},
    inputs: []const u8 = &.{},
    guards_ok: bool = false,
    pub fn outputBytes(r: Rig) []const u8 {
        return r.output;
    }
    pub fn inputBytes(r: Rig) []const u8 {
        return r.inputs;
    }
    pub fn intact(r: Rig) bool {
        return r.guards_ok;
    }
    pub fn overlap(_: Rig) bool {
        return false; // the in-place row is its own buffer; no binding range overlaps another
    }
};

pub const Arm = enum { kernel, control, subnormal };

pub const Penalize = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    dev: *const vk.Device,
    cmds: *vk.Commands,
    staging: vk.Buffer,
    timeout_ns: u64,
    /// penalize.spv, or the control's image when arm == .control.
    spirv: []align(4) const u8,
    arm: Arm,
    kind: Kind = .double,
    pipe: ?vk.dispatch.Pipeline = null,
    logits: ?vk.Buffer = null,
    table: ?vk.Buffer = null,
    row_words: usize = 0,
    n: u32 = 0,
    toks: []u32 = &.{},
    subs: []f32 = &.{},
    input: []u16 = &.{},
    want: []u16 = &.{},
    named: []bool = &.{},
    got: []u16 = &.{},
    shell: []u16 = &.{},
    table_back: []u8 = &.{},
    rig: Rig = .{},
    msg: [192]u8 = undefined,

    pub fn init(a: std.mem.Allocator, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, timeout_ns: u64, spirv: []align(4) const u8, arm: Arm, kind: Kind) Penalize {
        return .{
            .a = a,
            .arena = .init(a),
            .dev = dev,
            .cmds = cmds,
            .staging = staging,
            .timeout_ns = timeout_ns,
            .spirv = spirv,
            .arm = arm,
            .kind = kind,
        };
    }

    pub fn deinit(self: *Penalize) void {
        self.reset();
        self.arena.deinit();
    }

    fn reset(self: *Penalize) void {
        if (self.pipe) |*p| p.deinit();
        self.pipe = null;
        if (self.logits) |*b| b.deinit();
        self.logits = null;
        if (self.table) |*b| b.deinit();
        self.table = null;
        self.rig = .{};
    }

    /// The table is a comptime constant: `&g.hooks(Penalize)` would point at a
    /// stack temporary that dies when kernel() returns.
    const vt = g.hooks(Penalize);

    pub fn kernel(self: *Penalize) g.Kernel {
        return switch (self.arm) {
            .kernel => .{ .spec = spec, .ctx = self, .vt = &vt },
            .control => .{ .spec = controlSpec(self.kind), .ctx = self, .vt = &vt, .control = controlGate(self.kind) },
            .subnormal => .{ .spec = spec, .ctx = self, .vt = &vt }, // never registered; the runner drives it by hand
        };
    }

    pub fn prepare(self: *Penalize, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (self.arm == .subnormal) {
            self.drop();
            return self.build(.{ .rows = 1, .n = vocab, .k = 0 }, null, seed);
        }
        // The plan is the admission: a shape whose plan is out of range is
        // refused before any allocation (never a served shape — every shape
        // the spec generates plans).
        _ = planFor(shape) catch return false;
        self.drop();
        return self.build(shape, fill, seed);
    }

    fn drop(self: *Penalize) void {
        self.reset();
        _ = self.arena.reset(.retain_capacity);
    }

    fn build(self: *Penalize, shape: c.Shape, fill: ?g.Fill, seed: u64) !bool {
        const a = self.arena.allocator();
        const plan = planFor(shape) catch return false;
        self.row_words = shape.n;
        self.n = @intCast(shape.rows);
        self.input = try a.alloc(u16, self.row_words);
        self.want = try a.alloc(u16, self.row_words);
        self.named = try a.alloc(bool, self.row_words);
        self.got = try a.alloc(u16, plan.shell_words);
        self.shell = try a.alloc(u16, plan.shell_words);
        self.toks = try a.alloc(u32, self.n);
        self.subs = try a.alloc(f32, self.n);
        @memset(self.named, false);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (self.arm == .subnormal) {
            @memset(self.input, 0);
            self.input[subnormal_token] = subnormal_logit;
            self.toks[0] = @intCast(subnormal_token);
            self.subs[0] = subnormal_amount;
        } else {
            for (self.input) |*w| w.* = randomLogit(rng);
            const picks = try distinctTokens(a, rng, self.n);
            @memcpy(self.toks, picks);
            for (self.subs, self.toks) |*s, t| s.* = informativeAmount(rng, self.input[t]);
            if (fill == .nonfinite) plantNonfinite(self.toks, self.subs, self.input);
        }
        for (self.toks) |t| self.named[t] = true;
        @memcpy(self.want, self.input);
        for (self.toks, self.subs) |t, s| self.want[t] = wantBits(self.input[t], s);
        // the guarded shells: pattern, data, pattern
        @memset(self.shell, 0xa5a5);
        @memcpy(self.shell[guard_words..][0..self.row_words], self.input);
        const table = try a.alloc(u8, plan.table_bytes);
        @memset(table, pattern);
        @memcpy(table[@intCast(plan.toks_off)..][0..plan.toks_len], std.mem.sliceAsBytes(self.toks));
        @memcpy(table[@intCast(plan.subs_off)..][0..plan.subs_len], std.mem.sliceAsBytes(self.subs));
        self.table_back = try a.alloc(u8, plan.table_bytes);
        self.logits = try vk.Buffer.init(self.dev, .device, std.mem.sliceAsBytes(self.shell).len);
        self.table = try vk.Buffer.init(self.dev, .device, plan.table_bytes);
        try vk.orExit(vk.upload(self.cmds, self.logits.?, 0, self.staging, std.mem.sliceAsBytes(self.shell), self.timeout_ns));
        try vk.orExit(vk.upload(self.cmds, self.table.?, 0, self.staging, table, self.timeout_ns));
        self.pipe = try vk.dispatch.Pipeline.init(self.dev, self.spirv, .{ .bindings = 3, .push_bytes = @sizeOf(u32), .writes = 1 });
        var view = try vk.tensor.View.init(&self.logits.?, .bf16, guard, self.row_words);
        try self.pipe.?.bind(&.{
            view.bind(),
            .{ .buffer = &self.table.?, .offset = plan.toks_off, .len = plan.toks_len },
            .{ .buffer = &self.table.?, .offset = plan.subs_off, .len = plan.subs_len },
        });
        // gate 2's inputs() before the first run must be the prepared bytes;
        // run() overwrites both with what the device holds. The data window
        // only: the same slice run() reads back.
        self.rig = .{ .inputs = table[guard .. table.len - guard] };
        return true;
    }

    /// One decode: the row the head produced fresh (the prepared row, staged
    /// again — the engine never penalizes the same row twice), the dispatch at
    /// the engine's group math, then the row and the entry table read back.
    pub fn run(self: *Penalize) !void {
        try vk.orExit(vk.upload(self.cmds, self.logits.?, 0, self.staging, std.mem.sliceAsBytes(self.shell), self.timeout_ns));
        try vk.orExit(self.pipe.?.run(self.cmds, std.mem.asBytes(&self.n), .{ groups(self.n), 1, 1 }, self.timeout_ns));
        try vk.orExit(vk.download(self.cmds, self.logits.?, 0, self.staging, std.mem.sliceAsBytes(self.got), self.timeout_ns));
        try vk.orExit(vk.download(self.cmds, self.table.?, 0, self.staging, self.table_back, self.timeout_ns));
        // what the rig reports: the output window, the table as the device now
        // holds it, and the write contract — guards of both buffers intact and
        // every word no entry names bit-identical to its input.
        var ok = true;
        for (self.got, 0..) |w, i| {
            if ((i < guard_words or i >= guard_words + self.row_words) and w != 0xa5a5) {
                ok = false;
                break;
            }
        }
        if (ok) for (self.table_back, 0..) |v, i| {
            if (i < guard or i >= self.table_back.len - guard) {
                if (v != pattern) {
                    ok = false;
                    break;
                }
            }
        };
        if (ok) for (0..self.row_words) |t| {
            if (!self.named[t] and self.got[guard_words + t] != self.input[t]) {
                ok = false;
                break;
            }
        };
        self.rig = .{
            .output = std.mem.sliceAsBytes(self.got[guard_words..][0..self.row_words]),
            .inputs = self.table_back[guard .. self.table_back.len - guard],
            .guards_ok = ok,
        };
    }

    pub fn count(self: *Penalize) usize {
        return self.row_words;
    }

    pub fn reference(self: *Penalize, want: []f64, mag: []f64) !void {
        _ = self;
        @memset(want, 0);
        @memset(mag, 0); // never called: the checker hook owns every comparison
    }

    pub fn check(self: *Penalize, at: g.Gate) !?[]const u8 {
        _ = at;
        const got = self.got[guard_words..][0..self.row_words];
        if (firstFault(got, self.want, self.input, self.named)) |f| {
            return std.fmt.bufPrint(&self.msg, "token {d} reads 0x{x:0>4} ({s}), the f64 reference says 0x{x:0>4} ({s})", .{ f.token, f.got, classWord(f.got), f.want, classWord(f.want) }) catch "message too long";
        }
        return null;
    }

    /// The denormal row's gate: one dispatch over the fixed case, then the
    /// conditioned verdict (the runner counts it, KNOWN or not).
    pub fn subnormalGate(self: *Penalize, preserve32: bool) !Verdict {
        try self.run();
        return subnormalVerdict(self.got[guard_words..][0..self.row_words], self.want, self.input, preserve32);
    }
};

const testing = std.testing;

// ---------------------------------------------------------------------------
// Host tests: the reference, the checkers, the staging rule, the geometry,
// the control images — no GPU
// ---------------------------------------------------------------------------

test "the f64 reference is the CUDA formula: RNE ties, signed zero, the denormal row, classes" {
    // the penalties_fixture's hand-derived tie words (exact halves, to-even and away)
    try testing.expectEqual(@as(u16, 0x3f80), wantBits(0x3f80, -0.00390625));
    try testing.expectEqual(@as(u16, 0x3f82), wantBits(0x3f80, -0.01171875));
    try testing.expectEqual(@as(u16, 0xbf80), wantBits(0xbf80, 0.00390625));
    try testing.expectEqual(@as(u16, 0xbf82), wantBits(0xbf80, 0.01171875));
    // 1.0 minus 1.0 is POSITIVE zero, never 0x8000
    try testing.expectEqual(@as(u16, 0x0000), wantBits(0x3f80, 1.0));
    // the strict CUDA edge: smallest bf16 normal minus fp32 2^-133 lands on 0x007f
    try testing.expectEqual(@as(u16, 0x007f), wantBits(0x0080, subnormal_amount));
    // classes: infinities keep their sign, NaN stays NaN through either operand
    try testing.expectEqual(@as(u16, 0x7f80), wantBits(0x7f80, 1.0));
    try testing.expectEqual(@as(u16, 0xff80), wantBits(0xff80, 1.0));
    try testing.expect(std.math.isNan(c.fromBf16(wantBits(0x7fc0, 1.0))));
    try testing.expect(std.math.isNan(c.fromBf16(wantBits(0x3f80, std.math.nan(f32)))));
}

test "the checker refuses planted outputs: a wrong word, a scribbled neighbor, a class break" {
    var input = [_]u16{ 0x3f80, 0x4000, 0x7fc0 };
    var named = [_]bool{ true, false, false };
    var want: [3]u16 = undefined;
    want[0] = wantBits(input[0], 0.5);
    want[1] = input[1];
    want[2] = input[2];
    var got: [3]u16 = undefined;
    @memcpy(&got, &want);
    try testing.expectEqual(@as(?Fault, null), firstFault(&got, &want, &input, &named));
    got[0] ^= 1; // one bf16 ulp off at a named token
    try testing.expectEqual(@as(usize, 0), firstFault(&got, &want, &input, &named).?.token);
    @memcpy(&got, &want);
    got[1] ^= 0x0080; // a word no entry names moved
    try testing.expectEqual(@as(usize, 1), firstFault(&got, &want, &input, &named).?.token);
    @memcpy(&got, &want);
    got[2] = 0x3f80; // NaN planted, finite read back
    try testing.expectEqual(@as(usize, 2), firstFault(&got, &want, &input, &named).?.token);
    _ = &named;
}

test "the run's accounting counts caught controls as passed: the device's numbers give total 6 and exit 0" {
    // jw16 G13C: 1 kernel passing gates 1-4, 4 controls red as planted, the denormal row known
    const t: g.Tally = .{ .kernels = 1, .passed = 1, .failed = 0, .controls = 4, .caught = 4, .missed = 0, .known = 0 };
    const a = accounting(t, .known, false);
    try testing.expectEqual(@as(usize, 5), a.passed);
    try testing.expectEqual(@as(usize, 1), a.known);
    try testing.expectEqual(@as(usize, 6), a.total);
    try testing.expect(a.ok);
    // a missed control, a failed kernel, a failed denormal row, a refused device each fail the run
    var missed = t;
    missed.caught = 3;
    missed.missed = 1;
    try testing.expect(!accounting(missed, .known, false).ok);
    var failed = t;
    failed.passed = 0;
    failed.failed = 1;
    try testing.expect(!accounting(failed, .known, false).ok);
    try testing.expect(!accounting(t, .fail, false).ok);
    try testing.expect(!accounting(t, .known, true).ok);
    // a device that preserves denormals and passes the row: all six pass, none known
    const clean = accounting(t, .pass, false);
    try testing.expectEqual(@as(usize, 6), clean.passed);
    try testing.expectEqual(@as(usize, 0), clean.known);
    try testing.expect(clean.ok);
}

test "the denormal verdict is KNOWN only for the unchanged-logit flush on a device reporting no preservation" {
    var input: [8]u16 = @splat(0);
    input[subnormal_token] = subnormal_logit;
    var want = input;
    want[subnormal_token] = 0x007f;
    var named: [8]bool = @splat(false);
    named[subnormal_token] = true;
    // exact: pass, whatever the device reports
    try testing.expectEqual(Verdict.pass, subnormalVerdict(&want, &want, &input, true));
    try testing.expectEqual(Verdict.pass, subnormalVerdict(&want, &want, &input, false));
    // the whole row read back unchanged: the flush, KNOWN exactly when the
    // device reports no denormal preservation
    try testing.expectEqual(Verdict.known, subnormalVerdict(&input, &want, &input, false));
    try testing.expectEqual(Verdict.fail, subnormalVerdict(&input, &want, &input, true));
    // a flushed-to-zero RESULT is a different device behavior: never KNOWN
    var zeroed = input;
    zeroed[subnormal_token] = 0x0000;
    try testing.expectEqual(Verdict.fail, subnormalVerdict(&zeroed, &want, &input, false));
    // a scribble beside the token is a real fault: never KNOWN
    var scribbled = input;
    scribbled[3] = 0x2180; // a bf16 word at 2^-60: any wrong value fails
    try testing.expectEqual(Verdict.fail, subnormalVerdict(&scribbled, &want, &input, false));
}

test "every staged entry's amount moves its bf16 word: a plain random draw often does not" {
    var prng = std.Random.DefaultPrng.init(0x70e7);
    const rng = prng.random();
    for (0..2000) |_| {
        const x = randomLogit(rng);
        try testing.expect(wantBits(x, informativeAmount(rng, x)) != x);
    }
    // the condition that silenced the tail control on the device: an
    // unfiltered random amount frequently rounds straight back to its logit
    var noops: usize = 0;
    for (0..2000) |_| {
        const x = randomLogit(rng);
        if (wantBits(x, randomAmount(rng)) == x) noops += 1;
    }
    try testing.expect(noops > 0);
}

test "every control's host semantics are rejected by the exact check at every control shape" {
    inline for (control_served) |shape| {
        const n: usize = shape.rows;
        var prng = std.Random.DefaultPrng.init(0x70e8 + n);
        const rng = prng.random();
        const input = try testing.allocator.alloc(u16, vocab);
        defer testing.allocator.free(input);
        const want = try testing.allocator.alloc(u16, vocab);
        defer testing.allocator.free(want);
        const named = try testing.allocator.alloc(bool, vocab);
        defer testing.allocator.free(named);
        const toks = try distinctTokens(testing.allocator, rng, @intCast(n));
        defer testing.allocator.free(toks);
        const subs = try testing.allocator.alloc(f32, n);
        defer testing.allocator.free(subs);
        for (input) |*w| w.* = randomLogit(rng);
        for (subs, toks) |*s, t| s.* = informativeAmount(rng, input[t]);
        @memset(named, false);
        for (toks) |t| named[t] = true;
        @memcpy(want, input);
        for (toks, subs) |t, s| want[t] = wantBits(input[t], s);
        const rejects = struct {
            fn f(got: []const u16, w: []const u16, i: []const u16, nm: []const bool) !void {
                if (firstFault(got, w, i, nm) == null) return error.ControlPassedTheExactCheck;
            }
        }.f;
        // tail: the last entry unapplied
        const tail_got = try testing.allocator.dupe(u16, want);
        defer testing.allocator.free(tail_got);
        tail_got[toks[n - 1]] = input[toks[n - 1]];
        try rejects(tail_got, want, input, named);
        // double: every entry subtracted twice
        const dbl_got = try testing.allocator.alloc(u16, vocab);
        defer testing.allocator.free(dbl_got);
        @memcpy(dbl_got, input);
        for (toks, subs) |t, s| dbl_got[t] = doubleBits(input[t], s);
        try rejects(dbl_got, want, input, named);
        // bf16 subtract: both operands rounded before the subtract
        const bf_got = try testing.allocator.alloc(u16, vocab);
        defer testing.allocator.free(bf_got);
        @memcpy(bf_got, input);
        for (toks, subs) |t, s| bf_got[t] = bf16SubBits(input[t], s);
        try rejects(bf_got, want, input, named);
        // scribble: the word beside the first token
        const scr_got = try testing.allocator.dupe(u16, want);
        defer testing.allocator.free(scr_got);
        scr_got[toks[0] ^ 1] = input[toks[0] ^ 1] ^ 1;
        try rejects(scr_got, want, input, named);
    }
}

test "staging is one entry per distinct token with the engine's amounts; duplicates and out-of-range refuse" {
    const entries = try stageEntries(testing.allocator, &.{ 5, 7, 5, 7, 9 }, 1, 0.5);
    defer testing.allocator.free(entries);
    try testing.expectEqualSlices(Entry, &.{
        .{ .tok = 5, .sub = 2.0 }, // presence 1 + frequency 0.5 times count 2
        .{ .tok = 7, .sub = 2.0 },
        .{ .tok = 9, .sub = 1.5 }, // count 1: presence 1 + frequency 0.5
    }, entries);
    try testing.expectError(error.OutOfRange, stageEntries(testing.allocator, &.{vocab}, 1, 0));
    try testing.expect(!tokenInRange(vocab));
    try testing.expect(tokenInRange(vocab - 1));
}

test "the amounts offset is a 256-byte multiple past every toks range" {
    for ([_]u32{ 1, 2, 8, 63, 64, 65, 255, 256, 257, max_entries }) |cap| {
        try testing.expectEqual(@as(u64, 0), amountsAt(cap) % 256);
        try testing.expect(amountsAt(cap) >= @as(u64, cap) * 4);
        try testing.expect(amountsAt(cap) < @as(u64, cap) * 4 + 256);
    }
}

test "every shape the spec generates plans in range: allocations, 256-byte offsets, the launch floor" {
    const list = try c.shapes(testing.allocator, spec);
    defer testing.allocator.free(list);
    // served two + the 255/256/257 dispatch edges (1 and 7169 are served)
    try testing.expectEqual(@as(usize, 5), list.len);
    for (list) |shape| {
        const p = try planFor(shape);
        try testing.expectEqual(logitsBytes(shape.n), 2 * p.shell_words);
        // toks, its padding and the amounts fill the table's data window exactly
        try testing.expectEqual(@as(u64, p.table_bytes - 2 * guard), p.amounts_at + p.subs_len);
        try testing.expectEqual(p.amounts_at, p.subs_off - p.toks_off);
        try testing.expectEqual(@as(u64, 0), p.toks_off % 256);
        try testing.expectEqual(@as(u64, 0), p.subs_off % 256);
        try testing.expectEqual(groups(@intCast(shape.rows)), p.groups);
        try testing.expectEqual((shape.rows + 255) / 256, @as(usize, p.groups));
        // maxComputeWorkGroupCount[0]'s guaranteed floor: every served launch
        // fits any real device (dispatch.Pipeline.init enforces the device's own bound)
        try testing.expect(p.groups <= 65535);
    }
    // the refusals: a wrong row width, an empty entry list, a count past the engine's ceiling
    try testing.expectError(error.Invalid, planFor(.{ .rows = 1, .n = 4096, .k = 0 }));
    try testing.expectError(error.Invalid, planFor(.{ .rows = 0, .n = vocab, .k = 0 }));
    try testing.expectError(error.Invalid, planFor(.{ .rows = max_entries + 1, .n = vocab, .k = 0 }));
}

test "the random families stay in the normal range, so no fill trips a device flush" {
    var prng = std.Random.DefaultPrng.init(0x70e5);
    const rng = prng.random();
    for (0..1000) |_| {
        const w = randomLogit(rng);
        const exp: u16 = (w >> 7) & 0xff;
        try testing.expect(exp >= 67 and exp <= 187); // |x| in [2^-60, 2^60]
        const s = randomAmount(rng);
        try testing.expect(std.math.isFinite(s) and @abs(s) >= 0x1p-61 and @abs(s) <= 0x1p61);
    }
}

test "distinct tokens are distinct, even and inside the row" {
    var prng = std.Random.DefaultPrng.init(0x70e6);
    const rng = prng.random();
    const picks = try distinctTokens(testing.allocator, rng, 1000);
    defer testing.allocator.free(picks);
    for (picks, 0..) |t, i| {
        try testing.expect(tokenInRange(t));
        try testing.expect(t % 2 == 0);
        for (picks[i + 1 ..]) |o| try testing.expect(t != o);
    }
}

test "the control images and penalize compile: magic word, and no two are the same bytes" {
    const images = [_]struct { name: []const u8, spv: []align(4) const u8 }{
        .{ .name = "penalize", .spv = &shaders.penalize },
        .{ .name = "ctl_double", .spv = &shaders.ctl_double },
        .{ .name = "ctl_bf16_sub", .spv = &shaders.ctl_bf16_sub },
        .{ .name = "ctl_scribble", .spv = &shaders.ctl_scribble },
        .{ .name = "ctl_tail", .spv = &shaders.ctl_tail },
    };
    for (images) |img| {
        const words = std.mem.bytesAsSlice(u32, img.spv);
        try testing.expect(words.len > 5);
        try testing.expectEqual(@as(u32, 0x07230203), words[0]); // SPIR-V magic
    }
    for (images, 0..) |x, i| for (images[i + 1 ..]) |y| {
        try testing.expect(!std.mem.eql(u8, x.spv, y.spv)); // a define matching nothing would compile a twin
    };
}

test "the spec declares the served envelope: one vocab row, the engine's entry ceiling, in place" {
    try testing.expect(spec.in_place);
    try testing.expectEqual(c.Accum.none, spec.accum);
    try testing.expectEqual(@as(usize, 2), spec.served.len);
    try testing.expectEqual(@as(usize, 1), spec.served[0].rows);
    try testing.expectEqual(max_entries, spec.served[1].rows);
    for (spec.served) |s| try testing.expectEqual(vocab, s.n);
}
