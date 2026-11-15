// Qwen3.5-2B: context 1-7168 (256 default); 2 exact-ID runs, lowest ties; exits 0/1/2/124; teacher-forced per-step.
const std = @import("std");
const q35v = @import("qwen35_vulkan");
const engine_mod = q35v.engine;
const pinned = q35v.config;
const q35s = @import("qwen35_shaders");
const sys = @import("sys");

const receipt = @import("qwen35_model_receipt.zig");
const Receipt = receipt.Receipt;
const writeResult = receipt.writeResult;
const free_running_mode = receipt.free_running_mode;
const teacher_forced_mode = receipt.teacher_forced_mode;

const Engine = engine_mod.Engine;

const usage = "usage: tf-qwen35-model [--vendor ID] [--timeout-ms MS] [--context N] [--teacher-forced] [--sequential-prefill] MODEL_DIR INPUT_IDS.npy ORACLE_DIR OUTPUT_DIR\n" ++
    "  --help            show this text and exit without opening a loader or device\n" ++
    "  --vendor ID       refuse a device whose Vulkan vendorID differs (decimal or 0x hex; no default)\n" ++
    "  --timeout-ms MS   bound on every fence wait, 1 to 300000 (default 10000); a timeout ends the process\n" ++
    "  --context N       KV context capacity to request, 1 to 7168 (default 256)\n" ++
    "  --teacher-forced  feed each scored step's oracle id as the next input (labelled per-step mode; default off)\n" ++
    "  --sequential-prefill  prefill one prompt token per Engine.prefill call (test-only; default is the canonical whole-prompt 64-row chunking)\n";

const Elapsed = struct {
    io: std.Io,
    start: std.Io.Timestamp,

    fn ms(e: Elapsed) i64 {
        return e.start.durationTo(std.Io.Clock.awake.now(e.io)).toMilliseconds();
    }
};

const Options = struct {
    vendor: ?u32 = null,
    timeout_ms: u64 = 10_000,
    context: usize = 256,
    teacher_forced: bool = false,
    sequential_prefill: bool = false,
    positional: [4][]const u8 = undefined,
};

// The CLI's shape: flags then exactly four positional paths, or null; pure and host-tested.
pub fn parseArgs(args: []const []const u8) ?Options {
    var o: Options = .{};
    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--vendor")) {
            if (i + 1 >= args.len) return null;
            o.vendor = std.fmt.parseInt(u32, args[i + 1], 0) catch return null;
            i += 2;
        } else if (std.mem.eql(u8, arg, "--timeout-ms")) {
            if (i + 1 >= args.len) return null;
            o.timeout_ms = std.fmt.parseInt(u64, args[i + 1], 10) catch return null;
            if (o.timeout_ms == 0 or o.timeout_ms > 300_000) return null;
            i += 2;
        } else if (std.mem.eql(u8, arg, "--context")) {
            if (i + 1 >= args.len) return null;
            o.context = std.fmt.parseInt(usize, args[i + 1], 10) catch return null;
            if (o.context == 0 or o.context > q35v.backend.max_capacity) return null;
            i += 2;
        } else if (std.mem.eql(u8, arg, "--teacher-forced")) {
            o.teacher_forced = true;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--sequential-prefill")) {
            o.sequential_prefill = true;
            i += 1;
        } else if (std.mem.startsWith(u8, arg, "--")) {
            return null;
        } else break;
    }
    const rest = args[i..];
    if (rest.len != 4) return null;
    o.positional = .{ rest[0], rest[1], rest[2], rest[3] };
    return o;
}

// Canonical prefill chunking: how many 64-row chunks `rows` spans; pure, host-tested; the chunk size is the engine's.
pub fn chunkCount(rows: usize) usize {
    return (rows + engine_mod.max_prefill_rows - 1) / engine_mod.max_prefill_rows;
}

// The .npy header's verdict, or null unless a v1 little-endian u32 rank-1 array of `count` words; pure, host-tested.
pub const Npy = struct { count: usize, data_offset: usize };

pub fn parseNpy(bytes: []const u8) ?Npy {
    if (bytes.len < 8 or bytes[6] != 1) return null;
    const array = @import("core").npy.parse(bytes) catch return null;
    if (array.rank != 1 or array.shape[0] == 0 or !std.mem.eql(u8, array.descr, "<u4")) return null;
    return .{ .count = array.shape[0], .data_offset = bytes.len - array.data.len };
}

fn readPromptIds(a: std.mem.Allocator, bytes: []const u8) ![]u32 {
    if (bytes.len == 0 or bytes.len % 4 != 0) return error.BadPayload;
    const ids = try a.alloc(u32, bytes.len / 4);
    for (ids, 0..) |*id, i| id.* = std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little);
    return ids;
}

fn sha256HexA(a: std.mem.Allocator, data: []const u8) ![]const u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(data, &digest, .{});
    return a.dupe(u8, &std.fmt.bytesToHex(digest, .lower));
}

/// A 64-hex-character SHA-256 string, pure and host-tested.
pub fn isShaHex(s: []const u8) bool {
    if (s.len != 64) return false;
    for (s) |c| if (!std.ascii.isHex(c)) return false;
    return true;
}

// Two SHA-256 hex strings naming the same digest (case-insensitive, both well-formed), pure and host-tested.
pub fn sameHex(a: []const u8, b: []const u8) bool {
    return isShaHex(a) and isShaHex(b) and std.ascii.eqlIgnoreCase(a, b);
}

// Differing BF16 elements (2-byte pairs) between two logit dumps, null on unequal/torn lengths; the diagnostic counter.
pub fn countElementMismatches(a_bytes: []const u8, b_bytes: []const u8) ?u64 {
    if (a_bytes.len != b_bytes.len or a_bytes.len % 2 != 0) return null;
    var n: u64 = 0;
    var i: usize = 0;
    while (i < a_bytes.len) : (i += 2) {
        if (a_bytes[i] != b_bytes[i] or a_bytes[i + 1] != b_bytes[i + 1]) n += 1;
    }
    return n;
}

// The oracle's expected greedy ids from token_ids.json (a JSON array of non-negative integers), or why it is unusable.
fn readExpectedIds(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, oracle_dir: []const u8) ![]u32 {
    const path = try std.fmt.allocPrint(a, "{s}/token_ids.json", .{oracle_dir});
    defer a.free(path);
    const bytes = try dir.readFileAlloc(io, path, a, .limited(1 << 20));
    defer a.free(bytes);
    var parsed = try std.json.parseFromSlice(std.json.Value, a, bytes, .{});
    defer parsed.deinit();
    if (parsed.value != .array) return error.BadOracleIds;
    const items = parsed.value.array.items;
    const out = try a.alloc(u32, items.len);
    errdefer a.free(out);
    if (items.len == 0) return error.BadOracleIds;
    for (items, 0..) |item, i| {
        if (item != .integer or item.integer < 0 or item.integer > std.math.maxInt(u32)) return error.BadOracleIds;
        out[i] = @intCast(item.integer);
    }
    return out;
}

// Oracle result.json fields the receipt echoes; missing = unusable oracle, not a pass; arena-allocated, nothing freed.
const OracleMeta = struct {
    prompt_token_hash: []const u8,
    generated_ids_hash: []const u8,
    logits_bf16_hash: []const u8,
    logits_bytes: u64,
    steps: u64,
    note: []const u8,
};

fn readOracleMeta(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, oracle_dir: []const u8) !OracleMeta {
    const path = try std.fmt.allocPrint(a, "{s}/result.json", .{oracle_dir});
    defer a.free(path);
    const bytes = try dir.readFileAlloc(io, path, a, .limited(1 << 22));
    defer a.free(bytes);
    var parsed = try std.json.parseFromSlice(std.json.Value, a, bytes, .{});
    defer parsed.deinit();
    const root = if (parsed.value == .object) parsed.value.object else return error.BadOracleMeta;
    const str = struct {
        fn get(alloc: std.mem.Allocator, o: std.json.ObjectMap, key: []const u8) ![]const u8 {
            const v = o.get(key) orelse return error.BadOracleMeta;
            if (v != .string) return error.BadOracleMeta;
            return alloc.dupe(u8, v.string);
        }
    };
    const num = struct {
        fn get(o: std.json.ObjectMap, key: []const u8) !u64 {
            const v = o.get(key) orelse return error.BadOracleMeta;
            if (v != .integer or v.integer < 0) return error.BadOracleMeta;
            return @intCast(v.integer);
        }
    };
    var note: []const u8 = "";
    if (root.get("reference_limits")) |rl| {
        if (rl == .object) {
            if (rl.object.get("checkpoint_revision")) |v| {
                if (v == .string) note = try a.dupe(u8, v.string);
            }
        }
    }
    return .{
        .prompt_token_hash = try str.get(a, root, "prompt_token_hash"),
        .generated_ids_hash = try str.get(a, root, "generated_ids_hash"),
        .logits_bf16_hash = try str.get(a, root, "logits_bf16_hash"),
        .logits_bytes = try num.get(root, "logits_bytes"),
        .steps = try num.get(root, "steps"),
        .note = note,
    };
}

// The bundle: qwen35_shaders SPIR-V per ShaderBundle field, shared by probe and load test — the set the server serves.
fn bundle() engine_mod.ShaderBundle {
    return .{
        .ops = .{
            .qmm_bf16 = &q35s.qmm_bf16,
            .qmm_q4_vec4 = &q35s.qmm_q4_vec4,
            .qmm_f32 = &q35s.qmm_f32,
            .qmm_q4_tile = &q35s.qmm_q4_tile,
            .qmm_q4_tile_rows = &q35s.qmm_q4_tile_rows,
            .qmm_q4_tile_f32 = &q35s.qmm_q4_tile_f32,
            .dequant_bf16 = &q35s.dequant_bf16,
            .gather_u32 = &q35s.gather_u32,
            .gather_u16 = &q35s.gather_u16,
            .cast_bf16_f32 = &q35s.cast_bf16_f32,
            .cast_f32_bf16 = &q35s.cast_f32_bf16,
            .copy_u32 = &q35s.copy_u32,
            .copy_u16 = &q35s.copy_u16,
            .copy_f32_bf16 = &q35s.copy_f32_bf16,
        },
        .norm = .{
            .norm_bf16 = &q35s.norm_bf16,
            .add_bf16 = &q35s.add_bf16,
            .add_preserve_bf16 = &q35s.add_preserve_bf16,
            .swiglu_bf16 = &q35s.swiglu_bf16,
            .norm_f32 = &q35s.norm_f32,
            .norm_f32_in = &q35s.norm_f32_in,
            .norm_bf16_in = &q35s.norm_bf16_in,
            .add_f32 = &q35s.add_f32,
            .add_wide_bf16 = &q35s.add_wide_bf16,
            .add_bf16_f32 = &q35s.add_bf16_f32,
            .swiglu_f32 = &q35s.swiglu_f32,
        },
        .attention = .{
            .rope_bf16 = &q35s.rope_bf16,
            .sdpa_rows_bf16 = &q35s.sdpa_rows_bf16,
            .gate_bf16 = &q35s.gate_bf16,
            .rope_f32 = &q35s.rope_f32,
            .rope_f32_kcache = &q35s.rope_f32_kcache,
            .sdpa_rows_f32q = &q35s.sdpa_rows_f32q,
            .gate_f32 = &q35s.gate_f32,
        },
        .gdn = .{
            .conv_dw1d_bf16 = &q35s.conv_dw1d_bf16,
            .gdn_conv_decode_bf16 = &q35s.gdn_conv_decode_bf16,
            .gdn_split_bf16 = &q35s.gdn_split_bf16,
            .prefill_recur_bf16 = &q35s.prefill_recur_bf16,
            .gated_delta_decode_bf16 = &q35s.gated_delta_decode_bf16,
            .fast_norm_gated_bf16 = &q35s.fast_norm_gated_bf16,
            .copy_u16 = &q35s.copy_u16,
            .conv_dw1d_f32 = &q35s.conv_dw1d_f32,
            .gdn_split_f32 = &q35s.gdn_split_f32,
            .prefill_recur_f32 = &q35s.prefill_recur_f32,
            .fast_norm_gated_f32 = &q35s.fast_norm_gated_f32,
            .copy_u32 = &q35s.gdn_copy_u32,
        },
        .greedy = .{ .finite_logits = &q35s.finite_logits, .argreduce_chunk = &q35s.argreduce_chunk, .argreduce_combine = &q35s.argreduce_combine },
    };
}

pub fn main(init: std.process.Init) !u8 {
    const clock: Elapsed = .{ .io = init.io, .start = std.Io.Clock.awake.now(init.io) };
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    for (args[1..]) |arg| if (std.mem.eql(u8, arg, "--help")) {
        std.debug.print(usage, .{});
        return 0;
    };
    const opts = parseArgs(args[1..]) orelse {
        std.debug.print(usage, .{});
        return 2;
    };
    const a = init.gpa;
    const io = init.io;
    const cwd = std.Io.Dir.cwd();
    // Immutable hash/diagnostic strings ride the process arena: allocated once, freed only at exit, never on gpa.
    const diag = init.arena.allocator();

    // every input validated before any GPU work exists
    const npy_bytes = cwd.readFileAlloc(io, opts.positional[1], a, .limited(1 << 26)) catch |e| {
        std.debug.print("FAIL input: cannot read {s}: {s}\n", .{ opts.positional[1], @errorName(e) });
        return 2;
    };
    defer a.free(npy_bytes);
    const npy = parseNpy(npy_bytes) orelse {
        std.debug.print("FAIL input: {s} is not a v1 little-endian u32 rank-1 NPY with a nonempty shape\n", .{opts.positional[1]});
        return 2;
    };
    if (npy.count > q35v.backend.max_capacity) {
        std.debug.print("FAIL input: {d} prompt ids past the {d}-row context cap\n", .{ npy.count, q35v.backend.max_capacity });
        return 2;
    }
    const prompt = try readPromptIds(a, npy_bytes[npy.data_offset..]);
    defer a.free(prompt);
    for (prompt) |id| if (id >= pinned.vocab) {
        std.debug.print("FAIL input: token id {d} past the vocab\n", .{id});
        return 2;
    };

    const expected = readExpectedIds(a, io, cwd, opts.positional[2]) catch |e| {
        std.debug.print("FAIL input: oracle ids unreadable from {s}: {s}\n", .{ opts.positional[2], @errorName(e) });
        return 2;
    };
    defer a.free(expected);
    // immutable metadata rides the arena: readOracleMeta allocates, the process arena frees, no per-field cleanup
    const oracle = readOracleMeta(diag, io, cwd, opts.positional[2]) catch |e| {
        std.debug.print("FAIL input: oracle result.json unreadable from {s}: {s}\n", .{ opts.positional[2], @errorName(e) });
        return 2;
    };
    if (oracle.steps != expected.len) {
        std.debug.print("FAIL input: oracle steps {d} and ids {d} disagree\n", .{ oracle.steps, expected.len });
        return 2;
    }
    // same-prompt and expected-ids proof before any GPU work: the oracle's hashes must name this prompt and these ids
    for (expected) |id| if (id >= pinned.vocab) {
        std.debug.print("FAIL input: oracle id {d} past the vocab\n", .{id});
        return 2;
    };
    const prompt_hash = try sha256HexA(diag, std.mem.sliceAsBytes(prompt));
    const expected_hash = try sha256HexA(diag, std.mem.sliceAsBytes(expected));
    if (!sameHex(prompt_hash, oracle.prompt_token_hash)) {
        std.debug.print("FAIL input: prompt hash {s} is not the oracle's {s}\n", .{ prompt_hash, oracle.prompt_token_hash });
        return 2;
    }
    if (!sameHex(expected_hash, oracle.generated_ids_hash)) {
        std.debug.print("FAIL input: expected ids hash {s} is not the oracle's {s}\n", .{ expected_hash, oracle.generated_ids_hash });
        return 2;
    }
    if (!isShaHex(oracle.logits_bf16_hash)) {
        std.debug.print("FAIL input: oracle logits hash is not 64 hex characters\n", .{});
        return 2;
    }

    // the prompt and every step must fit the requested context — checked before the device or weights exist
    engine_mod.planFor(opts.context, 0, prompt.len, expected.len) catch |e| {
        std.debug.print("FAIL plan: {s} (prompt {d} rows, {d} steps, context {d})\n", .{ @errorName(e), prompt.len, expected.len, opts.context });
        return 1;
    };

    // the output directory must be new before the GPU exists
    cwd.createDir(io, opts.positional[3], .default_dir) catch |e| switch (e) {
        error.PathAlreadyExists => {
            std.debug.print("FAIL input: output dir {s} already exists\n", .{opts.positional[3]});
            return 2;
        },
        else => {
            std.debug.print("FAIL input: cannot create {s}: {s}\n", .{ opts.positional[3], @errorName(e) });
            return 2;
        },
    };
    const out = cwd.openDir(io, opts.positional[3], .{}) catch |e| {
        std.debug.print("FAIL input: cannot open {s}: {s}\n", .{ opts.positional[3], @errorName(e) });
        return 2;
    };
    defer out.close(io);

    const engine = Engine.init(a, io, bundle(), opts.positional[0], opts.context, opts.timeout_ms * std.time.ns_per_ms) catch |e| {
        std.debug.print("FAIL engine: {s}\n", .{@errorName(e)});
        return 1;
    };
    defer engine.deinit();
    std.debug.print("MILESTONE engine-ready elapsed_ms={d} capacity={d} model_context={d} logits_bytes={d}\n", .{ clock.ms(), engine.capacity(), engine.modelContext(), engine.logitsLen() });

    // TF_VK_KTIME=1 opts into per-dispatch GPU timestamps for the prefill and
    // the steady decode window of both runs; anything that refuses is named
    // and the run continues uninstrumented. The destroy defer registered here
    // runs before engine.deinit's, so the pool dies while the device lives.
    var kt: ?*q35v.vk.dispatch.Timing = null;
    if (std.c.getenv("TF_VK_KTIME")) |v| if (std.mem.eql(u8, std.mem.span(v), "1")) {
        kt = q35v.vk.dispatch.Timing.create(a, engine.dev) catch |err| blk: {
            std.debug.print("KTIME refused: {s}\n", .{@errorName(err)});
            break :blk null;
        };
    };
    defer if (kt) |t| t.destroy(a);
    if (opts.vendor) |want| if (engine.vendorId() != want) {
        std.debug.print("FAIL device: vendor 0x{x:0>4} is not the requested 0x{x:0>4}\n", .{ engine.vendorId(), want });
        return 1;
    };

    const row_len: usize = @intCast(engine.logitsLen());
    const rows_a = try a.alloc(u8, expected.len * row_len);
    defer a.free(rows_a);
    const rows_b = try a.alloc(u8, expected.len * row_len);
    defer a.free(rows_b);
    const ids_a = try a.alloc(u32, expected.len);
    defer a.free(ids_a);
    const ids_b = try a.alloc(u32, expected.len);
    defer a.free(ids_b);
    const samples_a = try a.alloc(SteadySample, expected.len - 1);
    defer a.free(samples_a);
    const samples_b = try a.alloc(SteadySample, expected.len - 1);
    defer a.free(samples_b);

    greedyRun(engine, opts.teacher_forced, opts.sequential_prefill, prompt, expected, ids_a, rows_a, row_len, samples_a, io, a, "run_a", kt) catch |e| {
        std.debug.print("FAIL run1: {s}\n", .{@errorName(e)});
        return 1;
    };
    std.debug.print("MILESTONE run1-done elapsed_ms={d}\n", .{clock.ms()});
    try engine.reset();
    greedyRun(engine, opts.teacher_forced, opts.sequential_prefill, prompt, expected, ids_b, rows_b, row_len, samples_b, io, a, "run_b", kt) catch |e| {
        std.debug.print("FAIL run2: {s}\n", .{@errorName(e)});
        return 1;
    };
    std.debug.print("MILESTONE run2-done elapsed_ms={d}\n", .{clock.ms()});

    // within-Vulkan determinism: fresh cache, real reset, identical bytes
    const ids_equal = std.mem.eql(u32, ids_a, ids_b);
    const logits_equal = std.mem.eql(u8, rows_a, rows_b);
    // the pass criterion: the exact oracle ids, both runs, no epsilon
    const match_a = std.mem.eql(u32, ids_a, expected);
    const match_b = std.mem.eql(u32, ids_b, expected);

    // diagnostic only: element-wise raw logits against the oracle's file
    var mismatches: ?u64 = null;
    const oracle_logits_path = try std.fmt.allocPrint(a, "{s}/logits_bf16.bin", .{opts.positional[2]});
    defer a.free(oracle_logits_path);
    if (cwd.readFileAlloc(io, oracle_logits_path, a, .limited(rows_a.len + (1 << 20)))) |oracle_bytes| {
        defer a.free(oracle_bytes);
        mismatches = countElementMismatches(oracle_bytes, rows_a);
        if (mismatches == null) std.debug.print("NOTE oracle logits {d} bytes vs run {d}; diagnostic skipped\n", .{ oracle_bytes.len, rows_a.len });
    } else |_| std.debug.print("NOTE oracle logits file unreadable; diagnostic skipped\n", .{});

    // receipts: the full BF16 logit rows of run A, then the result record
    out.writeFile(io, .{ .sub_path = "logits_bf16.bin", .data = rows_a }) catch |e| {
        std.debug.print("FAIL receipt: cannot write logits_bf16.bin: {s}\n", .{@errorName(e)});
        return 1;
    };
    const ids_hash = try sha256HexA(diag, std.mem.sliceAsBytes(ids_a));
    writeResult(a, io, out, .{
        .status = if (match_a and match_b and ids_equal and logits_equal) "ok" else "mismatch",
        .steps = expected.len,
        .execution_mode = if (opts.teacher_forced) teacher_forced_mode else free_running_mode,
        .prefill_schedule = if (opts.sequential_prefill) "rowwise" else "chunked",
        .context_cap = opts.context,
        .prompt_chunks = if (opts.sequential_prefill) prompt.len else chunkCount(prompt.len),
        .prefill_chunk_rows = if (opts.sequential_prefill) 1 else engine_mod.max_prefill_rows,
        .capacity = engine.capacity(),
        .model_context = engine.modelContext(),
        .prompt_tokens = prompt.len,
        .input_npy_sha256 = try sha256HexA(diag, npy_bytes),
        .prompt_u32le_sha256 = prompt_hash,
        .generated_ids = ids_a,
        .generated_ids_hash = ids_hash,
        .logits_bytes = rows_a.len,
        .logits_bf16_sha256 = try sha256HexA(diag, rows_a),
        .repeat_checks = .{ .generated_ids_equal = ids_equal, .logits_bytes_equal = logits_equal },
        .oracle_match = .{ .ids_run_a = match_a, .ids_run_b = match_b },
        .oracle = .{
            .generated_ids_hash = oracle.generated_ids_hash,
            .logits_bf16_hash = oracle.logits_bf16_hash,
            .logits_bytes = oracle.logits_bytes,
            .checkpoint_note = oracle.note,
        },
        .oracle_logits_element_mismatches = mismatches,
        .elapsed_ms = clock.ms(),
    }) catch |e| {
        std.debug.print("FAIL receipt: cannot write result.json: {s}\n", .{@errorName(e)});
        return 1;
    };

    std.debug.print("RESULT qwen35-model elapsed_ms={d}: ids_match_a={} ids_match_b={} repeats(ids={},logits={}) " ++
        "oracle_logits_mismatches={s} expected_hash={s} got_hash={s}\n", .{
        clock.ms(),                                                                           match_a,                   match_b,  ids_equal, logits_equal,
        if (mismatches) |n| (std.fmt.allocPrint(diag, "{d}", .{n}) catch "?") else "skipped", oracle.generated_ids_hash, ids_hash,
    });
    // mode labels ride only non-default modes, so the anchor default's RESULT line stays byte-identical
    if (opts.teacher_forced) std.debug.print("RESULT execution_mode=teacher-forced: oracle ids fed as previous tokens; per-step agreement, not a free-running chain proof\n", .{});
    if (opts.sequential_prefill) std.debug.print("RESULT prefill_schedule=rowwise: one Engine.prefill call per prompt token\n", .{});
    return if (match_a and match_b and ids_equal and logits_equal) 0 else 1;
}

/// One steady token's host wall time and the id its logits drew.
const SteadySample = struct { id: u32, wall_ns: u64 };

fn steadyToken(io: std.Io, e: *Engine, feed: u32, rows: []u8) !SteadySample {
    const t0 = std.Io.Clock.awake.now(io);
    try e.decode(feed);
    const id = try e.draw();
    try e.logitsBytes(rows);
    return .{ .id = id, .wall_ns = @intCast(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds()) };
}

/// The tally entry's kernel name, resolved against the probe's own shader
/// images; a module the probe does not embed prints as unknown<len>.
fn kernelName(a: std.mem.Allocator, spirv_ptr: usize, spirv_len: u32) []const u8 {
    const bytes = @as([*]const u8, @ptrFromInt(spirv_ptr))[0..spirv_len];
    inline for (comptime std.meta.declarations(q35s)) |name| {
        const image = &@field(q35s, name);
        if (std.mem.eql(u8, std.mem.sliceAsBytes(image), bytes)) return name;
    }
    return std.fmt.allocPrint(a, "unknown<{d}>", .{spirv_len}) catch "unknown";
}

/// The tally sorted heaviest-first, with a per-image rollup folded over the
/// per-shape lines (one image may serve several shapes).
const Sorted = struct {
    fn desc(_: void, x: q35v.vk.dispatch.TallyEntry, y: q35v.vk.dispatch.TallyEntry) bool {
        return x.total_ns > y.total_ns;
    }
};

const Rollup = struct { name: []const u8, calls: u64 = 0, total_ns: u64 = 0 };

fn sortedEntries(a: std.mem.Allocator, kt: *q35v.vk.dispatch.Timing) []q35v.vk.dispatch.TallyEntry {
    const entries = a.alloc(q35v.vk.dispatch.TallyEntry, kt.tally.n) catch {
        std.debug.print("KTIME report failed: OutOfMemory\n", .{});
        return &.{};
    };
    @memcpy(entries, kt.tally.entries[0..kt.tally.n]);
    std.mem.sort(q35v.vk.dispatch.TallyEntry, entries, {}, Sorted.desc);
    return entries;
}

fn rollupByImage(a: std.mem.Allocator, entries: []const q35v.vk.dispatch.TallyEntry) []Rollup {
    var out = a.alloc(Rollup, entries.len) catch return &.{};
    var n: usize = 0;
    for (entries) |e| {
        const name = kernelName(a, e.key.spirv, e.spirv_len);
        var found = false;
        for (out[0..n]) |*r| if (std.mem.eql(u8, r.name, name)) {
            r.calls += e.calls;
            r.total_ns += e.total_ns;
            found = true;
            break;
        };
        if (!found) {
            out[n] = .{ .name = name, .calls = e.calls, .total_ns = e.total_ns };
            n += 1;
        }
    }
    return out[0..n];
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

/// The prefill phase's report: one `KTIME prefill` line per invocation shape
/// sorted by total, a per-image rollup, then the GPU total, the phase's wall
/// time and the prompt token count. Diagnostic only — it never changes the
/// run's result or exit code.
fn prefillReport(a: std.mem.Allocator, kt: *q35v.vk.dispatch.Timing, wall_ns: u64, prompt_tokens: usize) void {
    const entries = sortedEntries(a, kt);
    for (entries) |e| {
        const name = kernelName(a, e.key.spirv, e.spirv_len);
        std.debug.print("KTIME prefill {s} calls={d} total_ms={d:.3} share={d:.4}\n", .{ name, e.calls, ms(e.total_ns), kt.tally.share(e.total_ns) });
    }
    for (rollupByImage(a, entries)) |r| {
        std.debug.print("KTIME prefill image {s} calls={d} total_ms={d:.3} share={d:.4}\n", .{ r.name, r.calls, ms(r.total_ns), kt.tally.share(r.total_ns) });
    }
    std.debug.print("KTIME prefill total_ms={d:.3} wall_ms={d:.3} prompt_tokens={d}\n", .{ ms(kt.tally.total_ns), ms(wall_ns), prompt_tokens });
}

/// One `KTIME decode` block per run's steady window: every tally entry
/// heaviest first (per-token stats), the per-image rollup, then the window's
/// per-token GPU total and wall time. Diagnostic only.
fn ktimeReport(a: std.mem.Allocator, kt: *q35v.vk.dispatch.Timing, run: []const u8, samples: []const SteadySample) void {
    const per_token: u64 = @max(samples.len, 1);
    var wall_ns: u64 = 0;
    for (samples) |s| wall_ns += s.wall_ns;
    const entries = sortedEntries(a, kt);
    for (entries) |e| {
        const name = kernelName(a, e.key.spirv, e.spirv_len);
        const calls_per_token = @as(f64, @floatFromInt(e.calls)) / @as(f64, @floatFromInt(per_token));
        const mean_us = @as(f64, @floatFromInt(e.total_ns)) / @as(f64, @floatFromInt(@max(e.calls, 1))) / 1000.0;
        const total_us_per_token = @as(f64, @floatFromInt(e.total_ns)) / 1000.0 / @as(f64, @floatFromInt(per_token));
        std.debug.print("KTIME {s} {s} push={x:0>8},{x:0>8},{x:0>8},{x:0>8} groups={d},{d},{d} calls_per_token={d:.2} mean_us={d:.2} total_us_per_token={d:.2} share={d:.4}\n", .{
            run,                          name,
            e.key.push_words[0],          e.key.push_words[1],
            e.key.push_words[2],          e.key.push_words[3],
            e.key.groups[0],              e.key.groups[1],
            e.key.groups[2],              calls_per_token,
            mean_us,                      total_us_per_token,
            kt.tally.share(e.total_ns),
        });
    }
    for (rollupByImage(a, entries)) |r| {
        std.debug.print("KTIME {s} image {s} calls_per_token={d:.2} total_us_per_token={d:.2} share={d:.4}\n", .{
            run, r.name,
            @as(f64, @floatFromInt(r.calls)) / @as(f64, @floatFromInt(per_token)),
            @as(f64, @floatFromInt(r.total_ns)) / 1000.0 / @as(f64, @floatFromInt(per_token)),
            kt.tally.share(r.total_ns),
        });
    }
    std.debug.print("KTIME total {s} gpu_ns_per_token={d} wall_ns_per_token={d}\n", .{ run, kt.tally.total_ns / per_token, wall_ns / per_token });
}

// Greedy: prefill (max_prefill_rows chunks; --sequential-prefill per token = boundary evidence); ids_out.len argreduce draws.
// TF_VK_KTIME=1 instruments two phases per run: the whole prefill (all
// chunks, one tally) and the steady tokens (the first draw rides prefill's
// logits, uninstrumented). Each phase's tally restarts; the reports are
// diagnostic and never change the run's result or exit code.
fn greedyRun(e: *Engine, teacher_forced: bool, sequential: bool, prompt: []const u32, expected: []const u32, ids_out: []u32, rows_out: []u8, row_len: usize, samples_out: []SteadySample, io: std.Io, a: std.mem.Allocator, run: []const u8, kt: ?*q35v.vk.dispatch.Timing) !void {
    if (samples_out.len != ids_out.len - 1) return error.Invalid;
    if (kt) |t| {
        t.tally = .{};
        q35v.vk.dispatch.timing = t;
    }
    const pf0 = std.Io.Clock.awake.now(io);
    if (sequential) {
        for (prompt) |tok| try e.prefill(&.{tok});
    } else {
        try e.prefill(prompt);
    }
    const pf_wall: u64 = @intCast(pf0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds());
    if (kt) |t| {
        q35v.vk.dispatch.timing = null;
        prefillReport(a, t, pf_wall, prompt.len);
    }
    // the warm-up token: prefill's first draw, uninstrumented like the prefill report above
    ids_out[0] = try e.draw();
    try e.logitsBytes(rows_out[0..row_len]);
    if (kt) |t| {
        t.tally = .{};
        q35v.vk.dispatch.timing = t;
    }
    defer if (kt != null) {
        q35v.vk.dispatch.timing = null;
    };
    for (1..ids_out.len) |step| {
        const feed: u32 = if (teacher_forced) expected[step - 1] else ids_out[step - 1];
        samples_out[step - 1] = try steadyToken(io, e, feed, rows_out[step * row_len ..][0..row_len]);
        ids_out[step] = samples_out[step - 1].id;
    }
    if (kt) |t| ktimeReport(a, t, run, samples_out);
}

const testing = std.testing;

test "isShaHex and sameHex gate the oracle hash cross-checks" {
    const h = "dcd774d847e199c7a8ad93ef794e597e5a6bf672ec448afb7a2e507578b261b3";
    try testing.expect(isShaHex(h));
    try testing.expect(sameHex(h, h));
    try testing.expect(sameHex(h, "DCD774D847E199C7A8AD93EF794E597E5A6BF672EC448AFB7A2E507578B261B3"));
    try testing.expect(!isShaHex(h[0..63])); // short
    try testing.expect(!isShaHex(&@as([64]u8, @splat('g')))); // not hex
    try testing.expect(!isShaHex("dcd774")); // far too short
    try testing.expect(!sameHex(h, "d1b45ec858151fe74125ce020f637fcf1963b399a3cd32564c7ca65966624589")); // different digest
}

test "countElementMismatches counts 2-byte pairs, not bytes" {
    const a = [_]u8{ 0x11, 0x22, 0x33, 0x44, 0x55, 0x66 };
    try testing.expectEqual(@as(?u64, 0), countElementMismatches(&a, &a));
    var b = a;
    b[1] ^= 0xff; // one byte of the first element flips: one element differs
    try testing.expectEqual(@as(?u64, 1), countElementMismatches(&a, &b));
    var c = a;
    c[2] ^= 0xff; // first element intact again, second element flipped
    try testing.expectEqual(@as(?u64, 1), countElementMismatches(&a, &c));
    try testing.expectEqual(@as(?u64, null), countElementMismatches(&a, a[0..4])); // length mismatch
    try testing.expectEqual(@as(?u64, null), countElementMismatches(&a, a[0..5])); // torn element
}

test "parseArgs takes the two flags and exactly four paths" {
    const ok = parseArgs(&.{ "--vendor", "0x1002", "--timeout-ms", "90000", "/m", "/i.npy", "/o", "/out" }).?;
    try testing.expectEqual(@as(?u32, 0x1002), ok.vendor);
    try testing.expectEqual(@as(u64, 90000), ok.timeout_ms);
    try testing.expectEqualStrings("/out", ok.positional[3]);
    try testing.expect(parseArgs(&.{"/m"}) == null);
    try testing.expect(parseArgs(&.{"--vendor"}) == null);
    try testing.expect(parseArgs(&.{ "--vendor", "x", "/m", "/i", "/o", "/out" }) == null);
    try testing.expect(parseArgs(&.{ "--timeout-ms", "0", "/m", "/i", "/o", "/out" }) == null);
    try testing.expect(parseArgs(&.{ "--wat", "/m", "/i", "/o", "/out" }) == null);
    try testing.expect(parseArgs(&.{ "/m", "/i", "/o", "/out", "/extra" }) == null);
}

test "parseArgs context and mode flags keep the anchor default and bound the context" {
    // the anchor default: no new flags, the historical 256 free-running run
    const plain = parseArgs(&.{ "/m", "/i", "/o", "/out" }).?;
    try testing.expectEqual(@as(usize, 256), plain.context);
    try testing.expect(!plain.teacher_forced);
    try testing.expect(!plain.sequential_prefill);

    // the boundary sizes the corpus runs: 33, 65, 7105 all parse
    for ([_][]const u8{ "33", "65", "7105", "7168" }) |n| {
        const o = parseArgs(&.{ "--context", n, "--teacher-forced", "--sequential-prefill", "/m", "/i", "/o", "/out" }).?;
        try testing.expect(o.teacher_forced);
        try testing.expect(o.sequential_prefill);
    }
    // one past the backend's capacity is refused at the CLI, no device
    try testing.expectEqual(@as(usize, 7168), q35v.backend.max_capacity);
    try testing.expect(parseArgs(&.{ "--context", "7169", "/m", "/i", "/o", "/out" }) == null);
    try testing.expect(parseArgs(&.{ "--context", "0", "/m", "/i", "/o", "/out" }) == null);
    try testing.expect(parseArgs(&.{ "--context", "-1", "/m", "/i", "/o", "/out" }) == null);
    try testing.expect(parseArgs(&.{ "--context", "abc", "/m", "/i", "/o", "/out" }) == null);
    try testing.expect(parseArgs(&.{"--context"}) == null);
    try testing.expect(parseArgs(&.{ "--context", "256", "/m", "/i", "/o", "/out", "/extra" }) == null);
}

test "chunkCount spans the canonical 64-row chunks without a configurable size" {
    try testing.expectEqual(@as(usize, engine_mod.max_prefill_rows), 64);
    try testing.expectEqual(@as(usize, 1), chunkCount(1));
    try testing.expectEqual(@as(usize, 1), chunkCount(64));
    try testing.expectEqual(@as(usize, 2), chunkCount(65)); // boundary: two chunks
    try testing.expectEqual(@as(usize, 2), chunkCount(128));
    try testing.expectEqual(@as(usize, 3), chunkCount(129)); // boundary: three chunks
    try testing.expectEqual(@as(usize, 112), chunkCount(7168)); // full context
}

test "parseNpy takes exactly a v1 LE u32 rank-1 array" {
    const dict = "{'descr': '<u4', 'fortran_order': False, 'shape': (3, ), }";
    var header: [10 + dict.len]u8 = undefined;
    @memcpy(header[0..6], "\x93NUMPY");
    header[6] = 1;
    header[7] = 0;
    std.mem.writeInt(u16, header[8..10], @intCast(dict.len), .little);
    @memcpy(header[10..], dict);
    var buf: [256]u8 = undefined;
    @memcpy(buf[0..header.len], &header);
    @memset(buf[header.len..], 0xab);
    const n = parseNpy(buf[0 .. header.len + 12]).?;
    try testing.expectEqual(@as(usize, 3), n.count);
    try testing.expectEqual(header.len, n.data_offset);
    try testing.expect(parseNpy(buf[0 .. header.len + 11]) == null); // torn data
    try testing.expect(parseNpy(buf[0 .. header.len + 13]) == null); // extra bytes
    buf[22] = 'i'; // "<u4" becomes "<i4": not the u32 descriptor
    try testing.expect(parseNpy(buf[0 .. header.len + 12]) == null);
    buf[22] = 'u';
    try testing.expect(parseNpy(buf[0 .. header.len + 12]) != null);
    try testing.expect(parseNpy(&.{}) == null);
    try testing.expect(parseNpy("\x93NUMPY\x02\x00\x00\x00\x00\x00") == null);
    // rank 2 is refused: two integers in the shape tuple
    const dict2 = "{'descr': '<u4', 'fortran_order': False, 'shape': (3, 4), }";
    var header2: [10 + dict2.len]u8 = undefined;
    @memcpy(header2[0..6], "\x93NUMPY");
    header2[6] = 1;
    header2[7] = 0;
    std.mem.writeInt(u16, header2[8..10], @intCast(dict2.len), .little);
    @memcpy(header2[10..], dict2);
    @memcpy(buf[0..header2.len], &header2);
    try testing.expect(parseNpy(buf[0 .. header2.len + 48]) == null);
}

test "prompt ids decode little-endian words without an alignment assumption" {
    const bytes = [_]u8{ 0xaa, 0x78, 0x56, 0x34, 0x12 };
    const ids = try readPromptIds(testing.allocator, bytes[1..]);
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &.{0x12345678}, ids);
    try testing.expectError(error.BadPayload, readPromptIds(testing.allocator, bytes[2..]));
}

test "the real engine load (TF_VULKAN_LOAD_DIR): device gates, pinned admission, every layer's uploads and state, then teardown; no forward" {
    const dir = sys.getenv("TF_VULKAN_LOAD_DIR") orelse return error.SkipZigTest;
    // init here: every device gate passed, the pinned-model admission read the real config, the zeroed state exists.
    const engine = try Engine.init(testing.allocator, testing.io, bundle(), dir, 256, 60 * std.time.ns_per_s);
    defer engine.deinit();
    try testing.expectEqual(@as(usize, pinned.layers), engine.model.layers.len);
    try testing.expectEqual(@as(u64, 0), engine.visible());
    std.debug.print("MILESTONE qwen35-load capacity={d} model_context={d} layers={d} logits_bytes={d} loaded, then torn down\n", .{ engine.capacity(), engine.modelContext(), engine.model.layers.len, engine.logitsLen() });
}
