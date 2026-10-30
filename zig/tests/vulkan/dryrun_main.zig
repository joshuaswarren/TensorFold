//! The dry-run harness: the whole engine on the CT's CPU with no GPU and no
//! Vulkan loader. The mock device backs device memory with real zeroed host
//! blocks (uploads, copies, fills and read-downs move real bytes), records
//! every recorded command, and executes nothing. Loads the real checkpoint,
//! prefills the real prompt ids and decodes three tokens; any error is the
//! failure. Env switches behave exactly as in the model probe.
const std = @import("std");
const q35v = @import("qwen35_vulkan");
const q35s = @import("qwen35_shaders");
const mock = @import("mock_device");
const probe = @import("probe");

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print("usage: tf-qwen35-dryrun MODEL_DIR PROMPT.npy [context]\\n", .{});
        return 2;
    }
    const align_env = std.c.getenv("TF_DRYRUN_ALIGN");
    const alignment: u64 = if (align_env) |v| std.fmt.parseInt(u64, std.mem.span(v), 10) catch 256 else 256;

    const npy_bytes = try std.Io.Dir.cwd().readFileAlloc(io, args[2], gpa, .limited(1 << 24));
    defer gpa.free(npy_bytes);
    const npy = probe.parseNpy(npy_bytes) orelse {
        std.debug.print("FAIL npy: unreadable header\\n", .{});
        return 2;
    };
    const words: [*]const u32 = @alignCast(@ptrCast(npy_bytes.ptr + npy.data_offset));
    const ids = words[0..npy.count];

    const md = try mock.device(gpa, alignment);
    defer md.mock.deinit();
    const rt = try gpa.create(q35v.vk.Runtime);
    rt.* = .{ .lib = undefined, .instance = null, .api = undefined };

    std.debug.print("DRYRUN engine load: {s} align={d} ids={d}\\n", .{ args[1], alignment, ids.len });
    const engine_ptr = try q35v.engine.Engine.initOnDevice(gpa, io, bundle(), args[1], 256, 30_000_000_000, md.dev, rt, null);
    // no engine.deinit(): the mock runtime's library handle is a dummy and
    // teardown would close it; the process exit frees everything.
    const engine = engine_ptr;
    std.debug.print("DRYRUN loaded dispatches={d} barriers={d}\\n", .{ md.mock.dispatchCount(), md.mock.barrierCount() });

    // The GPU never executes: the greedy verdict fails after every draw's
    // stream is recorded. The harness counts the refusals, resets (recorded,
    // deterministic) and feeds 0 on; numerics are not what it can see.
    var verdict_failures: u64 = 0;
    var feed: u32 = 0;
    q35v.vk.orExit(engine.prefill(ids)) catch |err| {
        std.debug.print("FAIL prefill: {s}\n", .{@errorName(err)});
        return 1;
    };
    const load_d = md.mock.dispatchCount();
    const load_b = md.mock.barrierCount();
    std.debug.print("DRYRUN prefill dispatches={d} barriers={d}\n", .{ load_d, load_b });

    if (q35v.vk.orExit(engine.draw())) |id| {
        feed = id;
    } else |_| {
        verdict_failures += 1;
        try q35v.vk.orExit(engine.reset());
        feed = 0;
    }
    var step: u64 = 0;
    while (step < 3) : (step += 1) {
        if (q35v.vk.orExit(engine.decodeDraw(feed, null))) |id| {
            feed = id;
        } else |err| {
            std.debug.print("DRYRUN decode step failed: {s}\n", .{@errorName(err)});
            verdict_failures += 1;
            try q35v.vk.orExit(engine.reset());
            feed = 0;
        }
    }
    const d = md.mock.dispatchCount();
    const b = md.mock.barrierCount();
    std.debug.print("DRYRUN decode 3 tokens dispatches={d} barriers={d} verdict_failures={d}\n", .{ d - load_d, b - load_b, verdict_failures });

    var sha: [32]u8 = undefined;
    try md.mock.logSha(&sha);
    std.debug.print("DRYRUN total dispatches={d} barriers={d} entries={d} log_sha256={s}\\n", .{
        d, b, md.mock.entries.items.len, &std.fmt.bytesToHex(sha, .lower),
    });
    return 0;
}

fn bundle() q35v.engine.ShaderBundle {
    return .{
        .ops = .{
            .qmm_bf16 = &q35s.qmm_bf16,
            .qmm_q4_pk = &q35s.qmm_q4_pk,
            .dequant_bf16 = &q35s.dequant_bf16,
            .gather_u32 = &q35s.gather_u32,
            .gather_u16 = &q35s.gather_u16,
            .cast_bf16_f32 = &q35s.cast_bf16_f32,
            .cast_f32_bf16 = &q35s.cast_f32_bf16,
            .copy_u32 = &q35s.copy_u32,
            .copy_u16 = &q35s.copy_u16,
            .qmm_q4_sg = &q35s.qmm_q4_sg,
        },
        .norm = .{ .norm_bf16 = &q35s.norm_bf16, .add_bf16 = &q35s.add_bf16, .add_preserve_bf16 = &q35s.add_preserve_bf16, .swiglu_bf16 = &q35s.swiglu_bf16 },
        .attention = .{ .rope_bf16 = &q35s.rope_bf16, .sdpa_rows_bf16 = &q35s.sdpa_rows_bf16, .gate_bf16 = &q35s.gate_bf16 },
        .gdn = .{
            .conv_dw1d_bf16 = &q35s.conv_dw1d_bf16,
            .gdn_conv_decode_bf16 = &q35s.gdn_conv_decode_bf16,
            .gdn_split_bf16 = &q35s.gdn_split_bf16,
            .prefill_recur_bf16 = &q35s.prefill_recur_bf16,
            .gated_delta_decode_bf16 = &q35s.gated_delta_decode_bf16,
            .fast_norm_gated_bf16 = &q35s.fast_norm_gated_bf16,
            .copy_u16 = &q35s.copy_u16,
        },
        .greedy = .{ .finite_logits = &q35s.finite_logits, .argreduce_chunk = &q35s.argreduce_chunk, .argreduce_combine = &q35s.argreduce_combine },
    };
}
