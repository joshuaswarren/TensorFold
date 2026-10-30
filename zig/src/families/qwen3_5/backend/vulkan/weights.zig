// Qwen3.5-2B Vulkan: fail-closed Q4 admission before reads; raw BF16 norms; heap views; staged uploads use orExit.
const std = @import("std");
const vk = @import("vulkan");
const pinned = @import("../../config.zig");
const loader = @import("../../loader.zig");

const Allocator = std.mem.Allocator;
const Buffer = vk.Buffer;
const Commands = vk.Commands;
const Device = vk.Device;
const Table = loader.table.Table;
const Tensor = loader.table.Tensor;
const View = vk.tensor.View;

pub const Error = loader.table.Error || pinned.Error || vk.runtime.Error ||
    error{UnsupportedModel};

/// The text tower's key prefix, as the shared loader names it.
const prefix = loader.checkpoint.prefix;

// A `Quant` stores packed affine words, per-group scales and biases, plus row/column geometry checked by qmm.
pub const Quant = struct {
    w: View,
    scales: View,
    biases: View,
    n: u64,
    k: u64,
};

pub const DenseMlp = struct {
    gate: Quant,
    up: Quant,
    down: Quant,
    /// gate and up concatenated for the one-row decode pass.
    gate_up: ?Group = null,
};

/// One fused decode group: the members' weight rows concatenated into three
/// shared buffers (the member `Quant`s are views into them), so the decode
/// path can run the group as one projection. `starts` are the members' first
/// output rows in the fused output (element offsets for a one-row batch).
pub const Group = struct {
    fused: Quant,
    starts: [vk.fuse.max_members]u64 = undefined,
    n_members: u32 = 0,

    /// Member i as a standalone Quant over its slice of the shared buffers.
    pub fn member(g: *const Group, i: u32, rows: usize) Error!Quant {
        const k = g.fused.k;
        return .{
            .w = try g.fused.w.slice(g.starts[i] * (k / 8), rows * k / 8),
            .scales = try g.fused.scales.slice(g.starts[i] * (k / 64), rows * k / 64),
            .biases = try g.fused.biases.slice(g.starts[i] * (k / 64), rows * k / 64),
            .n = rows,
            .k = k,
        };
    }
};

// Full attention uses Q4/raw BF16; `q` has 4096 interleaved query/gate rows (8×256), split by forward to match Metal.
pub const FullLayer = struct {
    input_norm: View,
    post_norm: View,
    q: Quant,
    k: Quant,
    v: Quant,
    o: Quant,
    q_norm: View,
    k_norm: View,
    mlp: DenseMlp,
    /// q, k and v of one layer concatenated for the one-row decode pass; the
    /// member Quant views above point into the same buffers.
    qkv: ?Group = null,
};

// GatedDeltaNet keeps raw BF16 taps/norms/`dt_bias`; `conv_dw1d` widens uint16, and `A_log` stays raw F32.
pub const LinearLayer = struct {
    input_norm: View,
    post_norm: View,
    qkv: Quant,
    z: Quant,
    a: Quant,
    b: Quant,
    out: Quant,
    gnorm: View,
    conv: View,
    a_log: View,
    dt_bias: View,
    mlp: DenseMlp,
    /// qkv, z, a and b concatenated for the one-row decode pass.
    qkvzab: ?Group = null,
};

pub const Layer = union(enum) { full: FullLayer, linear: LinearLayer };

// The one supported model, checked field by field against the pinned family constants before anything is trusted.
pub fn expectPinnedModel(s: loader.config.Spec) Error!void {
    const expected = .{
        .hidden = pinned.hidden,
        .vocab = pinned.vocab,
        .n_layers = pinned.layers,
        .intermediate = pinned.intermediate,
        .heads = pinned.query_heads,
        .kv_heads = pinned.kv_heads,
        .head_dim = pinned.head_dim,
        .key_heads = pinned.linear_heads,
        .value_heads = pinned.linear_heads,
        // Spec stores per-head dimensions and conv taps, not channel widths.
        .key_dim = pinned.linear_dim,
        .value_dim = pinned.linear_dim,
        .conv = pinned.conv_taps,
        .rotary_dim = pinned.rotary_dim,
        .full_every = 4,
        .bits = 4,
        .group = pinned.group,
        .experts = 0,
    };
    inline for (@typeInfo(@TypeOf(expected)).@"struct".field_names) |field| {
        if (@field(s, field) != @field(expected, field)) {
            std.debug.print("Vulkan weights admission: Spec.{s}: expected {d}, found {d}\n", .{ field, @field(expected, field), @field(s, field) });
            return error.UnsupportedModel;
        }
    }
}

comptime {
    if (pinned.layers % 4 != 0) @compileError("the pinned layer count must keep the full-attention interval exact");
    if (pinned.conv_dim != 2 * pinned.linear_heads * pinned.linear_dim + pinned.linear_heads * pinned.linear_dim)
        @compileError("conv channels drifted from 2*NK*DK + NV*DV");
    if (pinned.query_heads * pinned.head_dim * 2 != 4096) @compileError("qraw row count drifted from the interleaved q/gate layout");
}

fn kindOf(dt: loader.table.DType) ?vk.tensor.Kind {
    return switch (dt) {
        .f32 => .f32,
        .bf16 => .bf16,
        .u32 => .u32,
        else => null,
    };
}

/// A checked exact dtype/shape, logged by key when it refuses.
fn expectTensor(key: []const u8, t: Tensor, dt: loader.table.DType, shape: []const usize) Error!void {
    if (!t.is(dt, shape)) {
        std.log.err("{s}: expected {s} {any}, found {s} {any}", .{ key, @tagName(dt), shape, @tagName(t.dtype), t.shape[0..t.rank] });
        return error.UnsupportedModel;
    }
}

// `Builder` records heap-owned buffers; `deinit` frees created buffers on errors, preventing leaks and double-frees.
const Builder = struct {
    gpa: Allocator,
    dev: *const Device,
    cmds: *Commands,
    staging: Buffer,
    timeout_ns: u64,
    table: *const Table,
    keep: std.ArrayList(*Buffer) = .empty,
    /// TF_VK_FUSE: concatenate the independent decode groups at load time.
    fuse_groups: bool = false,

    fn deinit(b: *Builder) void {
        for (b.keep.items) |ptr| {
            var buf = ptr.*;
            buf.deinit();
            b.gpa.destroy(ptr);
        }
        b.keep.deinit(b.gpa);
        b.* = undefined;
    }

    fn key(buf: *[192]u8, comptime fmt: []const u8, args: anytype) Error![]const u8 {
        // Layer prefixes can point into buf: format elsewhere before copying.
        var scratch: [192]u8 = undefined;
        const name = std.fmt.bufPrint(&scratch, fmt, args) catch return error.UnsupportedModel;
        @memcpy(buf[0..name.len], name);
        return buf[0..name.len];
    }

    /// The named tensor, dtype-checked before its mapped bytes are touched.
    fn named(b: *Builder, name: []const u8) Error!Tensor {
        const t = b.table.get(name) catch |err| {
            std.debug.print("Vulkan weights tensor {s}: {s}\n", .{ name, @errorName(err) });
            return err;
        };
        switch (t.dtype) {
            .f32, .bf16, .u32 => {},
            else => {
                std.log.err("{s}: unsupported tensor dtype {s}", .{ name, @tagName(t.dtype) });
                return error.UnsupportedModel;
            },
        }
        return t;
    }

    /// A heap-owned registered device buffer of `bytes`, not yet written.
    fn device(b: *Builder, bytes: u64) !*Buffer {
        const ptr = try b.gpa.create(Buffer);
        ptr.* = Buffer.init(b.dev, .device, bytes) catch |e| {
            b.gpa.destroy(ptr);
            return e;
        };
        b.keep.append(b.gpa, ptr) catch |e| {
            var buf = ptr.*;
            buf.deinit();
            b.gpa.destroy(ptr);
            return e;
        };
        return ptr;
    }

    /// A device buffer holding `t`'s bytes exactly, streamed through the
    /// staging buffer; the returned view is typed and word-padded by `View`.
    ///
    /// Ownership transfers exactly once: until the slot is registered in
    /// `keep`, this function alone cleans up (slot plus the Vk allocation) on
    /// every failure; after registration only `Builder.deinit` frees it, so a
    /// failed upload never double-frees or leaves a dangling slot.
    fn tensor(b: *Builder, t: Tensor) !View {
        const kind = kindOf(t.dtype) orelse return error.UnsupportedModel;
        const elements: u64 = @intCast(t.numel());
        const bytes: u64 = @intCast(t.bytes.len);
        const ptr = try b.device(bytes);
        // registered: in-flight failures exit the process here (orExit),
        // never unwinding into the frees below this load's scope
        try vk.orExit(vk.upload(b.cmds, ptr.*, 0, b.staging, t.bytes, b.timeout_ns));
        return View.init(ptr, kind, 0, elements);
    }

    /// A raw tensor of the exact given dtype and shape, uploaded unchanged.
    fn raw(b: *Builder, name: []const u8, dt: loader.table.DType, shape: []const usize) !View {
        const t = try b.named(name);
        try expectTensor(name, t, dt, shape);
        return b.tensor(t);
    }

    // For `n` rows, `k` inputs, group 64: checked U32 `[n,k/8]` words and BF16 scales/biases `[n,k/64]`.
    fn quant(b: *Builder, name: []const u8, n: usize, k: usize) !Quant {
        var buf: [192]u8 = undefined;
        const w = try b.raw(try key(&buf, "{s}.weight", .{name}), .u32, &.{ n, k / 8 });
        const scales = try b.raw(try key(&buf, "{s}.scales", .{name}), .bf16, &.{ n, k / 64 });
        const biases = try b.raw(try key(&buf, "{s}.biases", .{name}), .bf16, &.{ n, k / 64 });
        return .{ .w = w, .scales = scales, .biases = biases, .n = n, .k = k };
    }

    /// The members of one decode group as rows of three shared buffers, with
    /// the pad rows the alignment plan asks for zeroed. Each member keeps a
    /// full `Quant` (views into the shared buffers), so the prefill path and
    /// the separate-dispatch fallback read the exact same bytes. The fused
    /// whole rides alongside for the one-row pass. A group past the device's
    /// range limits refuses here: the caller loads the members separately.
    fn groupQuant(b: *Builder, spec: []const struct { name: []const u8, rows: usize }, k: usize) !Group {
        const lim: vk.ops_q4.Limits = .{
            .max_range = b.dev.properties.limits.maxStorageBufferRange,
            .alignment = b.dev.properties.limits.minStorageBufferOffsetAlignment,
        };
        var rows: [vk.fuse.max_members]u64 = undefined;
        for (spec, 0..) |m, i| rows[i] = m.rows;
        const plan = try vk.fuse.planGroup(@intCast(k), rows[0..spec.len], lim);
        const w_buf = try b.device(plan.w_bytes);
        const s_buf = try b.device(plan.params_bytes);
        const b_buf = try b.device(plan.params_bytes);
        // the pad rows read as zero weights, zero scales, zero biases
        try vk.orExit(vk.zero(b.cmds, w_buf.*, 0, plan.w_bytes, b.timeout_ns));
        try vk.orExit(vk.zero(b.cmds, s_buf.*, 0, plan.params_bytes, b.timeout_ns));
        try vk.orExit(vk.zero(b.cmds, b_buf.*, 0, plan.params_bytes, b.timeout_ns));
        var g: Group = .{ .fused = undefined, .n_members = @intCast(spec.len) };
        for (spec, 0..) |m, i| {
            var kb: [192]u8 = undefined;
            const w_name = try key(&kb, "{s}.weight", .{m.name});
            const t_w = try b.named(w_name);
            try expectTensor(w_name, t_w, .u32, &.{ m.rows, k / 8 });
            try vk.orExit(vk.upload(b.cmds, w_buf.*, plan.members[i].start * (k / 2), b.staging, t_w.bytes, b.timeout_ns));
            var kbs: [192]u8 = undefined;
            const s_name = try key(&kbs, "{s}.scales", .{m.name});
            const t_s = try b.named(s_name);
            try expectTensor(s_name, t_s, .bf16, &.{ m.rows, k / 64 });
            try vk.orExit(vk.upload(b.cmds, s_buf.*, plan.members[i].start * (k / 64) * 2, b.staging, t_s.bytes, b.timeout_ns));
            var kbb: [192]u8 = undefined;
            const b_name = try key(&kbb, "{s}.biases", .{m.name});
            const t_b = try b.named(b_name);
            try expectTensor(b_name, t_b, .bf16, &.{ m.rows, k / 64 });
            try vk.orExit(vk.upload(b.cmds, b_buf.*, plan.members[i].start * (k / 64) * 2, b.staging, t_b.bytes, b.timeout_ns));
            g.starts[i] = plan.members[i].start;
        }
        g.fused = .{
            .w = try View.init(w_buf, .u32, 0, plan.total_rows * k / 8),
            .scales = try View.init(s_buf, .bf16, 0, plan.total_rows * k / 64),
            .biases = try View.init(b_buf, .bf16, 0, plan.total_rows * k / 64),
            .n = plan.total_rows,
            .k = k,
        };
        return g;
    }

    /// A raw BF16 norm gain, passed through byte for byte (the Metal oracle's
    /// semantics: the kernel multiplies the gain directly, no `1 + w`).
    fn norm(b: *Builder, name: []const u8, elements: usize) !View {
        return b.raw(name, .bf16, &.{elements});
    }

    fn mlp(b: *Builder, base: []const u8) !DenseMlp {
        var buf: [192]u8 = undefined;
        var buf2: [192]u8 = undefined;
        var buf3: [192]u8 = undefined;
        const gate_name = try key(&buf, "{s}mlp.gate_proj", .{base});
        const up_name = try key(&buf2, "{s}mlp.up_proj", .{base});
        const down_name = try key(&buf3, "{s}mlp.down_proj", .{base});
        if (b.fuse_groups) {
            if (b.groupQuant(&.{ .{ .name = gate_name, .rows = pinned.intermediate }, .{ .name = up_name, .rows = pinned.intermediate } }, pinned.hidden)) |g| {
                return .{
                    .gate = try g.member(0, pinned.intermediate),
                    .up = try g.member(1, pinned.intermediate),
                    .down = try b.quant(down_name, pinned.hidden, pinned.intermediate),
                    .gate_up = g,
                };
            } else |err| switch (err) {
                error.TooLarge => {}, // the merged weight does not fit the device: separate members
                else => return err,
            }
        }
        return .{
            .gate = try b.quant(gate_name, pinned.intermediate, pinned.hidden),
            .up = try b.quant(up_name, pinned.intermediate, pinned.hidden),
            .down = try b.quant(down_name, pinned.hidden, pinned.intermediate),
        };
    }

    fn layer(b: *Builder, index: usize) !Layer {
        var buf: [192]u8 = undefined;
        const base = try key(&buf, prefix ++ "layers.{d}.", .{index});
        const input_norm = try b.norm(try key(&buf, "{s}input_layernorm.weight", .{base}), pinned.hidden);
        const post_norm = try b.norm(try key(&buf, "{s}post_attention_layernorm.weight", .{base}), pinned.hidden);
        const m = try b.mlp(base);
        if (pinned.linear(index)) {
            const lin = try key(&buf, "{s}linear_attn.", .{base});
            // The mixer members load first, in the separate layout's order,
            // so the FUSE=0 stream matches the unit 12 tree byte for byte.
            var l: LinearLayer = .{
                .input_norm = input_norm,
                .post_norm = post_norm,
                .qkv = undefined,
                .z = undefined,
                .a = undefined,
                .b = undefined,
                .out = undefined,
                .gnorm = undefined,
                .conv = undefined,
                .a_log = undefined,
                .dt_bias = undefined,
                .mlp = m,
            };
            var done = false;
            if (b.fuse_groups) {
                // one key buffer per member: shared buffers alias earlier
                // slices with later (shorter) writes — in_proj_bkv
                var kb0: [192]u8 = undefined;
                var kb1: [192]u8 = undefined;
                var kb2: [192]u8 = undefined;
                var kb3: [192]u8 = undefined;
                if (b.groupQuant(&.{
                    .{ .name = try key(&kb0, "{s}in_proj_qkv", .{lin}), .rows = 3 * pinned.linear_heads * pinned.linear_dim },
                    .{ .name = try key(&kb1, "{s}in_proj_z", .{lin}), .rows = pinned.linear_heads * pinned.linear_dim },
                    .{ .name = try key(&kb2, "{s}in_proj_a", .{lin}), .rows = pinned.linear_heads },
                    .{ .name = try key(&kb3, "{s}in_proj_b", .{lin}), .rows = pinned.linear_heads },
                }, pinned.hidden)) |g| {
                    l.qkv = try g.member(0, 3 * pinned.linear_heads * pinned.linear_dim);
                    l.z = try g.member(1, pinned.linear_heads * pinned.linear_dim);
                    l.a = try g.member(2, pinned.linear_heads);
                    l.b = try g.member(3, pinned.linear_heads);
                    l.qkvzab = g;
                    done = true;
                } else |err| switch (err) {
                    error.TooLarge => {}, // the merged weight does not fit the device: separate members
                    else => return err,
                }
            }
            if (!done) {
                l.qkv = try b.quant(try key(&buf, "{s}in_proj_qkv", .{lin}), 3 * pinned.linear_heads * pinned.linear_dim, pinned.hidden);
                l.z = try b.quant(try key(&buf, "{s}in_proj_z", .{lin}), pinned.linear_heads * pinned.linear_dim, pinned.hidden);
                l.a = try b.quant(try key(&buf, "{s}in_proj_a", .{lin}), pinned.linear_heads, pinned.hidden);
                l.b = try b.quant(try key(&buf, "{s}in_proj_b", .{lin}), pinned.linear_heads, pinned.hidden);
            }
            l.out = try b.quant(try key(&buf, "{s}out_proj", .{lin}), pinned.hidden, pinned.linear_heads * pinned.linear_dim);
            l.gnorm = try b.norm(try key(&buf, "{s}norm.weight", .{lin}), pinned.linear_dim);
            l.conv = try b.raw(try key(&buf, "{s}conv1d.weight", .{lin}), .bf16, &.{ pinned.conv_dim, pinned.conv_taps, 1 });
            l.a_log = try b.raw(try key(&buf, "{s}A_log", .{lin}), .f32, &.{pinned.linear_heads});
            l.dt_bias = try b.raw(try key(&buf, "{s}dt_bias", .{lin}), .bf16, &.{pinned.linear_heads});
            return .{ .linear = l };
        }
        const attn = try key(&buf, "{s}self_attn.", .{base});
        var f: FullLayer = .{
            .input_norm = input_norm,
            .post_norm = post_norm,
            .q = undefined,
            .k = undefined,
            .v = undefined,
            .o = undefined,
            .q_norm = undefined,
            .k_norm = undefined,
            .mlp = m,
        };
        var done = false;
        if (b.fuse_groups) {
            var kb0: [192]u8 = undefined;
            var kb1: [192]u8 = undefined;
            var kb2: [192]u8 = undefined;
            if (b.groupQuant(&.{
                .{ .name = try key(&kb0, "{s}q_proj", .{attn}), .rows = 2 * pinned.query_heads * pinned.head_dim },
                .{ .name = try key(&kb1, "{s}k_proj", .{attn}), .rows = pinned.kv_heads * pinned.head_dim },
                .{ .name = try key(&kb2, "{s}v_proj", .{attn}), .rows = pinned.kv_heads * pinned.head_dim },
            }, pinned.hidden)) |g| {
                f.q = try g.member(0, 2 * pinned.query_heads * pinned.head_dim);
                f.k = try g.member(1, pinned.kv_heads * pinned.head_dim);
                f.v = try g.member(2, pinned.kv_heads * pinned.head_dim);
                f.qkv = g;
                done = true;
            } else |err| switch (err) {
                error.TooLarge => {},
                else => return err,
            }
        }
        if (!done) {
            f.q = try b.quant(try key(&buf, "{s}q_proj", .{attn}), 2 * pinned.query_heads * pinned.head_dim, pinned.hidden);
            f.k = try b.quant(try key(&buf, "{s}k_proj", .{attn}), pinned.kv_heads * pinned.head_dim, pinned.hidden);
            f.v = try b.quant(try key(&buf, "{s}v_proj", .{attn}), pinned.kv_heads * pinned.head_dim, pinned.hidden);
        }
        f.o = try b.quant(try key(&buf, "{s}o_proj", .{attn}), pinned.hidden, pinned.query_heads * pinned.head_dim);
        f.q_norm = try b.norm(try key(&buf, "{s}q_norm.weight", .{attn}), pinned.head_dim);
        f.k_norm = try b.norm(try key(&buf, "{s}k_norm.weight", .{attn}), pinned.head_dim);
        return .{ .full = f };
    }
};

// Typed text-tower views use owned heap buffers; checkpoint config decides whether the head ties to the embedding.
pub const Model = struct {
    gpa: Allocator,
    spec: loader.config.Spec,
    /// config.json's max_position_embeddings, as admission pinned it.
    context: usize,
    /// The head is the packed input embedding (`tie_word_embeddings`).
    tied: bool,
    buffers: []*Buffer,
    embed: Quant,
    final_norm: View,
    // When tied, `outputHead` returns `embed`; untied checkpoints require a separate head or loading is refused.
    head: ?Quant,
    layers: []Layer,
    /// TF_VK_FUSE's load-time decision: the groups are laid out fused.
    fuse_switch: bool,

    // Loads `dir` via `cmds` with nonempty `staging`; it checks admission first and keeps heap-backed views valid.
    pub fn load(
        gpa: Allocator,
        dev: *const Device,
        cmds: *Commands,
        io: std.Io,
        dir: []const u8,
        staging: Buffer,
        timeout_ns: u64,
    ) !Model {
        // Admission pins model type, geometry, quantization, and tying from `config.json` before other reads.
        const cfg = pinned.Config.read(gpa, io, dir) catch |err| {
            std.debug.print("Vulkan weights config.json admission: {s}\n", .{@errorName(err)});
            return err;
        };
        var ck = loader.checkpoint.Checkpoint.open(gpa, io, dir) catch |err| {
            std.debug.print("Vulkan weights Checkpoint.open: {s}\n", .{@errorName(err)});
            return err;
        };
        defer ck.close();
        try expectPinnedModel(ck.spec());

        var b: Builder = .{
            .gpa = gpa,
            .dev = dev,
            .cmds = cmds,
            .staging = staging,
            .timeout_ns = timeout_ns,
            .table = &ck.main,
            // TF_VK_FUSE, read once: the concatenation decision is made here,
            // when the weights are laid out. 0 loads exactly as today.
            .fuse_groups = vk.fuse.fuseFromEnv(std.c.getenv("TF_VK_FUSE")),
        };
        defer b.deinit();

        var layers: std.ArrayList(Layer) = .empty;
        errdefer layers.deinit(gpa);

        // Tied output uses the embedding; untied checkpoints require an existing `lm_head` or loading is refused.
        const head: ?Quant = if (cfg.tied) null else try b.quant("language_model.lm_head", pinned.vocab, pinned.hidden);

        const embed = try b.quant(prefix ++ "embed_tokens", pinned.vocab, pinned.hidden);
        const final_norm = try b.norm(prefix ++ "norm.weight", pinned.hidden);
        // Mapped pages are released per part, so host RSS holds one checkpoint layer, not the full file set.
        ck.release(null);
        for (0..pinned.layers) |i| {
            try layers.append(gpa, try b.layer(i));
            ck.release(i);
        }

        const owned_layers = try layers.toOwnedSlice(gpa);
        errdefer gpa.free(owned_layers);
        // ownership moves to the model; the builder's defer is now a no-op
        const buffers = try b.keep.toOwnedSlice(gpa);
        return .{
            .gpa = gpa,
            .spec = ck.spec(),
            .context = cfg.context,
            .tied = cfg.tied,
            .buffers = buffers,
            .embed = embed,
            .final_norm = final_norm,
            .head = head,
            .layers = owned_layers,
            .fuse_switch = b.fuse_groups,
        };
    }

    /// The output head: the embedding itself when the checkpoint ties them.
    pub fn outputHead(m: *const Model) Quant {
        return m.head orelse m.embed;
    }

    // Free device buffers only when no submission can touch a view; in-flight errors exit through `orExit`.
    pub fn deinit(m: *Model) void {
        for (m.buffers) |ptr| {
            var buf = ptr.*;
            buf.deinit();
            m.gpa.destroy(ptr);
        }
        m.gpa.free(m.buffers);
        m.gpa.free(m.layers);
        m.* = undefined;
    }
};

const testing = std.testing;

/// One tensor the loader will request: name, dtype and shape, all checkable
/// against a safetensors header without a GPU.
pub const Request = struct { name: []const u8, dtype: loader.table.DType, shape: [4]usize, rank: usize };

/// The layer prefix's tensor name, formatted into `buf` (the Builder's key).
pub fn formatKey(buf: *[192]u8, comptime fmt: []const u8, args: anytype) Error![]const u8 {
    var scratch: [192]u8 = undefined;
    const name = std.fmt.bufPrint(&scratch, fmt, args) catch return error.UnsupportedModel;
    @memcpy(buf[0..name.len], name);
    return buf[0..name.len];
}

const QuantShape = struct { w: [2]usize, p: [2]usize };

fn quantShape(rows: usize, k: usize) QuantShape {
    return .{ .w = .{ rows, k / 8 }, .p = .{ rows, k / 64 } };
}

fn pushQuant(gpa: Allocator, out: *std.ArrayList(Request), name: []const u8, rows: usize, k: usize) Error!void {
    const s = quantShape(rows, k);
    try out.append(gpa, .{ .name = try std.fmt.allocPrint(gpa, "{s}.weight", .{name}), .dtype = .u32, .shape = .{ s.w[0], s.w[1], 0, 0 }, .rank = 2 });
    try out.append(gpa, .{ .name = try std.fmt.allocPrint(gpa, "{s}.scales", .{name}), .dtype = .bf16, .shape = .{ s.p[0], s.p[1], 0, 0 }, .rank = 2 });
    try out.append(gpa, .{ .name = try std.fmt.allocPrint(gpa, "{s}.biases", .{name}), .dtype = .bf16, .shape = .{ s.p[0], s.p[1], 0, 0 }, .rank = 2 });
}

fn addRequest(gpa: Allocator, out: *std.ArrayList(Request), buf: *[192]u8, comptime fmt: []const u8, args: anytype, dtype: loader.table.DType, shape: [4]usize, rank: usize) Error!void {
    try out.append(gpa, .{ .name = try gpa.dupe(u8, try formatKey(buf, fmt, args)), .dtype = dtype, .shape = shape, .rank = rank });
}

/// Every tensor the loader requests for layer `index`: names, dtypes,
/// shapes. Names are heap copies, so the list outlives this call. The caller
/// frees.
pub fn layerRequests(gpa: Allocator, index: usize) Error![]Request {
    var out: std.ArrayList(Request) = .empty;
    errdefer out.deinit(gpa);
    var buf: [192]u8 = undefined;
    const base = try formatKey(&buf, prefix ++ "layers.{d}.", .{index});
    {
        var nb: [192]u8 = undefined;
        try addRequest(gpa, &out, &nb, "{s}input_layernorm.weight", .{base}, .bf16, .{ pinned.hidden, 0, 0, 0 }, 1);
        try addRequest(gpa, &out, &nb, "{s}post_attention_layernorm.weight", .{base}, .bf16, .{ pinned.hidden, 0, 0, 0 }, 1);
    }
    {
        var b0: [192]u8 = undefined;
        var b1: [192]u8 = undefined;
        var b2: [192]u8 = undefined;
        const gate = try formatKey(&b0, "{s}mlp.gate_proj", .{base});
        const up = try formatKey(&b1, "{s}mlp.up_proj", .{base});
        const down = try formatKey(&b2, "{s}mlp.down_proj", .{base});
        try pushQuant(gpa, &out, gate, pinned.intermediate, pinned.hidden);
        try pushQuant(gpa, &out, up, pinned.intermediate, pinned.hidden);
        try pushQuant(gpa, &out, down, pinned.hidden, pinned.intermediate);
    }
    if (pinned.linear(index)) {
        var lb: [192]u8 = undefined;
        const lin = try formatKey(&lb, "{s}linear_attn.", .{base});
        var kb: [192]u8 = undefined;
        try pushQuant(gpa, &out, try formatKey(&kb, "{s}out_proj", .{lin}), pinned.hidden, pinned.linear_heads * pinned.linear_dim);
        {
            var nb: [192]u8 = undefined;
            try addRequest(gpa, &out, &nb, "{s}norm.weight", .{lin}, .bf16, .{ pinned.linear_dim, 0, 0, 0 }, 1);
        }
        {
            var nb: [192]u8 = undefined;
            try addRequest(gpa, &out, &nb, "{s}conv1d.weight", .{lin}, .bf16, .{ pinned.conv_dim, pinned.conv_taps, 1, 0 }, 3);
            try addRequest(gpa, &out, &nb, "{s}A_log", .{lin}, .f32, .{ pinned.linear_heads, 0, 0, 0 }, 1);
            try addRequest(gpa, &out, &nb, "{s}dt_bias", .{lin}, .bf16, .{ pinned.linear_heads, 0, 0, 0 }, 1);
        }
        // the fused arm's four members each get their OWN key buffer: the
        // loader's group spec held four slices of one buffer, and the last
        // (shortest) write left stale tails the earlier slices read back as
        // in_proj_bkv
        var kb0: [192]u8 = undefined;
        var kb1: [192]u8 = undefined;
        var kb2: [192]u8 = undefined;
        var kb3: [192]u8 = undefined;
        const qkv = try formatKey(&kb0, "{s}in_proj_qkv", .{lin});
        const z = try formatKey(&kb1, "{s}in_proj_z", .{lin});
        const a = try formatKey(&kb2, "{s}in_proj_a", .{lin});
        const b = try formatKey(&kb3, "{s}in_proj_b", .{lin});
        try pushQuant(gpa, &out, qkv, 3 * pinned.linear_heads * pinned.linear_dim, pinned.hidden);
        try pushQuant(gpa, &out, z, pinned.linear_heads * pinned.linear_dim, pinned.hidden);
        try pushQuant(gpa, &out, a, pinned.linear_heads, pinned.hidden);
        try pushQuant(gpa, &out, b, pinned.linear_heads, pinned.hidden);
    } else {
        var ab: [192]u8 = undefined;
        const attn = try formatKey(&ab, "{s}self_attn.", .{base});
        var qb: [192]u8 = undefined;
        var kbb: [192]u8 = undefined;
        var vb: [192]u8 = undefined;
        const q = try formatKey(&qb, "{s}q_proj", .{attn});
        const k = try formatKey(&kbb, "{s}k_proj", .{attn});
        const v = try formatKey(&vb, "{s}v_proj", .{attn});
        try pushQuant(gpa, &out, q, 2 * pinned.query_heads * pinned.head_dim, pinned.hidden);
        try pushQuant(gpa, &out, k, pinned.kv_heads * pinned.head_dim, pinned.hidden);
        try pushQuant(gpa, &out, v, pinned.kv_heads * pinned.head_dim, pinned.hidden);
        {
            var nb: [192]u8 = undefined;
            try addRequest(gpa, &out, &nb, "{s}q_norm.weight", .{attn}, .bf16, .{ pinned.head_dim, 0, 0, 0 }, 1);
            try addRequest(gpa, &out, &nb, "{s}k_norm.weight", .{attn}, .bf16, .{ pinned.head_dim, 0, 0, 0 }, 1);
        }
        try pushQuant(gpa, &out, try formatKey(&ab, "{s}o_proj", .{attn}), pinned.hidden, pinned.query_heads * pinned.head_dim);
    }
    return out.toOwnedSlice(gpa);
}

test "a group member slices the shared weight buffers by whole rows, not by row indices" {
    const k: u64 = 2048;
    const total: u64 = 8336; // the GDN plan: qkv 6144, z 2048, a 16, b 16 with the pad rows
    const w_len: u64 = total * k / 2; // 4 bits an element
    const p_len: u64 = total * k / 32; // one bf16 scale or bias per 64 elements
    const wbuf: Buffer = .{ .dev = undefined, .handle = null, .memory = null, .len = w_len, .kind = .device, .mapped = null, .coherent = false };
    const sbuf: Buffer = .{ .dev = undefined, .handle = null, .memory = null, .len = p_len, .kind = .device, .mapped = null, .coherent = false };
    const bbuf: Buffer = .{ .dev = undefined, .handle = null, .memory = null, .len = p_len, .kind = .device, .mapped = null, .coherent = false };
    var g: Group = .{
        .fused = .{
            .w = try View.init(&wbuf, .u32, 0, total * k / 8),
            .scales = try View.init(&sbuf, .bf16, 0, total * k / 64),
            .biases = try View.init(&bbuf, .bf16, 0, total * k / 64),
            .n = total,
            .k = k,
        },
        .n_members = 4,
    };
    g.starts[0] = 0;
    g.starts[1] = 6144;
    g.starts[2] = 8192;
    g.starts[3] = 8320;
    const rows = [_]usize{ 6144, 2048, 16, 16 };
    for (rows, 0..) |r, i| {
        const m = try g.member(@intCast(i), r);
        try testing.expectEqual(g.starts[i] * k / 2, m.w.byte_offset);
        try testing.expectEqual(g.starts[i] * k / 32, m.scales.byte_offset);
        try testing.expectEqual(g.starts[i] * k / 32, m.biases.byte_offset);
        try testing.expectEqual(@as(u64, r) * k / 8, m.w.elements);
        try testing.expectEqual(@as(u64, r), m.n);
    }
}

test "the fused GDN group resolves every member name against a shared key buffer" {
    // the loader's group spec holds four names built into one buffer; the
    // member the loader touches first must read back as in_proj_qkv, not a
    // stale-tail mash like in_proj_bkv
    const a = testing.allocator;
    const reqs = try layerRequests(a, 0);
    defer {
        for (reqs) |r| a.free(r.name);
        a.free(reqs);
    }
    var qkv_seen = false;
    for (reqs) |r| {
        if (std.mem.indexOf(u8, r.name, "in_proj_") != null and std.mem.endsWith(u8, r.name, ".weight")) {
            const ok = std.mem.endsWith(u8, r.name, "in_proj_qkv.weight") or
                std.mem.endsWith(u8, r.name, "in_proj_z.weight") or
                std.mem.endsWith(u8, r.name, "in_proj_a.weight") or
                std.mem.endsWith(u8, r.name, "in_proj_b.weight");
            if (!ok) std.debug.print("BAD fused member name: {s}\n", .{r.name});
            try testing.expect(ok);
            if (std.mem.endsWith(u8, r.name, "in_proj_qkv.weight")) qkv_seen = true;
        }
    }
    try testing.expect(qkv_seen);
}

test "every requested tensor resolves against the real checkpoint header, both arms" {
    const path_env: ?[*:0]const u8 = std.c.getenv("TF_TEST_CHECKPOINT");
    const path: []const u8 = if (path_env) |p| std.mem.span(p) else {
        std.debug.print("SKIP real-checkpoint header test: set TF_TEST_CHECKPOINT to a model.safetensors\n", .{});
        return;
    };
    const file = std.Io.Dir.cwd().openFile(std.testing.io, path, .{}) catch |err| {
        std.debug.print("SKIP real-checkpoint header test: no checkpoint at {s} ({s})\n", .{ path, @errorName(err) });
        return;
    };
    defer file.close(std.testing.io);
    var len_bytes: [8]u8 = undefined;
    _ = try file.readPositionalAll(std.testing.io, &len_bytes, 0);
    const header_len = std.mem.readInt(u64, &len_bytes, .little);
    if (header_len > 64 << 20) return error.UnsupportedModel;
    const a = testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const header = try arena.alloc(u8, header_len);
    _ = try file.readPositionalAll(std.testing.io, header, 8);
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, arena, header, .{});
    for ([_]bool{ false, true }) |fuse| {
        for (0..pinned.layers) |i| {
            const reqs = try layerRequests(arena, i);
            for (reqs) |r| {
                const entry = doc.object.get(r.name) orelse {
                    std.debug.print("MISSING tensor (fuse={}): {s}\n", .{ fuse, r.name });
                    return error.TestUnexpectedResult;
                };
                const dt = entry.object.get("dtype").?.string;
                const shape = entry.object.get("shape").?.array;
                const want_dt: []const u8 = switch (r.dtype) {
                    .bf16 => "BF16",
                    .f32 => "F32",
                    .u32 => "U32",
                    else => "OTHER",
                };
                try testing.expectEqualStrings(want_dt, dt);
                try testing.expectEqual(@as(usize, r.rank), shape.items.len);
                for (r.shape[0..r.rank], 0..) |want, d| try testing.expectEqual(want, @as(usize, @intCast(shape.items[d].integer)));
            }
        }
    }
    // the head tensors, both arms (tied head: no lm_head request)
    for ([_][]const u8{ prefix ++ "embed_tokens.weight", prefix ++ "embed_tokens.scales", prefix ++ "embed_tokens.biases", prefix ++ "norm.weight" }) |name| {
        try testing.expect(doc.object.get(name) != null);
    }
}

test "kindOf maps exactly the dtypes the views carry" {
    try testing.expectEqual(@as(?vk.tensor.Kind, .f32), kindOf(.f32));
    try testing.expectEqual(@as(?vk.tensor.Kind, .bf16), kindOf(.bf16));
    try testing.expectEqual(@as(?vk.tensor.Kind, .u32), kindOf(.u32));
    try testing.expectEqual(@as(?vk.tensor.Kind, null), kindOf(.f16));
}

test "tensor keys can extend a prefix in the destination buffer" {
    var buf: [192]u8 = undefined;
    const base = try Builder.key(&buf, prefix ++ "layers.{d}.", .{0});
    const name = try Builder.key(&buf, "{s}input_layernorm.weight", .{base});
    try testing.expectEqualStrings(prefix ++ "layers.0.input_layernorm.weight", name);
    const oversized: [193]u8 = @splat('x');
    try testing.expectError(error.UnsupportedModel, Builder.key(&buf, "{s}", .{oversized[0..]}));
}

test "the real pinned config geometry is admitted and field drift is refused" {
    const bytes = @embedFile("qwen35_2b_config.json");
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    try testing.expectEqualStrings("beb7fc5a6e0405fe332821cf1a8ef7b69bb390a8c8933171647de5579debf949", &std.fmt.bytesToHex(digest, .lower));
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const doc = try pinned.declared(arena.allocator(), bytes);
    const s = try pinned.geometry(try pinned.textTower(doc), 4, 64);
    try testing.expectEqual(@as(usize, 128), s.key_dim);
    try testing.expectEqual(@as(usize, 128), s.value_dim);
    try testing.expectEqual(@as(usize, 4), s.conv);
    try expectPinnedModel(s);
    inline for (.{ "hidden", "vocab", "n_layers", "intermediate", "heads", "kv_heads", "head_dim", "key_heads", "value_heads", "key_dim", "value_dim", "conv", "rotary_dim", "full_every", "bits", "group", "experts" }) |field| {
        var drift = s;
        @field(drift, field) += 1;
        try testing.expectError(error.UnsupportedModel, expectPinnedModel(drift));
    }
}

test "tensor shape checks refuse rank and dim drift by name" {
    const t: Tensor = .{ .dtype = .bf16, .rank = 1, .shape = .{ 2048, 0, 0, 0, 0 }, .bytes = &.{} };
    try testing.expect(t.is(.bf16, &.{2048}));
    try testing.expect(!t.is(.f32, &.{2048}));
    try testing.expect(!t.is(.bf16, &.{2049}));
    try testing.expect(!t.is(.bf16, &.{ 2048, 1 }));
}
