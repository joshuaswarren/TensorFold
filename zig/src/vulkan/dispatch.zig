//! One compute pipeline over set-0 storage buffers and push constants: checked SPIR-V, limits, bindings and dispatch.
const std = @import("std");
const c = @import("vk");
const runtime = @import("runtime.zig");
const Device = @import("device.zig").Device;
const Buffer = @import("memory.zig").Buffer;
const commands = @import("commands.zig");

const check = runtime.check;

pub const Error = runtime.Error || error{ BadSpirv, NoLocalSize, TooManyBindings, BadPushSize, BadWorkgroup, BadGroupCount, BadBindings, Aliased, BatchOverflow, TimingOverflow, TimestampUnsupported };

pub const max_bindings = 10;

/// Per-dispatch GPU timestamps, off until a caller points `timing` at its
/// Timing. Null (the default) costs nothing: no query pool, no extra
/// commands, no allocations.
pub var timing: ?*Timing = null;

/// Query-pool capacity: the worst token records ~490 dispatch stamps plus
/// uploads between them; 16384 leaves order-of-magnitude headroom.
pub const timing_capacity: u32 = 16384;

/// What one timestamp followed: the dispatch's kernel (its SPIR-V image),
/// its push bytes (hash and first words) and its group counts.
pub const Stamp = struct {
    spirv_ptr: usize = 0,
    spirv_len: u32 = 0,
    push_hash: u64 = 0,
    push_words: [4]u32 = .{ 0, 0, 0, 0 },
    groups: [3]u32 = .{ 0, 0, 0 },
};

/// A Tally key: one kernel invocation shape. The push words ride along so a
/// report can name them; two dispatches with the same hash always share them.
pub const Key = struct { spirv: usize, push_hash: u64, push_words: [4]u32, groups: [3]u32 };

/// Where the deltas of one steady window accumulate, keyed by invocation
/// shape. Linear scan over a bounded table: a token's distinct shapes are
/// hundreds, adds happen once per dispatch after the fence, never in the
/// recording path.
pub const tally_cap: u32 = 4096;

pub const TallyEntry = struct { key: Key, spirv_len: u32, calls: u64 = 0, total_ns: u64 = 0 };

pub const Tally = struct {
    entries: [tally_cap]TallyEntry = undefined,
    n: u32 = 0,
    total_ns: u64 = 0,

    pub fn add(self: *Tally, key: Key, spirv_len: u32, ns: u64) Error!void {
        for (self.entries[0..self.n]) |*e| if (std.meta.eql(e.key, key)) {
            e.calls += 1;
            e.total_ns += ns;
            self.total_ns += ns;
            return;
        };
        if (self.n >= tally_cap) return error.TimingOverflow; // the caller logs the refusal
        self.entries[self.n] = .{ .key = key, .spirv_len = spirv_len, .calls = 1, .total_ns = ns };
        self.n += 1;
        self.total_ns += ns;
    }

    /// The fraction of the window's summed deltas this entry's total makes up.
    pub fn share(self: *const Tally, entry_total_ns: u64) f64 {
        if (self.total_ns == 0) return 0;
        return @as(f64, @floatFromInt(entry_total_ns)) / @as(f64, @floatFromInt(self.total_ns));
    }
};

pub const Phase = enum { record, flush, submit, fence_wait, invalidate, readdown, head_gpu, tail_gpu };

pub const Phases = struct {
    ns: [8]u64 = .{ 0, 0, 0, 0, 0, 0, 0, 0 },

    pub fn add(self: *Phases, phase: Phase, elapsed_ns: u64) void {
        self.ns[@intFromEnum(phase)] += elapsed_ns;
    }

    pub fn perTokenUs(self: *const Phases, phase: Phase, tokens: u64) f64 {
        if (tokens == 0) return 0;
        return @as(f64, @floatFromInt(self.ns[@intFromEnum(phase)])) / @as(f64, @floatFromInt(tokens)) / 1000.0;
    }
};

pub fn formatPhases(buf: []u8, run: []const u8, phases: Phases, tokens: u64) []const u8 {
    return std.fmt.bufPrint(buf, "KTIME phases run={s} per_token_us record={d:.2} flush={d:.2} submit={d:.2} fence_wait={d:.2} invalidate={d:.2} readdown={d:.2} tail_gpu={d:.2} head_gpu={d:.2}", .{
        run,
        phases.perTokenUs(.record, tokens),
        phases.perTokenUs(.flush, tokens),
        phases.perTokenUs(.submit, tokens),
        phases.perTokenUs(.fence_wait, tokens),
        phases.perTokenUs(.invalidate, tokens),
        phases.perTokenUs(.readdown, tokens),
        phases.perTokenUs(.tail_gpu, tokens),
        phases.perTokenUs(.head_gpu, tokens),
    }) catch "";
}

/// Monotonic host time; called only when the timing instrument is enabled.
pub fn hostClockNs() u64 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts) != 0) return 0;
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

test "phase accumulator sums phases and computes per-token microseconds" {
    var p: Phases = .{};
    p.add(.record, 1_000);
    p.add(.record, 3_000);
    p.add(.flush, 500);
    try std.testing.expectEqual(@as(u64, 4_000), p.ns[0]);
    try std.testing.expectEqual(@as(f64, 2.0), p.perTokenUs(.record, 2));
    try std.testing.expectEqual(@as(f64, 0.25), p.perTokenUs(.flush, 2));
    try std.testing.expectEqual(@as(f64, 0.0), p.perTokenUs(.record, 0));
}

test "phase line formatter emits the exact KTIME phases fields" {
    var p: Phases = .{};
    p.add(.record, 2_000_000);
    p.add(.flush, 1_000_000);
    p.add(.submit, 1_000_000);
    p.add(.fence_wait, 1_000_000);
    p.add(.invalidate, 1_000_000);
    p.add(.readdown, 1_000_000);
    p.add(.tail_gpu, 1_000_000);
    p.add(.head_gpu, 1_000_000);
    var buf: [256]u8 = undefined;
    try std.testing.expectEqualStrings(
        "KTIME phases run=run_a per_token_us record=2000.00 flush=1000.00 submit=1000.00 fence_wait=1000.00 invalidate=1000.00 readdown=1000.00 tail_gpu=1000.00 head_gpu=1000.00",
        formatPhases(&buf, "run_a", p, 1),
    );
}

/// Ticks are only valid in the family's low `valid_bits`; the delta wraps with the masked period.
pub fn timestampMask(valid_bits: u32) u64 {
    if (valid_bits == 0 or valid_bits >= 64) return ~@as(u64, 0);
    return (@as(u64, 1) << @intCast(valid_bits)) - 1;
}

pub fn stampDelta(prev: u64, cur: u64, mask: u64) u64 {
    const p = prev & mask;
    const q = cur & mask;
    return if (q >= p) q - p else q +% mask +% 1 -% p;
}

/// The enable-time gate: a family that cannot timestamp, or a device whose
/// tick length is unknown, refuses the instrument instead of producing junk.
pub fn checkTimestampSupport(valid_bits: u32, timestamp_period: f32) error{TimestampUnsupported}!void {
    if (valid_bits == 0 or timestamp_period == 0) return error.TimestampUnsupported;
}

/// One VkQueryPool and the host-side record of what each stamp followed.
/// Created only by an explicit enable; the pool lives and dies with it.
pub const Timing = struct {
    dev: *const Device,
    pool: c.VkQueryPool = null,
    stamps: [timing_capacity]u64 = undefined,
    entries: [timing_capacity]Stamp = undefined,
    n: u32 = 0,
    valid_mask: u64 = ~@as(u64, 0),
    io: ?std.Io = null,
    period: f32 = 1,
    tally: Tally = .{},
    phases: Phases = .{},

    pub fn resetWindow(self: *Timing) void {
        self.tally = .{};
        self.phases = .{};
    }

    pub fn addPhase(self: *Timing, phase: Phase, ns: u64) void {
        self.phases.add(phase, ns);
    }
    pub fn nowNs(self: *const Timing) u64 {
        if (self.io) |io| return @intCast(std.Io.Clock.awake.now(io).nanoseconds);
        return hostClockNs();
    }

    pub fn create(gpa: std.mem.Allocator, dev: *const Device) (runtime.Error || error{ OutOfMemory, TimestampUnsupported })!*Timing {
        var fams: [32]c.VkQueueFamilyProperties = undefined;
        var nf: u32 = fams.len;
        dev.rt.api.vkGetPhysicalDeviceQueueFamilyProperties(dev.physical, &nf, &fams);
        const bits: u32 = if (dev.queue_family < @min(nf, fams.len)) fams[dev.queue_family].timestampValidBits else 0;
        const period = dev.properties.limits.timestampPeriod;
        checkTimestampSupport(bits, period) catch |e| {
            std.log.err("vulkan: per-dispatch timing refused: queue family {d} timestampValidBits={d}, timestampPeriod={d}", .{ dev.queue_family, bits, period });
            return e;
        };
        const t = try gpa.create(Timing);
        errdefer gpa.destroy(t);
        t.* = .{ .dev = dev, .valid_mask = timestampMask(bits), .period = period };
        const info: c.VkQueryPoolCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = c.VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = timing_capacity,
        };
        try check(dev.api.vkCreateQueryPool(dev.handle, &info, null, &t.pool));
        return t;
    }

    /// Destroys the pool and the struct; call only after the last submission using it was fenced.
    pub fn destroy(self: *Timing, gpa: std.mem.Allocator) void {
        self.dev.api.vkDestroyQueryPool(self.dev.handle, self.pool, null);
        gpa.destroy(self);
    }

    /// Called from beginBatch's freshly begun command buffer: the stamp list
    /// restarts, the pool range resets, and stamp 0 anchors the batch.
    pub fn batchBegin(self: *Timing, cmd: c.VkCommandBuffer) Error!void {
        self.n = 0;
        self.dev.api.vkCmdResetQueryPool(cmd, self.pool, 0, timing_capacity);
        try self.write(cmd, 0, .{});
    }

    /// Called from record after its barrier, inside an open batch only.
    pub fn stamp(self: *Timing, cmd: c.VkCommandBuffer, spirv: []align(4) const u8, push: []const u8, groups: [3]u32) Error!void {
        var words: [4]u32 = .{ 0, 0, 0, 0 };
        for (words[0..@min(4, push.len / 4)], 0..) |*w, i| w.* = std.mem.readInt(u32, push[i * 4 ..][0..4], .little);
        try self.write(cmd, self.n, .{
            .spirv_ptr = @intFromPtr(spirv.ptr),
            .spirv_len = @intCast(spirv.len),
            .push_hash = std.hash.Wyhash.hash(0, push),
            .push_words = words,
            .groups = groups,
        });
    }

    fn write(self: *Timing, cmd: c.VkCommandBuffer, index: u32, entry: Stamp) Error!void {
        if (index >= timing_capacity) {
            std.log.err("vulkan: timing stamp {d} past capacity {d}; refusing", .{ index, timing_capacity });
            return error.TimingOverflow;
        }
        self.dev.api.vkCmdWriteTimestamp(cmd, c.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, self.pool, index);
        self.stamps[index] = 0;
        self.entries[index] = entry;
        self.n = index + 1;
    }

    /// The batch's final command follows the last dispatch and download copies.
    pub fn tailStamp(self: *Timing, cmd: c.VkCommandBuffer) Error!void {
        try self.write(cmd, self.n, .{});
    }

    /// After the batch's fence: fetch the stamps, convert the deltas to ns
    /// and fold them into the tally. Delta i names the dispatch of entry i.
    pub fn collect(self: *Timing) Error!void {
        if (self.n == 0) return;
        try check(self.dev.api.vkGetQueryPoolResults(
            self.dev.handle,
            self.pool,
            0,
            self.n,
            @as(u64, self.n) * @sizeOf(u64),
            &self.stamps,
            @sizeOf(u64),
            c.VK_QUERY_RESULT_64_BIT | c.VK_QUERY_RESULT_WAIT_BIT,
        ));
        for (self.entries[1..self.n], self.stamps[1..self.n], 0..) |e, cur, i| {
            const ns = self.ticksToNs(stampDelta(self.stamps[i], cur, self.valid_mask));
            if (e.spirv_ptr == 0) continue;
            self.tally.add(.{ .spirv = e.spirv_ptr, .push_hash = e.push_hash, .push_words = e.push_words, .groups = e.groups }, e.spirv_len, ns) catch |err| {
                std.log.err("vulkan: timing tally overflowed {d} distinct kernels at stamp {d}; refusing", .{ tally_cap, i + 1 });
                return err;
            };
        }
        if (self.n >= 2) self.phases.add(.head_gpu, self.ticksToNs(stampDelta(self.stamps[0], self.stamps[1], self.valid_mask)));
        if (self.n >= 3) self.phases.add(.tail_gpu, self.ticksToNs(stampDelta(self.stamps[self.n - 2], self.stamps[self.n - 1], self.valid_mask)));
    }

    fn ticksToNs(self: *const Timing, ticks: u64) u64 {
        return @intFromFloat(@as(f64, @floatFromInt(ticks)) * @as(f64, self.period));
    }
};

/// The bounded per-pipeline bank of immutable descriptor snapshots for
/// batched recording. 512 covers the worst token (489 dispatch records; no
/// single pipeline exceeds that within one token). A snapshot set, once
/// recorded, is never rewritten until the batch epoch advances, which
/// happens only after the previous batch's fence has been waited — so the
/// GPU never observes a set whose contents change under it.
pub const batch_cap: u32 = 512;

/// Batch session state shared by every pipeline; driven by commands.zig's
/// beginBatch (batchBegin) and after the fence (batchEnd).
pub var batch_open: bool = false;
/// Advances at every batch open; a pipeline whose seen epoch differs resets
/// its bank cursor, reusing snapshots only after the previous fence.
pub var batch_epoch: u32 = 0;

pub fn batchBegin() void {
    batch_open = true;
    batch_epoch +%= 1;
}

pub fn batchEnd() void {
    batch_open = false;
}

/// The next bank cursor for a pipeline whose snapshots were last reset at
/// epoch `seen`: a new epoch resets to 0 (legal only because beginBatch runs
/// after the previous batch's fence), and the bounded bank refuses past
/// batch_cap instead of silently reusing a set the GPU is still reading.
/// Pure and host-tested.
pub fn nextBatchCursor(seen: u32, cursor: u32) Error!u32 {
    const next = if (seen != batch_epoch) 0 else cursor;
    if (next >= batch_cap) return error.BatchOverflow;
    return next;
}

/// The shader's set 0: storage buffers 0..bindings-1, its push bytes, and a mask of the bindings it writes.
pub const Layout = struct { bindings: u32, push_bytes: u32, writes: u32 };

/// A device-local byte range for one binding.
pub const Binding = struct { buffer: *const Buffer, offset: u64 = 0, len: u64 };

pub const Pipeline = struct {
    dev: *const Device,
    layout: Layout,
    /// The entry point's workgroup size, read from the module itself (localSize).
    local: [3]u32,
    set_layout: c.VkDescriptorSetLayout = null,
    pipeline_layout: c.VkPipelineLayout = null,
    pipeline: c.VkPipeline = null,
    pool: c.VkDescriptorPool = null,
    set: c.VkDescriptorSet = null,
    bound: bool = false,
    // Batched-recording bank: lazily created pool, allocated sets, the write
    // cursor and the epoch it was last reset at. Bindings staged by bind()
    // while a batch is open are snapshotted into the bank at record time.
    batch_pool: c.VkDescriptorPool = null,
    batch_sets: [batch_cap]c.VkDescriptorSet = undefined,
    batch_n: u32 = 0,
    batch_cursor: u32 = 0,
    batch_epoch_seen: u32 = 0,
    stash: [max_bindings]Binding = undefined,
    stash_n: u32 = 0,
    /// The module this pipeline was built from, so a timing stamp can name its kernel.
    spirv: []align(4) const u8,

    /// Checks the module and the device's limits before creating anything; a failure destroys what was made.
    pub fn init(dev: *const Device, spirv: []align(4) const u8, layout: Layout) Error!Pipeline {
        const local = localSize(spirv) catch |e| {
            const hint = if (e == error.MissingFeature) " (LocalSizeId needs maintenance4: compile for SPIR-V 1.5)" else "";
            std.log.err("vulkan: SPIR-V module of {d} bytes refused: {s}{s}", .{ spirv.len, @errorName(e), hint });
            return e;
        };
        const l = dev.properties.limits;
        fits(l, layout, local) catch |e| {
            const say = "vulkan: {s}: {d} storage buffers, {d} push bytes, workgroup {any}; device: {d}/{d}/{d} " ++
                "buffers, {d} push bytes, workgroup {any} of {d}";
            const asked = .{ @errorName(e), layout.bindings, layout.push_bytes, local };
            const has = .{ l.maxPerStageDescriptorStorageBuffers, l.maxDescriptorSetStorageBuffers, l.maxPerStageResources };
            const room = .{ l.maxPushConstantsSize, l.maxComputeWorkGroupSize, l.maxComputeWorkGroupInvocations };
            std.log.err(say, asked ++ has ++ room);
            return e;
        };
        const api = &dev.api;
        const h = dev.handle;
        const module_info: c.VkShaderModuleCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spirv.len,
            .pCode = @ptrCast(spirv.ptr),
        };
        var module: c.VkShaderModule = null;
        try check(api.vkCreateShaderModule(h, &module_info, null, &module));
        defer api.vkDestroyShaderModule(h, module, null); // a created pipeline no longer needs its module
        var slots: [max_bindings]c.VkDescriptorSetLayoutBinding = undefined;
        for (slots[0..layout.bindings], 0..) |*s, i| s.* = .{
            .binding = @intCast(i),
            .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT,
        };
        var p: Pipeline = .{ .dev = dev, .layout = layout, .local = local, .spirv = spirv };
        const set_info: c.VkDescriptorSetLayoutCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = layout.bindings,
            .pBindings = &slots,
        };
        try check(api.vkCreateDescriptorSetLayout(h, &set_info, null, &p.set_layout));
        errdefer api.vkDestroyDescriptorSetLayout(h, p.set_layout, null);
        const range: c.VkPushConstantRange = .{ .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT, .size = layout.push_bytes };
        const layout_info: c.VkPipelineLayoutCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &p.set_layout,
            .pushConstantRangeCount = @intFromBool(layout.push_bytes > 0),
            .pPushConstantRanges = &range,
        };
        try check(api.vkCreatePipelineLayout(h, &layout_info, null, &p.pipeline_layout));
        errdefer api.vkDestroyPipelineLayout(h, p.pipeline_layout, null);
        const pipe_info: c.VkComputePipelineCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = .{
                .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = c.VK_SHADER_STAGE_COMPUTE_BIT,
                .module = module,
                .pName = "main",
            },
            .layout = p.pipeline_layout,
            .basePipelineIndex = -1,
        };
        check(api.vkCreateComputePipelines(h, null, 1, &pipe_info, null, &p.pipeline)) catch |e| {
            if (p.pipeline != null) api.vkDestroyPipeline(h, p.pipeline, null); // a driver may leave a partial handle
            return e;
        };
        errdefer api.vkDestroyPipeline(h, p.pipeline, null);
        const size: c.VkDescriptorPoolSize = .{ .type = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = layout.bindings };
        const pool_info: c.VkDescriptorPoolCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &size,
        };
        try check(api.vkCreateDescriptorPool(h, &pool_info, null, &p.pool));
        errdefer api.vkDestroyDescriptorPool(h, p.pool, null);
        const alloc_info: c.VkDescriptorSetAllocateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = p.pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &p.set_layout,
        };
        try check(api.vkAllocateDescriptorSets(h, &alloc_info, &p.set));
        return p;
    }

    /// Call only after the last submission that used this pipeline returned success.
    pub fn deinit(p: *Pipeline) void {
        const api = &p.dev.api;
        const h = p.dev.handle;
        if (p.batch_pool != null) api.vkDestroyDescriptorPool(h, p.batch_pool, null);
        api.vkDestroyDescriptorPool(h, p.pool, null);
        api.vkDestroyPipeline(h, p.pipeline, null);
        api.vkDestroyPipelineLayout(h, p.pipeline_layout, null);
        api.vkDestroyDescriptorSetLayout(h, p.set_layout, null);
    }

    /// Updates checked device-local bindings; call only when no submission using this descriptor set is pending.
    /// While a batch is open the bindings are staged instead and snapshotted
    /// into that dispatch's own banked set at record time — a batched token
    /// never rewrites a set the GPU has been commanded to read.
    pub fn bind(p: *Pipeline, bindings: []const Binding) Error!void {
        if (bindings.len != p.layout.bindings) return error.BadBindings;
        var infos: [max_bindings]c.VkDescriptorBufferInfo = undefined;
        for (bindings, 0..) |b, i| {
            if (b.buffer.kind != .device or b.buffer.dev != p.dev) return error.BadBindings;
            if (b.len % 4 != 0) return error.Misaligned; // 32-bit element arrays: a ragged range drops its last element
            infos[i] = try b.buffer.descriptor(b.offset, b.len);
        }
        try disjoint(bindings, p.layout.writes);
        if (batch_open) {
            @memcpy(p.stash[0..bindings.len], bindings);
            p.stash_n = @intCast(bindings.len);
            return;
        }
        p.stash_n = 0; // discard any unrecorded binding left by an aborted batch
        var writes: [max_bindings]c.VkWriteDescriptorSet = undefined;
        for (bindings, 0..) |_, i| {
            writes[i] = .{
                .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = p.set,
                .dstBinding = @intCast(i),
                .descriptorCount = 1,
                .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &infos[i],
            };
        }
        p.dev.api.vkUpdateDescriptorSets(p.dev.handle, p.layout.bindings, &writes, 0, null);
        p.bound = true;
    }

    /// The descriptor set this dispatch binds: the pipeline's live set on the
    /// immediate schedule, or a fresh immutable bank snapshot per batched
    /// dispatch (the staged bindings' vkUpdateDescriptorSets, counted by the
    /// profiler, happens here).
    fn snapshotSet(p: *Pipeline) Error!c.VkDescriptorSet {
        if (p.stash_n == 0) return p.set; // immediate schedule: bind already updated p.set
        const cursor = try nextBatchCursor(p.batch_epoch_seen, p.batch_cursor);
        if (p.batch_epoch_seen != batch_epoch) {
            p.batch_epoch_seen = batch_epoch;
            p.batch_cursor = cursor; // previous batch's fence has been waited; its snapshots are reusable
        }
        const api = &p.dev.api;
        const h = p.dev.handle;
        if (p.batch_pool == null) {
            // First batched use of this pipeline: the persistent bounded
            // bank. First-use cost only; later tokens reuse the sets.
            const size: c.VkDescriptorPoolSize = .{ .type = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = batch_cap * p.layout.bindings };
            const pool_info: c.VkDescriptorPoolCreateInfo = .{
                .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets = batch_cap,
                .poolSizeCount = 1,
                .pPoolSizes = &size,
            };
            try check(api.vkCreateDescriptorPool(h, &pool_info, null, &p.batch_pool));
        }
        if (p.batch_cursor == p.batch_n) {
            const alloc_info: c.VkDescriptorSetAllocateInfo = .{
                .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = p.batch_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &p.set_layout,
            };
            try check(api.vkAllocateDescriptorSets(h, &alloc_info, &p.batch_sets[p.batch_cursor]));
            p.batch_n += 1;
        }
        const set = p.batch_sets[p.batch_cursor];
        var infos: [max_bindings]c.VkDescriptorBufferInfo = undefined;
        var writes: [max_bindings]c.VkWriteDescriptorSet = undefined;
        for (p.stash[0..p.stash_n], 0..) |b, i| {
            infos[i] = try b.buffer.descriptor(b.offset, b.len);
            writes[i] = .{
                .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = set,
                .dstBinding = @intCast(i),
                .descriptorCount = 1,
                .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &infos[i],
            };
        }
        api.vkUpdateDescriptorSets(h, p.stash_n, &writes, 0, null);
        p.batch_cursor += 1;
        p.stash_n = 0;
        return set;
    }

    /// Records one dispatch, then a barrier making its writes visible to a later dispatch or transfer.
    pub fn record(p: *Pipeline, cmd: c.VkCommandBuffer, push: []const u8, groups: [3]u32) Error!void {
        if (p.stash_n == 0 and !p.bound) return error.BadBindings;
        if (push.len != p.layout.push_bytes) return error.BadPushSize;
        const most = p.dev.properties.limits.maxComputeWorkGroupCount;
        counts(most, groups) catch |e| {
            std.log.err("vulkan: dispatch {any} groups refused, device allows {any}", .{ groups, most });
            return e;
        };
        const set = try p.snapshotSet();
        const api = &p.dev.api;
        api.vkCmdBindPipeline(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
        api.vkCmdBindDescriptorSets(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline_layout, 0, 1, &set, 0, null);
        if (push.len > 0) api.vkCmdPushConstants(cmd, p.pipeline_layout, c.VK_SHADER_STAGE_COMPUTE_BIT, 0, p.layout.push_bytes, push.ptr);
        api.vkCmdDispatch(cmd, groups[0], groups[1], groups[2]);
        const reads = c.VK_ACCESS_SHADER_READ_BIT | c.VK_ACCESS_SHADER_WRITE_BIT | c.VK_ACCESS_TRANSFER_READ_BIT | c.VK_ACCESS_TRANSFER_WRITE_BIT;
        const stages = c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | c.VK_PIPELINE_STAGE_TRANSFER_BIT;
        commands.barrier(p.dev, cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_ACCESS_SHADER_WRITE_BIT, stages, reads);
        // Only inside an open batch: beginBatch anchors stamp 0 and endBatch
        // reads exactly this batch's range, so a stamp outside one would be
        // recorded but never collected.
        if (timing) |t| if (batch_open) try t.stamp(cmd, p.spirv, push, groups);
    }

    /// One dispatch in its own submission; wrap it in commands.orExit, which ends the process on error.Timeout.
    /// Inside an open batch the begin/submit pair is the session's no-op and
    /// the dispatch only records into the packed command buffer.
    pub fn run(p: *Pipeline, cmds: *commands.Commands, push: []const u8, groups: [3]u32, timeout_ns: u64) Error!void {
        try p.record(try cmds.begin(), push, groups);
        try cmds.submit(timeout_ns);
    }
};

const op = struct {
    const entry_point = 15;
    const execution_mode = 16;
    const constant = 43;
    const constant_composite = 44;
    const spec_constant = 50;
    const spec_constant_composite = 51;
    const decorate = 71;
    const execution_mode_id = 331;
};

/// Validates SPIR-V and reads main's workgroup size; refuses LocalSizeId because maintenance4 is not enabled.
pub fn localSize(code: []align(4) const u8) Error![3]u32 {
    if (code.len < 20 or code.len % 4 != 0) return error.BadSpirv;
    const w = std.mem.bytesAsSlice(u32, code);
    const minor = w[1] >> 8 & 0xff;
    if (w[0] != 0x07230203 or w[1] >> 16 != 1 or minor > 6 or w[3] == 0 or w[4] != 0) return error.BadSpirv;
    var entry: ?u32 = null;
    var literal: ?[3]u32 = null;
    var by_id = false;
    var builtin: ?u32 = null;
    var composite: ?[3]u32 = null;
    var i: usize = 5;
    while (i < w.len) : (i += w[i] >> 16) {
        const in = try instruction(w, i);
        const code_op = in[0] & 0xffff;
        const ours = in.len == 6 and entry != null and in[1] == entry.?;
        switch (code_op) {
            op.entry_point => if (in.len >= 5 and in[1] == 5 and std.mem.startsWith(u8, std.mem.sliceAsBytes(in[3..]), "main\x00")) {
                entry = in[2];
            },
            op.execution_mode => if (ours and in[2] == 17) {
                literal = in[3..6].*;
            },
            op.execution_mode_id => if (ours and in[2] == 38) {
                by_id = true;
            },
            op.decorate => if (in.len == 4 and in[2] == 11 and in[3] == 25) {
                builtin = in[1];
            },
            op.constant_composite, op.spec_constant_composite => if (in.len == 6 and builtin != null and in[2] == builtin.?) {
                composite = in[3..6].*;
            },
            else => {},
        }
    }
    if (entry == null) return error.NoLocalSize;
    if (by_id) return error.MissingFeature;
    if (builtin == null) return literal orelse error.NoLocalSize;
    const ids = composite orelse return error.NoLocalSize;
    var size: [3]?u32 = .{ null, null, null };
    i = 5;
    while (i < w.len) : (i += w[i] >> 16) {
        const in = try instruction(w, i);
        const scalar = in.len == 4 and (in[0] & 0xffff == op.constant or in[0] & 0xffff == op.spec_constant);
        if (scalar) for (ids, &size) |id, *s| if (in[2] == id) {
            s.* = in[3];
        };
    }
    var out: [3]u32 = undefined;
    for (size, &out) |s, *o| o.* = s orelse return error.NoLocalSize;
    return out;
}

fn instruction(w: []const u32, i: usize) Error![]const u32 {
    const n = w[i] >> 16;
    if (n == 0 or n > w.len - i) return error.BadSpirv;
    return w[i..][0..n];
}

fn fits(l: c.VkPhysicalDeviceLimits, layout: Layout, local: [3]u32) Error!void {
    const buffers = @min(l.maxPerStageDescriptorStorageBuffers, l.maxDescriptorSetStorageBuffers, l.maxPerStageResources);
    if (layout.bindings == 0 or layout.bindings > max_bindings or layout.bindings > buffers) return error.TooManyBindings;
    if (layout.writes >> @intCast(layout.bindings) != 0) return error.BadBindings;
    if (layout.push_bytes % 4 != 0 or layout.push_bytes > l.maxPushConstantsSize) return error.BadPushSize;
    var threads: u64 = 1;
    for (local, l.maxComputeWorkGroupSize) |n, most| {
        if (n == 0 or n > most) return error.BadWorkgroup;
        threads *= n; // threads was at most the u32 invocation limit, so this product fits u64
        if (threads > l.maxComputeWorkGroupInvocations) return error.BadWorkgroup;
    }
}

fn counts(most: [3]u32, groups: [3]u32) Error!void {
    for (groups, most) |g, m| if (g == 0 or g > m) return error.BadGroupCount;
}

/// A written range may not overlap any other binding's range in the same buffer.
fn disjoint(bindings: []const Binding, writes: u32) Error!void {
    for (bindings, 0..) |o, i| if (writes >> @intCast(i) & 1 != 0) for (bindings, 0..) |b, j| {
        if (i != j and o.buffer.handle == b.buffer.handle and o.offset < b.offset + b.len and b.offset < o.offset + o.len) return error.Aliased;
    };
}

const header = [_]u32{ 0x07230203, 0x00010600, 0, 30, 0 };
const main_entry = [_]u32{ 5 << 16 | 15, 5, 7, 0x6e69616d, 0 };
const local_256 = [_]u32{ 6 << 16 | 16, 7, 17, 256, 1, 1 };
const entry_words = header ++ main_entry ++ local_256;
const constants = [_]u32{ 4 << 16 | 43, 2, 10, 128, 4 << 16 | 43, 2, 11, 1 };

test "localSize reads the compute entry's size in each SPIR-V form and refuses malformed modules" {
    const bytes = std.mem.sliceAsBytes;
    try std.testing.expectEqual([3]u32{ 256, 1, 1 }, try localSize(bytes(&entry_words)));
    const by_id = header ++ main_entry ++ [_]u32{ 6 << 16 | 331, 7, 38, 10, 11, 11 } ++ constants;
    try std.testing.expectError(error.MissingFeature, localSize(bytes(&by_id)));
    const decorated = entry_words ++ [_]u32{ 4 << 16 | 71, 20, 11, 25 } ++ constants ++ [_]u32{ 6 << 16 | 44, 3, 20, 10, 11, 11 };
    try std.testing.expectEqual([3]u32{ 128, 1, 1 }, try localSize(bytes(&decorated)));
    try std.testing.expectError(error.NoLocalSize, localSize(bytes(decorated[0 .. decorated.len - 6])));
    try std.testing.expectError(error.BadSpirv, localSize(bytes(entry_words[0..15])));
    try std.testing.expectError(error.BadSpirv, localSize(bytes(&(entry_words ++ [_]u32{0}))));
    var bad = entry_words;
    bad[0] = 0x03022307;
    try std.testing.expectError(error.BadSpirv, localSize(bytes(&bad)));
    bad = entry_words;
    bad[11] = 8; // LocalSize of another function
    try std.testing.expectError(error.NoLocalSize, localSize(bytes(&bad)));
    bad = entry_words;
    bad[1] = 0x00010700;
    try std.testing.expectError(error.BadSpirv, localSize(bytes(&bad)));
}

test "device limits refuse bindings, push bytes, workgroups and group counts past them" {
    var l = std.mem.zeroes(c.VkPhysicalDeviceLimits);
    l.maxPerStageDescriptorStorageBuffers = 4;
    l.maxDescriptorSetStorageBuffers = 8;
    l.maxPerStageResources = 8;
    l.maxPushConstantsSize = 128;
    l.maxComputeWorkGroupSize = .{ 256, 1, 1 };
    l.maxComputeWorkGroupInvocations = 256;
    const qmm: Layout = .{ .bindings = 5, .push_bytes = 120, .writes = 1 << 4 };
    try std.testing.expectError(error.TooManyBindings, fits(l, qmm, .{ 256, 1, 1 }));
    l.maxPerStageDescriptorStorageBuffers = 8;
    try fits(l, qmm, .{ 256, 1, 1 });
    try std.testing.expectError(error.BadPushSize, fits(l, .{ .bindings = 5, .push_bytes = 132, .writes = 1 << 4 }, .{ 256, 1, 1 }));
    try std.testing.expectError(error.BadBindings, fits(l, .{ .bindings = 5, .push_bytes = 120, .writes = 1 << 5 }, .{ 256, 1, 1 }));
    try std.testing.expectError(error.BadWorkgroup, fits(l, qmm, .{ 512, 1, 1 }));
    l.maxComputeWorkGroupInvocations = 128;
    try std.testing.expectError(error.BadWorkgroup, fits(l, qmm, .{ 256, 1, 1 }));
    // restore the limit the negative case reduced, before the positive cases
    l.maxComputeWorkGroupInvocations = 256;
    try std.testing.expectError(error.BadGroupCount, counts(.{ 65535, 65535, 65535 }, .{ 0, 1, 1 }));
    try std.testing.expectError(error.BadGroupCount, counts(.{ 65535, 65535, 65535 }, .{ 65536, 1, 1 }));
    try counts(.{ 65535, 65535, 65535 }, .{ 2, 1, 1 });
    // The 10-binding GDN prefill/decode closures fit; 11 do not, and a device
    // offering fewer than asked still refuses (real per-stage/set capacity).
    l.maxPerStageDescriptorStorageBuffers = 10;
    l.maxDescriptorSetStorageBuffers = 10;
    l.maxPerStageResources = 10;
    try fits(l, .{ .bindings = max_bindings, .push_bytes = 120, .writes = 1 << 9 }, .{ 256, 1, 1 });
    try std.testing.expectError(error.TooManyBindings, fits(l, .{ .bindings = max_bindings + 1, .push_bytes = 120, .writes = 1 << 10 }, .{ 256, 1, 1 }));
    l.maxPerStageDescriptorStorageBuffers = 8;
    try std.testing.expectError(error.TooManyBindings, fits(l, .{ .bindings = max_bindings, .push_bytes = 120, .writes = 1 << 9 }, .{ 256, 1, 1 }));
}

test "timing deltas mask the valid bits and wrap with the masked period" {
    const full = ~@as(u64, 0);
    try std.testing.expectEqual(@as(u64, 5), stampDelta(10, 15, full));
    try std.testing.expectEqual(@as(u64, 0), stampDelta(7, 7, full));
    // a 24-bit family: the stamp wraps past the mask and the delta still counts the wrapped ticks
    const m24 = timestampMask(24);
    try std.testing.expectEqual(@as(u64, (1 << 24) - 1), m24);
    try std.testing.expectEqual(@as(u64, 3), stampDelta(0xfffffe, 0x1000001, m24));
    try std.testing.expectEqual(@as(u64, 4), stampDelta(0xfffffd, (1 << 24) + 1, m24));
    // 64 valid bits: the natural u64 wrap IS the period
    try std.testing.expectEqual(@as(u64, 1), stampDelta(std.math.maxInt(u64), 0, full));
}

test "the tally aggregates by key and shares partition the total" {
    var t: Tally = .{};
    const k1: Key = .{ .spirv = 0x1000, .push_hash = 1, .push_words = .{ 1, 0, 0, 0 }, .groups = .{ 4, 1, 1 } };
    const k2: Key = .{ .spirv = 0x2000, .push_hash = 2, .push_words = .{ 2, 0, 0, 0 }, .groups = .{ 8, 1, 1 } };
    try t.add(k1, 100, 300);
    try t.add(k2, 200, 100);
    try t.add(k1, 100, 100); // same key aggregates
    try std.testing.expectEqual(@as(u32, 2), t.n);
    try std.testing.expectEqual(@as(u64, 500), t.total_ns);
    try std.testing.expectEqual(@as(u64, 2), t.entries[0].calls);
    try std.testing.expectEqual(@as(u64, 400), t.entries[0].total_ns);
    try std.testing.expectEqual(@as(u64, 100), t.entries[1].total_ns);
    try std.testing.expectEqual(@as(f64, 0.8), t.share(t.entries[0].total_ns));
    try std.testing.expectEqual(@as(f64, 1.0), t.share(t.total_ns));
    // a distinct key past the bounded cap refuses loudly instead of dropping a kernel
    var full: Tally = .{};
    for (0..tally_cap) |i| try full.add(.{ .spirv = i, .push_hash = 0, .push_words = .{ 0, 0, 0, 0 }, .groups = .{ 1, 1, 1 } }, 0, 1);
    try std.testing.expectError(error.TimingOverflow, full.add(.{ .spirv = tally_cap, .push_hash = 0, .push_words = .{ 0, 0, 0, 0 }, .groups = .{ 1, 1, 1 } }, 0, 1));
}

test "timing refuses a queue family without usable timestamps" {
    try std.testing.expectError(error.TimestampUnsupported, checkTimestampSupport(0, 1.0));
    try std.testing.expectError(error.TimestampUnsupported, checkTimestampSupport(64, 0.0));
    try std.testing.expectEqual(@as(u64, (1 << 36) - 1), timestampMask(36));
    try std.testing.expectEqual(@as(u64, ~@as(u64, 0)), timestampMask(64));
    try checkTimestampSupport(64, 1.0);
    try checkTimestampSupport(36, 0.5);
}


test "the batched bank cursor resets per epoch and refuses past the bounded cap" {
    batch_epoch += 1; // beginBatch bumps: the previous batch's fence has been waited
    try std.testing.expectEqual(@as(u32, 0), try nextBatchCursor(0, 489)); // stale seen epoch resets
    try std.testing.expectEqual(@as(u32, 7), try nextBatchCursor(batch_epoch, 7)); // same epoch advances
    try std.testing.expectEqual(@as(u32, batch_cap - 1), try nextBatchCursor(batch_epoch, batch_cap - 1)); // last slot
    try std.testing.expectEqual(@as(u32, 0), try nextBatchCursor(batch_epoch -% 1, batch_cap)); // fresh epoch clears even a full cursor
    // exhausting the bank within one epoch refuses instead of rewriting a set in flight
    const full: u32 = batch_cap;
    batch_epoch += 1;
    try std.testing.expectError(error.BatchOverflow, nextBatchCursor(batch_epoch, full));
}
