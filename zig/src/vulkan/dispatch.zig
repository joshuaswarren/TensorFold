//! One compute pipeline over set-0 storage buffers and push constants: checked SPIR-V, limits, bindings and dispatch.
const std = @import("std");
const c = @import("vk");
const runtime = @import("runtime.zig");
const Device = @import("device.zig").Device;
const Buffer = @import("memory.zig").Buffer;
const commands = @import("commands.zig");
const hazard = @import("hazard.zig");

const check = runtime.check;

pub const Error = runtime.Error || error{ BadSpirv, NoLocalSize, TooManyBindings, BadPushSize, BadWorkgroup, BadGroupCount, BadBindings, Aliased };

pub const max_bindings = 10;

/// Dispatches recorded into one submission, so a descriptor set per dispatch; run submits at half of it.
pub const batch_max: u32 = 256;
const batch_flush: u32 = batch_max / 2;

/// Per-dispatch GPU timestamps, off until a caller points `timing` at its
/// Timing. Null (the default) costs nothing: no query pool, no extra
/// commands, no allocations.
pub var timing: ?*Timing = null;

/// Query-pool capacity. Measured bounds: one collect window is one Commands
/// submit, and run submits at `batch_flush`, so a window records at most
/// batch_flush (128) dispatch stamps plus the anchor; a whole 64-row prefill
/// chunk spans ~500 dispatches (24 layers times their kernel sites plus head
/// and tail), split across ~4 windows. 16384 covers a chunk several times
/// over either way.
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

/// Where the deltas of one instrumented phase accumulate, keyed by invocation
/// shape. Linear scan over a bounded table: a phase's distinct shapes are
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

    /// The fraction of the phase's summed deltas this entry's total makes up.
    pub fn share(self: *const Tally, entry_total_ns: u64) f64 {
        if (self.total_ns == 0) return 0;
        return @as(f64, @floatFromInt(entry_total_ns)) / @as(f64, @floatFromInt(self.total_ns));
    }
};

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
    period: f32 = 1,
    tally: Tally = .{},

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

    /// Called from Commands.begin's freshly begun command buffer: the stamp
    /// list restarts, the pool range resets, and stamp 0 anchors the window.
    pub fn batchBegin(self: *Timing, cmd: c.VkCommandBuffer) Error!void {
        self.n = 0;
        self.dev.api.vkCmdResetQueryPool(cmd, self.pool, 0, timing_capacity);
        try self.write(cmd, 0, .{});
    }

    /// Called from dispatchOnly right after the dispatch it follows; only
    /// meaningful inside an open command buffer (begin anchors stamp 0,
    /// submit collects exactly this range, so a stamp outside one would be
    /// recorded but never collected).
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

    /// After the submit's fence: fetch the stamps, convert the deltas to ns
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
            const ns: u64 = @intFromFloat(@as(f64, @floatFromInt(stampDelta(self.stamps[i], cur, self.valid_mask))) * @as(f64, self.period));
            self.tally.add(.{ .spirv = e.spirv_ptr, .push_hash = e.push_hash, .push_words = e.push_words, .groups = e.groups }, e.spirv_len, ns) catch |err| {
                std.log.err("vulkan: timing tally overflowed {d} distinct kernels at stamp {d}; refusing", .{ tally_cap, i + 1 });
                return err;
            };
        }
    }
};

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
    /// One set per dispatch recorded since the last submission; bind fills the next one, record binds the current one.
    sets: [batch_max]c.VkDescriptorSet = @splat(null),
    slot: usize = 0,
    bound: bool = false,
    /// The module this pipeline was built from, so a timing stamp can name its kernel.
    spirv: []align(4) const u8,
    /// The byte range of each binding of the current bind, for the hazard check of the next run.
    ranges: [max_bindings]hazard.Range = undefined,

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
        const size: c.VkDescriptorPoolSize = .{ .type = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = layout.bindings * batch_max };
        const pool_info: c.VkDescriptorPoolCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = batch_max,
            .poolSizeCount = 1,
            .pPoolSizes = &size,
        };
        try check(api.vkCreateDescriptorPool(h, &pool_info, null, &p.pool));
        errdefer api.vkDestroyDescriptorPool(h, p.pool, null);
        const set_layouts: [batch_max]c.VkDescriptorSetLayout = @splat(p.set_layout);
        const alloc_info: c.VkDescriptorSetAllocateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = p.pool,
            .descriptorSetCount = batch_max,
            .pSetLayouts = &set_layouts,
        };
        try check(api.vkAllocateDescriptorSets(h, &alloc_info, &p.sets));
        return p;
    }

    /// Call only after the last submission that used this pipeline returned success.
    pub fn deinit(p: *Pipeline) void {
        const api = &p.dev.api;
        const h = p.dev.handle;
        api.vkDestroyDescriptorPool(h, p.pool, null);
        api.vkDestroyPipeline(h, p.pipeline, null);
        api.vkDestroyPipelineLayout(h, p.pipeline_layout, null);
        api.vkDestroyDescriptorSetLayout(h, p.set_layout, null);
    }

    /// Fills the next descriptor set with checked device-local bindings; earlier recorded dispatches keep their sets.
    pub fn bind(p: *Pipeline, bindings: []const Binding) Error!void {
        if (bindings.len != p.layout.bindings) return error.BadBindings;
        var infos: [max_bindings]c.VkDescriptorBufferInfo = undefined;
        var writes: [max_bindings]c.VkWriteDescriptorSet = undefined;
        for (bindings, 0..) |b, i| {
            if (b.buffer.kind != .device or b.buffer.dev != p.dev) return error.BadBindings;
            if (b.len % 4 != 0) return error.Misaligned; // 32-bit element arrays: a ragged range drops its last element
            infos[i] = try b.buffer.descriptor(b.offset, b.len);
            writes[i] = .{
                .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = p.sets[(p.slot + 1) % batch_max],
                .dstBinding = @intCast(i),
                .descriptorCount = 1,
                .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &infos[i],
            };
        }
        try disjoint(bindings, p.layout.writes);
        for (bindings, 0..) |b, i| p.ranges[i] = .{ .buffer = @intFromPtr(b.buffer.handle), .start = b.offset, .end = b.offset + b.len };
        p.dev.api.vkUpdateDescriptorSets(p.dev.handle, p.layout.bindings, &writes, 0, null);
        p.slot = (p.slot + 1) % batch_max;
        p.bound = true;
    }

    /// Records one dispatch and a barrier after it, for callers that record into a command buffer themselves.
    pub fn record(p: *const Pipeline, cmd: c.VkCommandBuffer, push: []const u8, groups: [3]u32) Error!void {
        try p.dispatchOnly(cmd, push, groups);
        commands.dispatchBarrier(p.dev, cmd);
    }

    fn dispatchOnly(p: *const Pipeline, cmd: c.VkCommandBuffer, push: []const u8, groups: [3]u32) Error!void {
        if (!p.bound) return error.BadBindings;
        if (push.len != p.layout.push_bytes) return error.BadPushSize;
        const most = p.dev.properties.limits.maxComputeWorkGroupCount;
        counts(most, groups) catch |e| {
            std.log.err("vulkan: dispatch {any} groups refused, device allows {any}", .{ groups, most });
            return e;
        };
        const api = &p.dev.api;
        api.vkCmdBindPipeline(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
        api.vkCmdBindDescriptorSets(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline_layout, 0, 1, &p.sets[p.slot], 0, null);
        if (push.len > 0) api.vkCmdPushConstants(cmd, p.pipeline_layout, c.VK_SHADER_STAGE_COMPUTE_BIT, 0, p.layout.push_bytes, push.ptr);
        api.vkCmdDispatch(cmd, groups[0], groups[1], groups[2]);
        if (timing) |t| try t.stamp(cmd, p.spirv, push, groups);
    }

    /// Records one dispatch, after a barrier only on a conflict; submits at half a batch or at a copy or fill.
    pub fn run(p: *const Pipeline, cmds: *commands.Commands, push: []const u8, groups: [3]u32, timeout_ns: u64) Error!void {
        var reads: [max_bindings]hazard.Range = undefined;
        var writes: [max_bindings]hazard.Range = undefined;
        var n_reads: usize = 0;
        var n_writes: usize = 0;
        for (p.ranges[0..p.layout.bindings], 0..) |r, i| {
            if (p.layout.writes >> @intCast(i) & 1 != 0) {
                writes[n_writes] = r;
                n_writes += 1;
            } else {
                reads[n_reads] = r;
                n_reads += 1;
            }
        }
        const cmd = try cmds.begin();
        if (cmds.hazards.conflicts(reads[0..n_reads], writes[0..n_writes])) {
            commands.dispatchBarrier(p.dev, cmd);
            cmds.hazards.clear();
        }
        try p.dispatchOnly(cmd, push, groups);
        cmds.hazards.add(reads[0..n_reads], writes[0..n_writes]);
        cmds.pending += 1;
        if (cmds.pending >= batch_flush) try cmds.submit(timeout_ns);
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
    // The 10-binding GDN prefill/decode closures fit; 11 do not; fewer offered still refuses (real capacity).
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
