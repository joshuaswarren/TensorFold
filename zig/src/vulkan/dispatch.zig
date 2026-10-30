//! One compute pipeline over set-0 storage buffers and push constants: checked SPIR-V, limits, bindings and dispatch.
const std = @import("std");
const c = @import("vk");
const runtime = @import("runtime.zig");
const Device = @import("device.zig").Device;
const Buffer = @import("memory.zig").Buffer;
const commands = @import("commands.zig");

const check = runtime.check;

pub const Error = runtime.Error || error{ BadSpirv, NoLocalSize, TooManyBindings, BadPushSize, BadWorkgroup, BadGroupCount, BadBindings, Aliased, BatchOverflow };

/// Barrier elision: a dispatch's post-barrier is deferred until something hazards; TF_VK_NOBAR=0 restores it.
pub var elide_barriers: bool = true;

pub fn elideFromEnv(v: ?[*:0]const u8) bool {
    const s = v orelse return true;
    return !std.mem.eql(u8, std.mem.span(s), "0");
}

/// One buffer range a dispatch reads or writes: same buffer and intersecting range is a hazard.
pub const HazardRange = struct { buf: c.VkBuffer, offset: u64, len: u64 };

pub const hazard_range_cap = 32;

fn rangesOverlap(a: HazardRange, b: HazardRange) bool {
    return a.buf == b.buf and a.offset < b.offset + b.len and b.offset < a.offset + a.len;
}

/// Would this dispatch against unsynchronised work need the barrier? RAW, WAW or WAR do. Pure and host-tested.
pub fn hazardNeeded(pending_writes: []const HazardRange, pending_reads: []const HazardRange, next_writes: []const HazardRange, next_reads: []const HazardRange) bool {
    for (pending_writes) |w| {
        for (next_reads) |r| if (rangesOverlap(w, r)) return true;
        for (next_writes) |w2| if (rangesOverlap(w, w2)) return true;
    }
    for (pending_reads) |r| for (next_writes) |w| {
        if (rangesOverlap(r, w)) return true;
    };
    return false;
}

/// Unsynchronised writes and reads since the last barrier; past the cap it saturates and every check orders.
pub const Hazard = struct {
    writes: [hazard_range_cap]HazardRange = undefined,
    reads: [hazard_range_cap]HazardRange = undefined,
    n_writes: u32 = 0,
    n_reads: u32 = 0,
    saturated: bool = false,

    pub fn add(self: *Hazard, writes: []const HazardRange, reads: []const HazardRange) void {
        for (writes) |w| {
            var dup = false;
            for (self.writes[0..self.n_writes]) |*k| if (k.buf == w.buf and k.offset == w.offset and k.len == w.len) {
                dup = true;
                break;
            };
            if (dup) continue;
            if (self.n_writes < hazard_range_cap) {
                self.writes[self.n_writes] = w;
                self.n_writes += 1;
            } else self.saturated = true;
        }
        for (reads) |r| {
            var dup = false;
            for (self.reads[0..self.n_reads]) |*k| if (k.buf == r.buf and k.offset == r.offset and k.len == r.len) {
                dup = true;
                break;
            };
            if (dup) continue;
            if (self.n_reads < hazard_range_cap) {
                self.reads[self.n_reads] = r;
                self.n_reads += 1;
            } else self.saturated = true;
        }
    }

    pub fn hazard(self: *const Hazard, next_writes: []const HazardRange, next_reads: []const HazardRange) bool {
        if (self.saturated) return true;
        return hazardNeeded(self.writes[0..self.n_writes], self.reads[0..self.n_reads], next_writes, next_reads);
    }

    pub fn clear(self: *Hazard) void {
        self.* = .{};
    }
};

/// The one command buffer's pending barrier state; the engine is single-threaded, one buffer at a time.
pub var hazard_pending: Hazard = .{};

/// The barrier a deferred dispatch owes, emitted verbatim before the hazarding dispatch or command.
pub fn flushHazard(dev: *const Device, cmd: c.VkCommandBuffer) void {
    if (hazard_pending.n_writes == 0 and hazard_pending.n_reads == 0 and !hazard_pending.saturated) return;
    const b: c.VkMemoryBarrier = .{
        .sType = c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = c.VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = c.VK_ACCESS_SHADER_READ_BIT | c.VK_ACCESS_SHADER_WRITE_BIT | c.VK_ACCESS_TRANSFER_READ_BIT | c.VK_ACCESS_TRANSFER_WRITE_BIT,
    };
    dev.api.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | c.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &b, 0, null, 0, null);
    hazard_pending.clear();
}

pub const max_bindings = 10;

/// Bounded per-pipeline bank of immutable descriptor snapshots; never rewritten until the epoch advances.
pub const batch_cap: u32 = 512;

/// Batch session state shared by every pipeline; driven by beginBatch (batchBegin) and after the fence (batchEnd).
pub var batch_open: bool = false;
/// Advances at every batch open; a pipeline whose seen epoch differs resets its bank cursor.
pub var batch_epoch: u32 = 0;

pub fn batchBegin() void {
    batch_open = true;
    batch_epoch +%= 1;
}

pub fn batchEnd() void {
    batch_open = false;
}

/// Next bank cursor for a pipeline last reset at `seen`: a new epoch resets to 0; the bank refuses past batch_cap.
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
    // Batched-recording bank: lazy pool, sets, write cursor, epoch; staged bindings are snapshotted at record time.
    batch_pool: c.VkDescriptorPool = null,
    batch_sets: [batch_cap]c.VkDescriptorSet = undefined,
    batch_n: u32 = 0,
    batch_cursor: u32 = 0,
    batch_epoch_seen: u32 = 0,
    stash: [max_bindings]Binding = undefined,
    stash_n: u32 = 0,
    /// The pending dispatch's bindings, kept for the hazard check in record (bind is the single door).
    last_bindings: [max_bindings]Binding = undefined,
    last_n: u32 = 0,

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
        var p: Pipeline = .{ .dev = dev, .layout = layout, .local = local };
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
    /// While a batch is open the bindings are staged and snapshotted into the dispatch banked set at record time.
    pub fn bind(p: *Pipeline, bindings: []const Binding) Error!void {
        if (bindings.len != p.layout.bindings) return error.BadBindings;
        var infos: [max_bindings]c.VkDescriptorBufferInfo = undefined;
        for (bindings, 0..) |b, i| {
            if (b.buffer.kind != .device or b.buffer.dev != p.dev) return error.BadBindings;
            if (b.len % 4 != 0) return error.Misaligned; // 32-bit element arrays: a ragged range drops its last element
            infos[i] = try b.buffer.descriptor(b.offset, b.len);
        }
        try disjoint(bindings, p.layout.writes);
        @memcpy(p.last_bindings[0..bindings.len], bindings);
        p.last_n = @intCast(bindings.len);
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

    /// The set this dispatch binds: the pipeline live set immediately, or a fresh bank snapshot per batched dispatch.
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
            // First batched use: the persistent bounded bank; first-use cost only, later tokens reuse the sets.
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
        const reads = c.VK_ACCESS_SHADER_READ_BIT | c.VK_ACCESS_SHADER_WRITE_BIT | c.VK_ACCESS_TRANSFER_READ_BIT | c.VK_ACCESS_TRANSFER_WRITE_BIT;
        const stages = c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | c.VK_PIPELINE_STAGE_TRANSFER_BIT;
        // TF_VK_NOBAR=0: today's stream verbatim — one full barrier after every dispatch.
        if (!elide_barriers) {
            api.vkCmdBindPipeline(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
            api.vkCmdBindDescriptorSets(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline_layout, 0, 1, &set, 0, null);
            if (push.len > 0) api.vkCmdPushConstants(cmd, p.pipeline_layout, c.VK_SHADER_STAGE_COMPUTE_BIT, 0, p.layout.push_bytes, push.ptr);
            api.vkCmdDispatch(cmd, groups[0], groups[1], groups[2]);
            api.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, stages, 0, 1, &c.VkMemoryBarrier{
                .sType = c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = c.VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = reads,
            }, 0, null, 0, null);
            return;
        }
        //  The earlier unsynchronised dispatches' deferred barrier is owed here only on a real hazard.
        const bindings = p.last_bindings[0..p.last_n];
        var next_w: [max_bindings]HazardRange = undefined;
        var next_r: [max_bindings]HazardRange = undefined;
        var nw: u32 = 0;
        var nr: u32 = 0;
        for (bindings, 0..) |b, i| {
            const r = HazardRange{ .buf = b.buffer.handle, .offset = b.offset, .len = b.len };
            if (p.layout.writes >> @intCast(i) & 1 != 0) {
                next_w[nw] = r;
                nw += 1;
            } else {
                next_r[nr] = r;
                nr += 1;
            }
        }
        if (hazard_pending.hazard(next_w[0..nw], next_r[0..nr])) {
            api.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, stages, 0, 1, &c.VkMemoryBarrier{
                .sType = c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = c.VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = reads,
            }, 0, null, 0, null);
            hazard_pending.clear();
        }
        api.vkCmdBindPipeline(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
        api.vkCmdBindDescriptorSets(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline_layout, 0, 1, &set, 0, null);
        if (push.len > 0) api.vkCmdPushConstants(cmd, p.pipeline_layout, c.VK_SHADER_STAGE_COMPUTE_BIT, 0, p.layout.push_bytes, push.ptr);
        api.vkCmdDispatch(cmd, groups[0], groups[1], groups[2]);
        hazard_pending.add(next_w[0..nw], next_r[0..nr]);
    }

    /// One dispatch in its own submission; wrap it in commands.orExit, which ends the process on error.Timeout.
    /// Inside an open batch begin/submit are the session no-op and the dispatch only records into the buffer.
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
    // The 10-binding GDN prefill/decode closures fit; 11 do not; fewer offered still refuses (real capacity).
    l.maxPerStageDescriptorStorageBuffers = 10;
    l.maxDescriptorSetStorageBuffers = 10;
    l.maxPerStageResources = 10;
    try fits(l, .{ .bindings = max_bindings, .push_bytes = 120, .writes = 1 << 9 }, .{ 256, 1, 1 });
    try std.testing.expectError(error.TooManyBindings, fits(l, .{ .bindings = max_bindings + 1, .push_bytes = 120, .writes = 1 << 10 }, .{ 256, 1, 1 }));
    l.maxPerStageDescriptorStorageBuffers = 8;
    try std.testing.expectError(error.TooManyBindings, fits(l, .{ .bindings = max_bindings, .push_bytes = 120, .writes = 1 << 9 }, .{ 256, 1, 1 }));
}

fn hbuf(comptime v: usize) c.VkBuffer {
    return @ptrFromInt(v);
}

fn hr(buf: c.VkBuffer, offset: u64, len: u64) HazardRange {
    return .{ .buf = buf, .offset = offset, .len = len };
}

test "hazardNeeded orders RAW, WAW and WAR and admits independent dispatches" {
    const in = hbuf(0x10);
    const qkv = hbuf(0x20);
    const z = hbuf(0x30);
    // no pending writes: nothing to order against
    try std.testing.expect(!hazardNeeded(&.{hr(qkv, 0, 64)}, &.{}, &.{hr(in, 0, 64)}, &.{hr(in, 0, 64)}));
    const pending = [_]HazardRange{hr(qkv, 0, 64)};
    try std.testing.expect(hazardNeeded(&pending, &.{}, &.{}, &.{hr(qkv, 0, 32)})); // RAW: next reads qkv
    try std.testing.expect(hazardNeeded(&pending, &.{}, &.{hr(qkv, 32, 32)}, &.{})); // WAW: next writes qkv
    // WAR: the next writes a range an unsynchronised dispatch read
    const pending_r = [_]HazardRange{hr(in, 0, 64)};
    try std.testing.expect(hazardNeeded(&.{hr(qkv, 0, 64)}, &pending_r, &.{hr(in, 32, 32)}, &.{}));
    // two dispatches read the same activation and write different buffers: no barrier
    try std.testing.expect(!hazardNeeded(&pending, &.{hr(in, 0, 64)}, &.{hr(z, 0, 64)}, &.{hr(in, 0, 64)}));
    // same buffer, disjoint byte ranges: no barrier; overlapping: barrier
    try std.testing.expect(!hazardNeeded(&.{hr(qkv, 0, 64)}, &.{}, &.{}, &.{hr(qkv, 64, 64)}));
    try std.testing.expect(hazardNeeded(&.{hr(qkv, 0, 65)}, &.{}, &.{}, &.{hr(qkv, 64, 64)}));
}

test "the unsynchronised set accumulates across elided dispatches until a flush" {
    const a_out = hbuf(0x40);
    const b_out = hbuf(0x50);
    const shared_in = hbuf(0x10);
    var h: Hazard = .{};
    h.add(&.{hr(a_out, 0, 64)}, &.{hr(shared_in, 0, 64)}); // a: reads shared, writes a_out
    try std.testing.expect(!h.hazard(&.{}, &.{hr(shared_in, 0, 64)})); // b: reads shared, writes b_out: independent of a
    h.add(&.{hr(b_out, 0, 64)}, &.{hr(shared_in, 0, 64)});
    // c reads a's output (not b's): the barrier is still needed — a is unsynchronised
    try std.testing.expect(h.hazard(&.{}, &.{hr(a_out, 0, 64)}));
    h.clear();
    try std.testing.expect(!h.hazard(&.{}, &.{hr(a_out, 0, 64)})); // after the flush, nothing pending
    // a saturated set needs a barrier rather than dropping ranges
    var full: Hazard = .{};
    for (0..hazard_range_cap + 1) |i| full.add(&.{hr(a_out, @intCast(i * 64), 64)}, &.{});
    try std.testing.expect(full.hazard(&.{}, &.{hr(b_out, 0, 64)}));
    // duplicate ranges do not consume slots twice
    var dup: Hazard = .{};
    for (0..hazard_range_cap) |_| dup.add(&.{hr(a_out, 0, 64)}, &.{});
    try std.testing.expect(!dup.hazard(&.{}, &.{hr(b_out, 0, 64)}));
}

test "elideFromEnv pins today's barriers only for the exact 0" {
    try std.testing.expect(elideFromEnv(null));
    try std.testing.expect(elideFromEnv("1"));
    try std.testing.expect(!elideFromEnv("0"));
    try std.testing.expect(elideFromEnv("off"));
}

test "the batched bank cursor resets per epoch and refuses past the bounded cap" {
    batch_epoch += 1; // beginBatch bumps: the previous batch's fence has been waited
    try std.testing.expectEqual(@as(u32, 7), try nextBatchCursor(batch_epoch, 7)); // same epoch advances
    try std.testing.expectEqual(@as(u32, batch_cap - 1), try nextBatchCursor(batch_epoch, batch_cap - 1)); // last slot
    try std.testing.expectEqual(@as(u32, 0), try nextBatchCursor(batch_epoch -% 1, batch_cap)); // fresh epoch clears even a full cursor
    // exhausting the bank within one epoch refuses instead of rewriting a set in flight
    const full: u32 = batch_cap;
    batch_epoch += 1;
    try std.testing.expectError(error.BatchOverflow, nextBatchCursor(batch_epoch, full));
}
