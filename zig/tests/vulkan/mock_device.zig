//! The dry-run harness's mock Vulkan device: every DeviceApi entry point is a
//! C stub that (a) hands out fake handles, (b) backs all device memory with
//! real zeroed host blocks so staged uploads, copies, fills and read-downs
//! move real bytes, (c) records the command stream (dispatches with pipeline
//! identity and push-constant hash, barriers, copies, fills, submits) and
//! (d) never executes anything. Test-only: nothing in the product imports it.
const std = @import("std");
const vulkan = @import("vulkan");
const c = vulkan.c;

pub const Entry = union(enum) {
    dispatch: struct { pipeline: u64, groups: [3]u32, push_hash: u64 },
    barrier: void,
    copy: struct { src_buf: u64, dst_buf: u64, src_off: u64, dst_off: u64, bytes: u64 },
    fill: struct { buf: u64, offset: u64, bytes: u64, value: u32 },
    push: struct { pipeline_layout: u64, offset: u32, size: u32, hash: u64 },
    bind_sets: struct { first: u32, sets: [4]u64 },
    submit: void,
};

pub const SetWrite = struct { set: u64, binding: u32, buffer: u64, offset: u64, range: u64 };

pub const Mock = struct {
    gpa: std.mem.Allocator,
    alignment: u64 = 256,
    next: u64 = 0x1000,
    entries: std.ArrayList(Entry) = .empty,
    set_writes: std.ArrayList(SetWrite) = .empty,
    memory_host: std.AutoHashMapUnmanaged(u64, []u8) = .empty,
    buffer_mem: std.AutoHashMapUnmanaged(u64, struct { mem: u64, offset: u64, size: u64 }) = .empty,
    shader_hash: std.AutoHashMapUnmanaged(u64, u64) = .empty,
    pipeline_shader: std.AutoHashMapUnmanaged(u64, u64) = .empty,
    set_of: std.AutoHashMapUnmanaged(u64, u64) = .empty, // bound set handle -> set id
    current_pipeline: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, alignment: u64) !*Mock {
        const m = try gpa.create(Mock);
        m.* = .{ .gpa = gpa, .alignment = alignment };
        global = m;
        return m;
    }

    pub fn deinit(m: *Mock) void {
        var it = m.memory_host.valueIterator();
        while (it.next()) |block| m.gpa.free(block.*);
        m.memory_host.deinit(m.gpa);
        m.buffer_mem.deinit(m.gpa);
        m.shader_hash.deinit(m.gpa);
        m.pipeline_shader.deinit(m.gpa);
        m.set_of.deinit(m.gpa);
        m.entries.deinit(m.gpa);
        m.set_writes.deinit(m.gpa);
        m.gpa.destroy(m);
        global = null;
    }

    fn handle(m: *Mock) u64 {
        m.next += 0x40;
        return m.next;
    }

    fn log(m: *Mock, e: Entry) void {
        m.entries.append(m.gpa, e) catch {};
    }

    /// The host bytes behind a buffer: its memory block at the bound offset.
    fn resolve(m: *Mock, buffer: u64, offset: u64) ?[*]u8 {
        const bm = m.buffer_mem.get(buffer) orelse return null;
        const block = m.memory_host.get(bm.mem) orelse return null;
        return block.ptr + bm.offset + offset;
    }

    pub fn dispatchCount(m: *const Mock) usize {
        var n: usize = 0;
        for (m.entries.items) |e| if (e == .dispatch) {
            n += 1;
        };
        return n;
    }

    pub fn barrierCount(m: *const Mock) usize {
        var n: usize = 0;
        for (m.entries.items) |e| if (e == .barrier) {
            n += 1;
        };
        return n;
    }

    /// sha256 over the normalized stream: one line per entry, handles
    /// normalized to their allocation ordinal so run-to-run handle values
    /// never change the hash.
    pub fn logSha(m: *const Mock, out: *[32]u8) !void {
        var bytes: std.ArrayList(u8) = .empty;
        defer bytes.deinit(m.gpa);
        const dump = std.c.getenv("TF_DRYRUN_DUMP") != null;
        var ords: std.AutoHashMapUnmanaged(u64, u64) = .empty;
        defer ords.deinit(m.gpa);
        const dump = std.c.getenv("TF_DRYRUN_DUMP") != null;
        for (m.entries.items) |e| {
            const W = struct {
                fn f(l: *std.ArrayList(u8), al: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !void {
                    try l.appendSlice(al, try std.fmt.allocPrint(al, fmt, args));
                }
            }.f;
            const G = m.gpa;
            switch (e) {
                .dispatch => |d| try W(&bytes, G, "D {d} {d},{d},{d}\n", .{ m.ord(&ords, d.pipeline), d.groups[0], d.groups[1], d.groups[2] }),
                .barrier => try W(&bytes, G, "B\n", .{}),
                .copy => |cp| try W(&bytes, G, "C {d} {d} {d} {d} {d}\n", .{ m.ord(&ords, cp.src_buf), m.ord(&ords, cp.dst_buf), cp.src_off, cp.dst_off, cp.bytes }),
                .fill => |f| try W(&bytes, G, "F {d} {d} {d} {d}\n", .{ m.ord(&ords, f.buf), f.offset, f.bytes, f.value }),
                .push => |p| try W(&bytes, G, "P {d} {d} {d}\n", .{ m.ord(&ords, p.pipeline_layout), p.offset, p.size }),
                .bind_sets => |b| try W(&bytes, G, "S {d} {d},{d},{d},{d}\n", .{ b.first, m.ord(&ords, b.sets[0]), m.ord(&ords, b.sets[1]), m.ord(&ords, b.sets[2]), m.ord(&ords, b.sets[3]) }),
                .submit => try W(&bytes, G, "Q\n", .{}),
            }
        }
        if (dump) std.debug.print("{s}", .{bytes.items});
        std.crypto.hash.sha2.Sha256.hash(bytes.items, out, .{});
    }

    fn ord(m: *const Mock, map: *std.AutoHashMapUnmanaged(u64, u64), v: u64) u64 {
        const r = map.getOrPut(m.gpa, v) catch return v;
        if (!r.found_existing) r.value_ptr.* = map.count();
        return r.value_ptr.*;
    }
};

// The stubs run on the engine's single thread; one file-scope instance.
var global: ?*Mock = null;

fn g() *Mock {
    return global.?;
}

fn ok() c.VkResult {
    return c.VK_SUCCESS;
}

export fn mockDestroyDevice(_: ?*anyopaque, _: ?*const anyopaque) void {}

export fn mockGetDeviceQueue(_: ?*anyopaque, _: u32, _: u32, pQueue: ?*?*anyopaque) void {
    pQueue.?.* = @ptrFromInt(g().handle());
}

export fn mockQueueSubmit(_: ?*anyopaque, _: u32, _: ?*const anyopaque, _: ?*anyopaque) c.VkResult {
    g().log(.submit);
    return ok();
}

export fn mockCreateBuffer(_: ?*anyopaque, pCreateInfo: ?*const c.VkBufferCreateInfo, _: ?*const anyopaque, pBuffer: ?*c.VkBuffer) c.VkResult {
    if (std.c.getenv("TF_DRYRUN_TRACE") != null) std.debug.print("MOCK createBuffer size={d}\n", .{pCreateInfo.?.size});
    const h = g().handle();
    pBuffer.?.* = @ptrFromInt(h);
    _ = g().buffer_mem.put(g().gpa, h, .{ .mem = 0, .offset = 0, .size = pCreateInfo.?.size }) catch return c.VK_ERROR_OUT_OF_HOST_MEMORY;
    return ok();
}

export fn mockDestroyBuffer(_: ?*anyopaque, _: ?*anyopaque, _: ?*const anyopaque) void {}

export fn mockGetBufferMemoryRequirements(_: ?*anyopaque, buffer: ?*anyopaque, pMemoryRequirements: ?*c.VkMemoryRequirements) void {
    const bm = g().buffer_mem.get(@intFromPtr(buffer)).?;
    pMemoryRequirements.?.size = bm.size;
    pMemoryRequirements.?.alignment = @max(g().alignment, 16);
    pMemoryRequirements.?.memoryTypeBits = 1;
}

export fn mockAllocateMemory(_: ?*anyopaque, pAllocateInfo: ?*const c.VkMemoryAllocateInfo, _: ?*const anyopaque, pMemory: ?*c.VkDeviceMemory) c.VkResult {
    const size: usize = @intCast(pAllocateInfo.?.allocationSize);
    if (std.c.getenv("TF_DRYRUN_TRACE") != null) std.debug.print("MOCK alloc size={d}\n", .{size});
    const block = g().gpa.alloc(u8, @max(size, 1 << 12)) catch return c.VK_ERROR_OUT_OF_HOST_MEMORY;
    @memset(block, 0);
    const h = g().handle();
    g().memory_host.put(g().gpa, h, block) catch return c.VK_ERROR_OUT_OF_HOST_MEMORY;
    pMemory.?.* = @ptrFromInt(h);
    return ok();
}

export fn mockFreeMemory(_: ?*anyopaque, memory: ?*c.VkDeviceMemory, _: ?*const anyopaque) void {
    if (memory) |mem| {
        const h = @intFromPtr(mem);
        if (g().memory_host.fetchRemove(h)) |kv| g().gpa.free(kv.value);
    }
}

export fn mockBindBufferMemory(_: ?*anyopaque, buffer: ?*anyopaque, memory: ?*c.VkDeviceMemory, memoryOffset: c.VkDeviceSize) c.VkResult {
    const h = @intFromPtr(buffer);
    if (g().buffer_mem.getPtr(h)) |bm| {
        bm.mem = @intFromPtr(memory);
        bm.offset = @intCast(memoryOffset);
    }
    return ok();
}

export fn mockMapMemory(_: ?*anyopaque, memory: ?*c.VkDeviceMemory, offset: c.VkDeviceSize, _: c.VkDeviceSize, _: u32, ppData: ?*?*anyopaque) c.VkResult {
    const block = g().memory_host.get(@intFromPtr(memory)) orelse return c.VK_ERROR_OUT_OF_HOST_MEMORY;
    ppData.?.* = block.ptr + @as(usize, @intCast(offset));
    return ok();
}

export fn mockUnmapMemory(_: ?*anyopaque, _: ?*c.VkDeviceMemory) void {}

export fn mockFlushRanges(_: ?*anyopaque, _: u32, _: ?*const anyopaque) c.VkResult {
    return ok();
}

export fn mockInvalidateRanges(_: ?*anyopaque, _: u32, _: ?*const anyopaque) c.VkResult {
    return ok();
}

export fn mockCreateCommandPool(_: ?*anyopaque, _: ?*const anyopaque, _: ?*const anyopaque, pCommandPool: ?*c.VkCommandPool) c.VkResult {
    pCommandPool.?.* = @ptrFromInt(g().handle());
    return ok();
}

export fn mockDestroyCommandPool(_: ?*anyopaque, _: ?*anyopaque, _: ?*const anyopaque) void {}

export fn mockAllocateCommandBuffers(_: ?*anyopaque, _: ?*const anyopaque, pCommandBuffers: ?*c.VkCommandBuffer) c.VkResult {
    pCommandBuffers.?.* = @ptrFromInt(g().handle());
    return ok();
}

export fn mockResetCommandBuffer(_: ?*anyopaque, _: u32) c.VkResult {
    return ok();
}

export fn mockBeginCommandBuffer(_: ?*anyopaque, _: ?*const anyopaque) c.VkResult {
    return ok();
}

export fn mockEndCommandBuffer(_: ?*anyopaque) c.VkResult {
    return ok();
}

export fn mockCreateFence(_: ?*anyopaque, _: ?*const anyopaque, _: ?*const anyopaque, pFence: ?*c.VkFence) c.VkResult {
    pFence.?.* = @ptrFromInt(g().handle());
    return ok();
}

export fn mockDestroyFence(_: ?*anyopaque, _: ?*anyopaque, _: ?*const anyopaque) void {}

export fn mockWaitForFences(_: ?*anyopaque, _: u32, _: ?*const ?*anyopaque, _: u32, _: u64) c.VkResult {
    return c.VK_SUCCESS;
}

export fn mockResetFences(_: ?*anyopaque, _: u32, _: ?*const ?*anyopaque) c.VkResult {
    return ok();
}

export fn mockCmdPipelineBarrier(_: ?*anyopaque, _: u32, _: u32, _: u32, _: u32, _: ?*const anyopaque, _: u32, _: ?*const anyopaque, _: u32, _: ?*const anyopaque) void {
    g().log(.barrier);
}

export fn mockCmdFillBuffer(_: ?*anyopaque, dstBuffer: ?*anyopaque, dstOffset: c.VkDeviceSize, size: c.VkDeviceSize, data: u32) void {
    const m = g();
    if (m.resolve(@intFromPtr(dstBuffer), @intCast(dstOffset))) |p| {
        const n: usize = @intCast(size);
        var i: usize = 0;
        while (i + 4 <= n) : (i += 4) @as(*align(1) u32, @ptrCast(p + i)).* = data;
    }
    m.log(.{ .fill = .{ .buf = @intFromPtr(dstBuffer), .offset = @intCast(dstOffset), .bytes = @intCast(size), .value = data } });
}

export fn mockCmdCopyBuffer(_: ?*anyopaque, srcBuffer: ?*anyopaque, dstBuffer: ?*anyopaque, regionCount: u32, pRegions: ?*const c.VkBufferCopy) void {
    const m = g();
    var i: u32 = 0;
    const regions: [*]const c.VkBufferCopy = @ptrCast(pRegions.?);
    while (i < regionCount) : (i += 1) {
        const region = regions[i];
        if (m.resolve(@intFromPtr(srcBuffer), @intCast(region.srcOffset))) |src| {
            if (m.resolve(@intFromPtr(dstBuffer), @intCast(region.dstOffset))) |dst| {
                @memcpy(dst[0..@intCast(region.size)], src[0..@intCast(region.size)]);
                // The engine's only 12-byte copy is the greedy verdict slice
                // (Result { finite_flag, first_bad, index }). A real device
                // would have executed the argreduce chain; the mock writes
                // what it would produce on all-finite logits: flag 0, chosen
                // id 0 (mock-only; never product code).
                if (region.size == 12) {
                    const verdict: [3]u32 = .{ 0, 0, 0 };
                    @memcpy(dst[0..12], std.mem.sliceAsBytes(&verdict)[0..12]);
                }
            }
        }
        m.log(.{ .copy = .{ .src_buf = @intFromPtr(srcBuffer), .dst_buf = @intFromPtr(dstBuffer), .src_off = @intCast(region.srcOffset), .dst_off = @intCast(region.dstOffset), .bytes = @intCast(region.size) } });
    }
}

export fn mockCmdBindPipeline(_: ?*anyopaque, _: u32, pipeline: ?*anyopaque) void {
    g().current_pipeline = @intFromPtr(pipeline);
}

export fn mockCmdBindDescriptorSets(_: ?*anyopaque, _: u32, _: ?*anyopaque, firstSet: u32, setCount: u32, pSets: ?*const ?*anyopaque, _: u32, _: ?*const u32) void {
    const m = g();
    var sets: [4]u64 = .{ 0, 0, 0, 0 };
    var i: u32 = 0;
    const sets_in: [*]const ?*anyopaque = @ptrCast(pSets.?);
    while (i < @min(setCount, 4)) : (i += 1) {
        sets[i] = @intFromPtr(sets_in[i]);
    }
    m.log(.{ .bind_sets = .{ .first = firstSet, .sets = sets } });
}

export fn mockCmdPushConstants(_: ?*anyopaque, pipeline_layout: ?*anyopaque, _: u32, offset: u32, size: u32, pValues: ?*const anyopaque) void {
    const m = g();
    const bytes: [*]const u8 = @ptrCast(pValues.?);
    m.log(.{ .push = .{ .pipeline_layout = @intFromPtr(pipeline_layout), .offset = offset, .size = size, .hash = std.hash.Wyhash.hash(0, bytes[0..size]) } });
}

export fn mockCmdDispatch(_: ?*anyopaque, x: u32, y: u32, z: u32) void {
    const m = g();
    m.log(.{ .dispatch = .{ .pipeline = m.current_pipeline, .groups = .{ x, y, z }, .push_hash = 0 } });
}

export fn mockCreateShaderModule(_: ?*anyopaque, pCreateInfo: ?*const c.VkShaderModuleCreateInfo, _: ?*const anyopaque, pShaderModule: ?*c.VkShaderModule) c.VkResult {
    const m = g();
    const info = pCreateInfo.?;
    const bytes: []const u8 = @as([*]const u8, @ptrCast(info.pCode.?))[0..info.codeSize];
    const hash = std.hash.Wyhash.hash(0, bytes);
    const h = m.handle();
    m.shader_hash.put(m.gpa, h, hash) catch return c.VK_ERROR_OUT_OF_HOST_MEMORY;
    pShaderModule.?.* = @ptrFromInt(h);
    return ok();
}

export fn mockDestroyShaderModule(_: ?*anyopaque, _: ?*anyopaque, _: ?*const anyopaque) void {}

export fn mockCreateDescriptorSetLayout(_: ?*anyopaque, _: ?*const anyopaque, _: ?*const anyopaque, pSetLayout: ?*c.VkDescriptorSetLayout) c.VkResult {
    pSetLayout.?.* = @ptrFromInt(g().handle());
    return ok();
}

export fn mockDestroyDescriptorSetLayout(_: ?*anyopaque, _: ?*anyopaque, _: ?*const anyopaque) void {}

export fn mockCreatePipelineLayout(_: ?*anyopaque, _: ?*const anyopaque, _: ?*const anyopaque, pPipelineLayout: ?*c.VkPipelineLayout) c.VkResult {
    pPipelineLayout.?.* = @ptrFromInt(g().handle());
    return ok();
}

export fn mockDestroyPipelineLayout(_: ?*anyopaque, _: ?*anyopaque, _: ?*const anyopaque) void {}

export fn mockCreateComputePipelines(_: ?*anyopaque, _: ?*anyopaque, createInfoCount: u32, pCreateInfos: ?*const c.VkComputePipelineCreateInfo, _: ?*const anyopaque, pPipelines: ?*c.VkPipeline) c.VkResult {
    const m = g();
    var i: u32 = 0;
    const infos: [*]const c.VkComputePipelineCreateInfo = @ptrCast(pCreateInfos.?);
    while (i < createInfoCount) : (i += 1) {
        const stage = infos[i].stage;
        const module = @intFromPtr(stage.module);
        const h = m.handle();
        m.pipeline_shader.put(m.gpa, h, m.shader_hash.get(module) orelse 0) catch return c.VK_ERROR_OUT_OF_HOST_MEMORY;
        const out_pipes: [*]c.VkPipeline = @ptrCast(pPipelines.?);
        out_pipes[i] = @ptrFromInt(h);
    }
    return ok();
}

export fn mockDestroyPipeline(_: ?*anyopaque, _: ?*anyopaque, _: ?*const anyopaque) void {}

export fn mockCreateDescriptorPool(_: ?*anyopaque, _: ?*const anyopaque, _: ?*const anyopaque, pDescriptorPool: ?*c.VkDescriptorPool) c.VkResult {
    pDescriptorPool.?.* = @ptrFromInt(g().handle());
    return ok();
}

export fn mockDestroyDescriptorPool(_: ?*anyopaque, _: ?*anyopaque, _: ?*const anyopaque) void {}

export fn mockAllocateDescriptorSets(_: ?*anyopaque, pAllocateInfo: ?*const c.VkDescriptorSetAllocateInfo, pDescriptorSets: ?*c.VkDescriptorSet) c.VkResult {
    const m = g();
    var i: u32 = 0;
    const out_sets: [*]c.VkDescriptorSet = @ptrCast(pDescriptorSets.?);
    while (i < pAllocateInfo.?.descriptorSetCount) : (i += 1) {
        const h = m.handle();
        out_sets[i] = @ptrFromInt(h);
        m.set_of.put(m.gpa, h, h) catch return c.VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    return ok();
}

export fn mockUpdateDescriptorSets(_: ?*anyopaque, descriptorWriteCount: u32, pDescriptorWrites: ?*const c.VkWriteDescriptorSet, _: u32, _: ?*const anyopaque) void {
    const m = g();
    var i: u32 = 0;
    const writes: [*]const c.VkWriteDescriptorSet = @ptrCast(pDescriptorWrites.?);
    while (i < descriptorWriteCount) : (i += 1) {
        const w = writes[i];
        const set = @intFromPtr(w.dstSet);
        const bi_opt: ?*const c.VkDescriptorBufferInfo = w.pBufferInfo;
        if (bi_opt) |bi| {
            m.set_writes.append(m.gpa, .{ .set = set, .binding = w.dstBinding, .buffer = @intFromPtr(bi.buffer), .offset = @intCast(bi.offset), .range = @intCast(bi.range) }) catch {};
        }
    }
}

/// The mock DeviceApi: every field points at its stub.
pub fn api() vulkan.device.DeviceApi {
    return .{
        .vkDestroyDevice = @ptrCast(&mockDestroyDevice),
        .vkGetDeviceQueue = @ptrCast(&mockGetDeviceQueue),
        .vkQueueSubmit = @ptrCast(&mockQueueSubmit),
        .vkCreateBuffer = @ptrCast(&mockCreateBuffer),
        .vkDestroyBuffer = @ptrCast(&mockDestroyBuffer),
        .vkGetBufferMemoryRequirements = @ptrCast(&mockGetBufferMemoryRequirements),
        .vkAllocateMemory = @ptrCast(&mockAllocateMemory),
        .vkFreeMemory = @ptrCast(&mockFreeMemory),
        .vkBindBufferMemory = @ptrCast(&mockBindBufferMemory),
        .vkMapMemory = @ptrCast(&mockMapMemory),
        .vkUnmapMemory = @ptrCast(&mockUnmapMemory),
        .vkFlushMappedMemoryRanges = @ptrCast(&mockFlushRanges),
        .vkInvalidateMappedMemoryRanges = @ptrCast(&mockInvalidateRanges),
        .vkCreateCommandPool = @ptrCast(&mockCreateCommandPool),
        .vkDestroyCommandPool = @ptrCast(&mockDestroyCommandPool),
        .vkAllocateCommandBuffers = @ptrCast(&mockAllocateCommandBuffers),
        .vkResetCommandBuffer = @ptrCast(&mockResetCommandBuffer),
        .vkBeginCommandBuffer = @ptrCast(&mockBeginCommandBuffer),
        .vkEndCommandBuffer = @ptrCast(&mockEndCommandBuffer),
        .vkCreateFence = @ptrCast(&mockCreateFence),
        .vkDestroyFence = @ptrCast(&mockDestroyFence),
        .vkWaitForFences = @ptrCast(&mockWaitForFences),
        .vkResetFences = @ptrCast(&mockResetFences),
        .vkCmdPipelineBarrier = @ptrCast(&mockCmdPipelineBarrier),
        .vkCmdFillBuffer = @ptrCast(&mockCmdFillBuffer),
        .vkCmdCopyBuffer = @ptrCast(&mockCmdCopyBuffer),
        .vkCmdBindPipeline = @ptrCast(&mockCmdBindPipeline),
        .vkCmdBindDescriptorSets = @ptrCast(&mockCmdBindDescriptorSets),
        .vkCmdPushConstants = @ptrCast(&mockCmdPushConstants),
        .vkCmdDispatch = @ptrCast(&mockCmdDispatch),
        .vkCreateShaderModule = @ptrCast(&mockCreateShaderModule),
        .vkDestroyShaderModule = @ptrCast(&mockDestroyShaderModule),
        .vkCreateDescriptorSetLayout = @ptrCast(&mockCreateDescriptorSetLayout),
        .vkDestroyDescriptorSetLayout = @ptrCast(&mockDestroyDescriptorSetLayout),
        .vkCreatePipelineLayout = @ptrCast(&mockCreatePipelineLayout),
        .vkDestroyPipelineLayout = @ptrCast(&mockDestroyPipelineLayout),
        .vkCreateComputePipelines = @ptrCast(&mockCreateComputePipelines),
        .vkDestroyPipeline = @ptrCast(&mockDestroyPipeline),
        .vkCreateDescriptorPool = @ptrCast(&mockCreateDescriptorPool),
        .vkDestroyDescriptorPool = @ptrCast(&mockDestroyDescriptorPool),
        .vkAllocateDescriptorSets = @ptrCast(&mockAllocateDescriptorSets),
        .vkUpdateDescriptorSets = @ptrCast(&mockUpdateDescriptorSets),
    };
}

/// A device-shaped struct over the mock table, with the properties the
/// engine's gates read (alignment as a parameter; subgroup 32 + arithmetic;
/// no denormal preserve) and storage16/int16 enabled.
pub fn device(gpa: std.mem.Allocator, alignment: u64) !struct { dev: *vulkan.Device, mock: *Mock } {
    const m = try Mock.init(gpa, alignment);
    const dev = try gpa.create(vulkan.Device);
    dev.* = .{
        .rt = undefined,
        .physical = null,
        .handle = @ptrFromInt(m.handle()),
        .api = api(),
        .queue = @ptrFromInt(m.handle()),
        .queue_family = 0,
        .properties = std.mem.zeroInit(c.VkPhysicalDeviceProperties, .{}),
        .v11 = .{ .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES },
        .v12 = .{ .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES },
        .v13 = .{ .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES },
        .memory = std.mem.zeroes(c.VkPhysicalDeviceMemoryProperties), // populated below
        .features = .{ .storage16 = true, .int16 = true },
    };
    dev.properties.limits.minStorageBufferOffsetAlignment = @intCast(alignment);
    dev.properties.limits.maxStorageBufferRange = (1 << 32) - 1;
    dev.properties.limits.maxComputeWorkGroupCount = .{ 65535, 65535, 65535 };
    dev.properties.limits.maxComputeWorkGroupSize = .{ 1024, 1024, 1024 };
    dev.properties.limits.maxComputeWorkGroupInvocations = 1024;
    dev.properties.limits.maxPushConstantsSize = 256;
    dev.properties.limits.maxPerStageDescriptorStorageBuffers = 128;
    dev.properties.limits.maxDescriptorSetStorageBuffers = 128;
    dev.properties.limits.maxPerStageResources = 128;
    dev.properties.limits.minMemoryMapAlignment = 64;
    dev.properties.limits.nonCoherentAtomSize = 256;
    dev.v11.subgroupSize = 32;
    dev.v11.subgroupSupportedOperations = 1 << 2 | 1 << 3 | 1 << 5; // ARITHMETIC | VOTE | SHUFFLE_RELATIVE
    dev.v11.subgroupSupportedStages = c.VK_SHADER_STAGE_COMPUTE_BIT;
    dev.properties.limits.maxComputeSharedMemorySize = 64 << 10;
    dev.v11.maxMemoryAllocationSize = (1 << 32) - 1;
    dev.v12.shaderDenormPreserveFloat32 = 0;
    dev.v13.maxBufferSize = (1 << 32) - 1;
    dev.memory.memoryTypeCount = 1;
    dev.memory.memoryTypes[0] = .{
        .propertyFlags = c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        .heapIndex = 0,
    };
    dev.memory.memoryHeapCount = 1;
    dev.memory.memoryHeaps[0] = .{ .size = 1 << 40, .flags = c.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT };
    return .{ .dev = dev, .mock = m };
}
