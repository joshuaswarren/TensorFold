//! The MXFP8 128x64 prompt tile's launch geometry is exact; its compiled usage stays pending until a nvcc build measures it.
const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");

fn mxfp8(comptime wide: bool, comptime fused: bool) registry.CudaRecord {
    return .{ .launch = .{
        .name = "qmmf MXFP8 prompt 128x64 " ++ (if (wide) "fp32" else "bf16") ++ (if (fused) " fused slices" else " unsplit"),
        .symbol = std.fmt.comptimePrint("_ZN7tf_qmmf11qmmf_kernelILi2ELi128ELi64ELi2ELi4ELi4ELb{d}ELb0ELb{d}EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .{ @intFromBool(wide), @intFromBool(fused) }),
        .image = "qmmf",
        .threads = 256,
        .dynamic = 82432,
        .opt_in = true,
    }, .usage = registry.none };
}

pub const records = [_]registry.CudaRecord{ mxfp8(false, false), mxfp8(false, true), mxfp8(true, false), mxfp8(true, true) };

pub fn owns(name: []const u8) bool {
    for (records) |r| if (std.mem.eql(u8, r.launch.name, name)) return true;
    return false;
}

test "qmmf MXFP8 prompt: launch geometry pinned, unmeasured compiler usage stays pending" {
    for (records) |r| {
        try std.testing.expectEqual(@as(u32, 256), r.launch.threads);
        try std.testing.expectEqual(@as(u32, 82432), r.launch.dynamic);
        try std.testing.expect(r.launch.opt_in);
        for ([_]?limits.Nvcc{ null, .{ .major = 13, .minor = 0 }, .{ .major = 13, .minor = 1 }, .{ .major = 13, .minor = 2 }, .{ .major = 13, .minor = 3 } }) |version| {
            try std.testing.expectEqualDeep(registry.none, registry.usageFor(r, version));
        }
    }
    try std.testing.expectEqual(@as(usize, 20), try registry.checkCuda(std.testing.allocator, &records, true));
}
