//! A `native_engines` stand-in for the CLI's host tests: the families the fixtures serve, on any host.
pub const families: []const struct { model_type: []const u8, formats: []const []const u8 } = &.{
    .{ .model_type = "nemotron_h", .formats = &.{"mlx-q4g64"} },
    .{ .model_type = "qwen4_exp", .formats = &.{"mlx-q6g32"} },
};
