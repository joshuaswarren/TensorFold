// Qwen3.5 root resolves backend ../../ imports; backend.zig cannot resolve them alone.
pub const backend = @import("backend/vulkan/backend.zig");
pub const engine = @import("backend/vulkan/engine.zig");
pub const config = @import("config.zig");
/// The engine's Vulkan backend module, for tools that inspect engine state
/// (the model probe's barrier-elision milestone line).
pub const vk = @import("vulkan");
