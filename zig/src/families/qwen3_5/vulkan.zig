//! The qwen3.5 family's thin Vulkan root: a module anchored here can resolve
//! the backend's ../../ relative imports; backend.zig alone cannot.
pub const backend = @import("backend/vulkan/backend.zig");
pub const engine = @import("backend/vulkan/engine.zig");
pub const config = @import("config.zig");
/// The engine's Vulkan backend module, for tools that inspect engine state
/// (the model probe's barrier-elision milestone line).
=======
/// The engine's Vulkan backend module, for the dry-run harness and tools.
>>>>>>> 55183265 (perf(vulkan): the harness's dump hook and the backend module re-export it needs)
pub const vk = @import("vulkan");
