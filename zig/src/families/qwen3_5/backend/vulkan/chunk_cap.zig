//! The engine's per-pass forward chunk ceiling, shared verbatim with the kernel contract's row
//! envelope (the projection's row counts are the forward pass's chunk sizes; backend.zig's
//! request-facing max_prompt_rows is the same 32).
pub const max_prefill_rows: usize = 32;
