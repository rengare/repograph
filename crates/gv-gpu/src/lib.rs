//! Device ownership and the graph buffers shared by rendering and compute.
//!
//! This crate exists to break what would otherwise be a cycle. The node buffer
//! is written by the compute passes in `gv-layout-gpu` and read as vertex data
//! by `gv-render`; neither should depend on the other, so both depend on this.
//!
//! The single most important consequence of using wgpu here: the node buffer
//! carries `STORAGE | VERTEX` usage, so the layout passes mutate exactly the
//! memory the vertex shader reads. There is no per-frame copy, which is what
//! the original achieved by binding the same GL name as both an SSBO and an
//! array buffer.

pub mod buffers;
pub mod context;

pub use buffers::GraphBuffers;
pub use context::GpuContext;
