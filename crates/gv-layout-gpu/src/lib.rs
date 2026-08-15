//! Layout algorithms as wgpu compute passes — the reason this project exists.

pub mod bh_gpu;
pub mod fr_gpu;
pub mod radix_sort;

pub use bh_gpu::BhGpuLayout;
pub use fr_gpu::FrGpuLayout;

use anyhow::Result;
use gv_layout::LayoutParams;

/// A layout that advances node state held in a [`gv_gpu::GraphBuffers`].
///
/// Distinct from [`gv_layout::CpuLayout`] because the node array never reaches
/// host memory: a step records dispatches into a caller-supplied encoder so it
/// can share a submission with the frame's draw calls.
pub trait GpuLayout {
    fn name(&self) -> &'static str;

    /// Records this step's dispatches. The caller submits.
    fn record_step(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        params: &LayoutParams,
    ) -> Result<()>;
}
