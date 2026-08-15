//! Adapter selection and device creation.

use std::sync::Arc;

use anyhow::{Context, Result};

/// The wgpu instance, adapter, device and queue, kept together and cloned
/// cheaply into whatever needs them.
#[derive(Debug, Clone)]
pub struct GpuContext {
    pub instance: Arc<wgpu::Instance>,
    pub adapter: Arc<wgpu::Adapter>,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

impl GpuContext {
    /// Requests an adapter able to present to `surface`, then a device.
    ///
    /// Backend selection honours `WGPU_BACKEND`; on this project's aarch64
    /// target the default resolves to Vulkan.
    pub async fn new(surface: Option<&wgpu::Surface<'_>>) -> Result<Self> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        Self::from_instance(instance, surface).await
    }

    /// As [`Self::new`], but reusing an instance that has already created a
    /// surface — an adapter must be requested from the same instance the
    /// surface came from.
    pub async fn from_instance(
        instance: wgpu::Instance,
        surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<Self> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: surface,
            })
            .await
            .context(if cfg!(target_arch = "wasm32") {
                "no WebGPU adapter — enable WebGPU (Chrome/Edge 113+, or Firefox with \
                 dom.webgpu.enabled) and serve over https or localhost"
            } else {
                "no GPU adapter available; set WGPU_BACKEND to try another backend"
            })?;

        let info = adapter.get_info();
        log::info!("adapter: {} ({:?}, {:?})", info.name, info.device_type, info.backend);

        // Ask for everything the adapter offers rather than the downlevel
        // defaults: the storage-buffer binding size is what caps graph size,
        // and the defaults would cut the largest bundled datasets short.
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("gv-gpu device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .context("requesting a device from the selected adapter")?;

        Ok(Self {
            instance: Arc::new(instance),
            adapter: Arc::new(adapter),
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }

    /// Limits the graph must fit inside — chiefly
    /// `max_storage_buffer_binding_size`, which caps how many nodes a single
    /// binding can hold and so caps graph size.
    pub fn limits(&self) -> wgpu::Limits {
        self.device.limits()
    }

    /// The largest graph a single storage binding can hold under `limits`.
    ///
    /// Pure arithmetic, so it is testable without an adapter — and it is the
    /// check that turns an opaque driver-side failure on a 500k-edge dataset
    /// into a message naming the limit that was exceeded.
    pub fn max_nodes_for(limits: &wgpu::Limits) -> u64 {
        limits.max_storage_buffer_binding_size / size_of::<gv_graph::Node>() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_limits_admit_the_largest_bundled_dataset() {
        // array.1.edges is 999,999 edges; node counts are far smaller, but the
        // CSR neighbour buffer is 2 * edges. Downlevel defaults must clear it.
        let limits = wgpu::Limits::downlevel_defaults();
        let neighbors_bytes = 2 * 999_999u64 * size_of::<u32>() as u64;
        assert!(
            neighbors_bytes <= limits.max_storage_buffer_binding_size,
            "CSR neighbours ({neighbors_bytes} bytes) exceed the downlevel binding limit"
        );
    }

    #[test]
    fn max_nodes_divides_the_binding_limit_by_the_node_size() {
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: 480,
            ..wgpu::Limits::downlevel_defaults()
        };
        assert_eq!(GpuContext::max_nodes_for(&limits), 10);
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn creates_a_device_without_a_surface() {
        let context = pollster::block_on(GpuContext::new(None)).expect("adapter available");
        assert!(context.limits().max_compute_workgroup_size_x >= 256);
    }
}
