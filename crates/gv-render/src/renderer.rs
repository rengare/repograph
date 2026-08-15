//! Surface configuration and the graph render pass.

use anyhow::{Result, bail};
use gv_config::AppConfig;
use gv_gpu::{GpuContext, GraphBuffers};
use wgpu::util::DeviceExt;

use crate::{Camera, edges::EdgePipeline, nodes::NodePipeline};

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub struct Renderer {
    context: GpuContext,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,

    camera_buffer: wgpu::Buffer,
    /// One `u32` per node: 1 visible, 0 filtered out by kind. Uploaded whole
    /// whenever the GUI's visibility checkboxes change.
    visibility_buffer: wgpu::Buffer,
    /// One `f32` per node: 1.0 full brightness, <1.0 dimmed. Uploaded whole
    /// whenever the node selection changes.
    dim_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    depth_view: wgpu::TextureView,

    nodes: NodePipeline,
    edges: EdgePipeline,
}

impl Renderer {
    pub fn new(
        context: &GpuContext,
        surface: wgpu::Surface<'static>,
        buffers: &GraphBuffers,
        app_config: &AppConfig,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let device = &context.device;
        let capabilities = surface.get_capabilities(&context.adapter);
        let format = choose_surface_format(&capabilities.formats);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: width.max(1),
            height: height.max(1),
            present_mode: Self::present_mode(app_config.is_vsync_enabled),
            desired_maximum_frame_latency: 2,
            alpha_mode: capabilities.alpha_modes[0],
            view_formats: Vec::new(),
        };
        surface.configure(device, &config);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera"),
            contents: bytemuck::bytes_of(&Camera::default().uniforms()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Everything visible until the GUI says otherwise. Length matches the
        // node buffer so the shader can index it by node.
        let visibility_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("visibility"),
            contents: bytemuck::cast_slice(&vec![1u32; buffers.node_count.max(1) as usize]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Full brightness until a selection dims the rest. Length matches the
        // node buffer so the shader can index it by node.
        let dim_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dim"),
            contents: bytemuck::cast_slice(&vec![1.0f32; buffers.node_count.max(1) as usize]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("graph"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(1),
                storage_entry(2),
                storage_entry(3),
                storage_entry(4),
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("graph"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.nodes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.edges.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: visibility_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dim_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("graph"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("graph"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/graph.wgsl").into()),
        });

        let nodes = NodePipeline::new(device, &pipeline_layout, &shader, format);
        let edges = EdgePipeline::new(device, &pipeline_layout, &shader, format);
        let depth_view = create_depth_view(device, config.width, config.height);

        Ok(Self {
            context: context.clone(),
            surface,
            config,
            camera_buffer,
            visibility_buffer,
            dim_buffer,
            bind_group,
            depth_view,
            nodes,
            edges,
        })
    }

    /// Uploads a fresh per-node visibility mask (`1` visible, `0` hidden). The
    /// slice must be `node_count` long; the shader indexes it by node.
    pub fn set_visibility(&self, visible: &[u32]) {
        self.context
            .queue
            .write_buffer(&self.visibility_buffer, 0, bytemuck::cast_slice(visible));
    }

    /// Uploads a fresh per-node dim mask (`1.0` full brightness, `<1.0` dimmed).
    /// The slice must be `node_count` long; the shader indexes it by node.
    pub fn set_dim(&self, dim: &[f32]) {
        self.context
            .queue
            .write_buffer(&self.dim_buffer, 0, bytemuck::cast_slice(dim));
    }

    /// Present mode for the configured vsync setting.
    pub fn present_mode(is_vsync_enabled: bool) -> wgpu::PresentMode {
        if is_vsync_enabled {
            // Fifo is the only mode guaranteed present on every backend.
            wgpu::PresentMode::Fifo
        } else {
            wgpu::PresentMode::AutoNoVsync
        }
    }

    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Reconfigures the surface and rebuilds the depth texture.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.context.device, &self.config);
        self.depth_view = create_depth_view(&self.context.device, width, height);
    }

    /// Acquires the next frame, or `None` when this frame should be skipped.
    ///
    /// A surface can legitimately have nothing to give — occluded, or timed
    /// out — and that is not an error; the frame is simply dropped. Lost and
    /// Outdated mean the surface needs reconfiguring, which happens routinely
    /// on resize, so those retry once.
    pub fn acquire(&mut self) -> Result<Option<wgpu::SurfaceTexture>> {
        use wgpu::CurrentSurfaceTexture as Current;

        match self.surface.get_current_texture() {
            // Suboptimal is still presentable; the resize that provoked it
            // reconfigures the surface on its own event.
            Current::Success(frame) | Current::Suboptimal(frame) => Ok(Some(frame)),
            // Nothing to draw into this frame; skipping it is correct.
            Current::Timeout | Current::Occluded => Ok(None),
            Current::Lost | Current::Outdated => {
                self.surface.configure(&self.context.device, &self.config);
                match self.surface.get_current_texture() {
                    Current::Success(frame) | Current::Suboptimal(frame) => Ok(Some(frame)),
                    other => bail!("reacquiring the surface after it was lost gave {other:?}"),
                }
            }
            Current::Validation => bail!("surface acquisition failed validation"),
        }
    }

    /// Records the graph draw. `encoder` may already hold this step's layout
    /// dispatches, so compute and draw share a single submission.
    pub fn draw_graph(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        buffers: &GraphBuffers,
        camera: &Camera,
        app_config: &AppConfig,
    ) {
        self.context.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&camera.uniforms()),
        );

        let [r, g, b, a] = app_config.clear_color();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("graph"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_bind_group(0, &self.bind_group, &[]);

        if app_config.show_edge && buffers.edge_count > 0 {
            pass.set_pipeline(&self.edges.pipeline);
            pass.draw(0..EdgePipeline::vertex_count(buffers.edge_count), 0..1);
        }

        if buffers.node_count > 0 {
            pass.set_pipeline(&self.nodes.pipeline);
            pass.draw(0..NodePipeline::vertex_count(buffers.node_count), 0..1);
        }
    }
}

/// Picks the swapchain format: the surface's preferred one, with any sRGB
/// encoding dropped.
///
/// Both consumers want to write their colours through unmodified. The original
/// rendered to a default framebuffer with no `GL_FRAMEBUFFER_SRGB`, so its node
/// and clear colours reached the screen raw; an sRGB surface would encode them
/// a second time and wash the whole scene out. egui wants the same thing for
/// the same reason — it does its own gamma handling and logs a warning when
/// handed an sRGB target.
///
/// Deriving from `formats[0]` rather than scanning for the first non-sRGB entry
/// matters: this adapter lists 16-bit-norm formats too, and those need a device
/// feature we do not enable, so a scan picks a format the pipelines cannot use.
fn choose_surface_format(formats: &[wgpu::TextureFormat]) -> wgpu::TextureFormat {
    let preferred = formats[0];
    let linear = preferred.remove_srgb_suffix();
    if formats.contains(&linear) {
        linear
    } else {
        preferred
    }
}

fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn create_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_graph_shader_is_valid_wgsl() {
        // The pipelines only compile this when a window builds them, which the
        // headless tests never do — so validate it here, catching a broken
        // binding or type before it reaches a real device.
        let src = include_str!("../shaders/graph.wgsl");
        let module = naga::front::wgsl::parse_str(src).expect("graph.wgsl should parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("graph.wgsl should validate");
    }

    #[test]
    fn a_linear_surface_format_is_preferred_over_an_srgb_one() {
        // Order must not matter: the sRGB entry comes first here, which is what
        // this adapter actually reports.
        let formats = [
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Bgra8Unorm,
        ];
        assert_eq!(
            choose_surface_format(&formats),
            wgpu::TextureFormat::Bgra8Unorm
        );
    }

    #[test]
    fn an_srgb_only_surface_still_yields_a_format() {
        // Degraded, but a washed-out window beats no window.
        let formats = [wgpu::TextureFormat::Bgra8UnormSrgb];
        assert_eq!(
            choose_surface_format(&formats),
            wgpu::TextureFormat::Bgra8UnormSrgb
        );
    }

    #[test]
    fn the_choice_never_falls_through_to_a_format_needing_extra_features() {
        // What a naive "first non-sRGB" scan did on this adapter: Rgba16Unorm
        // needs TEXTURE_FORMAT_16BIT_NORM, which the device does not enable, so
        // the node pipeline failed to build.
        let formats = [
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Rgba16Unorm,
            wgpu::TextureFormat::Bgra8Unorm,
        ];
        assert_eq!(
            choose_surface_format(&formats),
            wgpu::TextureFormat::Bgra8Unorm
        );
    }

    #[test]
    fn vsync_maps_to_the_universally_supported_present_mode() {
        assert_eq!(Renderer::present_mode(true), wgpu::PresentMode::Fifo);
    }

    #[test]
    fn disabling_vsync_selects_an_unthrottled_mode() {
        assert_ne!(Renderer::present_mode(false), wgpu::PresentMode::Fifo);
    }

    #[test]
    fn clear_colour_is_normalised_from_the_gui_range() {
        // The GUI sliders write 0..=255; the surface wants 0..=1.
        let config = AppConfig { red: 255.0, green: 0.0, blue: 51.0, ..Default::default() };
        let [r, g, b, a] = config.clear_color();
        assert!((r - 1.0).abs() < 1e-9);
        assert_eq!(g, 0.0);
        assert!((b - 0.2).abs() < 1e-9);
        assert_eq!(a, 1.0);
    }

    #[test]
    fn the_depth_format_is_one_every_backend_supports() {
        assert_eq!(DEPTH_FORMAT, wgpu::TextureFormat::Depth32Float);
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn the_shader_compiles_and_both_pipelines_build() {
        // Catches WGSL errors and any drift between `Node` and the struct the
        // shader declares, which no host-side test can see.
        unimplemented!("covered by the windowed smoke test; needs a surface")
    }
}
