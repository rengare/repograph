//! Edge pipeline: a `LineList` draw of `2 * edge_count` vertices.
//!
//! No vertex buffer and no per-step copy. The vertex shader derives the edge
//! from `@builtin(vertex_index) / 2` and the endpoint from `index % 2`, reads
//! the id out of the edge storage buffer, and reads the position out of the
//! node storage buffer. This is what makes the original's `lines` compute pass
//! and its duplicate `edgeSsbo` unnecessary.

use crate::renderer::DEPTH_FORMAT;

pub struct EdgePipeline {
    pub(crate) pipeline: wgpu::RenderPipeline,
}

impl EdgePipeline {
    pub fn new(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        format: wgpu::TextureFormat,
    ) -> Self {
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("edges"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_edges"),
                compilation_options: Default::default(),
                // The property this design turns on: nothing is bound here.
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_edges"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        Self { pipeline }
    }

    /// Vertices to draw for `edge_count` edges — two per edge.
    pub fn vertex_count(edge_count: u32) -> u32 {
        edge_count * 2
    }

    /// The `(edge, endpoint)` a given vertex index resolves to.
    ///
    /// This is the arithmetic the vertex shader performs, lifted out so it can
    /// be checked on the host. Endpoint 0 is `from`, 1 is `to`.
    pub fn resolve_vertex(vertex_index: u32) -> (u32, u32) {
        (vertex_index / 2, vertex_index % 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_edge_becomes_two_vertices() {
        assert_eq!(EdgePipeline::vertex_count(0), 0);
        assert_eq!(EdgePipeline::vertex_count(3), 6);
    }

    #[test]
    fn vertices_resolve_to_alternating_endpoints() {
        assert_eq!(EdgePipeline::resolve_vertex(0), (0, 0));
        assert_eq!(EdgePipeline::resolve_vertex(1), (0, 1));
        assert_eq!(EdgePipeline::resolve_vertex(2), (1, 0));
        assert_eq!(EdgePipeline::resolve_vertex(7), (3, 1));
    }

    #[test]
    fn every_vertex_of_every_edge_is_visited_exactly_once() {
        let edge_count = 64;
        let mut seen = vec![[false; 2]; edge_count as usize];
        for vertex in 0..EdgePipeline::vertex_count(edge_count) {
            let (edge, endpoint) = EdgePipeline::resolve_vertex(vertex);
            assert!(!seen[edge as usize][endpoint as usize], "vertex {vertex} repeated");
            seen[edge as usize][endpoint as usize] = true;
        }
        assert!(seen.iter().all(|pair| pair == &[true, true]));
    }
}
