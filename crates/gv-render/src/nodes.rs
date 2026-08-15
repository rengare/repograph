//! Node pipeline: one camera-facing quad per node, masked to a disc.
//!
//! wgpu has no equivalent of `GL_PROGRAM_POINT_SIZE` and no `gl_PointSize`, so
//! the original's `GL_POINTS` draw does not port directly. Each node becomes a
//! quad — six vertices generated in the vertex shader from
//! `@builtin(vertex_index)`, sized in clip space the way `circle.vert` sized
//! its point (`size * 500 / -viewPosition.z`) — with the fragment shader
//! discarding outside the unit disc, as `circle.frag` did against
//! `gl_PointCoord`.

use crate::renderer::DEPTH_FORMAT;

pub struct NodePipeline {
    pub(crate) pipeline: wgpu::RenderPipeline,
}

impl NodePipeline {
    /// Vertices per node: two triangles.
    ///
    /// A triangle strip would be four, but a single strip draw would join
    /// consecutive nodes into one connected ribbon.
    pub const VERTICES_PER_NODE: u32 = 6;

    pub fn new(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        format: wgpu::TextureFormat,
    ) -> Self {
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("nodes"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_nodes"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_nodes"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                // The quad winding flips when a sprite is mirrored, so culling
                // would drop half the nodes.
                cull_mode: None,
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

    /// Vertices to draw for `node_count` nodes.
    pub fn vertex_count(node_count: u32) -> u32 {
        node_count * Self::VERTICES_PER_NODE
    }

    /// Screen-space diameter of a node, matching `circle.vert`'s
    /// `gl_PointSize = size * (500 / -modelViewPosition.z)`.
    ///
    /// `view_z` is the node's z in view space, which is negative in front of a
    /// right-handed camera. Behind or on the camera plane the sprite is
    /// degenerate, so this returns 0 and the shader culls it.
    pub fn point_diameter(size: f32, view_z: f32) -> f32 {
        if view_z >= 0.0 {
            return 0.0;
        }
        size * (500.0 / -view_z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_node_becomes_two_triangles() {
        assert_eq!(NodePipeline::vertex_count(0), 0);
        assert_eq!(NodePipeline::vertex_count(3), 18);
    }

    #[test]
    fn point_size_matches_the_original_perspective_scale() {
        // size 10 at view z = -500 gave gl_PointSize = 10 exactly.
        assert_eq!(NodePipeline::point_diameter(10.0, -500.0), 10.0);
    }

    #[test]
    fn point_size_grows_as_a_node_approaches_the_camera() {
        let near = NodePipeline::point_diameter(10.0, -100.0);
        let far = NodePipeline::point_diameter(10.0, -1000.0);
        assert!(near > far, "{near} should exceed {far}");
    }

    #[test]
    fn nodes_at_or_behind_the_camera_plane_collapse() {
        // Guards the division: view_z == 0 would otherwise be an infinity that
        // propagates into clip space and blanks the frame.
        assert_eq!(NodePipeline::point_diameter(10.0, 0.0), 0.0);
        assert_eq!(NodePipeline::point_diameter(10.0, 25.0), 0.0);
    }
}
