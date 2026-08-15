//! Fruchterman-Reingold on the GPU.
//!
//! # Differences from the original's four GLSL passes
//!
//! The original dispatched `repulsive`, `attractive`, `positionupdate` and
//! `lines`. This has three, and two of the surviving passes are corrected:
//!
//! - **attractive** was one invocation *per edge*, doing an unsynchronised
//!   read-modify-write of `data[from]` and `data[to]`. Every edge incident to
//!   a node raced with every other, so contributions were silently lost and no
//!   two runs agreed. Here it is one invocation *per node*, gathering over that
//!   node's CSR neighbour list, so each invocation writes only its own node.
//! - **repulsive** had `if (globalIndex > graphDataSize) return;` — one
//!   invocation past the end read out of bounds on every dispatch. It also
//!   looped `j` from `globalIndex + 1`, computing each pair once but applying
//!   the force to only one of the two nodes, which is not symmetric. Both are
//!   fixed: `>=` for the bound, and a full `0..n` loop.
//! - **lines** is gone entirely; see [`gv_gpu::GraphBuffers`].
//!
//! Uniforms are a single bound struct rather than the original's six
//! `glUniform1*v` calls at hardcoded locations — and they are written *before*
//! the dispatch, which `FRModel::UpdateNodes` failed to do for the repulsive
//! pass.

use anyhow::Result;
use gv_gpu::{GpuContext, GraphBuffers};
use gv_layout::LayoutParams;

use crate::GpuLayout;

/// Workgroup size, matching the original's `local_size_x = 256`.
pub const WORKGROUP_SIZE: u32 = 256;

/// Uniform block shared by all three passes. `repr(C)` with explicit padding
/// to a 16-byte multiple, as WGSL uniform layout requires.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FrUniforms {
    pub node_count: u32,
    pub edge_count: u32,
    pub three_d: u32,
    pub _pad: u32,

    pub k: f32,
    pub speed_scale: f32,
    pub gravity: f32,
    pub max_displace: f32,
}

impl FrUniforms {
    pub fn new(params: &LayoutParams, node_count: u32, edge_count: u32) -> Self {
        Self {
            node_count,
            edge_count,
            three_d: params.three_d as u32,
            _pad: 0,
            k: params.k(node_count as usize),
            speed_scale: params.speed_scale(),
            gravity: params.gravity,
            max_displace: params.max_displace(),
        }
    }
}

pub struct FrGpuLayout {
    /// Holds strong references to every buffer it binds, so the layout does not
    /// need to own the `GraphBuffers` the renderer is also drawing from.
    bind_group: wgpu::BindGroup,
    uniforms: wgpu::Buffer,
    repulsive: wgpu::ComputePipeline,
    attractive: wgpu::ComputePipeline,
    position_update: wgpu::ComputePipeline,
    node_count: u32,
    edge_count: u32,
}

impl FrGpuLayout {
    pub fn new(context: &GpuContext, buffers: &GraphBuffers) -> Result<Self> {
        let device = &context.device;

        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fr uniforms"),
            size: size_of::<FrUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fr"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // The node array is the only writable binding: the layout
                // mutates positions and displacements in place, and the draw
                // pipelines read the same allocation with no copy between them.
                storage_entry(1, false),
                storage_entry(2, true),
                storage_entry(3, true),
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fr"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.nodes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.csr_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.csr_neighbors.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fr"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fr"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fr.wgsl").into()),
        });

        let pipeline = |entry_point: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        Ok(Self {
            repulsive: pipeline("repulsive"),
            attractive: pipeline("attractive"),
            position_update: pipeline("position_update"),
            bind_group,
            uniforms,
            node_count: buffers.node_count,
            edge_count: buffers.edge_count,
        })
    }

    /// Uploads this step's derived constants.
    ///
    /// Written *before* the dispatch, which `FRModel::UpdateNodes` failed to do
    /// for the repulsive pass — it uploaded that pass's uniforms after
    /// dispatching it, so the first step ran on whatever was there before.
    pub fn write_uniforms(&self, queue: &wgpu::Queue, params: &LayoutParams) {
        queue.write_buffer(
            &self.uniforms,
            0,
            bytemuck::bytes_of(&FrUniforms::new(params, self.node_count, self.edge_count)),
        );
    }

    /// Records attraction and the position update — everything after the
    /// repulsion term.
    ///
    /// Split out because that term is the only thing Barnes-Hut replaces:
    /// [`crate::bh_gpu::BhGpuLayout`] records its octree walk in place of the
    /// repulsive pass and then calls this, so the two paths cannot drift in how
    /// a step is closed out. The equivalent on the CPU side is
    /// `gv_layout::integrate`.
    pub fn record_after_repulsion(&self, encoder: &mut wgpu::CommandEncoder) {
        for (label, pipeline) in [
            ("fr attractive", &self.attractive),
            ("fr position update", &self.position_update),
        ] {
            self.record_pass(encoder, label, pipeline);
        }
    }

    /// One dispatch in a compute pass of its own. See [`Self::record_step`] for
    /// why the passes are never batched.
    fn record_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        pipeline: &wgpu::ComputePipeline,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count(self.node_count), 1, 1);
    }
}

impl GpuLayout for FrGpuLayout {
    fn name(&self) -> &'static str {
        "F-R gpu"
    }

    fn record_step(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        params: &LayoutParams,
    ) -> Result<()> {
        if self.node_count == 0 {
            return Ok(());
        }

        self.write_uniforms(queue, params);

        // One compute pass *per* dispatch, not one pass with three dispatches.
        //
        // The three passes are strictly dependent — repulsion seeds the
        // displacement, attraction accumulates into it, and only then may
        // positions move — so every invocation of a pass must see all of the
        // previous pass's writes. Batching them into a single pass does not
        // give that: the node buffer stays in the same read-write storage usage
        // throughout, so wgpu's resource tracker sees no state transition and
        // emits no barrier between the dispatches.
        //
        // The failure is invisible below 257 nodes, where one workgroup runs
        // the whole graph and the ordering is incidental. Above that it is
        // severe: position-update invocations read displacements that
        // attraction had not written yet, so the layout ran on repulsion alone
        // and converged to a graph roughly five times too spread out. Splitting
        // the passes is what makes wgpu insert the barrier.
        self.record_pass(encoder, "fr repulsive", &self.repulsive);
        self.record_after_repulsion(encoder);

        Ok(())
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Number of workgroups needed to cover `count` invocations.
///
/// The original wrote `(n / 256) + 1`, which dispatches a wholly empty extra
/// group whenever `n` is a multiple of 256, and paired it with a `>` bound
/// check that let one invocation past the end read out of bounds. This is the
/// ceiling division, to be paired with a `>=` check in the shader.
pub fn workgroup_count(count: u32) -> u32 {
    count.div_ceil(WORKGROUP_SIZE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniforms_satisfy_wgsl_uniform_alignment() {
        assert_eq!(size_of::<FrUniforms>(), 32);
        assert_eq!(size_of::<FrUniforms>() % 16, 0);
    }

    #[test]
    fn workgroup_count_covers_every_invocation() {
        for count in [0, 1, 255, 256, 257, 512, 100_000] {
            let groups = workgroup_count(count);
            assert!(
                groups * WORKGROUP_SIZE >= count,
                "{count} invocations not covered by {groups} groups"
            );
        }
    }

    #[test]
    fn workgroup_count_does_not_over_dispatch_on_exact_multiples() {
        // The original's `(n / 256) + 1` gave 2 here, wasting a whole group.
        assert_eq!(workgroup_count(256), 1);
        assert_eq!(workgroup_count(512), 2);
        assert_eq!(workgroup_count(257), 2);
    }

    #[test]
    fn an_empty_graph_dispatches_nothing() {
        assert_eq!(workgroup_count(0), 0);
    }

    #[test]
    fn uniforms_carry_the_derived_constants_not_the_raw_knobs() {
        // k, speed_scale and max_displace are computed once on the host rather
        // than recomputed by every invocation, as the original's GLSL did.
        let params = LayoutParams { area: 1000.0, speed: 100.0, gravity: 2.0, three_d: true };
        let uniforms = FrUniforms::new(&params, 999, 4);

        assert_eq!(uniforms.k, params.k(999));
        assert_eq!(uniforms.speed_scale, params.speed_scale());
        assert_eq!(uniforms.max_displace, params.max_displace());
        assert_eq!(uniforms.gravity, 2.0);
        assert_eq!(uniforms.three_d, 1);
        assert_eq!(uniforms.node_count, 999);
        assert_eq!(uniforms.edge_count, 4);
    }

    #[test]
    fn two_d_mode_is_encoded_as_zero() {
        let params = LayoutParams { three_d: false, ..Default::default() };
        assert_eq!(FrUniforms::new(&params, 8, 4).three_d, 0);
    }

    #[test]
    fn uniforms_are_zeroable_so_padding_never_leaks_stack_bytes() {
        let zeroed: FrUniforms = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed._pad, 0);
        assert_eq!(bytemuck::bytes_of(&zeroed).len(), 32);
    }

    /// Relative tolerance for a single step of CPU-versus-GPU comparison.
    ///
    /// Measured agreement on this machine's Adreno X1-85 is actually at f32
    /// epsilon — the two paths sum in the same order, so the only divergence
    /// available is fused multiply-add contraction and how `length` is lowered.
    /// This is deliberately looser than the measurement so a different driver
    /// making different contraction choices does not fail the suite, and it is
    /// still four orders of magnitude tighter than any real defect: a dropped
    /// neighbour or a sign error moves a node by a fraction of `k`, not by a
    /// fraction of a ulp.
    const TOLERANCE: f32 = 1e-5;

    /// Runs `steps` GPU steps over `graph` and reads the node array back.
    fn run_on_gpu(
        graph: &gv_graph::GraphData,
        params: &LayoutParams,
        steps: usize,
    ) -> Vec<gv_graph::Node> {
        let context = pollster::block_on(GpuContext::new(None)).expect("adapter available");
        let buffers = GraphBuffers::upload(&context, graph).expect("upload");
        let mut layout = FrGpuLayout::new(&context, &buffers).expect("pipelines build");

        for _ in 0..steps {
            let mut encoder = context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            layout
                .record_step(&mut encoder, &context.queue, params)
                .expect("record");
            context.queue.submit([encoder.finish()]);
        }

        pollster::block_on(buffers.read_nodes(&context)).expect("readback")
    }

    fn run_on_cpu(
        graph: &gv_graph::GraphData,
        params: &LayoutParams,
        steps: usize,
    ) -> Vec<gv_graph::Node> {
        use gv_layout::CpuLayout;

        let mut graph = graph.clone();
        let mut layout = gv_layout::fr_cpu::FrCpuLayout;
        for _ in 0..steps {
            layout.step(&mut graph, params);
        }
        graph.nodes
    }

    /// A graph large enough to need more than one workgroup, with degrees that
    /// vary — the shape that exposes a missing barrier between passes.
    fn multi_workgroup_graph(node_count: u32) -> gv_graph::GraphData {
        let mut state = 12345u64;
        let mut next = move |n: u32| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as u32) % n
        };

        // A spanning tree so nothing is isolated, then extra edges so some rows
        // are much longer than others.
        let mut edges: Vec<gv_graph::Edge> = (1..node_count)
            .map(|i| gv_graph::Edge { from: i, to: next(i) })
            .collect();
        for _ in 0..node_count {
            edges.push(gv_graph::Edge { from: next(node_count), to: next(node_count) });
        }

        gv_graph::testing::from_edges(node_count as usize, edges, 0)
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_step_matches_the_cpu_reference() {
        // The milestone test. Same seed, same params, one step each; the two
        // paths must agree to float tolerance. Until this passes, the GPU
        // layout is not known to compute Fruchterman-Reingold at all — which
        // is precisely the assurance the original never had, since its
        // attractive pass raced.
        //
        // One step, not many: the layout is chaotic, so a divergence of one ulp
        // is amplified without bound over time. What is being checked is that
        // the two implement the same arithmetic, and a single step is where
        // that is visible without the comparison becoming meaningless.
        let params = LayoutParams::default();
        let graph = gv_graph::testing::path(64);

        let actual = run_on_gpu(&graph, &params, 1);
        let expected = run_on_cpu(&graph, &params, 1);

        assert_positions_close(&actual, &expected, TOLERANCE);
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn the_passes_are_ordered_across_more_than_one_workgroup() {
        // The regression for the barrier bug, and the reason the fixtures above
        // are not sufficient on their own: at 64 nodes a single workgroup runs
        // the whole graph, so the passes are ordered incidentally and a missing
        // barrier is invisible. Past 256 they are not.
        //
        // Unfixed, this failed at step one with a relative error of 17 — the
        // position update was reading displacements attraction had not written.
        let params = LayoutParams::default();
        let graph = multi_workgroup_graph(1024);
        assert!(
            workgroup_count(graph.node_count() as u32) > 1,
            "fixture must span more than one workgroup or it proves nothing"
        );

        let actual = run_on_gpu(&graph, &params, 1);
        let expected = run_on_cpu(&graph, &params, 1);

        // Looser than TOLERANCE: the repulsion sum runs over 1024 terms rather
        // than 64, so more rounding accumulates in it.
        assert_positions_close(&actual, &expected, 1e-4);
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn both_paths_converge_to_the_same_layout() {
        // Individual positions diverge chaotically — by step 20 on this graph
        // they disagree outright — so over a long run the only meaningful
        // comparison is an aggregate one. Mean edge length is the headline
        // quality number, and it is what caught the barrier bug: the two
        // converged to stable but different equilibria, 1035 against 5204,
        // while every single-step assertion still passed.
        let params = LayoutParams::default();
        let graph = multi_workgroup_graph(1024);

        let gpu = mean_edge_length(&graph, &run_on_gpu(&graph, &params, 400));
        let cpu = mean_edge_length(&graph, &run_on_cpu(&graph, &params, 400));

        let difference = (gpu - cpu).abs() / cpu;
        assert!(
            difference < 0.02,
            "converged layouts disagree: gpu mean edge {gpu}, cpu {cpu} ({:.1}% apart)",
            difference * 100.0
        );
    }

    fn mean_edge_length(graph: &gv_graph::GraphData, nodes: &[gv_graph::Node]) -> f32 {
        let total: f64 = graph
            .edges
            .iter()
            .map(|edge| {
                let a = nodes[edge.from as usize].position;
                let b = nodes[edge.to as usize].position;
                let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
                f64::from(dx * dx + dy * dy + dz * dz).sqrt()
            })
            .sum();
        (total / graph.edges.len() as f64) as f32
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_and_cpu_agree_in_three_d_and_on_a_denser_graph() {
        // 2D zeroes z on every step, which would hide a broken z lane. The
        // triangle also exercises a node whose CSR row has more than one entry.
        let params = LayoutParams { three_d: true, ..Default::default() };
        let graph = gv_graph::testing::triangle();

        let actual = run_on_gpu(&graph, &params, 1);
        let expected = run_on_cpu(&graph, &params, 1);

        assert!(
            actual.iter().any(|node| node.position[2] != 0.0),
            "3D run left every node in the plane; the z lane is not being exercised"
        );
        assert_positions_close(&actual, &expected, TOLERANCE);
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn every_neighbour_contributes_on_the_gpu_too() {
        // The regression for the original's per-edge race. A hub with four
        // neighbours all at positive x must be pulled toward them; if CSR rows
        // were being dropped or overwritten the hub would barely move.
        //
        // `area` puts k (= 5) well inside the 100..400 spacing, since attraction
        // only outweighs repulsion beyond d = k.
        let params = LayoutParams { speed: 100.0, area: 0.06, gravity: 0.0, three_d: false };
        let mut star = gv_graph::testing::from_edges(
            5,
            (1..5).map(|i| gv_graph::Edge { from: 0, to: i }).collect(),
            0,
        );
        for (index, node) in star.nodes.iter_mut().enumerate() {
            node.position = [100.0 * index as f32, 0.0, 0.0, 1.0];
        }

        let actual = run_on_gpu(&star, &params, 1);
        assert!(
            actual[0].position[0] > star.nodes[0].position[0],
            "hub moved to {} from {}",
            actual[0].position[0],
            star.nodes[0].position[0]
        );
        assert_positions_close(&actual, &run_on_cpu(&star, &params, 1), TOLERANCE);
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn repeated_gpu_steps_are_deterministic() {
        // Directly regresses the original's race: two identical runs must
        // produce byte-identical node buffers. Many steps here, not one —
        // a race needs time and contention to show itself, and byte equality
        // is the one comparison that stays meaningful as error compounds.
        let params = LayoutParams::default();
        let graph = gv_graph::testing::path(256);

        let first = run_on_gpu(&graph, &params, 20);
        let second = run_on_gpu(&graph, &params, 20);

        assert_eq!(first, second, "two identical GPU runs disagreed");
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn an_edgeless_graph_steps_without_dispatching_into_nothing() {
        // Repulsion only, and every CSR row empty — the shape most likely to
        // index past the end of a padded buffer.
        let params = LayoutParams::default();
        let graph = gv_graph::testing::dust(8);

        let actual = run_on_gpu(&graph, &params, 5);
        assert_positions_close(&actual, &run_on_cpu(&graph, &params, 5), 1e-3);
    }

    /// Compares positions with a relative tolerance.
    ///
    /// Absolute equality is not available and not the point: the two paths
    /// differ in fused multiply-add contraction and in how `length` is lowered,
    /// and `k` is on the order of 10⁵, so the absolute error scales with the
    /// magnitudes involved.
    fn assert_positions_close(
        actual: &[gv_graph::Node],
        expected: &[gv_graph::Node],
        tolerance: f32,
    ) {
        assert_eq!(actual.len(), expected.len(), "node count changed");

        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            for axis in 0..3 {
                let (a, e) = (actual.position[axis], expected.position[axis]);
                assert!(a.is_finite(), "node {index} axis {axis} is not finite: {a}");

                let error = (a - e).abs() / e.abs().max(1.0);
                assert!(
                    error <= tolerance,
                    "node {index} axis {axis}: gpu {a} vs cpu {e} (relative error {error})"
                );
            }
        }
    }
}
