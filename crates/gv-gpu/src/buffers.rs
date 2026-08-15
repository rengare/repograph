//! The GPU-side graph: node state, edges and adjacency.

use anyhow::{Result, bail};
use gv_graph::{GraphData, Node};
use wgpu::util::DeviceExt;

use crate::GpuContext;

/// Every buffer the layout passes and the draw calls share.
///
/// Replaces the original's three SSBOs at bindings 0, 8 and 11. The third of
/// those — `edgeSsbo`, two duplicated `GraphicData` entries per edge, filled
/// each step by a dedicated `lines` compute pass — is gone: the edge vertex
/// shader indexes `nodes` through `edges` using `@builtin(vertex_index)`
/// instead, which removes both the pass and `2 * edge_count * 48` bytes.
#[derive(Debug)]
pub struct GraphBuffers {
    /// `Node[]`, usage `STORAGE | VERTEX | COPY_SRC`. Mutated in place by the
    /// layout passes and drawn from directly.
    pub nodes: wgpu::Buffer,
    /// `Edge[]`, usage `STORAGE`. Immutable after upload.
    pub edges: wgpu::Buffer,
    /// CSR row offsets, `u32[node_count + 1]`, usage `STORAGE`.
    pub csr_offsets: wgpu::Buffer,
    /// CSR neighbour ids, `u32[2 * edge_count]`, usage `STORAGE`.
    pub csr_neighbors: wgpu::Buffer,

    pub node_count: u32,
    pub edge_count: u32,
}

impl GraphBuffers {
    /// Usages the node buffer must be created with.
    ///
    /// Named rather than inlined so the test below can assert on the property
    /// the whole no-copy design rests on: one allocation serves compute and
    /// drawing, so a layout step is visible to the next frame with no transfer.
    pub const NODE_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
        .union(wgpu::BufferUsages::VERTEX)
        .union(wgpu::BufferUsages::COPY_SRC)
        .union(wgpu::BufferUsages::COPY_DST);

    /// Usages for the immutable edge and CSR buffers.
    pub const READ_ONLY_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE;

    /// Uploads `graph` and allocates the layout scratch it needs.
    pub fn upload(context: &GpuContext, graph: &GraphData) -> Result<Self> {
        let limits = context.limits();
        let max_nodes = GpuContext::max_nodes_for(&limits);
        if graph.node_count() as u64 > max_nodes {
            bail!(
                "graph has {} nodes; this adapter's max_storage_buffer_binding_size \
                 ({} bytes) allows {max_nodes}",
                graph.node_count(),
                limits.max_storage_buffer_binding_size,
            );
        }

        let device = &context.device;

        // wgpu rejects zero-sized buffers, and an edgeless graph is a valid
        // thing to visualise, so empty arrays get one dummy element. The
        // counts below — not the buffer lengths — bound every dispatch and
        // draw, so the padding is never read.
        let nodes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("graph nodes"),
            contents: bytemuck::cast_slice(pad(&graph.nodes, Node::default())),
            usage: Self::NODE_USAGE,
        });

        let edges = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("graph edges"),
            contents: bytemuck::cast_slice(pad(&graph.edges, Default::default())),
            usage: Self::READ_ONLY_USAGE,
        });

        let csr_offsets = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("graph csr offsets"),
            contents: bytemuck::cast_slice(pad(&graph.adjacency.offsets, 0u32)),
            usage: Self::READ_ONLY_USAGE,
        });

        let csr_neighbors = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("graph csr neighbors"),
            contents: bytemuck::cast_slice(pad(&graph.adjacency.neighbors, 0u32)),
            usage: Self::READ_ONLY_USAGE,
        });

        Ok(Self {
            nodes,
            edges,
            csr_offsets,
            csr_neighbors,
            node_count: graph.node_count() as u32,
            edge_count: graph.edge_count() as u32,
        })
    }

    /// Overwrites the node buffer from host memory.
    ///
    /// The cost a CPU layout pays every step and a GPU layout does not.
    pub fn write_nodes(&self, context: &GpuContext, nodes: &[Node]) {
        if nodes.is_empty() {
            return;
        }
        context
            .queue
            .write_buffer(&self.nodes, 0, bytemuck::cast_slice(nodes));
    }

    /// Reads the node array back to host memory.
    ///
    /// Used by the GPU-versus-CPU comparison in phase 3 and by headless runs;
    /// it stalls the pipeline, so it is not on the frame path.
    pub async fn read_nodes(&self, context: &GpuContext) -> Result<Vec<Node>> {
        if self.node_count == 0 {
            return Ok(Vec::new());
        }

        let bytes = u64::from(self.node_count) * size_of::<Node>() as u64;
        let staging = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("node readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("node readback"),
            });
        encoder.copy_buffer_to_buffer(&self.nodes, 0, &staging, 0, bytes);
        context.queue.submit([encoder.finish()]);

        let (sender, receiver) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        context.device.poll(wgpu::PollType::wait_indefinitely())?;
        receiver.recv()??;

        let view = staging.slice(..).get_mapped_range();
        let nodes = bytemuck::cast_slice::<u8, Node>(&view).to_vec();
        drop(view);
        staging.unmap();
        Ok(nodes)
    }

    /// Web: non-blocking node readback. [`read_nodes`](Self::read_nodes) blocks the
    /// device with `poll(wait_indefinitely)`, which the browser forbids; here the
    /// `map_async` callback (carrying no `Send` bound on wasm) delivers the pulled
    /// nodes once the browser resolves the mapping.
    #[cfg(target_arch = "wasm32")]
    pub fn read_nodes_callback(
        &self,
        context: &GpuContext,
        done: impl FnOnce(Vec<Node>) + Send + 'static,
    ) {
        if self.node_count == 0 {
            done(Vec::new());
            return;
        }
        let bytes = u64::from(self.node_count) * size_of::<Node>() as u64;
        let staging = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("node readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("node readback"),
            });
        encoder.copy_buffer_to_buffer(&self.nodes, 0, &staging, 0, bytes);
        context.queue.submit([encoder.finish()]);

        let mapped = staging.clone();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            if result.is_ok() {
                let view = mapped.slice(..).get_mapped_range();
                let nodes = bytemuck::cast_slice::<u8, Node>(&view).to_vec();
                drop(view);
                mapped.unmap();
                done(nodes);
            }
        });
    }

    /// Bytes the buffers will occupy for a graph of this shape.
    pub fn byte_size(node_count: u64, edge_count: u64) -> u64 {
        node_count * size_of::<Node>() as u64
            + edge_count * size_of::<gv_graph::Edge>() as u64
            + (node_count + 1) * size_of::<u32>() as u64
            + 2 * edge_count * size_of::<u32>() as u64
    }
}

/// Borrows `data`, or a one-element slice of `fallback` when it is empty.
fn pad<T: bytemuck::Pod>(data: &[T], fallback: T) -> &[T] {
    if data.is_empty() {
        // Leaked once per empty buffer at startup, never in a loop.
        Box::leak(Box::new([fallback]))
    } else {
        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_node_buffer_is_both_storage_and_vertex() {
        // If either of these is ever dropped, the layout passes and the draw
        // calls stop sharing memory and a per-frame copy sneaks back in.
        assert!(GraphBuffers::NODE_USAGE.contains(wgpu::BufferUsages::STORAGE));
        assert!(GraphBuffers::NODE_USAGE.contains(wgpu::BufferUsages::VERTEX));
    }

    #[test]
    fn read_only_buffers_are_not_writable_from_the_host() {
        assert!(!GraphBuffers::READ_ONLY_USAGE.contains(wgpu::BufferUsages::COPY_DST));
        assert!(!GraphBuffers::READ_ONLY_USAGE.contains(wgpu::BufferUsages::MAP_WRITE));
    }

    #[test]
    fn byte_size_accounts_for_every_buffer() {
        // 2 nodes, 1 edge: 96 nodes + 8 edge + 12 offsets + 8 neighbours.
        assert_eq!(GraphBuffers::byte_size(2, 1), 96 + 8 + 12 + 8);
    }

    #[test]
    fn dropping_the_lines_pass_saves_what_it_claims_to() {
        // The original allocated an extra `2 * edges` Nodes for the duplicated
        // edge vertex array. This records the saving the design note asserts.
        let edges = 100_000u64;
        let saved = 2 * edges * size_of::<Node>() as u64;
        assert_eq!(saved, 9_600_000);
    }

    #[test]
    fn padding_substitutes_a_single_element_for_an_empty_slice() {
        // wgpu rejects a zero-sized buffer, so an edgeless graph would fail to
        // upload without this.
        assert_eq!(pad::<u32>(&[], 0).len(), 1);
        assert_eq!(pad(&[7u32, 8], 0), &[7, 8]);
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn upload_round_trips_node_state() {
        let context = pollster::block_on(GpuContext::new(None)).expect("adapter available");
        let graph = gv_graph::testing::triangle();
        let buffers = GraphBuffers::upload(&context, &graph).unwrap();

        let read_back = pollster::block_on(buffers.read_nodes(&context)).unwrap();
        assert_eq!(read_back, graph.nodes);
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn an_edgeless_graph_uploads() {
        let context = pollster::block_on(GpuContext::new(None)).expect("adapter available");
        let graph = gv_graph::testing::dust(4);
        let buffers = GraphBuffers::upload(&context, &graph).unwrap();
        assert_eq!(buffers.edge_count, 0);
    }
}
