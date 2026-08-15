//! Graph storage and edge-list loading.
//!
//! This crate is deliberately free of `wgpu`, `winit` and `egui`: it is the
//! one place where the CPU and GPU layout paths agree on what a graph *is*,
//! and keeping it dependency-light means the layout algorithms can be unit
//! tested without a display or an adapter.
//!
//! [`Node`] is laid out to match the `GraphicData` struct the original's GLSL
//! compute shaders declared, so the same buffer serves as a storage buffer and
//! a vertex buffer with no repacking.

pub mod loader;
pub mod seed;

#[cfg(any(test, feature = "testing"))]
pub mod testing;

use bytemuck::{Pod, Zeroable};

/// One graph node, 48 bytes, 16-byte aligned.
///
/// Field order and padding mirror the original `VertexData`/`GraphicData`:
/// `vec4 position; vec4 color; float size; float dx, dy, dz;`
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Node {
    /// `xyz` is the position; `w` is padding to satisfy `vec4` alignment.
    pub position: [f32; 4],
    pub color: [f32; 4],
    /// Point size in pixels before perspective scaling.
    pub size: f32,
    /// Displacement accumulated during the current layout step.
    pub disp: [f32; 3],
}

/// A directed edge over dense node indices.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct Edge {
    pub from: u32,
    pub to: u32,
}

/// Undirected adjacency in compressed sparse row form.
///
/// The attractive-force pass gathers per node rather than scattering per edge —
/// that is what removes the read-modify-write race the original's
/// `fruchtermanreingold_attractive.comp` had — and a gather needs each node's
/// neighbour list contiguously. Both arrays upload to the GPU as-is.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Csr {
    /// Length `node_count + 1`; node `i` owns `neighbors[offsets[i]..offsets[i + 1]]`.
    pub offsets: Vec<u32>,
    /// Length `2 * edge_count` — every edge appears from both endpoints.
    pub neighbors: Vec<u32>,
}

impl Csr {
    /// Builds the adjacency for `node_count` nodes from a directed edge list,
    /// treating every edge as undirected.
    pub fn build(node_count: usize, edges: &[Edge]) -> Self {
        let mut degrees = vec![0u32; node_count];
        for edge in edges {
            degrees[edge.from as usize] += 1;
            degrees[edge.to as usize] += 1;
        }

        let mut offsets = Vec::with_capacity(node_count + 1);
        let mut running = 0u32;
        offsets.push(0);
        for degree in &degrees {
            running += degree;
            offsets.push(running);
        }

        let mut cursor = offsets.clone();
        let mut neighbors = vec![0u32; running as usize];
        for edge in edges {
            neighbors[cursor[edge.from as usize] as usize] = edge.to;
            cursor[edge.from as usize] += 1;
            neighbors[cursor[edge.to as usize] as usize] = edge.from;
            cursor[edge.to as usize] += 1;
        }

        Self { offsets, neighbors }
    }

    /// Neighbours of `node`.
    pub fn neighbors_of(&self, node: usize) -> &[u32] {
        let start = self.offsets[node] as usize;
        let end = self.offsets[node + 1] as usize;
        &self.neighbors[start..end]
    }
}

/// What a node represents, when loaded from a knowledge-graph sidecar. Mirrors
/// `rkg_core::NodeKind` but is duplicated here to keep `gv-graph` a leaf crate the
/// knowledge-graph crates depend on, not the other way around.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NodeCategory {
    Dir,
    File,
    Doc,
    Section,
    Symbol,
    #[default]
    Unknown,
}

impl NodeCategory {
    /// Parses the lowercase tag written in `nodes.tsv`.
    pub fn from_tag(tag: &str) -> Self {
        match tag {
            "dir" => NodeCategory::Dir,
            "file" => NodeCategory::File,
            "doc" => NodeCategory::Doc,
            "sec" => NodeCategory::Section,
            "sym" => NodeCategory::Symbol,
            _ => NodeCategory::Unknown,
        }
    }

    /// Short lowercase label for display in the search list.
    pub fn tag_short(self) -> &'static str {
        match self {
            NodeCategory::Dir => "dir",
            NodeCategory::File => "file",
            NodeCategory::Doc => "doc",
            NodeCategory::Section => "sec",
            NodeCategory::Symbol => "sym",
            NodeCategory::Unknown => "?",
        }
    }

    /// A fixed, legible RGBA colour per category, used when the loaded graph
    /// carries kinds instead of anonymous ids.
    pub fn color(self) -> [f32; 4] {
        match self {
            NodeCategory::Dir => [0.55, 0.55, 0.60, 1.0], // grey
            NodeCategory::File => [0.30, 0.65, 1.00, 1.0], // blue
            NodeCategory::Doc => [0.35, 0.80, 0.45, 1.0],  // green
            NodeCategory::Section => [0.55, 0.85, 0.55, 1.0], // pale green
            NodeCategory::Symbol => [1.00, 0.70, 0.25, 1.0], // amber
            NodeCategory::Unknown => [0.80, 0.80, 0.80, 1.0], // light grey
        }
    }
}

/// Per-node metadata from a knowledge-graph sidecar (`nodes.tsv`). Empty on a
/// plain edge-list load; populated by [`loader::load_with_sidecar`], which is what
/// the searchable browser reads.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NodeMeta {
    pub id: String,
    pub name: String,
    pub kind: NodeCategory,
    pub path: String,
    /// 1-based inclusive line span, when the node is a symbol/section.
    pub span: Option<(u32, u32)>,
    /// The symbol's full signature, when known.
    pub signature: Option<String>,
    /// Symbol sub-kind (`fn`, `struct`, `method`, …).
    pub symbol_kind: Option<String>,
    /// Enclosing scope (`impl Csr`, a class/module name).
    pub container: Option<String>,
    /// Doc comment / summary.
    pub doc: Option<String>,
}

/// A loaded graph: node state, edges, adjacency, and the original string ids.
#[derive(Clone, Debug, Default)]
pub struct GraphData {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub adjacency: Csr,
    /// `labels[i]` is the id that node `i` had in the input file. Kept for
    /// display and for reproducing a run against the source data.
    pub labels: Vec<String>,
    /// `meta[i]` is node `i`'s knowledge-graph metadata, or empty if the graph
    /// was loaded from a plain edge list without a sidecar.
    pub meta: Vec<NodeMeta>,
}

impl GraphData {
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_matches_the_glsl_struct_layout() {
        // `GraphicData` in the original compute shaders, under std430.
        assert_eq!(size_of::<Node>(), 48);
        assert_eq!(align_of::<Node>(), 4);
        assert_eq!(size_of::<Edge>(), 8);
    }

    #[test]
    fn csr_is_undirected_and_contiguous() {
        //  0 -- 1 -- 2
        //   \________/
        let edges = [
            Edge { from: 0, to: 1 },
            Edge { from: 1, to: 2 },
            Edge { from: 0, to: 2 },
        ];
        let csr = Csr::build(3, &edges);

        assert_eq!(csr.offsets, vec![0, 2, 4, 6]);
        assert_eq!(csr.neighbors.len(), 2 * edges.len());

        let mut zero = csr.neighbors_of(0).to_vec();
        zero.sort_unstable();
        assert_eq!(zero, vec![1, 2]);

        let mut one = csr.neighbors_of(1).to_vec();
        one.sort_unstable();
        assert_eq!(one, vec![0, 2]);
    }

    #[test]
    fn csr_handles_isolated_and_self_looped_nodes() {
        // Node 1 has no edges; node 2 loops to itself.
        let edges = [Edge { from: 2, to: 2 }];
        let csr = Csr::build(3, &edges);

        assert_eq!(csr.offsets, vec![0, 0, 0, 2]);
        assert!(csr.neighbors_of(1).is_empty());
        assert_eq!(csr.neighbors_of(2), &[2, 2]);
    }
}
