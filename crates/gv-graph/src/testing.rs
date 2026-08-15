//! Small graphs with known structure, for testing layout algorithms.
//!
//! Gated behind the `testing` feature so it is absent from release builds;
//! `gv-layout` enables it as a dev-dependency.

use crate::{Csr, Edge, GraphData, Node, seed};

/// Builds a graph from an edge list over `node_count` nodes, scattered with
/// [`seed::SeedOptions::default`] at the given seed.
pub fn from_edges(node_count: usize, edges: Vec<Edge>, rng_seed: u64) -> GraphData {
    let adjacency = Csr::build(node_count, &edges);
    let mut graph = GraphData {
        nodes: vec![Node::default(); node_count],
        edges,
        adjacency,
        labels: (0..node_count).map(|i| i.to_string()).collect(),
        meta: Vec::new(),
    };
    seed::scatter(
        &mut graph,
        seed::SeedOptions { seed: rng_seed, ..Default::default() },
    );
    graph
}

/// Two nodes joined by one edge — the smallest graph with a force balance,
/// which settles at a separation near `k`.
pub fn dumbbell() -> GraphData {
    from_edges(2, vec![Edge { from: 0, to: 1 }], 0)
}

/// A path `0 - 1 - ... - (n - 1)`.
pub fn path(node_count: usize) -> GraphData {
    let edges = (0..node_count.saturating_sub(1))
        .map(|i| Edge { from: i as u32, to: i as u32 + 1 })
        .collect();
    from_edges(node_count, edges, 0)
}

/// A triangle: every pair joined, so the layout should tend to equilateral.
pub fn triangle() -> GraphData {
    from_edges(
        3,
        vec![
            Edge { from: 0, to: 1 },
            Edge { from: 1, to: 2 },
            Edge { from: 0, to: 2 },
        ],
        0,
    )
}

/// `node_count` nodes and no edges — repulsion only, so it should expand
/// without bound until gravity balances it.
pub fn dust(node_count: usize) -> GraphData {
    from_edges(node_count, Vec::new(), 0)
}

/// Euclidean distance between two nodes, ignoring the padding lane.
pub fn distance(graph: &GraphData, a: usize, b: usize) -> f32 {
    let (p, q) = (graph.nodes[a].position, graph.nodes[b].position);
    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
}

/// True if every node position is finite — the cheapest guard against a
/// layout that has diverged or divided by zero.
pub fn all_finite(graph: &GraphData) -> bool {
    graph
        .nodes
        .iter()
        .all(|node| node.position.iter().all(|axis| axis.is_finite()))
}
