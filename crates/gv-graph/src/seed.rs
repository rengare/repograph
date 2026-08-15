//! Initial node placement, colour and size.
//!
//! The original drew these from a global `rand()`; this takes an explicit seed
//! so a CPU run and a GPU run can start from byte-identical state, which is
//! what makes the two paths comparable.

use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::GraphData;

#[derive(Debug, Clone, Copy)]
pub struct SeedOptions {
    /// Positions are drawn uniformly from `-extent..=extent` per axis.
    pub extent: f32,
    pub size_range: (f32, f32),
    /// When false, `z` is pinned to 0 and the layout stays planar.
    pub three_d: bool,
    pub seed: u64,
}

impl Default for SeedOptions {
    fn default() -> Self {
        Self {
            extent: 1000.0,
            size_range: (10.0, 30.0),
            three_d: true,
            seed: 0,
        }
    }
}

/// Places, colours and sizes every node in `graph`.
pub fn scatter(graph: &mut GraphData, options: SeedOptions) {
    let mut rng = StdRng::seed_from_u64(options.seed);
    let (min_size, max_size) = options.size_range;

    for node in &mut graph.nodes {
        node.position = [
            rng.random_range(-options.extent..=options.extent),
            rng.random_range(-options.extent..=options.extent),
            if options.three_d {
                rng.random_range(-options.extent..=options.extent)
            } else {
                0.0
            },
            1.0,
        ];
        node.color = [
            rng.random_range(0.1..=1.0),
            rng.random_range(0.1..=1.0),
            rng.random_range(0.1..=1.0),
            1.0,
        ];
        node.size = rng.random_range(min_size..=max_size);
        node.disp = [0.0; 3];
    }
}

/// Overrides node colours from their knowledge-graph kind, when the graph was
/// loaded with a sidecar (`GraphData::meta`). Call after [`scatter`] to keep the
/// random positions/sizes but replace the random colours with legible per-kind
/// ones. A no-op when `meta` is empty (a plain edge-list load).
pub fn colorize_by_kind(graph: &mut GraphData) {
    if graph.meta.len() != graph.nodes.len() {
        return;
    }
    for (node, meta) in graph.nodes.iter_mut().zip(&graph.meta) {
        node.color = meta.kind.color();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Node, NodeCategory, NodeMeta};

    fn graph_of(count: usize) -> GraphData {
        GraphData {
            nodes: vec![Node::default(); count],
            ..Default::default()
        }
    }

    #[test]
    fn same_seed_gives_identical_placement() {
        let options = SeedOptions::default();
        let (mut a, mut b) = (graph_of(64), graph_of(64));
        scatter(&mut a, options);
        scatter(&mut b, options);
        assert_eq!(a.nodes, b.nodes);
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = graph_of(64);
        let mut b = graph_of(64);
        scatter(&mut a, SeedOptions { seed: 1, ..Default::default() });
        scatter(&mut b, SeedOptions { seed: 2, ..Default::default() });
        assert_ne!(a.nodes, b.nodes);
    }

    #[test]
    fn two_d_mode_pins_z_to_zero() {
        let mut graph = graph_of(32);
        scatter(&mut graph, SeedOptions { three_d: false, ..Default::default() });
        assert!(graph.nodes.iter().all(|node| node.position[2] == 0.0));
    }

    #[test]
    fn colorize_by_kind_overrides_random_colors() {
        let mut graph = GraphData {
            nodes: vec![Node::default(); 2],
            meta: vec![
                NodeMeta { kind: NodeCategory::File, ..Default::default() },
                NodeMeta { kind: NodeCategory::Doc, ..Default::default() },
            ],
            ..Default::default()
        };
        scatter(&mut graph, SeedOptions::default());
        colorize_by_kind(&mut graph);
        assert_eq!(graph.nodes[0].color, NodeCategory::File.color());
        assert_eq!(graph.nodes[1].color, NodeCategory::Doc.color());
    }

    #[test]
    fn colorize_is_noop_without_meta() {
        let mut graph = graph_of(4);
        scatter(&mut graph, SeedOptions::default());
        let before: Vec<_> = graph.nodes.iter().map(|n| n.color).collect();
        colorize_by_kind(&mut graph);
        let after: Vec<_> = graph.nodes.iter().map(|n| n.color).collect();
        assert_eq!(before, after);
    }

    #[test]
    fn stays_inside_the_requested_bounds() {
        let mut graph = graph_of(256);
        let options = SeedOptions { extent: 500.0, size_range: (2.0, 4.0), ..Default::default() };
        scatter(&mut graph, options);

        for node in &graph.nodes {
            for axis in &node.position[..3] {
                assert!((-500.0..=500.0).contains(axis), "{axis} out of bounds");
            }
            assert!((2.0..=4.0).contains(&node.size));
        }
    }
}
