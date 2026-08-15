//! Random jitter — the original's `RandomModel`.
//!
//! Produces no meaningful layout. It exists as a baseline: it costs one write
//! per node per step, so it measures the cost of everything that is not the
//! layout itself (buffer upload, draw, present).

use gv_graph::GraphData;

use crate::{CpuLayout, LayoutParams};

#[derive(Debug, Default)]
pub struct RandomLayout;

impl CpuLayout for RandomLayout {
    fn name(&self) -> &'static str {
        "Random"
    }

    fn step(&mut self, _graph: &mut GraphData, _params: &LayoutParams) {
        todo!("phase 5: per-node jitter baseline")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gv_graph::testing;

    #[test]
    fn names_itself_for_the_picker() {
        assert_eq!(RandomLayout.name(), "Random");
    }

    #[test]
    #[ignore = "phase 5"]
    fn every_node_moves() {
        let mut graph = testing::dust(64);
        let before: Vec<_> = graph.nodes.iter().map(|n| n.position).collect();
        RandomLayout.step(&mut graph, &LayoutParams::default());

        for (index, previous) in before.iter().enumerate() {
            assert_ne!(graph.nodes[index].position, *previous, "node {index} did not move");
        }
    }

    #[test]
    #[ignore = "phase 5"]
    fn jitter_respects_the_displacement_ceiling() {
        let params = LayoutParams::default();
        let mut graph = testing::dust(64);
        let before: Vec<_> = graph.nodes.iter().map(|n| n.position).collect();
        RandomLayout.step(&mut graph, &params);

        let ceiling = params.max_displace() * params.speed_scale();
        for (index, previous) in before.iter().enumerate() {
            let current = graph.nodes[index].position;
            let moved = ((current[0] - previous[0]).powi(2)
                + (current[1] - previous[1]).powi(2)
                + (current[2] - previous[2]).powi(2))
            .sqrt();
            assert!(moved <= ceiling * 1.001, "node {index} moved {moved} > {ceiling}");
        }
    }

    #[test]
    #[ignore = "phase 5"]
    fn two_d_mode_never_leaves_the_plane() {
        let mut graph = testing::dust(32);
        for node in &mut graph.nodes {
            node.position[2] = 0.0;
        }
        RandomLayout.step(&mut graph, &LayoutParams { three_d: false, ..Default::default() });
        assert!(graph.nodes.iter().all(|node| node.position[2] == 0.0));
    }

    #[test]
    #[ignore = "phase 5"]
    fn an_empty_graph_is_a_no_op() {
        let mut graph = testing::dust(0);
        RandomLayout.step(&mut graph, &LayoutParams::default());
        assert_eq!(graph.node_count(), 0);
    }
}
