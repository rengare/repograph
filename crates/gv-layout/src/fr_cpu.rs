//! Fruchterman-Reingold with brute-force O(n²) repulsion.
//!
//! This is the reference implementation: the GPU port in `gv-layout-gpu` is
//! validated against it, so it favours being obviously correct over being
//! fast. Repulsion parallelises over nodes with rayon (the original was
//! single-threaded); the attractive pass gathers over CSR adjacency so each
//! node's displacement is written by exactly one worker.

use gv_graph::GraphData;
use rayon::prelude::*;

use crate::{CpuLayout, LayoutParams, attraction, integrate, separation};

#[derive(Debug, Default)]
pub struct FrCpuLayout;

impl CpuLayout for FrCpuLayout {
    fn name(&self) -> &'static str {
        "F-R cpu"
    }

    fn step(&mut self, graph: &mut GraphData, params: &LayoutParams) {
        let node_count = graph.nodes.len();
        if node_count == 0 {
            return;
        }

        let k = params.k(node_count);
        let three_d = params.three_d;

        // Positions are snapshotted so every node sees the same state, and so
        // the parallel loop below borrows nothing mutably. This is also what
        // makes the result independent of iteration order — the property the
        // original's in-place shader writes destroyed.
        let positions: Vec<[f32; 4]> = graph.nodes.iter().map(|node| node.position).collect();
        let adjacency = &graph.adjacency;

        let displacements: Vec<[f32; 3]> = (0..node_count)
            .into_par_iter()
            .map(|i| {
                let mut displacement = [0.0f32; 3];
                let position = positions[i];

                // Repulsion: k² / d, away from every other node.
                //
                // The full 0..n loop, not the original's j = i + 1: that
                // half-loop computed each pair once but applied the force to
                // only one of the two nodes, which is not a symmetric force.
                for (j, other) in positions.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    let (delta, distance) = separation(&position, other, three_d);
                    if distance > 0.0 {
                        let magnitude = (k * k) / distance;
                        for (component, delta) in displacement.iter_mut().zip(&delta) {
                            *component += delta / distance * magnitude;
                        }
                    }
                }

                // Attraction: d² / k, toward every neighbour.
                let attractive = attraction(i, &positions, adjacency, k, three_d);
                for (component, attractive) in displacement.iter_mut().zip(&attractive) {
                    *component += attractive;
                }

                displacement
            })
            .collect();

        integrate(graph, displacements, params, k);
    }
}

/// The behaviour phase 1 has to deliver.
///
/// These are `#[ignore]`d because [`FrCpuLayout::step`] is still `todo!()` and
/// would panic. Drop the attribute as the implementation lands; the assertions
/// themselves are the specification and should not need editing.
#[cfg(test)]
mod tests {
    use super::*;
    use gv_graph::testing;

    fn run(graph: &mut GraphData, params: &LayoutParams, steps: usize) {
        let mut layout = FrCpuLayout;
        for _ in 0..steps {
            layout.step(graph, params);
        }
    }

    #[test]
    fn names_itself_for_the_picker() {
        assert_eq!(FrCpuLayout.name(), "F-R cpu");
    }

    #[test]
    fn a_single_step_leaves_positions_finite() {
        let mut graph = testing::path(64);
        run(&mut graph, &LayoutParams::default(), 1);
        assert!(testing::all_finite(&graph));
    }

    #[test]
    fn two_joined_nodes_settle_near_the_ideal_edge_length() {
        // Attraction grows as d²/k, repulsion falls as k²/d; alone, they
        // balance at d = k. Gravity has to be off for that to be the whole
        // story — at the default area it is the dominant term for a two-node
        // graph (0.01·k·|p| against k²/d, with k ≈ 1.7e5) and the pair
        // collapses toward the origin instead.
        //
        // `area` is chosen so k lands within reach of the displacement ceiling
        // in the step budget: k = 2000, ceiling ≈ 1.94/step, ~465 steps to
        // cross from the starting separation.
        let params = LayoutParams { speed: 100.0, area: 12.0, gravity: 0.0, three_d: false };
        let mut graph = testing::dumbbell();
        graph.nodes[0].position = [-100.0, 0.0, 0.0, 1.0];
        graph.nodes[1].position = [100.0, 0.0, 0.0, 1.0];

        run(&mut graph, &params, 2_000);

        let separation = testing::distance(&graph, 0, 1);
        let k = params.k(graph.node_count());
        assert!(
            (0.5 * k..2.0 * k).contains(&separation),
            "separation {separation} is not near k = {k}"
        );
    }

    #[test]
    fn identical_input_gives_identical_output() {
        // The property the original could not offer: its GPU passes raced, so
        // no two runs agreed. The CPU reference must be exactly reproducible
        // or it cannot serve as the oracle for the GPU port.
        let params = LayoutParams::default();
        let (mut a, mut b) = (testing::triangle(), testing::triangle());
        run(&mut a, &params, 50);
        run(&mut b, &params, 50);
        assert_eq!(a.nodes, b.nodes);
    }

    #[test]
    fn two_d_mode_never_leaves_the_plane() {
        let params = LayoutParams { three_d: false, ..Default::default() };
        let mut graph = testing::path(32);
        for node in &mut graph.nodes {
            node.position[2] = 0.0;
        }
        run(&mut graph, &params, 100);
        assert!(graph.nodes.iter().all(|node| node.position[2] == 0.0));
    }

    #[test]
    fn unconnected_nodes_repel_each_other() {
        let mut graph = testing::dust(2);
        // Place them close together so repulsion dominates gravity.
        graph.nodes[0].position = [-1.0, 0.0, 0.0, 1.0];
        graph.nodes[1].position = [1.0, 0.0, 0.0, 1.0];
        let before = testing::distance(&graph, 0, 1);

        run(&mut graph, &LayoutParams::default(), 1);
        assert!(testing::distance(&graph, 0, 1) > before);
    }

    #[test]
    fn coincident_nodes_do_not_produce_nan() {
        // Distance zero is a division hazard in every force term.
        let mut graph = testing::dust(4);
        for node in &mut graph.nodes {
            node.position = [0.0, 0.0, 0.0, 1.0];
        }
        run(&mut graph, &LayoutParams::default(), 10);
        assert!(testing::all_finite(&graph));
    }

    #[test]
    fn an_empty_graph_is_a_no_op() {
        let mut graph = testing::dust(0);
        run(&mut graph, &LayoutParams::default(), 5);
        assert_eq!(graph.node_count(), 0);
    }

    #[test]
    fn no_step_moves_a_node_further_than_the_displacement_ceiling() {
        // 3D, matching how the fixture is seeded: in 2D the step also flattens
        // z to zero, which is a legitimate move larger than the ceiling.
        let params = LayoutParams { three_d: true, ..Default::default() };
        let mut graph = testing::path(32);
        let before: Vec<_> = graph.nodes.iter().map(|n| n.position).collect();

        run(&mut graph, &params, 1);

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
    fn every_edge_contributes_to_attraction() {
        // The defect this whole design works around: the original's per-edge
        // scatter lost contributions to a race, so a node with many neighbours
        // was pulled as if it had few. A high-degree node must feel more
        // attraction than a low-degree one at equal distance.
        let mut star = testing::from_edges(
            5,
            (1..5).map(|i| gv_graph::Edge { from: 0, to: i }).collect(),
            0,
        );
        for (index, node) in star.nodes.iter_mut().enumerate() {
            node.position = [100.0 * index as f32, 0.0, 0.0, 1.0];
        }
        let hub_before = star.nodes[0].position[0];

        // Attraction only outweighs repulsion beyond d = k, so `area` is set
        // to put k (= 5) well inside the 100..400 spacing. At the default area
        // k ≈ 8.3e4 and repulsion swamps the signal this test is looking for.
        let params = LayoutParams { speed: 100.0, area: 0.06, gravity: 0.0, three_d: false };
        run(&mut star, &params, 1);

        // All four neighbours sit at positive x, so the hub must move that way.
        assert!(
            star.nodes[0].position[0] > hub_before,
            "hub moved to {} from {hub_before}",
            star.nodes[0].position[0]
        );
    }
}
