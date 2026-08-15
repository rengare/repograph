//! Fruchterman-Reingold with Barnes-Hut approximated repulsion.
//!
//! Replaces the O(n²) inner loop with an octree walk: a cell whose width over
//! distance falls below `theta` contributes as a single aggregate body. This
//! is the improvement the original's README conceded it never made ("brute
//! force ... although it can be improved"), and it replaces the nanoflann
//! k-d-tree radius search of `FRModelCpuKdTree`, which approximated by
//! ignoring distant nodes outright rather than aggregating them.

use gv_graph::GraphData;
use rayon::prelude::*;

use crate::octree::Octree;
use crate::{CpuLayout, LayoutParams, attraction, integrate};

#[derive(Debug)]
pub struct BarnesHutLayout {
    /// Opening angle. Smaller is more accurate and slower; 0.5 is the usual
    /// default, 0 degrades to brute force.
    pub theta: f32,
}

impl Default for BarnesHutLayout {
    fn default() -> Self {
        Self { theta: 0.5 }
    }
}

impl CpuLayout for BarnesHutLayout {
    fn name(&self) -> &'static str {
        "F-R cpu barnes-hut"
    }

    fn step(&mut self, graph: &mut GraphData, params: &LayoutParams) {
        let node_count = graph.nodes.len();
        if node_count == 0 {
            return;
        }

        let k = params.k(node_count);
        let three_d = params.three_d;

        let positions: Vec<[f32; 4]> = graph.nodes.iter().map(|node| node.position).collect();

        // The tree is built on the coordinates the forces actually act in, so
        // in 2D z is flattened here rather than being dropped later. Otherwise
        // the tree would subdivide along an axis that contributes nothing to
        // any distance, wasting depth and weakening the opening criterion.
        let bodies: Vec<[f32; 3]> = positions
            .iter()
            .map(|p| [p[0], p[1], if three_d { p[2] } else { 0.0 }])
            .collect();
        let tree = Octree::build(&bodies);

        let adjacency = &graph.adjacency;
        let theta = self.theta;

        // Walked in Morton order, not index order: see `Octree::order`. The
        // results come back permuted and have to be scattered home, which costs
        // one linear pass and is repaid many times over by the locality.
        let walked: Vec<[f32; 3]> = tree
            .order
            .par_iter()
            .map(|&body| {
                let i = body as usize;
                let mut displacement = tree.repulsion(body, bodies[i], k, theta);

                let attractive = attraction(i, &positions, adjacency, k, three_d);
                for (component, attractive) in displacement.iter_mut().zip(&attractive) {
                    *component += attractive;
                }

                displacement
            })
            .collect();

        let mut displacements = vec![[0.0f32; 3]; node_count];
        for (slot, &body) in tree.order.iter().enumerate() {
            displacements[body as usize] = walked[slot];
        }

        integrate(graph, displacements, params, k);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fr_cpu::FrCpuLayout;
    use gv_graph::testing;

    #[test]
    fn names_itself_for_the_picker() {
        assert_eq!(BarnesHutLayout::default().name(), "F-R cpu barnes-hut");
    }

    #[test]
    fn defaults_to_the_conventional_opening_angle() {
        assert_eq!(BarnesHutLayout::default().theta, 0.5);
    }

    #[test]
    fn theta_zero_reproduces_brute_force() {
        // With no cell ever accepted as an aggregate, the octree walk visits
        // every body and must agree with the reference to float tolerance.
        // This is the test that proves the approximation is the only source of
        // difference between the two implementations.
        let params = LayoutParams::default();
        let mut approximate = testing::path(48);
        let mut exact = approximate.clone();

        BarnesHutLayout { theta: 0.0 }.step(&mut approximate, &params);
        FrCpuLayout.step(&mut exact, &params);

        for (a, b) in approximate.nodes.iter().zip(&exact.nodes) {
            for axis in 0..3 {
                let delta = (a.position[axis] - b.position[axis]).abs();
                assert!(delta < 1e-2, "diverged by {delta} on axis {axis}");
            }
        }
    }

    #[test]
    fn the_default_theta_stays_close_to_brute_force() {
        let params = LayoutParams::default();
        let mut approximate = testing::path(200);
        let mut exact = approximate.clone();

        BarnesHutLayout::default().step(&mut approximate, &params);
        FrCpuLayout.step(&mut exact, &params);

        let worst = approximate
            .nodes
            .iter()
            .zip(&exact.nodes)
            .map(|(a, b)| {
                (0..3)
                    .map(|i| (a.position[i] - b.position[i]).abs())
                    .fold(0.0_f32, f32::max)
            })
            .fold(0.0_f32, f32::max);

        // Loose: this asserts the approximation is sane, not that it is exact.
        assert!(worst < params.max_displace(), "worst-case drift {worst}");
    }

    #[test]
    fn coincident_nodes_do_not_produce_nan() {
        // Degenerate cells — every body at one point — are the classic way an
        // octree build recurses forever or divides by a zero cell width.
        let mut graph = testing::dust(16);
        for node in &mut graph.nodes {
            node.position = [0.0, 0.0, 0.0, 1.0];
        }
        BarnesHutLayout::default().step(&mut graph, &LayoutParams::default());
        assert!(testing::all_finite(&graph));
    }

    #[test]
    fn an_empty_graph_is_a_no_op() {
        let mut graph = testing::dust(0);
        BarnesHutLayout::default().step(&mut graph, &LayoutParams::default());
        assert_eq!(graph.node_count(), 0);
    }

    #[test]
    fn identical_input_gives_identical_output() {
        let params = LayoutParams::default();
        let (mut a, mut b) = (testing::triangle(), testing::triangle());
        BarnesHutLayout::default().step(&mut a, &params);
        BarnesHutLayout::default().step(&mut b, &params);
        assert_eq!(a.nodes, b.nodes);
    }
}
