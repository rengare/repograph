//! The on-device tree and walk, against the paths that already work.
//!
//! Order matters here, and it is the order that has actually caught bugs in
//! this project:
//!
//! 1. **theta = 0 must match brute force.** The anchor that says the
//!    approximation is the only difference between this and the exact path.
//! 2. **Converged mean edge length must match.** Single-step assertions passed
//!    all the way through the phase 3 barrier bug; this is what caught it.
//! 3. **Two identical runs must be byte-identical.** The tree is appended by an
//!    atomic, so this is not a formality.
//!
//! And every fixture exceeds one workgroup — 257 bodies at minimum. Below that
//! a single workgroup runs the whole graph, pass ordering is incidental, and a
//! missing barrier is invisible.

use gv_graph::GraphData;
use gv_layout::LayoutParams;

use super::*;
use crate::FrGpuLayout;

/// Runs `steps` Barnes-Hut steps and hands back the layout and the node array.
fn run(
    graph: &GraphData,
    params: &LayoutParams,
    theta: f32,
    steps: usize,
) -> (GpuContext, BhGpuLayout, Vec<gv_graph::Node>) {
    let context = pollster::block_on(GpuContext::new(None)).expect("adapter available");
    let buffers = GraphBuffers::upload(&context, graph).expect("upload");
    let mut layout = BhGpuLayout::new(&context, &buffers).expect("pipelines build");
    layout.theta = theta;

    for _ in 0..steps {
        let mut encoder = context.device.create_command_encoder(&Default::default());
        layout
            .record_step(&mut encoder, &context.queue, params)
            .expect("record");
        context.queue.submit([encoder.finish()]);
    }

    let nodes = pollster::block_on(buffers.read_nodes(&context)).expect("readback");
    (context, layout, nodes)
}

/// The same graph through the exact GPU path, for comparison.
fn run_brute_force(
    graph: &GraphData,
    params: &LayoutParams,
    steps: usize,
) -> Vec<gv_graph::Node> {
    let context = pollster::block_on(GpuContext::new(None)).expect("adapter available");
    let buffers = GraphBuffers::upload(&context, graph).expect("upload");
    let mut layout = FrGpuLayout::new(&context, &buffers).expect("pipelines build");

    for _ in 0..steps {
        let mut encoder = context.device.create_command_encoder(&Default::default());
        layout
            .record_step(&mut encoder, &context.queue, params)
            .expect("record");
        context.queue.submit([encoder.finish()]);
    }

    pollster::block_on(buffers.read_nodes(&context)).expect("readback")
}

/// A graph over one workgroup with degrees that vary, matching the fixture the
/// exact GPU path is regression-tested on.
fn multi_workgroup_graph(node_count: u32) -> GraphData {
    let mut state = 12345u64;
    let mut next = move |n: u32| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as u32) % n
    };

    let mut edges: Vec<gv_graph::Edge> = (1..node_count)
        .map(|i| gv_graph::Edge { from: i, to: next(i) })
        .collect();
    for _ in 0..node_count {
        edges.push(gv_graph::Edge { from: next(node_count), to: next(node_count) });
    }

    gv_graph::testing::from_edges(node_count as usize, edges, 0)
}

fn bodies_of(nodes: &[gv_graph::Node], three_d: bool) -> Vec<[f32; 3]> {
    nodes
        .iter()
        .map(|node| {
            let p = node.position;
            [p[0], p[1], if three_d { p[2] } else { 0.0 }]
        })
        .collect()
}

fn mean_edge_length(graph: &GraphData, nodes: &[gv_graph::Node]) -> f32 {
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
                "node {index} axis {axis}: barnes-hut {a} vs brute force {e} \
                 (relative error {error})"
            );
        }
    }
}

// ---------------------------------------------------------------- the tree

#[test]
#[ignore = "requires a GPU adapter"]
fn the_tree_is_a_walkable_depth_first_array() {
    // The invariants the stackless walk rests on, and the only reason it can do
    // without a per-thread stack. If any of these breaks the walk either loops
    // forever or reads past the end, and both look like a hang, not a wrong
    // answer — so they are checked directly rather than inferred from forces.
    let params = LayoutParams::default();
    let graph = multi_workgroup_graph(2000);
    let (_context, layout, _) = run(&graph, &params, DEFAULT_THETA, 1);

    let cells = pollster::block_on(layout.read_cells()).expect("readback");
    let count = cells.len() as u32;
    assert!(count > 0, "the build emitted no cells");

    for (index, cell) in cells.iter().enumerate() {
        let index = index as u32;
        assert!(cell.escape > index, "cell {index} escapes backwards to {}", cell.escape);
        assert!(cell.escape <= count, "cell {index} escapes past the end");
        assert!(cell.first <= cell.last, "cell {index} covers an empty range");
        assert!(cell.last < graph.node_count() as u32, "cell {index} covers a body that is not there");

        // Descending is `index + 1`, so a cell with children must be followed
        // immediately by its subtree, and every one of them must fall inside
        // its range.
        let mut child = index + 1;
        let mut covered = 0u32;
        while child < cell.escape {
            let inner = &cells[child as usize];
            assert!(
                inner.first >= cell.first && inner.last <= cell.last,
                "cell {child} is inside cell {index}'s subtree but not inside its range"
            );
            covered += inner.last - inner.first + 1;
            child = inner.escape;
        }
        if cell.escape > index + 1 {
            assert_eq!(
                covered,
                cell.last - cell.first + 1,
                "cell {index}'s children do not partition its range"
            );
        }
    }

    // The root covers every body, and its escape is the cell count — which is
    // how the walk finds the end of the array without binding the counter.
    assert_eq!(cells[0].first, 0);
    assert_eq!(cells[0].last, graph.node_count() as u32 - 1);
    assert_eq!(cells[0].escape, count, "the root does not escape past the last cell");
}

#[test]
#[ignore = "requires a GPU adapter"]
fn every_cell_is_the_centre_of_mass_of_its_own_range() {
    // The stage with no structural symptom: a wrong sum still walks, still
    // terminates, and just quietly puts the forces in the wrong direction. So
    // it is checked against a host recomputation over the same body ranges.
    let params = LayoutParams { three_d: true, ..Default::default() };
    let graph = multi_workgroup_graph(1500);
    let (_context, layout, nodes) = run(&graph, &params, DEFAULT_THETA, 1);

    let cells = pollster::block_on(layout.read_cells()).expect("readback");
    let order = pollster::block_on(layout.read_order()).expect("readback");

    // The tree was built from the positions the step *started* from, so the
    // comparison uses those, not the ones it ended at.
    let bodies = bodies_of(&graph.nodes, true);

    for (index, cell) in cells.iter().enumerate() {
        let range = &order[cell.first as usize..=cell.last as usize];
        assert_eq!(cell.mass, range.len() as f32, "cell {index} has the wrong mass");

        let mut sum = [0.0f64; 3];
        for &body in range {
            for (sum, axis) in sum.iter_mut().zip(&bodies[body as usize]) {
                *sum += f64::from(*axis);
            }
        }

        for (axis, sum) in sum.iter().enumerate() {
            let expected = sum / f64::from(cell.mass);
            let error = (f64::from(cell.center[axis]) - expected).abs();
            let scale = expected.abs().max(1.0);
            assert!(
                error / scale < 1e-4,
                "cell {index} axis {axis}: device {} vs host {expected}",
                cell.center[axis]
            );
        }
    }

    assert!(!nodes.is_empty());
}

#[test]
#[ignore = "requires a GPU adapter"]
fn the_build_stays_inside_the_capacity_it_allocates() {
    // `enumerate` appends by atomic and drops anything past the end, so an
    // undercounted capacity would silently truncate the tree rather than fail.
    // The bound is 2n - 1 cells; this is what checks the reasoning behind it.
    let params = LayoutParams::default();
    for graph in [
        multi_workgroup_graph(1000),
        gv_graph::testing::path(1024),
        gv_graph::testing::dust(300),
    ] {
        let (_context, layout, _) = run(&graph, &params, DEFAULT_THETA, 1);
        let count = pollster::block_on(layout.read_cell_count()).expect("readback");
        let capacity = Tree::capacity_for(graph.node_count() as u32);
        assert!(
            count < 2 * graph.node_count() as u32,
            "{} bodies produced {count} cells, past the 2n - 1 bound",
            graph.node_count()
        );
        assert!(count < capacity, "{count} cells filled the {capacity}-slot capacity");
    }
}

#[test]
#[ignore = "requires a GPU adapter"]
fn the_tree_is_path_compressed() {
    // The clause that makes the build affordable on a clustered graph. Without
    // it, bodies sharing a long prefix produce a chain of single-child cells
    // all covering the same range; with it, every internal cell splits. A path
    // graph seeded into a cloud is exactly the shape that would show the chain.
    let params = LayoutParams::default();
    let graph = gv_graph::testing::path(4096);
    let (_context, layout, _) = run(&graph, &params, DEFAULT_THETA, 1);

    let cells = pollster::block_on(layout.read_cells()).expect("readback");
    for (index, cell) in cells.iter().enumerate() {
        if cell.escape == index as u32 + 1 {
            continue;
        }
        let first_child = &cells[index + 1];
        assert!(
            first_child.escape < cell.escape,
            "cell {index} has exactly one child; the compression clause is not firing"
        );
    }
}

// ---------------------------------------------------------------- the walk

#[test]
#[ignore = "requires a GPU adapter"]
fn theta_zero_reproduces_the_brute_force_gpu_path() {
    // The exactness anchor. With nothing accepted as an aggregate the walk
    // reaches every leaf, and a leaf iterates its bodies rather than lumping
    // them, so the only difference from the O(n²) pass is the order the
    // contributions are summed in.
    let params = LayoutParams::default();
    let graph = multi_workgroup_graph(1024);

    let (_context, _layout, actual) = run(&graph, &params, 0.0, 1);
    let expected = run_brute_force(&graph, &params, 1);

    assert_positions_close(&actual, &expected, 1e-4);
}

#[test]
#[ignore = "requires a GPU adapter"]
fn the_default_theta_stays_close_to_the_brute_force_path() {
    let params = LayoutParams::default();
    let graph = multi_workgroup_graph(1024);

    let (_context, _layout, actual) = run(&graph, &params, DEFAULT_THETA, 1);
    let expected = run_brute_force(&graph, &params, 1);

    // A step is clamped to `max_displace * speed_scale`, so the honest scale
    // for "how far apart did one step put them" is that ceiling.
    let ceiling = params.max_displace() * params.speed_scale();
    let worst = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| {
            (0..3)
                .map(|i| (a.position[i] - b.position[i]).abs())
                .fold(0.0f32, f32::max)
        })
        .fold(0.0f32, f32::max);

    assert!(
        worst < 0.2 * ceiling,
        "worst-case drift {worst} against a step ceiling of {ceiling}"
    );
}

#[test]
#[ignore = "requires a GPU adapter"]
fn both_paths_converge_to_the_same_layout() {
    // Individual positions diverge chaotically, so over a long run the only
    // meaningful comparison is an aggregate one — and mean edge length is what
    // caught the phase 3 barrier bug while every single-step assertion passed.
    // Both paths converged then, stably, to 1035 against 5204.
    let params = LayoutParams::default();
    let graph = multi_workgroup_graph(1024);

    let (_context, _layout, approximate) = run(&graph, &params, DEFAULT_THETA, 400);
    let exact = run_brute_force(&graph, &params, 400);

    let approximate = mean_edge_length(&graph, &approximate);
    let exact = mean_edge_length(&graph, &exact);

    let difference = (approximate - exact).abs() / exact;
    assert!(
        difference < 0.05,
        "converged layouts disagree: barnes-hut mean edge {approximate}, \
         brute force {exact} ({:.1}% apart)",
        difference * 100.0
    );
}

#[test]
#[ignore = "requires a GPU adapter"]
fn repeated_runs_are_byte_identical() {
    // The tree is appended with an atomic, so cells land in slots in whatever
    // order invocations reach the counter. What restores determinism is the
    // sort that follows, keyed on `first * 16 + level` — unique per node. This
    // is the test that would fail if that key were ever weakened.
    let params = LayoutParams::default();
    let graph = multi_workgroup_graph(1024);

    let (_context, _layout, first) = run(&graph, &params, DEFAULT_THETA, 20);
    let (_context, _layout, second) = run(&graph, &params, DEFAULT_THETA, 20);

    assert_eq!(first, second, "two identical Barnes-Hut runs disagreed");
}

#[test]
#[ignore = "requires a GPU adapter"]
fn three_d_is_laid_out_in_three_dimensions() {
    // 2D zeroes z on every step, which would hide a z lane that the tree drops
    // — and the tree quantises z into the code whether or not the forces use it.
    let params = LayoutParams { three_d: true, ..Default::default() };
    let graph = multi_workgroup_graph(512);

    let (_context, _layout, actual) = run(&graph, &params, 0.0, 1);
    let expected = run_brute_force(&graph, &params, 1);

    assert!(
        actual.iter().any(|node| node.position[2] != 0.0),
        "3D run left every node in the plane; the z lane is not being exercised"
    );
    assert_positions_close(&actual, &expected, 1e-4);
}

#[test]
#[ignore = "requires a GPU adapter"]
fn coincident_bodies_neither_hang_nor_produce_nan() {
    // Every body at one point: one code, so one cell, a leaf covering the whole
    // array. The walk must iterate it and find every distance zero. This is the
    // shape that makes an unguarded k²/d an infinity — and the shape a
    // subdividing build recurses forever on.
    let params = LayoutParams::default();
    let mut graph = gv_graph::testing::dust(512);
    for node in &mut graph.nodes {
        node.position = [3.0, 3.0, 3.0, 1.0];
    }

    let (_context, layout, actual) = run(&graph, &params, DEFAULT_THETA, 3);

    assert_eq!(
        pollster::block_on(layout.read_cell_count()).expect("readback"),
        1,
        "coincident bodies should collapse to a single leaf"
    );
    assert!(
        actual.iter().all(|node| node.position.iter().all(|axis| axis.is_finite())),
        "coincident bodies produced a non-finite position"
    );
}

#[test]
#[ignore = "requires a GPU adapter"]
fn a_distant_cluster_is_felt_as_its_aggregate() {
    // The point of the whole structure, stated as a force rather than as a cell
    // count: a tight clump far away must push a lone body as hard as its mass
    // says, whether the walk opened it or took it whole. If the aggregate were
    // ever built from the wrong mass, this is where it shows.
    let params = LayoutParams { area: 0.1, gravity: 0.0, speed: 100.0, three_d: false };
    let mut graph = gv_graph::testing::dust(513);
    for (index, node) in graph.nodes.iter_mut().enumerate() {
        node.position = if index == 0 {
            [0.0, 0.0, 0.0, 1.0]
        } else {
            // A clump 500 away, tight enough that theta accepts it whole.
            [500.0 + (index % 4) as f32 * 0.01, (index % 7) as f32 * 0.01, 0.0, 1.0]
        };
    }

    let (_context, _layout, approximate) = run(&graph, &params, DEFAULT_THETA, 1);
    let exact = run_brute_force(&graph, &params, 1);

    assert!(
        approximate[0].position[0] < 0.0,
        "the lone body was not pushed away from the clump: {:?}",
        approximate[0].position
    );
    assert_positions_close(&approximate, &exact, 1e-3);
}

#[test]
#[ignore = "requires a GPU adapter"]
fn an_edgeless_graph_steps() {
    // Repulsion only, and every CSR row empty — the shape most likely to index
    // past the end of a padded buffer.
    let params = LayoutParams::default();
    let graph = gv_graph::testing::dust(300);

    let (_context, _layout, actual) = run(&graph, &params, 0.0, 3);
    let expected = run_brute_force(&graph, &params, 3);

    assert_positions_close(&actual, &expected, 1e-3);
}

#[test]
#[ignore = "benchmark; requires a GPU adapter"]
fn stress_100k() {
    use std::time::Instant;

    const BODIES: usize = 100_000;
    const STEPS: u32 = 50;

    // 3D, because that is what the application defaults to and it is the harder
    // case: flattening z turns the octree into a quadtree, which is shallower,
    // has fewer cells and diverges far less inside a subgroup. Reporting the 2D
    // number as the headline would be flattering rather than honest.
    let params = LayoutParams { three_d: true, ..Default::default() };
    let graph = gv_graph::testing::path(BODIES);

    let context = pollster::block_on(GpuContext::new(None)).expect("adapter available");
    let buffers = GraphBuffers::upload(&context, &graph).expect("upload");
    let mut layout = BhGpuLayout::new(&context, &buffers).expect("pipelines build");

    // One untimed step first: the first submission pays for pipeline
    // compilation and buffer residency, which is not what is being measured.
    let mut encoder = context.device.create_command_encoder(&Default::default());
    layout.record_step(&mut encoder, &context.queue, &params).expect("record");
    context.queue.submit([encoder.finish()]);
    context.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");

    let started = Instant::now();
    for _ in 0..STEPS {
        let mut encoder = context.device.create_command_encoder(&Default::default());
        layout.record_step(&mut encoder, &context.queue, &params).expect("record");
        context.queue.submit([encoder.finish()]);
    }
    // The clock stops after the device drains, not after the last submit.
    context.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let per_step = started.elapsed().as_secs_f64() * 1000.0 / f64::from(STEPS);

    let cells = pollster::block_on(layout.read_cell_count()).expect("readback");
    println!("gpu barnes-hut over {BODIES} bodies: {per_step:.3} ms/step, {cells} cells");

    // Correctness is not suspended for a benchmark: a build that quietly
    // truncated would otherwise look like a speedup.
    assert!(cells > 0 && cells < Tree::capacity_for(BODIES as u32));
    let nodes = pollster::block_on(buffers.read_nodes(&context)).expect("readback");
    assert!(nodes.iter().all(|n| n.position.iter().all(|a| a.is_finite())));
}


