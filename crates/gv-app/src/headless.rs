//! Windowless layout runs.
//!
//! Loads a graph, steps a layout a fixed number of times and reports what came
//! out. This is how the CPU reference is exercised before there is anything to
//! render, and — since phase 3 — how the two paths are timed against each
//! other without a window's vsync in the way.

use std::fmt;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use gv_gpu::{GpuContext, GraphBuffers};
use gv_graph::GraphData;
use gv_gui::LayoutChoice;
use gv_layout::{CpuLayout, LayoutParams, barnes_hut::BarnesHutLayout, fr_cpu::FrCpuLayout,
                random::RandomLayout};
use gv_layout_gpu::{BhGpuLayout, FrGpuLayout, GpuLayout};

/// Axis-aligned bounds of a laid-out graph.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Bounds {
    /// Bounds of every node position. `None` for an empty graph, which has no
    /// meaningful extent.
    pub fn of(graph: &GraphData) -> Option<Self> {
        let mut nodes = graph.nodes.iter();
        let first = nodes.next()?;

        let mut bounds = Self {
            min: [first.position[0], first.position[1], first.position[2]],
            max: [first.position[0], first.position[1], first.position[2]],
        };

        for node in nodes {
            for axis in 0..3 {
                bounds.min[axis] = bounds.min[axis].min(node.position[axis]);
                bounds.max[axis] = bounds.max[axis].max(node.position[axis]);
            }
        }

        Some(bounds)
    }

    pub fn extent(&self) -> [f32; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }
}

/// Mean length of an edge, or `None` when there are no edges.
///
/// The headline quality number: a converged Fruchterman-Reingold layout puts
/// this near `k`, so it is the cheapest signal that a run did something
/// sensible rather than collapsing to a point or flying apart.
pub fn mean_edge_length(graph: &GraphData) -> Option<f32> {
    if graph.edges.is_empty() {
        return None;
    }

    let total: f64 = graph
        .edges
        .iter()
        .map(|edge| {
            let a = graph.nodes[edge.from as usize].position;
            let b = graph.nodes[edge.to as usize].position;
            let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
            f64::from(dx * dx + dy * dy + dz * dz).sqrt()
        })
        .sum();

    Some((total / graph.edges.len() as f64) as f32)
}

#[derive(Debug, Clone)]
pub struct Report {
    pub layout: &'static str,
    pub node_count: usize,
    pub edge_count: usize,
    pub steps: u32,
    pub elapsed: Duration,
    pub bounds: Option<Bounds>,
    pub mean_edge_length: Option<f32>,
    /// Ideal edge length for this graph, for comparison against the mean.
    pub k: f32,
}

impl Report {
    pub fn millis_per_step(&self) -> f64 {
        if self.steps == 0 {
            return 0.0;
        }
        self.elapsed.as_secs_f64() * 1000.0 / f64::from(self.steps)
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "layout          {}", self.layout)?;
        writeln!(f, "nodes           {}", self.node_count)?;
        writeln!(f, "edges           {}", self.edge_count)?;
        writeln!(f, "steps           {}", self.steps)?;
        writeln!(
            f,
            "elapsed         {:.3} s ({:.3} ms/step)",
            self.elapsed.as_secs_f64(),
            self.millis_per_step()
        )?;
        writeln!(f, "ideal edge (k)  {:.1}", self.k)?;

        match self.mean_edge_length {
            Some(mean) => writeln!(f, "mean edge       {mean:.1} ({:.2}k)", mean / self.k)?,
            None => writeln!(f, "mean edge       n/a (no edges)")?,
        }

        match self.bounds {
            Some(bounds) => {
                let [x, y, z] = bounds.extent();
                writeln!(f, "extent          {x:.1} x {y:.1} x {z:.1}")?;
            }
            None => writeln!(f, "extent          n/a (no nodes)")?,
        }

        Ok(())
    }
}

/// Instantiates the CPU layout behind a picker choice.
///
/// The GPU layout is deliberately absent: it does not implement [`CpuLayout`]
/// at all, because it never brings the node array into host memory. Headless
/// GPU runs go through [`step_on_gpu`] instead.
pub fn cpu_layout_for(choice: LayoutChoice) -> Result<Box<dyn CpuLayout>> {
    Ok(match choice {
        LayoutChoice::FrCpu => Box::new(FrCpuLayout),
        LayoutChoice::FrBarnesHut => Box::new(BarnesHutLayout::default()),
        LayoutChoice::Random => Box::new(RandomLayout),
        LayoutChoice::FrGpu | LayoutChoice::FrGpuBarnesHut => {
            bail!("{choice:?} is not a CpuLayout; headless runs it through step_on_gpu")
        }
    })
}

/// Instantiates the device layout behind a picker choice.
fn gpu_layout_for(
    choice: LayoutChoice,
    context: &GpuContext,
    buffers: &GraphBuffers,
) -> Result<Box<dyn GpuLayout>> {
    Ok(match choice {
        LayoutChoice::FrGpu => Box::new(FrGpuLayout::new(context, buffers)?),
        LayoutChoice::FrGpuBarnesHut => Box::new(BhGpuLayout::new(context, buffers)?),
        other => bail!("{other:?} does not run on the device; use cpu_layout_for"),
    })
}

/// Runs the layout on a device, leaving the result in `graph`.
///
/// Returns the layout's name and how long the steps took.
///
/// The clock stops after a device poll, not after the last submit. Submitting
/// is nearly free — without the wait this would report the cost of building
/// command buffers and claim a speedup that is not there.
pub fn step_on_gpu(
    graph: &mut GraphData,
    params: &LayoutParams,
    choice: LayoutChoice,
    steps: u32,
) -> Result<(&'static str, Duration)> {
    let context = pollster::block_on(GpuContext::new(None))
        .context("headless GPU runs need an adapter; try --layout cpu")?;
    let buffers = GraphBuffers::upload(&context, graph)?;
    let mut layout = gpu_layout_for(choice, &context, &buffers)?;

    let started = Instant::now();
    for _ in 0..steps {
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("headless step"),
            });
        layout.record_step(&mut encoder, &context.queue, params)?;
        context.queue.submit([encoder.finish()]);
    }
    context.device.poll(wgpu::PollType::wait_indefinitely())?;
    let elapsed = started.elapsed();

    // The host copy is stale after a GPU run; every number the report quotes
    // is derived from it.
    graph.nodes = pollster::block_on(buffers.read_nodes(&context))?;

    Ok((layout.name(), elapsed))
}

/// Steps `graph` in place and reports the result.
pub fn run(
    graph: &mut GraphData,
    params: &LayoutParams,
    choice: LayoutChoice,
    steps: u32,
) -> Result<Report> {
    let (name, elapsed) = if choice.is_gpu() {
        step_on_gpu(graph, params, choice, steps)?
    } else {
        let mut layout = cpu_layout_for(choice)?;
        let started = Instant::now();
        for _ in 0..steps {
            layout.step(graph, params);
        }
        (layout.name(), started.elapsed())
    };

    Ok(Report {
        layout: name,
        node_count: graph.node_count(),
        edge_count: graph.edge_count(),
        steps,
        elapsed,
        bounds: Bounds::of(graph),
        mean_edge_length: mean_edge_length(graph),
        k: params.k(graph.node_count()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gv_graph::{Csr, Edge, GraphData, Node};

    fn graph_at(positions: &[[f32; 3]], edges: Vec<Edge>) -> GraphData {
        let nodes = positions
            .iter()
            .map(|p| Node {
                position: [p[0], p[1], p[2], 1.0],
                ..Default::default()
            })
            .collect::<Vec<_>>();
        let adjacency = Csr::build(nodes.len(), &edges);
        GraphData { nodes, edges, adjacency, labels: Vec::new(), meta: Vec::new() }
    }

    #[test]
    fn bounds_span_every_node() {
        let graph = graph_at(&[[-1.0, 2.0, 0.0], [3.0, -4.0, 5.0]], Vec::new());
        let bounds = Bounds::of(&graph).unwrap();

        assert_eq!(bounds.min, [-1.0, -4.0, 0.0]);
        assert_eq!(bounds.max, [3.0, 2.0, 5.0]);
        assert_eq!(bounds.extent(), [4.0, 6.0, 5.0]);
    }

    #[test]
    fn bounds_of_an_empty_graph_are_absent() {
        assert_eq!(Bounds::of(&graph_at(&[], Vec::new())), None);
    }

    #[test]
    fn bounds_of_a_single_node_have_zero_extent() {
        let graph = graph_at(&[[7.0, 7.0, 7.0]], Vec::new());
        assert_eq!(Bounds::of(&graph).unwrap().extent(), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn mean_edge_length_averages_over_edges() {
        // Two edges of length 3 and 5.
        let graph = graph_at(
            &[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 5.0, 0.0]],
            vec![Edge { from: 0, to: 1 }, Edge { from: 0, to: 2 }],
        );
        assert_eq!(mean_edge_length(&graph), Some(4.0));
    }

    #[test]
    fn mean_edge_length_is_absent_without_edges() {
        assert_eq!(mean_edge_length(&graph_at(&[[0.0; 3]], Vec::new())), None);
    }

    #[test]
    fn the_gpu_layout_is_not_a_cpu_layout() {
        // It never brings the node array into host memory, so it cannot
        // implement the trait; `run` routes it to `step_on_gpu` instead.
        let Err(error) = cpu_layout_for(LayoutChoice::FrGpu) else {
            panic!("the GPU layout should not resolve to a CpuLayout");
        };
        assert!(error.to_string().contains("step_on_gpu"), "{error}");
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn a_headless_gpu_run_reports_the_same_shape_as_a_cpu_run() {
        let params = LayoutParams { speed: 100.0, area: 12.0, gravity: 0.0, three_d: false };
        let mut graph = graph_at(
            &[[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            vec![Edge { from: 0, to: 1 }],
        );

        let report = run(&mut graph, &params, LayoutChoice::FrGpu, 100).unwrap();

        assert_eq!(report.layout, "F-R gpu");
        // Proves the read-back happened: without it every position would still
        // be the input, and the mean edge length would be exactly 20.
        assert_ne!(report.mean_edge_length, Some(20.0));
        assert!(report.mean_edge_length.is_some_and(f32::is_finite));
        assert!(report.elapsed > Duration::ZERO, "the clock did not run");
    }

    #[test]
    fn cpu_choices_resolve_to_their_layouts() {
        assert_eq!(cpu_layout_for(LayoutChoice::FrCpu).unwrap().name(), "F-R cpu");
        assert_eq!(
            cpu_layout_for(LayoutChoice::FrBarnesHut).unwrap().name(),
            "F-R cpu barnes-hut"
        );
    }

    #[test]
    fn zero_steps_reports_no_time_per_step() {
        let mut graph = graph_at(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], Vec::new());
        let report = run(&mut graph, &LayoutParams::default(), LayoutChoice::FrCpu, 0).unwrap();

        assert_eq!(report.steps, 0);
        assert_eq!(report.millis_per_step(), 0.0);
        assert_eq!(report.node_count, 2);
    }

    #[test]
    fn a_run_reports_the_shape_of_the_result() {
        let mut graph = graph_at(
            &[[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            vec![Edge { from: 0, to: 1 }],
        );
        let params = LayoutParams { speed: 100.0, area: 12.0, gravity: 0.0, three_d: false };
        let report = run(&mut graph, &params, LayoutChoice::FrCpu, 500).unwrap();

        assert_eq!(report.layout, "F-R cpu");
        assert_eq!(report.edge_count, 1);
        assert!(report.mean_edge_length.is_some());
        assert!(report.bounds.is_some());
        assert_eq!(report.k, params.k(2));
    }

    #[test]
    fn the_report_renders_every_field() {
        let mut graph = graph_at(
            &[[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            vec![Edge { from: 0, to: 1 }],
        );
        let report = run(&mut graph, &LayoutParams::default(), LayoutChoice::FrCpu, 1).unwrap();
        let text = report.to_string();

        for field in ["layout", "nodes", "edges", "steps", "elapsed", "mean edge", "extent"] {
            assert!(text.contains(field), "{field:?} missing from:\n{text}");
        }
    }

    #[test]
    fn an_edgeless_run_renders_without_dividing_by_zero() {
        let mut graph = graph_at(&[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], Vec::new());
        let report = run(&mut graph, &LayoutParams::default(), LayoutChoice::FrCpu, 5).unwrap();
        assert!(report.to_string().contains("n/a (no edges)"));
    }
}
