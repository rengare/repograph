//! The one place the CPU and GPU layout traits meet.
//!
//! `gv_layout::CpuLayout` mutates a `GraphData` in host memory;
//! `gv_layout_gpu::GpuLayout` records dispatches over a buffer the host never
//! reads. Neither trait can express the other without dragging in a dependency
//! it should not have, so the reconciliation is an enum here — the top of the
//! dependency graph, the only crate that already depends on both.
//!
//! A CPU layout therefore costs an upload of the node array each step; a GPU
//! layout costs nothing extra. That asymmetry is the honest one: it is exactly
//! what the original paid too, and it is part of what the benchmark measures.

use anyhow::Result;
use gv_gpu::{GpuContext, GraphBuffers};
use gv_graph::GraphData;
use gv_gui::LayoutChoice;
use gv_layout::{CpuLayout, LayoutParams};
use gv_layout_gpu::GpuLayout;

pub enum LayoutSlot {
    Cpu(Box<dyn CpuLayout>),
    Gpu(Box<dyn GpuLayout>),
}

impl LayoutSlot {
    /// Builds the layout behind a picker choice.
    ///
    /// Unlike the original's `ModelCreator::GetModelByType`, the match is
    /// exhaustive: [`LayoutChoice`] is an enum, so there is no index outside
    /// the known set and no uninitialised-pointer path.
    /// `gpu` is `None` before a window and adapter exist — `App` is constructed
    /// first and only initialises the device on `resumed` — and in headless
    /// runs, which have no device at all. The GPU choice degrades to the CPU
    /// reference there rather than failing, and `App::init` rebuilds the slot
    /// once the device is up.
    pub fn for_choice(
        choice: LayoutChoice,
        gpu: Option<(&GpuContext, &GraphBuffers)>,
    ) -> Result<Self> {
        Ok(match choice {
            LayoutChoice::FrCpu => Self::Cpu(Box::new(gv_layout::fr_cpu::FrCpuLayout)),
            LayoutChoice::FrGpu => match gpu {
                Some((context, buffers)) => {
                    Self::Gpu(Box::new(gv_layout_gpu::FrGpuLayout::new(context, buffers)?))
                }
                None => {
                    log::debug!("no device yet; the GPU layout starts on the CPU reference");
                    Self::Cpu(Box::new(gv_layout::fr_cpu::FrCpuLayout))
                }
            },
            LayoutChoice::FrBarnesHut => {
                Self::Cpu(Box::new(gv_layout::barnes_hut::BarnesHutLayout::default()))
            }
            LayoutChoice::FrGpuBarnesHut => match gpu {
                Some((context, buffers)) => {
                    Self::Gpu(Box::new(gv_layout_gpu::BhGpuLayout::new(context, buffers)?))
                }
                None => {
                    log::debug!("no device yet; the GPU tree starts on the CPU tree");
                    Self::Cpu(Box::new(gv_layout::barnes_hut::BarnesHutLayout::default()))
                }
            },
            // Still a `todo!()` body. Its `step` would abort the process, and
            // the picker is live, so a click on it must not take the window
            // down with it. This arm goes away when phase 5 lands.
            LayoutChoice::Random => Self::pending("the random baseline", 5),
        })
    }

    /// The CPU reference standing in for a layout that has not landed yet.
    fn pending(what: &str, phase: u32) -> Self {
        log::warn!("{what} arrives in phase {phase}; running the CPU reference instead");
        Self::Cpu(Box::new(gv_layout::fr_cpu::FrCpuLayout))
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu(layout) => layout.name(),
            Self::Gpu(layout) => layout.name(),
        }
    }

    /// True when a step leaves the authoritative node state on the GPU, so the
    /// host copy in `GraphData` is stale until read back.
    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::Gpu(_))
    }

    /// Advances one step. CPU layouts mutate `graph` and re-upload; GPU
    /// layouts record into `encoder` and leave `graph` untouched.
    pub fn step(
        &mut self,
        context: &GpuContext,
        buffers: &GraphBuffers,
        graph: &mut GraphData,
        encoder: &mut wgpu::CommandEncoder,
        params: &LayoutParams,
    ) -> Result<()> {
        match self {
            Self::Cpu(layout) => {
                layout.step(graph, params);
                // Without this the frame draws the previous step's positions.
                buffers.write_nodes(context, &graph.nodes);
                Ok(())
            }
            Self::Gpu(layout) => layout.record_step(encoder, &context.queue, params),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gv_layout::fr_cpu::FrCpuLayout;

    #[test]
    fn a_cpu_layout_reports_its_name_through_the_slot() {
        let slot = LayoutSlot::Cpu(Box::new(FrCpuLayout));
        assert_eq!(slot.name(), "F-R cpu");
        assert!(!slot.is_gpu());
    }

    #[test]
    fn cpu_layouts_are_object_safe() {
        // The enum only works if both traits can be boxed; this fails to
        // compile rather than fails at runtime if that ever stops holding.
        let layouts: Vec<Box<dyn CpuLayout>> = vec![
            Box::new(FrCpuLayout),
            Box::new(gv_layout::barnes_hut::BarnesHutLayout::default()),
            Box::new(gv_layout::random::RandomLayout),
        ];
        assert_eq!(layouts.len(), 3);
    }

    #[test]
    fn every_picker_choice_builds_a_layout() {
        // The exhaustiveness the original's switch lacked.
        for choice in LayoutChoice::ALL {
            let slot = LayoutSlot::for_choice(choice, None).expect("every choice must resolve");
            assert!(!slot.name().is_empty());
        }
    }

    #[test]
    fn the_gpu_choice_degrades_to_the_cpu_reference_without_a_device() {
        // App is constructed before `resumed` fires, and headless runs never
        // get a device at all. Neither is an error; `App::init` rebuilds the
        // slot for real once the adapter is up.
        let slot = LayoutSlot::for_choice(LayoutChoice::FrGpu, None).unwrap();
        assert!(!slot.is_gpu());
        assert_eq!(slot.name(), "F-R cpu");
    }

    #[test]
    fn no_picker_choice_can_abort_the_process() {
        // The window is live from phase 2 on, but Barnes-Hut and Random are
        // still `todo!()`. Stepping every choice here is what proves a click on
        // one of them cannot take the application down.
        let params = LayoutParams::default();
        for choice in LayoutChoice::ALL {
            let mut graph = gv_graph::testing::triangle();
            let LayoutSlot::Cpu(mut layout) = LayoutSlot::for_choice(choice, None).unwrap() else {
                panic!("{choice:?} should still be a CPU fallback");
            };
            layout.step(&mut graph, &params);
            assert!(
                gv_graph::testing::all_finite(&graph),
                "{choice:?} produced non-finite positions"
            );
        }
    }

    /// Device, buffers and one stepped frame's worth of scaffolding.
    fn on_device(graph: &GraphData) -> (GpuContext, GraphBuffers) {
        let context = pollster::block_on(GpuContext::new(None)).expect("adapter available");
        let buffers = GraphBuffers::upload(&context, graph).expect("upload");
        (context, buffers)
    }

    /// Runs one `LayoutSlot::step` and waits for the device to finish it.
    fn step_once(
        slot: &mut LayoutSlot,
        context: &GpuContext,
        buffers: &GraphBuffers,
        graph: &mut GraphData,
        params: &LayoutParams,
    ) {
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        slot.step(context, buffers, graph, &mut encoder, params)
            .expect("step");
        context.queue.submit([encoder.finish()]);
        context
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn a_cpu_step_uploads_the_node_array() {
        // The asymmetry this enum exists to make explicit: after a CPU step the
        // GPU buffer must match host memory, or the frame draws stale positions.
        let params = LayoutParams::default();
        let mut graph = gv_graph::testing::path(64);
        let (context, buffers) = on_device(&graph);

        let mut slot = LayoutSlot::Cpu(Box::new(FrCpuLayout));
        step_once(&mut slot, &context, &buffers, &mut graph, &params);

        let on_gpu = pollster::block_on(buffers.read_nodes(&context)).expect("readback");
        assert_eq!(on_gpu, graph.nodes, "the GPU buffer is a step behind the host");
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn a_gpu_step_does_not_touch_host_node_state() {
        let params = LayoutParams::default();
        let mut graph = gv_graph::testing::path(64);
        let (context, buffers) = on_device(&graph);
        let before = graph.nodes.clone();

        let mut slot = LayoutSlot::for_choice(LayoutChoice::FrGpu, Some((&context, &buffers)))
            .expect("the GPU layout builds when a device is available");
        assert!(slot.is_gpu(), "a device was available; this should be the GPU path");

        step_once(&mut slot, &context, &buffers, &mut graph, &params);

        assert_eq!(graph.nodes, before, "a GPU step wrote to host memory");
        // ...and the work really did happen, on the device.
        let on_gpu = pollster::block_on(buffers.read_nodes(&context)).expect("readback");
        assert_ne!(on_gpu, before, "the GPU step did not move anything");
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn the_gpu_tree_is_selected_when_a_device_exists() {
        // The arm that used to fall back to the CPU tree with a warning. The
        // fixture is over one workgroup because that is the size at which
        // anything about GPU pass ordering is testable at all.
        let params = LayoutParams::default();
        let mut graph = gv_graph::testing::path(1024);
        let (context, buffers) = on_device(&graph);
        let before = graph.nodes.clone();

        let mut slot = LayoutSlot::for_choice(
            LayoutChoice::FrGpuBarnesHut,
            Some((&context, &buffers)),
        )
        .expect("the GPU tree builds when a device is available");

        assert!(slot.is_gpu(), "a device was available; this should be the GPU path");
        assert_eq!(slot.name(), "F-R gpu barnes-hut");

        step_once(&mut slot, &context, &buffers, &mut graph, &params);

        assert_eq!(graph.nodes, before, "a GPU step wrote to host memory");
        let on_gpu = pollster::block_on(buffers.read_nodes(&context)).expect("readback");
        assert_ne!(on_gpu, before, "the GPU tree did not move anything");
    }
}
