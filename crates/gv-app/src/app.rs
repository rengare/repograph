//! The winit `ApplicationHandler` and the frame loop.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use gv_config::AppConfig;
use gv_gpu::{GpuContext, GraphBuffers};
use gv_graph::{GraphData, seed::SeedOptions};
use gv_gui::{GuiActions, GuiState, KindVisibility, LayoutChoice};
use gv_layout::LayoutParams;
use gv_render::{Camera, Renderer};
use glam::{Vec3, Vec4};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::LayoutSlot;
use crate::input::InputState;

/// Everything that only exists once a window and adapter do.
struct Active {
    window: Arc<Window>,
    context: GpuContext,
    renderer: Renderer,
    buffers: GraphBuffers,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
}

pub struct App {
    pub config: AppConfig,
    pub graph: GraphData,
    pub params: LayoutParams,
    pub choice: LayoutChoice,
    pub camera: Camera,
    pub layout: LayoutSlot,
    /// Options the initial scatter used, so `Reseed` can reproduce it.
    seed_options: SeedOptions,
    /// Search-panel state, persisted across frames.
    search: gv_gui::SearchState,
    /// The node currently enlarged by a search focus, and its original size, so
    /// the highlight can be undone when the selection moves.
    selected_highlight: Option<(usize, f32)>,
    /// `is_update_on` as of last frame, to catch the moment the layout stops.
    was_updating: bool,
    /// `show_labels` as of last frame, to catch the moment labels are switched on.
    was_show_labels: bool,
    /// Set when the layout has just stopped: pull the final GPU positions back to
    /// the host on the next frame so the (now visible) labels sit on the nodes.
    label_sync_pending: bool,
    /// Latest cursor position in physical pixels, for click picking.
    cursor: (f32, f32),
    /// The node whose inspector is open, if any.
    inspected: Option<usize>,
    /// The last edge clicked and which endpoint was last jumped to (0 = from,
    /// 1 = to), so re-clicking the same edge hops to the other end.
    edge_jump: Option<(usize, u8)>,

    active: Option<Active>,
    input: InputState,
    last_frame: Option<Instant>,
    frame_time: Duration,
    /// Wall-clock time the layout has been running, as the original's
    /// stopwatch reported. Accumulates only while `is_update_on`.
    layout_elapsed: Duration,
    /// Frames since the last throughput log, and when that log was.
    frames_since_report: u32,
    last_report: Option<Instant>,
}

/// How often the frame-rate log fires, when debug logging is on.
const REPORT_INTERVAL: Duration = Duration::from_secs(2);

/// How much a search-focused node is enlarged, so it stands out from its
/// neighbours without a shader change.
const HIGHLIGHT_SCALE: f32 = 2.5;

/// How far the rest of the graph fades when a node is selected — its RGB is
/// multiplied by this, so a low value pushes unselected nodes toward black
/// while the selection and its edge-connected neighbours stay at full 1.0.
const DIM_FACTOR: f32 = 0.18;

/// Distance the camera sits from a node it focuses on — the default framing.
const FOCUS_STANDOFF: f32 = 700.0;

/// Smallest clickable node radius in pixels — a distant node's sprite can shrink
/// below the cursor's precision, so picking is forgiving.
const PICK_MIN_RADIUS: f32 = 7.0;

/// How close, in pixels, the cursor must fall to an edge line to select it.
const PICK_EDGE_THRESHOLD: f32 = 6.0;

/// Cap on how many node labels are drawn per frame when labels are on, so a
/// large repository stays interactive; nearer/earlier nodes win.
const MAX_LABELS: usize = 800;

/// Points a label sits above its node's projected centre.
const LABEL_OFFSET: f32 = 12.0;

/// Projects each node's name to a screen position for the label overlay.
///
/// A free function taking the pieces it needs (rather than `&self`) so it can run
/// while `self.active` is mutably borrowed for the egui pass. Nodes behind the
/// camera or outside the view frustum are skipped; positions are returned in egui
/// points (physical pixels divided by `pixels_per_point`).
fn project_labels(
    camera: &Camera,
    graph: &GraphData,
    visible: &KindVisibility,
    width: f32,
    height: f32,
    pixels_per_point: f32,
) -> Vec<(egui::Pos2, String)> {
    let view_proj = camera.projection_matrix() * camera.view_matrix();
    let mut out = Vec::new();
    for (i, node) in graph.nodes.iter().enumerate() {
        let Some(meta) = graph.meta.get(i) else {
            break; // no metadata past here
        };
        // Hidden kinds carry no label, matching the scene.
        if meta.name.is_empty() || !visible.allows(meta.kind) {
            continue;
        }
        let clip = view_proj
            * Vec4::new(node.position[0], node.position[1], node.position[2], 1.0);
        if clip.w <= 0.0 {
            continue; // behind the camera
        }
        let inv = 1.0 / clip.w;
        let (ndc_x, ndc_y, ndc_z) = (clip.x * inv, clip.y * inv, clip.z * inv);
        // wgpu clips depth to 0..1; x/y to -1..1.
        if !(0.0..=1.0).contains(&ndc_z) || ndc_x.abs() > 1.0 || ndc_y.abs() > 1.0 {
            continue;
        }
        let px = (ndc_x * 0.5 + 0.5) * width;
        let py = (1.0 - (ndc_y * 0.5 + 0.5)) * height;
        out.push((
            egui::pos2(px / pixels_per_point, py / pixels_per_point - LABEL_OFFSET),
            meta.name.clone(),
        ));
        if out.len() >= MAX_LABELS {
            break;
        }
    }
    out
}

/// Distance from point `p` to the segment `a`–`b`, all in screen pixels.
fn point_segment_distance(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    let (abx, aby) = (b.0 - a.0, b.1 - a.1);
    let (apx, apy) = (p.0 - a.0, p.1 - a.1);
    let len_sq = abx * abx + aby * aby;
    // A degenerate edge (both endpoints projected to one point) is point-distance.
    let t = if len_sq > 0.0 {
        ((apx * abx + apy * aby) / len_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (cx, cy) = (a.0 + t * abx, a.1 + t * aby);
    ((p.0 - cx).powi(2) + (p.1 - cy).powi(2)).sqrt()
}

/// Paints the projected labels onto a foreground egui layer.
fn draw_labels(ctx: &egui::Context, labels: &[(egui::Pos2, String)]) {
    if labels.is_empty() {
        return;
    }
    // Background order so the panels paint over the labels — text tucked under
    // a window stays hidden instead of bleeding across it.
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Background,
        egui::Id::new("node-labels"),
    ));
    for (pos, text) in labels {
        // A dark outline keeps the white text legible over bright nodes.
        painter.text(
            *pos + egui::vec2(1.0, 1.0),
            egui::Align2::CENTER_BOTTOM,
            text,
            egui::FontId::proportional(12.0),
            egui::Color32::from_black_alpha(180),
        );
        painter.text(
            *pos,
            egui::Align2::CENTER_BOTTOM,
            text,
            egui::FontId::proportional(12.0),
            egui::Color32::WHITE,
        );
    }
}

impl App {
    pub fn new(
        config: AppConfig,
        graph: GraphData,
        choice: LayoutChoice,
        seed_options: SeedOptions,
    ) -> Result<Self> {
        let params = LayoutParams {
            three_d: config.graph_type_3d,
            ..Default::default()
        };

        Ok(Self {
            // No device until `resumed`; `init` rebuilds this once there is one.
            layout: LayoutSlot::for_choice(choice, None)?,
            camera: Camera::default(),
            config,
            graph,
            params,
            choice,
            seed_options,
            search: gv_gui::SearchState::default(),
            selected_highlight: None,
            was_updating: false,
            was_show_labels: false,
            label_sync_pending: false,
            cursor: (0.0, 0.0),
            inspected: None,
            edge_jump: None,
            active: None,
            input: InputState::default(),
            last_frame: None,
            frame_time: Duration::ZERO,
            layout_elapsed: Duration::ZERO,
            frames_since_report: 0,
            last_report: None,
        })
    }

    /// Rebuilds the active layout after the picker changes.
    pub fn set_layout(&mut self, choice: LayoutChoice) -> Result<()> {
        // Positions carry over, so switching algorithms continues from the
        // current layout rather than restarting — that is what makes comparing
        // them useful.
        self.sync_from_gpu()?;

        self.choice = choice;
        let gpu = self.active.as_ref().map(|a| (&a.context, &a.buffers));
        let slot = LayoutSlot::for_choice(choice, gpu)?;
        self.layout = slot;
        Ok(())
    }

    /// Logs frames per second every [`REPORT_INTERVAL`].
    ///
    /// The GUI already shows this, but a window is not always something you can
    /// read a number off — under a benchmark, over a remote session, or while
    /// the frame rate is low enough that the panel itself is hard to catch.
    /// Averaged over the interval rather than taken from the last frame, which
    /// on a loaded device is noisy.
    fn report_throughput(&mut self, now: Instant) {
        self.frames_since_report += 1;
        let since = self.last_report.get_or_insert(now);

        let elapsed = now - *since;
        if elapsed < REPORT_INTERVAL {
            return;
        }

        let frames = f64::from(self.frames_since_report);
        log::debug!(
            "{:.1} fps ({:.1} ms/frame) over {frames} frames",
            frames / elapsed.as_secs_f64(),
            elapsed.as_secs_f64() * 1000.0 / frames,
        );

        self.frames_since_report = 0;
        self.last_report = Some(now);
    }

    /// Brings host node state back up to date after a GPU layout has been
    /// running.
    ///
    /// A GPU step leaves the authoritative positions in the node buffer and
    /// never touches `self.graph`. Without this, switching to a CPU layout
    /// would resume from wherever the last CPU step left off and discard
    /// everything the GPU did. It stalls the pipeline, so it happens only on a
    /// layout change, never on the frame path.
    fn sync_from_gpu(&mut self) -> Result<()> {
        if !self.layout.is_gpu() {
            return Ok(());
        }
        let Some(active) = &self.active else {
            return Ok(());
        };
        self.graph.nodes = pollster::block_on(active.buffers.read_nodes(&active.context))?;
        Ok(())
    }

    pub fn apply_gui_actions(&mut self, actions: GuiActions) -> Result<()> {
        if actions.layout_changed {
            let choice = self.choice;
            self.set_layout(choice)?;
        }

        if actions.reseed {
            gv_graph::seed::scatter(&mut self.graph, self.seed_options);
            self.layout_elapsed = Duration::ZERO;
            if let Some(active) = &self.active {
                active.buffers.write_nodes(&active.context, &self.graph.nodes);
            }
        }

        if actions.reset_camera {
            self.camera.reset();
        }

        if let Some(idx) = actions.focus_node {
            self.focus_node(idx)?;
        }

        if actions.filters_changed {
            self.apply_visibility();
        }

        if actions.close_inspect {
            self.clear_inspection();
        }

        Ok(())
    }

    /// Closes the inspector and removes the highlight from the inspected node.
    fn clear_inspection(&mut self) {
        if let Some((prev, size)) = self.selected_highlight.take() {
            if prev < self.graph.nodes.len() {
                self.graph.nodes[prev].size = size;
                if let Some(active) = &self.active {
                    active.buffers.write_nodes(&active.context, &self.graph.nodes);
                }
            }
        }
        self.inspected = None;
        self.search.selected = None;
        self.edge_jump = None;
        self.clear_dim();
    }

    /// Dims every node and edge except `idx` and its edge-connected neighbours,
    /// bringing the selection to the foreground. The dim mask is a separate GPU
    /// buffer, so it does not disturb positions or the kind-visibility mask.
    fn apply_dim(&mut self, idx: usize) {
        let mask = self.dim_mask(idx);
        if let Some(active) = &self.active {
            active.renderer.set_dim(&mask);
        }
    }

    /// Builds the per-node dim mask for a selection of `idx`: `1.0` for the
    /// selected node and every node sharing an edge with it, `DIM_FACTOR` for
    /// the rest.
    fn dim_mask(&self, idx: usize) -> Vec<f32> {
        let count = self.graph.nodes.len();
        let mut mask = vec![DIM_FACTOR; count];
        if idx < count {
            mask[idx] = 1.0;
        }
        // Neighbours reachable across a single edge in either direction.
        for e in &self.graph.edges {
            let (from, to) = (e.from as usize, e.to as usize);
            if from == idx && to < count {
                mask[to] = 1.0;
            } else if to == idx && from < count {
                mask[from] = 1.0;
            }
        }
        mask
    }

    /// Restores full brightness to every node and edge.
    fn clear_dim(&mut self) {
        if let Some(active) = &self.active {
            active.renderer.set_dim(&vec![1.0; self.graph.nodes.len()]);
        }
    }

    /// Uploads the per-node visibility mask from the current kind filters, so the
    /// shader hides the unchecked kinds' nodes and their edges. A no-op without a
    /// sidecar (no kinds to filter) or a window.
    fn apply_visibility(&self) {
        let Some(active) = &self.active else {
            return;
        };
        if self.graph.meta.is_empty() {
            return;
        }
        let mask: Vec<u32> = self
            .graph
            .meta
            .iter()
            .map(|m| u32::from(self.search.visible.allows(m.kind)))
            .collect();
        active.renderer.set_visibility(&mask);
    }

    /// Keeps the Labels and Update toggles consistent, run once after the GUI.
    ///
    /// Labels are hidden while the layout runs (they would lag the live GPU
    /// positions), so switching Labels on while Update is running stops the
    /// layout; and any transition to stopped schedules a one-off position sync so
    /// the labels reappear exactly on the settled nodes.
    fn reconcile_labels_and_update(&mut self) {
        let show_labels = self.config.show_labels;

        if show_labels && !self.was_show_labels && self.config.is_update_on {
            self.config.is_update_on = false;
        }
        if self.was_updating && !self.config.is_update_on {
            self.label_sync_pending = true;
        }

        self.was_updating = self.config.is_update_on;
        self.was_show_labels = show_labels;
    }

    /// Enlarges node `idx`, undoing any previous highlight, and opens its
    /// inspector. Does not move the camera — used when the user clicks a node
    /// that is already on screen.
    ///
    /// A GPU layout keeps the authoritative positions in the node buffer, so we
    /// [`sync_from_gpu`](Self::sync_from_gpu) first — otherwise re-uploading the
    /// node buffer to apply the size bump would overwrite the device's positions
    /// with stale host ones and the graph would jump.
    fn highlight_node(&mut self, idx: usize) -> Result<()> {
        if idx >= self.graph.nodes.len() {
            return Ok(());
        }
        self.sync_from_gpu()?;

        // Undo the previous highlight before applying the new one.
        if let Some((prev, size)) = self.selected_highlight.take() {
            if prev < self.graph.nodes.len() {
                self.graph.nodes[prev].size = size;
            }
        }
        let original = self.graph.nodes[idx].size;
        self.selected_highlight = Some((idx, original));
        self.graph.nodes[idx].size = original * HIGHLIGHT_SCALE;
        self.search.selected = Some(idx);
        self.inspected = Some(idx);

        if let Some(active) = &self.active {
            active.buffers.write_nodes(&active.context, &self.graph.nodes);
        }
        Ok(())
    }

    /// Highlights node `idx` and centres the camera on it.
    fn focus_node(&mut self, idx: usize) -> Result<()> {
        if idx >= self.graph.nodes.len() {
            return Ok(());
        }
        self.highlight_node(idx)?;
        self.apply_dim(idx);
        let p = self.graph.nodes[idx].position;
        self.camera
            .focus_on(Vec3::new(p[0], p[1], p[2]), FOCUS_STANDOFF);
        Ok(())
    }

    /// Reacts to a left click: a node under the cursor opens its inspector; else
    /// an edge under the cursor jumps the camera to one endpoint, and re-clicking
    /// the same edge hops to the other end.
    fn handle_click(&mut self) -> Result<()> {
        let Some((width, height)) = self.active.as_ref().map(|a| a.renderer.size()) else {
            return Ok(());
        };
        // Picking reads host positions; refresh them if a GPU layout is running.
        if self.config.is_update_on && self.layout.is_gpu() {
            self.sync_from_gpu()?;
        }
        let (cx, cy) = self.cursor;
        let (w, h) = (width as f32, height as f32);

        if let Some(idx) = self.pick_node(cx, cy, w, h) {
            self.edge_jump = None;
            // Clicking the already-selected node toggles the selection off,
            // restoring full brightness; otherwise select it and dim the rest.
            if self.inspected == Some(idx) {
                self.clear_inspection();
            } else {
                self.highlight_node(idx)?;
                self.apply_dim(idx);
            }
        } else if let Some(edge) = self.pick_edge(cx, cy, w, h) {
            let endpoint = match self.edge_jump {
                Some((prev, ep)) if prev == edge => 1 - ep,
                _ => 0,
            };
            self.edge_jump = Some((edge, endpoint));
            let e = self.graph.edges[edge];
            let node = if endpoint == 0 { e.from } else { e.to } as usize;
            self.focus_node(node)?;
        }
        Ok(())
    }

    /// The front-most node whose sprite covers the cursor, or `None`. Honors the
    /// kind-visibility filter so hidden nodes are not pickable.
    fn pick_node(&self, cx: f32, cy: f32, w: f32, h: f32) -> Option<usize> {
        let view = self.camera.view_matrix();
        let proj = self.camera.projection_matrix();
        let mut best: Option<(usize, f32)> = None;

        for (i, node) in self.graph.nodes.iter().enumerate() {
            if !self.kind_visible(i) {
                continue;
            }
            let view_pos = view * Vec4::new(node.position[0], node.position[1], node.position[2], 1.0);
            if view_pos.z >= 0.0 {
                continue; // behind the camera
            }
            let clip = proj * view_pos;
            if clip.w <= 0.0 {
                continue;
            }
            let inv = 1.0 / clip.w;
            let (ndc_x, ndc_y, ndc_z) = (clip.x * inv, clip.y * inv, clip.z * inv);
            let px = (ndc_x * 0.5 + 0.5) * w;
            let py = (1.0 - (ndc_y * 0.5 + 0.5)) * h;
            // The sprite's on-screen radius, matching the node shader's sizing.
            let radius = (node.size * (500.0 / -view_pos.z) * 0.5).max(PICK_MIN_RADIUS);
            let dist = ((px - cx).powi(2) + (py - cy).powi(2)).sqrt();
            if dist <= radius && best.is_none_or(|(_, z)| ndc_z < z) {
                best = Some((i, ndc_z));
            }
        }
        best.map(|(i, _)| i)
    }

    /// The nearest edge line within [`PICK_EDGE_THRESHOLD`] of the cursor, or
    /// `None`. Edges with a hidden endpoint are not pickable.
    fn pick_edge(&self, cx: f32, cy: f32, w: f32, h: f32) -> Option<usize> {
        let view = self.camera.view_matrix();
        let proj = self.camera.projection_matrix();
        let project = |pos: [f32; 4]| -> Option<(f32, f32)> {
            let view_pos = view * Vec4::new(pos[0], pos[1], pos[2], 1.0);
            if view_pos.z >= 0.0 {
                return None;
            }
            let clip = proj * view_pos;
            if clip.w <= 0.0 {
                return None;
            }
            let inv = 1.0 / clip.w;
            Some((
                (clip.x * inv * 0.5 + 0.5) * w,
                (1.0 - (clip.y * inv * 0.5 + 0.5)) * h,
            ))
        };

        let mut best: Option<(usize, f32)> = None;
        for (i, edge) in self.graph.edges.iter().enumerate() {
            let (from, to) = (edge.from as usize, edge.to as usize);
            if !self.kind_visible(from) || !self.kind_visible(to) {
                continue;
            }
            let (Some(a), Some(b)) = (
                project(self.graph.nodes[from].position),
                project(self.graph.nodes[to].position),
            ) else {
                continue;
            };
            let dist = point_segment_distance((cx, cy), a, b);
            if dist <= PICK_EDGE_THRESHOLD && best.is_none_or(|(_, d)| dist < d) {
                best = Some((i, dist));
            }
        }
        best.map(|(i, _)| i)
    }

    /// Whether node `idx`'s kind is currently shown (always true without a sidecar).
    fn kind_visible(&self, idx: usize) -> bool {
        self.graph
            .meta
            .get(idx)
            .is_none_or(|m| self.search.visible.allows(m.kind))
    }

    fn init(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let attributes = Window::default_attributes()
            .with_title(&self.config.name)
            .with_inner_size(winit::dpi::LogicalSize::new(
                self.config.width,
                self.config.height,
            ));

        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .context("creating the window")?,
        );

        // The adapter must come from the instance that owns the surface, so
        // the instance is created here rather than inside GpuContext::new.
        let instance = wgpu::Instance::new(
            wgpu::InstanceDescriptor::new_with_display_handle_from_env(Box::new(window.clone())),
        );
        let surface = instance
            .create_surface(window.clone())
            .context("creating a surface for the window")?;
        let context = pollster::block_on(GpuContext::from_instance(instance, Some(&surface)))?;

        let size = window.inner_size();
        let buffers = GraphBuffers::upload(&context, &self.graph)?;
        let renderer = Renderer::new(
            &context,
            surface,
            &buffers,
            &self.config,
            size.width,
            size.height,
        )?;

        // Now that there is a device, the picker's choice can be honoured for
        // real — this is where a GPU layout actually gets built.
        self.layout = LayoutSlot::for_choice(self.choice, Some((&context, &buffers)))?;
        log::info!(
            "layout: {} ({})",
            self.layout.name(),
            if self.layout.is_gpu() { "on device" } else { "on host" }
        );

        self.camera.resize(size.width, size.height);

        let egui_context = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_context,
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &context.device,
            renderer.surface_format(),
            egui_wgpu::RendererOptions {
                depth_stencil_format: None,
                ..Default::default()
            },
        );

        self.active = Some(Active {
            window,
            context,
            renderer,
            buffers,
            egui_state,
            egui_renderer,
        });

        Ok(())
    }

    /// Steps the layout and draws one frame.
    ///
    /// Compute and draw share one `CommandEncoder`, so a GPU layout step and
    /// the frame it produces reach the queue in a single submission.
    pub fn frame(&mut self) -> Result<()> {
        if self.active.is_none() {
            return Ok(());
        }

        // The layout just stopped: refresh host node state from the GPU so the
        // labels — hidden while it ran — reappear on the settled positions.
        if self.label_sync_pending {
            self.sync_from_gpu()?;
            self.label_sync_pending = false;
        }

        let now = Instant::now();
        let delta = self.last_frame.map(|last| now - last).unwrap_or_default();
        self.last_frame = Some(now);
        self.frame_time = delta;
        if self.config.is_update_on {
            self.layout_elapsed += delta;
        }
        self.report_throughput(now);

        // Movement keys read the camera's velocity; keep it live with the setting.
        self.camera.velocity = Vec3::splat(self.config.move_speed);
        self.input.apply_to(&mut self.camera);

        let active = self.active.as_mut().expect("active checked above");
        let Some(frame) = active.renderer.acquire()? else {
            return Ok(());
        };
        let target = frame.texture.create_view(&Default::default());

        let mut encoder =
            active
                .context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("frame"),
                });

        if self.config.is_update_on {
            self.layout.step(
                &active.context,
                &active.buffers,
                &mut self.graph,
                &mut encoder,
                &self.params,
            )?;
        }

        active.renderer.draw_graph(
            &mut encoder,
            &target,
            &active.buffers,
            &self.camera,
            &self.config,
        );

        let actions = if self.config.show_gui {
            self.draw_gui(&mut encoder, &target)?
        } else {
            GuiActions::default()
        };
        self.reconcile_labels_and_update();

        let active = self.active.as_mut().expect("active checked above");
        active.context.queue.submit([encoder.finish()]);
        frame.present();
        active.window.request_redraw();

        self.apply_gui_actions(actions)?;
        Ok(())
    }

    /// Runs the egui pass. Loads the colour attachment rather than clearing,
    /// so the graph drawn above stays visible underneath.
    fn draw_gui(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
    ) -> Result<GuiActions> {
        let node_count = self.graph.node_count();
        let edge_count = self.graph.edge_count();
        let frame_time_ms = self.frame_time.as_secs_f32() * 1000.0;
        let layout_seconds = self.layout_elapsed.as_secs_f64();

        let active = self.active.as_mut().expect("active checked by caller");

        // Project node labels before the egui borrow starts, so the overlay is
        // drawn in the same frame as the panels. Hidden while the layout runs —
        // the live positions are on the GPU, so labels would lag the nodes.
        let labels = if self.config.show_labels
            && !self.config.is_update_on
            && !self.graph.meta.is_empty()
        {
            let (width, height) = active.renderer.size();
            let ppp = active.egui_state.egui_ctx().pixels_per_point();
            project_labels(
                &self.camera,
                &self.graph,
                &self.search.visible,
                width as f32,
                height as f32,
                ppp,
            )
        } else {
            Vec::new()
        };

        let input = active.egui_state.take_egui_input(&active.window);

        let mut actions = GuiActions::default();
        let output = active.egui_state.egui_ctx().run_ui(input, |ctx| {
            actions = gv_gui::draw(
                ctx,
                GuiState {
                    config: &mut self.config,
                    params: &mut self.params,
                    choice: &mut self.choice,
                    node_count,
                    edge_count,
                    frame_time_ms,
                    layout_seconds,
                    search: &mut self.search,
                    nodes_meta: &self.graph.meta,
                    inspected: self.inspected,
                },
            );
            draw_labels(ctx, &labels);
        });

        active
            .egui_state
            .handle_platform_output(&active.window, output.platform_output);

        let pixels_per_point = active.egui_state.egui_ctx().pixels_per_point();
        let jobs = active
            .egui_state
            .egui_ctx()
            .tessellate(output.shapes, pixels_per_point);

        for (id, delta) in &output.textures_delta.set {
            active
                .egui_renderer
                .update_texture(&active.context.device, &active.context.queue, *id, delta);
        }

        let (width, height) = active.renderer.size();
        let descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point,
        };

        active.egui_renderer.update_buffers(
            &active.context.device,
            &active.context.queue,
            encoder,
            &jobs,
            &descriptor,
        );

        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            active
                .egui_renderer
                .render(&mut pass.forget_lifetime(), &jobs, &descriptor);
        }

        for id in &output.textures_delta.free {
            active.egui_renderer.free_texture(id);
        }

        Ok(actions)
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.active.is_some() {
            return;
        }
        if let Err(error) = self.init(event_loop) {
            log::error!("{error:#}");
            event_loop.exit();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // Let egui claim the event first; if a panel has the pointer or
        // keyboard, the camera must not also act on it.
        let consumed = if let Some(active) = &mut self.active {
            let response = active.egui_state.on_window_event(&active.window, &event);
            if response.repaint {
                active.window.request_redraw();
            }
            response.consumed
        } else {
            false
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                if let Some(active) = &mut self.active {
                    active.renderer.resize(size.width, size.height);
                }
                self.camera.resize(size.width, size.height);
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if consumed {
                    return;
                }
                let PhysicalKey::Code(code) = event.physical_key else {
                    return;
                };
                if code == KeyCode::Escape {
                    event_loop.exit();
                    return;
                }
                self.input.set_key(code, event.state == ElementState::Pressed);
            }

            WindowEvent::MouseInput { button: MouseButton::Right, state, .. } => {
                self.input.set_looking(!consumed && state == ElementState::Pressed);
            }

            WindowEvent::MouseInput { button: MouseButton::Left, state, .. } => {
                if !consumed && state == ElementState::Pressed {
                    if let Err(error) = self.handle_click() {
                        log::error!("click picking failed: {error:#}");
                    }
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                // Let egui scroll its panels (the search results) first; only
                // zoom when the wheel wasn't over a panel.
                if consumed {
                    return;
                }
                // A pixel-precise trackpad reports far larger deltas than a
                // notched wheel, so scale it down to a comparable step count.
                let steps = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 50.0,
                };
                self.camera.zoom(steps * self.config.wheel_zoom_speed);
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = (position.x as f32, position.y as f32);
                self.input
                    .set_cursor(position.x as f32, position.y as f32, &mut self.camera);
            }

            WindowEvent::RedrawRequested => {
                if let Err(error) = self.frame() {
                    log::error!("frame failed: {error:#}");
                    event_loop.exit();
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(active) = &self.active {
            active.window.request_redraw();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_layout_choice_has_a_slot_variant() {
        // The exhaustive-match guarantee the original lacked: its
        // `GetModelByType` switch had no default, so an index outside 0..=3
        // returned an uninitialised pointer.
        for choice in LayoutChoice::ALL {
            assert!(LayoutSlot::for_choice(choice, None).is_ok());
        }
    }

    #[test]
    fn default_params_match_the_originals_starting_values() {
        let params = LayoutParams::default();
        assert_eq!(params.speed, 100.0);
        assert_eq!(params.area, 1000.0);
        assert_eq!(params.gravity, 1.0);
    }

    fn app_of(graph: GraphData) -> App {
        App::new(
            AppConfig::default(),
            graph,
            LayoutChoice::FrCpu,
            SeedOptions::default(),
        )
        .unwrap()
    }

    #[test]
    fn the_three_d_toggle_reaches_the_layout_params() {
        let config = AppConfig { graph_type_3d: false, ..Default::default() };
        let app = App::new(
            config,
            gv_graph::testing::triangle(),
            LayoutChoice::FrCpu,
            SeedOptions::default(),
        )
        .unwrap();
        assert!(!app.params.three_d);
    }

    #[test]
    fn switching_layout_preserves_node_positions() {
        // Swapping algorithms mid-run must continue from the current layout,
        // not restart from the seed — that is what makes comparing them useful.
        let mut app = app_of(gv_graph::testing::triangle());
        let before = app.graph.nodes.clone();

        app.set_layout(LayoutChoice::FrBarnesHut).unwrap();

        assert_eq!(app.graph.nodes, before);
        assert_eq!(app.choice, LayoutChoice::FrBarnesHut);
    }

    #[test]
    fn reseed_returns_the_graph_to_its_initial_scatter() {
        let mut app = app_of(gv_graph::testing::triangle());
        let seeded = app.graph.nodes.clone();

        // Disturb it, then ask for a reseed.
        app.graph.nodes[0].position = [999.0, 999.0, 999.0, 1.0];
        app.apply_gui_actions(GuiActions { reseed: true, ..Default::default() })
            .unwrap();

        assert_eq!(app.graph.nodes, seeded);
    }

    #[test]
    fn reseed_restarts_the_stopwatch() {
        let mut app = app_of(gv_graph::testing::triangle());
        app.layout_elapsed = Duration::from_secs(42);

        app.apply_gui_actions(GuiActions { reseed: true, ..Default::default() })
            .unwrap();

        assert_eq!(app.layout_elapsed, Duration::ZERO);
    }

    #[test]
    fn resetting_the_camera_clears_rotation_but_not_position() {
        let mut app = app_of(gv_graph::testing::triangle());
        app.camera.forward();
        let moved = app.camera.position;
        app.camera.rotation.y = 33.0;

        app.apply_gui_actions(GuiActions { reset_camera: true, ..Default::default() })
            .unwrap();

        assert_eq!(app.camera.rotation.y, 0.0);
        assert_eq!(app.camera.position, moved);
    }

    #[test]
    fn focusing_a_node_enlarges_it_and_moves_the_camera() {
        let mut app = app_of(gv_graph::testing::triangle());
        let original = app.graph.nodes[1].size;

        app.apply_gui_actions(GuiActions { focus_node: Some(1), ..Default::default() })
            .unwrap();

        assert_eq!(app.graph.nodes[1].size, original * HIGHLIGHT_SCALE);
        assert_eq!(app.selected_highlight, Some((1, original)));
        // Camera cleared its rotation to frame the node.
        assert_eq!(app.camera.rotation, glam::Vec3::ZERO);
    }

    #[test]
    fn refocusing_restores_the_previous_highlight() {
        let mut app = app_of(gv_graph::testing::triangle());
        let size0 = app.graph.nodes[0].size;
        let size2 = app.graph.nodes[2].size;

        app.apply_gui_actions(GuiActions { focus_node: Some(0), ..Default::default() })
            .unwrap();
        app.apply_gui_actions(GuiActions { focus_node: Some(2), ..Default::default() })
            .unwrap();

        // Node 0 back to its original size, node 2 now the enlarged one.
        assert_eq!(app.graph.nodes[0].size, size0);
        assert_eq!(app.graph.nodes[2].size, size2 * HIGHLIGHT_SCALE);
        assert_eq!(app.selected_highlight, Some((2, size2)));
    }

    #[test]
    fn dim_mask_lights_the_selection_and_its_edge_neighbours_only() {
        use gv_graph::Edge;

        let mut app = app_of(gv_graph::testing::triangle());
        // A path 0 — 1 — 2 (node 0 not connected to node 2 directly).
        app.graph.edges = vec![
            Edge { from: 0, to: 1 },
            Edge { from: 1, to: 2 },
        ];

        let mask = app.dim_mask(1);

        // Selected node 1 and both its edge neighbours (0 and 2) are lit.
        assert_eq!(mask[0], 1.0);
        assert_eq!(mask[1], 1.0);
        assert_eq!(mask[2], 1.0);

        // Selecting an endpoint lights only it and its single neighbour.
        let mask0 = app.dim_mask(0);
        assert_eq!(mask0[0], 1.0);
        assert_eq!(mask0[1], 1.0);
        assert_eq!(mask0[2], DIM_FACTOR);
    }

    #[test]
    fn labels_project_visible_nodes_and_skip_ones_behind_the_camera() {
        use gv_graph::{Node, NodeCategory, NodeMeta};

        let mut camera = Camera::default();
        camera.resize(800, 600);

        let graph = GraphData {
            nodes: vec![
                Node { position: [0.0, 0.0, 0.0, 1.0], ..Default::default() },
                // Past the camera (which sits at z=-700 looking toward +z).
                Node { position: [0.0, 0.0, 5000.0, 1.0], ..Default::default() },
            ],
            meta: vec![
                NodeMeta { name: "origin".into(), kind: NodeCategory::File, ..Default::default() },
                NodeMeta { name: "behind".into(), kind: NodeCategory::File, ..Default::default() },
            ],
            ..Default::default()
        };

        let all = KindVisibility::default();
        let labels = project_labels(&camera, &graph, &all, 800.0, 600.0, 1.0);

        // Only the origin node is in front of the camera.
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0].1, "origin");

        // Hiding files removes both labels.
        let no_files = KindVisibility { file: false, ..Default::default() };
        assert!(project_labels(&camera, &graph, &no_files, 800.0, 600.0, 1.0).is_empty());
        // It projects to roughly the centre of an 800x600 view.
        assert!((labels[0].0.x - 400.0).abs() < 1.0, "x={}", labels[0].0.x);
        assert!((labels[0].0.y - (300.0 - LABEL_OFFSET)).abs() < 1.0, "y={}", labels[0].0.y);
    }

    #[test]
    fn enabling_labels_while_updating_stops_the_layout_and_schedules_a_sync() {
        let mut app = app_of(gv_graph::testing::triangle());
        app.was_updating = true;
        app.config.is_update_on = true;
        app.was_show_labels = false;
        app.config.show_labels = true; // just ticked Labels

        app.reconcile_labels_and_update();

        assert!(!app.config.is_update_on, "layout should stop");
        assert!(app.label_sync_pending, "positions should be pulled");
    }

    #[test]
    fn stopping_the_layout_schedules_a_label_sync() {
        let mut app = app_of(gv_graph::testing::triangle());
        app.was_updating = true;
        app.config.is_update_on = false; // user unticked Update
        app.was_show_labels = true;
        app.config.show_labels = true;

        app.reconcile_labels_and_update();

        assert!(app.label_sync_pending);
    }

    #[test]
    fn starting_the_layout_with_labels_already_on_is_allowed() {
        // Labels just get hidden while it runs; Update is not forced back off.
        let mut app = app_of(gv_graph::testing::triangle());
        app.was_updating = false;
        app.config.is_update_on = true; // just ticked Update
        app.was_show_labels = true;
        app.config.show_labels = true; // labels were already on (not a rising edge)

        app.reconcile_labels_and_update();

        assert!(app.config.is_update_on, "update should be allowed to run");
        assert!(!app.label_sync_pending);
    }

    #[test]
    fn point_segment_distance_handles_interior_and_endpoints() {
        // Perpendicular from a point above the middle of a horizontal segment.
        assert!((point_segment_distance((0.0, 5.0), (-10.0, 0.0), (10.0, 0.0)) - 5.0).abs() < 1e-4);
        // Beyond an endpoint clamps to that endpoint.
        assert!((point_segment_distance((13.0, 0.0), (-10.0, 0.0), (10.0, 0.0)) - 3.0).abs() < 1e-4);
        // On the segment is zero.
        assert!(point_segment_distance((0.0, 0.0), (-10.0, 0.0), (10.0, 0.0)) < 1e-4);
    }

    #[test]
    fn pick_node_hits_the_sprite_under_the_cursor() {
        let graph = GraphData {
            nodes: vec![gv_graph::Node {
                position: [0.0, 0.0, 0.0, 1.0],
                size: 20.0,
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut app = app_of(graph);
        app.camera.resize(800, 600);

        // The node sits at the origin, which the default camera frames at the
        // screen centre.
        assert_eq!(app.pick_node(400.0, 300.0, 800.0, 600.0), Some(0));
        // A click in the corner misses it.
        assert_eq!(app.pick_node(50.0, 50.0, 800.0, 600.0), None);
    }

    #[test]
    fn pick_edge_hits_the_line_between_two_nodes() {
        let graph = GraphData {
            nodes: vec![
                gv_graph::Node { position: [0.0, 0.0, 0.0, 1.0], size: 5.0, ..Default::default() },
                gv_graph::Node { position: [300.0, 0.0, 0.0, 1.0], size: 5.0, ..Default::default() },
            ],
            edges: vec![gv_graph::Edge { from: 0, to: 1 }],
            ..Default::default()
        };
        let mut app = app_of(graph);
        app.camera.resize(800, 600);

        // Node 0 is at the screen centre and is an endpoint of the only edge.
        assert_eq!(app.pick_edge(400.0, 300.0, 800.0, 600.0), Some(0));
        // Far above the line: no edge.
        assert_eq!(app.pick_edge(400.0, 20.0, 800.0, 600.0), None);
    }

    #[test]
    fn labels_are_empty_without_metadata() {
        let camera = Camera::default();
        let graph = gv_graph::testing::triangle(); // no meta
        let all = KindVisibility::default();
        assert!(project_labels(&camera, &graph, &all, 800.0, 600.0, 1.0).is_empty());
    }

    #[test]
    fn focusing_an_out_of_range_node_is_ignored() {
        let mut app = app_of(gv_graph::testing::triangle());
        app.apply_gui_actions(GuiActions { focus_node: Some(999), ..Default::default() })
            .unwrap();
        assert_eq!(app.selected_highlight, None);
    }

    #[test]
    fn a_frame_without_a_window_is_a_no_op() {
        // `frame` is driven by RedrawRequested, which cannot arrive before
        // `resumed`, but returning Ok rather than panicking keeps the
        // invariant local rather than spread across the event handler.
        let mut app = app_of(gv_graph::testing::triangle());
        assert!(app.frame().is_ok());
    }
}
