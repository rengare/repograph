//! The egui panels, ported from the original's ImGui windows.
//!
//! Depends on `egui` alone — not on `egui-winit`, `egui-wgpu`, `wgpu` or
//! `winit`. Platform and paint integration live in `gv-app`, which keeps the
//! panel definitions testable and makes it obvious that the GUI only reads and
//! writes plain state.

use gv_config::AppConfig;
use gv_graph::{NodeCategory, NodeMeta};
use gv_layout::LayoutParams;

/// Which node kinds are drawn in the scene. Unchecking one hides its nodes, their
/// edges, and their labels. All visible by default.
#[derive(Debug, Clone, Copy)]
pub struct KindVisibility {
    pub dir: bool,
    pub file: bool,
    pub doc: bool,
    pub section: bool,
    pub symbol: bool,
}

impl Default for KindVisibility {
    fn default() -> Self {
        Self {
            dir: true,
            file: true,
            doc: true,
            section: true,
            symbol: true,
        }
    }
}

impl KindVisibility {
    /// Whether nodes of `kind` are currently drawn. Unknown kinds always show.
    pub fn allows(&self, kind: NodeCategory) -> bool {
        match kind {
            NodeCategory::Dir => self.dir,
            NodeCategory::File => self.file,
            NodeCategory::Doc => self.doc,
            NodeCategory::Section => self.section,
            NodeCategory::Symbol => self.symbol,
            NodeCategory::Unknown => true,
        }
    }
}

/// Persistent state for the node search panel, owned by the app across frames.
#[derive(Debug, Default, Clone)]
pub struct SearchState {
    /// Current name/path query.
    pub query: String,
    /// Restrict results to one kind, or `None` for all.
    pub kind_filter: Option<NodeCategory>,
    /// Index of the node last focused from the panel, so it can be marked.
    pub selected: Option<usize>,
    /// Which kinds are drawn in the scene.
    pub visible: KindVisibility,
}

/// How many matches the panel lists before it stops, to keep the frame cheap on
/// a large repository.
const MAX_RESULTS: usize = 60;

/// The kinds offered in the search filter, with their labels.
const FILTER_KINDS: [(Option<NodeCategory>, &str); 6] = [
    (None, "all"),
    (Some(NodeCategory::Dir), "dir"),
    (Some(NodeCategory::File), "file"),
    (Some(NodeCategory::Doc), "doc"),
    (Some(NodeCategory::Section), "section"),
    (Some(NodeCategory::Symbol), "symbol"),
];

/// Which algorithm the picker currently has selected.
///
/// The original keyed this on a bare `int` (0, 1, 2, 3) shared between the
/// radio buttons and a `switch` with no default case, so an unexpected value
/// returned an uninitialised pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LayoutChoice {
    #[default]
    FrGpu,
    FrGpuBarnesHut,
    FrCpu,
    FrBarnesHut,
    Random,
}

impl LayoutChoice {
    pub const ALL: [Self; 5] = [
        Self::FrGpu,
        Self::FrGpuBarnesHut,
        Self::FrCpu,
        Self::FrBarnesHut,
        Self::Random,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::FrGpu => "F-R gpu",
            Self::FrGpuBarnesHut => "F-R gpu barnes-hut",
            Self::FrCpu => "F-R cpu",
            Self::FrBarnesHut => "F-R cpu barnes-hut",
            Self::Random => "Random",
        }
    }

    /// True if this layout runs as compute passes rather than on the host.
    pub fn is_gpu(self) -> bool {
        matches!(self, Self::FrGpu | Self::FrGpuBarnesHut)
    }

    /// True if this layout approximates repulsion with a tree rather than
    /// summing every pair.
    pub fn is_approximate(self) -> bool {
        matches!(self, Self::FrGpuBarnesHut | Self::FrBarnesHut)
    }

    /// Whether this layout actually runs what its label says.
    ///
    /// The picker offers every variant so the set is visible and the CLI is
    /// stable, but two are not written yet and quietly run something else.
    /// Keeping that on the type — rather than only in a `log::warn!` nobody
    /// reads — is what stops the picker from claiming to do something it does
    /// not. Delete an arm as its implementation lands.
    pub fn is_implemented(self) -> bool {
        !matches!(self, Self::Random)
    }

    /// What actually runs when [`Self::is_implemented`] is false.
    pub fn falls_back_to(self) -> Self {
        match self {
            Self::Random => Self::FrCpu,
            implemented => implemented,
        }
    }

    /// Short name accepted on the command line.
    pub fn slug(self) -> &'static str {
        match self {
            Self::FrGpu => "gpu",
            Self::FrGpuBarnesHut => "gpu-barnes-hut",
            Self::FrCpu => "cpu",
            Self::FrBarnesHut => "barnes-hut",
            Self::Random => "random",
        }
    }
}

impl std::str::FromStr for LayoutChoice {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|choice| choice.slug().eq_ignore_ascii_case(text))
            .ok_or_else(|| {
                let known: Vec<_> = Self::ALL.iter().map(|c| c.slug()).collect();
                format!("unknown layout {text:?}; expected one of {}", known.join(", "))
            })
    }
}

/// Everything the panels read or write, in one borrow.
pub struct GuiState<'a> {
    pub config: &'a mut AppConfig,
    pub params: &'a mut LayoutParams,
    pub choice: &'a mut LayoutChoice,
    pub node_count: usize,
    pub edge_count: usize,
    pub frame_time_ms: f32,
    /// Wall-clock seconds the layout has been running, as the original's
    /// stopwatch reported.
    pub layout_seconds: f64,
    /// Search panel state, persisted by the app between frames.
    pub search: &'a mut SearchState,
    /// Per-node knowledge-graph metadata. Empty when the graph was loaded from a
    /// plain edge list, in which case the search panel is hidden.
    pub nodes_meta: &'a [NodeMeta],
    /// The node the user clicked, if any — shown in the inspector panel.
    pub inspected: Option<usize>,
}

/// Set when a panel asks the app to do something it cannot do itself.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct GuiActions {
    /// The algorithm picker changed; rebuild the active layout.
    pub layout_changed: bool,
    /// Re-scatter the nodes and start over.
    pub reseed: bool,
    pub reset_camera: bool,
    /// A search result was clicked; focus the camera on this node index and
    /// highlight it.
    pub focus_node: Option<usize>,
    /// A kind-visibility checkbox changed; rebuild the scene visibility mask.
    pub filters_changed: bool,
    /// The inspector's Close button was clicked; clear the inspected node.
    pub close_inspect: bool,
}

/// Bounds the original clamped `speed` into inside its `InputFloat` handler.
pub const SPEED_RANGE: std::ops::RangeInclusive<f32> = 0.1..=1000.0;
/// `area` must stay positive: `k` is proportional to it, and a negative `k`
/// inverts every force. The original clamped this too.
pub const MIN_AREA: f32 = 0.1;

/// Width of each edge-anchored panel window.
const PANEL_WIDTH: f32 = 290.0;
/// Inset of each panel from the window edge it anchors to.
const PANEL_MARGIN: f32 = 8.0;
/// Vertical padding placed around each section so they read as evenly spaced.
const SECTION_GAP: f32 = 12.0;

/// A titled block inside a panel, with even padding above and below and a
/// divider, so stacked sections space consistently.
fn section(ui: &mut egui::Ui, title: &str, add: impl FnOnce(&mut egui::Ui)) {
    ui.add_space(SECTION_GAP);
    ui.heading(title);
    ui.add_space(SECTION_GAP * 0.5);
    add(ui);
    ui.add_space(SECTION_GAP);
    ui.separator();
}

/// Draws every panel and reports what the user asked for.
///
/// The GUI is sorted into two edge-anchored windows — configuration pinned to the
/// right edge, browsing to the left — leaving the centre clear for the graph. Each
/// stacks titled sections with uniform spacing.
pub fn draw(ctx: &egui::Context, state: GuiState<'_>) -> GuiActions {
    let mut actions = GuiActions::default();
    let GuiState {
        config,
        params,
        choice,
        node_count,
        edge_count,
        frame_time_ms,
        layout_seconds,
        search,
        nodes_meta,
        inspected,
    } = state;

    // Right edge: configuration — system, forces, algorithm.
    egui::Window::new("Configuration")
        .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-PANEL_MARGIN, PANEL_MARGIN))
        .resizable(false)
        .default_width(PANEL_WIDTH)
        .show(ctx, |ui| {
            section(ui, "System settings", |ui| {
                ui.add(egui::Slider::new(&mut config.red, 0.0..=255.0).text("red"));
                ui.add(egui::Slider::new(&mut config.green, 0.0..=255.0).text("green"));
                ui.add(egui::Slider::new(&mut config.blue, 0.0..=255.0).text("blue"));
                ui.add_space(SECTION_GAP * 0.5);
                if ui.button("Reset camera").clicked() {
                    actions.reset_camera = true;
                }
            });

            section(ui, "Graph settings", |ui| {
                // Clamped after every edit, not only on commit: an intermediate
                // value still feeds the next layout step.
                if ui.add(egui::DragValue::new(&mut params.speed).prefix("speed: ")).changed() {
                    params.speed = params.speed.clamp(*SPEED_RANGE.start(), *SPEED_RANGE.end());
                }
                if ui.add(egui::DragValue::new(&mut params.area).prefix("area: ")).changed() {
                    params.area = params.area.max(MIN_AREA);
                }
                ui.add(egui::DragValue::new(&mut params.gravity).prefix("gravity: "));
            });

            section(ui, "Algorithms", |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.checkbox(&mut config.show_edge, "Show edge");
                    ui.checkbox(&mut config.is_update_on, "Update");
                    if ui.checkbox(&mut config.graph_type_3d, "3d").changed() {
                        params.three_d = config.graph_type_3d;
                    }
                    ui.checkbox(&mut config.show_labels, "Labels");
                });
                ui.separator();
                for option in LayoutChoice::ALL {
                    // An unimplemented layout still runs — it falls back — so the
                    // label must say so rather than imply it executes.
                    let label = if option.is_implemented() {
                        option.label().to_string()
                    } else {
                        format!(
                            "{} — not built, runs {}",
                            option.label(),
                            option.falls_back_to().label()
                        )
                    };
                    if ui.radio_value(choice, option, label).clicked() {
                        actions.layout_changed = true;
                    }
                }
                ui.add_space(SECTION_GAP * 0.5);
                if ui.button("Reseed").clicked() {
                    actions.reseed = true;
                }
            });
        });

    // Left edge: browsing — stats, search, inspector.
    egui::Window::new("Browser")
        .anchor(egui::Align2::LEFT_TOP, egui::vec2(PANEL_MARGIN, PANEL_MARGIN))
        .resizable(false)
        .default_width(PANEL_WIDTH)
        .show(ctx, |ui| {
            section(ui, "Graph", |ui| {
                ui.label(format!("nodes count: {node_count}"));
                ui.label(format!("edges count: {edge_count}"));
                ui.label(format!("algorithm duration: {layout_seconds:.0} s"));
                ui.label(format!(
                    "{frame_time_ms:.3} ms/frame ({:.1} FPS)",
                    if frame_time_ms > 0.0 { 1000.0 / frame_time_ms } else { 0.0 }
                ));
            });

            // The search section only exists for graphs loaded with a
            // knowledge-graph sidecar — an edge list has no names to search.
            if !nodes_meta.is_empty() {
                section(ui, "Search", |ui| {
                    draw_search(ui, search, nodes_meta, &mut actions);
                });
            }

            if let Some(meta) = inspected.and_then(|i| nodes_meta.get(i)) {
                section(ui, "Node", |ui| {
                    draw_inspector(ui, meta, &mut actions);
                });
            }
        });

    actions
}

/// Draws the node search controls: a query box, a kind filter, per-kind scene
/// visibility, and a scrollable list of matches that focus the camera when
/// clicked. Rendered inside the left panel's "Search" section.
fn draw_search(
    ui: &mut egui::Ui,
    search: &mut SearchState,
    nodes_meta: &[NodeMeta],
    actions: &mut GuiActions,
) {
    ui.horizontal(|ui| {
        ui.label("find:");
        ui.text_edit_singleline(&mut search.query);
        if ui.button("✕").on_hover_text("clear").clicked() {
            search.query.clear();
        }
    });

    ui.horizontal_wrapped(|ui| {
        for (kind, label) in FILTER_KINDS {
            ui.selectable_value(&mut search.kind_filter, kind, label);
        }
    });

    // Per-kind scene visibility: unticking hides those nodes, edges, labels.
    ui.horizontal_wrapped(|ui| {
        ui.label("show:");
        let v = &mut search.visible;
        let mut changed = false;
        changed |= ui.checkbox(&mut v.dir, "dir").changed();
        changed |= ui.checkbox(&mut v.file, "file").changed();
        changed |= ui.checkbox(&mut v.doc, "doc").changed();
        changed |= ui.checkbox(&mut v.section, "sec").changed();
        changed |= ui.checkbox(&mut v.symbol, "sym").changed();
        if changed {
            actions.filters_changed = true;
        }
    });
    ui.separator();

    let matches = search_matches(nodes_meta, &search.query, search.kind_filter);
    ui.label(format!(
        "{} match{}",
        matches.len(),
        if matches.len() == 1 { "" } else { "es" }
    ));

    egui::ScrollArea::vertical()
        .max_height(240.0)
        .show(ui, |ui| {
            for &i in matches.iter().take(MAX_RESULTS) {
                let meta = &nodes_meta[i];
                let selected = search.selected == Some(i);
                let text = format!("[{}] {}", meta.kind.tag_short(), meta.path);
                if ui.selectable_label(selected, text).clicked() {
                    search.selected = Some(i);
                    actions.focus_node = Some(i);
                }
            }
            if matches.len() > MAX_RESULTS {
                ui.weak(format!("… {} more", matches.len() - MAX_RESULTS));
            }
        });
}

/// Draws the metadata for a clicked node inside the left panel's "Node" section.
fn draw_inspector(ui: &mut egui::Ui, meta: &NodeMeta, actions: &mut GuiActions) {
    ui.horizontal(|ui| {
        ui.strong(&meta.name);
        // Prefer the specific symbol sub-kind (fn/struct/…) over the coarse kind.
        let tag = meta.symbol_kind.as_deref().unwrap_or_else(|| meta.kind.tag_short());
        ui.label(format!("[{tag}]"));
    });

    egui::Grid::new("node-meta").num_columns(2).show(ui, |ui| {
        ui.label("id:");
        ui.label(&meta.id);
        ui.end_row();
        ui.label("path:");
        ui.label(&meta.path);
        ui.end_row();
        if let Some(container) = &meta.container {
            ui.label("in:");
            ui.label(container);
            ui.end_row();
        }
        if let Some((start, end)) = meta.span {
            ui.label("lines:");
            ui.label(format!("{start}–{end}"));
            ui.end_row();
        }
    });

    if let Some(sig) = &meta.signature {
        ui.add_space(SECTION_GAP * 0.5);
        ui.label("signature:");
        ui.code(sig);
    }

    if let Some(doc) = &meta.doc {
        ui.add_space(SECTION_GAP * 0.5);
        ui.label("doc:");
        ui.label(egui::RichText::new(doc).italics().weak());
    }

    ui.add_space(SECTION_GAP * 0.5);
    if ui.button("Close").clicked() {
        actions.close_inspect = true;
    }
}

/// Indices of nodes whose name or path contains `query` (case-insensitive) and
/// match `kind_filter`, ranked with name-prefix hits first. An empty query with
/// no filter returns nothing, so the list starts empty rather than dumping the
/// whole repository.
fn search_matches(
    nodes_meta: &[NodeMeta],
    query: &str,
    kind_filter: Option<NodeCategory>,
) -> Vec<usize> {
    let needle = query.trim().to_lowercase();
    if needle.is_empty() && kind_filter.is_none() {
        return Vec::new();
    }
    let mut hits: Vec<usize> = nodes_meta
        .iter()
        .enumerate()
        .filter(|(_, m)| kind_filter.is_none_or(|k| m.kind == k))
        .filter(|(_, m)| {
            needle.is_empty()
                || m.name.to_lowercase().contains(&needle)
                || m.path.to_lowercase().contains(&needle)
        })
        .map(|(i, _)| i)
        .collect();

    hits.sort_by_key(|&i| {
        let name = nodes_meta[i].name.to_lowercase();
        if name == needle {
            0
        } else if name.starts_with(&needle) {
            1
        } else {
            2
        }
    });
    hits
}

/// Applies the clamps [`draw`] enforces, so they can be checked without a
/// rendering context.
pub fn clamp_params(params: &mut LayoutParams) {
    params.speed = params.speed.clamp(*SPEED_RANGE.start(), *SPEED_RANGE.end());
    params.area = params.area.max(MIN_AREA);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn exactly_the_device_layouts_report_as_gpu() {
        let gpu: Vec<_> = LayoutChoice::ALL.iter().filter(|c| c.is_gpu()).collect();
        assert_eq!(
            gpu,
            vec![&LayoutChoice::FrGpu, &LayoutChoice::FrGpuBarnesHut]
        );
    }

    #[test]
    fn an_unimplemented_layout_falls_back_to_an_implemented_one() {
        // Otherwise the fallback chain could loop, or point at another stub.
        for choice in LayoutChoice::ALL {
            let target = choice.falls_back_to();
            assert!(
                target.is_implemented(),
                "{choice:?} falls back to {target:?}, which is not implemented either"
            );
            if choice.is_implemented() {
                assert_eq!(target, choice, "{choice:?} is implemented but redirects");
            }
        }
    }

    #[test]
    fn exactly_the_tree_layouts_report_as_approximate() {
        // The distinction the picker has to make visible: two of these four
        // compute the same forces by different means, and two approximate them.
        let approximate: Vec<_> = LayoutChoice::ALL
            .iter()
            .filter(|c| c.is_approximate())
            .collect();
        assert_eq!(
            approximate,
            vec![&LayoutChoice::FrGpuBarnesHut, &LayoutChoice::FrBarnesHut]
        );
    }

    #[test]
    fn each_repulsion_strategy_is_offered_on_both_processors() {
        // Brute force and Barnes-Hut, each on CPU and GPU — the four-way grid
        // the benchmark compares. A missing pairing would make one of those
        // comparisons unavailable from the picker.
        for approximate in [false, true] {
            for gpu in [false, true] {
                assert!(
                    LayoutChoice::ALL
                        .iter()
                        .any(|c| c.is_gpu() == gpu && c.is_approximate() == approximate),
                    "no layout with gpu={gpu}, approximate={approximate}"
                );
            }
        }
    }

    #[test]
    fn all_lists_every_variant_exactly_once() {
        // Guards against a new variant being added without extending ALL,
        // which would silently drop it from the picker.
        let mut slugs: Vec<_> = LayoutChoice::ALL.iter().map(|c| c.slug()).collect();
        slugs.sort_unstable();
        slugs.dedup();
        assert_eq!(slugs.len(), LayoutChoice::ALL.len());
    }

    #[test]
    fn slug_round_trips_through_from_str() {
        for choice in LayoutChoice::ALL {
            assert_eq!(LayoutChoice::from_str(choice.slug()), Ok(choice));
        }
    }

    #[test]
    fn from_str_is_case_insensitive() {
        assert_eq!(LayoutChoice::from_str("GPU"), Ok(LayoutChoice::FrGpu));
        assert_eq!(LayoutChoice::from_str("Barnes-Hut"), Ok(LayoutChoice::FrBarnesHut));
    }

    #[test]
    fn from_str_rejects_unknown_names_and_lists_the_valid_ones() {
        let error = LayoutChoice::from_str("kd-tree").unwrap_err();
        assert!(error.contains("kd-tree"), "{error}");
        assert!(error.contains("barnes-hut"), "{error}");
    }

    #[test]
    fn labels_are_distinct() {
        let mut labels: Vec<_> = LayoutChoice::ALL.iter().map(|c| c.label()).collect();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels.len(), LayoutChoice::ALL.len());
    }

    #[test]
    fn default_is_the_gpu_layout_the_project_exists_for() {
        assert_eq!(LayoutChoice::default(), LayoutChoice::FrGpu);
    }

    #[test]
    fn gui_actions_default_to_doing_nothing() {
        let actions = GuiActions::default();
        assert!(!actions.layout_changed);
        assert!(!actions.reseed);
        assert!(!actions.reset_camera);
    }

    /// Runs the panels headlessly. `egui::Context::run_ui` needs no window, so
    /// the real `draw` is exercised — not a stand-in.
    fn run_draw(config: &mut AppConfig, params: &mut LayoutParams, choice: &mut LayoutChoice)
    -> GuiActions {
        let ctx = egui::Context::default();
        let mut actions = GuiActions::default();
        let mut search = SearchState::default();
        let _ = ctx.run_ui(egui::RawInput::default(), |ctx| {
            actions = draw(
                ctx,
                GuiState {
                    config,
                    params,
                    choice,
                    node_count: 10,
                    edge_count: 20,
                    frame_time_ms: 16.0,
                    layout_seconds: 1.0,
                    search: &mut search,
                    nodes_meta: &[],
                    inspected: None,
                },
            );
        });
        actions
    }

    fn meta(name: &str, path: &str, kind: NodeCategory) -> NodeMeta {
        NodeMeta {
            id: format!("{}:{path}", kind.tag_short()),
            name: name.to_string(),
            kind,
            path: path.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn empty_query_without_filter_matches_nothing() {
        let nodes = [meta("a", "src/a.rs", NodeCategory::File)];
        assert!(search_matches(&nodes, "", None).is_empty());
    }

    #[test]
    fn search_matches_name_and_path_case_insensitively() {
        let nodes = [
            meta("loader", "src/loader.rs", NodeCategory::File),
            meta("README", "README.md", NodeCategory::Doc),
        ];
        assert_eq!(search_matches(&nodes, "LOAD", None), vec![0]);
        // Path substring also matches.
        assert_eq!(search_matches(&nodes, "readme.md", None), vec![1]);
    }

    #[test]
    fn kind_filter_narrows_results() {
        let nodes = [
            meta("a", "src/a.rs", NodeCategory::File),
            meta("docs", "docs", NodeCategory::Dir),
        ];
        // Empty query but a filter still lists that kind.
        assert_eq!(search_matches(&nodes, "", Some(NodeCategory::Dir)), vec![1]);
        assert!(search_matches(&nodes, "a", Some(NodeCategory::Doc)).is_empty());
    }

    #[test]
    fn exact_name_ranks_before_substring() {
        let nodes = [
            meta("loader_ext", "a/loader_ext.rs", NodeCategory::File),
            meta("loader", "b/loader.rs", NodeCategory::File),
        ];
        // "loader" exact should come before "loader_ext".
        assert_eq!(search_matches(&nodes, "loader", None), vec![1, 0]);
    }

    #[test]
    fn drawing_without_interaction_asks_for_nothing() {
        let mut config = AppConfig::default();
        let mut params = LayoutParams::default();
        let mut choice = LayoutChoice::default();

        let actions = run_draw(&mut config, &mut params, &mut choice);

        assert_eq!(actions, GuiActions::default());
    }

    #[test]
    fn drawing_leaves_valid_state_untouched() {
        let mut config = AppConfig::default();
        let mut params = LayoutParams::default();
        let before = params;
        let mut choice = LayoutChoice::default();

        run_draw(&mut config, &mut params, &mut choice);

        assert_eq!(params, before);
    }

    #[test]
    fn speed_is_clamped_into_the_originals_range() {
        // The original clamped speed into 0.1..=1000 inside its InputFloat
        // handler; a negative speed reverses every displacement.
        let mut params = LayoutParams { speed: -5.0, ..Default::default() };
        clamp_params(&mut params);
        assert_eq!(params.speed, *SPEED_RANGE.start());

        params.speed = 5_000.0;
        clamp_params(&mut params);
        assert_eq!(params.speed, *SPEED_RANGE.end());
    }

    #[test]
    fn area_cannot_be_driven_to_zero_or_below() {
        // k is proportional to area, and a non-positive k inverts every force.
        let mut params = LayoutParams { area: 0.0, ..Default::default() };
        clamp_params(&mut params);
        assert_eq!(params.area, MIN_AREA);

        params.area = -100.0;
        clamp_params(&mut params);
        assert_eq!(params.area, MIN_AREA);
    }

    #[test]
    fn clamping_leaves_values_in_range_alone() {
        let mut params = LayoutParams { speed: 100.0, area: 1000.0, ..Default::default() };
        let before = params;
        clamp_params(&mut params);
        assert_eq!(params, before);
    }

    #[test]
    fn gravity_is_deliberately_unclamped() {
        // Negative gravity pushes nodes away from the origin, which is a
        // legitimate thing to explore; the original did not clamp it either.
        let mut params = LayoutParams { gravity: -3.0, ..Default::default() };
        clamp_params(&mut params);
        assert_eq!(params.gravity, -3.0);
    }
}
