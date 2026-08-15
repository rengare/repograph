use std::cell::RefCell;
use std::collections::VecDeque;
use std::f64::consts::TAU;
use std::rc::Rc;

use gv_graph::{GraphData, NodeCategory, loader};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, Document, HtmlCanvasElement, HtmlInputElement, HtmlSelectElement};

thread_local! {
    static VIEWER: RefCell<Option<Rc<RefCell<Viewer>>>> = const { RefCell::new(None) };
}

struct Viewer {
    document: Document,
    canvas: HtmlCanvasElement,
    context: CanvasRenderingContext2d,
    graph: GraphData,
    positions: Vec<(f64, f64)>,
    selected: Option<usize>,
    selection_depth: u32,
    show_labels: bool,
    visible: [bool; 5],
    pan: (f64, f64),
    zoom: f64,
    drag_origin: Option<(f64, f64)>,
}

#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    Ok(())
}

/// Loads the exact `repo.edges` and `nodes.tsv` text format accepted by the
/// native graph visualizer. The files never leave the browser.
#[wasm_bindgen]
pub fn load_graph(edges_text: String, nodes_text: String) -> Result<(), JsValue> {
    let graph = loader::from_exported_text(&edges_text, &nodes_text)
        .map_err(|error| JsValue::from_str(&format!("Invalid graph files: {error:#}")))?;
    let document = window()?.document().ok_or_else(|| JsValue::from_str("missing document"))?;
    let canvas = element::<HtmlCanvasElement>(&document, "graph-canvas")?;
    let context = canvas
        .get_context("2d")?
        .ok_or_else(|| JsValue::from_str("Canvas 2D is unavailable"))?
        .dyn_into::<CanvasRenderingContext2d>()?;
    let count = graph.node_count().max(1);
    let positions = (0..graph.node_count())
        .map(|index| {
            let angle = index as f64 * 2.399_963_229_728_653;
            let radius = 35.0 + 16.0 * (index as f64).sqrt();
            (radius * angle.cos(), radius * angle.sin())
        })
        .collect();
    let viewer = Rc::new(RefCell::new(Viewer {
        document,
        canvas,
        context,
        graph,
        positions,
        selected: None,
        selection_depth: input_value("selection-depth")?.parse().unwrap_or(1),
        show_labels: input_checked("show-labels")?,
        visible: [true; 5],
        pan: (0.0, 0.0),
        zoom: (700.0 / count as f64).clamp(0.45, 2.2),
        drag_origin: None,
    }));
    bind_controls(&viewer)?;
    viewer.borrow_mut().refresh_ui();
    viewer.borrow().render();
    VIEWER.with(|slot| *slot.borrow_mut() = Some(viewer));
    Ok(())
}

fn bind_controls(viewer: &Rc<RefCell<Viewer>>) -> Result<(), JsValue> {
    let document = viewer.borrow().document.clone();
    let redraw = |id: &str| -> Result<(), JsValue> {
        let viewer = viewer.clone();
        let callback = Closure::<dyn FnMut(web_sys::Event)>::new(move |_| {
            let mut viewer = viewer.borrow_mut();
            viewer.sync_controls();
            viewer.refresh_ui();
            viewer.render();
        });
        element::<web_sys::Element>(&document, id)?
            .add_event_listener_with_callback("input", callback.as_ref().unchecked_ref())?;
        callback.forget();
        Ok(())
    };
    for id in ["selection-depth", "show-labels", "show-dir", "show-file", "show-doc", "show-section", "show-symbol"] {
        redraw(id)?;
    }

    let viewer_for_search = viewer.clone();
    let search = Closure::<dyn FnMut(web_sys::Event)>::new(move |_| {
        let mut viewer = viewer_for_search.borrow_mut();
        viewer.refresh_ui();
        viewer.render();
    });
    element::<web_sys::Element>(&document, "search")?
        .add_event_listener_with_callback("input", search.as_ref().unchecked_ref())?;
    search.forget();

    let viewer_for_results = viewer.clone();
    let results = Closure::<dyn FnMut(web_sys::Event)>::new(move |_| {
        let mut viewer = viewer_for_results.borrow_mut();
        if let Ok(results) = element::<HtmlSelectElement>(&viewer.document, "search-results") {
            viewer.selected = results.value().parse().ok();
            viewer.refresh_ui();
            viewer.render();
        }
    });
    element::<web_sys::Element>(&document, "search-results")?
        .add_event_listener_with_callback("change", results.as_ref().unchecked_ref())?;
    results.forget();

    let canvas = viewer.borrow().canvas.clone();
    let viewer_for_down = viewer.clone();
    let down = Closure::<dyn FnMut(web_sys::MouseEvent)>::new(move |event: web_sys::MouseEvent| {
        viewer_for_down.borrow_mut().drag_origin = Some((event.offset_x() as f64, event.offset_y() as f64));
    });
    canvas.add_event_listener_with_callback("mousedown", down.as_ref().unchecked_ref())?;
    down.forget();

    let canvas = viewer.borrow().canvas.clone();
    let viewer_for_move = viewer.clone();
    let moved = Closure::<dyn FnMut(web_sys::MouseEvent)>::new(move |event: web_sys::MouseEvent| {
        let mut viewer = viewer_for_move.borrow_mut();
        if let Some((x, y)) = viewer.drag_origin {
            let next = (event.offset_x() as f64, event.offset_y() as f64);
            viewer.pan.0 += next.0 - x;
            viewer.pan.1 += next.1 - y;
            viewer.drag_origin = Some(next);
            viewer.render();
        }
    });
    canvas.add_event_listener_with_callback("mousemove", moved.as_ref().unchecked_ref())?;
    moved.forget();

    let canvas = viewer.borrow().canvas.clone();
    let viewer_for_up = viewer.clone();
    let up = Closure::<dyn FnMut(web_sys::MouseEvent)>::new(move |event: web_sys::MouseEvent| {
        let mut viewer = viewer_for_up.borrow_mut();
        let was_dragging = viewer.drag_origin.take();
        if was_dragging.is_some_and(|(x, y)| (x - event.offset_x() as f64).abs() < 3.0 && (y - event.offset_y() as f64).abs() < 3.0) {
            viewer.pick(event.offset_x() as f64, event.offset_y() as f64);
            viewer.refresh_ui();
            viewer.render();
        }
    });
    canvas.add_event_listener_with_callback("mouseup", up.as_ref().unchecked_ref())?;
    up.forget();

    let canvas = viewer.borrow().canvas.clone();
    let viewer_for_wheel = viewer.clone();
    let wheel = Closure::<dyn FnMut(web_sys::WheelEvent)>::new(move |event: web_sys::WheelEvent| {
        event.prevent_default();
        let mut viewer = viewer_for_wheel.borrow_mut();
        viewer.zoom = (viewer.zoom * if event.delta_y() > 0.0 { 0.88 } else { 1.14 }).clamp(0.12, 8.0);
        viewer.render();
    });
    canvas.add_event_listener_with_callback("wheel", wheel.as_ref().unchecked_ref())?;
    wheel.forget();
    Ok(())
}

impl Viewer {
    fn sync_controls(&mut self) {
        self.selection_depth = input_value("selection-depth").ok().and_then(|value| value.parse().ok()).unwrap_or(1);
        self.show_labels = input_checked("show-labels").unwrap_or(false);
        self.visible = [
            input_checked("show-dir").unwrap_or(true),
            input_checked("show-file").unwrap_or(true),
            input_checked("show-doc").unwrap_or(true),
            input_checked("show-section").unwrap_or(true),
            input_checked("show-symbol").unwrap_or(true),
        ];
    }

    fn neighborhood(&self) -> Option<Vec<bool>> {
        let selected = self.selected?;
        let mut included = vec![false; self.graph.node_count()];
        included[selected] = true;
        let mut pending = VecDeque::from([(selected, 0u32)]);
        while let Some((node, depth)) = pending.pop_front() {
            if depth == self.selection_depth {
                continue;
            }
            for &neighbor in self.graph.adjacency.neighbors_of(node) {
                let neighbor = neighbor as usize;
                if !included[neighbor] {
                    included[neighbor] = true;
                    pending.push_back((neighbor, depth + 1));
                }
            }
        }
        Some(included)
    }

    fn render(&self) {
        let width = self.canvas.client_width().max(1) as u32;
        let height = self.canvas.client_height().max(1) as u32;
        self.canvas.set_width(width);
        self.canvas.set_height(height);
        let context = &self.context;
        context.set_fill_style_str("#111827");
        context.fill_rect(0.0, 0.0, width as f64, height as f64);
        let included = self.neighborhood();
        for edge in &self.graph.edges {
            let (from, to) = (edge.from as usize, edge.to as usize);
            if !self.is_visible(from) || !self.is_visible(to) {
                continue;
            }
            let bright = included.as_ref().is_none_or(|mask| mask[from] && mask[to]);
            context.set_stroke_style_str(if bright { "#64748b" } else { "#263244" });
            context.begin_path();
            let (x, y) = self.screen_position(from, width, height);
            let (to_x, to_y) = self.screen_position(to, width, height);
            context.move_to(x, y);
            context.line_to(to_x, to_y);
            context.stroke();
        }
        for index in 0..self.graph.node_count() {
            if !self.is_visible(index) {
                continue;
            }
            let (x, y) = self.screen_position(index, width, height);
            let bright = included.as_ref().is_none_or(|mask| mask[index]);
            context.set_global_alpha(if bright { 1.0 } else { 0.18 });
            context.set_fill_style_str(category_color(self.graph.meta[index].kind));
            context.begin_path();
            let _ = context.arc(x, y, if self.selected == Some(index) { 7.0 } else { 4.5 }, 0.0, TAU);
            context.fill();
            if self.show_labels && bright {
                context.set_fill_style_str("#e5e7eb");
                context.set_font("12px ui-sans-serif, system-ui");
                let _ = context.fill_text(&self.graph.meta[index].name, x + 7.0, y - 7.0);
            }
        }
        context.set_global_alpha(1.0);
    }

    fn screen_position(&self, index: usize, width: u32, height: u32) -> (f64, f64) {
        let (x, y) = self.positions[index];
        (width as f64 * 0.5 + self.pan.0 + x * self.zoom, height as f64 * 0.5 + self.pan.1 + y * self.zoom)
    }

    fn pick(&mut self, x: f64, y: f64) {
        let width = self.canvas.width();
        let height = self.canvas.height();
        self.selected = (0..self.graph.node_count())
            .filter(|&index| self.is_visible(index))
            .filter_map(|index| {
                let (node_x, node_y) = self.screen_position(index, width, height);
                let distance = (node_x - x).hypot(node_y - y);
                (distance <= 10.0).then_some((index, distance))
            })
            .min_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index);
    }

    fn is_visible(&self, index: usize) -> bool {
        match self.graph.meta[index].kind {
            NodeCategory::Dir => self.visible[0],
            NodeCategory::File => self.visible[1],
            NodeCategory::Doc => self.visible[2],
            NodeCategory::Section => self.visible[3],
            NodeCategory::Symbol => self.visible[4],
            NodeCategory::Unknown => true,
        }
    }

    fn refresh_ui(&mut self) {
        let query = input_value("search").unwrap_or_default().to_lowercase();
        let results = element::<HtmlSelectElement>(&self.document, "search-results").unwrap();
        results.set_inner_html("");
        for (index, meta) in self.graph.meta.iter().enumerate().filter(|(_, meta)| {
            query.is_empty() || meta.name.to_lowercase().contains(&query) || meta.path.to_lowercase().contains(&query)
        }).take(80) {
            let option = self.document.create_element("option").unwrap();
            option.set_attribute("value", &index.to_string()).unwrap();
            option.set_text_content(Some(&format!("[{}] {}", meta.kind.tag_short(), meta.path)));
            let _ = results.append_child(&option);
        }
        element::<web_sys::Element>(&self.document, "stats").unwrap().set_text_content(Some(&format!("{} nodes · {} edges", self.graph.node_count(), self.graph.edge_count())));
        let inspector = element::<web_sys::Element>(&self.document, "inspector").unwrap();
        inspector.set_text_content(self.selected.and_then(|index| self.graph.meta.get(index)).map(|meta| {
            format!("{}\n[{}]\n{}\n{}", meta.name, meta.kind.tag_short(), meta.path, meta.signature.as_deref().unwrap_or(""))
        }).as_deref());
    }
}

fn window() -> Result<web_sys::Window, JsValue> {
    web_sys::window().ok_or_else(|| JsValue::from_str("missing browser window"))
}

fn element<T: JsCast>(document: &Document, id: &str) -> Result<T, JsValue> {
    document
        .get_element_by_id(id)
        .ok_or_else(|| JsValue::from_str(&format!("missing #{id}")))?
        .dyn_into::<T>()
        .map_err(Into::into)
}

fn input_value(id: &str) -> Result<String, JsValue> {
    Ok(element::<HtmlInputElement>(&window()?.document().ok_or_else(|| JsValue::from_str("missing document"))?, id)?.value())
}

fn input_checked(id: &str) -> Result<bool, JsValue> {
    Ok(element::<HtmlInputElement>(&window()?.document().ok_or_else(|| JsValue::from_str("missing document"))?, id)?.checked())
}

fn category_color(kind: NodeCategory) -> &'static str {
    match kind {
        NodeCategory::Dir => "#9499a3",
        NodeCategory::File => "#4dabf7",
        NodeCategory::Doc => "#69db7c",
        NodeCategory::Section => "#a9e34b",
        NodeCategory::Symbol => "#ffd43b",
        NodeCategory::Unknown => "#d1d5db",
    }
}
