//! WebAssembly entry point for the graph viewer.
//!
//! This crate is a thin shim: the browser's `app.js` collects the `repo.edges`
//! and `nodes.tsv` text the user picks (nothing is uploaded), then calls
//! [`run`], which loads the graph and hands it to the desktop viewer
//! ([`gv_app::run_web`]) — the very same wgpu/WebGPU renderer, egui GUI, camera,
//! picking, filters, and layouts, compiled to wasm. Rendering does not start
//! until files are supplied.
//!
//! The crate is wasm-only; on native targets it compiles to nothing.
#![cfg(target_arch = "wasm32")]

use gv_config::AppConfig;
use gv_gui::LayoutChoice;
use wasm_bindgen::prelude::*;

/// Installs the panic hook and a console logger. Called once at module load.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);
}

/// Loads the exported graph from the supplied file contents and starts the
/// viewer on the page's `<canvas id="graph-canvas">`. Call this only after the
/// user has chosen both files.
#[wasm_bindgen]
pub fn run(edges_text: String, nodes_text: String) -> Result<(), JsValue> {
    run_inner(&edges_text, &nodes_text).map_err(|error| JsValue::from_str(&format!("{error:#}")))
}

fn run_inner(edges_text: &str, nodes_text: &str) -> anyhow::Result<()> {
    let config = AppConfig::default();
    let options = gv_app::seed_options(&config, 0);

    let mut graph = gv_graph::loader::from_exported_text(edges_text, nodes_text)?;
    gv_graph::seed::scatter(&mut graph, options);
    gv_graph::seed::colorize_by_kind(&mut graph);

    log::info!(
        "loaded {} nodes, {} edges",
        graph.node_count(),
        graph.edge_count()
    );

    // Match the desktop default (F-R gpu): the layout step runs as a WebGPU
    // compute pass, keeping pace with the desktop rather than dropping to the
    // O(n²) single-threaded wasm CPU layout. The CPU layouts remain selectable.
    gv_app::run_web(config, graph, LayoutChoice::default(), options)
}
