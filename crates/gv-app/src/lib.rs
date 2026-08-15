//! Wiring: CLI, the winit event loop, and the enum that unifies the CPU and
//! GPU layout traits.
//!
//! Everything substantive lives in the crates below this one; the binary at
//! `src/bin/graphvisualizer.rs` is a three-line shim so that all of the code
//! stays in libraries and stays testable.

pub mod app;
pub mod cli;
pub mod headless;
pub mod input;
pub mod layout_slot;

pub use app::App;
pub use cli::Cli;
pub use layout_slot::LayoutSlot;

use std::path::PathBuf;

use anyhow::{Context, Result};
use gv_config::AppConfig;
use gv_graph::{GraphData, loader, seed::SeedOptions};
use gv_layout::LayoutParams;

const DEFAULT_SETTINGS: &str = "settings.json";

/// Parses arguments, loads settings and the graph, and runs to completion.
pub fn run(cli: Cli) -> Result<()> {
    if cli.help {
        print!("{}", cli::USAGE);
        return Ok(());
    }

    let config = load_config(&cli)?;
    let options = seed_options(&config, cli.seed.unwrap_or(0));
    let mut graph = load_graph(&config, options)?;
    let choice = cli.layout.unwrap_or_default();

    match cli.headless_steps {
        Some(steps) => {
            let params = LayoutParams {
                three_d: config.graph_type_3d,
                ..Default::default()
            };
            let report = headless::run(&mut graph, &params, choice, steps)?;
            print!("{report}");
            Ok(())
        }
        None => run_windowed(config, graph, choice, options),
    }
}

fn run_windowed(
    config: AppConfig,
    graph: GraphData,
    choice: gv_gui::LayoutChoice,
    options: SeedOptions,
) -> Result<()> {
    let mut app = App::new(config, graph, choice, options)?;

    let event_loop =
        winit::event_loop::EventLoop::new().context("creating the winit event loop")?;
    // Poll rather than Wait: the layout advances every frame, so there is
    // always something to redraw.
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    event_loop.run_app(&mut app).context("running the event loop")?;

    Ok(())
}

/// Reads the settings file, then lets CLI overrides shadow it.
///
/// A missing settings file is not an error — the defaults stand — but a
/// malformed one is, and so is a missing file the user named explicitly.
pub fn load_config(cli: &Cli) -> Result<AppConfig> {
    let explicit = cli.settings.is_some();
    let path: PathBuf = cli
        .settings
        .clone()
        .unwrap_or_else(|| PathBuf::from(DEFAULT_SETTINGS));

    let mut config = if explicit || path.exists() {
        AppConfig::load(&path)?
    } else {
        log::debug!("no {DEFAULT_SETTINGS}; using defaults");
        AppConfig::default()
    };

    cli.apply_to(&mut config);
    Ok(config)
}

/// The initial scatter for this configuration.
///
/// Shared by startup and by the GUI's Reseed button, so the two cannot drift.
pub fn seed_options(config: &AppConfig, rng_seed: u64) -> SeedOptions {
    SeedOptions {
        extent: 1000.0,
        size_range: (config.node_size_range_start, config.node_size_range_end),
        three_d: config.graph_type_3d,
        seed: rng_seed,
    }
}

/// Loads the edge list named by `config` and scatters its nodes.
///
/// When `config.nodes_input` is set, the edge list is loaded index-aligned to that
/// knowledge-graph sidecar (via [`loader::load_with_sidecar`]) and nodes are
/// coloured by their kind instead of randomly — this is what turns the anonymous
/// visualizer into a browsable repository map.
pub fn load_graph(config: &AppConfig, options: SeedOptions) -> Result<GraphData> {
    let path = &config.edge_input;
    let mut graph = match &config.nodes_input {
        Some(nodes) => loader::load_with_sidecar(path, nodes).with_context(|| {
            format!(
                "loading graph from {} with sidecar {}",
                path.display(),
                nodes.display()
            )
        })?,
        None => {
            loader::load(path).with_context(|| format!("loading graph from {}", path.display()))?
        }
    };

    gv_graph::seed::scatter(&mut graph, options);
    if config.nodes_input.is_some() {
        gv_graph::seed::colorize_by_kind(&mut graph);
    }

    log::info!(
        "loaded {} nodes and {} edges from {}",
        graph.node_count(),
        graph.edge_count(),
        path.display()
    );
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_default_settings_file_is_not_an_error() {
        // Run from a directory with no settings.json: the defaults stand.
        let cli = Cli { settings: None, ..Default::default() };
        let config = load_config(&cli).expect("defaults should apply");
        assert_eq!(config.width, AppConfig::default().width);
    }

    #[test]
    fn an_explicitly_named_missing_settings_file_is_an_error() {
        // Silently ignoring this would leave the user wondering why their
        // settings had no effect.
        let cli = Cli {
            settings: Some(PathBuf::from("/nonexistent/settings.json")),
            ..Default::default()
        };
        assert!(load_config(&cli).is_err());
    }

    #[test]
    fn cli_edge_input_overrides_the_config() {
        let cli = Cli {
            edge_input: Some(PathBuf::from("from-cli.edges")),
            ..Default::default()
        };
        let config = load_config(&cli).unwrap();
        assert_eq!(config.edge_input, PathBuf::from("from-cli.edges"));
    }

    #[test]
    fn help_short_circuits_before_any_file_is_touched() {
        let cli = Cli {
            help: true,
            edge_input: Some(PathBuf::from("/nonexistent/graph.edges")),
            ..Default::default()
        };
        assert!(run(cli).is_ok());
    }

    #[test]
    fn a_missing_edge_list_names_the_file_it_could_not_open() {
        let config = AppConfig {
            edge_input: PathBuf::from("/nonexistent/graph.edges"),
            ..Default::default()
        };
        let error = load_graph(&config, SeedOptions::default()).unwrap_err();
        assert!(
            format!("{error:#}").contains("graph.edges"),
            "unhelpful error: {error:#}"
        );
    }

    #[test]
    fn seed_options_take_the_node_size_range_from_the_config() {
        let config = AppConfig {
            node_size_range_start: 3.0,
            node_size_range_end: 9.0,
            graph_type_3d: false,
            ..Default::default()
        };
        let options = seed_options(&config, 7);

        assert_eq!(options.size_range, (3.0, 9.0));
        assert!(!options.three_d);
        assert_eq!(options.seed, 7);
    }

    #[test]
    fn the_cli_seed_reaches_the_scatter() {
        let config = AppConfig::default();
        assert_eq!(seed_options(&config, 0).seed, 0);
        assert_eq!(seed_options(&config, 99).seed, 99);
    }
}
