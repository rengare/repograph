//! Application settings, deserialised from the same `settings.json` the C++
//! original used.
//!
//! The six shader-path fields of the original `AppConfig` are gone: WGSL
//! sources are compiled into the binary with `include_str!`, so there is
//! nothing to point at. Every remaining key keeps its original camelCase
//! spelling, so an old `settings.json` still loads.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AppConfig {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub position_x: i32,
    pub position_y: i32,
    pub show_gui: bool,
    pub is_vsync_enabled: bool,

    /// Clear colour, 0..=255 per channel — the range the original's ImGui
    /// sliders wrote, divided by 255 at upload time.
    pub red: f32,
    pub green: f32,
    pub blue: f32,

    pub graph_type_3d: bool,
    pub show_edge: bool,
    pub is_update_on: bool,

    /// Force-layout parameters, shared by the GUI and headless runs.
    pub speed: f32,
    pub area: f32,
    pub gravity: f32,
    /// How many edge hops remain highlighted after selecting a node. Zero keeps
    /// only the selected node visible at full brightness.
    pub selection_depth: u32,

    /// Draw each node's name (from the knowledge-graph sidecar) as a text label
    /// above it. Off by default; a no-op for anonymous edge-list graphs.
    pub show_labels: bool,

    pub node_size_range_start: f32,
    pub node_size_range_end: f32,

    /// Camera translation speed (world units per frame) for the `W`/`S`/`A`/`D`/
    /// `R`/`F` movement keys.
    pub move_speed: f32,
    /// Multiplier applied to mouse-wheel notches when zooming.
    pub wheel_zoom_speed: f32,

    /// Path to the whitespace-separated edge list to load.
    pub edge_input: PathBuf,

    /// Optional knowledge-graph node sidecar (`nodes.tsv` from `rkg export`).
    /// When set, the edge list is loaded index-aligned to it and nodes are
    /// coloured by kind. `None` keeps the plain anonymous edge-list behaviour.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nodes_input: Option<PathBuf>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            name: "Graph visualizer".to_owned(),
            width: 1280,
            height: 620,
            position_x: 400,
            position_y: 100,
            show_gui: true,
            is_vsync_enabled: true,
            red: 0.0,
            green: 0.0,
            blue: 0.0,
            graph_type_3d: true,
            show_edge: false,
            is_update_on: false,
            speed: 100.0,
            area: 1000.0,
            gravity: 1.0,
            selection_depth: 1,
            show_labels: false,
            node_size_range_start: 10.0,
            node_size_range_end: 30.0,
            move_speed: 100.0,
            wheel_zoom_speed: 6.0,
            edge_input: PathBuf::from("array.edges"),
            nodes_input: None,
        }
    }
}

impl AppConfig {
    /// Reads a settings file. Missing keys fall back to [`Default`], so a
    /// partial file — or no file at all — is valid.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading settings from {}", path.display()))?;
        serde_json::from_str(&text)
            .with_context(|| format!("parsing settings from {}", path.display()))
    }

    /// Updates graph controls while retaining settings this version does not
    /// model, such as the legacy shader-path keys.
    pub fn save_graph_settings(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => "{}".to_owned(),
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("reading settings from {}", path.display()))
            }
        };
        let mut json: serde_json::Value = serde_json::from_str(&text)
            .with_context(|| format!("parsing settings from {}", path.display()))?;
        let settings = json
            .as_object_mut()
            .context("settings JSON must be an object")?;
        settings.insert("speed".into(), self.speed.into());
        settings.insert("area".into(), self.area.into());
        settings.insert("gravity".into(), self.gravity.into());
        settings.insert("selectionDepth".into(), self.selection_depth.into());
        let json = serde_json::to_string_pretty(&json)?;
        std::fs::write(path, json)
            .with_context(|| format!("writing settings to {}", path.display()))
    }

    /// Clear colour normalised to the 0..=1 range wgpu expects.
    ///
    /// Widens before dividing: doing the division in f32 and casting after
    /// bakes in f32 rounding that the f64 target would otherwise not have.
    pub fn clear_color(&self) -> [f64; 4] {
        [
            f64::from(self.red) / 255.0,
            f64::from(self.green) / 255.0,
            f64::from(self.blue) / 255.0,
            1.0,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `settings.json` shipped with the C++ original, minus the shader
    /// paths, must still deserialise.
    #[test]
    fn reads_original_settings_json() {
        let json = r#"{
            "name": "Graph visualizer",
            "showGui": true,
            "width": 1280,
            "height": 620,
            "positionX": 400,
            "positionY": 100,
            "isVsyncEnabled": true,
            "red": 0, "green": 0, "blue": 0,
            "graphType3d": true,
            "showEdge": false,
            "isUpdateOn": false,
            "nodeSizeRangeStart": 10,
            "nodeSizeRangeEnd": 30,
            "nodeShaderName": "nodes",
            "nodeShaderVertexPath": "res/shaders/circle.vert",
            "edgeInput": "array.edges"
        }"#;

        let config: AppConfig = serde_json::from_str(json).expect("should parse");
        assert_eq!(config.width, 1280);
        assert!(config.graph_type_3d);
        assert_eq!(config.node_size_range_end, 30.0);
        assert_eq!(config.speed, 100.0);
        assert_eq!(config.area, 1000.0);
        assert_eq!(config.gravity, 1.0);
        assert_eq!(config.selection_depth, 1);
        assert_eq!(config.edge_input, PathBuf::from("array.edges"));
    }

    #[test]
    fn empty_object_yields_defaults() {
        let config: AppConfig = serde_json::from_str("{}").expect("should parse");
        assert_eq!(config, AppConfig::default());
    }

    #[test]
    fn saving_graph_settings_preserves_unrelated_settings() {
        let path = std::env::temp_dir().join(format!("gv-config-{}.json", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let config = AppConfig {
            speed: 42.0,
            area: 12.5,
            gravity: -3.0,
            selection_depth: 3,
            ..Default::default()
        };
        std::fs::write(&path, r#"{ "legacyShader": "nodes" }"#).unwrap();

        config.save_graph_settings(&path).unwrap();

        let saved: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(saved["legacyShader"], "nodes");
        assert_eq!(saved["speed"], 42.0);
        assert_eq!(saved["area"], 12.5);
        assert_eq!(saved["gravity"], -3.0);
        assert_eq!(saved["selectionDepth"], 3);
        std::fs::remove_file(path).unwrap();
    }
}
