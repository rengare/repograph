//! `rkg` — build, query, and export a repository knowledge graph.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use rkg_core::export::{self, ExportOptions};
use rkg_core::query::{self, Direction};
use rkg_core::{EdgeKind, Graph, NodeKind};

const DEFAULT_GRAPH: &str = ".rkg/graph.json";

#[derive(Parser)]
#[command(name = "rkg", about = "Repository knowledge graph")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Scan a repository and persist its knowledge graph.
    Build(BuildArgs),
    /// Query a previously built graph.
    Query(QueryArgs),
    /// Project the graph to the visualizer's edge list + node sidecar.
    Export(ExportArgs),
    /// Install rkg-mcp and configure an AI coding client for this repository.
    Mcp(McpArgs),
}

#[derive(Args)]
struct BuildArgs {
    /// Repository root to scan.
    #[arg(default_value = ".")]
    path: String,
    /// Where to write the graph JSON.
    #[arg(short, long, default_value = DEFAULT_GRAPH)]
    output: String,
}

#[derive(Args)]
struct QueryArgs {
    /// Path to a graph JSON produced by `rkg build`.
    #[arg(short, long, default_value = DEFAULT_GRAPH, global = true)]
    graph: String,
    /// Emit machine-readable JSON instead of text.
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    what: QueryCmd,
}

#[derive(Subcommand)]
enum QueryCmd {
    /// Find nodes by name/path/id substring.
    Find {
        query: String,
        /// Restrict to a kind: dir|file|doc|sec|sym.
        #[arg(long)]
        kind: Option<String>,
    },
    /// List nodes near an id.
    Neighbors {
        id: String,
        #[arg(long, default_value_t = 1)]
        depth: u32,
        /// out|in|both.
        #[arg(long, default_value = "both")]
        direction: String,
        /// Restrict to edge kinds (repeatable): contains|imports|links|…
        #[arg(long = "edge-kind")]
        edge_kinds: Vec<String>,
    },
    /// Build a token-budgeted context pack around a seed node.
    Context {
        seed: String,
        #[arg(long, default_value_t = 2000)]
        budget: u32,
    },
    /// Print the induced subgraph over the given ids.
    Subgraph { ids: Vec<String> },
    /// Shortest (undirected) path between two nodes.
    Path { a: String, b: String },
}

#[derive(Args)]
struct ExportArgs {
    #[arg(short, long, default_value = DEFAULT_GRAPH)]
    graph: String,
    /// Output edge list.
    #[arg(long, default_value = "repo.edges")]
    edges: String,
    /// Output node-attribute sidecar.
    #[arg(long, default_value = "nodes.tsv")]
    nodes: String,
    /// Only project these edge kinds (repeatable).
    #[arg(long = "edge-kind")]
    edge_kinds: Vec<String>,
}

#[derive(Args)]
struct McpArgs {
    #[command(subcommand)]
    command: McpCommand,
}

#[derive(Subcommand)]
enum McpCommand {
    /// Install rkg-mcp and add a project-scoped MCP configuration.
    Install(McpInstallArgs),
}

#[derive(Args)]
struct McpInstallArgs {
    /// Client to configure.
    #[arg(value_enum)]
    provider: McpProvider,
    /// Project directory that receives the project-scoped configuration (any
    /// directory — it need not be the repograph checkout).
    #[arg(long, default_value = ".")]
    project_dir: PathBuf,
    /// repograph checkout to build `rkg-mcp` from with `cargo install --path
    /// crates/rkg-mcp`. Defaults to the project directory when that is itself a
    /// repograph checkout; otherwise the build is skipped and the config points at
    /// an `rkg-mcp` already on your PATH.
    #[arg(long)]
    source: Option<PathBuf>,
    /// Add configuration only; never build `rkg-mcp`.
    #[arg(long)]
    no_install: bool,
}

#[derive(Clone, Copy, ValueEnum)]
enum McpProvider {
    Opencode,
    ClaudeCode,
    Codex,
    Junie,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Build(args) => build(args),
        Command::Query(args) => run_query(args),
        Command::Export(args) => run_export(args),
        Command::Mcp(args) => match args.command {
            McpCommand::Install(args) => install_mcp(args),
        },
    }
}

/// A directory is a repograph checkout we can build the server from if it holds
/// the `rkg-mcp` crate.
fn is_repograph_source(dir: &Path) -> bool {
    dir.join("crates/rkg-mcp/Cargo.toml").exists()
}

/// Whether an executable named `rkg-mcp` is resolvable on `PATH`.
fn rkg_mcp_on_path() -> bool {
    let name = if cfg!(windows) { "rkg-mcp.exe" } else { "rkg-mcp" };
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|dir| dir.join(name).is_file())
}

fn install_mcp(args: McpInstallArgs) -> Result<()> {
    // The project directory only receives config files; it need not be a Cargo
    // workspace (that requirement was the reason this failed outside the checkout).
    let project_dir = args
        .project_dir
        .canonicalize()
        .with_context(|| format!("resolving project directory {}", args.project_dir.display()))?;

    // Building the server needs a repograph checkout. Use --source if given, else
    // the project dir when it happens to be one, else skip the build entirely.
    let source = match &args.source {
        Some(source) => {
            let source = source
                .canonicalize()
                .with_context(|| format!("resolving --source {}", source.display()))?;
            if !is_repograph_source(&source) {
                bail!(
                    "--source {} is not a repograph checkout (no crates/rkg-mcp/Cargo.toml)",
                    source.display()
                );
            }
            Some(source)
        }
        None => is_repograph_source(&project_dir).then(|| project_dir.clone()),
    };

    let built = if args.no_install {
        false
    } else if let Some(source) = &source {
        let status = ProcessCommand::new("cargo")
            .args(["install", "--locked", "--path", "crates/rkg-mcp"])
            .current_dir(source)
            .status()
            .context("running cargo install for rkg-mcp")?;
        if !status.success() {
            bail!("cargo install for rkg-mcp failed with {status}");
        }
        true
    } else {
        false
    };

    let path = match args.provider {
        McpProvider::Opencode => install_opencode(&project_dir)?,
        McpProvider::ClaudeCode => install_json_mcp(&project_dir.join(".mcp.json"))?,
        McpProvider::Codex => install_codex(&project_dir)?,
        McpProvider::Junie => install_json_mcp(&project_dir.join(".junie/mcp/mcp.json"))?,
    };
    println!("configured {}", path.display());

    // If we didn't build the server, the config points at whatever `rkg-mcp` is on
    // PATH — warn if there isn't one so it doesn't silently fail to start later.
    if !built && !rkg_mcp_on_path() {
        eprintln!(
            "note: `rkg-mcp` was not found on your PATH. Install it with `cargo install \
             --locked --path crates/rkg-mcp` from your repograph checkout, or re-run with \
             `--source <repograph-dir>`."
        );
    }
    Ok(())
}

fn install_opencode(project_dir: &Path) -> Result<PathBuf> {
    let json = project_dir.join("opencode.json");
    let jsonc = project_dir.join("opencode.jsonc");
    if jsonc.exists() {
        bail!(
            "{} exists; add the repograph MCP entry there manually to preserve JSONC comments",
            jsonc.display()
        );
    }
    install_opencode_json(&json)?;
    Ok(json)
}

fn install_opencode_json(path: &Path) -> Result<()> {
    let mut root = read_json_object(path)?;
    let mcp = root.entry("mcp").or_insert_with(|| serde_json::json!({}));
    let servers = mcp
        .as_object_mut()
        .context("the existing `mcp` field must be an object")?;
    if servers.contains_key("repograph") {
        bail!(
            "{} already configures an MCP server named repograph",
            path.display()
        );
    }
    servers.insert(
        "repograph".into(),
        serde_json::json!({
            "type": "local",
            "command": ["rkg-mcp", "--graph", DEFAULT_GRAPH],
            "cwd": ".",
            "timeout": 30000,
        }),
    );
    write_json(path, &root)
}

fn install_json_mcp(path: &Path) -> Result<PathBuf> {
    let mut root = read_json_object(path)?;
    let servers = root
        .entry("mcpServers")
        .or_insert_with(|| serde_json::json!({}))
        .as_object_mut()
        .context("the existing `mcpServers` field must be an object")?;
    if servers.contains_key("repograph") {
        bail!(
            "{} already configures an MCP server named repograph",
            path.display()
        );
    }
    servers.insert(
        "repograph".into(),
        serde_json::json!({
            "command": "rkg-mcp",
            "args": ["--graph", DEFAULT_GRAPH],
        }),
    );
    write_json(path, &root)?;
    Ok(path.to_owned())
}

fn install_codex(project_dir: &Path) -> Result<PathBuf> {
    let path = project_dir.join(".codex/config.toml");
    let table = "[mcp_servers.repograph]";
    let existing = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => return Err(error).with_context(|| format!("reading {}", path.display())),
    };
    if existing.lines().any(|line| line.trim() == table) {
        bail!(
            "{} already configures an MCP server named repograph",
            path.display()
        );
    }
    let mut updated = existing;
    if !updated.is_empty() && !updated.ends_with('\n') {
        updated.push('\n');
    }
    if !updated.is_empty() {
        updated.push('\n');
    }
    updated.push_str(&format!(
        "{table}\ncommand = \"rkg-mcp\"\nargs = [\"--graph\", \"{DEFAULT_GRAPH}\"]\ncwd = \".\"\nstartup_timeout_sec = 30\ntool_timeout_sec = 300\n"
    ));
    write_text(&path, &updated)?;
    Ok(path)
}

fn read_json_object(path: &Path) -> Result<serde_json::Map<String, serde_json::Value>> {
    let value = match fs::read_to_string(path) {
        Ok(text) => {
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => serde_json::json!({}),
        Err(error) => return Err(error).with_context(|| format!("reading {}", path.display())),
    };
    value
        .as_object()
        .cloned()
        .context("MCP configuration must be a JSON object")
}

fn write_json(path: &Path, value: &serde_json::Map<String, serde_json::Value>) -> Result<()> {
    write_text(path, &format!("{}\n", serde_json::to_string_pretty(value)?))
}

fn write_text(path: &Path, text: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}

fn build(args: BuildArgs) -> Result<()> {
    let graph = rkg_ingest::build_graph(&args.path)
        .with_context(|| format!("building graph for {}", args.path))?;
    graph.save(&args.output)?;
    eprintln!(
        "built {} nodes, {} edges -> {}",
        graph.node_count(),
        graph.edge_count(),
        args.output
    );
    Ok(())
}

fn run_query(args: QueryArgs) -> Result<()> {
    let graph = Graph::load(&args.graph)
        .with_context(|| format!("loading {} (run `rkg build` first?)", args.graph))?;

    match args.what {
        QueryCmd::Find { query, kind } => {
            let kind = kind.as_deref().map(parse_node_kind).transpose()?;
            let hits = query::find_node(&graph, &query, kind);
            if args.json {
                println!("{}", serde_json::to_string_pretty(&hits)?);
            } else if hits.is_empty() {
                println!("no matches");
            } else {
                for n in hits {
                    println!("{}\t[{}]\t{}", n.id, n.kind.tag(), n.path);
                }
            }
        }
        QueryCmd::Neighbors {
            id,
            depth,
            direction,
            edge_kinds,
        } => {
            let dir = parse_direction(&direction)?;
            let kinds = parse_edge_kinds(&edge_kinds)?;
            let out = query::neighbors(&graph, &id, depth, kinds.as_deref(), dir);
            if args.json {
                println!("{}", serde_json::to_string_pretty(&out)?);
            } else if out.is_empty() {
                println!("no neighbors");
            } else {
                for id in out {
                    println!("{id}");
                }
            }
        }
        QueryCmd::Context { seed, budget } => {
            let pack = query::context_pack(&graph, &seed, budget)
                .with_context(|| format!("seed {seed:?} not found in graph"))?;
            if args.json {
                println!("{}", serde_json::to_string_pretty(&pack)?);
            } else {
                println!("context for {} (~{} tokens):", pack.seed, pack.est_tokens);
                for e in &pack.entries {
                    println!("  [{}] {}  — {}", e.kind.tag(), e.id, e.reason);
                    if let Some(sig) = &e.signature {
                        println!("        {sig}");
                    }
                }
            }
        }
        QueryCmd::Subgraph { ids } => {
            let sub = query::subgraph(&graph, &ids);
            if args.json {
                println!("{}", serde_json::to_string_pretty(&sub)?);
            } else {
                println!("{} nodes, {} edges", sub.node_count(), sub.edge_count());
            }
        }
        QueryCmd::Path { a, b } => match query::path_between(&graph, &a, &b) {
            Some(path) => {
                if args.json {
                    println!("{}", serde_json::to_string_pretty(&path)?);
                } else {
                    println!("{}", path.join(" -> "));
                }
            }
            None => println!("no path"),
        },
    }
    Ok(())
}

fn run_export(args: ExportArgs) -> Result<()> {
    let graph = Graph::load(&args.graph)
        .with_context(|| format!("loading {} (run `rkg build` first?)", args.graph))?;
    let opts = ExportOptions {
        edge_kinds: parse_edge_kinds(&args.edge_kinds)?,
    };
    let written = export::write_edges(&graph, &args.edges, &opts)?;
    export::write_nodes_tsv(&graph, &args.nodes)?;
    eprintln!(
        "exported {} nodes, {} edges -> {} + {}",
        graph.node_count(),
        written,
        args.edges,
        args.nodes
    );
    Ok(())
}

fn parse_node_kind(s: &str) -> Result<NodeKind> {
    NodeKind::from_tag(s).with_context(|| format!("unknown node kind {s:?}"))
}

fn parse_direction(s: &str) -> Result<Direction> {
    Ok(match s {
        "out" => Direction::Out,
        "in" => Direction::In,
        "both" => Direction::Both,
        other => bail!("unknown direction {other:?} (expected out|in|both)"),
    })
}

fn parse_edge_kinds(tags: &[String]) -> Result<Option<Vec<EdgeKind>>> {
    if tags.is_empty() {
        return Ok(None);
    }
    let kinds = tags
        .iter()
        .map(|t| EdgeKind::from_tag(t).with_context(|| format!("unknown edge kind {t:?}")))
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(kinds))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    static NEXT_TEMP_DIR: AtomicUsize = AtomicUsize::new(0);

    fn temp_dir() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "rkg-cli-mcp-{}-{}",
            std::process::id(),
            NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn temp_project() -> PathBuf {
        let path = temp_dir();
        fs::write(path.join("Cargo.toml"), "[workspace]\n").unwrap();
        path
    }

    #[test]
    fn configures_a_project_that_is_not_a_cargo_workspace() {
        // The reported bug: running the installer outside the repograph checkout
        // must still write config. `--no-install` keeps the build out of the test.
        let project = temp_dir();
        assert!(!project.join("Cargo.toml").exists());

        install_mcp(McpInstallArgs {
            provider: McpProvider::ClaudeCode,
            project_dir: project.clone(),
            source: None,
            no_install: true,
        })
        .expect("config-only install should succeed anywhere");

        let config: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(project.join(".mcp.json")).unwrap()).unwrap();
        assert_eq!(config["mcpServers"]["repograph"]["command"], "rkg-mcp");
        fs::remove_dir_all(project).unwrap();
    }

    #[test]
    fn rejects_a_source_that_is_not_a_repograph_checkout() {
        let project = temp_dir();
        let error = install_mcp(McpInstallArgs {
            provider: McpProvider::ClaudeCode,
            project_dir: project.clone(),
            source: Some(project.clone()), // no crates/rkg-mcp/Cargo.toml here
            no_install: true,
        })
        .unwrap_err();
        assert!(error.to_string().contains("not a repograph checkout"), "{error}");
        fs::remove_dir_all(project).unwrap();
    }

    #[test]
    fn installs_project_configuration_for_every_provider() {
        let project = temp_project();

        install_opencode(&project).unwrap();
        install_json_mcp(&project.join(".mcp.json")).unwrap();
        install_codex(&project).unwrap();
        install_json_mcp(&project.join(".junie/mcp/mcp.json")).unwrap();

        let opencode: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(project.join("opencode.json")).unwrap())
                .unwrap();
        assert_eq!(opencode["mcp"]["repograph"]["type"], "local");
        assert_eq!(opencode["mcp"]["repograph"]["command"][0], "rkg-mcp");

        for path in [
            project.join(".mcp.json"),
            project.join(".junie/mcp/mcp.json"),
        ] {
            let config: serde_json::Value =
                serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
            assert_eq!(config["mcpServers"]["repograph"]["command"], "rkg-mcp");
        }

        let codex = fs::read_to_string(project.join(".codex/config.toml")).unwrap();
        assert!(codex.contains("[mcp_servers.repograph]"));
        assert!(codex.contains("tool_timeout_sec = 300"));
        fs::remove_dir_all(project).unwrap();
    }

    #[test]
    fn does_not_overwrite_an_existing_server() {
        let project = temp_project();
        install_json_mcp(&project.join(".mcp.json")).unwrap();
        let error = install_json_mcp(&project.join(".mcp.json")).unwrap_err();
        assert!(error.to_string().contains("already configures"));
        fs::remove_dir_all(project).unwrap();
    }

    #[test]
    fn appends_to_existing_json_configuration() {
        let project = temp_project();
        let path = project.join(".mcp.json");
        fs::write(
            &path,
            r#"{
  "keep": true,
  "mcpServers": {
    "other": { "command": "other-mcp" }
  }
}
"#,
        )
        .unwrap();

        install_json_mcp(&path).unwrap();

        let config: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(config["keep"], true);
        assert_eq!(config["mcpServers"]["other"]["command"], "other-mcp");
        assert_eq!(config["mcpServers"]["repograph"]["command"], "rkg-mcp");
        fs::remove_dir_all(project).unwrap();
    }
}
