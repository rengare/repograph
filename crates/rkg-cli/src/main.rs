//! `rkg` — build, query, and export a repository knowledge graph.

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
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

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Build(args) => build(args),
        Command::Query(args) => run_query(args),
        Command::Export(args) => run_export(args),
    }
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
