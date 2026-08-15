//! `rkg-mcp` — an MCP stdio server exposing the repository knowledge graph.
//!
//! Reads newline-delimited JSON-RPC from stdin and writes responses to stdout
//! (MCP's stdio transport). The graph is loaded once from a `rkg build` output;
//! all query logic lives in [`rkg_core::query`] via [`server::Server`].
//!
//! Register in Claude Code with an `.mcp.json` entry:
//! ```json
//! { "mcpServers": { "repograph": {
//!     "command": "rkg-mcp", "args": ["--graph", ".rkg/graph.json"] } } }
//! ```

mod server;

use std::io::{BufRead, Write};
use std::path::Path;

use anyhow::{Context, Result};
use clap::Parser;
use rkg_core::Graph;
use serde_json::{Value, json};

use crate::server::Server;

#[derive(Parser)]
#[command(name = "rkg-mcp", about = "MCP server over a repository knowledge graph")]
struct Cli {
    /// Path to a graph JSON produced by `rkg build`.
    #[arg(short, long, env = "RKG_GRAPH", default_value = ".rkg/graph.json")]
    graph: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    // Start empty when the graph does not exist yet — the client can populate it
    // with the `build` tool. A file that exists but fails to parse is a real error.
    let graph = if Path::new(&cli.graph).exists() {
        Graph::load(&cli.graph).with_context(|| format!("loading {}", cli.graph))?
    } else {
        eprintln!(
            "rkg-mcp: {} not found — starting empty; call the `build` tool to populate it",
            cli.graph
        );
        Graph::new()
    };
    let server = Server::new(graph, cli.graph.clone().into());

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    for line in stdin.lock().lines() {
        let line = line.context("reading stdin")?;
        if line.trim().is_empty() {
            continue;
        }

        let response = match serde_json::from_str::<Value>(&line) {
            Ok(request) => server.handle(&request),
            // A malformed line still deserves a JSON-RPC parse error, per spec.
            Err(error) => Some(json!({
                "jsonrpc": "2.0",
                "id": Value::Null,
                "error": { "code": -32700, "message": format!("parse error: {error}") },
            })),
        };

        if let Some(response) = response {
            writeln!(out, "{}", serde_json::to_string(&response)?)?;
            out.flush()?;
        }
    }
    Ok(())
}
