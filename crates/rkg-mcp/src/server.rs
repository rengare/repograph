//! The MCP request handler, split from the stdio plumbing so it can be tested
//! without pipes.
//!
//! Speaks JSON-RPC 2.0 (MCP's wire format): [`Server::handle`] takes one parsed
//! request and returns the response to send, or `None` for notifications (which
//! get no reply). Tools are thin adapters over [`rkg_core::query`].

use std::cell::RefCell;
use std::path::PathBuf;

use rkg_core::query::{self, Direction};
use rkg_core::{EdgeKind, Graph, NodeKind};
use serde_json::{Value, json};

/// The MCP protocol revision reported when a client omits its own.
const DEFAULT_PROTOCOL: &str = "2024-11-05";

pub struct Server {
    /// Interior mutability so the `build` tool can swap in a freshly scanned
    /// graph while `handle` only needs `&self` (the stdio loop is single-threaded).
    graph: RefCell<Graph>,
    /// Where `build` writes the graph JSON when no explicit output is given.
    graph_path: PathBuf,
}

impl Server {
    pub fn new(graph: Graph, graph_path: PathBuf) -> Self {
        Server {
            graph: RefCell::new(graph),
            graph_path,
        }
    }

    /// Handles one JSON-RPC message. Returns `Some(response)` for requests and
    /// `None` for notifications (a message with no `id`).
    pub fn handle(&self, request: &Value) -> Option<Value> {
        let id = request.get("id").cloned();
        let method = request.get("method").and_then(Value::as_str).unwrap_or("");
        let params = request.get("params").cloned().unwrap_or(Value::Null);

        // Notifications carry no id and never get a reply.
        let id = id?;

        let result = match method {
            "initialize" => Ok(self.initialize(&params)),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(self.tools_list()),
            "tools/call" => Ok(self.tools_call(&params)),
            other => Err((-32601, format!("method not found: {other}"))),
        };

        Some(match result {
            Ok(value) => json!({ "jsonrpc": "2.0", "id": id, "result": value }),
            Err((code, message)) => json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": { "code": code, "message": message },
            }),
        })
    }

    fn initialize(&self, params: &Value) -> Value {
        // Echo the client's protocol version when given, so we agree on a shared
        // revision rather than forcing ours.
        let protocol = params
            .get("protocolVersion")
            .and_then(Value::as_str)
            .unwrap_or(DEFAULT_PROTOCOL);
        json!({
            "protocolVersion": protocol,
            "capabilities": { "tools": {} },
            "serverInfo": {
                "name": "rkg-mcp",
                "version": env!("CARGO_PKG_VERSION"),
            },
        })
    }

    fn tools_list(&self) -> Value {
        json!({ "tools": tool_schemas() })
    }

    fn tools_call(&self, params: &Value) -> Value {
        let name = params.get("name").and_then(Value::as_str).unwrap_or("");
        let args = params.get("arguments").cloned().unwrap_or(Value::Null);

        match self.dispatch(name, &args) {
            Ok(value) => json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string_pretty(&value).unwrap_or_default(),
                }],
            }),
            // MCP convention: tool-level failures are results with isError, not
            // JSON-RPC errors, so the model can read and react to them.
            Err(message) => json!({
                "content": [{ "type": "text", "text": message }],
                "isError": true,
            }),
        }
    }

    /// Runs one tool by name, returning its result payload or an error message.
    fn dispatch(&self, name: &str, args: &Value) -> Result<Value, String> {
        // `build` mutates; everything else reads. Read tools take a shared borrow
        // of the current graph.
        if name == "build" {
            return self.build(args);
        }

        let graph = self.graph.borrow();
        match name {
            "find_node" => {
                let query = str_arg(args, "query")?;
                let kind = opt_str(args, "kind")
                    .map(|t| NodeKind::from_tag(t).ok_or_else(|| format!("unknown kind {t:?}")))
                    .transpose()?;
                Ok(json!(query::find_node(&graph, query, kind)))
            }
            "neighbors" => {
                let id = str_arg(args, "id")?;
                let depth = args.get("depth").and_then(Value::as_u64).unwrap_or(1) as u32;
                let direction = parse_direction(opt_str(args, "direction").unwrap_or("both"))?;
                let edge_kinds = parse_edge_kinds(args.get("edge_kinds"))?;
                Ok(json!(query::neighbors(
                    &graph,
                    id,
                    depth,
                    edge_kinds.as_deref(),
                    direction
                )))
            }
            "context_pack" => {
                let seed = str_arg(args, "seed")?;
                let budget = args.get("budget").and_then(Value::as_u64).unwrap_or(2000) as u32;
                let pack = query::context_pack(&graph, seed, budget)
                    .ok_or_else(|| format!("seed {seed:?} not found in graph"))?;
                Ok(json!(pack))
            }
            "subgraph" => {
                let ids = str_array(args, "ids")?;
                Ok(json!(query::subgraph(&graph, &ids)))
            }
            "path_between" => {
                let a = str_arg(args, "a")?;
                let b = str_arg(args, "b")?;
                Ok(json!(query::path_between(&graph, a, b)))
            }
            other => Err(format!("unknown tool: {other}")),
        }
    }

    /// Scans a repository, persists the graph JSON, and swaps it in as the live
    /// graph the other tools query.
    fn build(&self, args: &Value) -> Result<Value, String> {
        let path = str_arg(args, "path")?;
        let output = opt_str(args, "output")
            .map(PathBuf::from)
            .unwrap_or_else(|| self.graph_path.clone());

        let graph = rkg_ingest::build_graph(path)
            .map_err(|e| format!("building graph for {path}: {e:#}"))?;
        graph
            .save(&output)
            .map_err(|e| format!("writing {}: {e:#}", output.display()))?;

        let (nodes, edges) = (graph.node_count(), graph.edge_count());
        *self.graph.borrow_mut() = graph;
        Ok(json!({
            "nodes": nodes,
            "edges": edges,
            "output": output.display().to_string(),
        }))
    }
}

fn str_arg<'a>(args: &'a Value, key: &str) -> Result<&'a str, String> {
    args.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing required string argument {key:?}"))
}

fn opt_str<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(Value::as_str)
}

fn str_array(args: &Value, key: &str) -> Result<Vec<String>, String> {
    let arr = args
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing required array argument {key:?}"))?;
    Ok(arr
        .iter()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect())
}

fn parse_direction(s: &str) -> Result<Direction, String> {
    match s {
        "out" => Ok(Direction::Out),
        "in" => Ok(Direction::In),
        "both" => Ok(Direction::Both),
        other => Err(format!("unknown direction {other:?} (expected out|in|both)")),
    }
}

fn parse_edge_kinds(value: Option<&Value>) -> Result<Option<Vec<EdgeKind>>, String> {
    let Some(arr) = value.and_then(Value::as_array) else {
        return Ok(None);
    };
    let kinds = arr
        .iter()
        .filter_map(Value::as_str)
        .map(|t| EdgeKind::from_tag(t).ok_or_else(|| format!("unknown edge kind {t:?}")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(kinds))
}

/// The `tools/list` schemas. Kept as data so the wire contract is in one place.
fn tool_schemas() -> Value {
    json!([
        {
            "name": "build",
            "description": "Scan a repository into the knowledge graph, persist it, and make it the live graph the other tools query. Run this first if the graph is empty or stale.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Repository root to scan." },
                    "output": { "type": "string", "description": "Where to write the graph JSON. Defaults to the server's --graph path." }
                },
                "required": ["path"]
            }
        },
        {
            "name": "find_node",
            "description": "Find graph nodes by name/path/id substring, optionally filtered by kind (dir|file|doc|sec|sym).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Case-insensitive substring." },
                    "kind": { "type": "string", "enum": ["dir", "file", "doc", "sec", "sym"] }
                },
                "required": ["query"]
            }
        },
        {
            "name": "neighbors",
            "description": "Node ids within `depth` hops of a node, following edges in a direction and optional edge kinds.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string" },
                    "depth": { "type": "integer", "default": 1 },
                    "direction": { "type": "string", "enum": ["out", "in", "both"], "default": "both" },
                    "edge_kinds": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["id"]
            }
        },
        {
            "name": "context_pack",
            "description": "A ranked, token-budgeted neighborhood around a seed node — the compact context bundle to read instead of whole files.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "seed": { "type": "string", "description": "Seed node id, e.g. file:src/x.rs or sym:src/x.rs::parse." },
                    "budget": { "type": "integer", "default": 2000, "description": "Approximate token budget." }
                },
                "required": ["seed"]
            }
        },
        {
            "name": "subgraph",
            "description": "The induced subgraph over a set of node ids (nodes plus edges with both endpoints in the set).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "ids": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["ids"]
            }
        },
        {
            "name": "path_between",
            "description": "Shortest undirected path between two node ids, or null if disconnected.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "a": { "type": "string" },
                    "b": { "type": "string" }
                },
                "required": ["a", "b"]
            }
        }
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use rkg_core::{Edge, Node};

    fn server() -> Server {
        let mut g = Graph::new();
        g.add_node(Node::new(NodeKind::File, "src/a.rs", "a"));
        g.add_node(Node::new(NodeKind::File, "src/b.rs", "b"));
        g.add_node(Node::new(NodeKind::Doc, "README.md", "README"));
        g.add_edge(Edge::new("file:src/a.rs", "file:src/b.rs", EdgeKind::Imports));
        g.add_edge(Edge::new("doc:README.md", "file:src/a.rs", EdgeKind::Links));
        Server::new(g, PathBuf::from(".rkg/graph.json"))
    }

    fn call(server: &Server, name: &str, args: Value) -> Value {
        let req = json!({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": { "name": name, "arguments": args },
        });
        server.handle(&req).unwrap()["result"].clone()
    }

    /// Parses the text payload of a successful tools/call back into JSON.
    fn payload(result: &Value) -> Value {
        assert!(
            result.get("isError").is_none(),
            "unexpected tool error: {result}"
        );
        let text = result["content"][0]["text"].as_str().unwrap();
        serde_json::from_str(text).unwrap()
    }

    #[test]
    fn initialize_reports_tool_capability_and_echoes_protocol() {
        let s = server();
        let req = json!({
            "jsonrpc": "2.0", "id": 0, "method": "initialize",
            "params": { "protocolVersion": "2025-06-18" },
        });
        let resp = s.handle(&req).unwrap();
        assert_eq!(resp["result"]["protocolVersion"], "2025-06-18");
        assert!(resp["result"]["capabilities"]["tools"].is_object());
        assert_eq!(resp["result"]["serverInfo"]["name"], "rkg-mcp");
    }

    #[test]
    fn notifications_get_no_reply() {
        let s = server();
        let note = json!({ "jsonrpc": "2.0", "method": "notifications/initialized" });
        assert!(s.handle(&note).is_none());
    }

    #[test]
    fn tools_list_advertises_every_tool() {
        let s = server();
        let req = json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" });
        let resp = s.handle(&req).unwrap();
        let names: Vec<&str> = resp["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        for expected in ["build", "find_node", "neighbors", "context_pack", "subgraph", "path_between"] {
            assert!(names.contains(&expected), "missing tool {expected}");
        }
    }

    #[test]
    fn build_scans_a_repo_and_replaces_the_live_graph() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);

        // A throwaway repo with one Rust file.
        let root = std::env::temp_dir().join(format!(
            "rkg-mcp-build-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/lib.rs"), "pub fn hello() {}\n").unwrap();
        let out = root.join("graph.json");

        let s = server();
        let result = call(
            &s,
            "build",
            json!({ "path": root.to_str().unwrap(), "output": out.to_str().unwrap() }),
        );
        let built = payload(&result);
        assert!(built["nodes"].as_u64().unwrap() >= 2); // dir + file + symbol
        assert!(out.exists(), "graph.json should be written");

        // The live graph is now the scanned repo, not the seed fixture.
        let found = payload(&call(&s, "find_node", json!({ "query": "hello" })));
        assert!(!found.as_array().unwrap().is_empty());

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn find_node_returns_matches() {
        let s = server();
        let out = payload(&call(&s, "find_node", json!({ "query": "a.rs" })));
        assert_eq!(out.as_array().unwrap().len(), 1);
        assert_eq!(out[0]["id"], "file:src/a.rs");
    }

    #[test]
    fn neighbors_honours_direction_and_kind() {
        let s = server();
        let out = payload(&call(
            &s,
            "neighbors",
            json!({ "id": "file:src/a.rs", "direction": "out", "edge_kinds": ["imports"] }),
        ));
        assert_eq!(out, json!(["file:src/b.rs"]));
    }

    #[test]
    fn context_pack_starts_at_the_seed() {
        let s = server();
        let out = payload(&call(&s, "context_pack", json!({ "seed": "file:src/a.rs" })));
        assert_eq!(out["seed"], "file:src/a.rs");
        assert_eq!(out["entries"][0]["id"], "file:src/a.rs");
    }

    #[test]
    fn a_missing_seed_is_a_tool_error() {
        let s = server();
        let result = call(&s, "context_pack", json!({ "seed": "file:nope.rs" }));
        assert_eq!(result["isError"], true);
    }

    #[test]
    fn a_missing_required_argument_is_a_tool_error() {
        let s = server();
        let result = call(&s, "find_node", json!({}));
        assert_eq!(result["isError"], true);
    }

    #[test]
    fn an_unknown_method_is_a_jsonrpc_error() {
        let s = server();
        let req = json!({ "jsonrpc": "2.0", "id": 9, "method": "does/not/exist" });
        let resp = s.handle(&req).unwrap();
        assert_eq!(resp["error"]["code"], -32601);
    }
}
