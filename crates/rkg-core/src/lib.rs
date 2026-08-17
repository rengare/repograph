//! The typed knowledge-graph model, its on-disk form, and adjacency.
//!
//! This crate is deliberately free of GPU/window/parser dependencies — it is the
//! one place the CLI, the ingest passes, and (later) the MCP server agree on what a
//! knowledge graph *is*, so it can be unit-tested without a repo or a display. It
//! mirrors `gv-graph`'s discipline of a dependency-light core.
//!
//! Where `gv-graph::GraphData` is an anonymous, undirected, GPU-layout-oriented
//! graph (positions/colors/sizes), this model is **typed and directed**: every node
//! carries a [`NodeKind`] and metadata, every edge a [`EdgeKind`]. Projection down
//! to the `gv-graph` edge list for visualization lives in [`export`].

pub mod export;
pub mod query;

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Stable, human-meaningful node identity. Kind-prefixed so new kinds (notably
/// `sym:` symbols in a later phase) slot in without renumbering anything:
/// `dir:src`, `file:src/foo.rs`, `doc:README.md`, `sec:README.md#Format`,
/// `sym:src/foo.rs::parse`.
pub type NodeId = String;

/// What a node represents. `Symbol` is reserved now and populated once tree-sitter
/// extraction lands; keeping it in the enum from day one means the schema, the
/// serialized form, and the viewer's kind filter never have to change for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NodeKind {
    Dir,
    File,
    Doc,
    Section,
    Symbol,
}

impl NodeKind {
    /// Lowercase tag used in `nodes.tsv` and the `id` prefix.
    pub fn tag(self) -> &'static str {
        match self {
            NodeKind::Dir => "dir",
            NodeKind::File => "file",
            NodeKind::Doc => "doc",
            NodeKind::Section => "sec",
            NodeKind::Symbol => "sym",
        }
    }

    /// Parses a tag back into a kind; used by the viewer's sidecar loader.
    pub fn from_tag(tag: &str) -> Option<Self> {
        Some(match tag {
            "dir" => NodeKind::Dir,
            "file" => NodeKind::File,
            "doc" => NodeKind::Doc,
            "sec" => NodeKind::Section,
            "sym" => NodeKind::Symbol,
            _ => return None,
        })
    }
}

/// How two nodes relate. Directed: `from` → `to`. `Defines`/`Calls`/`References`
/// are reserved for symbol-level extraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EdgeKind {
    /// Directory contains a child dir/file, or a doc contains a section.
    Contains,
    /// Source file imports/uses another file (Rust `use`/`mod`, JS `import`).
    Imports,
    /// Markdown link to another in-repo file/doc.
    Links,
    /// Symbol references another symbol (reserved).
    References,
    /// File defines a symbol (reserved).
    Defines,
    /// Symbol calls another symbol (reserved).
    Calls,
}

impl EdgeKind {
    pub fn tag(self) -> &'static str {
        match self {
            EdgeKind::Contains => "contains",
            EdgeKind::Imports => "imports",
            EdgeKind::Links => "links",
            EdgeKind::References => "references",
            EdgeKind::Defines => "defines",
            EdgeKind::Calls => "calls",
        }
    }

    pub fn from_tag(tag: &str) -> Option<Self> {
        Some(match tag {
            "contains" => EdgeKind::Contains,
            "imports" => EdgeKind::Imports,
            "links" => EdgeKind::Links,
            "references" => EdgeKind::References,
            "defines" => EdgeKind::Defines,
            "calls" => EdgeKind::Calls,
            _ => return None,
        })
    }
}

/// A line span within a file, 1-based and inclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    pub start_line: u32,
    pub end_line: u32,
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// A type on a parameter or return value — either declared in the source or
/// derived by local, single-function heuristic inference (never cross-file).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeRef {
    pub ty: String,
    /// True when guessed from local syntax (a literal or constructor call) rather
    /// than read from a source annotation.
    #[serde(default, skip_serializing_if = "is_false")]
    pub inferred: bool,
}

impl TypeRef {
    pub fn declared(ty: impl Into<String>) -> Self {
        TypeRef { ty: ty.into(), inferred: false }
    }

    pub fn inferred(ty: impl Into<String>) -> Self {
        TypeRef { ty: ty.into(), inferred: true }
    }
}

/// A function parameter: its name and, when known, its type.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Param {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ty: Option<TypeRef>,
}

/// One node with its metadata. Everything past `name` is optional context an AI
/// consumer can use without opening the file; `signature`/`summary`/`span` fill in
/// for symbols and docs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub kind: NodeKind,
    /// Repo-relative path (for sections, `path#anchor`).
    pub path: String,
    /// Display name (file stem, symbol name, heading text).
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lang: Option<String>,
    /// Sub-kind for `Symbol` nodes: `fn`, `struct`, `method`, `class`, …
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub symbol_kind: Option<String>,
    /// Enclosing scope of a symbol — `impl Csr`, a class or module name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span: Option<Span>,
    /// Full declaration signature (params + return type) for symbols.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    /// Doc comment / summary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    /// Variable names declared in a symbol's scope (parameters + local declarations).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub locals: Vec<String>,
    /// Callee names invoked in a symbol's body (call-position identifiers) — the
    /// strongest structural signal of what the code does.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub calls: Vec<String>,
    /// Heuristic role of a symbol: `constructor`, `accessor`, `predicate`,
    /// `handler`, `test`, `factory`, `converter`, `io`, `entrypoint`, `mutator`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Parameters with optional (declared or locally inferred) types.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub params: Vec<Param>,
    /// Return type, declared or locally inferred.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub returns: Option<TypeRef>,
    /// A synthesized one-line description of what the symbol does, filled in only
    /// when no real doc comment (`summary`) is present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Lines of content, 0 for structural nodes.
    #[serde(default)]
    pub loc: u32,
}

impl Node {
    /// A structural node (dir/file/doc) with no source-derived metadata yet.
    pub fn new(kind: NodeKind, path: impl Into<String>, name: impl Into<String>) -> Self {
        let path = path.into();
        Node {
            id: format!("{}:{}", kind.tag(), path),
            kind,
            path,
            name: name.into(),
            lang: None,
            symbol_kind: None,
            container: None,
            span: None,
            signature: None,
            summary: None,
            locals: Vec::new(),
            calls: Vec::new(),
            role: None,
            params: Vec::new(),
            returns: None,
            description: None,
            loc: 0,
        }
    }
}

/// A directed, typed edge over [`NodeId`]s.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
    pub kind: EdgeKind,
    #[serde(default = "one")]
    pub weight: f32,
}

fn one() -> f32 {
    1.0
}

impl Edge {
    pub fn new(from: impl Into<NodeId>, to: impl Into<NodeId>, kind: EdgeKind) -> Self {
        Edge {
            from: from.into(),
            to: to.into(),
            kind,
            weight: 1.0,
        }
    }
}

/// The knowledge graph. `nodes`/`edges` are the serialized state; the index and
/// adjacency are derived and rebuilt after load via [`Graph::reindex`].
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,

    /// `id -> position in nodes`.
    #[serde(skip)]
    index: HashMap<NodeId, usize>,
    /// Per node, indices into `edges` for outgoing edges.
    #[serde(skip)]
    out_adj: Vec<Vec<usize>>,
    /// Per node, indices into `edges` for incoming edges.
    #[serde(skip)]
    in_adj: Vec<Vec<usize>>,
}

impl Graph {
    pub fn new() -> Self {
        Graph::default()
    }

    /// Inserts a node, or returns the existing index if its id is already present.
    /// Later structural passes may re-assert the same dir/file; first write wins.
    pub fn add_node(&mut self, node: Node) -> usize {
        if let Some(&i) = self.index.get(&node.id) {
            return i;
        }
        let i = self.nodes.len();
        self.index.insert(node.id.clone(), i);
        self.nodes.push(node);
        self.out_adj.push(Vec::new());
        self.in_adj.push(Vec::new());
        i
    }

    /// Adds an edge only if both endpoints exist and the same (from,to,kind) triple
    /// is not already present. Returns false if it was a dangling or duplicate edge.
    pub fn add_edge(&mut self, edge: Edge) -> bool {
        let (Some(&from), Some(&to)) =
            (self.index.get(&edge.from), self.index.get(&edge.to))
        else {
            return false;
        };
        if self.out_adj[from]
            .iter()
            .any(|&e| self.edges[e].to == edge.to && self.edges[e].kind == edge.kind)
        {
            return false;
        }
        let ei = self.edges.len();
        self.out_adj[from].push(ei);
        self.in_adj[to].push(ei);
        self.edges.push(edge);
        true
    }

    pub fn node(&self, id: &str) -> Option<&Node> {
        self.index.get(id).map(|&i| &self.nodes[i])
    }

    /// Mutable access to an existing node (e.g. to fill in a derived summary after
    /// the node was first inserted).
    pub fn node_mut(&mut self, id: &str) -> Option<&mut Node> {
        self.index.get(id).map(|&i| &mut self.nodes[i])
    }

    pub fn contains(&self, id: &str) -> bool {
        self.index.contains_key(id)
    }

    /// Outgoing edges of `id`.
    pub fn out_edges(&self, id: &str) -> impl Iterator<Item = &Edge> {
        self.index
            .get(id)
            .into_iter()
            .flat_map(|&i| self.out_adj[i].iter().map(|&e| &self.edges[e]))
    }

    /// Incoming edges of `id`.
    pub fn in_edges(&self, id: &str) -> impl Iterator<Item = &Edge> {
        self.index
            .get(id)
            .into_iter()
            .flat_map(|&i| self.in_adj[i].iter().map(|&e| &self.edges[e]))
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Rebuilds the id index and adjacency from `nodes`/`edges`. Call after
    /// deserializing, since those derived fields are `#[serde(skip)]`.
    pub fn reindex(&mut self) {
        self.index = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.id.clone(), i))
            .collect();
        self.out_adj = vec![Vec::new(); self.nodes.len()];
        self.in_adj = vec![Vec::new(); self.nodes.len()];
        for (ei, edge) in self.edges.iter().enumerate() {
            if let (Some(&from), Some(&to)) =
                (self.index.get(&edge.from), self.index.get(&edge.to))
            {
                self.out_adj[from].push(ei);
                self.in_adj[to].push(ei);
            }
        }
    }

    /// Serializes to pretty JSON at `path`.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating {}", parent.display()))?;
            }
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json).with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    /// Loads a graph from JSON and rebuilds its adjacency.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let json = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let mut graph: Graph =
            serde_json::from_str(&json).with_context(|| format!("parsing {}", path.display()))?;
        graph.reindex();
        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Graph {
        let mut g = Graph::new();
        g.add_node(Node::new(NodeKind::File, "src/a.rs", "a"));
        g.add_node(Node::new(NodeKind::File, "src/b.rs", "b"));
        g.add_edge(Edge::new("file:src/a.rs", "file:src/b.rs", EdgeKind::Imports));
        g
    }

    #[test]
    fn add_node_dedupes_by_id() {
        let mut g = Graph::new();
        let first = g.add_node(Node::new(NodeKind::File, "src/a.rs", "a"));
        let again = g.add_node(Node::new(NodeKind::File, "src/a.rs", "a"));
        assert_eq!(first, again);
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn add_edge_rejects_dangling_and_duplicate() {
        let mut g = sample();
        // Duplicate of the existing a->b import.
        assert!(!g.add_edge(Edge::new(
            "file:src/a.rs",
            "file:src/b.rs",
            EdgeKind::Imports
        )));
        // Dangling: endpoint does not exist.
        assert!(!g.add_edge(Edge::new(
            "file:src/a.rs",
            "file:src/missing.rs",
            EdgeKind::Imports
        )));
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn out_and_in_edges_resolve() {
        let g = sample();
        assert_eq!(g.out_edges("file:src/a.rs").count(), 1);
        assert_eq!(g.in_edges("file:src/b.rs").count(), 1);
        assert_eq!(g.in_edges("file:src/a.rs").count(), 0);
    }

    #[test]
    fn json_round_trip_preserves_and_reindexes() {
        let g = sample();
        let json = serde_json::to_string(&g).unwrap();
        let mut back: Graph = serde_json::from_str(&json).unwrap();
        back.reindex();
        assert_eq!(back.node_count(), 2);
        assert_eq!(back.out_edges("file:src/a.rs").count(), 1);
    }
}
