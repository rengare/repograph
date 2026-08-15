//! Projection of the typed knowledge graph down to the visualizer's format.
//!
//! The viewer (`gv-graph`) speaks an anonymous integer edge list plus — with this
//! project's extension — a node-attribute sidecar. Node ids can contain spaces
//! (paths, headings), which the whitespace-split loader could not round-trip, so we
//! emit **dense integer `from to` pairs** keyed by node position and carry the real
//! id/name/kind/path in `nodes.tsv` alongside.

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};

use crate::{EdgeKind, Graph};

/// Which edge kinds to include in the projection. `None` means all.
#[derive(Clone, Debug, Default)]
pub struct ExportOptions {
    pub edge_kinds: Option<Vec<EdgeKind>>,
}

impl ExportOptions {
    fn includes(&self, kind: EdgeKind) -> bool {
        match &self.edge_kinds {
            None => true,
            Some(kinds) => kinds.contains(&kind),
        }
    }
}

fn dense_index(graph: &Graph) -> HashMap<&str, u32> {
    graph
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.id.as_str(), i as u32))
        .collect()
}

/// Writes the integer edge list. Returns the number of edges written.
pub fn write_edges(graph: &Graph, path: impl AsRef<Path>, opts: &ExportOptions) -> Result<usize> {
    let path = path.as_ref();
    let index = dense_index(graph);
    let mut out = String::new();
    out.push_str("# from to (dense indices; see the nodes sidecar for identities)\n");
    let mut written = 0;
    for edge in &graph.edges {
        if !opts.includes(edge.kind) {
            continue;
        }
        // Endpoints are guaranteed present: add_edge rejects dangling edges.
        let (from, to) = (index[edge.from.as_str()], index[edge.to.as_str()]);
        out.push_str(&format!("{from} {to}\n"));
        written += 1;
    }
    std::fs::write(path, out).with_context(|| format!("writing {}", path.display()))?;
    Ok(written)
}

/// Writes the node-attribute sidecar consumed by the searchable viewer:
/// `index<TAB>id<TAB>name<TAB>kind<TAB>path<TAB>span<TAB>signature<TAB>symbol_kind<TAB>container<TAB>doc`,
/// one row per node in dense order. Trailing columns are optional — a shorter
/// sidecar still loads — so the layout can grow without breaking older files.
pub fn write_nodes_tsv(graph: &Graph, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let mut buf = Vec::new();
    writeln!(
        buf,
        "# index\tid\tname\tkind\tpath\tspan\tsignature\tsymbol_kind\tcontainer\tdoc\tlocals"
    )?;
    for (i, n) in graph.nodes.iter().enumerate() {
        let span = n
            .span
            .map(|s| format!("{}-{}", s.start_line, s.end_line))
            .unwrap_or_default();
        // Free-text cells could contain tabs/newlines that break the columns, so
        // flatten any whitespace to single spaces.
        let signature = n.signature.as_deref().map(sanitize_cell).unwrap_or_default();
        let symbol_kind = n.symbol_kind.as_deref().unwrap_or("");
        let container = n.container.as_deref().map(sanitize_cell).unwrap_or_default();
        let doc = n.summary.as_deref().map(sanitize_cell).unwrap_or_default();
        let locals = n.locals.join(" ");
        writeln!(
            buf,
            "{i}\t{}\t{}\t{}\t{}\t{span}\t{signature}\t{symbol_kind}\t{container}\t{doc}\t{locals}",
            n.id,
            n.name,
            n.kind.tag(),
            n.path
        )?;
    }
    std::fs::write(path, buf).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Replaces tabs and any control whitespace with single spaces so a value is
/// safe to place in a TSV cell.
fn sanitize_cell(s: &str) -> String {
    s.chars()
        .map(|c| if c == '\t' || c == '\n' || c == '\r' { ' ' } else { c })
        .collect()
}

/// Builds a `gv_graph::GraphData` directly from the knowledge graph, reusing the
/// viewer's `Edge`/`Csr` types. Node state is left zeroed for the seeder to fill;
/// `labels[i]` is node `i`'s id. Useful for embedding the viewer in-process.
pub fn to_graph_data(graph: &Graph, opts: &ExportOptions) -> gv_graph::GraphData {
    let index = dense_index(graph);
    let labels: Vec<String> = graph.nodes.iter().map(|n| n.id.clone()).collect();
    let edges: Vec<gv_graph::Edge> = graph
        .edges
        .iter()
        .filter(|e| opts.includes(e.kind))
        .map(|e| gv_graph::Edge {
            from: index[e.from.as_str()],
            to: index[e.to.as_str()],
        })
        .collect();
    let adjacency = gv_graph::Csr::build(labels.len(), &edges);
    gv_graph::GraphData {
        nodes: vec![gv_graph::Node::default(); labels.len()],
        edges,
        adjacency,
        labels,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Edge, Node, NodeKind};

    fn sample() -> Graph {
        let mut g = Graph::new();
        g.add_node(Node::new(NodeKind::File, "src/a.rs", "a"));
        g.add_node(Node::new(NodeKind::File, "src/b.rs", "b"));
        g.add_node(Node::new(NodeKind::Doc, "README.md", "README"));
        g.add_edge(Edge::new("file:src/a.rs", "file:src/b.rs", EdgeKind::Imports));
        g.add_edge(Edge::new("doc:README.md", "file:src/a.rs", EdgeKind::Links));
        g
    }

    #[test]
    fn edge_filter_selects_kinds() {
        let g = sample();
        let all = ExportOptions::default();
        let gd = to_graph_data(&g, &all);
        assert_eq!(gd.edges.len(), 2);
        assert_eq!(gd.labels.len(), 3);

        let imports_only = ExportOptions {
            edge_kinds: Some(vec![EdgeKind::Imports]),
        };
        let gd = to_graph_data(&g, &imports_only);
        assert_eq!(gd.edges.len(), 1);
    }

    #[test]
    fn edges_are_written_as_dense_indices() {
        let g = sample();
        let dir = std::env::temp_dir().join(format!("rkg-export-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let edges_path = dir.join("repo.edges");
        let n = write_edges(&g, &edges_path, &ExportOptions::default()).unwrap();
        assert_eq!(n, 2);
        let text = std::fs::read_to_string(&edges_path).unwrap();
        // a=0, b=1, README=2: a->b is "0 1", README->a is "2 0".
        assert!(text.contains("0 1"));
        assert!(text.contains("2 0"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
