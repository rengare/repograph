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
        "# index\tid\tname\tkind\tpath\tspan\tsignature\tsymbol_kind\tcontainer\tdoc\tlocals\tcalls\trole\treturns\tdescription"
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
        // Locals can be multi-line destructuring patterns, so sanitize each one
        // (and the id/name/path cells) before joining into the tab-separated row.
        let locals = n
            .locals
            .iter()
            .map(|l| sanitize_cell(l))
            .collect::<Vec<_>>()
            .join(" ");
        let calls = n
            .calls
            .iter()
            .map(|c| sanitize_cell(c))
            .collect::<Vec<_>>()
            .join(" ");
        let role = n.role.as_deref().unwrap_or("");
        // A leading `~` marks a locally-inferred (vs declared) return type.
        let returns = n
            .returns
            .as_ref()
            .map(|t| if t.inferred { format!("~{}", t.ty) } else { t.ty.clone() })
            .map(|s| sanitize_cell(&s))
            .unwrap_or_default();
        let description = n.description.as_deref().map(sanitize_cell).unwrap_or_default();
        let id = sanitize_cell(&n.id);
        let name = sanitize_cell(&n.name);
        let path = sanitize_cell(&n.path);
        writeln!(
            buf,
            "{i}\t{id}\t{name}\t{}\t{path}\t{span}\t{signature}\t{symbol_kind}\t{container}\t{doc}\t{locals}\t{calls}\t{role}\t{returns}\t{description}",
            n.kind.tag()
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
    fn nodes_tsv_rows_survive_multiline_fields() {
        // A symbol whose signature and locals span multiple lines (as happens with
        // destructuring parameters in JS/TS) must still emit a single, well-formed
        // TSV row so the viewer's column-count parser doesn't choke.
        let mut g = Graph::new();
        let mut n = Node::new(NodeKind::Symbol, "src/a.tsx::View", "View");
        n.signature = Some("View = () => {\n  return <div/>;\n}".to_owned());
        n.locals = vec!["{\n  a,\n  b\n}".to_owned(), "c".to_owned()];
        g.add_node(n);

        let dir = std::env::temp_dir().join(format!("rkg-tsv-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("nodes.tsv");
        write_nodes_tsv(&g, &path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();

        for line in text.lines().filter(|l| !l.starts_with('#')) {
            let cols = line.split('\t').count();
            assert!(
                cols >= 5,
                "row has too few columns ({cols}): {line:?}"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn nodes_tsv_round_trips_semantic_columns() {
        use crate::{Param, TypeRef};
        let mut g = Graph::new();
        let mut n = Node::new(NodeKind::Symbol, "src/a.rs::build", "build");
        n.calls = vec!["read".to_owned(), "parse".to_owned()];
        n.role = Some("factory".to_owned());
        n.returns = Some(TypeRef::inferred("Csr")); // inferred -> `~Csr` on the wire
        n.params = vec![Param {
            name: "path".to_owned(),
            ty: Some(TypeRef::declared("&str")),
        }];
        n.description = Some("build (factory) takes path: &str; calls read, parse".to_owned());
        g.add_node(n);

        let dir = std::env::temp_dir().join(format!("rkg-tsv-rt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("nodes.tsv");
        write_nodes_tsv(&g, &path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();

        // Read back through the viewer's own sidecar parser.
        let gd = gv_graph::loader::from_exported_text("# from to\n", &text).unwrap();
        let meta = gd.meta.iter().find(|m| m.name == "build").expect("build meta");
        assert_eq!(meta.calls, vec!["read".to_owned(), "parse".to_owned()]);
        assert_eq!(meta.role.as_deref(), Some("factory"));
        assert_eq!(meta.returns.as_deref(), Some("Csr"));
        assert!(meta.returns_inferred, "the ~ marker should decode to inferred=true");
        assert!(meta.description.as_deref().unwrap().contains("factory"));
        std::fs::remove_dir_all(&dir).ok();
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
