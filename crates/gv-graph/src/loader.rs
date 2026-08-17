//! Edge-list parsing and id interning.
//!
//! Input is one `from to` pair per line, whitespace separated, ids arbitrary
//! strings. Unlike the original's `while (inputFile >> x >> y)` loop this skips
//! `#` comment headers, which every SNAP dataset in `data/` carries and which
//! the original silently parsed as node ids.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use anyhow::{Context, Result};

use crate::{Csr, Edge, GraphData, NodeCategory, NodeMeta};

/// Parses an edge list into dense indices plus the labels they came from.
///
/// Labels are sorted before indices are assigned — matching the original's
/// `sort` + `unique` + map-build — so the same file always produces the same
/// node numbering regardless of the order edges appear in.
pub fn parse(reader: impl Read) -> Result<(Vec<Edge>, Vec<String>)> {
    let mut pairs: Vec<(String, String)> = Vec::new();
    let mut labels: Vec<String> = Vec::new();

    for (line_number, line) in BufReader::new(reader).lines().enumerate() {
        let line = line.with_context(|| format!("reading line {}", line_number + 1))?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('%') {
            continue;
        }

        let mut fields = line.split_whitespace();
        let (Some(from), Some(to)) = (fields.next(), fields.next()) else {
            anyhow::bail!("line {}: expected two fields, got {line:?}", line_number + 1);
        };

        labels.push(from.to_owned());
        labels.push(to.to_owned());
        pairs.push((from.to_owned(), to.to_owned()));
    }

    labels.sort_unstable();
    labels.dedup();

    let index: HashMap<&str, u32> = labels
        .iter()
        .enumerate()
        .map(|(i, label)| (label.as_str(), i as u32))
        .collect();

    let edges = pairs
        .iter()
        .map(|(from, to)| Edge {
            from: index[from.as_str()],
            to: index[to.as_str()],
        })
        .collect();

    Ok((edges, labels))
}

/// Loads an edge list from disk and builds the adjacency. Node state is left
/// zeroed; call [`crate::seed`] to place and colour the nodes.
pub fn load(path: impl AsRef<Path>) -> Result<GraphData> {
    let path = path.as_ref();
    let file = std::fs::File::open(path)
        .with_context(|| format!("opening edge list {}", path.display()))?;
    let (edges, labels) = parse(file)
        .with_context(|| format!("parsing edge list {}", path.display()))?;

    let adjacency = Csr::build(labels.len(), &edges);
    Ok(GraphData {
        nodes: vec![Default::default(); labels.len()],
        edges,
        adjacency,
        labels,
        meta: Vec::new(),
    })
}

/// Loads an edge list together with a knowledge-graph node sidecar (`nodes.tsv`,
/// as written by `rkg export`). Unlike [`load`], node indices come straight from
/// the sidecar's `index` column and the edge file's integers are used verbatim —
/// **no lexicographic re-interning** — so the two files stay aligned (`"10"` does
/// not sort ahead of `"2"`). Populates [`GraphData::meta`] so the viewer can search
/// by name/path and colour by kind.
pub fn load_with_sidecar(edges: impl AsRef<Path>, nodes: impl AsRef<Path>) -> Result<GraphData> {
    let nodes = nodes.as_ref();
    let sidecar = std::fs::read_to_string(nodes)
        .with_context(|| format!("opening node sidecar {}", nodes.display()))?;
    let edges_path = edges.as_ref();
    let edge_text = std::fs::read_to_string(edges_path)
        .with_context(|| format!("opening edge list {}", edges_path.display()))?;
    from_exported_text(&edge_text, &sidecar)
        .with_context(|| format!("parsing graph from {}", edges_path.display()))
}

/// Builds a searchable graph from the text of `rkg export`'s `repo.edges` and
/// `nodes.tsv` files. Shared by the native file loader and browser viewer so the
/// two surfaces accept precisely the same artifact format.
pub fn from_exported_text(edges_text: &str, nodes_text: &str) -> Result<GraphData> {
    let meta = parse_sidecar(nodes_text).context("parsing node sidecar")?;
    let edges = parse_indexed_edges(edges_text, meta.len()).context("parsing indexed edges")?;

    let labels = meta.iter().map(|m| m.name.clone()).collect::<Vec<_>>();
    let adjacency = Csr::build(meta.len(), &edges);
    Ok(GraphData {
        nodes: vec![Default::default(); meta.len()],
        edges,
        adjacency,
        labels,
        meta,
    })
}

/// Parses a `nodes.tsv` sidecar (`index<TAB>id<TAB>name<TAB>kind<TAB>path`) into a
/// dense `Vec<NodeMeta>` ordered by the `index` column. `#` comment lines skipped.
fn parse_sidecar(text: &str) -> Result<Vec<NodeMeta>> {
    let mut rows: Vec<(usize, NodeMeta)> = Vec::new();
    let mut max_index = 0;
    for (n, line) in text.lines().enumerate() {
        let line = line.trim_end();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut cols = line.split('\t');
        let (Some(index), Some(id), Some(name), Some(kind), Some(path)) = (
            cols.next(),
            cols.next(),
            cols.next(),
            cols.next(),
            cols.next(),
        ) else {
            anyhow::bail!("line {}: expected at least 5 tab-separated columns", n + 1);
        };
        // Every column past `path` is optional (older/narrower sidecars omit them).
        let optional = |c: Option<&str>| c.filter(|s| !s.is_empty()).map(str::to_owned);
        let span = cols.next().and_then(parse_span);
        let signature = optional(cols.next());
        let symbol_kind = optional(cols.next());
        let container = optional(cols.next());
        let doc = optional(cols.next());
        let split_names = |c: Option<&str>| -> Vec<String> {
            c.filter(|s| !s.is_empty())
                .map(|s| s.split_whitespace().map(str::to_owned).collect())
                .unwrap_or_default()
        };
        let locals = split_names(cols.next());
        let calls = split_names(cols.next());
        let role = optional(cols.next());
        // A leading `~` marks a locally-inferred (vs declared) return type.
        let (returns, returns_inferred) = match optional(cols.next()) {
            Some(s) => match s.strip_prefix('~') {
                Some(rest) => (Some(rest.to_owned()), true),
                None => (Some(s), false),
            },
            None => (None, false),
        };
        let description = optional(cols.next());

        let index: usize = index
            .parse()
            .with_context(|| format!("line {}: bad index {index:?}", n + 1))?;
        max_index = max_index.max(index);
        rows.push((
            index,
            NodeMeta {
                id: id.to_owned(),
                name: name.to_owned(),
                kind: NodeCategory::from_tag(kind),
                path: path.to_owned(),
                span,
                signature,
                symbol_kind,
                container,
                doc,
                locals,
                calls,
                role,
                returns,
                returns_inferred,
                description,
            },
        ));
    }

    let mut meta = vec![NodeMeta::default(); max_index + 1];
    for (index, m) in rows {
        meta[index] = m;
    }
    Ok(meta)
}

/// Parses a `start-end` span cell into a line range, or `None` if empty/malformed.
fn parse_span(cell: &str) -> Option<(u32, u32)> {
    let (start, end) = cell.split_once('-')?;
    Some((start.parse().ok()?, end.parse().ok()?))
}

/// Parses an edge list of integer index pairs (as [`load_with_sidecar`] expects),
/// validating that every index is within `node_count`.
fn parse_indexed_edges(text: &str, node_count: usize) -> Result<Vec<Edge>> {
    let mut edges = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('%') {
            continue;
        }
        let mut fields = line.split_whitespace();
        let (Some(from), Some(to)) = (fields.next(), fields.next()) else {
            anyhow::bail!("line {}: expected two fields, got {line:?}", n + 1);
        };
        let from: u32 = from.parse().with_context(|| format!("line {}", n + 1))?;
        let to: u32 = to.parse().with_context(|| format!("line {}", n + 1))?;
        if from as usize >= node_count || to as usize >= node_count {
            anyhow::bail!("line {}: edge {from}->{to} exceeds node count {node_count}", n + 1);
        }
        edges.push(Edge { from, to });
    }
    Ok(edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assigns_dense_indices_in_sorted_label_order() {
        let (edges, labels) = parse("b c\na b\n".as_bytes()).unwrap();

        assert_eq!(labels, vec!["a", "b", "c"]);
        assert_eq!(
            edges,
            vec![Edge { from: 1, to: 2 }, Edge { from: 0, to: 1 }]
        );
    }

    #[test]
    fn skips_snap_comment_headers_and_blank_lines() {
        let input = "# Directed graph\n# FromNodeId\tToNodeId\n\n0 1\n1 2\n";
        let (edges, labels) = parse(input.as_bytes()).unwrap();

        assert_eq!(labels, vec!["0", "1", "2"]);
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn tolerates_tabs_and_trailing_columns() {
        // Some SNAP files carry a weight or timestamp in a third column.
        let (edges, _) = parse("0\t1\t-1\n1\t2\t1\n".as_bytes()).unwrap();
        assert_eq!(edges, vec![Edge { from: 0, to: 1 }, Edge { from: 1, to: 2 }]);
    }

    #[test]
    fn rejects_a_line_with_one_field() {
        let error = parse("0 1\n2\n".as_bytes()).unwrap_err();
        assert!(error.to_string().contains("expected two fields"));
    }

    /// Labels sort lexicographically, not numerically — "10" precedes "9".
    /// That is the original's behaviour and callers depend on it only for
    /// determinism, not for ordering, so it is pinned here rather than fixed.
    #[test]
    fn label_order_is_lexicographic() {
        let (_, labels) = parse("9 10\n".as_bytes()).unwrap();
        assert_eq!(labels, vec!["10", "9"]);
    }

    #[test]
    fn sidecar_preserves_index_order_without_re_sorting() {
        // Indices 0,1,2,10 — a lexicographic sort would misorder 10 before 2.
        let tsv = "# index\tid\tname\tkind\tpath\n\
                   0\tdir:.\troot\tdir\t.\n\
                   1\tfile:a.rs\ta\tfile\ta.rs\n\
                   2\tfile:b.rs\tb\tfile\tb.rs\n\
                   10\tdoc:R.md\tR\tdoc\tR.md\n";
        let meta = parse_sidecar(tsv).unwrap();
        assert_eq!(meta.len(), 11); // 0..=10
        assert_eq!(meta[10].kind, NodeCategory::Doc);
        assert_eq!(meta[1].name, "a");
        // Gaps (indices 3..=9) default to Unknown.
        assert_eq!(meta[5].kind, NodeCategory::Unknown);
    }

    #[test]
    fn indexed_edges_use_integers_verbatim() {
        let edges = parse_indexed_edges("# c\n0 10\n2 1\n", 11).unwrap();
        assert_eq!(edges, vec![Edge { from: 0, to: 10 }, Edge { from: 2, to: 1 }]);
    }

    #[test]
    fn indexed_edges_reject_out_of_range() {
        let err = parse_indexed_edges("0 5\n", 3).unwrap_err();
        assert!(err.to_string().contains("exceeds node count"));
    }

    #[test]
    fn exported_text_loads_the_same_graph_as_the_native_viewer() {
        let nodes = "# index\tid\tname\tkind\tpath\n\
                     0\tdir:.\troot\tdir\t.\n\
                     1\tfile:src/a.rs\ta\tfile\tsrc/a.rs\n";
        let graph = from_exported_text("0 1\n", nodes).unwrap();

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
        assert_eq!(graph.meta[1].name, "a");
        assert_eq!(graph.adjacency.neighbors_of(0), &[1]);
    }
}
