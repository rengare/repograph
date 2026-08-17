//! The read side: lookups, bounded traversal, and the token-budgeted
//! [`context_pack`] that makes the graph useful to an AI consumer.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};

use serde::Serialize;

use crate::{EdgeKind, Graph, Node, NodeId, NodeKind, Param, TypeRef};

/// Which direction to traverse edges in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Out,
    In,
    Both,
}

/// Case-insensitive substring search over node name, path, and id, with an
/// optional kind filter. Results are ordered by a light relevance heuristic:
/// exact name match first, then name-prefix, then everything else.
pub fn find_node<'g>(graph: &'g Graph, query: &str, kind: Option<NodeKind>) -> Vec<&'g Node> {
    let needle = query.to_lowercase();
    let mut hits: Vec<&Node> = graph
        .nodes
        .iter()
        .filter(|n| kind.is_none_or(|k| n.kind == k))
        .filter(|n| {
            n.name.to_lowercase().contains(&needle)
                || n.path.to_lowercase().contains(&needle)
                || n.id.to_lowercase().contains(&needle)
        })
        .collect();

    hits.sort_by_key(|n| {
        let name = n.name.to_lowercase();
        if name == needle {
            0
        } else if name.starts_with(&needle) {
            1
        } else if name.contains(&needle) {
            2
        } else {
            3
        }
    });
    hits
}

/// Node ids reachable from `start` within `depth` hops, following edges in
/// `direction`, optionally restricted to `edge_kinds`. Excludes `start` itself.
/// Order is breadth-first (nearest first).
pub fn neighbors(
    graph: &Graph,
    start: &str,
    depth: u32,
    edge_kinds: Option<&[EdgeKind]>,
    direction: Direction,
) -> Vec<NodeId> {
    let mut seen: HashMap<&str, ()> = HashMap::new();
    let mut out = Vec::new();
    if !graph.contains(start) {
        return out;
    }
    let mut queue: VecDeque<(&str, u32)> = VecDeque::new();
    queue.push_back((start, 0));
    seen.insert(start, ());

    while let Some((id, d)) = queue.pop_front() {
        if d == depth {
            continue;
        }
        for (neighbor, _) in step(graph, id, edge_kinds, direction) {
            if seen.insert(neighbor, ()).is_none() {
                out.push(neighbor.to_string());
                queue.push_back((neighbor, d + 1));
            }
        }
    }
    out
}

/// One hop from `id`: yields `(neighbor_id, edge_kind)` pairs honoring direction
/// and the kind filter.
fn step<'g>(
    graph: &'g Graph,
    id: &str,
    edge_kinds: Option<&[EdgeKind]>,
    direction: Direction,
) -> Vec<(&'g str, EdgeKind)> {
    let allow = |k: EdgeKind| edge_kinds.is_none_or(|ks| ks.contains(&k));
    let mut result = Vec::new();
    if matches!(direction, Direction::Out | Direction::Both) {
        for e in graph.out_edges(id) {
            if allow(e.kind) {
                result.push((e.to.as_str(), e.kind));
            }
        }
    }
    if matches!(direction, Direction::In | Direction::Both) {
        for e in graph.in_edges(id) {
            if allow(e.kind) {
                result.push((e.from.as_str(), e.kind));
            }
        }
    }
    result
}

/// The induced subgraph over `ids` — every node in `ids` that exists, plus every
/// edge whose endpoints are both in the set. Returned graph is reindexed.
pub fn subgraph(graph: &Graph, ids: &[String]) -> Graph {
    let keep: HashMap<&str, ()> = ids.iter().map(|s| (s.as_str(), ())).collect();
    let mut out = Graph::new();
    for id in ids {
        if let Some(n) = graph.node(id) {
            out.add_node(n.clone());
        }
    }
    for edge in &graph.edges {
        if keep.contains_key(edge.from.as_str()) && keep.contains_key(edge.to.as_str()) {
            out.add_edge(edge.clone());
        }
    }
    out
}

/// Shortest path between `a` and `b` treating edges as undirected, or `None` if
/// they are disconnected. Includes both endpoints.
pub fn path_between(graph: &Graph, a: &str, b: &str) -> Option<Vec<NodeId>> {
    if !graph.contains(a) || !graph.contains(b) {
        return None;
    }
    if a == b {
        return Some(vec![a.to_string()]);
    }
    let mut prev: HashMap<&str, &str> = HashMap::new();
    let mut queue: VecDeque<&str> = VecDeque::new();
    queue.push_back(a);
    prev.insert(a, a);

    while let Some(id) = queue.pop_front() {
        for (neighbor, _) in step(graph, id, None, Direction::Both) {
            if prev.contains_key(neighbor) {
                continue;
            }
            prev.insert(neighbor, id);
            if neighbor == b {
                let mut path = vec![b.to_string()];
                let mut cur = b;
                while cur != a {
                    cur = prev[cur];
                    path.push(cur.to_string());
                }
                path.reverse();
                return Some(path);
            }
            queue.push_back(neighbor);
        }
    }
    None
}

/// One entry in a [`ContextPack`]: a node plus why it was included and how far it
/// sits from the seed.
#[derive(Clone, Debug, Serialize)]
pub struct ContextEntry {
    pub id: NodeId,
    pub kind: NodeKind,
    pub name: String,
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol_kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub locals: Vec<String>,
    /// Callee names invoked in the body — what this code does.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub calls: Vec<String>,
    /// Heuristic behavioural role.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Parameters with declared-or-inferred types.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub params: Vec<Param>,
    /// Return type, declared or locally inferred.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub returns: Option<TypeRef>,
    /// Synthesised "what it does" line for undocumented symbols.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Hops from the seed (0 = the seed itself).
    pub distance: u32,
    /// Human-readable reason this entry earned its place.
    pub reason: String,
    /// Estimated token cost of this entry's text.
    pub est_tokens: u32,
}

/// A compact, budgeted bundle of context around a seed — the token-saving payload
/// an AI reads instead of opening files.
#[derive(Clone, Debug, Serialize)]
pub struct ContextPack {
    pub seed: NodeId,
    pub entries: Vec<ContextEntry>,
    pub est_tokens: u32,
}

/// Rough token estimate: ~4 chars per token over a node's textual footprint.
fn estimate_tokens(node: &Node) -> u32 {
    let chars = node.id.len()
        + node.name.len()
        + node.path.len()
        + node.signature.as_deref().map_or(0, str::len)
        + node.summary.as_deref().map_or(0, str::len)
        + node.description.as_deref().map_or(0, str::len)
        + node.calls.iter().map(|c| c.len() + 1).sum::<usize>()
        + node
            .params
            .iter()
            .map(|p| p.name.len() + p.ty.as_ref().map_or(0, |t| t.ty.len()) + 2)
            .sum::<usize>()
        + node.returns.as_ref().map_or(0, |t| t.ty.len());
    (chars / 4).max(1) as u32
}

/// Relevance cost of crossing an edge of a given kind. Semantic edges (what
/// imports/links/calls what) are cheap; structural containment is expensive, so a
/// file's imports rank far above its directory siblings when packing context.
fn edge_cost(kind: EdgeKind) -> u32 {
    match kind {
        EdgeKind::Imports
        | EdgeKind::Links
        | EdgeKind::Defines
        | EdgeKind::Calls
        | EdgeKind::References => 1,
        EdgeKind::Contains => 6,
    }
}

/// Builds a ranked, token-budgeted neighborhood around `seed`.
///
/// Reachability is explored by *relevance cost* (a shortest-path search where each
/// edge kind is weighted by [`edge_cost`]) rather than raw hops, so semantically
/// related nodes — imports, links, definitions — are preferred over structural
/// directory neighbors that a plain BFS would surface just as eagerly. Entries are
/// admitted in ascending cost until `token_budget` would be exceeded; the seed is
/// always first.
pub fn context_pack(graph: &Graph, seed: &str, token_budget: u32) -> Option<ContextPack> {
    graph.node(seed)?;

    // Dijkstra from the seed over relevance cost.
    let mut best: HashMap<&str, u32> = HashMap::new();
    let mut hops: HashMap<&str, u32> = HashMap::new();
    let mut via: HashMap<&str, EdgeKind> = HashMap::new();
    let mut heap: BinaryHeap<Reverse<(u32, u32, &str)>> = BinaryHeap::new();
    best.insert(seed, 0);
    hops.insert(seed, 0);
    heap.push(Reverse((0, 0, seed)));

    // Cost-ordered visitation order, seed first.
    let mut order: Vec<(&str, u32, u32)> = Vec::new();
    while let Some(Reverse((cost, hop, id))) = heap.pop() {
        if cost > best[id] {
            continue; // stale heap entry
        }
        order.push((id, cost, hop));
        for (neighbor, kind) in step(graph, id, None, Direction::Both) {
            let next = cost + edge_cost(kind);
            if best.get(neighbor).is_none_or(|&c| next < c) {
                best.insert(neighbor, next);
                hops.insert(neighbor, hop + 1);
                via.insert(neighbor, kind);
                heap.push(Reverse((next, hop + 1, neighbor)));
            }
        }
    }

    let mut pack = ContextPack {
        seed: seed.to_string(),
        entries: Vec::new(),
        est_tokens: 0,
    };
    for (id, _cost, hop) in order {
        let node = graph.node(id).expect("id came from the graph");
        let est = estimate_tokens(node);
        if hop != 0 && pack.est_tokens + est > token_budget {
            continue; // over budget; keep scanning for cheaper cousins
        }
        let reason = if hop == 0 {
            "seed".to_string()
        } else {
            format!(
                "{} hop(s) away via {}",
                hop,
                via.get(id).map_or("edge", |k| k.tag())
            )
        };
        pack.est_tokens += est;
        pack.entries.push(ContextEntry {
            id: node.id.clone(),
            kind: node.kind,
            name: node.name.clone(),
            path: node.path.clone(),
            symbol_kind: node.symbol_kind.clone(),
            container: node.container.clone(),
            signature: node.signature.clone(),
            summary: node.summary.clone(),
            locals: node.locals.clone(),
            calls: node.calls.clone(),
            role: node.role.clone(),
            params: node.params.clone(),
            returns: node.returns.clone(),
            description: node.description.clone(),
            distance: hop,
            reason,
            est_tokens: est,
        });
    }
    Some(pack)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Edge, Node};

    fn chain() -> Graph {
        // a -> b -> c, and a doc linking to a.
        let mut g = Graph::new();
        for name in ["a", "b", "c"] {
            g.add_node(Node::new(NodeKind::File, format!("src/{name}.rs"), name));
        }
        g.add_node(Node::new(NodeKind::Doc, "README.md", "README"));
        g.add_edge(Edge::new("file:src/a.rs", "file:src/b.rs", EdgeKind::Imports));
        g.add_edge(Edge::new("file:src/b.rs", "file:src/c.rs", EdgeKind::Imports));
        g.add_edge(Edge::new("doc:README.md", "file:src/a.rs", EdgeKind::Links));
        g
    }

    #[test]
    fn find_ranks_exact_name_first() {
        let g = chain();
        let hits = find_node(&g, "a", None);
        assert_eq!(hits[0].name, "a");
    }

    #[test]
    fn find_respects_kind_filter() {
        let g = chain();
        let docs = find_node(&g, "readme", Some(NodeKind::Doc));
        assert_eq!(docs.len(), 1);
        assert!(find_node(&g, "readme", Some(NodeKind::File)).is_empty());
    }

    #[test]
    fn neighbors_are_depth_bounded() {
        let g = chain();
        let d1 = neighbors(&g, "file:src/a.rs", 1, None, Direction::Both);
        assert!(d1.contains(&"file:src/b.rs".to_string()));
        assert!(d1.contains(&"doc:README.md".to_string()));
        assert!(!d1.contains(&"file:src/c.rs".to_string()));

        let d2 = neighbors(&g, "file:src/a.rs", 2, None, Direction::Both);
        assert!(d2.contains(&"file:src/c.rs".to_string()));
    }

    #[test]
    fn neighbors_respect_direction_and_kind() {
        let g = chain();
        // Only outgoing imports from a: just b (the doc link is incoming).
        let out = neighbors(
            &g,
            "file:src/a.rs",
            1,
            Some(&[EdgeKind::Imports]),
            Direction::Out,
        );
        assert_eq!(out, vec!["file:src/b.rs".to_string()]);
    }

    #[test]
    fn path_between_is_undirected() {
        let g = chain();
        let path = path_between(&g, "doc:README.md", "file:src/c.rs").unwrap();
        assert_eq!(
            path,
            vec![
                "doc:README.md",
                "file:src/a.rs",
                "file:src/b.rs",
                "file:src/c.rs",
            ]
        );
    }

    #[test]
    fn context_pack_starts_with_seed_and_respects_budget() {
        let g = chain();
        let pack = context_pack(&g, "file:src/a.rs", 1000).unwrap();
        assert_eq!(pack.entries[0].id, "file:src/a.rs");
        assert_eq!(pack.entries[0].distance, 0);
        // Everything is reachable and cheap, so all four nodes fit.
        assert_eq!(pack.entries.len(), 4);

        // A tiny budget still yields the seed and stays within budget afterward.
        let tight = context_pack(&g, "file:src/a.rs", 1).unwrap();
        assert_eq!(tight.entries[0].distance, 0);
    }
}
