//! Markdown structure extraction: ATX headings become `Section` nodes contained by
//! their document, and `[text](target)` links to in-repo files/docs become `Links`.

use rkg_core::{Edge, EdgeKind, Node, NodeKind};

use crate::path::{join_normalize, parent};
use crate::resolver::Resolver;

/// Extracts `(section nodes, edges)` for one markdown document at `rel`.
pub fn extract(content: &str, rel: &str, resolver: &Resolver) -> (Vec<Node>, Vec<Edge>) {
    let doc_id = format!("doc:{rel}");
    let dir = parent(rel);
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut in_fence = false;

    for raw in content.lines() {
        let line = raw.trim_end();
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }

        if let Some(text) = heading(line) {
            let slug = slugify(text);
            if slug.is_empty() {
                continue;
            }
            let path = format!("{rel}#{slug}");
            let node = Node::new(NodeKind::Section, path.clone(), text.to_string());
            let sec_id = node.id.clone();
            nodes.push(node);
            edges.push(Edge::new(doc_id.clone(), sec_id, EdgeKind::Contains));
        }

        for target in link_targets(line) {
            if is_external(&target) {
                continue;
            }
            let clean = target.split('#').next().unwrap_or(&target);
            if clean.is_empty() {
                continue; // pure in-page anchor
            }
            let base = join_normalize(&dir, clean);
            if let Some(to) = resolver.node_id(&base) {
                if to != doc_id {
                    edges.push(Edge::new(doc_id.clone(), to, EdgeKind::Links));
                }
            }
        }
    }
    (nodes, edges)
}

/// The text of an ATX heading (`#`..`######` followed by a space), or `None`.
fn heading(line: &str) -> Option<&str> {
    let trimmed = line.trim_start();
    let hashes = trimmed.chars().take_while(|c| *c == '#').count();
    if (1..=6).contains(&hashes) {
        let rest = &trimmed[hashes..];
        if rest.starts_with(' ') {
            return Some(rest.trim());
        }
    }
    None
}

/// GitHub-style anchor slug: lowercase, spaces to hyphens, drop other punctuation.
fn slugify(text: &str) -> String {
    let mut out = String::new();
    for c in text.trim().to_lowercase().chars() {
        if c.is_alphanumeric() {
            out.push(c);
        } else if c == ' ' || c == '-' || c == '_' {
            out.push('-');
        }
        // everything else is dropped
    }
    out.trim_matches('-').to_string()
}

/// Link targets on a line: for every `](target)` the `target` up to whitespace.
fn link_targets(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = line.as_bytes();
    let mut i = 0;
    while let Some(rel) = line[i..].find("](") {
        let start = i + rel + 2;
        if let Some(close) = line[start..].find(')') {
            let inside = &line[start..start + close];
            // Strip an optional `"title"` after the url.
            let target = inside.split_whitespace().next().unwrap_or("").to_string();
            if !target.is_empty() {
                out.push(target);
            }
            i = start + close + 1;
        } else {
            break;
        }
        if i >= bytes.len() {
            break;
        }
    }
    out
}

fn is_external(target: &str) -> bool {
    let t = target.trim_start();
    t.starts_with("http://")
        || t.starts_with("https://")
        || t.starts_with("mailto:")
        || t.starts_with('#')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolver(files: &[(&str, NodeKind)]) -> Resolver {
        let mut r = Resolver::new();
        for (f, k) in files {
            r.insert(f, *k);
        }
        r
    }

    #[test]
    fn headings_become_sections() {
        let r = resolver(&[]);
        let (nodes, edges) = extract("# Intro\n## The Format\n", "README.md", &r);
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].id, "sec:README.md#intro");
        assert_eq!(nodes[1].id, "sec:README.md#the-format");
        assert!(edges.iter().all(|e| e.kind == EdgeKind::Contains));
    }

    #[test]
    fn links_to_repo_files_resolve() {
        let r = resolver(&[("src/loader.rs", NodeKind::File), ("docs/x.md", NodeKind::Doc)]);
        let (_, edges) = extract(
            "see [loader](src/loader.rs) and [x](docs/x.md#anchor)\n",
            "README.md",
            &r,
        );
        let links: Vec<_> = edges.iter().filter(|e| e.kind == EdgeKind::Links).collect();
        assert_eq!(links.len(), 2);
        assert!(links.iter().any(|e| e.to == "file:src/loader.rs"));
        assert!(links.iter().any(|e| e.to == "doc:docs/x.md"));
    }

    #[test]
    fn external_and_anchor_links_ignored() {
        let r = resolver(&[]);
        let (_, edges) = extract(
            "[site](https://example.com) [top](#intro)\n",
            "README.md",
            &r,
        );
        assert!(edges.iter().all(|e| e.kind != EdgeKind::Links));
    }

    #[test]
    fn headings_in_code_fences_are_skipped() {
        let r = resolver(&[]);
        let (nodes, _) = extract("```\n# not a heading\n```\n# Real\n", "README.md", &r);
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].name, "Real");
    }
}
