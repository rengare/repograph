//! Heuristic C / C++ include extraction. `#include "x.h"` is a local include,
//! resolved relative to the including file's directory first, then by suffix match
//! anywhere in the repo (for `-I` include-dir layouts). `#include <x>` is a system
//! header and is skipped. Shared by the `c` and `cpp` languages.

use rkg_core::{Edge, EdgeKind};

use crate::path::{join_normalize, parent};
use crate::resolver::Resolver;

/// Extracts `Imports` edges from one C/C++ file at `rel`.
pub fn extract(content: &str, rel: &str, resolver: &Resolver) -> Vec<Edge> {
    let from = format!("file:{rel}");
    let dir = parent(rel);
    let mut edges = Vec::new();

    for raw in content.lines() {
        let line = raw.trim_start();
        let Some(rest) = line.strip_prefix('#') else {
            continue;
        };
        let rest = rest.trim_start();
        let Some(rest) = rest.strip_prefix("include") else {
            continue;
        };
        // Only quoted (local) includes; `<...>` system headers are skipped.
        let rest = rest.trim_start();
        let Some(inner) = rest.strip_prefix('"').and_then(|r| r.split('"').next()) else {
            continue;
        };
        if inner.is_empty() {
            continue;
        }

        // Relative to this file first, then anywhere in the repo.
        let relative = join_normalize(&dir, inner);
        let target = resolver
            .node_id(&relative)
            .or_else(|| resolver.resolve_suffix_path(inner));
        if let Some(to) = target {
            if to != from {
                edges.push(Edge::new(from.clone(), to, EdgeKind::Imports));
            }
        }
    }
    edges
}

#[cfg(test)]
mod tests {
    use super::*;
    use rkg_core::NodeKind;

    fn resolver(files: &[&str]) -> Resolver {
        let mut r = Resolver::new();
        for f in files {
            r.insert(f, NodeKind::File);
        }
        r
    }

    #[test]
    fn local_include_resolves_relative_to_the_file() {
        let r = resolver(&["src/main.c", "src/util.h"]);
        let edges = extract("#include \"util.h\"\n", "src/main.c", &r);
        assert_eq!(edges[0].to, "file:src/util.h");
    }

    #[test]
    fn include_falls_back_to_suffix_match() {
        let r = resolver(&["src/main.c", "include/lib/api.h"]);
        let edges = extract("#include \"lib/api.h\"\n", "src/main.c", &r);
        assert_eq!(edges[0].to, "file:include/lib/api.h");
    }

    #[test]
    fn system_include_is_skipped() {
        let r = resolver(&["src/main.c"]);
        assert!(extract("#include <stdio.h>\n", "src/main.c", &r).is_empty());
    }

    #[test]
    fn spacing_variations_still_parse() {
        let r = resolver(&["a.c", "b.h"]);
        let edges = extract("#  include   \"b.h\"\n", "a.c", &r);
        assert_eq!(edges[0].to, "file:b.h");
    }
}
