//! Heuristic JS/TS import extraction: `import … from '…'`, `export … from '…'`,
//! side-effect `import '…'`, and `require('…')`. Only *relative* specifiers are
//! resolved to files; bare specifiers (packages) are left out of the graph.

use rkg_core::{Edge, EdgeKind};

use crate::path::{join_normalize, parent};
use crate::resolver::Resolver;

/// Suffix candidates tried, in order, when resolving a relative specifier.
const SUFFIXES: &[&str] = &[
    "", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", "/index.ts", "/index.tsx", "/index.js",
    "/index.jsx",
];

/// Extracts `Imports` edges out of one JS/TS source file at `rel`.
pub fn extract(content: &str, rel: &str, resolver: &Resolver) -> Vec<Edge> {
    let from = format!("file:{rel}");
    let dir = parent(rel);
    let mut edges = Vec::new();

    for raw in content.lines() {
        let line = raw.trim();
        if line.starts_with("//") {
            continue;
        }
        for spec in specifiers(line) {
            if !is_relative(&spec) {
                continue;
            }
            let base = join_normalize(&dir, &spec);
            if let Some(target) = resolver.resolve_with_suffixes(&base, SUFFIXES) {
                if target != from {
                    edges.push(Edge::new(from.clone(), target, EdgeKind::Imports));
                }
            }
        }
    }
    edges
}

/// Module specifiers referenced on one line (usually zero or one).
fn specifiers(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(s) = quoted_after(line, " from ") {
        out.push(s);
    }
    if let Some(s) = quoted_after(line, "require(") {
        out.push(s);
    }
    // Side-effect import: `import '…';`.
    if out.is_empty() && line.starts_with("import ") {
        if let Some(rest) = line.strip_prefix("import ") {
            if let Some(s) = leading_quoted(rest.trim()) {
                out.push(s);
            }
        }
    }
    out
}

/// The string literal immediately following the first occurrence of `marker`.
fn quoted_after(line: &str, marker: &str) -> Option<String> {
    let idx = line.find(marker)? + marker.len();
    leading_quoted(line[idx..].trim_start())
}

/// A leading `'…'`, `"…"`, or `` `…` `` literal at the start of `s`.
fn leading_quoted(s: &str) -> Option<String> {
    let quote = s.chars().next()?;
    if !matches!(quote, '\'' | '"' | '`') {
        return None;
    }
    let rest = &s[quote.len_utf8()..];
    let end = rest.find(quote)?;
    Some(rest[..end].to_string())
}

fn is_relative(spec: &str) -> bool {
    spec.starts_with("./") || spec.starts_with("../") || spec.starts_with('/')
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
    fn import_from_relative_resolves_with_extension_guess() {
        let r = resolver(&["src/app.ts", "src/util.ts"]);
        let edges = extract("import { x } from './util';\n", "src/app.ts", &r);
        assert_eq!(edges[0].to, "file:src/util.ts");
    }

    #[test]
    fn import_resolves_index_file() {
        let r = resolver(&["src/app.ts", "src/util/index.ts"]);
        let edges = extract("import x from './util';\n", "src/app.ts", &r);
        assert_eq!(edges[0].to, "file:src/util/index.ts");
    }

    #[test]
    fn require_and_parent_dir() {
        let r = resolver(&["src/a/app.js", "src/shared.js"]);
        let edges = extract("const s = require('../shared');\n", "src/a/app.js", &r);
        assert_eq!(edges[0].to, "file:src/shared.js");
    }

    #[test]
    fn bare_specifier_is_ignored() {
        let r = resolver(&["src/app.ts"]);
        let edges = extract("import React from 'react';\n", "src/app.ts", &r);
        assert!(edges.is_empty());
    }

    #[test]
    fn side_effect_import() {
        let r = resolver(&["src/app.ts", "src/styles.ts"]);
        let edges = extract("import './styles';\n", "src/app.ts", &r);
        assert_eq!(edges[0].to, "file:src/styles.ts");
    }
}
