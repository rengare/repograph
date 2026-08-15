//! Heuristic Rust import extraction: `mod` declarations (which map deterministically
//! to files) and `use crate::…` paths (mapped to the crate root or a top module).
//! This is intentionally line-based; tree-sitter replaces it in the symbol phase.

use rkg_core::{Edge, EdgeKind};

use crate::path::{join_normalize, parent};
use crate::resolver::Resolver;

/// Extracts `Imports` edges out of one Rust source file at `rel`.
pub fn extract(content: &str, rel: &str, resolver: &Resolver) -> Vec<Edge> {
    let from = format!("file:{rel}");
    let mut edges = Vec::new();
    let src_dir = crate_src_dir(rel);

    for raw in content.lines() {
        let line = raw.trim();
        if line.starts_with("//") {
            continue;
        }

        if let Some(name) = mod_name(line) {
            if let Some(target) = resolve_mod(rel, name, resolver) {
                edges.push(Edge::new(from.clone(), target, EdgeKind::Imports));
            }
        } else if let Some(target) = use_crate_target(line, src_dir.as_deref(), resolver) {
            edges.push(Edge::new(from.clone(), target, EdgeKind::Imports));
        }
    }
    edges
}

/// The module name in a `mod foo;` / `pub mod foo;` declaration (not `mod foo {`).
fn mod_name(line: &str) -> Option<&str> {
    let rest = line
        .strip_prefix("pub mod ")
        .or_else(|| line.strip_prefix("mod "))?;
    let rest = rest.trim();
    let name = rest.trim_end_matches(';');
    // Inline modules (`mod foo { … }`) declare no file.
    if rest.ends_with(';') && !name.is_empty() && is_ident(name) {
        Some(name)
    } else {
        None
    }
}

/// Resolves `mod name;` within `rel` to a module file, per Rust 2018 layout.
fn resolve_mod(rel: &str, name: &str, resolver: &Resolver) -> Option<String> {
    let parent_dir = parent(rel);
    let stem = file_stem(rel);
    let base_dir = if matches!(stem, "lib" | "main" | "mod") {
        parent_dir
    } else {
        join_normalize(&parent_dir, stem)
    };
    let flat = join_normalize(&base_dir, name);
    resolver
        .node_id(&format!("{flat}.rs"))
        .or_else(|| resolver.node_id(&format!("{flat}/mod.rs")))
}

/// For a `use crate::…` line, the file it should point at: a top-level module file
/// if the first segment names one, else the crate root (`lib.rs`/`main.rs`).
fn use_crate_target(line: &str, src_dir: Option<&str>, resolver: &Resolver) -> Option<String> {
    let src_dir = src_dir?;
    let after = use_body(line)?.strip_prefix("crate::")?;
    let root = resolver
        .resolve_with_suffixes(&format!("{src_dir}/lib"), &[".rs"])
        .or_else(|| resolver.resolve_with_suffixes(&format!("{src_dir}/main"), &[".rs"]));

    // `use crate::{ … }` — items live at the crate root.
    let first = after.trim_start();
    if first.starts_with('{') || first.starts_with('*') {
        return root;
    }
    let seg: String = first
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    if seg.is_empty() {
        return root;
    }
    // A lowercase first segment that resolves to a module file wins; otherwise it
    // is an item re-exported at the crate root.
    let module = resolver
        .node_id(&format!("{src_dir}/{seg}.rs"))
        .or_else(|| resolver.node_id(&format!("{src_dir}/{seg}/mod.rs")));
    module.or(root)
}

/// The path body of a `use …;` statement (handles a leading `pub`).
fn use_body(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("use ").or_else(|| {
        line.strip_prefix("pub use ")
            .or_else(|| line.strip_prefix("pub(crate) use "))
    })?;
    Some(rest.trim())
}

/// The nearest ancestor `src` directory of `rel`, e.g. `crates/x/src/a.rs` →
/// `crates/x/src`. `None` when the file lives outside any `src` tree.
fn crate_src_dir(rel: &str) -> Option<String> {
    let parts: Vec<&str> = rel.split('/').collect();
    let idx = parts.iter().rposition(|p| *p == "src")?;
    Some(parts[..=idx].join("/"))
}

fn file_stem(rel: &str) -> &str {
    let name = rel.rsplit_once('/').map_or(rel, |(_, n)| n);
    name.rsplit_once('.').map_or(name, |(stem, _)| stem)
}

fn is_ident(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_alphanumeric() || c == '_')
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
    fn mod_declaration_resolves_to_flat_file() {
        let r = resolver(&["src/lib.rs", "src/loader.rs"]);
        let edges = extract("pub mod loader;\n", "src/lib.rs", &r);
        assert_eq!(edges[0].to, "file:src/loader.rs");
    }

    #[test]
    fn mod_declaration_resolves_to_mod_rs() {
        let r = resolver(&["src/lib.rs", "src/lang/mod.rs"]);
        let edges = extract("mod lang;\n", "src/lib.rs", &r);
        assert_eq!(edges[0].to, "file:src/lang/mod.rs");
    }

    #[test]
    fn nested_mod_uses_stem_subdirectory() {
        // In `src/lang.rs`, `mod rust;` -> `src/lang/rust.rs`.
        let r = resolver(&["src/lang.rs", "src/lang/rust.rs"]);
        let edges = extract("mod rust;\n", "src/lang.rs", &r);
        assert_eq!(edges[0].to, "file:src/lang/rust.rs");
    }

    #[test]
    fn use_crate_points_at_crate_root() {
        let r = resolver(&["crates/x/src/lib.rs", "crates/x/src/loader.rs"]);
        let edges = extract("use crate::{Csr, Edge};\n", "crates/x/src/loader.rs", &r);
        assert_eq!(edges[0].to, "file:crates/x/src/lib.rs");
    }

    #[test]
    fn use_crate_module_points_at_module_file() {
        let r = resolver(&["src/main.rs", "src/loader.rs"]);
        let edges = extract("use crate::loader::parse;\n", "src/main.rs", &r);
        assert_eq!(edges[0].to, "file:src/loader.rs");
    }

    #[test]
    fn inline_mod_is_ignored() {
        let r = resolver(&["src/lib.rs"]);
        let edges = extract("mod tests {\n", "src/lib.rs", &r);
        assert!(edges.is_empty());
    }
}
