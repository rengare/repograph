//! Turns a repository on disk into an [`rkg_core::Graph`].
//!
//! A `.gitignore`-aware walk builds the directory/file skeleton (`Contains` edges),
//! then a second pass, driven by the language [`registry`], extracts per-file import
//! edges (heuristic line scanners in [`lang`]) and symbol nodes with `Defines` /
//! `References` edges (tree-sitter, in [`symbols`]). Markdown headings become
//! sections and `[..](..)` links become `Links` edges. Supported languages and how
//! to add one live in [`registry`].

mod path;
pub mod registry;
pub mod resolver;
pub mod symbols;
pub mod lang {
    pub mod c;
    pub mod csharp;
    pub mod java;
    pub mod js;
    pub mod kotlin;
    pub mod markdown;
    pub mod python;
    pub mod rust;
}

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use ignore::WalkBuilder;
use rkg_core::{Edge, EdgeKind, Graph, Node, NodeKind, Span};

use crate::resolver::Resolver;
use crate::symbols::{SymbolDef, SymbolExtractor};

/// Node kind for a file by extension: markdown is a `Doc`, everything else a `File`.
fn kind_for(ext: &str) -> NodeKind {
    match ext {
        "md" | "markdown" => NodeKind::Doc,
        _ => NodeKind::File,
    }
}

fn extension(rel: &str) -> &str {
    rel.rsplit_once('/')
        .map_or(rel, |(_, name)| name)
        .rsplit_once('.')
        .map_or("", |(_, ext)| ext)
}

fn file_name(rel: &str) -> &str {
    rel.rsplit_once('/').map_or(rel, |(_, name)| name)
}

/// Builds the knowledge graph for the repository rooted at `root`.
pub fn build_graph(root: impl AsRef<Path>) -> Result<Graph> {
    let root = root.as_ref();
    let mut graph = Graph::new();

    // Pass 1: skeleton. Collect files so imports can resolve against the real set.
    let mut files: Vec<(String, NodeKind)> = Vec::new();
    let mut resolver = Resolver::new();

    // Root directory node so top-level entries have a parent.
    let root_name = root
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| ".".to_string());
    graph.add_node(Node::new(NodeKind::Dir, ".", root_name));

    for entry in WalkBuilder::new(root).build() {
        let entry = entry.context("walking the repository")?;
        let Some(rel) = path::relative(root, entry.path()) else {
            continue; // the root itself
        };
        let is_dir = entry.file_type().is_some_and(|t| t.is_dir());

        if is_dir {
            let name = file_name(&rel).to_string();
            graph.add_node(Node::new(NodeKind::Dir, rel.clone(), name));
            add_contains(&mut graph, &rel);
        } else {
            let ext = extension(&rel);
            let kind = kind_for(ext);
            let name = file_name(&rel).to_string();
            let mut node = Node::new(kind, rel.clone(), name);
            node.lang = registry::for_extension(ext).map(|l| l.tag.to_string());
            graph.add_node(node);
            add_contains(&mut graph, &rel);
            resolver.insert(&rel, kind);
            files.push((rel, kind));
        }
    }

    // Pass 2: content-derived edges, section nodes, and symbols.
    let mut extractor = SymbolExtractor::new()?;
    for (rel, kind) in &files {
        let abs = root.join(rel);
        let Ok(bytes) = std::fs::read(&abs) else {
            continue;
        };
        let Ok(content) = String::from_utf8(bytes) else {
            continue; // binary / non-utf8: skeleton only
        };

        if *kind == NodeKind::Doc {
            let (nodes, edges) = lang::markdown::extract(&content, rel, &resolver);
            for node in nodes {
                graph.add_node(node);
            }
            for edge in edges {
                graph.add_edge(edge);
            }
        } else if let Some(language) = registry::for_extension(extension(rel)) {
            // Import/include edges, then symbol-level nodes — both from the registry.
            if let Some(import_fn) = language.imports {
                for edge in import_fn(&content, rel, &resolver) {
                    graph.add_edge(edge);
                }
            }
            let syms = extractor.extract(extension(rel), &content);
            add_file_symbols(&mut graph, rel, Some(language.tag), syms);
        }
    }

    Ok(graph)
}

/// Adds `Symbol` nodes for one file's definitions, a `Defines` edge from the file
/// to each, and `References` edges between symbols that mention one another within
/// the same file (a conservative, no-false-cross-file call graph).
fn add_file_symbols(
    graph: &mut Graph,
    rel: &str,
    lang: Option<&'static str>,
    symbols: Vec<SymbolDef>,
) {
    let file_id = format!("file:{rel}");
    let mut name_to_id: HashMap<String, String> = HashMap::new();
    let mut created: Vec<(SymbolDef, String)> = Vec::new();

    for sym in symbols {
        // Two same-named symbols in one file (e.g. methods on different impls)
        // get disambiguated by line so neither is dropped by node dedup.
        let base = format!("sym:{rel}::{}", sym.name);
        let id = if graph.contains(&base) {
            format!("{base}@{}", sym.start_line)
        } else {
            base
        };

        let mut node = Node::new(NodeKind::Symbol, format!("{rel}::{}", sym.name), sym.name.clone());
        node.id = id.clone();
        node.lang = lang.map(str::to_string);
        node.symbol_kind = Some(sym.kind.clone());
        node.container = sym.container.clone();
        node.span = Some(Span {
            start_line: sym.start_line,
            end_line: sym.end_line,
        });
        node.signature = Some(sym.signature.clone());
        node.summary = sym.doc.clone();
        node.locals = sym.locals.clone();
        node.loc = sym.end_line.saturating_sub(sym.start_line) + 1;
        graph.add_node(node);
        graph.add_edge(Edge::new(file_id.clone(), id.clone(), EdgeKind::Defines));

        name_to_id.entry(sym.name.clone()).or_insert_with(|| id.clone());
        created.push((sym, id));
    }

    for (sym, id) in &created {
        for referenced in &sym.refs {
            if let Some(target) = name_to_id.get(referenced) {
                if target != id {
                    graph.add_edge(Edge::new(id.clone(), target.clone(), EdgeKind::References));
                }
            }
        }
    }
}

/// Adds a `Contains` edge from `rel`'s parent directory to `rel`.
fn add_contains(graph: &mut Graph, rel: &str) {
    let parent = path::parent(rel);
    let parent_id = format!("dir:{parent}");
    let child = graph.node(&node_id_of(graph, rel)).map(|n| n.id.clone());
    if let Some(child_id) = child {
        graph.add_edge(Edge::new(parent_id, child_id, EdgeKind::Contains));
    }
}

/// Finds the id a just-added node got (dir or file/doc) for `rel`.
fn node_id_of(graph: &Graph, rel: &str) -> String {
    for kind in [NodeKind::Dir, NodeKind::Doc, NodeKind::File] {
        let id = format!("{}:{}", kind.tag(), rel);
        if graph.contains(&id) {
            return id;
        }
    }
    format!("file:{rel}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Writes a throwaway repo tree and returns its root.
    fn scratch_repo(files: &[(&str, &str)]) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "rkg-ingest-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&root);
        for (rel, content) in files {
            let abs = root.join(rel);
            fs::create_dir_all(abs.parent().unwrap()).unwrap();
            fs::write(abs, content).unwrap();
        }
        root
    }

    #[test]
    fn builds_skeleton_with_contains_edges() {
        let root = scratch_repo(&[("src/lib.rs", "// lib\n"), ("README.md", "# Title\n")]);
        let g = build_graph(&root).unwrap();

        assert!(g.contains("file:src/lib.rs"));
        assert!(g.contains("doc:README.md"));
        assert!(g.contains("dir:src"));
        // src contains lib.rs; root contains src and README.
        assert!(
            g.out_edges("dir:src")
                .any(|e| e.to == "file:src/lib.rs" && e.kind == EdgeKind::Contains)
        );
        assert!(g.out_edges("dir:.").any(|e| e.to == "dir:src"));
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn rust_mod_and_use_edges() {
        let root = scratch_repo(&[
            ("src/lib.rs", "pub mod loader;\n"),
            ("src/loader.rs", "use crate::Thing;\n"),
        ]);
        let g = build_graph(&root).unwrap();
        // lib.rs declares `mod loader;` -> imports loader.rs
        assert!(
            g.out_edges("file:src/lib.rs")
                .any(|e| e.to == "file:src/loader.rs" && e.kind == EdgeKind::Imports)
        );
        // loader.rs `use crate::` -> imports the crate root lib.rs
        assert!(
            g.out_edges("file:src/loader.rs")
                .any(|e| e.to == "file:src/lib.rs" && e.kind == EdgeKind::Imports)
        );
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn symbols_become_nodes_with_defines_and_references() {
        let root = scratch_repo(&[(
            "src/lib.rs",
            "struct Csr {}\npub fn build() -> Csr { Csr {} }\n",
        )]);
        let g = build_graph(&root).unwrap();

        assert!(g.contains("sym:src/lib.rs::Csr"));
        assert!(g.contains("sym:src/lib.rs::build"));
        // File defines its symbols.
        assert!(
            g.out_edges("file:src/lib.rs")
                .any(|e| e.to == "sym:src/lib.rs::build" && e.kind == EdgeKind::Defines)
        );
        // `build` references `Csr` in its body.
        assert!(
            g.out_edges("sym:src/lib.rs::build")
                .any(|e| e.to == "sym:src/lib.rs::Csr" && e.kind == EdgeKind::References)
        );
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn polyglot_symbols_and_imports() {
        let root = scratch_repo(&[
            ("app.py", "from util import helper\n\ndef main():\n    return helper()\n"),
            ("util/helper.py", "def helper():\n    return 2\n"),
            ("src/main.c", "#include \"util.h\"\nint add(int a) { return a; }\n"),
            ("src/util.h", "int sub(int a);\n"),
            ("A.java", "import lib.Thing;\nclass A { void run() {} }\n"),
            ("lib/Thing.java", "class Thing {}\n"),
        ]);
        let g = build_graph(&root).unwrap();

        // Python: symbol + cross-file import to the resolved submodule.
        assert!(g.contains("sym:app.py::main"));
        assert!(
            g.out_edges("file:app.py")
                .any(|e| e.to == "file:util/helper.py" && e.kind == EdgeKind::Imports)
        );
        // C: symbol without a name field + local #include.
        assert!(g.contains("sym:src/main.c::add"));
        assert!(
            g.out_edges("file:src/main.c")
                .any(|e| e.to == "file:src/util.h" && e.kind == EdgeKind::Imports)
        );
        // Java: symbol + dotted import resolved by path suffix.
        assert!(g.contains("sym:A.java::A"));
        assert!(
            g.out_edges("file:A.java")
                .any(|e| e.to == "file:lib/Thing.java" && e.kind == EdgeKind::Imports)
        );
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn markdown_sections_and_links() {
        let root = scratch_repo(&[
            ("README.md", "# Intro\nsee [loader](src/loader.rs)\n## Format\n"),
            ("src/loader.rs", "// x\n"),
        ]);
        let g = build_graph(&root).unwrap();
        assert!(g.contains("sec:README.md#intro"));
        assert!(g.contains("sec:README.md#format"));
        assert!(
            g.out_edges("doc:README.md")
                .any(|e| e.to == "file:src/loader.rs" && e.kind == EdgeKind::Links)
        );
        fs::remove_dir_all(&root).ok();
    }
}
