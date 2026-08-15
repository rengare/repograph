//! Heuristic Python import extraction: `import a.b`, `from a.b import c`, and
//! relative `from .mod import x` / `from . import y`. Absolute modules are
//! resolved by suffix match (the source root is unknown); relative ones from the
//! importing file's package directory. A module resolves to `a/b.py` or its
//! package `a/b/__init__.py`.

use rkg_core::{Edge, EdgeKind};

use crate::path::{join_normalize, parent};
use crate::resolver::Resolver;

/// Extracts `Imports` edges from one Python file at `rel`.
pub fn extract(content: &str, rel: &str, resolver: &Resolver) -> Vec<Edge> {
    let from = format!("file:{rel}");
    let dir = parent(rel);
    let mut edges = Vec::new();
    let mut push = |target: Option<String>| {
        if let Some(to) = target {
            if to != from {
                edges.push(Edge::new(from.clone(), to, EdgeKind::Imports));
            }
        }
    };

    for raw in content.lines() {
        let line = raw.trim();
        if let Some(rest) = line.strip_prefix("import ") {
            // `import a.b, c.d` — each comma-separated dotted module.
            for item in rest.split(',') {
                let module = item.split_whitespace().next().unwrap_or("");
                if !module.is_empty() {
                    push(resolve_absolute(module, resolver));
                }
            }
        } else if let Some(rest) = line.strip_prefix("from ") {
            let Some((left, right)) = rest.split_once(" import ") else {
                continue;
            };
            let level = left.chars().take_while(|c| *c == '.').count();
            let module = left.trim_start_matches('.').trim();
            // Each imported name may itself be a submodule (`from pkg import mod`).
            let names: Vec<&str> = right
                .split(',')
                .filter_map(|n| n.split_whitespace().next())
                .filter(|n| *n != "*")
                .collect();

            if level == 0 {
                // Absolute: the module/package, plus each name as a submodule.
                push(resolve_absolute(module, resolver));
                for name in &names {
                    push(resolve_absolute(&format!("{module}.{name}"), resolver));
                }
            } else {
                // Relative: ascend `level - 1` directories from this file's dir.
                let mut base = dir.clone();
                for _ in 0..level.saturating_sub(1) {
                    base = parent(&base);
                }
                let module_path = module.replace('.', "/");
                if module.is_empty() {
                    // `from . import x, y` — each name is a sibling module.
                    for name in &names {
                        push(resolve_relative(&base, name, resolver));
                    }
                } else {
                    push(resolve_relative(&base, &module_path, resolver));
                    for name in &names {
                        push(resolve_relative(&base, &format!("{module_path}/{name}"), resolver));
                    }
                }
            }
        }
    }
    edges
}

/// Absolute dotted module → a file ending `a/b.py` or `a/b/__init__.py`.
fn resolve_absolute(dotted: &str, resolver: &Resolver) -> Option<String> {
    let path = dotted.replace('.', "/");
    resolver
        .resolve_suffix_path(&format!("{path}.py"))
        .or_else(|| resolver.resolve_suffix_path(&format!("{path}/__init__.py")))
}

/// Relative module under `base_dir` → `base/mod.py` or `base/mod/__init__.py`.
fn resolve_relative(base_dir: &str, module: &str, resolver: &Resolver) -> Option<String> {
    let base = join_normalize(base_dir, module);
    resolver.resolve_with_suffixes(&base, &[".py", "/__init__.py"])
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
    fn absolute_dotted_import_resolves_to_module_file() {
        let r = resolver(&["pkg/app.py", "pkg/util/helpers.py"]);
        let edges = extract("from pkg.util import helpers\n", "pkg/app.py", &r);
        assert_eq!(edges[0].to, "file:pkg/util/helpers.py");
    }

    #[test]
    fn import_resolves_package_init() {
        let r = resolver(&["a.py", "pkg/__init__.py"]);
        let edges = extract("import pkg\n", "a.py", &r);
        assert_eq!(edges[0].to, "file:pkg/__init__.py");
    }

    #[test]
    fn relative_from_dot_resolves_sibling_module() {
        let r = resolver(&["pkg/app.py", "pkg/util.py"]);
        let edges = extract("from . import util\n", "pkg/app.py", &r);
        assert_eq!(edges[0].to, "file:pkg/util.py");
    }

    #[test]
    fn relative_parent_ascends_a_package() {
        let r = resolver(&["pkg/sub/app.py", "pkg/shared.py"]);
        let edges = extract("from ..shared import thing\n", "pkg/sub/app.py", &r);
        assert_eq!(edges[0].to, "file:pkg/shared.py");
    }

    #[test]
    fn unresolved_import_is_dropped() {
        let r = resolver(&["a.py"]);
        assert!(extract("import numpy\n", "a.py", &r).is_empty());
    }
}
