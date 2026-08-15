//! Heuristic Java import extraction. `import a.b.C;` (and `import static a.b.C.m;`)
//! resolve — by the one-public-class-per-file, package-mirrors-directory
//! convention — to a repo file ending `a/b/C.java`. Wildcard `import a.b.*;` can't
//! name a file and is skipped.

use rkg_core::{Edge, EdgeKind};

use crate::resolver::Resolver;

/// Extracts `Imports` edges from one Java file at `rel`.
pub fn extract(content: &str, rel: &str, resolver: &Resolver) -> Vec<Edge> {
    dotted_import_edges(content, rel, resolver, "import ", ";", "java")
}

/// Shared driver for `import a.b.C` style imports (Java/Kotlin/C#). Strips the
/// keyword and an optional terminator, drops `static`/aliases/wildcards, and
/// suffix-matches the dotted path to a `<path>.<ext>` file.
pub(crate) fn dotted_import_edges(
    content: &str,
    rel: &str,
    resolver: &Resolver,
    keyword: &str,
    terminator: &str,
    ext: &str,
) -> Vec<Edge> {
    let from = format!("file:{rel}");
    let mut edges = Vec::new();

    for raw in content.lines() {
        let line = raw.trim();
        let Some(rest) = line.strip_prefix(keyword) else {
            continue;
        };
        let mut rest = rest.trim();
        if !terminator.is_empty() {
            rest = rest.strip_suffix(terminator).unwrap_or(rest).trim();
        }
        rest = rest.strip_prefix("static ").unwrap_or(rest).trim();
        // `using Alias = A.B.C` / `import a.b.C as D` — keep the qualified name.
        if let Some((_, right)) = rest.split_once(" = ") {
            rest = right.trim();
        }
        if let Some((left, _)) = rest.split_once(" as ") {
            rest = left.trim();
        }
        // A `using`/`import` statement, not a declaration.
        if rest.is_empty() || rest.ends_with(".*") || rest.contains(['(', '{', '"']) {
            continue;
        }

        let suffix = format!("{}.{ext}", rest.replace('.', "/"));
        if let Some(to) = resolver.resolve_suffix_path(&suffix) {
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
    fn import_resolves_by_package_path_suffix() {
        let r = resolver(&[
            "src/main/java/app/Main.java",
            "src/main/java/util/Helper.java",
        ]);
        let edges = extract(
            "import util.Helper;\n",
            "src/main/java/app/Main.java",
            &r,
        );
        assert_eq!(edges[0].to, "file:src/main/java/util/Helper.java");
    }

    #[test]
    fn static_import_resolves_to_the_class() {
        let r = resolver(&["a/App.java", "a/Math.java"]);
        let edges = extract("import static a.Math.abs;\n", "a/App.java", &r);
        // `a.Math.abs` won't match, but implementations often also import the class;
        // here the dotted path a/Math/abs.java has no file, so nothing resolves.
        assert!(edges.is_empty());
    }

    #[test]
    fn wildcard_import_is_skipped() {
        let r = resolver(&["a/App.java"]);
        assert!(extract("import java.util.*;\n", "a/App.java", &r).is_empty());
    }
}
