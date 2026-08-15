//! Heuristic Kotlin import extraction. `import a.b.C` resolves by suffix match to a
//! repo file ending `a/b/C.kt`. Kotlin allows several top-level declarations per
//! file and file names need not match a class, so this is best-effort (it matches
//! the common class-per-file layout); wildcards and aliases are handled like Java.

use rkg_core::Edge;

use crate::lang::java::dotted_import_edges;
use crate::resolver::Resolver;

/// Extracts `Imports` edges from one Kotlin file at `rel`.
pub fn extract(content: &str, rel: &str, resolver: &Resolver) -> Vec<Edge> {
    // Kotlin imports have no trailing terminator.
    dotted_import_edges(content, rel, resolver, "import ", "", "kt")
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
    fn import_resolves_by_path_suffix() {
        let r = resolver(&["app/Main.kt", "util/Helper.kt"]);
        let edges = extract("import util.Helper\n", "app/Main.kt", &r);
        assert_eq!(edges[0].to, "file:util/Helper.kt");
    }

    #[test]
    fn aliased_import_still_resolves() {
        let r = resolver(&["app/Main.kt", "util/Helper.kt"]);
        let edges = extract("import util.Helper as H\n", "app/Main.kt", &r);
        assert_eq!(edges[0].to, "file:util/Helper.kt");
    }

    #[test]
    fn wildcard_is_skipped() {
        let r = resolver(&["app/Main.kt"]);
        assert!(extract("import kotlin.collections.*\n", "app/Main.kt", &r).is_empty());
    }
}
