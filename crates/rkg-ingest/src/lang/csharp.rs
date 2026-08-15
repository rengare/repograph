//! Heuristic C# using extraction. `using A.B.C;` imports a *namespace*, which does
//! not map to a single file the way a class does, so resolution is best-effort:
//! we suffix-match the dotted path to a repo file ending `A/B/C.cs` (works for the
//! file-per-type, folder-mirrors-namespace layout many projects use). `using`
//! statements and aliases are handled by the shared Java driver.

use rkg_core::Edge;

use crate::lang::java::dotted_import_edges;
use crate::resolver::Resolver;

/// Extracts `Imports` edges from one C# file at `rel`.
pub fn extract(content: &str, rel: &str, resolver: &Resolver) -> Vec<Edge> {
    dotted_import_edges(content, rel, resolver, "using ", ";", "cs")
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
    fn using_resolves_when_folders_mirror_the_namespace() {
        let r = resolver(&["src/App.cs", "src/Util/Helper.cs"]);
        let edges = extract("using Util.Helper;\n", "src/App.cs", &r);
        assert_eq!(edges[0].to, "file:src/Util/Helper.cs");
    }

    #[test]
    fn using_alias_keeps_the_qualified_name() {
        let r = resolver(&["A.cs", "Lib/Thing.cs"]);
        let edges = extract("using T = Lib.Thing;\n", "A.cs", &r);
        assert_eq!(edges[0].to, "file:Lib/Thing.cs");
    }

    #[test]
    fn using_statement_is_skipped() {
        let r = resolver(&["A.cs"]);
        assert!(extract("using (var f = Open()) { }\n", "A.cs", &r).is_empty());
    }
}
