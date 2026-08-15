//! Maps candidate repo-relative paths to the node ids they were interned as, so
//! import/link extractors emit edges only to files that actually exist.

use std::collections::HashMap;

use rkg_core::{NodeId, NodeKind};

/// The set of ingested files and the kind each was classified as.
#[derive(Debug, Default)]
pub struct Resolver {
    kinds: HashMap<String, NodeKind>,
}

impl Resolver {
    pub fn new() -> Self {
        Resolver::default()
    }

    pub fn insert(&mut self, rel: &str, kind: NodeKind) {
        self.kinds.insert(rel.to_string(), kind);
    }

    /// Node id for an exact repo-relative path, or `None` if no such file exists.
    pub fn node_id(&self, rel: &str) -> Option<NodeId> {
        self.kinds
            .get(rel)
            .map(|kind| format!("{}:{}", kind.tag(), rel))
    }

    /// Resolves `base` against a list of suffix candidates (e.g. extensions or
    /// `index` files), returning the first that exists. `base` is already the
    /// path stem to which each candidate is appended.
    pub fn resolve_with_suffixes(&self, base: &str, suffixes: &[&str]) -> Option<NodeId> {
        for suffix in suffixes {
            let candidate = format!("{base}{suffix}");
            if let Some(id) = self.node_id(&candidate) {
                return Some(id);
            }
        }
        None
    }

    /// Finds a file whose repo-relative path is `suffix` or ends with `/suffix`,
    /// preferring the shortest (closest to the repo root). Used for Java/Kotlin/C#
    /// dotted imports (`a.b.C` → a file ending `a/b/C.java`), where the source
    /// root is unknown so an exact path can't be formed.
    pub fn resolve_suffix_path(&self, suffix: &str) -> Option<NodeId> {
        let tail = format!("/{suffix}");
        self.kinds
            .iter()
            .filter(|(path, _)| path.as_str() == suffix || path.ends_with(&tail))
            .min_by_key(|(path, _)| path.len())
            .map(|(path, kind)| format!("{}:{}", kind.tag(), path))
    }
}
