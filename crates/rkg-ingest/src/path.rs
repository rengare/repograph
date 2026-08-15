//! Small path helpers working in normalized, forward-slash, repo-relative strings.

use std::path::Path;

/// Repo-relative, forward-slash form of `path` under `root`, or `None` if `path`
/// *is* the root (nothing to name).
pub fn relative(root: &Path, path: &Path) -> Option<String> {
    let rel = path.strip_prefix(root).ok()?;
    if rel.as_os_str().is_empty() {
        return None;
    }
    let mut s = String::new();
    for (i, comp) in rel.components().enumerate() {
        if i > 0 {
            s.push('/');
        }
        s.push_str(&comp.as_os_str().to_string_lossy());
    }
    Some(s)
}

/// Parent directory of a repo-relative path; top-level entries parent to `.`.
pub fn parent(rel: &str) -> String {
    match rel.rsplit_once('/') {
        Some((dir, _)) => dir.to_string(),
        None => ".".to_string(),
    }
}

/// Joins `specifier` onto `base_dir` (a directory) and collapses `.`/`..`.
/// Both are repo-relative; the result stays repo-relative (a specifier escaping
/// the root yields a best-effort collapsed path).
pub fn join_normalize(base_dir: &str, specifier: &str) -> String {
    let mut stack: Vec<&str> = if base_dir == "." || base_dir.is_empty() {
        Vec::new()
    } else {
        base_dir.split('/').collect()
    };
    for part in specifier.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                stack.pop();
            }
            other => stack.push(other),
        }
    }
    if stack.is_empty() {
        ".".to_string()
    } else {
        stack.join("/")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parent_of_nested_and_top_level() {
        assert_eq!(parent("a/b/c.rs"), "a/b");
        assert_eq!(parent("x.rs"), ".");
    }

    #[test]
    fn join_collapses_dot_and_dotdot() {
        assert_eq!(join_normalize("src/models", "../loader"), "src/loader");
        assert_eq!(join_normalize("src", "./a/b"), "src/a/b");
        assert_eq!(join_normalize(".", "index"), "index");
    }
}
