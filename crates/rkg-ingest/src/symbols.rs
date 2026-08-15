//! Symbol-level extraction with tree-sitter, driven by the language registry.
//!
//! For each source file the matching grammar parses it into a syntax tree and this
//! module pulls out the definitions it declares — functions, types, classes — as
//! [`SymbolDef`]s carrying a signature, doc, container scope, and a line span. The
//! per-language details (which node kinds are definitions, how docs attach) live in
//! [`crate::registry::SymbolSpec`]; the tree walk here is generic over them.

use std::collections::HashMap;

use anyhow::{Context, Result};
use tree_sitter::{Node, Parser};

use crate::registry::{self, DocStrategy, SymbolSpec};

/// One definition found in a source file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolDef {
    /// Declared name (e.g. `parse`, `GraphData`).
    pub name: String,
    /// Kind of definition: `fn`, `method`, `struct`, `enum`, `class`, `interface`, …
    pub kind: String,
    /// The full declaration up to the body, whitespace-collapsed.
    pub signature: String,
    /// The leading doc comment / docstring, cleaned of markers.
    pub doc: Option<String>,
    /// The enclosing scope's name — `impl Csr`, a class, a namespace — when nested.
    pub container: Option<String>,
    /// 1-based inclusive line span.
    pub start_line: u32,
    pub end_line: u32,
    /// Distinct identifier names used inside the body, minus the symbol's own.
    pub refs: Vec<String>,
}

/// Owns one parser per language tag, reused across files.
pub struct SymbolExtractor {
    parsers: HashMap<&'static str, Parser>,
}

impl SymbolExtractor {
    pub fn new() -> Result<Self> {
        let mut parsers = HashMap::new();
        for language in registry::LANGUAGES {
            let mut parser = Parser::new();
            parser
                .set_language(&(language.grammar)())
                .with_context(|| format!("setting grammar for {}", language.tag))?;
            parsers.insert(language.tag, parser);
        }
        Ok(Self { parsers })
    }

    /// Extracts the definitions from `content`, choosing the grammar by file
    /// extension via the registry. Unknown extensions yield an empty list.
    pub fn extract(&mut self, ext: &str, content: &str) -> Vec<SymbolDef> {
        let Some(language) = registry::for_extension(ext) else {
            return Vec::new();
        };
        let Some(parser) = self.parsers.get_mut(language.tag) else {
            return Vec::new();
        };
        let Some(tree) = parser.parse(content, None) else {
            return Vec::new();
        };
        let src = content.as_bytes();

        let mut out = Vec::new();
        collect(tree.root_node(), &language.symbols, src, &mut out);
        out
    }
}

/// Depth-first walk collecting every definition node under `node`.
fn collect(node: Node, spec: &SymbolSpec, src: &[u8], out: &mut Vec<SymbolDef>) {
    if let Some(name_node) = symbol_name(node, spec) {
        if let Ok(name) = name_node.utf8_text(src) {
            out.push(build_symbol(node, name, spec, src));
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect(child, spec, src, out);
    }
}

/// The name node of `node` if it is a definition under `spec`, else `None`.
fn symbol_name<'t>(node: Node<'t>, spec: &SymbolSpec) -> Option<Node<'t>> {
    let kind = node.kind();
    if spec.defs.contains(&kind) {
        return name_of(node);
    }
    // JS/TS `const f = () => …`.
    if spec.value_fn_decl && kind == "variable_declarator" {
        let value = node.child_by_field_name("value")?;
        if matches!(value.kind(), "arrow_function" | "function" | "function_expression") {
            return node.child_by_field_name("name");
        }
    }
    None
}

/// A definition's name node: the `name` field when present, otherwise the innermost
/// identifier reached by descending `declarator` fields (C/C++ have no `name` field
/// — the name is nested in the declarator).
fn name_of(node: Node) -> Option<Node> {
    if let Some(name) = node.child_by_field_name("name") {
        return Some(name);
    }
    let mut cur = node;
    while let Some(declarator) = cur.child_by_field_name("declarator") {
        cur = declarator;
    }
    match cur.kind() {
        "identifier" | "field_identifier" | "type_identifier" => Some(cur),
        // `void Foo::bar()` — take the last identifier of the qualified name.
        "qualified_identifier" | "scoped_identifier" => {
            cur.child_by_field_name("name").or_else(|| last_identifier(cur))
        }
        _ => None,
    }
}

fn last_identifier(node: Node) -> Option<Node> {
    let mut cursor = node.walk();
    node.children(&mut cursor)
        .filter(|c| c.kind().ends_with("identifier"))
        .last()
}

fn build_symbol(node: Node, name: &str, spec: &SymbolSpec, src: &[u8]) -> SymbolDef {
    let mut refs = std::collections::HashSet::new();
    gather_refs(node, spec, src, name, &mut refs);
    let mut refs: Vec<String> = refs.into_iter().collect();
    refs.sort_unstable();

    SymbolDef {
        name: name.to_string(),
        kind: kind_label(node.kind()).to_string(),
        signature: signature_text(node, src),
        doc: doc_of(node, src, &spec.doc),
        container: container_of(node, src, spec),
        start_line: node.start_position().row as u32 + 1,
        end_line: node.end_position().row as u32 + 1,
        refs,
    }
}

/// A short, friendly label for a definition's grammar node kind, across all
/// supported languages (kinds don't collide in meaning between grammars).
fn kind_label(node_kind: &str) -> &'static str {
    match node_kind {
        "function_item" | "function_declaration" | "generator_function_declaration"
        | "function_definition" | "variable_declarator" => "fn",
        "method_definition" | "method_declaration" => "method",
        "constructor_declaration" => "constructor",
        "struct_item" | "struct_specifier" | "struct_declaration" => "struct",
        "enum_item" | "enum_declaration" | "enum_specifier" => "enum",
        "union_item" | "union_specifier" => "union",
        "trait_item" => "trait",
        "type_item" | "type_alias_declaration" | "type_definition" => "type",
        "const_item" => "const",
        "static_item" => "static",
        "macro_definition" => "macro",
        "class_declaration" | "abstract_class_declaration" | "class_definition"
        | "class_specifier" => "class",
        "interface_declaration" => "interface",
        "record_declaration" => "record",
        "object_declaration" => "object",
        "namespace_definition" | "namespace_declaration" => "namespace",
        _ => "symbol",
    }
}

/// Body-block node kinds, so `signature_text` can stop at the body even when the
/// grammar doesn't expose it as a `body` field (e.g. Kotlin).
const BODY_KINDS: &[&str] = &[
    "block", "function_body", "compound_statement", "declaration_list", "class_body",
    "field_declaration_list", "interface_body", "enum_body", "enumerator_list",
];

/// The declaration text up to (but excluding) the body, collapsed to one line.
fn signature_text(node: Node, src: &[u8]) -> String {
    let start = node.start_byte();
    let end = body_start(node).unwrap_or_else(|| node.end_byte());
    let raw = std::str::from_utf8(&src[start..end.max(start)]).unwrap_or("");
    let collapsed = collapse_ws(raw);
    let trimmed = collapsed.trim().trim_end_matches(['{', '(']).trim();
    truncate(trimmed, 240)
}

fn body_start(node: Node) -> Option<usize> {
    if let Some(body) = node.child_by_field_name("body") {
        return Some(body.start_byte());
    }
    let mut cursor = node.walk();
    node.children(&mut cursor)
        .find(|c| BODY_KINDS.contains(&c.kind()))
        .map(|c| c.start_byte())
}

fn doc_of(node: Node, src: &[u8], strategy: &DocStrategy) -> Option<String> {
    match strategy {
        DocStrategy::None => None,
        DocStrategy::Docstring => docstring_of(node, src),
        DocStrategy::LineComments(prefixes) => line_comment_doc(node, src, prefixes),
    }
}

/// Comment node kinds across grammars (a kind a grammar lacks simply never matches).
const COMMENT_KINDS: &[&str] = &["comment", "line_comment", "block_comment"];

/// Leading doc comment: contiguous preceding comment siblings whose text starts
/// with a doc prefix, cleaned and joined. Attributes/annotations are stepped over.
fn line_comment_doc(node: Node, src: &[u8], prefixes: &[&str]) -> Option<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut sibling = node.prev_sibling();
    while let Some(s) = sibling {
        let kind = s.kind();
        if matches!(
            kind,
            "attribute_item" | "decorator" | "modifiers" | "annotation" | "attribute_list"
        ) {
            sibling = s.prev_sibling();
            continue;
        }
        if COMMENT_KINDS.contains(&kind) {
            let text = s.utf8_text(src).unwrap_or("");
            if prefixes.iter().any(|p| text.trim_start().starts_with(p)) {
                chunks.push(clean_doc(text));
                sibling = s.prev_sibling();
                continue;
            }
        }
        break;
    }
    if chunks.is_empty() {
        return None;
    }
    chunks.reverse();
    Some(truncate(&collapse_ws(&chunks.join(" ")), 400))
}

/// Python docstring: the first string-literal statement inside the body.
fn docstring_of(node: Node, src: &[u8]) -> Option<String> {
    let body = node.child_by_field_name("body")?;
    let first = body.named_child(0)?;
    let string = if first.kind() == "expression_statement" {
        first.named_child(0)?
    } else {
        return None;
    };
    if string.kind() != "string" {
        return None;
    }
    let mut cursor = string.walk();
    let content = string
        .children(&mut cursor)
        .find(|c| c.kind() == "string_content")
        .and_then(|c| c.utf8_text(src).ok())
        .map(str::to_string)
        .unwrap_or_else(|| {
            string
                .utf8_text(src)
                .unwrap_or("")
                .trim_matches(['"', '\''])
                .to_string()
        });
    Some(truncate(&collapse_ws(&content), 400))
}

/// The nearest enclosing definition's descriptor, walking up from `node`.
fn container_of(node: Node, src: &[u8], spec: &SymbolSpec) -> Option<String> {
    let mut current = node.parent();
    while let Some(parent) = current {
        let kind = parent.kind();
        if Some(kind) == spec.impl_kind {
            let ty = parent
                .child_by_field_name("type")
                .and_then(|t| t.utf8_text(src).ok())
                .unwrap_or("");
            return Some(format!("impl {ty}").trim().to_string());
        }
        if spec.container_kinds.contains(&kind) {
            if let Some(name) = parent
                .child_by_field_name("name")
                .and_then(|n| n.utf8_text(src).ok())
            {
                return Some(name.to_string());
            }
        }
        current = parent.parent();
    }
    None
}

/// Collects identifier texts inside `node`, excluding the symbol's own name.
fn gather_refs(
    node: Node,
    spec: &SymbolSpec,
    src: &[u8],
    own: &str,
    out: &mut std::collections::HashSet<String>,
) {
    if out.len() >= 200 {
        return;
    }
    if spec.ref_kinds.contains(&node.kind()) {
        if let Ok(text) = node.utf8_text(src) {
            if text != own {
                out.insert(text.to_string());
            }
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        gather_refs(child, spec, src, own, out);
    }
}

/// Strips comment markers (`/**`, `///`, `//!`, `//`, `*`, `*/`) from each line.
fn clean_doc(text: &str) -> String {
    text.lines()
        .map(|line| {
            line.trim()
                .trim_start_matches("/**")
                .trim_start_matches("///")
                .trim_start_matches("//!")
                .trim_start_matches("//")
                .trim_start_matches('*')
                .trim_end_matches("*/")
                .trim()
        })
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let cut: String = s.chars().take(max).collect();
    format!("{cut}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(symbols: &[SymbolDef]) -> Vec<&str> {
        symbols.iter().map(|s| s.name.as_str()).collect()
    }

    fn find<'a>(symbols: &'a [SymbolDef], name: &str) -> &'a SymbolDef {
        symbols.iter().find(|s| s.name == name).expect("symbol")
    }

    #[test]
    fn extracts_rust_functions_structs_and_methods() {
        let mut ex = SymbolExtractor::new().unwrap();
        let src = "\
pub struct Csr { offsets: Vec<u32> }

/// Builds an adjacency.
pub fn build(n: usize) -> Csr { Csr { offsets: vec![] } }

impl Csr {
    pub fn len(&self) -> usize { self.offsets.len() }
}
";
        let syms = ex.extract("rs", src);
        let build = find(&syms, "build");
        assert_eq!(build.kind, "fn");
        assert_eq!(build.signature, "pub fn build(n: usize) -> Csr");
        assert_eq!(build.doc.as_deref(), Some("Builds an adjacency."));
        assert_eq!(find(&syms, "len").container.as_deref(), Some("impl Csr"));
        assert_eq!(find(&syms, "Csr").kind, "struct");
    }

    #[test]
    fn extracts_js_arrow_consts_and_ts_interfaces() {
        let mut ex = SymbolExtractor::new().unwrap();
        let js_syms = ex.extract("js", "const parse = (s) => s;\nclass Store {}\n");
        let js = names(&js_syms);
        assert!(js.contains(&"parse") && js.contains(&"Store"));
        let ts_syms = ex.extract("ts", "interface Node { id: string }\ntype Id = string;\n");
        let ts = names(&ts_syms);
        assert!(ts.contains(&"Node") && ts.contains(&"Id"));
    }

    #[test]
    fn extracts_python_functions_classes_and_docstrings() {
        let mut ex = SymbolExtractor::new().unwrap();
        let src = "\
def parse(x, y=1):
    \"\"\"Parse a thing.\"\"\"
    return x

class Store:
    def get(self):
        return 1
";
        let syms = ex.extract("py", src);
        let parse = find(&syms, "parse");
        assert_eq!(parse.kind, "fn");
        assert_eq!(parse.signature, "def parse(x, y=1):");
        assert_eq!(parse.doc.as_deref(), Some("Parse a thing."));
        assert_eq!(find(&syms, "get").container.as_deref(), Some("Store"));
        assert_eq!(find(&syms, "Store").kind, "class");
    }

    #[test]
    fn extracts_c_functions_and_structs_without_a_name_field() {
        let mut ex = SymbolExtractor::new().unwrap();
        let src = "typedef struct Point { int x; } Point;\nint add(int a, int b) { return a + b; }\n";
        let syms = ex.extract("c", src);
        let add = find(&syms, "add");
        assert_eq!(add.kind, "fn");
        assert_eq!(add.signature, "int add(int a, int b)");
    }

    #[test]
    fn extracts_cpp_classes_methods_and_namespaces() {
        let mut ex = SymbolExtractor::new().unwrap();
        let src = "namespace ns { class Foo { public: void bar() {} }; }\n";
        let syms = ex.extract("cpp", src);
        assert_eq!(find(&syms, "Foo").kind, "class");
        assert_eq!(find(&syms, "bar").container.as_deref(), Some("Foo"));
    }

    #[test]
    fn extracts_java_classes_methods_and_docs() {
        let mut ex = SymbolExtractor::new().unwrap();
        let src = "/** A store. */\npublic class Store {\n  public int get() { return 1; }\n}\ninterface I { void m(); }\n";
        let syms = ex.extract("java", src);
        assert_eq!(find(&syms, "Store").kind, "class");
        assert_eq!(find(&syms, "Store").doc.as_deref(), Some("A store."));
        assert_eq!(find(&syms, "get").kind, "method");
        assert_eq!(find(&syms, "get").container.as_deref(), Some("Store"));
        assert_eq!(find(&syms, "I").kind, "interface");
    }

    #[test]
    fn extracts_kotlin_functions_and_classes() {
        let mut ex = SymbolExtractor::new().unwrap();
        let src = "fun parse(x: Int): Int { return x }\nclass Store { fun get(): Int = 1 }\nobject O\n";
        let syms = ex.extract("kt", src);
        let parse = find(&syms, "parse");
        assert_eq!(parse.kind, "fn");
        assert_eq!(parse.signature, "fun parse(x: Int): Int");
        assert_eq!(find(&syms, "get").container.as_deref(), Some("Store"));
        assert_eq!(find(&syms, "O").kind, "object");
    }

    #[test]
    fn extracts_csharp_classes_and_methods() {
        let mut ex = SymbolExtractor::new().unwrap();
        let src = "namespace A { public class Store { public int Get() { return 1; } } interface I {} }\n";
        let syms = ex.extract("cs", src);
        assert_eq!(find(&syms, "Store").kind, "class");
        assert_eq!(find(&syms, "Get").kind, "method");
        assert_eq!(find(&syms, "Get").container.as_deref(), Some("Store"));
        assert_eq!(find(&syms, "I").kind, "interface");
    }

    #[test]
    fn references_link_within_a_file() {
        let mut ex = SymbolExtractor::new().unwrap();
        let syms = ex.extract("rs", "struct Csr {}\nfn build() -> Csr { Csr {} }\n");
        assert!(find(&syms, "build").refs.contains(&"Csr".to_string()));
    }

    #[test]
    fn unknown_extension_yields_nothing() {
        let mut ex = SymbolExtractor::new().unwrap();
        assert!(ex.extract("txt", "hello\n").is_empty());
    }
}
