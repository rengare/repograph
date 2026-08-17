//! Symbol-level extraction with tree-sitter, driven by the language registry.
//!
//! For each source file the matching grammar parses it into a syntax tree and this
//! module pulls out the definitions it declares — functions, types, classes — as
//! [`SymbolDef`]s carrying a signature, doc, container scope, and a line span. The
//! per-language details (which node kinds are definitions, how docs attach) live in
//! [`crate::registry::SymbolSpec`]; the tree walk here is generic over them.

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};
use rkg_core::{Param, TypeRef};
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
    /// Variable names declared in this symbol's scope — its parameters and local
    /// declarations (not those of nested functions). Empty for non-scoped symbols.
    pub locals: Vec<String>,
    /// Distinct callee names invoked in call position inside the body (a subset of
    /// `refs` restricted to calls/constructions) — what this code does.
    pub calls: Vec<String>,
    /// Parameters with optional declared-or-inferred types.
    pub params: Vec<Param>,
    /// Return type, declared (from the grammar) or locally inferred from the body.
    pub returns: Option<TypeRef>,
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
    let mut refs = HashSet::new();
    gather_refs(node, spec, src, name, &mut refs);
    let mut refs: Vec<String> = refs.into_iter().collect();
    refs.sort_unstable();

    let mut calls_set = HashSet::new();
    gather_calls(node, spec, src, name, &mut calls_set);
    let mut calls: Vec<String> = calls_set.into_iter().collect();
    calls.sort_unstable();
    calls.truncate(100);

    SymbolDef {
        name: name.to_string(),
        kind: kind_label(node.kind()).to_string(),
        signature: signature_text(node, src),
        doc: doc_of(node, src, &spec.doc),
        container: container_of(node, src, spec),
        start_line: node.start_position().row as u32 + 1,
        end_line: node.end_position().row as u32 + 1,
        refs,
        locals: collect_locals(node, spec, src, name),
        calls,
        params: extract_params(node, src),
        returns: return_of(node, spec, src),
    }
}

/// The variable names bound in `node`'s own scope: its parameters plus the local
/// variables it declares. Does not descend into nested definitions, so a
/// function's locals don't leak in from an inner function/class.
fn collect_locals(node: Node, spec: &SymbolSpec, src: &[u8], own: &str) -> Vec<String> {
    let mut set = HashSet::new();
    if let Some(params) = find_params(node) {
        collect_param_names(params, src, &mut set);
    }
    walk_locals(node, spec, src, &mut set, true);
    set.remove(own);
    let mut names: Vec<String> = set.into_iter().collect();
    names.sort_unstable();
    names.truncate(100);
    names
}

fn walk_locals(node: Node, spec: &SymbolSpec, src: &[u8], set: &mut HashSet<String>, is_root: bool) {
    if !is_root && spec.defs.contains(&node.kind()) {
        return; // a nested definition owns its own scope
    }
    if !is_root && spec.var_kinds.contains(&node.kind()) {
        binding_names(node, src, set);
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if set.len() >= 100 {
            break;
        }
        walk_locals(child, spec, src, set, false);
    }
}

/// The parameter-list node of a definition, if any — a `parameters` field, the
/// list inside a C/C++ `function_declarator`, or an unfielded child list (Kotlin).
fn find_params(node: Node) -> Option<Node> {
    if let Some(params) = node.child_by_field_name("parameters") {
        return Some(params);
    }
    let mut cur = node;
    while let Some(declarator) = cur.child_by_field_name("declarator") {
        if let Some(params) = declarator.child_by_field_name("parameters") {
            return Some(params);
        }
        cur = declarator;
    }
    let mut cursor = node.walk();
    node.children(&mut cursor).find(|c| {
        matches!(
            c.kind(),
            "parameters" | "formal_parameters" | "parameter_list" | "function_value_parameters"
        )
    })
}

fn collect_param_names(params: Node, src: &[u8], set: &mut HashSet<String>) {
    let mut cursor = params.walk();
    for param in params.children(&mut cursor) {
        if !param.is_named() {
            continue;
        }
        match param.kind() {
            "identifier" | "shorthand_property_identifier_pattern" => push_ident(param, src, set),
            "self_parameter" | "this" => {}
            _ => binding_names(param, src, set),
        }
    }
}

/// Extracts the declared name(s) of a binding node (a local declaration or a
/// parameter), handling `name`/`pattern`/`left` fields, C declarator descent, and
/// grammars that keep the name as a bare identifier child.
fn binding_names(node: Node, src: &[u8], set: &mut HashSet<String>) {
    if let Some(name) = node.child_by_field_name("name") {
        push_ident(name, src, set);
        return;
    }
    if let Some(pattern) = node.child_by_field_name("pattern") {
        collect_pattern_idents(pattern, src, set);
        return;
    }
    if let Some(left) = node.child_by_field_name("left") {
        collect_pattern_idents(left, src, set); // Python `x = …`
        return;
    }
    let mut cur = node;
    let mut descended = false;
    while let Some(declarator) = cur.child_by_field_name("declarator") {
        cur = declarator;
        descended = true;
    }
    if descended && cur.kind().ends_with("identifier") {
        push_ident(cur, src, set);
        return;
    }
    let mut cursor = node.walk();
    if let Some(id) = node.children(&mut cursor).find(|c| c.kind() == "identifier") {
        push_ident(id, src, set);
    }
}

/// Collects every `identifier` under a pattern (handles tuple/list destructuring).
fn collect_pattern_idents(node: Node, src: &[u8], set: &mut HashSet<String>) {
    if node.kind() == "identifier" {
        push_ident(node, src, set);
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_pattern_idents(child, src, set);
    }
}

fn push_ident(node: Node, src: &[u8], set: &mut HashSet<String>) {
    if let Ok(text) = node.utf8_text(src) {
        // `self`/`this` are receivers, not meaningful scope locals.
        if !text.is_empty() && text != "self" && text != "this" && set.len() < 100 {
            set.insert(text.to_string());
        }
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

/// Collects callee names in call position inside `node` — the identifier being
/// called in a `call_kinds` node — so `calls` captures behaviour rather than every
/// mentioned identifier. Excludes the symbol's own name (direct recursion).
fn gather_calls(node: Node, spec: &SymbolSpec, src: &[u8], own: &str, out: &mut HashSet<String>) {
    if out.len() >= 100 {
        return;
    }
    if spec.call_kinds.contains(&node.kind()) {
        if let Some(name) = callee_name(node, src) {
            if name != own {
                out.insert(name);
            }
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        gather_calls(child, spec, src, own, out);
    }
}

/// The final identifier segment being called: `read`, `Config` from `Config::new`,
/// `push` from `v.push`. `None` for computed/odd callees.
fn callee_name(call: Node, src: &[u8]) -> Option<String> {
    let callee = call
        .child_by_field_name("function")
        .or_else(|| call.child_by_field_name("constructor"))
        .or_else(|| call.child_by_field_name("macro"))
        .or_else(|| call.child_by_field_name("name"))
        .or_else(|| call.named_child(0))?;
    last_segment(callee.utf8_text(src).ok()?)
}

/// The last `.`/`::`-separated segment of a path, stripped of any generic/call
/// tail, if it is a plain identifier.
fn last_segment(text: &str) -> Option<String> {
    let seg = text.split(['.', ':']).filter(|s| !s.is_empty()).next_back()?;
    let seg = seg.split(['<', '(', '!', '[']).next().unwrap_or(seg).trim();
    (!seg.is_empty() && seg.chars().all(|c| c.is_alphanumeric() || c == '_')).then(|| seg.to_string())
}

/// Parameters of a definition with their names and, when knowable, types (declared
/// annotation first, else inferred from a default value). Reuses the same binding
/// walk as scope locals; unnamed/`self` params are skipped.
fn extract_params(node: Node, src: &[u8]) -> Vec<Param> {
    let Some(params) = find_params(node) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut cursor = params.walk();
    for param in params.children(&mut cursor) {
        if !param.is_named() || out.len() >= 50 {
            continue;
        }
        if matches!(param.kind(), "self_parameter" | "this") {
            continue;
        }
        let mut names = HashSet::new();
        match param.kind() {
            "identifier" | "shorthand_property_identifier_pattern" => push_ident(param, src, &mut names),
            _ => binding_names(param, src, &mut names),
        }
        let ty = param_type(param, src);
        let mut names: Vec<String> = names.into_iter().collect();
        names.sort_unstable();
        for name in names {
            out.push(Param { name, ty: ty.clone() });
        }
    }
    out
}

/// A parameter's type: a declared `type` field, else inference from its default
/// value (`x = 5`, `y = Foo()`).
fn param_type(param: Node, src: &[u8]) -> Option<TypeRef> {
    if let Some(t) = param.child_by_field_name("type") {
        if let Ok(text) = t.utf8_text(src) {
            let s = clean_type(text);
            if !s.is_empty() {
                return Some(TypeRef::declared(s));
            }
        }
    }
    let default = param
        .child_by_field_name("value")
        .or_else(|| param.child_by_field_name("default"));
    default.and_then(|v| infer_type(v, src))
}

/// A definition's return type: the declared `return_field` when present, else a
/// best-effort inference from the body's return expressions.
fn return_of(node: Node, spec: &SymbolSpec, src: &[u8]) -> Option<TypeRef> {
    if let Some(field) = spec.return_field {
        if let Some(rt) = node.child_by_field_name(field) {
            if let Ok(text) = rt.utf8_text(src) {
                let s = clean_type(text);
                if !s.is_empty() {
                    return Some(TypeRef::declared(s));
                }
            }
        }
    }
    infer_return_from_body(node, spec, src)
}

/// Infers a return type when every `return <expr>` in the body (not counting
/// nested definitions) infers to the *same* type; otherwise `None` (never guesses).
fn infer_return_from_body(node: Node, spec: &SymbolSpec, src: &[u8]) -> Option<TypeRef> {
    let body = node.child_by_field_name("body")?;
    let mut agreed: Option<String> = None;
    let mut consistent = true;
    let mut saw_return = false;
    collect_return_types(body, spec, src, &mut |ty| {
        saw_return = true;
        match &agreed {
            None => agreed = Some(ty),
            Some(prev) if *prev != ty => consistent = false,
            _ => {}
        }
    });
    match agreed {
        Some(ty) if consistent && saw_return => Some(TypeRef::inferred(ty)),
        _ => None,
    }
}

/// Visits `return` statements under `node`, calling `sink` with each returned
/// expression's inferred type. Stops at nested definitions (their returns are not
/// this symbol's). A return whose value can't be inferred yields nothing.
fn collect_return_types(
    node: Node,
    spec: &SymbolSpec,
    src: &[u8],
    sink: &mut impl FnMut(String),
) {
    if node.kind() == "return_statement" {
        if let Some(value) = node.named_child(0) {
            if let Some(t) = infer_type(value, src) {
                sink(t.ty);
            }
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        // Don't descend into a nested definition's own body.
        if spec.defs.contains(&child.kind()) {
            continue;
        }
        collect_return_types(child, spec, src, sink);
    }
}

/// Best-effort, single-expression type inference from local syntax: literals and
/// constructor/factory calls. Returns an `inferred` [`TypeRef`], or `None` when the
/// expression is ambiguous (a variable, arithmetic, an unknown call, …).
fn infer_type(expr: Node, src: &[u8]) -> Option<TypeRef> {
    if let Some(ty) = constructor_type(expr, src) {
        return Some(TypeRef::inferred(ty));
    }
    let k = expr.kind();
    let label = if k == "true" || k == "false" || k == "boolean_literal" || k.contains("boolean") {
        "bool"
    } else if k.contains("float") {
        "float"
    } else if k.contains("char") && !k.contains("character_") {
        "char"
    } else if k == "string" || k.contains("string_literal") || k == "template_string"
        || k == "raw_string_literal" || k == "interpolated_string_expression"
    {
        "string"
    } else if k == "integer" || k == "integer_literal" || k == "int_literal"
        || k == "decimal_integer_literal"
    {
        "int"
    } else if k == "number" || k == "number_literal" {
        "number"
    } else if k == "array" || k.contains("array_") || k == "list" || k == "list_expression" {
        "array"
    } else if k == "object" || k == "dictionary" || k == "map_literal" || k.contains("dictionary") {
        "map"
    } else if k == "set" {
        "set"
    } else if k == "tuple" || k == "tuple_expression" {
        "tuple"
    } else {
        return None;
    };
    Some(TypeRef::inferred(label))
}

/// The constructed type of a `new Foo()` / `Foo(...)` / `Foo::new(...)` expression,
/// or `None` when it isn't a recognisable construction.
fn constructor_type(expr: Node, src: &[u8]) -> Option<String> {
    match expr.kind() {
        "new_expression" | "object_creation_expression" => {
            let ty = expr
                .child_by_field_name("constructor")
                .or_else(|| expr.child_by_field_name("type"))
                .or_else(|| expr.named_child(0))?;
            last_segment(ty.utf8_text(src).ok()?)
        }
        "call_expression" | "call" => {
            let callee = expr
                .child_by_field_name("function")
                .or_else(|| expr.named_child(0))?;
            constructor_from_callee(callee.utf8_text(src).ok()?)
        }
        _ => None,
    }
}

/// Reads a callee path as a construction: `Foo::new`/`Foo.create` ⇒ `Foo`, or a
/// single Uppercase callee `Foo(...)` ⇒ `Foo`. Lowercase free functions ⇒ `None`.
fn constructor_from_callee(text: &str) -> Option<String> {
    let segs: Vec<&str> = text.split(['.', ':']).filter(|s| !s.is_empty()).collect();
    let last = *segs.last()?;
    const CTOR_METHODS: &[&str] = &["new", "create", "from", "default", "with_capacity", "of", "make"];
    if segs.len() >= 2 && CTOR_METHODS.contains(&last) {
        if let Some(ty) = segs.iter().rev().skip(1).find(|s| starts_upper(s)) {
            return last_segment(ty);
        }
    }
    if segs.len() == 1 && starts_upper(last) {
        return last_segment(last);
    }
    None
}

fn starts_upper(s: &str) -> bool {
    s.chars().next().is_some_and(|c| c.is_uppercase())
}

/// Normalises a type annotation to a compact one-line form: leading `->`/`:`
/// stripped, whitespace collapsed, capped.
fn clean_type(text: &str) -> String {
    let s = collapse_ws(text);
    let s = s
        .trim()
        .trim_start_matches("->")
        .trim()
        .trim_start_matches(':')
        .trim();
    truncate(s, 80)
}

/// A heuristic behavioural role for a callable symbol, from its name (and, for
/// tests, its container). `None` for types and unrecognised names — an honest
/// "unknown" rather than a forced label.
pub fn classify_role(sym: &SymbolDef) -> Option<String> {
    if sym.kind == "constructor" {
        return Some("constructor".to_string());
    }
    if !matches!(sym.kind.as_str(), "fn" | "method") {
        return None;
    }
    let name = sym.name.as_str();
    let lower = name.to_ascii_lowercase();
    if lower.starts_with("test")
        || lower.ends_with("_test")
        || sym.container.as_deref().is_some_and(|c| c.contains("Test"))
    {
        return Some("test".to_string());
    }
    if lower == "main" {
        return Some("entrypoint".to_string());
    }
    let role = match first_word(name).as_str() {
        "new" | "make" | "create" | "build" | "construct" | "init" | "with" => "factory",
        "get" | "fetch" | "find" | "lookup" | "peek" => "accessor",
        "set" | "put" | "update" | "insert" | "add" | "remove" | "delete" | "push" | "pop"
        | "clear" | "reset" | "append" => "mutator",
        "is" | "has" | "can" | "should" | "contains" | "exists" | "equals" | "matches" => "predicate",
        "to" | "into" | "as" | "from" | "parse" | "serialize" | "deserialize" | "convert"
        | "format" | "encode" | "decode" | "render" => "converter",
        "on" | "handle" => "handler",
        "read" | "write" | "open" | "close" | "flush" | "print" | "send" | "recv" | "connect"
        | "load" | "save" | "fetch_url" => "io",
        _ => return None,
    };
    Some(role.to_string())
}

/// The first word of an identifier, lowercased — splitting on `_` (snake_case) or a
/// lower→Upper boundary (camelCase/PascalCase). `parseConfig` ⇒ `parse`.
fn first_word(name: &str) -> String {
    let mut out = String::new();
    for c in name.trim_start_matches('_').chars() {
        if c == '_' {
            break;
        }
        if !out.is_empty()
            && c.is_uppercase()
            && out.chars().last().is_some_and(|p| p.is_lowercase())
        {
            break;
        }
        out.push(c);
    }
    out.to_ascii_lowercase()
}

/// A synthesised one-line description of what a callable does, from its role,
/// typed params, return, and top callees. Used only when the symbol has no real
/// doc comment. `None` when there's nothing meaningful to say.
pub fn synthesize_description(sym: &SymbolDef, role: Option<&str>) -> Option<String> {
    if !matches!(sym.kind.as_str(), "fn" | "method" | "constructor") {
        return None;
    }
    if sym.params.is_empty() && sym.returns.is_none() && sym.calls.is_empty() && role.is_none() {
        return None;
    }
    let mut s = match role {
        Some(r) => format!("{} ({r})", sym.name),
        None => sym.name.clone(),
    };
    if !sym.params.is_empty() {
        let ps: Vec<String> = sym
            .params
            .iter()
            .take(4)
            .map(|p| match &p.ty {
                Some(t) => format!("{}: {}", p.name, t.ty),
                None => p.name.clone(),
            })
            .collect();
        s.push_str(&format!(" takes {}", ps.join(", ")));
    }
    if let Some(ret) = &sym.returns {
        s.push_str(&format!("; returns {}", ret.ty));
    }
    if !sym.calls.is_empty() {
        let cs: Vec<&str> = sym.calls.iter().take(4).map(String::as_str).collect();
        s.push_str(&format!("; calls {}", cs.join(", ")));
    }
    Some(truncate(&s, 240))
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
    fn captures_calls_typed_params_and_declared_return() {
        let mut ex = SymbolExtractor::new().unwrap();
        let src = "\
pub fn build(path: &str, n: usize) -> Csr {
    let data = read(path);
    parse(data)
}
";
        let syms = ex.extract("rs", src);
        let build = find(&syms, "build");

        // Params carry their declared types (inferred = false).
        let path = build.params.iter().find(|p| p.name == "path").expect("path param");
        assert_eq!(path.ty.as_ref().unwrap().ty, "&str");
        assert!(!path.ty.as_ref().unwrap().inferred);
        let n = build.params.iter().find(|p| p.name == "n").expect("n param");
        assert_eq!(n.ty.as_ref().unwrap().ty, "usize");

        // Declared return type, from the grammar's return_type field.
        let ret = build.returns.as_ref().expect("return type");
        assert_eq!(ret.ty, "Csr");
        assert!(!ret.inferred);

        // `calls` holds call-position callees only — `read`/`parse`, not `data`.
        assert!(build.calls.contains(&"read".to_string()), "calls: {:?}", build.calls);
        assert!(build.calls.contains(&"parse".to_string()));
        assert!(!build.calls.contains(&"data".to_string()));

        assert_eq!(classify_role(build).as_deref(), Some("factory"));
    }

    #[test]
    fn infers_python_types_roles_and_descriptions() {
        let mut ex = SymbolExtractor::new().unwrap();
        let src = "\
def get_count(items):
    return 5

def make_store():
    return Store()

def find_it(x):
    return x
";
        let syms = ex.extract("py", src);

        // Return inferred from a literal.
        let get = find(&syms, "get_count");
        let ret = get.returns.as_ref().expect("inferred return");
        assert_eq!(ret.ty, "int");
        assert!(ret.inferred);
        assert_eq!(classify_role(get).as_deref(), Some("accessor"));

        // Return inferred from a constructor call.
        let make = find(&syms, "make_store");
        assert_eq!(make.returns.as_ref().unwrap().ty, "Store");
        assert!(make.returns.as_ref().unwrap().inferred);
        assert_eq!(classify_role(make).as_deref(), Some("factory"));

        // Un-inferable return (a bare variable) stays None rather than guessing.
        let find_it = find(&syms, "find_it");
        assert!(find_it.returns.is_none());
        assert_eq!(classify_role(find_it).as_deref(), Some("accessor"));

        // Synthesised description only when there is no doc comment.
        let desc = synthesize_description(get, Some("accessor")).expect("description");
        assert!(desc.contains("get_count") && desc.contains("accessor"), "{desc}");
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
    fn captures_scope_locals_params_and_declarations() {
        let mut ex = SymbolExtractor::new().unwrap();
        // `total` is a local, `path` a parameter; `read` is a call (a ref, not a
        // local); the nested fn's `inner_only` must not leak into count's scope.
        let src = "\
fn count(path: &str) -> usize {
    let total = read(path);
    fn helper() { let inner_only = 1; }
    total
}
";
        let syms = ex.extract("rs", src);
        let count = find(&syms, "count");
        assert!(count.locals.contains(&"path".to_string()), "locals: {:?}", count.locals);
        assert!(count.locals.contains(&"total".to_string()));
        assert!(!count.locals.contains(&"inner_only".to_string()), "nested leaked");
        assert!(!count.locals.contains(&"read".to_string()), "call is not a local");
    }

    #[test]
    fn captures_python_and_js_locals() {
        let mut ex = SymbolExtractor::new().unwrap();
        let py = ex.extract("py", "def f(a, b):\n    total = a + b\n    return total\n");
        let f = find(&py, "f");
        assert!(f.locals.contains(&"a".to_string()) && f.locals.contains(&"total".to_string()));

        let js = ex.extract("js", "function g(x) { const y = x + 1; return y; }\n");
        let g = find(&js, "g");
        assert!(g.locals.contains(&"x".to_string()) && g.locals.contains(&"y".to_string()));
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
