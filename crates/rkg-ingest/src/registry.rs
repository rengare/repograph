//! The language registry: one table describing every supported language — its file
//! extensions, tree-sitter grammar, symbol spec, and optional import extractor.
//! Adding a language is (mostly) a new [`Language`] entry here.

use rkg_core::Edge;

use crate::lang;
use crate::resolver::Resolver;

/// An import/include extractor: turns one file's source into `Imports` edges.
pub type ImportFn = fn(&str, &str, &Resolver) -> Vec<Edge>;

/// How a language attaches documentation to a definition.
pub enum DocStrategy {
    /// Leading comment lines whose trimmed text starts with one of these prefixes
    /// (e.g. `///`, `//!`, `/**`).
    LineComments(&'static [&'static str]),
    /// Python-style: the first string-literal statement inside the body.
    Docstring,
    /// No documentation extraction.
    None,
}

/// The tree-sitter node kinds that make up a language's symbol model, plus its
/// doc convention. Drives the generic walk in [`crate::symbols`].
pub struct SymbolSpec {
    /// Definition node kinds to emit as symbols.
    pub defs: &'static [&'static str],
    /// Identifier node kinds gathered inside a body for `References` edges.
    pub ref_kinds: &'static [&'static str],
    /// Node kinds that name an enclosing scope (via their `name` field).
    pub container_kinds: &'static [&'static str],
    /// Rust's `impl_item`, reported as `impl <type>` (uses the `type` field).
    pub impl_kind: Option<&'static str>,
    /// Whether `const f = () => …` value-function declarators count as defs (JS/TS).
    pub value_fn_decl: bool,
    pub doc: DocStrategy,
}

/// A supported language.
pub struct Language {
    pub tag: &'static str,
    pub extensions: &'static [&'static str],
    pub grammar: fn() -> tree_sitter::Language,
    pub symbols: SymbolSpec,
    pub imports: Option<ImportFn>,
}

/// The language whose extension set contains `ext`, if any.
pub fn for_extension(ext: &str) -> Option<&'static Language> {
    LANGUAGES.iter().find(|l| l.extensions.contains(&ext))
}

// --- grammar factories (named so they're usable in the `static` below) ---
fn g_rust() -> tree_sitter::Language { tree_sitter_rust::LANGUAGE.into() }
fn g_js() -> tree_sitter::Language { tree_sitter_javascript::LANGUAGE.into() }
fn g_ts() -> tree_sitter::Language { tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into() }
fn g_tsx() -> tree_sitter::Language { tree_sitter_typescript::LANGUAGE_TSX.into() }
fn g_python() -> tree_sitter::Language { tree_sitter_python::LANGUAGE.into() }
fn g_c() -> tree_sitter::Language { tree_sitter_c::LANGUAGE.into() }
fn g_cpp() -> tree_sitter::Language { tree_sitter_cpp::LANGUAGE.into() }
fn g_java() -> tree_sitter::Language { tree_sitter_java::LANGUAGE.into() }
fn g_kotlin() -> tree_sitter::Language { tree_sitter_kotlin_ng::LANGUAGE.into() }
fn g_csharp() -> tree_sitter::Language { tree_sitter_c_sharp::LANGUAGE.into() }

/// Comment-based doc prefixes shared by the C-family and JSDoc-style languages.
const JSDOC: &[&str] = &["/**"];
const RUSTDOC: &[&str] = &["///", "//!", "/**"];
const XMLDOC: &[&str] = &["///", "/**"];

pub static LANGUAGES: &[Language] = &[
    Language {
        tag: "rust",
        extensions: &["rs"],
        grammar: g_rust,
        symbols: SymbolSpec {
            defs: &[
                "function_item", "struct_item", "enum_item", "trait_item", "type_item",
                "const_item", "static_item", "union_item", "macro_definition",
            ],
            ref_kinds: &["identifier", "type_identifier"],
            container_kinds: &["struct_item", "enum_item", "trait_item", "union_item", "mod_item"],
            impl_kind: Some("impl_item"),
            value_fn_decl: false,
            doc: DocStrategy::LineComments(RUSTDOC),
        },
        imports: Some(lang::rust::extract),
    },
    Language {
        tag: "javascript",
        extensions: &["js", "jsx", "mjs", "cjs"],
        grammar: g_js,
        symbols: SymbolSpec {
            defs: &[
                "function_declaration", "generator_function_declaration",
                "class_declaration", "method_definition",
            ],
            ref_kinds: &["identifier", "property_identifier"],
            container_kinds: &["class_declaration"],
            impl_kind: None,
            value_fn_decl: true,
            doc: DocStrategy::LineComments(JSDOC),
        },
        imports: Some(lang::js::extract),
    },
    Language {
        tag: "typescript",
        extensions: &["ts", "mts", "cts"],
        grammar: g_ts,
        symbols: TS_SYMBOLS,
        imports: Some(lang::js::extract),
    },
    Language {
        tag: "tsx",
        extensions: &["tsx"],
        grammar: g_tsx,
        symbols: TS_SYMBOLS,
        imports: Some(lang::js::extract),
    },
    Language {
        tag: "python",
        extensions: &["py", "pyi"],
        grammar: g_python,
        symbols: SymbolSpec {
            defs: &["function_definition", "class_definition"],
            ref_kinds: &["identifier"],
            container_kinds: &["class_definition"],
            impl_kind: None,
            value_fn_decl: false,
            doc: DocStrategy::Docstring,
        },
        imports: Some(lang::python::extract),
    },
    Language {
        tag: "c",
        extensions: &["c"],
        grammar: g_c,
        symbols: SymbolSpec {
            defs: &[
                "function_definition", "struct_specifier", "enum_specifier",
                "union_specifier", "type_definition",
            ],
            ref_kinds: &["identifier", "type_identifier", "field_identifier"],
            container_kinds: &["struct_specifier", "union_specifier"],
            impl_kind: None,
            value_fn_decl: false,
            doc: DocStrategy::LineComments(XMLDOC),
        },
        imports: Some(lang::c::extract),
    },
    Language {
        tag: "cpp",
        // `.h` is ambiguous C/C++; the C++ grammar is a superset, so it parses C
        // headers correctly too — the safe default for a header extension.
        extensions: &["cpp", "cc", "cxx", "c++", "hpp", "hh", "hxx", "h"],
        grammar: g_cpp,
        symbols: SymbolSpec {
            defs: &[
                "function_definition", "struct_specifier", "class_specifier",
                "enum_specifier", "union_specifier", "namespace_definition",
                "type_definition",
            ],
            ref_kinds: &["identifier", "type_identifier", "field_identifier"],
            container_kinds: &[
                "class_specifier", "struct_specifier", "namespace_definition", "union_specifier",
            ],
            impl_kind: None,
            value_fn_decl: false,
            doc: DocStrategy::LineComments(XMLDOC),
        },
        imports: Some(lang::c::extract),
    },
    Language {
        tag: "java",
        extensions: &["java"],
        grammar: g_java,
        symbols: SymbolSpec {
            defs: &[
                "class_declaration", "interface_declaration", "enum_declaration",
                "record_declaration", "method_declaration", "constructor_declaration",
            ],
            ref_kinds: &["identifier", "type_identifier"],
            container_kinds: &[
                "class_declaration", "interface_declaration", "enum_declaration", "record_declaration",
            ],
            impl_kind: None,
            value_fn_decl: false,
            doc: DocStrategy::LineComments(JSDOC),
        },
        imports: Some(lang::java::extract),
    },
    Language {
        tag: "kotlin",
        extensions: &["kt", "kts"],
        grammar: g_kotlin,
        symbols: SymbolSpec {
            defs: &["function_declaration", "class_declaration", "object_declaration"],
            ref_kinds: &["identifier", "type_identifier"],
            container_kinds: &["class_declaration", "object_declaration"],
            impl_kind: None,
            value_fn_decl: false,
            doc: DocStrategy::LineComments(JSDOC),
        },
        imports: Some(lang::kotlin::extract),
    },
    Language {
        tag: "csharp",
        extensions: &["cs"],
        grammar: g_csharp,
        symbols: SymbolSpec {
            defs: &[
                "class_declaration", "interface_declaration", "struct_declaration",
                "enum_declaration", "record_declaration", "method_declaration",
                "constructor_declaration",
            ],
            ref_kinds: &["identifier"],
            container_kinds: &[
                "class_declaration", "interface_declaration", "struct_declaration",
                "record_declaration", "namespace_declaration",
            ],
            impl_kind: None,
            value_fn_decl: false,
            doc: DocStrategy::LineComments(XMLDOC),
        },
        imports: Some(lang::csharp::extract),
    },
];

/// TypeScript and TSX share the same symbol model.
const TS_SYMBOLS: SymbolSpec = SymbolSpec {
    defs: &[
        "function_declaration", "generator_function_declaration", "class_declaration",
        "abstract_class_declaration", "method_definition", "interface_declaration",
        "type_alias_declaration", "enum_declaration",
    ],
    ref_kinds: &["identifier", "property_identifier", "type_identifier"],
    container_kinds: &["class_declaration", "abstract_class_declaration", "interface_declaration"],
    impl_kind: None,
    value_fn_decl: true,
    doc: DocStrategy::LineComments(JSDOC),
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extensions_are_unique_across_languages() {
        let mut seen = std::collections::HashSet::new();
        for lang in LANGUAGES {
            for ext in lang.extensions {
                assert!(seen.insert(*ext), "extension {ext:?} claimed by two languages");
            }
        }
    }

    #[test]
    fn for_extension_finds_languages() {
        assert_eq!(for_extension("rs").unwrap().tag, "rust");
        assert_eq!(for_extension("py").unwrap().tag, "python");
        assert_eq!(for_extension("hpp").unwrap().tag, "cpp");
        assert_eq!(for_extension("cs").unwrap().tag, "csharp");
        assert!(for_extension("txt").is_none());
    }
}
