pub mod resolver;
pub mod bloom;
pub mod pipeline;

use anyhow::{Result, anyhow};
use tree_sitter::{Parser, Language};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SupportedLanguage {
    Rust,
    C,
    Cpp,
    CSharp,
    Java,
    Go,
    JavaScript,
    TypeScript,
    Python,
    Ruby,
    PHP,
    Swift,
    Kotlin,
    Solidity,
    SQL,
}

pub struct CodeParser {
    parser: Parser,
}

impl CodeParser {
    pub fn new(lang: SupportedLanguage) -> Result<Self> {
        let mut parser = Parser::new();
        
        // In a real build, each of these would be a separate crate dependency:
        // tree-sitter-rust, tree-sitter-java, etc.
        // We simulate the binding logic here.
        let language = match lang {
            SupportedLanguage::Rust => {
                extern "C" { fn tree_sitter_rust() -> Language; }
                unsafe { tree_sitter_rust() }
            }
            SupportedLanguage::C => {
                extern "C" { fn tree_sitter_c() -> Language; }
                unsafe { tree_sitter_c() }
            }
            SupportedLanguage::Cpp => {
                extern "C" { fn tree_sitter_cpp() -> Language; }
                unsafe { tree_sitter_cpp() }
            }
            SupportedLanguage::CSharp => {
                extern "C" { fn tree_sitter_c_sharp() -> Language; }
                unsafe { tree_sitter_c_sharp() }
            }
            SupportedLanguage::Java => {
                extern "C" { fn tree_sitter_java() -> Language; }
                unsafe { tree_sitter_java() }
            }
            SupportedLanguage::Go => {
                extern "C" { fn tree_sitter_go() -> Language; }
                unsafe { tree_sitter_go() }
            }
            SupportedLanguage::JavaScript => {
                extern "C" { fn tree_sitter_javascript() -> Language; }
                unsafe { tree_sitter_javascript() }
            }
            SupportedLanguage::TypeScript => {
                extern "C" { fn tree_sitter_typescript() -> Language; }
                unsafe { tree_sitter_typescript() }
            }
            SupportedLanguage::Python => {
                extern "C" { fn tree_sitter_python() -> Language; }
                unsafe { tree_sitter_python() }
            }
            SupportedLanguage::Ruby => {
                extern "C" { fn tree_sitter_ruby() -> Language; }
                unsafe { tree_sitter_ruby() }
            }
            SupportedLanguage::PHP => {
                extern "C" { fn tree_sitter_php() -> Language; }
                unsafe { tree_sitter_php() }
            }
            SupportedLanguage::Swift => {
                extern "C" { fn tree_sitter_swift() -> Language; }
                unsafe { tree_sitter_swift() }
            }
            SupportedLanguage::Kotlin => {
                extern "C" { fn tree_sitter_kotlin() -> Language; }
                unsafe { tree_sitter_kotlin() }
            }
            SupportedLanguage::Solidity => {
                extern "C" { fn tree_sitter_solidity() -> Language; }
                unsafe { tree_sitter_solidity() }
            }
            SupportedLanguage::SQL => {
                extern "C" { fn tree_sitter_sql() -> Language; }
                unsafe { tree_sitter_sql() }
            }
        };

        parser.set_language(language)
            .map_err(|e| anyhow!("Failed to set language: {:?}", e))?;

        Ok(Self { parser })
    }

    pub fn parse(&mut self, source_code: &str) -> Result<tree_sitter::Tree> {
        self.parser.parse(source_code, None)
            .ok_or_else(|| anyhow!("Parsing failed"))
    }
}
