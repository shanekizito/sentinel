pub mod resolver;
pub mod bloom;
pub mod pipeline;

use anyhow::{Result, anyhow};
use tree_sitter::Parser;

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
            SupportedLanguage::Rust => tree_sitter_rust::language(),
            SupportedLanguage::C => tree_sitter_c::language(),
            SupportedLanguage::Cpp => tree_sitter_cpp::language(),
            SupportedLanguage::Java => tree_sitter_java::language(),
            SupportedLanguage::Go => tree_sitter_go::language(),
            SupportedLanguage::JavaScript => tree_sitter_javascript::language(),
            SupportedLanguage::TypeScript => tree_sitter_typescript::language_typescript(),
            SupportedLanguage::Python => tree_sitter_python::language(),
            SupportedLanguage::CSharp | SupportedLanguage::Ruby | SupportedLanguage::PHP | SupportedLanguage::Swift | SupportedLanguage::Kotlin | SupportedLanguage::Solidity | SupportedLanguage::SQL => {
                return Err(anyhow!("Language {:?} is currently in development (SoV Phase 2)", lang));
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
