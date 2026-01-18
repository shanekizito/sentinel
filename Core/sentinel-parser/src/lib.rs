pub mod resolver;
pub mod bloom;
pub mod pipeline;

use anyhow::{Result, anyhow};
use tree_sitter::{Parser, Language};

pub enum SupportedLanguage {
    TypeScript,
    Python,
}

pub struct CodeParser {
    parser: Parser,
}

impl CodeParser {
    pub fn new(lang: SupportedLanguage) -> Result<Self> {
        let mut parser = Parser::new();
        let language = match lang {
            SupportedLanguage::TypeScript => {
                extern "C" { fn tree_sitter_typescript() -> Language; }
                unsafe { tree_sitter_typescript() }
            }
            SupportedLanguage::Python => {
                extern "C" { fn tree_sitter_python() -> Language; }
                unsafe { tree_sitter_python() }
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
