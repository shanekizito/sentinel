use anyhow::{Result, anyhow};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use crate::{CodeParser, SupportedLanguage};

/// A massively parallel parsing pipeline capable of processing million-file projects.
pub struct IndustrialPipeline {
    language: SupportedLanguage,
    concurrency_limit: usize,
}

impl IndustrialPipeline {
    pub fn new(lang: SupportedLanguage) -> Self {
        Self {
            language: lang,
            concurrency_limit: num_cpus::get(),
        }
    }

    /// Parallelized file analysis using Rayon.
    /// Distributes parsing tasks across all available CPU cores.
    pub fn analyze_files(&self, file_paths: &[PathBuf]) -> Result<Vec<(PathBuf, tree_sitter::Tree)>> {
        tracing::info!("Industrial Pipeline: Spawning {} parallel parsing threads...", self.concurrency_limit);
        
        let results = Arc::new(Mutex::new(Vec::new()));
        
        file_paths.par_iter().for_each(|path| {
            if let Ok(source) = std::fs::read_to_string(path) {
                if let Ok(mut parser) = CodeParser::new(self.language.clone()) {
                    if let Ok(tree) = parser.parse(&source) {
                        let mut res = results.lock().unwrap();
                        res.push((path.clone(), tree));
                    }
                }
            }
        });
        
        let final_results = Arc::try_unwrap(results).unwrap().into_inner().unwrap();
        tracing::info!("Industrial Pipeline: Successfully analyzed {} files.", final_results.len());
        
        Ok(final_results)
    }
}

/// Global State for Cross-Module Symbol Discovery.
pub struct SemanticContext {
    pub definitions: dashmap::DashMap<String, (PathBuf, u64)>,
    pub call_sites: dashmap::DashMap<String, Vec<(PathBuf, u64)>>,
}

impl SemanticContext {
    pub fn new() -> Self {
        Self {
            definitions: dashmap::DashMap::new(),
            call_sites: dashmap::DashMap::new(),
        }
    }

    /// Registers a symbol definition across file boundaries.
    pub fn register_definition(&self, name: String, path: PathBuf, id: u64) {
        self.definitions.insert(name, (path, id));
    }
}
