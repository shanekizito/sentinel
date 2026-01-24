use anyhow::{Result, anyhow};
use rayon::prelude::*;
use crossbeam::channel::{bounded, Sender, Receiver};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use dashmap::DashMap;
use tracing::{info, span, Level};
use crate::{CodeParser, SupportedLanguage};

// =============================================================================
// SECTION 1: OMEGA INDUSTRIAL PIPELINE
// =============================================================================

/// A massively parallel parsing pipeline with work-stealing and lock-free collection.
pub struct OmegaIndustrialPipeline {
    language: SupportedLanguage,
    worker_count: usize,
    queue_depth: usize,
}

impl OmegaIndustrialPipeline {
    pub fn new(lang: SupportedLanguage) -> Self {
        let workers = num_cpus::get();
        Self {
            language: lang,
            worker_count: workers,
            queue_depth: workers * 4, // 4x depth for load balancing
        }
    }

    /// Streaming parallel analysis with bounded memory.
    pub fn analyze_files_streaming<F>(&self, file_paths: &[PathBuf], mut callback: F) -> Result<usize>
    where
        F: FnMut(PathBuf, tree_sitter::Tree) + Send,
    {
        let _span = span!(Level::INFO, "OmegaPipeline", workers = self.worker_count).entered();
        info!("Omega Pipeline: Starting {} workers for {} files", self.worker_count, file_paths.len());

        let (tx, rx): (Sender<(PathBuf, tree_sitter::Tree)>, Receiver<_>) = bounded(self.queue_depth);
        let processed = AtomicUsize::new(0);
        let lang = self.language.clone();

        // Spawn producer workers using rayon
        let tx_clone = tx.clone();
        let files = file_paths.to_vec();
        let producer = thread::spawn(move || {
            files.par_iter().for_each(|path| {
                if let Ok(source) = std::fs::read_to_string(path) {
                    if let Ok(mut parser) = CodeParser::new(lang.clone()) {
                        if let Ok(tree) = parser.parse(&source) {
                            let _ = tx_clone.send((path.clone(), tree));
                        }
                    }
                }
            });
            drop(tx_clone); // Signal completion
        });

        // Consume results
        drop(tx); // Close sender side
        for (path, tree) in rx {
            callback(path, tree);
            processed.fetch_add(1, Ordering::Relaxed);
        }

        producer.join().map_err(|_| anyhow!("Producer thread panicked"))?;
        let count = processed.load(Ordering::Relaxed);
        info!("Omega Pipeline: Completed {} files", count);
        Ok(count)
    }

    /// Batch analysis returning all results (for compatibility).
    pub fn analyze_files(&self, file_paths: &[PathBuf]) -> Result<Vec<(PathBuf, tree_sitter::Tree)>> {
        let results = DashMap::new();
        
        file_paths.par_iter().for_each(|path| {
            if let Ok(source) = std::fs::read_to_string(path) {
                if let Ok(mut parser) = CodeParser::new(self.language.clone()) {
                    if let Ok(tree) = parser.parse(&source) {
                        results.insert(path.clone(), tree);
                    }
                }
            }
        });

        Ok(results.into_iter().collect())
    }
}

// =============================================================================
// SECTION 2: SEMANTIC CONTEXT (Lock-Free)
// =============================================================================

/// Global State for Cross-Module Symbol Discovery using lock-free structures.
pub struct OmegaSemanticContext {
    pub definitions: DashMap<String, (PathBuf, u64, String)>, // (path, id, type)
    pub call_sites: DashMap<String, Vec<(PathBuf, u64)>>,
    pub imports: DashMap<PathBuf, Vec<String>>,
    definition_count: AtomicUsize,
}

impl OmegaSemanticContext {
    pub fn new() -> Self {
        Self {
            definitions: DashMap::new(),
            call_sites: DashMap::new(),
            imports: DashMap::new(),
            definition_count: AtomicUsize::new(0),
        }
    }

    pub fn register_definition(&self, name: String, path: PathBuf, id: u64, kind: String) {
        self.definitions.insert(name, (path, id, kind));
        self.definition_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn register_call_site(&self, name: String, path: PathBuf, id: u64) {
        self.call_sites.entry(name).or_default().push((path, id));
    }

    pub fn register_import(&self, file: PathBuf, module: String) {
        self.imports.entry(file).or_default().push(module);
    }

    pub fn resolve_symbol(&self, name: &str) -> Option<(PathBuf, u64)> {
        self.definitions.get(name).map(|r| (r.0.clone(), r.1))
    }

    pub fn stats(&self) -> (usize, usize) {
        (self.definition_count.load(Ordering::Relaxed), self.call_sites.len())
    }
}

// =============================================================================
// SECTION 3: INCREMENTAL PARSING CACHE
// =============================================================================

/// Caches parsed ASTs to avoid re-parsing unchanged files.
pub struct IncrementalParseCache {
    cache: DashMap<PathBuf, (u64, tree_sitter::Tree)>, // (mtime, tree)
}

impl IncrementalParseCache {
    pub fn new() -> Self {
        Self { cache: DashMap::new() }
    }

    pub fn get_or_parse(&self, path: &PathBuf, parser: &mut CodeParser) -> Result<tree_sitter::Tree> {
        let mtime = std::fs::metadata(path)?.modified()?.duration_since(std::time::UNIX_EPOCH)?.as_secs();
        
        if let Some(entry) = self.cache.get(path) {
            if entry.0 == mtime {
                return Ok(entry.1.clone());
            }
        }

        let source = std::fs::read_to_string(path)?;
        let tree = parser.parse(&source)?;
        self.cache.insert(path.clone(), (mtime, tree.clone()));
        Ok(tree)
    }

    pub fn invalidate(&self, path: &PathBuf) {
        self.cache.remove(path);
    }

    pub fn clear(&self) {
        self.cache.clear();
    }
}
