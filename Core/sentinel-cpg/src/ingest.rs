use anyhow::{Result, anyhow, Context};
use std::path::{Path, PathBuf};
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::sync::atomic::{AtomicUsize, Ordering};
use rayon::prelude::*;
use crossbeam::channel::{bounded, Sender};
use tracing::{info, warn, error, span, Level};
use memmap2::MmapMut;
use crate::graph::SovereignGraph;
use crate::{Node, NodeType, Edge, EdgeType};
use sentinel_parser::SupportedLanguage;

// =============================================================================
// SECTION 1: OMEGA DATA INGESTOR
// =============================================================================

/// The Ultimate Ingestion Engine.
/// Maps Raw Source Code -> Abstract CPG -> Binary Shards with maximum parallelism.
pub struct OmegaDataIngestor {
    pub source_root: PathBuf,
    pub output_dir: PathBuf,
    pub shard_size: usize,
    pub worker_count: usize,
    total_nodes: AtomicUsize,
    total_files: AtomicUsize,
}

impl OmegaDataIngestor {
    pub fn new<P: AsRef<Path>>(source: P, output: P) -> Self {
        Self {
            source_root: source.as_ref().to_path_buf(),
            output_dir: output.as_ref().to_path_buf(),
            shard_size: 50_000,
            worker_count: num_cpus::get(),
            total_nodes: AtomicUsize::new(0),
            total_files: AtomicUsize::new(0),
        }
    }

    /// Parallel ingestion with streaming shard output.
    pub fn run_ingestion(&self) -> Result<IngestionStats> {
        let _span = span!(Level::INFO, "OmegaIngestion").entered();
        info!("Omega Ingestor: Starting parallel ingestion from {:?}", self.source_root);
        
        // 1. Parallel file discovery
        let files = self.scan_directory_parallel(&self.source_root)?;
        info!("Omega Ingestor: Discovered {} files", files.len());

        // 2. Create output directory
        fs::create_dir_all(&self.output_dir)?;

        // 3. Parallel processing with sharding
        let (tx, rx) = bounded::<(PathBuf, SovereignGraph)>(self.worker_count * 2);
        let output_dir = self.output_dir.clone();
        let shard_size = self.shard_size;
        
        // Shard writer thread
        let writer_handle = std::thread::spawn(move || {
            let mut current_graph = SovereignGraph::new();
            let mut current_count = 0usize;
            let mut shard_id = 0usize;
            
            for (_path, sub_graph) in rx {
                current_graph.merge(&sub_graph);
                current_count += sub_graph.node_count();
                
                if current_count >= shard_size {
                    let filename = format!("logic_shard_{:04}.bin", shard_id);
                    let path = output_dir.join(&filename);
                    if let Err(e) = current_graph.save_to_binary_v6(&path) {
                        error!("Failed to write shard {}: {}", shard_id, e);
                    } else {
                        info!("Omega Ingestor: Flushed shard {} ({} nodes)", shard_id, current_count);
                    }
                    current_graph = SovereignGraph::new();
                    current_count = 0;
                    shard_id += 1;
                }
            }
            
            // Flush remaining
            if current_count > 0 {
                let filename = format!("logic_shard_{:04}.bin", shard_id);
                let path = output_dir.join(&filename);
                let _ = current_graph.save_to_binary_v6(&path);
                shard_id += 1;
            }
            
            shard_id
        });

        // Parallel file processing
        let tx_clone = tx;
        let total_nodes = &self.total_nodes;
        let total_files = &self.total_files;
        
        files.par_iter().for_each(|path| {
            if let Some(graph) = self.process_file_parallel(path) {
                total_nodes.fetch_add(graph.node_count(), Ordering::Relaxed);
                total_files.fetch_add(1, Ordering::Relaxed);
                let _ = tx_clone.send((path.clone(), graph));
            }
        });
        
        drop(tx_clone);
        let shard_count = writer_handle.join().map_err(|_| anyhow!("Writer thread panicked"))?;

        Ok(IngestionStats {
            files_processed: self.total_files.load(Ordering::Relaxed),
            nodes_created: self.total_nodes.load(Ordering::Relaxed),
            shards_written: shard_count,
        })
    }

    fn scan_directory_parallel(&self, dir: &Path) -> Result<Vec<PathBuf>> {
        let entries: Vec<PathBuf> = walkdir::WalkDir::new(dir)
            .into_iter()
            .par_bridge()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .filter(|e| !e.path().to_string_lossy().contains(".git"))
            .filter(|e| !e.path().to_string_lossy().contains("node_modules"))
            .filter(|e| !e.path().to_string_lossy().contains("target"))
            .filter(|e| self.detect_language(e.path()).is_some())
            .map(|e| e.path().to_path_buf())
            .collect();
        Ok(entries)
    }

    fn detect_language(&self, path: &Path) -> Option<SupportedLanguage> {
        match path.extension().and_then(|s| s.to_str()) {
            Some("rs") => Some(SupportedLanguage::Rust),
            Some("c") | Some("h") => Some(SupportedLanguage::C),
            Some("cpp") | Some("hpp") | Some("cc") | Some("cxx") => Some(SupportedLanguage::Cpp),
            Some("cs") => Some(SupportedLanguage::CSharp),
            Some("java") => Some(SupportedLanguage::Java),
            Some("go") => Some(SupportedLanguage::Go),
            Some("js") | Some("jsx") | Some("mjs") => Some(SupportedLanguage::JavaScript),
            Some("ts") | Some("tsx") => Some(SupportedLanguage::TypeScript),
            Some("py") | Some("pyi") => Some(SupportedLanguage::Python),
            Some("rb") => Some(SupportedLanguage::Ruby),
            Some("php") => Some(SupportedLanguage::PHP),
            Some("swift") => Some(SupportedLanguage::Swift),
            Some("kt") | Some("kts") => Some(SupportedLanguage::Kotlin),
            Some("sol") => Some(SupportedLanguage::Solidity),
            Some("sql") => Some(SupportedLanguage::SQL),
            _ => None,
        }
    }

    fn process_file_parallel(&self, path: &Path) -> Option<SovereignGraph> {
        let lang = self.detect_language(path)?;
        let code = fs::read_to_string(path).ok()?;
        
        let mut graph = SovereignGraph::new();
        
        // Create file node
        let file_node = Node {
            id: 0,
            node_type: NodeType::File,
            name: path.file_name()?.to_string_lossy().to_string(),
            code: None,
            line_start: 0, line_end: code.lines().count() as u32,
            col_start: 0, col_end: 0,
        };
        graph.add_node(file_node).ok()?;
        
        // Parse and extract symbols (simplified)
        let lines: Vec<&str> = code.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if line.contains("fn ") || line.contains("def ") || line.contains("function ") {
                let func_node = Node {
                    id: 0,
                    node_type: NodeType::Function,
                    name: format!("func_{}_{}", path.file_stem()?.to_string_lossy(), i),
                    code: Some(line.to_string()),
                    line_start: i as u32, line_end: i as u32,
                    col_start: 0, col_end: line.len() as u32,
                };
                graph.add_node(func_node).ok()?;
            }
        }

        Some(graph)
    }
}

// =============================================================================
// SECTION 2: INGESTION STATISTICS
// =============================================================================

#[derive(Debug, Clone)]
pub struct IngestionStats {
    pub files_processed: usize,
    pub nodes_created: usize,
    pub shards_written: usize,
}

impl std::fmt::Display for IngestionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Files: {}, Nodes: {}, Shards: {}", 
            self.files_processed, self.nodes_created, self.shards_written)
    }
}
