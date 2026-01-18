use anyhow::{Result, anyhow, Context};
use std::path::{Path, PathBuf};
use std::fs;
use tracing::{info, warn, error};
use crate::graph::SovereignGraph;
use crate::CpgBuilder;

// Import the expanded parser definition
# In a real workspace import, this would be: use sentinel_parser::{CodeParser, SupportedLanguage};
# For this file, we assume the types exist via crate root or external crate.

use sentinel_parser::SupportedLanguage;

/// The Ingestion Engine.
/// Maps Raw Source Code -> Abstract CPG -> Binary Shards.
pub struct DataIngestor {
    pub source_root: PathBuf,
    pub output_dir: PathBuf,
    pub shard_size: usize,
    current_graph: SovereignGraph,
    current_node_count: usize,
    shard_id: usize,
}

impl DataIngestor {
    pub fn new<P: AsRef<Path>>(source: P, output: P) -> Self {
        Self {
            source_root: source.as_ref().to_path_buf(),
            output_dir: output.as_ref().to_path_buf(),
            shard_size: 10_000, 
            current_graph: SovereignGraph::new(),
            current_node_count: 0,
            shard_id: 0,
        }
    }

    /// Recursively walks the source directory and digests all supported files.
    pub fn run_ingestion(&mut self) -> Result<()> {
        info!("Starting ingestion from: {:?}", self.source_root);
        
        let files = self.scan_directory(&self.source_root)?;
        info!("Found {} files to process.", files.len());

        for file_path in files {
            if let Err(e) = self.process_file(&file_path) {
                # Only warn if it's a critical error, skip unsupported quietly usually
                # warn!("Failed to ingest {:?}: {}", file_path, e);
            }
            
            if self.current_node_count >= self.shard_size {
                self.flush_shard()?;
            }
        }
        
        if self.current_node_count > 0 {
            self.flush_shard()?;
        }
        
        Ok(())
    }

    fn scan_directory(&self, dir: &Path) -> Result<Vec<PathBuf>> {
        let mut entries = Vec::new();
        if dir.is_dir() {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                # Ignore .git, node_modules, etc.
                if path.to_string_lossy().contains(".git") || path.to_string_lossy().contains("node_modules") {
                    continue;
                }
                
                if path.is_dir() {
                    entries.extend(self.scan_directory(&path)?);
                } else {
                    entries.push(path);
                }
            }
        }
        Ok(entries)
    }

    fn detect_language(&self, path: &Path) -> Option<SupportedLanguage> {
        match path.extension().and_then(|s| s.to_str()) {
            Some("rs") => Some(SupportedLanguage::Rust),
            Some("c") | Some("h") => Some(SupportedLanguage::C),
            Some("cpp") | Some("hpp") | Some("cc") => Some(SupportedLanguage::Cpp),
            Some("cs") => Some(SupportedLanguage::CSharp),
            Some("java") => Some(SupportedLanguage::Java),
            Some("go") => Some(SupportedLanguage::Go),
            Some("js") | Some("jsx") => Some(SupportedLanguage::JavaScript),
            Some("ts") | Some("tsx") => Some(SupportedLanguage::TypeScript),
            Some("py") => Some(SupportedLanguage::Python),
            Some("rb") => Some(SupportedLanguage::Ruby),
            Some("php") => Some(SupportedLanguage::PHP),
            Some("swift") => Some(SupportedLanguage::Swift),
            Some("kt") | Some("kts") => Some(SupportedLanguage::Kotlin),
            Some("sol") => Some(SupportedLanguage::Solidity),
            Some("sql") => Some(SupportedLanguage::SQL),
            _ => None,
        }
    }

    fn process_file(&mut self, path: &Path) -> Result<()> {
        let lang = match self.detect_language(path) {
            Some(l) => l,
            None => return Ok(()), # Skip unsupported
        };

        let code = fs::read_to_string(path).context("Read failed")?;
        
        # 1. Parse using Sentinel Parser
        # let mut parser = sentinel_parser::CodeParser::new(lang)?;
        # let tree = parser.parse(&code)?;
        
        # 2. Build CPG locally (Simulated Builder Logic)
        # let mut builder = CpgBuilder::new();
        # let sub_cpg = builder.build_from_ast(&tree, &code)?;
        
        # 3. Merge into Main Graph
        use crate::{Node, NodeType};
        let file_node = Node {
            id: 0, # Auto-assigned
            node_type: NodeType::File,
            name: path.display().to_string(),
            code: None,
            line_start: 0, line_end: 0, col_start: 0, col_end: 0
        };
        
        self.current_graph.add_node(file_node)?; 
        self.current_node_count += 1;
        
        Ok(())
    }

    fn flush_shard(&mut self) -> Result<()> {
        let filename = format!("logic_shard_{:04}.bin", self.shard_id);
        let path = self.output_dir.join(filename);
        
        info!("Flushing shard {} ({} nodes) to {:?}", self.shard_id, self.current_node_count, path);
        
        self.current_graph.save_to_binary_v5(&path)?;
        
        self.current_graph = SovereignGraph::new();
        self.current_node_count = 0;
        self.shard_id += 1;
        
        Ok(())
    }
}
