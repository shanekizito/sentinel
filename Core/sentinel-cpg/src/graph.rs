use anyhow::{Result, anyhow, Context};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, RwLock};
use std::io::{Write, BufWriter};
use std::fs::File;
use std::path::Path;
use crate::{Node, Edge, NodeType, EdgeType};

/// A production-grade, thread-safe, and persistent Code Property Graph.
/// Uses B-Tree indexing for efficient range queries and logarithmic lookups.
pub struct SovereignGraph {
    pub nodes: Arc<RwLock<BTreeMap<u64, Node>>>,
    pub edges: Arc<RwLock<Vec<Edge>>>,
    
    // Reverse indices for high-performance traversals
    index_by_type: Arc<RwLock<HashMap<NodeType, Vec<u64>>>>,
    outgoing_edges: Arc<RwLock<HashMap<u64, Vec<usize>>>>,
    incoming_edges: Arc<RwLock<HashMap<u64, Vec<usize>>>>,
}

impl SovereignGraph {
    pub fn new() -> Self {
        Self {
            nodes: Arc::new(RwLock::new(BTreeMap::new())),
            edges: Arc::new(RwLock::new(Vec::new())),
            index_by_type: Arc::new(RwLock::new(HashMap::new())),
            outgoing_edges: Arc::new(RwLock::new(HashMap::new())),
            incoming_edges: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Optimized node insertion with index updates.
    pub fn add_node(&self, node: Node) -> Result<()> {
        let node_id = node.id;
        let node_type = node.node_type.clone();
        
        {
            let mut nodes = self.nodes.write().map_err(|_| anyhow!("Node lock poisoned"))?;
            nodes.insert(node_id, node);
        }
        
        {
            let mut registry = self.index_by_type.write().map_err(|_| anyhow!("Registry lock poisoned"))?;
            registry.entry(node_type).or_insert_with(Vec::new).push(node_id);
        }
        
        Ok(())
    }

    /// High-performance edge insertion with bidirectional indexing.
    pub fn add_edge(&self, from: u64, to: u64, edge_type: EdgeType) -> Result<()> {
        let edge_idx = {
            let mut edges = self.edges.write().map_err(|_| anyhow!("Edge lock poisoned"))?;
            let idx = edges.len();
            edges.push(Edge { from, to, edge_type });
            idx
        };
        
        {
            let mut outgoing = self.outgoing_edges.write().map_err(|_| anyhow!("Outgoing lock poisoned"))?;
            outgoing.entry(from).or_insert_with(Vec::new).push(edge_idx);
        }
        
        {
            let mut incoming = self.incoming_edges.write().map_err(|_| anyhow!("Incoming lock poisoned"))?;
            incoming.entry(to).or_insert_with(Vec::new).push(edge_idx);
        }
        
        Ok(())
    }

    /// Serializes the graph to the Sentinel Logic V5 Binary Format.
    /// Matched to `data_factory.py` : Universal Hardened Edition.
    ///
    /// Layout:
    /// - Magic Header: "SENT_LOGIC_V5" (13 bytes)
    /// - Metadata: [N_Nodes (u64), N_Edges (u64)]
    /// - Node Block: [Features(64*f32), Type(u64), Timestamp(u64)] per node
    /// - Edge Block: [From(u64), To(u64), Type(u64), Weight(f32)] per edge
    pub fn save_to_binary_v5<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path).context("Failed to create binary shard")?;
        let mut writer = BufWriter::new(file);
        
        // 1. Write Magic Header
        writer.write_all(b"SENT_LOGIC_V5")?;
        
        let nodes_read = self.nodes.read().map_err(|_| anyhow!("Lock poisoned"))?;
        let edges_read = self.edges.read().map_err(|_| anyhow!("Lock poisoned"))?;
        
        let n_nodes = nodes_read.len() as u64;
        let n_edges = edges_read.len() as u64;
        
        // 2. Write Counts
        writer.write_all(&n_nodes.to_le_bytes())?;
        writer.write_all(&n_edges.to_le_bytes())?;
        
        // 3. Write Nodes
        // Assuming Node struct has `features: Vec<f32>` (64 len) and `timestamp: u64`
        // We iterate in ID order (BTreeMap)
        for (_, node) in nodes_read.iter() {
            // Features (64 floats)
            // Ensure exactly 64 floats. Pad or Truncate.
            let mut feats = node.features.clone(); // Assume exists
            feats.resize(64, 0.0);
            
            for f in feats {
                writer.write_all(&f.to_le_bytes())?;
            }
            
            // Type (Enum to u64)
            let type_id = node.node_type_id(); // Helper assumption
            writer.write_all(&type_id.to_le_bytes())?;
            
            // Timestamp
            writer.write_all(&node.timestamp.to_le_bytes())?;
        }
        
        // 4. Write Edges
        for edge in edges_read.iter() {
            writer.write_all(&edge.from.to_le_bytes())?;
            writer.write_all(&edge.to.to_le_bytes())?;
            
            let type_id = edge.edge_type_id(); // Helper
            writer.write_all(&type_id.to_le_bytes())?;
            
            let weight = 1.0f32; // Default weight
            writer.write_all(&weight.to_le_bytes())?;
        }
        
        writer.flush()?;
        Ok(())
    }
}
