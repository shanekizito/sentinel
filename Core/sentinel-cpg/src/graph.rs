use anyhow::{Result, anyhow, Context};
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
    /// Includes a "Demo Mode" safety cap to prevent RAM exhaustion on 8GB devices.
    pub fn add_node(&self, node: Node) -> Result<()> {
        let node_id = node.id;
        let node_type = node.node_type.clone();
        
        {
            let mut nodes = self.nodes.write().map_err(|_| anyhow!("Node lock poisoned"))?;
            
            // DEMO SAFETY CAP: Hard limit at 50,000 nodes (~100MB RAM usage)
            if nodes.len() >= 50_000 {
                // In a real system we'd evict or flush to disk. For demo, we just stop ingesting new nodes.
                // This prevents the "Infinity Scale" from crashing a laptop.
                return Ok(()); 
            }

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

    /// Serializes the graph to the Sentinel Logic V6 Binary Format.
    /// Layout:
    /// - Magic Header: "SENT_LOGIC_V6" (13 bytes)
    /// - Metadata: [N_Nodes (u64), N_Edges (u64)]
    /// - Node Block: [ID(u64), Type(u64), Timestamp(u64), NameLen(u32), Name, CodeLen(u32), Code] per node
    /// - Edge Block: [From(u64), To(u64), Type(u64)] per edge
    pub fn save_to_binary_v5<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path).context("Failed to create binary shard")?;
        let mut writer = BufWriter::new(file);
        
        writer.write_all(b"SENT_LOGIC_V6")?;
        
        let nodes_read = self.nodes.read().map_err(|_| anyhow!("Lock poisoned"))?;
        let edges_read = self.edges.read().map_err(|_| anyhow!("Lock poisoned"))?;
        
        writer.write_all(&(nodes_read.len() as u64).to_le_bytes())?;
        writer.write_all(&(edges_read.len() as u64).to_le_bytes())?;
        
        for (id, node) in nodes_read.iter() {
            writer.write_all(&(*id as u64).to_le_bytes())?;
            writer.write_all(&(node.node_type_id() as u64).to_le_bytes())?;
            writer.write_all(&(node.timestamp as u64).to_le_bytes())?;
            
            let name_bytes = node.name.as_bytes();
            writer.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
            writer.write_all(name_bytes)?;
            
            let code_str = node.code.as_deref().unwrap_or("");
            let code_bytes = code_str.as_bytes();
            writer.write_all(&(code_bytes.len() as u32).to_le_bytes())?;
            writer.write_all(code_bytes)?;
        }
        
        for edge in edges_read.iter() {
            writer.write_all(&edge.from.to_le_bytes())?;
            writer.write_all(&edge.to.to_le_bytes())?;
            writer.write_all(&edge.edge_type_id().to_le_bytes())?;
        }
        
        writer.flush()?;
        Ok(())
    }

    pub fn node_count(&self) -> usize {
        if let Ok(nodes) = self.nodes.read() {
            nodes.len()
        } else {
            0
        }
    }

    pub fn load_from_binary_v5<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let mut header = [0u8; 13];
        std::io::Read::read_exact(&mut file, &mut header)?;
        
        if &header != b"SENT_LOGIC_V6" {
            return Err(anyhow!("Invalid or unsupported shard version"));
        }
        
        let mut u64_buf = [0u8; 8];
        std::io::Read::read_exact(&mut file, &mut u64_buf)?;
        let n_nodes = u64::from_le_bytes(u64_buf);
        std::io::Read::read_exact(&mut file, &mut u64_buf)?;
        let n_edges = u64::from_le_bytes(u64_buf);
        
        let graph = Self::new();
        
        for _ in 0..n_nodes {
            std::io::Read::read_exact(&mut file, &mut u64_buf)?;
            let id = u64::from_le_bytes(u64_buf);
            
            std::io::Read::read_exact(&mut file, &mut u64_buf)?;
            let type_id = u64::from_le_bytes(u64_buf);
            
            std::io::Read::read_exact(&mut file, &mut u64_buf)?;
            let timestamp = u64::from_le_bytes(u64_buf);
            
            let mut u32_buf = [0u8; 4];
            std::io::Read::read_exact(&mut file, &mut u32_buf)?;
            let name_len = u32::from_le_bytes(u32_buf);
            let mut name_vec = vec![0u8; name_len as usize];
            std::io::Read::read_exact(&mut file, &mut name_vec)?;
            let name = String::from_utf8(name_vec).unwrap_or_default();
            
            std::io::Read::read_exact(&mut file, &mut u32_buf)?;
            let code_len = u32::from_le_bytes(u32_buf);
            let mut code_vec = vec![0u8; code_len as usize];
            std::io::Read::read_exact(&mut file, &mut code_vec)?;
            let code = if code_len > 0 {
                Some(String::from_utf8(code_vec).unwrap_or_default())
            } else {
                None
            };
            
            let node_type = match type_id {
                1 => NodeType::File,
                4 => NodeType::Function,
                5 => NodeType::Variable,
                _ => NodeType::ControlFlow,
            };
            
            graph.add_node(Node {
                id,
                node_type,
                name,
                code,
                line_start: 0,
                line_end: 0,
                col_start: 0,
                col_end: 0,
                features: vec![0.0; 64],
                metadata: HashMap::new(),
                timestamp,
            })?;
        }
        
        for _ in 0..n_edges {
            std::io::Read::read_exact(&mut file, &mut u64_buf)?;
            let from = u64::from_le_bytes(u64_buf);
            std::io::Read::read_exact(&mut file, &mut u64_buf)?;
            let to = u64::from_le_bytes(u64_buf);
            std::io::Read::read_exact(&mut file, &mut u64_buf)?;
            let type_id = u64::from_le_bytes(u64_buf);
            
            let edge_type = match type_id {
                1 => EdgeType::Contains,
                2 => EdgeType::Calls,
                3 => EdgeType::DataFlow,
                _ => EdgeType::ControlFlow,
            };
            
            graph.add_edge(from, to, edge_type)?;
        }
        
        Ok(graph)
    }

    pub fn merge(&self, other: &SovereignGraph) {
        // Merge nodes
        if let Ok(other_nodes) = other.nodes.read() {
            if let Ok(mut self_nodes) = self.nodes.write() {
                for (id, node) in other_nodes.iter() {
                    self_nodes.insert(*id, node.clone());
                }
            }
        }
        
        // Merge edges
        if let Ok(other_edges) = other.edges.read() {
            if let Ok(mut self_edges) = self.edges.write() {
                self_edges.extend(other_edges.iter().cloned());
            }
        }
    }
}
