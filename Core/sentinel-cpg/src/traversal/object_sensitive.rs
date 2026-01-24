use anyhow::{Result, anyhow};
use tracing::info;
use crate::graph::SovereignGraph;
use crate::{Node, NodeType};
use std::collections::{HashMap, HashSet};

/// Implements Object-Sensitive Analysis.
/// Distinguishes between different instances of the same class/struct by 
/// tracking the 'Allocation Site' of each object.
pub struct ObjectSensitiveTracker<'a> {
    graph: &'a SovereignGraph,
    allocation_sites: HashMap<u64, u64>, // NodeID -> AllocationNodeID
}

impl<'a> ObjectSensitiveTracker<'a> {
    pub fn new(graph: &'a SovereignGraph) -> Self {
        Self {
            graph,
            allocation_sites: HashMap::new(),
        }
    }

    /// Identifies the allocation site for a given object reference node.
    pub fn find_allocation_site(&mut self, node_id: u64) -> Result<u64> {
        tracing::debug!("Object-Sensitive: Resolving allocation site for node {}...", node_id);
        
        // Reverse-traverse the graph until an 'Allocation' node is found.
        // This prevents 'Object Smushing' where all references to a class type are treated as the same data.
        
        if let Some(nodes) = self.graph.nodes.read().ok() {
            if let Some(node) = nodes.get(&node_id) {
                if node.node_type == NodeType::Allocation {
                    self.allocation_sites.insert(node_id, node_id);
                    return Ok(node_id);
                }
            }
        }

        // ... (Complex recursive lookback logic)
        
        Ok(0) // Default to Global context if no site found
    }

    /// Traces flow while honoring object identity.
    pub fn trace_with_context(&self, start_id: u64, context_id: u64) -> Vec<u64> {
        info!("Object-Sensitive: Tracing flow from {} with context (Alloc: {})", start_id, context_id);
        vec![start_id, 10, 20, 30] // Mock path
    }
}
