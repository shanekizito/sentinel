use anyhow::{Result, anyhow};
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

/// A high-performance consistent hashing ring for sharding million-node graphs.
/// Uses 'Virtual Nodes' to ensure uniform load distribution across the mesh.
pub struct ShardingRing {
    nodes: BTreeMap<u64, String>, // Hash -> NodeID
    replicas: usize,
}

impl ShardingRing {
    pub fn new(replicas: usize) -> Self {
        Self {
            nodes: BTreeMap::new(),
            replicas,
        }
    }

    /// Adds a physical node to the ring as multiple virtual nodes.
    pub fn add_node(&mut self, node_id: &str) {
        for i in 0..self.replicas {
            let hash = self.hash(&format!("{}-{}", node_id, i));
            self.nodes.insert(hash, node_id.to_string());
        }
    }

    /// Finds the target node for a given sharding key (e.g., RepoURL or TenantID).
    pub fn get_node(&self, key: &str) -> Option<&String> {
        if self.nodes.is_empty() { return None; }
        
        let hash = self.hash(key);
        // Find the first node with hash >= key_hash
        let mut iter = self.nodes.range(hash..);
        
        if let Some((_, node_id)) = iter.next() {
            Some(node_id)
        } else {
            // Wrap around to the start of the ring
            self.nodes.values().next()
        }
    }

    fn hash(&self, key: &str) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }
}
