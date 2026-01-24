pub mod walker;

use anyhow::Result;
use sentinel_cpg::{CodePropertyGraph, NodeType};
use tracing::info;

pub struct SymbolicExecutor {
    pub max_depth: usize,
}

impl SymbolicExecutor {
    pub fn new(max_depth: usize) -> Self {
        Self { max_depth }
    }

    pub fn analyze_paths(&self, cpg: &CodePropertyGraph) -> Result<Vec<String>> {
        let mut paths = Vec::new();
        
        // 1. Identify "Sinks" (Vulnerable functions)
        let sinks: Vec<_> = cpg.nodes.iter()
            .filter(|n| matches!(n.node_type, NodeType::Call) && n.name.contains("execute"))
            .collect();

        info!("Symbolic Executor: Discovered {} potential logic sinks in CPG.", sinks.len());

        for sink in sinks {
            let mut visited = std::collections::HashSet::new();
            let mut current_path = Vec::new();
            
            // 2. Perform Recursive Backwards Traversal
            self.explore_backwards(sink.id, cpg, &mut visited, &mut current_path, &mut paths);
        }

        Ok(paths)
    }

    fn explore_backwards(&self, node_id: u64, cpg: &CodePropertyGraph, visited: &mut std::collections::HashSet<u64>, current_path: &mut Vec<u64>, results: &mut Vec<String>) {
        if visited.len() > self.max_depth || visited.contains(&node_id) {
            return;
        }
        visited.insert(node_id);
        current_path.push(node_id);

        // Check if node is an Input Source
        if let Some(node) = cpg.nodes.iter().find(|n| n.id == node_id) {
            if node.name.contains("input") || matches!(node.node_type, NodeType::Literal) {
                let path_str = current_path.iter().map(|id| format!("node_{}", id)).collect::<Vec<_>>().join(" <- ");
                results.push(format!("Vulnerable Path Found: {}", path_str));
            }
        }

        // Traverse incoming DataFlow edges
        for edge in &cpg.edges {
            if edge.to == node_id && matches!(edge.edge_type, sentinel_cpg::EdgeType::DataFlow) {
                self.explore_backwards(edge.from, cpg, visited, current_path, results);
            }
        }

        current_path.pop();
        visited.remove(&node_id);
    }
}
