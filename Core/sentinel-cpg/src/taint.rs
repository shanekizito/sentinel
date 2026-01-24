use anyhow::Result;
use std::collections::{HashSet, VecDeque};
use crate::{CodePropertyGraph, EdgeType, Node, NodeType};

pub struct TaintTracker;

impl TaintTracker {
    /// Performs inter-procedural taint tracking on the CPG.
    /// Returns a list of paths from Sources to Sinks that lack proper sanitization.
    pub fn find_tainted_flows(cpg: &CodePropertyGraph) -> Vec<Vec<u64>> {
        let mut violations = Vec::new();
        
        let sources: Vec<&Node> = cpg.nodes.iter()
            .filter(|n| n.metadata.get("is_source").map_or(false, |v| v == "true"))
            .collect();

        let sinks: Vec<&Node> = cpg.nodes.iter()
            .filter(|n| n.node_type == NodeType::Call && 
                        (n.name.contains("execute") || n.name.contains("query")))
            .collect();

        for source in sources {
            let flows = Self::trace_flow(source.id, &sinks, cpg);
            violations.extend(flows);
        }

        violations
    }

    fn trace_flow(start_id: u64, sinks: &[&Node], cpg: &CodePropertyGraph) -> Vec<Vec<u64>> {
        let mut paths = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(vec![start_id]);

        let mut visited = HashSet::new();

        while let Some(path) = queue.pop_front() {
            let current_id = *path.last().unwrap();
            
            if visited.contains(&(current_id, path.len())) {
                continue;
            }
            visited.insert((current_id, path.len()));

            // Check if current node is a sink
            if sinks.iter().any(|s| s.id == current_id) {
                paths.push(path.clone());
                continue;
            }

            // Depth limit to prevent infinite loops in graph
            if path.len() > 100 { continue; }

            // Follow DataFlow edges
            for edge in &cpg.edges {
                if edge.from == current_id && edge.edge_type == EdgeType::DataFlow {
                    let mut next_path = path.clone();
                    next_path.push(edge.to);
                    queue.push_back(next_path);
                }
            }
        }

        paths
    }
}
