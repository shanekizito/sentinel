pub mod taint;
pub mod mmap_engine;
pub mod graph;
pub mod traversal;
pub mod ingest;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeType {
    File,
    Namespace,
    Class,
    Function,
    Variable,
    Literal,
    Call,
    ControlFlow, // If, Loop, etc.
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: u64,
    pub node_type: NodeType,
    pub name: String,
    pub code: Option<String>,
    pub line_start: usize,
    pub line_end: usize,
    pub col_start: usize,
    pub col_end: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeType {
    Contains,      // File -> Function
    Calls,         // Function -> Function
    DataFlow,      // Variable -> Variable
    ControlFlow,   // Op -> Op
    Inherits,      // Class -> Class
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub from: u64,
    pub to: u64,
    pub edge_type: EdgeType,
}

pub struct CodePropertyGraph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

pub struct CpgBuilder {
    cpg: CodePropertyGraph,
    next_id: u64,
}

impl CpgBuilder {
    pub fn new() -> Self {
        Self {
            cpg: CodePropertyGraph::new(),
            next_id: 1,
        }
    }

    fn next_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn build_from_ast(&mut self, tree: &tree_sitter::Tree, source: &str) -> Result<CodePropertyGraph, anyhow::Error> {
        let root_node = tree.root_node();
        self.visit_node(root_node, source, 0)?;
        Ok(std::mem::replace(&mut self.cpg, CodePropertyGraph::new()))
    }

    fn visit_node(&mut self, ts_node: tree_sitter::Node, source: &str, parent_id: u64) -> Result<u64, anyhow::Error> {
        let id = self.next_id();
        let name = ts_node.kind().to_string();
        
        // Map Tree-sitter node kinds to Sentinel NodeTypes (simplified mapping)
        let node_type = match ts_node.kind() {
            "function_declaration" | "method_definition" => NodeType::Function,
            "class_declaration" => NodeType::Class,
            "variable_declarator" => NodeType::Variable,
            "call_expression" => NodeType::Call,
            _ => NodeType::ControlFlow,
        };

        let node = Node {
            id,
            node_type,
            name,
            code: Some(ts_node.utf8_text(source.as_bytes())?.to_string()),
            line_start: ts_node.start_position().row,
            line_end: ts_node.end_position().row,
            col_start: ts_node.start_position().column,
            col_end: ts_node.end_position().column,
        };

        self.cpg.add_node(node);

        if parent_id != 0 {
            self.cpg.add_edge(parent_id, id, EdgeType::Contains);
        }

        // Recursively visit children
        let mut cursor = ts_node.walk();
        for child in ts_node.children(&mut cursor) {
            self.visit_node(child, source, id)?;
        }

        Ok(id)
    }
}
