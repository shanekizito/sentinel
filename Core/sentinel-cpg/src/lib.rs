pub mod taint;
pub mod mmap_engine;
pub mod graph;
pub mod traversal;
pub mod ingest;

use serde::{Deserialize, Serialize};

use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum NodeType {
    File,
    Namespace,
    Class,
    Function,
    Variable,
    Literal,
    Call,
    MemberAccess,
    Allocation,
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
    pub features: Vec<f32>,
    pub metadata: HashMap<String, String>,
    pub timestamp: u64,
}

impl Node {
    pub fn node_type_id(&self) -> u64 {
        match self.node_type {
            NodeType::File => 1,
            NodeType::Namespace => 2,
            NodeType::Class => 3,
            NodeType::Function => 4,
            NodeType::Variable => 5,
            NodeType::Literal => 6,
            NodeType::Call => 7,
            NodeType::MemberAccess => 8,
            NodeType::Allocation => 9,
            NodeType::ControlFlow => 10,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
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

impl Edge {
    pub fn edge_type_id(&self) -> u64 {
        match self.edge_type {
            EdgeType::Contains => 1,
            EdgeType::Calls => 2,
            EdgeType::DataFlow => 3,
            EdgeType::ControlFlow => 4,
            EdgeType::Inherits => 5,
        }
    }
}

pub struct CodePropertyGraph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

impl CodePropertyGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    pub fn add_node(&mut self, node: Node) {
        self.nodes.push(node);
    }

    pub fn add_edge(&mut self, from: u64, to: u64, edge_type: EdgeType) {
        self.edges.push(Edge { from, to, edge_type });
    }
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
            features: vec![0.0; 64],
            metadata: HashMap::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
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
