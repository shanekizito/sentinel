pub mod cwe;

use anyhow::Result;
use sentinel_cpg::{CodePropertyGraph, NodeType};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Finding {
    pub rule_id: String,
    pub message: String,
    pub severity: Severity,
    pub line: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Severity {
    High,
    Medium,
    Low,
}

pub trait Rule {
    fn metadata(&self) -> RuleMetadata;
    fn check(&self, cpg: &CodePropertyGraph) -> Result<Vec<Finding>>;
}

#[derive(Debug)]
pub struct RuleMetadata {
    pub id: String,
    pub description: String,
}

pub struct RuleEngine {
    rules: Vec<Box<dyn Rule>>,
}

impl RuleEngine {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    pub fn add_rule(&mut self, rule: Box<dyn Rule>) {
        self.rules.push(rule);
    }

    pub fn run(&self, cpg: &CodePropertyGraph) -> Result<Vec<Finding>> {
        let mut all_findings = Vec::new();
        for rule in &self.rules {
            let findings = rule.check(cpg)?;
            all_findings.extend(findings);
        }
        Ok(all_findings)
    }
}

// Example Rule: Search for hardcoded API keys/secrets in literals
pub struct SecretsRule;

impl Rule for SecretsRule {
    fn metadata(&self) -> RuleMetadata {
        RuleMetadata {
            id: "SENT-001".to_string(),
            description: "Detects potential hardcoded secrets in strings".to_string(),
        }
    }

    fn check(&self, cpg: &CodePropertyGraph) -> Result<Vec<Finding>> {
        let mut findings = Vec::new();
        let secret_patterns = ["key", "password", "secret", "token"];

        for node in &cpg.nodes {
            if matches!(node.node_type, NodeType::Variable | NodeType::Literal) {
                if let Some(code) = &node.code {
                    let code_lower = code.to_lowercase();
                    if secret_patterns.iter().any(|&p| code_lower.contains(p) && code.len() > 10) {
                        findings.push(Finding {
                            rule_id: self.metadata().id,
                            message: format!("Potential secret found in: {}", node.name),
                            severity: Severity::High,
                            line: node.line_start,
                        });
                    }
                }
            }
        }
        Ok(findings)
    }
}
