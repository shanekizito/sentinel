pub mod cwe;

use anyhow::Result;
use sentinel_cpg::{CodePropertyGraph, NodeType};
use serde::{Deserialize, Serialize};
use rayon::prelude::*;
use aho_corasick::{AhoCorasick, AhoCorasickBuilder};
use std::sync::Arc;

// =============================================================================
// SECTION 1: CORE DATA STRUCTURES
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    pub rule_id: String,
    pub message: String,
    pub severity: Severity,
    pub line: usize,
    pub column: usize,
    pub file: Option<String>,
    pub cwe_id: Option<String>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Critical = 4,
    High = 3,
    Medium = 2,
    Low = 1,
    Info = 0,
}

pub trait Rule: Send + Sync {
    fn metadata(&self) -> RuleMetadata;
    fn check(&self, cpg: &CodePropertyGraph) -> Result<Vec<Finding>>;
}

#[derive(Debug, Clone)]
pub struct RuleMetadata {
    pub id: String,
    pub description: String,
    pub cwe_ids: Vec<String>,
    pub severity: Severity,
}

// =============================================================================
// SECTION 2: OMEGA RULE ENGINE
// =============================================================================

pub struct OmegaRuleEngine {
    rules: Vec<Arc<dyn Rule>>,
    min_severity: Severity,
}

impl OmegaRuleEngine {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            min_severity: Severity::Low,
        }
    }

    pub fn with_severity_filter(mut self, min: Severity) -> Self {
        self.min_severity = min;
        self
    }

    pub fn add_rule(&mut self, rule: Arc<dyn Rule>) {
        self.rules.push(rule);
    }

    /// Parallel rule execution across all rules.
    pub fn run(&self, cpg: &CodePropertyGraph) -> Result<Vec<Finding>> {
        let findings: Vec<Finding> = self.rules
            .par_iter()
            .flat_map(|rule| {
                rule.check(cpg).unwrap_or_default()
            })
            .filter(|f| f.severity >= self.min_severity)
            .collect();

        Ok(findings)
    }

    /// Export findings in SARIF format.
    pub fn to_sarif(&self, findings: &[Finding]) -> String {
        serde_json::json!({
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [{
                "tool": {
                    "driver": {
                        "name": "Sentinel Omega",
                        "version": "7.0.0"
                    }
                },
                "results": findings.iter().map(|f| {
                    serde_json::json!({
                        "ruleId": f.rule_id,
                        "message": { "text": f.message },
                        "level": match f.severity {
                            Severity::Critical | Severity::High => "error",
                            Severity::Medium => "warning",
                            _ => "note"
                        }
                    })
                }).collect::<Vec<_>>()
            }]
        }).to_string()
    }
}

// =============================================================================
// SECTION 3: AHO-CORASICK PATTERN RULE
// =============================================================================

/// High-performance multi-pattern matching rule using Aho-Corasick automaton.
pub struct AhoCorasickPatternRule {
    metadata: RuleMetadata,
    automaton: AhoCorasick,
    patterns: Vec<String>,
}

impl AhoCorasickPatternRule {
    pub fn new(id: &str, description: &str, patterns: Vec<String>, severity: Severity) -> Self {
        let automaton = AhoCorasickBuilder::new()
            .ascii_case_insensitive(true)
            .build(&patterns)
            .unwrap();
        
        Self {
            metadata: RuleMetadata {
                id: id.to_string(),
                description: description.to_string(),
                cwe_ids: vec![],
                severity,
            },
            automaton,
            patterns,
        }
    }
}

impl Rule for AhoCorasickPatternRule {
    fn metadata(&self) -> RuleMetadata {
        self.metadata.clone()
    }

    fn check(&self, cpg: &CodePropertyGraph) -> Result<Vec<Finding>> {
        let mut findings = Vec::new();

        for node in &cpg.nodes {
            if let Some(code) = &node.code {
                for mat in self.automaton.find_iter(code) {
                    findings.push(Finding {
                        rule_id: self.metadata.id.clone(),
                        message: format!(
                            "Pattern '{}' found",
                            &self.patterns[mat.pattern().as_usize()]
                        ),
                        severity: self.metadata.severity,
                        line: node.line_start,
                        column: mat.start(),
                        file: Some(node.name.clone()),
                        cwe_id: self.metadata.cwe_ids.first().cloned(),
                        confidence: 0.9,
                    });
                }
            }
        }

        Ok(findings)
    }
}

// =============================================================================
// SECTION 4: SECRETS DETECTION RULE
// =============================================================================

pub struct OmegaSecretsRule {
    automaton: AhoCorasick,
    entropy_threshold: f64,
}

impl OmegaSecretsRule {
    pub fn new() -> Self {
        let patterns = vec![
            "api_key", "apikey", "api-key",
            "password", "passwd", "pwd",
            "secret", "token", "bearer",
            "aws_access", "aws_secret",
            "private_key", "ssh-rsa",
        ];
        
        Self {
            automaton: AhoCorasickBuilder::new()
                .ascii_case_insensitive(true)
                .build(&patterns)
                .unwrap(),
            entropy_threshold: 3.5,
        }
    }

    fn shannon_entropy(&self, s: &str) -> f64 {
        let mut freq = [0u32; 256];
        for b in s.bytes() {
            freq[b as usize] += 1;
        }
        let len = s.len() as f64;
        freq.iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / len;
                -p * p.log2()
            })
            .sum()
    }
}

impl Rule for OmegaSecretsRule {
    fn metadata(&self) -> RuleMetadata {
        RuleMetadata {
            id: "SENT-001".to_string(),
            description: "Detects hardcoded secrets using pattern + entropy analysis".to_string(),
            cwe_ids: vec!["CWE-798".to_string()],
            severity: Severity::Critical,
        }
    }

    fn check(&self, cpg: &CodePropertyGraph) -> Result<Vec<Finding>> {
        let mut findings = Vec::new();

        for node in &cpg.nodes {
            if !matches!(node.node_type, NodeType::Variable | NodeType::Literal) {
                continue;
            }

            if let Some(code) = &node.code {
                // Pattern match
                if self.automaton.find(code).is_some() {
                    // Entropy check (high entropy = likely real secret)
                    if code.len() > 8 && self.shannon_entropy(code) > self.entropy_threshold {
                        findings.push(Finding {
                            rule_id: "SENT-001".to_string(),
                            message: format!("High-entropy secret detected in '{}'", node.name),
                            severity: Severity::Critical,
                            line: node.line_start,
                            column: 0,
                            file: Some(node.name.clone()),
                            cwe_id: Some("CWE-798".to_string()),
                            confidence: 0.95,
                        });
                    }
                }
            }
        }

        Ok(findings)
    }
}
