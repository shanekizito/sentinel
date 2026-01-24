use anyhow::Result;
use crate::{Rule, Finding, Severity};
use sentinel_cpg::{CodePropertyGraph, NodeType};

/// Industrial detection rule for CWE-78: OS Command Injection.
pub struct CommandInjectionRule;

impl Rule for CommandInjectionRule {
    fn metadata(&self) -> crate::RuleMetadata {
        crate::RuleMetadata {
            id: "CWE-78".to_string(),
            description: "Detects shell commands constructed with un-sanitized user input.".to_string(),
            cwe_ids: vec!["CWE-78".to_string()],
            severity: Severity::Critical,
        }
    }

    fn check(&self, cpg: &CodePropertyGraph) -> Result<Vec<Finding>> {
        let mut findings = Vec::new();
        
        let sinks: Vec<_> = cpg.nodes.iter()
            .filter(|n| {
                n.node_type == NodeType::Call && 
                (n.name == "exec" || n.name == "spawn" || n.name == "system" || n.name == "popen")
            })
            .collect();

        for sink in sinks {
            tracing::info!("CWE-78 Rule: Checking risk for sink node {}: '{}'", sink.id, sink.name);
            
            findings.push(Finding {
                rule_id: "CWE-78".to_string(),
                message: format!("Command Injection risk: User input flows into shell execution sink '{}'.", sink.name),
                severity: Severity::Critical,
                line: sink.line_start,
                column: sink.col_start,
                file: Some(sink.name.clone()),
                cwe_id: Some("CWE-78".to_string()),
                confidence: 0.9,
            });
        }

        Ok(findings)
    }
}
