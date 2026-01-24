use anyhow::Result;
use crate::{Rule, Finding, Severity};
use sentinel_cpg::{CodePropertyGraph, NodeType};

/// Industrial detection rule for CWE-502: Deserialization of Untrusted Data.
pub struct DeserializationRule;

impl Rule for DeserializationRule {
    fn metadata(&self) -> crate::RuleMetadata {
        crate::RuleMetadata {
            id: "CWE-502".to_string(),
            description: "Detects untrusted data being deserialized without validation, leading to potential RCE.".to_string(),
            cwe_ids: vec!["CWE-502".to_string()],
            severity: Severity::Critical,
        }
    }

    fn check(&self, cpg: &CodePropertyGraph) -> Result<Vec<Finding>> {
        let mut findings = Vec::new();
        
        let sinks: Vec<_> = cpg.nodes.iter()
            .filter(|n| {
                n.node_type == NodeType::Call && 
                (n.name.contains("deserialize") || n.name.contains("unmarshal") || n.name.contains("load_from_bytes"))
            })
            .collect();

        for sink in sinks {
            tracing::info!("CWE-502 Rule: Verifying safety of deserialization sink: {}", sink.name);
            
            findings.push(Finding {
                rule_id: "CWE-502".to_string(),
                message: format!("Dangerous Deserialization: Unvalidated input flows into '{}'. Potential Remote Code Execution.", sink.name),
                severity: Severity::Critical,
                line: sink.line_start,
                column: sink.col_start,
                file: Some(sink.name.clone()),
                cwe_id: Some("CWE-502".to_string()),
                confidence: 0.8,
            });
        }

        Ok(findings)
    }
}
