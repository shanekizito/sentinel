use sentinel_cpg::CodePropertyGraph;

pub struct SmtTranslator {
    ctx_name: String,
}

impl SmtTranslator {
    pub fn new(name: &str) -> Self {
        Self { ctx_name: name.to_string() }
    }

    /// Translates a suspicious data flow path into a Z3-compatible SMT-LIB script.
    /// The script attempts to find if there's ANY input that can reach a sink without sanitization.
    pub fn translate_path_to_smt(&self, path: &[u64], cpg: &CodePropertyGraph) -> String {
        let mut smt = String::new();
        smt.push_str("(set-logic QF_BV)\n"); // Bit-vector logic for precise modeling
        smt.push_str(&format!("; Context: {}\n", self.ctx_name));

        for &node_id in path {
            if let Some(node) = cpg.nodes.iter().find(|n| n.id == node_id) {
                // Declare each node in the path as a 32-bit BitVector
                smt.push_str(&format!("(declare-fun node_{} () (_ BitVec 32))\n", node.id));
                
                // If it's a literal or constant, assert its value if known
                if let Some(val) = node.metadata.get("value") {
                    if let Ok(int_val) = val.parse::<i64>() {
                        smt.push_str(&format!("(assert (= node_{} #x{:08x}))\n", node.id, int_val));
                    }
                }
            }
        }

        // Add constraints for each hop in the path
        for i in 0..path.len() - 1 {
            smt.push_str(&format!("(assert (= node_{} node_{})) ; Data flow propagation\n", path[i+1], path[i]));
        }

        // Final Invariant: The value at the sink MUST be 'Safe'
        // In this model, we'll assume #x00000000 is malicious (Exploit payload)
        let sink_id = path.last().unwrap();
        smt.push_str(&format!("; Safety Property: Sink {} must not be exploitable\n", sink_id));
        smt.push_str(&format!("(assert (= node_{} #x00000000))\n", sink_id));

        smt.push_str("(check-sat)\n");
        smt.push_str("(get-model)\n");

        smt
    }
}
