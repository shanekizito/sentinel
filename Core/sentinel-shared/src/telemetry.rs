use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryPacket {
    pub job_id: String,
    pub node_id: String,
    pub region: String,
    pub stats: AnalysisStats,
    pub heartbeat_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisStats {
    pub nodes_analyzed: u64,
    pub edges_traversed: u64,
    pub proofs_resolved: u32,
    pub latency_ms: f32,
}

impl TelemetryPacket {
    pub fn new(job_id: &str, node_id: &str) -> Self {
        Self {
            job_id: job_id.to_string(),
            node_id: node_id.to_string(),
            region: "unassigned".to_string(),
            stats: AnalysisStats {
                nodes_analyzed: 0,
                edges_traversed: 0,
                proofs_resolved: 0,
                latency_ms: 0.0,
            },
            heartbeat_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }
}
