use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct TenantId(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobMetadata {
    pub job_id: String,
    pub tenant_id: TenantId,
    pub repo_url: String,
    pub created_at: std::time::SystemTime,
}

pub mod telemetry;

pub mod types {
    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Pulse {
        pub node_id: String,
        pub load: f32,
        pub active_scans: usize,
    }
}

/// A highly-optimized Bloom Filter configuration for global symbol tables.
pub const BLOOM_FILTER_SIZE: usize = 100 * 1024 * 1024; // 100MB
