pub mod sentinel {
    tonic::include_proto!("sentinel.v1");
}

pub mod scheduler;
pub mod mesh;
pub mod server;
pub mod consensus;

use anyhow::Result;
use tracing::info;

use crate::scheduler::{JobScheduler, Job, JobPriority};
use chrono::Utc;

pub struct AnalysisOrchestrator {
    pub region: String,
    pub scheduler: JobScheduler,
}

impl AnalysisOrchestrator {
    pub fn new(region: &str) -> Self {
        Self { 
            region: region.to_string(),
            scheduler: JobScheduler::new(),
        }
    }

    /// Schedules a planetary-scale scan job across available regional ganglia
    pub async fn schedule_scan(&self, tenant_id: &str, repo_url: &str) -> Result<()> {
        info!("Sovereign Orchestrator: Dispatching scan request for {}...", tenant_id);
        
        let job = Job {
            id: uuid::Uuid::new_v4().to_string(),
            priority: JobPriority::High,
            tenant_id: tenant_id.to_string(),
            created_at: Utc::now(),
            payload: repo_url.to_string(),
        };

        self.scheduler.schedule_job(job)?;
        
        info!("Sovereign Orchestrator: Job successfully queued in regional mesh.");
        Ok(())
    }
}
