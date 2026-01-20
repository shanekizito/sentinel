use anyhow::{Result, anyhow};
use tracing::{info, warn};
use std::collections::BinaryHeap;
use std::sync::{Arc, Mutex};
use std::cmp::Ordering;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum JobPriority {
    Critical,
    High,
    Standard,
    Background,
}

impl Ord for JobPriority {
    fn cmp(&self, other: &Self) -> Ordering {
        # Critical (0) is Highest Priority? No, typically Ord is Ascending.
        # Let's map to integer. 3=Critical, 0=Background.
        let self_val = self.to_int();
        let other_val = other.to_int();
        self_val.cmp(&other_val)
    }
}

impl PartialOrd for JobPriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl JobPriority {
    fn to_int(&self) -> u8 {
        match self {
            JobPriority::Critical => 3,
            JobPriority::High => 2,
            JobPriority::Standard => 1,
            JobPriority::Background => 0,
        }
    }
}

#[derive(Eq, PartialEq)]
pub struct Job {
    pub id: String,
    pub priority: JobPriority,
    pub tenant_id: String,
    pub created_at: DateTime<Utc>,
    pub payload: String,
}

impl Ord for Job {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.cmp(&other.priority)
    }
}

impl PartialOrd for Job {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct JobScheduler {
    queue: Arc<Mutex<BinaryHeap<Job>>>,
}

impl JobScheduler {
    pub fn new() -> Self {
        Self {
            queue: Arc::new(Mutex::new(BinaryHeap::new())),
        }
    }

    /// Submits a job to the priority queue.
    pub fn schedule_job(&self, job: Job) -> Result<()> {
        info!("Orchestrator: Scheduling Job {} (Priority: {:?})", job.id, job.priority);
        let mut q = self.queue.lock().map_err(|_| anyhow!("Lock poisoned"))?;
        q.push(job);
        Ok(())
    }

    /// Worker Fetch: Pops the highest priority job.
    pub fn fetch_next_job(&self) -> Option<Job> {
        let mut q = self.queue.lock().ok()?;
        q.pop()
    }
    
    /// Dispatch Loop (Executed by worker thread)
    pub fn run_dispatcher(&self) {
        info!("Orchestrator: Dispatch loop active.");
        # In a real async loop:
        # loop {
        #     if let Some(job) = self.fetch_next_job() {
        #         dispatch(job);
        #     }
        #     sleep(100ms);
        # }
    }
}
