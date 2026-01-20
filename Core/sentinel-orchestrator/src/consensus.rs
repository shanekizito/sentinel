use anyhow::{Result, anyhow};
use tracing::{info, warn, error, span, Level};
use std::time::{Duration, Instant};
use std::sync::Arc;
use rand::Rng;
use futures::stream::{FuturesUnordered, StreamExt};
use tokio::sync::RwLock;
use crate::sentinel::{
    consensus_service_client::ConsensusServiceClient,
    VoteRequest, VoteResponse, AppendEntriesRequest, AppendEntriesResponse, RaftLogEntry
};
use tonic::transport::Channel;

// =============================================================================
// SECTION 1: NODE STATE
// =============================================================================

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum NodeState {
    Follower,
    PreCandidate,
    Candidate,
    Leader,
    Zombie,
}

#[derive(Debug, Clone)]
pub struct LogEntry {
    pub term: u64,
    pub index: u64,
    pub command: Vec<u8>,
}

// =============================================================================
// SECTION 2: OMEGA CONSENSUS ENGINE
// =============================================================================

pub struct OmegaConsensus {
    pub node_id: String,
    pub region: String,
    pub state: NodeState,
    pub current_term: u64,
    pub voted_for: Option<String>,
    pub log: Vec<LogEntry>,
    pub commit_index: u64,
    pub last_applied: u64,
    
    // Leader State
    pub next_index: std::collections::HashMap<String, u64>,
    pub match_index: std::collections::HashMap<String, u64>,
    
    // Timers
    pub election_timeout: Duration,
    pub last_heartbeat: Instant,
    pub next_election: Instant,
    
    // Peers
    pub peers: Vec<String>,
    
    // Stats
    pub elections_won: u64,
    pub heartbeats_sent: u64,
}

impl OmegaConsensus {
    pub fn new(node_id: &str, region: &str, peers: Vec<String>) -> Self {
        let mut rng = rand::thread_rng();
        let timeout = Duration::from_millis(rng.gen_range(1500..3000));
        
        Self {
            node_id: node_id.to_string(),
            region: region.to_string(),
            state: NodeState::Follower,
            current_term: 0,
            voted_for: None,
            log: Vec::new(),
            commit_index: 0,
            last_applied: 0,
            next_index: std::collections::HashMap::new(),
            match_index: std::collections::HashMap::new(),
            election_timeout: timeout,
            last_heartbeat: Instant::now(),
            next_election: Instant::now() + timeout,
            peers,
            elections_won: 0,
            heartbeats_sent: 0,
        }
    }

    pub async fn tick(&mut self) -> Result<()> {
        match self.state {
            NodeState::Follower => {
                if Instant::now() > self.next_election {
                    self.start_pre_vote().await?;
                }
            }
            NodeState::PreCandidate => {
                // Pre-vote succeeded, start real election
                self.initiate_election().await?;
            }
            NodeState::Candidate => {
                if Instant::now() > self.next_election {
                    self.initiate_election().await?;
                }
            }
            NodeState::Leader => {
                self.broadcast_heartbeats().await?;
            }
            NodeState::Zombie => {
                self.state = NodeState::Follower;
                self.reset_election_timer();
            }
        }
        Ok(())
    }

    /// Pre-Vote Protocol: Prevents disruption from partitioned nodes.
    async fn start_pre_vote(&mut self) -> Result<()> {
        info!("Omega Consensus: [{} - Term {}] Starting Pre-Vote...", self.node_id, self.current_term);
        
        let request = VoteRequest {
            term: self.current_term + 1, // Hypothetical next term
            candidate_id: self.node_id.clone(),
            last_log_index: self.log.len() as u64,
            last_log_term: self.log.last().map_or(0, |e| e.term),
        };

        let votes = self.parallel_vote_request(request).await;
        let quorum = (self.peers.len() + 1) / 2 + 1;

        if votes >= quorum {
            self.state = NodeState::PreCandidate;
        } else {
            self.reset_election_timer();
        }

        Ok(())
    }

    async fn initiate_election(&mut self) -> Result<()> {
        self.current_term += 1;
        self.state = NodeState::Candidate;
        self.voted_for = Some(self.node_id.clone());
        self.reset_election_timer();
        
        let _span = span!(Level::INFO, "Election", term = self.current_term).entered();
        info!("Omega Consensus: [{} - Term {}] Initiating Election...", self.node_id, self.current_term);
        
        let request = VoteRequest {
            term: self.current_term,
            candidate_id: self.node_id.clone(),
            last_log_index: self.log.len() as u64,
            last_log_term: self.log.last().map_or(0, |e| e.term),
        };

        let votes = self.parallel_vote_request(request).await;
        let quorum = (self.peers.len() + 1) / 2 + 1;

        if votes >= quorum {
            info!("Omega Consensus: [{} - Term {}] ELECTION WON with {} votes!", self.node_id, self.current_term, votes);
            self.state = NodeState::Leader;
            self.elections_won += 1;
            self.initialize_leader_state();
            self.broadcast_heartbeats().await?;
        }

        Ok(())
    }

    /// Parallel vote requests using FuturesUnordered.
    async fn parallel_vote_request(&self, request: VoteRequest) -> usize {
        let mut futures = FuturesUnordered::new();
        
        for peer in &self.peers {
            let addr = peer.clone();
            let req = request.clone();
            futures.push(async move {
                match ConsensusServiceClient::connect(addr).await {
                    Ok(mut client) => {
                        match client.request_vote(req).await {
                            Ok(resp) => resp.into_inner().vote_granted,
                            Err(_) => false,
                        }
                    }
                    Err(_) => false,
                }
            });
        }

        let mut votes = 1; // Self vote
        while let Some(granted) = futures.next().await {
            if granted {
                votes += 1;
            }
        }
        votes
    }

    async fn broadcast_heartbeats(&mut self) -> Result<()> {
        if Instant::now().duration_since(self.last_heartbeat) < Duration::from_millis(150) {
            return Ok(());
        }
        
        self.last_heartbeat = Instant::now();
        self.heartbeats_sent += 1;
        
        let mut futures = FuturesUnordered::new();
        
        for peer in &self.peers {
            let addr = peer.clone();
            let request = AppendEntriesRequest {
                term: self.current_term,
                leader_id: self.node_id.clone(),
                prev_log_index: self.log.len() as u64,
                prev_log_term: self.log.last().map_or(0, |e| e.term),
                entries: vec![],
                leader_commit: self.commit_index,
            };
            
            futures.push(async move {
                if let Ok(mut client) = ConsensusServiceClient::connect(addr).await {
                    let _ = client.append_entries(request).await;
                }
            });
        }

        while futures.next().await.is_some() {}
        Ok(())
    }

    fn initialize_leader_state(&mut self) {
        let next = self.log.len() as u64 + 1;
        for peer in &self.peers {
            self.next_index.insert(peer.clone(), next);
            self.match_index.insert(peer.clone(), 0);
        }
    }

    fn reset_election_timer(&mut self) {
        let mut rng = rand::thread_rng();
        self.next_election = Instant::now() + Duration::from_millis(rng.gen_range(1500..3000));
    }

    pub fn stats(&self) -> (u64, u64, NodeState) {
        (self.elections_won, self.heartbeats_sent, self.state)
    }
}
