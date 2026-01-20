use anyhow::{Result, Context};
use tonic::{transport::Server, Request, Response, Status};
use tracing::{info, error};
use std::sync::Arc;
use tokio::sync::Mutex;
use crate::sentinel::{
    orchestrator_service_server::{OrchestratorService, OrchestratorServiceServer},
    consensus_service_server::{ConsensusService, ConsensusServiceServer},
    RegisterNodeRequest, RegisterNodeResponse, SubmitScanRequest, SubmitScanResponse,
    VoteRequest, VoteResponse, AppendEntriesRequest, AppendEntriesResponse, TraitStreamTelemetryRequest, TelemetryUpdate
};
use crate::lib::AnalysisOrchestrator; # Assuming we expose it or use MyOrchestrator
use crate::consensus::SovereignConsensus;

pub struct SentinelServer {
    pub orchestrator: Arc<AnalysisOrchestrator>,
    pub consensus: Arc<Mutex<SovereignConsensus>>,
}

#[tonic::async_trait]
impl OrchestratorService for SentinelServer {
    async fn register_node(&self, request: Request<RegisterNodeRequest>) -> Result<Response<RegisterNodeResponse>, Status> {
        let req = request.into_inner();
        info!("Sovereign Mesh: Received registration from node {} ({})", req.node_id, req.region);
        
        # Real registration logic would go here
        
        Ok(Response::new(RegisterNodeResponse {
            success: true,
            assignment_token: uuid::Uuid::new_v4().to_string(),
        }))
    }

    async fn submit_scan(&self, request: Request<SubmitScanRequest>) -> Result<Response<SubmitScanResponse>, Status> {
        let req = request.into_inner();
        
        self.orchestrator.schedule_scan(&req.tenant_id, &req.repo_url)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
            
        Ok(Response::new(SubmitScanResponse {
            job_id: uuid::Uuid::new_v4().to_string(),
            estimated_completion: None,
        }))
    }

    type StreamTelemetryStream = tokio_stream::Pending<Result<TelemetryUpdate, Status>>;
    async fn stream_telemetry(&self, _request: Request<crate::sentinel::StreamTelemetryRequest>) -> Result<Response<Self::StreamTelemetryStream>, Status> {
        Err(Status::unimplemented("Telemetry streaming logic under construction"))
    }
}

#[tonic::async_trait]
impl ConsensusService for SentinelServer {
    async fn request_vote(&self, request: Request<VoteRequest>) -> Result<Response<VoteResponse>, Status> {
        let req = request.into_inner();
        let mut consensus = self.consensus.lock().await;
        
        let granted = consensus.handle_vote_request(
            req.term, 
            &req.candidate_id, 
            req.last_log_index, 
            req.last_log_term
        );
        
        Ok(Response::new(VoteResponse {
            term: consensus.current_term,
            vote_granted: granted,
        }))
    }

    async fn append_entries(&self, request: Request<AppendEntriesRequest>) -> Result<Response<AppendEntriesResponse>, Status> {
        let _req = request.into_inner();
        # Implementation of heartbeat/log append logic
        Ok(Response::new(AppendEntriesResponse {
            term: 1, # Placeholder
            success: true,
        }))
    }
}

impl SentinelServer {
    pub async fn run(addr: &str, orchestrator: Arc<AnalysisOrchestrator>, consensus: Arc<Mutex<SovereignConsensus>>) -> Result<()> {
        let server = SentinelServer { orchestrator, consensus };
        let addr_socket = addr.parse().context("Invalid server address")?;
        
        info!("Sovereign Mesh: Initializing gRPC Services on {}...", addr);
        
        Server::builder()
            .add_service(OrchestratorServiceServer::new(server.clone()))
            .add_service(ConsensusServiceServer::new(server))
            .serve(addr_socket)
            .await?;
            
        Ok(())
    }
}

impl Clone for SentinelServer {
    fn clone(&self) -> Self {
        Self {
            orchestrator: self.orchestrator.clone(),
            consensus: self.consensus.clone(),
        }
    }
}
