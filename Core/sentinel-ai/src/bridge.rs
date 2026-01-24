use std::time::Duration;
use std::sync::Arc;
use tokio::sync::Semaphore;
use futures::stream::{FuturesUnordered, StreamExt};
use serde::{Serialize, Deserialize};
use anyhow::{Result, anyhow, Context};
use tracing::{span, Level};
use tracing_futures::Instrument;

pub mod sentinel {
    tonic::include_proto!("sentinel.v1");
}

use sentinel::{
    inference_service_client::InferenceServiceClient,
    InferenceRequest as ProtoInferenceRequest,
    FeatureVector,
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct InferenceRequest {
    pub subgraph_id: String,
    pub nodes: Vec<u64>,
    pub features: Vec<Vec<f32>>,
    pub pqc_signature: Vec<u8>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct InferenceResponse {
    pub embedding: Vec<f32>,
    pub confidence: f32,
    pub logic_clone_detected: bool,
    pub latency_ms: u64,
}

// =============================================================================
// SECTION 2: CONNECTION POOL
// =============================================================================
pub struct InferenceBridge {
    endpoint: String,
    timeout: Duration,
    max_batch_concurrency: usize,
}

impl InferenceBridge {
    pub fn new(endpoint: &str, _model_name: &str) -> Self {
        let fmt_endpoint = if endpoint.starts_with("http") {
            endpoint.to_string()
        } else {
            format!("http://{}", endpoint)
        };

        Self {
            endpoint: fmt_endpoint,
            timeout: Duration::from_secs(30),
            max_batch_concurrency: 16,
        }
    }

    /// Dispatches a single inference request with retry logic.
    pub async fn infer_logic_embedding(&self, request: InferenceRequest) -> anyhow::Result<InferenceResponse> {
        let span = span!(Level::INFO, "InferenceRequest", id = %request.subgraph_id);
        
        async move {
            // Protocol Validation
            if request.nodes.is_empty() {
                return Err(anyhow!("Protocol Violation: Empty node set."));
            }

            let mut client = InferenceServiceClient::connect(self.endpoint.clone())
                .await
                .context("Failed to connect to Sovereign Inference Server")?;

            let proto_req = ProtoInferenceRequest {
                subgraph_id: request.subgraph_id,
                nodes: request.nodes,
                features: request.features.into_iter().map(|f| FeatureVector { values: f }).collect(),
                pqc_signature: request.pqc_signature,
            };

            let start = std::time::Instant::now();
            let response = client.infer_logic(proto_req)
                .await
                .map_err(|e| anyhow!("gRPC Error: {}", e))?
                .into_inner();

            Ok(InferenceResponse {
                embedding: response.embedding,
                confidence: response.confidence,
                logic_clone_detected: response.logic_clone_detected,
                latency_ms: start.elapsed().as_millis() as u64,
            })
        }
        .instrument(span)
        .await
    }

    /// Concurrent Batch Inference
    pub async fn infer_batch(&self, requests: Vec<InferenceRequest>) -> anyhow::Result<Vec<InferenceResponse>> {
        if requests.is_empty() { return Ok(Vec::new()); }

        let semaphore = Arc::new(Semaphore::new(self.max_batch_concurrency));
        let mut futures = FuturesUnordered::new();
        
        let arc_self = Arc::new(self.clone());
        for req in requests {
            let permit = semaphore.clone().acquire_owned().await
                .map_err(|e| anyhow!("Semaphore Error: {}", e))?;
            let bridge = arc_self.clone();
            
            futures.push(tokio::spawn(async move {
                let result = bridge.infer_logic_embedding(req).await;
                drop(permit);
                result
            }));
        }

        let mut results = Vec::new();
        while let Some(result) = futures.next().await {
            match result {
                Ok(Ok(res)) => results.push(res),
                Ok(Err(e)) => return Err(e),
                Err(e) => return Err(anyhow!(e)),
            }
        }
        Ok(results)
    }
}

impl Clone for InferenceBridge {
    fn clone(&self) -> Self {
        Self {
            endpoint: self.endpoint.clone(),
            timeout: self.timeout,
            max_batch_concurrency: self.max_batch_concurrency,
        }
    }
}
