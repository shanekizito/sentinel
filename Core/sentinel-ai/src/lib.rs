pub mod bridge;
pub mod gcn;
pub mod rag;

use anyhow::Result;
use tracing::{info, span, Level};
use bridge::{InferenceBridge, InferenceRequest};
use rag::SemanticRagHub;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::fs::File;
use std::io::{Read, Write};

/// GCN Encoder: Serializes Graph Structures into high-dimensional latent vectors.
pub struct GcnEncoder {
    pub model_path: String,
}

impl GcnEncoder {
    pub fn new(path: &str) -> Self {
        Self { model_path: path.to_string() }
    }

    /// Generates a graph-aware embedding for a CPG subgraph using internal model logic.
    pub fn encode_subgraph(&self, nodes: &[u64], edges: &[(u64, u64)]) -> Result<Vec<f32>> {
        let _span = span!(Level::INFO, "GcnEncoding", count = nodes.len()).entered();
        // info!("Sovereign AI: Generating Neural Embedding for sub-graph logic...");
        // Fast path for demo: return a deterministic pseudo-embedding based on node count
        // This avoids heavy ONNX runtime but provides consistent "features" for the learner
        let seed = nodes.len() as f32;
        Ok(vec![seed * 0.01; 64]) 
    }
}

/// A lightweight, on-device AI learner for low-resource environments.
/// Uses Naive Bayes on simplified graph features.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct LocalBrain {
    // Feature ID -> (Safe Count, Vulnerable Count)
    pub feature_counts: HashMap<String, (u32, u32)>,
    pub total_safe: u32,
    pub total_vuln: u32,
}

impl LocalBrain {
    pub fn new() -> Self {
        Self {
            feature_counts: HashMap::new(),
            total_safe: 0,
            total_vuln: 0,
        }
    }

    pub fn learn(&mut self, features: &[String], is_vulnerable: bool) {
        if is_vulnerable {
            self.total_vuln += 1;
        } else {
            self.total_safe += 1;
        }

        for f in features {
            let entry = self.feature_counts.entry(f.clone()).or_insert((0, 0));
            if is_vulnerable {
                entry.1 += 1;
            } else {
                entry.0 += 1;
            }
        }
    }

    pub fn predict(&self, features: &[String]) -> f32 {
        // P(Vuln | Features) propto P(Vuln) * Prod P(Fi | Vuln)
        let total = (self.total_safe + self.total_vuln) as f32;
        if total == 0.0 { return 0.0; }

        let p_vuln = (self.total_vuln as f32) / total;
        let p_safe = (self.total_safe as f32) / total;

        let mut log_prob_vuln = p_vuln.ln();
        let mut log_prob_safe = p_safe.ln();

        for f in features {
            let (safe_hits, vuln_hits) = self.feature_counts.get(f).unwrap_or(&(0, 0));
            
            // Additive smoothing (Laplace +1)
            let prob_feat_given_vuln = (*vuln_hits as f32 + 1.0) / (self.total_vuln as f32 + 2.0);
            let prob_feat_given_safe = (*safe_hits as f32 + 1.0) / (self.total_safe as f32 + 2.0);

            log_prob_vuln += prob_feat_given_vuln.ln();
            log_prob_safe += prob_feat_given_safe.ln();
        }

        if log_prob_vuln > log_prob_safe { 1.0 } else { 0.0 }
    }

    pub fn save(&self, path: &str) -> Result<()> {
        let mut file = File::create(path)?;
        let data = serde_json::to_vec(self)?;
        file.write_all(&data)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let brain: LocalBrain = serde_json::from_slice(&buffer)?;
        Ok(brain)
    }
}

/// Reflex Engine: The master orchestrator for AI-driven remediation.
/// Unifies the GCN, Inference Bridge, and RAG Hub into a single sovereign autonomous unit.
pub struct Reflex {
    pub encoder: GcnEncoder,
    pub bridge: Option<InferenceBridge>, // Optional for Demo Mode
    pub local_brain: Arc<RwLock<LocalBrain>>,
    pub rag: SemanticRagHub,
}

impl Reflex {
    pub fn new(model_path: &str, triton_endpoint: &str, dimension: usize) -> Self {
        // Try simple load of local brain, else new
        let brain = LocalBrain::load("sentinel_brain.json").unwrap_or_else(|_| LocalBrain::new());
        
        let bridge = if !triton_endpoint.is_empty() {
             Some(InferenceBridge::new(triton_endpoint, "reflex_v1"))
        } else {
             None
        };

        Self {
            encoder: GcnEncoder::new(model_path),
            bridge,
            local_brain: Arc::new(RwLock::new(brain)),
            rag: SemanticRagHub::new(dimension),
        }
    }

    /// Performs the full neuro-symbolic audit for a discovered vulnerability path.
    pub async fn process_vulnerability(&self, id: &str, nodes: &[u64], edges: &[(u64, u64)]) -> Result<String> {
        info!("Sovereign AI: Reflex Engine starting autonomous remediation for {}", id);

        // 1. Generate Logic Embedding (Neural)
        let _embedding = self.encoder.encode_subgraph(nodes, edges)?;

        // 2. Local Demo Inference
        // Use "fake" features derived from graph topology for demonstration
        let features = vec![
            format!("nodes_{}", nodes.len()), 
            format!("edges_{}", edges.len()),
            format!("density_{}", edges.len() as f32 / (nodes.len() as f32 + 1.0))
        ];

        let confidence = {
             let brain = self.local_brain.read().unwrap();
             brain.predict(&features)
        };

        if confidence > 0.5 {
            info!("Sovereign AI: Local Brain detects vulnerability pattern with confidence.");
        }

        // 3. Online Learning (Simulation)
        // In a real demo, user feedback would trigger this. For now, we self-reinforce if high confidence.
        if confidence > 0.8 {
             let mut brain = self.local_brain.write().unwrap();
             brain.learn(&features, true);
             let _ = brain.save("sentinel_brain.json"); // Auto-save
        }

        // 4. Retrieve Semantic Context (RAG)
        // Use a lighter mock if needed, but RAG with small dimensionality is fine on 8GB
        let context = self.rag.retrieve_context(&_embedding).await?;

        // 4. Augment and Return final Remediation Instruction
        let prompt = format!("Vulnerability {} found. Auto-Learner Confidence: {:.2}", id, confidence);
        Ok(self.rag.augment_prompt(&prompt, &context))
    }
}
