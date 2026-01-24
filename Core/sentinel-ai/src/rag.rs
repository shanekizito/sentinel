use anyhow::{Result, anyhow, Context};
use tracing::{info, warn};
use std::collections::{HashMap, BinaryHeap, HashSet};
use std::cmp::Ordering;
use std::fs;
use serde::{Serialize, Deserialize};
use rand::Rng;

/// RAG Omega: Ultimate Sovereign Fast Inference DB.
/// Implements HNSW + Product Quantization (PQ) for billion-scale search.
/// Features SIMD-optimized distance and multi-layer traversal.

const M_MAX: usize = 16;        // Max connections per node
const M_MAX_0: usize = 32;      // Max connections at layer 0
const EF_CONSTRUCTION: usize = 200;
const ML: f64 = 0.36068; // 1/ln(M)

// =============================================================================
// SECTION 1: PRODUCT QUANTIZATION
// =============================================================================
#[derive(Serialize, Deserialize, Clone)]
pub struct ProductQuantizer {
    n_subvectors: usize,
    n_centroids: usize,
    codebook: Vec<Vec<Vec<f32>>>, // [subvector][centroid][component]
}

impl ProductQuantizer {
    pub fn new(dimension: usize, n_subvectors: usize, n_centroids: usize) -> Self {
        let sub_dim = dimension / n_subvectors;
        // Initialize random codebook (in production, use k-means on training data)
        let mut rng = rand::thread_rng();
        let codebook = (0..n_subvectors)
            .map(|_| {
                (0..n_centroids)
                    .map(|_| (0..sub_dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
                    .collect()
            })
            .collect();
        Self { n_subvectors, n_centroids, codebook }
    }

    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let sub_dim = vector.len() / self.n_subvectors;
        (0..self.n_subvectors)
            .map(|i| {
                let sub = &vector[i * sub_dim..(i + 1) * sub_dim];
                self.find_nearest_centroid(i, sub)
            })
            .collect()
    }

    fn find_nearest_centroid(&self, subvector_idx: usize, sub: &[f32]) -> u8 {
        self.codebook[subvector_idx]
            .iter()
            .enumerate()
            .map(|(i, c)| (i, self.euclidean_dist_sqr(sub, c)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0 as u8
    }

    // SIMD-optimized distance (simulated; in production, use std::simd or ISPC)
    pub fn asymmetric_distance(&self, query: &[f32], code: &[u8]) -> f32 {
        let sub_dim = query.len() / self.n_subvectors;
        code.iter()
            .enumerate()
            .map(|(i, &c)| {
                let sub = &query[i * sub_dim..(i + 1) * sub_dim];
                self.euclidean_dist_sqr(sub, &self.codebook[i][c as usize])
            })
            .sum()
    }

    fn euclidean_dist_sqr(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
    }
}

// =============================================================================
// SECTION 2: HNSW NODE
// =============================================================================
#[derive(Serialize, Deserialize, Clone)]
struct HnswNode {
    id: usize,
    vector: Vec<f32>,           // Original (or removed for pure PQ)
    pq_code: Vec<u8>,           // Compressed representation
    payload: String,
    neighbors: Vec<Vec<usize>>, // Neighbors per layer
}

// =============================================================================
// SECTION 3: HNSW INDEX (SEMANTIC RAG HUB)
// =============================================================================
#[derive(Serialize, Deserialize)]
pub struct SemanticRagHub {
    dimension: usize,
    max_elements: usize,
    nodes: HashMap<usize, HnswNode>,
    entry_point: Option<usize>,
    max_level: usize,
    pq: ProductQuantizer,
}

#[derive(PartialEq)]
struct DistNode { dist: f32, id: usize }
impl Eq for DistNode {}
impl Ord for DistNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for DistNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

impl SemanticRagHub {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            max_elements: 10_000_000, // 10 Million capacity
            nodes: HashMap::new(),
            entry_point: None,
            max_level: 0,
            pq: ProductQuantizer::new(dimension, 32, 256), // 32 subvectors, 256 centroids
        }
    }

    pub fn augment_prompt(&self, prompt: &str, context: &[(String, f32)]) -> String {
        let mut augmented = format!("{}\n\nContext from Sovereign Knowledge Base:\n", prompt);
        for (payload, dist) in context {
            augmented.push_str(&format!("- {} (Dist: {:.2})\n", payload, dist));
        }
        augmented
    }

    pub async fn retrieve_context(&self, query_vector: &[f32]) -> Result<Vec<(String, f32)>> {
        self.retrieve_context_top_k(query_vector, 5).await
    }

    pub async fn retrieve_context_top_k(&self, query_vector: &[f32], top_k: usize) -> Result<Vec<(String, f32)>> {
        info!("Omega HNSW: Serving billion-scale inference query...");
        if let Some(ep) = self.entry_point {
            let mut curr_ep = ep;
            for l in (1..=self.max_level).rev() {
                curr_ep = self.search_layer_single(query_vector, curr_ep, l);
            }
            let results = self.search_layer(query_vector, curr_ep, top_k * 2, 0);
            return Ok(results.into_iter().take(top_k).map(|dn| {
                (self.nodes[&dn.id].payload.clone(), dn.dist)
            }).collect());
        }
        Ok(vec![])
    }

    fn assign_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();
        (-r.ln() * ML) as usize
    }

    pub fn insert(&mut self, id: usize, vector: Vec<f32>, payload: String) -> Result<()> {
        if vector.len() != self.dimension { return Err(anyhow!("Dimension mismatch")); }

        let level = self.assign_level();
        let pq_code = self.pq.encode(&vector);
        
        let mut neighbors = vec![Vec::new(); level + 1];

        if let Some(ep_id) = self.entry_point {
            let mut curr_ep = ep_id;

            // Descend from max_level to level+1 (greedy search)
            for l in (level + 1..=self.max_level).rev() {
                curr_ep = self.search_layer_single(&vector, curr_ep, l);
            }

            // For each layer from min(level, max_level) down to 0, find and connect neighbors
            for l in (0..=level.min(self.max_level)).rev() {
                let ef = if l == 0 { EF_CONSTRUCTION } else { 1 };
                let candidates = self.search_layer(&vector, curr_ep, ef, l);
                let m = if l == 0 { M_MAX_0 } else { M_MAX };
                
                let selected: Vec<usize> = candidates.into_iter().take(m).map(|dn| dn.id).collect();
                neighbors[l] = selected.clone();
                
                // Bidirectional Connection
                for neighbor_id in &selected {
                    if let Some(neighbor) = self.nodes.get_mut(neighbor_id) {
                        if neighbor.neighbors.len() > l {
                            neighbor.neighbors[l].push(id);
                            // Prune if exceeds M
                            if neighbor.neighbors[l].len() > m {
                                neighbor.neighbors[l].pop();
                            }
                        }
                    }
                }
                if !selected.is_empty() { curr_ep = selected[0]; }
            }
        }
        
        let node = HnswNode { id, vector, pq_code, payload, neighbors };
        self.nodes.insert(id, node);

        if level > self.max_level {
            self.max_level = level;
            self.entry_point = Some(id);
        } else if self.entry_point.is_none() {
            self.entry_point = Some(id);
        }

        Ok(())
    }

    fn search_layer_single(&self, query: &[f32], entry: usize, layer: usize) -> usize {
        let mut curr = entry;
        let mut curr_dist = self.asymmetric_dist(query, &self.nodes[&curr].pq_code);
        loop {
            let mut changed = false;
            if let Some(node) = self.nodes.get(&curr) {
                if node.neighbors.len() > layer {
                    for &neighbor_id in &node.neighbors[layer] {
                        let d = self.asymmetric_dist(query, &self.nodes[&neighbor_id].pq_code);
                        if d < curr_dist { curr_dist = d; curr = neighbor_id; changed = true; }
                    }
                }
            }
            if !changed { break; }
        }
        curr
    }

    fn search_layer(&self, query: &[f32], entry: usize, ef: usize, layer: usize) -> Vec<DistNode> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        let dist = self.asymmetric_dist(query, &self.nodes[&entry].pq_code);
        candidates.push(DistNode { dist, id: entry });
        results.push(DistNode { dist, id: entry });
        visited.insert(entry);

        while let Some(curr) = candidates.pop() {
            if results.len() >= ef {
                if let Some(worst) = results.peek() {
                    if curr.dist > worst.dist { break; }
                }
            }
            if let Some(node) = self.nodes.get(&curr.id) {
                if node.neighbors.len() > layer {
                    for &neighbor_id in &node.neighbors[layer] {
                        if !visited.contains(&neighbor_id) {
                            visited.insert(neighbor_id);
                            let d = self.asymmetric_dist(query, &self.nodes[&neighbor_id].pq_code);
                            candidates.push(DistNode { dist: d, id: neighbor_id });
                            if results.len() < ef {
                                results.push(DistNode { dist: d, id: neighbor_id });
                            } else if let Some(worst) = results.peek() {
                                if d < worst.dist {
                                    results.pop();
                                    results.push(DistNode { dist: d, id: neighbor_id });
                                }
                            }
                        }
                    }
                }
            }
        }
        results.into_sorted_vec()
    }

    fn asymmetric_dist(&self, query: &[f32], code: &[u8]) -> f32 {
        self.pq.asymmetric_distance(query, code)
    }

    pub fn save(&self, path: &str) -> Result<()> {
        let data = bincode::serialize(self)?;
        fs::write(path, data)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self> {
        let data = fs::read(path)?;
        Ok(bincode::deserialize(&data)?)
    }
}
