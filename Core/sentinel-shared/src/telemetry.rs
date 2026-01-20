use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{info, span, Level};

// =============================================================================
// SECTION 1: CORE TELEMETRY STRUCTURES
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryPacket {
    pub job_id: String,
    pub node_id: String,
    pub region: String,
    pub stats: AnalysisStats,
    pub heartbeat_ms: u64,
    pub spans: Vec<SpanData>,
    pub metrics: MetricSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnalysisStats {
    pub nodes_analyzed: u64,
    pub edges_traversed: u64,
    pub proofs_resolved: u32,
    pub latency_ms: f32,
    pub throughput_ops_sec: f64,
    pub memory_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanData {
    pub trace_id: String,
    pub span_id: String,
    pub parent_id: Option<String>,
    pub operation: String,
    pub start_ns: u64,
    pub duration_ns: u64,
    pub status: SpanStatus,
    pub attributes: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpanStatus {
    Ok,
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetricSnapshot {
    pub counters: HashMap<String, u64>,
    pub gauges: HashMap<String, f64>,
    pub histograms: HashMap<String, HistogramData>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HistogramData {
    pub count: u64,
    pub sum: f64,
    pub min: f64,
    pub max: f64,
    pub buckets: Vec<(f64, u64)>, // (upper_bound, count)
}

// =============================================================================
// SECTION 2: TELEMETRY COLLECTOR
// =============================================================================

/// High-performance telemetry collector with atomic counters.
pub struct TelemetryCollector {
    job_id: String,
    node_id: String,
    region: String,
    nodes_analyzed: AtomicU64,
    edges_traversed: AtomicU64,
    proofs_resolved: AtomicU32,
    start_time: Instant,
    spans: parking_lot::Mutex<Vec<SpanData>>,
    latency_histogram: parking_lot::Mutex<HistogramData>,
}

impl TelemetryCollector {
    pub fn new(job_id: &str, node_id: &str, region: &str) -> Self {
        Self {
            job_id: job_id.to_string(),
            node_id: node_id.to_string(),
            region: region.to_string(),
            nodes_analyzed: AtomicU64::new(0),
            edges_traversed: AtomicU64::new(0),
            proofs_resolved: AtomicU32::new(0),
            start_time: Instant::now(),
            spans: parking_lot::Mutex::new(Vec::new()),
            latency_histogram: parking_lot::Mutex::new(HistogramData::default()),
        }
    }

    pub fn inc_nodes(&self, count: u64) {
        self.nodes_analyzed.fetch_add(count, Ordering::Relaxed);
    }

    pub fn inc_edges(&self, count: u64) {
        self.edges_traversed.fetch_add(count, Ordering::Relaxed);
    }

    pub fn inc_proofs(&self) {
        self.proofs_resolved.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_latency(&self, latency_ms: f64) {
        let mut hist = self.latency_histogram.lock();
        hist.count += 1;
        hist.sum += latency_ms;
        if latency_ms < hist.min || hist.count == 1 { hist.min = latency_ms; }
        if latency_ms > hist.max { hist.max = latency_ms; }
    }

    /// Creates a new span for tracing.
    pub fn start_span(&self, operation: &str) -> SpanGuard {
        SpanGuard::new(operation.to_string(), self)
    }

    fn record_span(&self, span: SpanData) {
        self.spans.lock().push(span);
    }

    /// Generates a snapshot packet for export.
    pub fn snapshot(&self) -> TelemetryPacket {
        let elapsed = self.start_time.elapsed();
        let nodes = self.nodes_analyzed.load(Ordering::Relaxed);
        let throughput = if elapsed.as_secs_f64() > 0.0 {
            nodes as f64 / elapsed.as_secs_f64()
        } else { 0.0 };

        TelemetryPacket {
            job_id: self.job_id.clone(),
            node_id: self.node_id.clone(),
            region: self.region.clone(),
            stats: AnalysisStats {
                nodes_analyzed: nodes,
                edges_traversed: self.edges_traversed.load(Ordering::Relaxed),
                proofs_resolved: self.proofs_resolved.load(Ordering::Relaxed),
                latency_ms: elapsed.as_millis() as f32,
                throughput_ops_sec: throughput,
                memory_mb: self.get_memory_usage(),
            },
            heartbeat_ms: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64,
            spans: self.spans.lock().clone(),
            metrics: MetricSnapshot {
                counters: [
                    ("nodes".to_string(), nodes),
                    ("edges".to_string(), self.edges_traversed.load(Ordering::Relaxed)),
                ].into_iter().collect(),
                gauges: [
                    ("throughput".to_string(), throughput),
                ].into_iter().collect(),
                histograms: [
                    ("latency_ms".to_string(), self.latency_histogram.lock().clone()),
                ].into_iter().collect(),
            },
        }
    }

    fn get_memory_usage(&self) -> u64 {
        // Platform-specific memory query (simplified)
        0
    }
}

// =============================================================================
// SECTION 3: SPAN GUARD (RAII)
// =============================================================================

pub struct SpanGuard<'a> {
    operation: String,
    start: Instant,
    collector: &'a TelemetryCollector,
    status: SpanStatus,
}

impl<'a> SpanGuard<'a> {
    fn new(operation: String, collector: &'a TelemetryCollector) -> Self {
        Self {
            operation,
            start: Instant::now(),
            collector,
            status: SpanStatus::Ok,
        }
    }

    pub fn set_error(&mut self, msg: String) {
        self.status = SpanStatus::Error(msg);
    }
}

impl<'a> Drop for SpanGuard<'a> {
    fn drop(&mut self) {
        let span = SpanData {
            trace_id: uuid::Uuid::new_v4().to_string(),
            span_id: uuid::Uuid::new_v4().to_string(),
            parent_id: None,
            operation: self.operation.clone(),
            start_ns: 0,
            duration_ns: self.start.elapsed().as_nanos() as u64,
            status: self.status.clone(),
            attributes: HashMap::new(),
        };
        self.collector.record_span(span);
    }
}

// =============================================================================
// SECTION 4: GLOBAL TELEMETRY REGISTRY
// =============================================================================

lazy_static::lazy_static! {
    pub static ref GLOBAL_TELEMETRY: Arc<TelemetryCollector> = 
        Arc::new(TelemetryCollector::new("global", "local", "default"));
}

pub fn global() -> Arc<TelemetryCollector> {
    GLOBAL_TELEMETRY.clone()
}
