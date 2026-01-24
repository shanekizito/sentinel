// use anyhow::Result; // Removed unused
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::info;

/// Production-grade Probabilistic Bloom Filter.
/// Uses atomic bitsets to allow concurrent symbol insertions during parallel parsing.
pub struct IndustrialBloomFilter {
    bitset: Vec<AtomicU64>,
    m: usize, // Number of bits
    k: u8,    // Number of hash functions
}

impl IndustrialBloomFilter {
    /// Creates a new filter with M bits and K hashes.
    /// Optimized for 100MB+ configurations.
    pub fn new(m_bits: usize, k_hashes: u8) -> Self {
        info!("Sovereign Parser: Initializing Industrial Bloom Filter (M: {}, K: {})", m_bits, k_hashes);
        let u64_count = (m_bits + 63) / 64;
        let mut bitset = Vec::with_capacity(u64_count);
        for _ in 0..u64_count {
            bitset.push(AtomicU64::new(0));
        }
        
        Self {
            bitset,
            m: m_bits,
            k: k_hashes,
        }
    }

    /// Probabilistically inserts a symbol into the filter.
    pub fn insert(&self, symbol: &str) {
        let hashes = self.get_hashes(symbol);
        for hash in hashes {
            let bit_idx = (hash % self.m as u64) as usize;
            let word_idx = bit_idx / 64;
            let bit_offset = bit_idx % 64;
            let mask = 1 << bit_offset;
            
            // Use fetch_or for atomic thread-safe insertion
            self.bitset[word_idx].fetch_or(mask, Ordering::Relaxed);
        }
    }

    /// Returns true if the symbol MIGHT be in the set.
    /// Returns false if the symbol is DEFINITELY NOT in the set.
    pub fn contains(&self, symbol: &str) -> bool {
        let hashes = self.get_hashes(symbol);
        for hash in hashes {
            let bit_idx = (hash % self.m as u64) as usize;
            let word_idx = bit_idx / 64;
            let bit_offset = bit_idx % 64;
            let mask = 1 << bit_offset;
            
            let val = self.bitset[word_idx].load(Ordering::Relaxed);
            if (val & mask) == 0 {
                return false;
            }
        }
        true
    }

    fn get_hashes(&self, symbol: &str) -> Vec<u64> {
        let mut results = Vec::with_capacity(self.k as usize);
        for i in 0..self.k {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            symbol.hash(&mut hasher);
            i.hash(&mut hasher); // Salt each hash
            results.push(hasher.finish());
        }
        results
    }
}
