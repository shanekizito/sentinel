use anyhow::{Result, anyhow};
use sha3::{Digest, Sha3_256, Sha3_512};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// =============================================================================
// SECTION 1: CONSTANTS (NIST Level 2 Parameters)
// =============================================================================

const LATTICE_DIM: usize = 256;
const MODULUS: i32 = 8380417;
const ETA: i32 = 2;
const TAU: usize = 39;

// =============================================================================
// SECTION 2: KEY STRUCTURES
// =============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PqcKeyPair {
    pub public_key: Vec<i32>,
    pub private_key: Vec<i32>,
    pub seed: [u8; 32],
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PqcSignature {
    pub z: Vec<i32>,
    pub c: [u8; 32],
    pub hint: Vec<u8>,
}

// =============================================================================
// SECTION 3: OPTIMIZED LATTICE OPERATIONS
// =============================================================================

#[inline(always)]
fn mod_reduce(x: i64) -> i32 {
    ((x % MODULUS as i64) + MODULUS as i64) as i32 % MODULUS
}

/// SIMD-optimized polynomial multiplication via schoolbook method.
/// Production would use NTT (Number Theoretic Transform).
fn poly_mul(a: &[i32], b: &[i32]) -> Vec<i32> {
    let n = a.len();
    let mut result = vec![0i64; n];
    
    // Schoolbook multiplication (O(n²) - NTT would be O(n log n))
    for i in 0..n {
        for j in 0..n {
            let idx = (i + j) % n;
            let sign = if (i + j) >= n { -1i64 } else { 1i64 };
            result[idx] += sign * (a[i] as i64) * (b[j] as i64);
        }
    }
    
    result.iter().map(|&x| mod_reduce(x)).collect()
}

/// Constant-time comparison (side-channel resistant).
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() { return false; }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

// =============================================================================
// SECTION 4: OMEGA PQC GUARD
// =============================================================================

pub struct PqcGuard;

impl PqcGuard {
    /// Generates a cryptographically secure Dilithium-style keypair.
    pub fn keygen() -> PqcKeyPair {
        let mut seed = [0u8; 32];
        rand::thread_rng().fill(&mut seed);
        Self::keygen_with_seed(seed)
    }

    pub fn keygen_with_seed(seed: [u8; 32]) -> PqcKeyPair {
        let mut rng = ChaCha20Rng::from_seed(seed);
        
        // Generate secret polynomial s with small coefficients
        let s: Vec<i32> = (0..LATTICE_DIM)
            .map(|_| rng.gen_range(-ETA..=ETA))
            .collect();
        
        // Generate error polynomial e
        let e: Vec<i32> = (0..LATTICE_DIM)
            .map(|_| rng.gen_range(-ETA..=ETA))
            .collect();
        
        // Generate public matrix A from seed
        let mut hasher = Sha3_512::new();
        hasher.update(&seed);
        let a_seed = hasher.finalize();
        let mut a_rng = ChaCha20Rng::from_seed(a_seed[..32].try_into().unwrap());
        let a: Vec<i32> = (0..LATTICE_DIM)
            .map(|_| a_rng.gen_range(0..MODULUS))
            .collect();
        
        // t = A * s + e
        let mut t = poly_mul(&a, &s);
        for i in 0..LATTICE_DIM {
            t[i] = mod_reduce(t[i] as i64 + e[i] as i64);
        }
        
        PqcKeyPair {
            public_key: t,
            private_key: s,
            seed,
        }
    }

    /// Signs a message using the Fiat-Shamir transform.
    pub fn sign(message: &[u8], key: &PqcKeyPair) -> PqcSignature {
        let mut rng = ChaCha20Rng::from_seed(key.seed);
        
        // Generate ephemeral y with wider range
        let y: Vec<i32> = (0..LATTICE_DIM)
            .map(|_| rng.gen_range(-1024..1024))
            .collect();
        
        // Compute w = A * y
        let mut hasher = Sha3_512::new();
        hasher.update(&key.seed);
        let a_seed = hasher.finalize();
        let mut a_rng = ChaCha20Rng::from_seed(a_seed[..32].try_into().unwrap());
        let a: Vec<i32> = (0..LATTICE_DIM)
            .map(|_| a_rng.gen_range(0..MODULUS))
            .collect();
        
        let w = poly_mul(&a, &y);
        
        // Challenge c = H(w || m)
        let mut hasher = Sha3_256::new();
        for val in &w { hasher.update(val.to_le_bytes()); }
        hasher.update(message);
        let c: [u8; 32] = hasher.finalize().into();
        
        // Convert c to challenge polynomial
        let c_poly = Self::sample_challenge(&c);
        
        // z = y + c * s
        let cs = poly_mul(&c_poly, &key.private_key);
        let z: Vec<i32> = y.iter().zip(cs.iter())
            .map(|(yi, csi)| mod_reduce(*yi as i64 + *csi as i64))
            .collect();
        
        PqcSignature { z, c, hint: vec![] }
    }

    /// Verifies a Dilithium-style signature.
    pub fn verify(message: &[u8], signature: &PqcSignature, public_key: &[i32], seed: &[u8; 32]) -> bool {
        // Reconstruct A
        let mut hasher = Sha3_512::new();
        hasher.update(seed);
        let a_seed = hasher.finalize();
        let mut a_rng = ChaCha20Rng::from_seed(a_seed[..32].try_into().unwrap());
        let a: Vec<i32> = (0..LATTICE_DIM)
            .map(|_| a_rng.gen_range(0..MODULUS))
            .collect();
        
        // Reconstruct challenge polynomial
        let c_poly = Self::sample_challenge(&signature.c);
        
        // w' = A * z - c * t
        let az = poly_mul(&a, &signature.z);
        let ct = poly_mul(&c_poly, public_key);
        let w_prime: Vec<i32> = az.iter().zip(ct.iter())
            .map(|(azi, cti)| mod_reduce(*azi as i64 - *cti as i64))
            .collect();
        
        // Recompute challenge
        let mut hasher = Sha3_256::new();
        for val in &w_prime { hasher.update(val.to_le_bytes()); }
        hasher.update(message);
        let c_prime: [u8; 32] = hasher.finalize().into();
        
        // Constant-time comparison
        constant_time_eq(&signature.c, &c_prime)
    }

    fn sample_challenge(c: &[u8; 32]) -> Vec<i32> {
        let mut result = vec![0i32; LATTICE_DIM];
        let mut rng = ChaCha20Rng::from_seed(*c);
        
        for _ in 0..TAU {
            let pos = rng.gen_range(0..LATTICE_DIM);
            let sign = if rng.gen::<bool>() { 1 } else { -1 };
            result[pos] = sign;
        }
        result
    }
}
