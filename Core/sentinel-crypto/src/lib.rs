pub mod pqc;
pub mod zkp;

use anyhow::Result;

pub struct PqcGuard;

impl PqcGuard {
    pub fn new() -> Self {
        Self
    }
}
