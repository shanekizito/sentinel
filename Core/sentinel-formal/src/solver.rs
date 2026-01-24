use anyhow::{Result, anyhow, Context};
use tracing::{info, warn, span, Level};
use std::process::{Command, Stdio, Child};
use std::io::{Write, BufRead, BufReader};
use std::time::{Duration, Instant};
use std::sync::Arc;
use parking_lot::Mutex;
// Crossbeam removed as unused


// =============================================================================
// SECTION 1: SMT RESULT
// =============================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum SmtResult {
    Satisfiable(String),
    Unsatisfiable,
    Unknown,
    Timeout,
    Error(String),
}

// =============================================================================
// SECTION 2: Z3 PROCESS POOL
// =============================================================================

struct Z3Process {
    child: Child,
}

impl Z3Process {
    fn spawn() -> Result<Self> {
        let child = Command::new("z3")
            .arg("-in")
            .arg("-smt2")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("Failed to spawn Z3. Is it installed and in PATH?")?;
        Ok(Self { child })
    }

    fn query(&mut self, script: &str, timeout: Duration) -> Result<SmtResult> {
        let stdin = self.child.stdin.as_mut().ok_or(anyhow!("No stdin"))?;
        
        // Write script with check-sat and get-model
        writeln!(stdin, "{}", script)?;
        writeln!(stdin, "(check-sat)")?;
        writeln!(stdin, "(get-model)")?;
        stdin.flush()?;

        // Read response (simplified - production uses async with timeout)
        let stdout = self.child.stdout.as_mut().ok_or(anyhow!("No stdout"))?;
        let mut reader = BufReader::new(stdout);
        let mut response = String::new();
        
        let start = Instant::now();
        while start.elapsed() < timeout {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
            response.push_str(&line);
            if line.contains("sat") || line.contains("unsat") || line.contains("unknown") {
                break;
            }
        }

        if start.elapsed() >= timeout {
            return Ok(SmtResult::Timeout);
        }

        if response.contains("unsat") {
            Ok(SmtResult::Unsatisfiable)
        } else if response.contains("sat") {
            Ok(SmtResult::Satisfiable(response))
        } else if response.contains("unknown") {
            Ok(SmtResult::Unknown)
        } else {
            Ok(SmtResult::Error(response))
        }
    }

    fn reset(&mut self) -> Result<()> {
        if let Some(stdin) = self.child.stdin.as_mut() {
            writeln!(stdin, "(reset)")?;
            stdin.flush()?;
        }
        Ok(())
    }
}

impl Drop for Z3Process {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

// =============================================================================
// SECTION 3: OMEGA Z3 SOLVER POOL
// =============================================================================

pub struct OmegaZ3SolverPool {
    pool: Arc<Mutex<Vec<Z3Process>>>,
    pool_size: usize,
    default_timeout: Duration,
}

impl OmegaZ3SolverPool {
    pub fn new(pool_size: usize) -> Self {
        Self {
            pool: Arc::new(Mutex::new(Vec::new())),
            pool_size,
            default_timeout: Duration::from_secs(30),
        }
    }

    fn acquire(&self) -> Result<Z3Process> {
        let mut pool = self.pool.lock();
        if let Some(proc) = pool.pop() {
            Ok(proc)
        } else {
            Z3Process::spawn()
        }
    }

    fn release(&self, mut proc: Z3Process) {
        if let Ok(()) = proc.reset() {
            let mut pool = self.pool.lock();
            if pool.len() < self.pool_size {
                pool.push(proc);
            }
        }
    }

    pub fn solve(&self, script: &str) -> Result<SmtResult> {
        let _span = span!(Level::INFO, "Z3Solve").entered();
        info!("Omega Z3: Dispatching proof...");
        
        let mut proc = self.acquire()?;
        let result = proc.query(script, self.default_timeout);
        self.release(proc);
        result
    }

    /// Parallel batch solving.
    pub fn solve_batch(&self, scripts: Vec<String>) -> Vec<Result<SmtResult>> {
        use rayon::prelude::*;
        
        scripts.par_iter().map(|script| {
            self.solve(script)
        }).collect()
    }
}

// =============================================================================
// SECTION 4: SIMPLE Z3 SOLVER (Stateless)
// =============================================================================

pub struct Z3Solver;

impl Z3Solver {
    pub fn new() -> Self { Self }

    pub fn solve(&self, smt_script: &str) -> Result<SmtResult> {
        info!("Omega Z3: Dispatching proof to Z3 binary...");
        
        let mut child = Command::new("z3")
            .arg("-in")
            .arg("-smt2")
            .arg("-t:30000")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("Failed to spawn Z3. Is it installed and in PATH?")?;
        
        {
            let stdin = child.stdin.as_mut().ok_or(anyhow!("Failed to open Z3 stdin"))?;
            stdin.write_all(smt_script.as_bytes())?;
        }
        
        let output = child.wait_with_output()?;
        
        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr);
            warn!("Z3 Process Error: {}", err);
            return Ok(SmtResult::Error(err.to_string()));
        }
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        
        if stdout.contains("unsat") {
            Ok(SmtResult::Unsatisfiable)
        } else if stdout.contains("sat") {
            Ok(SmtResult::Satisfiable(stdout.to_string()))
        } else {
            Ok(SmtResult::Unknown)
        }
    }
}
