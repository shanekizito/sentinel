// Sentinel Sovereign Test Suite: Concurrency Data Race
// Scenario: Data Race on shared state via improper synchronization.

use std::sync::{Arc, Mutex};
use std::thread;

struct GlobalAccount {
    balance: i64,
}

fn drain_account(account: Arc<GlobalAccount>) {
    // VIOLATION: Accessing balance without locking a Mutex
    // Note: In real Rust, this wouldn't compile if balance isn't Atomic or Mutex-wrapped,
    // but in unsafe blocks or logic-heavy C++/Java, this is a major issue.
    // For this benchmark, we're modeling a logic-race.
    
    let current = account.balance;
    thread::sleep(std::time::Duration::from_millis(10));
    // Simulated Race condition
    // account.balance = current - 100;
}

fn main() {
    let account = Arc::new(GlobalAccount { balance: 1000 });
    
    let mut handles = vec![];
    for _ in 0..10 {
        let acc = Arc::clone(&account);
        handles.push(thread::spawn(move || {
            drain_account(acc);
        }));
    }
    
    for h in handles { h.join().unwrap(); }
}
