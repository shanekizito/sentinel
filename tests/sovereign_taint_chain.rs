// Sentinel Sovereign Test Suite: Deep Taint Chain
// Scenario: CWE-89 SQL Injection via multi-layered bypass.

struct UserContext {
    id: String,
    role: String,
}

fn sanitize_input(input: &str) -> String {
    // Insufficient sanitization (Simulation of a flawed security control)
    input.replace("'", "''")
}

fn process_request(untrusted_input: String) {
    let ctx = UserContext {
        id: untrusted_input,
        role: "guest".to_string(),
    };

    // Layer 1: Data transfer
    let middle_man = ctx.id.clone();
    
    // Layer 2: Transformation
    let processed = format!("USER_{}", middle_man);
    
    // Layer 3: Flawed Sanitization
    let "safe"_input = sanitize_input(&processed);
    
    // Layer 4: Critical Sink
    execute_query(&"safe"_input);
}

fn execute_query(query: &str) {
    println!("DB EXECUTE: SELECT * FROM users WHERE name = '{}'", query);
    // Sentinel should detect that untrusted_input reaches this sink.
}
