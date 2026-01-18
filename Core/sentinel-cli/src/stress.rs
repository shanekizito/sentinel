use anyhow::Result;
use std::fs::{File, create_dir_all};
use std::io::Write;
use std::path::Path;
use tracing::info;

/// Generates massive synthetic codebases for performance benchmarking.
/// Can generate 1,000,000+ lines of TypeScript code in seconds.
pub struct StressGenerator;

impl StressGenerator {
    pub fn generate_monorepo<P: AsRef<Path>>(root: P, file_count: usize) -> Result<()> {
        info!("Stress Generator: Building synthetic monorepo with {} files...", file_count);
        create_dir_all(&root)?;

        for i in 0..file_count {
            let file_path = root.as_ref().join(format!("service_{}.ts", i));
            let mut file = File::create(file_path)?;
            
            // Generate complex logic to stress the CPG builder
            let content = self::generate_complex_ts_source(i);
            file.write_all(content.as_bytes())?;
            
            if i % 1000 == 0 && i > 0 {
                info!("  [Progress] Generated {} files (approx {}k lines)", i, i * 100 / 1000);
            }
        }

        info!("Stress Generator: Complete. Million-line environment ready for audit.");
        Ok(())
    }

    fn generate_complex_ts_source(index: usize) -> String {
        format!(
            r#"
/**
 * Sentinel Stress Test Service v{}
 * Auto-generated for performance benchmarking.
 */
import {{ db }} from './db_adapter';

export class Service{} {{
    private secret_key: string = "SENTINEL_DEMO_KEY_{}";

    public async executeTransaction(userInput: string): Promise<void> {{
        console.log("Processing transaction for index: {}");
        
        // Potential SQL Injection Sink
        const query = "SELECT * FROM users WHERE id = " + userInput;
        await db.query(query);

        // Potential Command Injection Sink
        if (userInput.startsWith("admin")) {{
            const {{ exec }} = require('child_process');
            exec("rm -rf /tmp/" + userInput);
        }

        this.internalStateChange();
    }}

    private internalStateChange(): void {{
        // Deep call stack to stress inter-procedural analysis
        this.step1();
    }}

    private step1() {{ this.step2(); }}
    private step2() {{ this.step3(); }}
    private step3() {{ console.log("State stabilized."); }}
}}
"#,
            index, index, index, index
        )
    }
}
