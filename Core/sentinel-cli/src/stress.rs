use anyhow::{Result, Context};
use std::path::Path;
use std::fs;
use std::io::Write;
use rand::Rng;
use rand::prelude::SliceRandom;
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};

pub struct StressGenerator;

impl StressGenerator {
    /// Generates a massive synthetic monorepo with realistic topology.
    pub fn generate_monorepo(target_dir: &str, file_count: usize) -> Result<()> {
        let root = Path::new(target_dir);
        if root.exists() {
            fs::remove_dir_all(root).context("Failed to clear bench dir")?;
        }
        fs::create_dir_all(root)?;

        println!("Generating Industry-Standard Benchmark: {} Files...", file_count);
        let pb = ProgressBar::new(file_count as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap());

// 1. Generate Logical Topology (DAG)
        // We assign each file a "layer" so imports only go downwards (preventing cycles generally, though cycles are good for stress)
        // For this bench, we want mostly DAG but some cycles.
        
        // Parallel Generation
        (0..file_count).into_par_iter().for_each(|i| {
            let mut rng = rand::thread_rng();
            let lang_type = rng.gen_range(0..3); // 0=Rust, 1=Py, 2=TS
            
            let (filename, content) = match lang_type {
                0 => generate_rust_module(i, file_count),
                1 => generate_python_module(i, file_count),
                _ => generate_ts_module(i, file_count),
            };
            
            let file_path = root.join(filename);
            if let Ok(mut f) = fs::File::create(&file_path) {
                let _ = f.write_all(content.as_bytes());
            }
            pb.inc(1);
        });

        pb.finish_with_message("Done");
        Ok(())
    }
}

fn generate_rust_module(id: usize, total: usize) -> (String, String) {
    let mut rng = rand::thread_rng();
    let name = format!("mod_{:06}", id);
    let mut code = String::new();
    
    // Imports
    code.push_str("// Synthetic Rust Module\n");
    let num_imports = rng.gen_range(2..10);
    for _ in 0..num_imports {
        let target = rng.gen_range(0..total);
        if target != id {
             code.push_str(&format!("use crate::mod_{:06};\n", target));
        }
    }
    
    // Complex Functions
    code.push_str("\npub struct DataPacket {\n    pub id: u64,\n    pub payload: Vec<u8>,\n}\n\n");
    
    let num_fns = rng.gen_range(5..20);
    for f in 0..num_fns {
        code.push_str(&format!("\npub fn logic_{}(input: DataPacket) -> Result<u64, String> {{\n", f));
        // Generate complexity (Control Flow)
        code.push_str("    let mut x = input.id;\n");
        code.push_str("    for i in 0..100 {\n");
        code.push_str("        if x % 2 == 0 { x = x.wrapping_div(2); }\n");
        code.push_str("        else { x = x.wrapping_mul(3).wrapping_add(1); }\n");
        code.push_str("    }\n");
        code.push_str("    if x == 0 { return Err(\"Overflow\".to_string()); }\n");
        code.push_str("    Ok(x)\n}\n");
    }
    
    (format!("{}.rs", name), code)
}

fn generate_python_module(id: usize, total: usize) -> (String, String) {
    let mut rng = rand::thread_rng();
    let name = format!("mod_{:06}", id);
    let mut code = String::new();
    
    code.push_str("# Synthetic Python Logic\nimport os\nimport sys\n");
    
    let num_imports = rng.gen_range(2..8);
    for _ in 0..num_imports {
        let target = rng.gen_range(0..total);
        if target != id {
             code.push_str(&format!("import mod_{:06}\n", target));
        }
    }
    
    code.push_str("\nclass BusinessLogic:\n");
    code.push_str("    def __init__(self, context):\n        self.ctx = context\n");
    
    let num_methods = rng.gen_range(3..15);
    for m in 0..num_methods {
        code.push_str(&format!("\n    def process_transaction_{}(self, data):\n", m));
        code.push_str("        result = 0\n");
        code.push_str("        try:\n");
        code.push_str("            if data['value'] > 1000:\n");
        code.push_str("                result = data['value'] * 0.9\n");
        code.push_str("            else:\n");
        code.push_str("                result = data['value']\n");
        code.push_str("        except KeyError:\n");
        code.push_str("            return None\n");
        code.push_str("        return result\n");
    }

    (format!("{}.py", name), code)
}

fn generate_ts_module(id: usize, total: usize) -> (String, String) {
    let mut rng = rand::thread_rng();
    let name = format!("mod_{:06}", id);
    let mut code = String::new();
    
    code.push_str("// Synthetic TypeScript Component\n");
    
    let num_imports = rng.gen_range(2..8);
    for _ in 0..num_imports {
        let target = rng.gen_range(0..total);
        if target != id {
             code.push_str(&format!("import {{ component_{} }} from './mod_{:06}';\n", target, target));
        }
    }
    
    code.push_str("\ninterface Props { id: number; active: boolean; }\n");
    code.push_str(&format!("\nexport const component_{} = (props: Props) => {{\n", id));
    code.push_str("    const [state, setState] = useState(0);\n");
    code.push_str("    useEffect(() => {\n");
    code.push_str("        if (props.active) { setState(s => s + 1); }\n");
    code.push_str("    }, [props.active]);\n");
    code.push_str("    return <div>{state}</div>;\n};\n");

    (format!("{}.tsx", name), code)
}
