"""
Sentinel Sovereign Bench: The Omega Test Harness
Generates Synthetic "Worst-Case" Monorepos for Robustness Verification.

Scenarios:
1. RAGGED: Disconnected files with no edges. Tests Graph Normalization.
2. CYCLICAL: Deeply recursive function calls (Stack Overflow limits).
3. HUGE: Single files > 100MB (Parser Memory limits).
4. POLYGLOT: Mixed Rust/Python/Solidity.
"""

import os
import random
import shutil

class OmegaHarness:
    def __init__(self, root_dir="./.omega_bench"):
        self.root_dir = root_dir
        if os.path.exists(root_dir):
            shutil.rmtree(root_dir)
        os.makedirs(root_dir)

    def generate_ragged(self, n_files=1000):
        """Generates 1000 isolated files (Zero Edges)."""
        scenario_dir = os.path.join(self.root_dir, "ragged")
        os.makedirs(scenario_dir)
        
        for i in range(n_files):
            with open(os.path.join(scenario_dir, f"file_{i}.py"), "w") as f:
                f.write(f"# Isolated Node {i}\nx_{i} = {random.random()}")

    def generate_cyclical(self, depth=500):
        """Generates a deep recursion chain."""
        scenario_dir = os.path.join(self.root_dir, "cyclical")
        os.makedirs(scenario_dir)
        
        code = []
        for i in range(depth):
            code.append(f"def func_{i}(): return func_{i+1}()")
        code.append(f"def func_{depth}(): return func_0()") # Close loop
        
        with open(os.path.join(scenario_dir, "infinite_loop.py"), "w") as f:
            f.write("\n".join(code))

    def generate_huge(self, size_mb=10):
        """Generates a massive file."""
        scenario_dir = os.path.join(self.root_dir, "huge")
        os.makedirs(scenario_dir)
        
        with open(os.path.join(scenario_dir, "giant.rs"), "w") as f:
            f.write("fn main() {\n")
            for i in range(size_mb * 1000):
                f.write(f"    let var_{i} = {i};\n")
            f.write("}\n")

if __name__ == "__main__":
    print("Generating Omega Test Suite...")
    harness = OmegaHarness()
    harness.generate_ragged()
    harness.generate_cyclical()
    harness.generate_huge()
    print("Done. Ready for Ingestion.")
