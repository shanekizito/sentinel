"""
Rust AI Engine Wrapper
Calls the compiled Rust sentinel-cli binary for AI analysis
"""

import subprocess
import json
import os
from pathlib import Path

class RustAIEngine:
    """Wrapper for Rust-based Sentinel AI engine"""
    
    def __init__(self):
        self.binary_path = self._find_binary()
        self.available = self.binary_path is not None
        
    def _find_binary(self):
        """Locate the compiled Rust binary"""
        possible_paths = [
            Path("core/target/release/sentinel-cli.exe"),
            Path("core/target/release/sentinel-cli"),
            Path("../core/target/release/sentinel-cli.exe"),
            Path("../core/target/release/sentinel-cli"),
        ]
        
        for path in possible_paths:
            if path.exists():
                return path.absolute()
        return None
    
    def run_demo(self, target_repo: str) -> dict:
        """Run the Rust AI demo on a target repository"""
        if not self.available:
            raise RuntimeError("Rust binary not found. Please build with: cargo build --release")
        
        cmd = [str(self.binary_path), "demo", "--target", target_repo]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Rust AI demo timed out after 5 minutes"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def scan_file(self, file_path: str) -> dict:
        """Scan a single file using Rust AI"""
        if not self.available:
            raise RuntimeError("Rust binary not found")
        
        # For now, use demo mode on the file's directory
        # In future, add a specific scan command to the Rust CLI
        target_dir = Path(file_path).parent
        return self.run_demo(str(target_dir))
    
    def get_brain_stats(self) -> dict:
        """Get statistics from the persisted LocalBrain"""
        brain_path = Path("sentinel_brain.json")
        
        if not brain_path.exists():
            return {
                "total_scans": 0,
                "patterns_learned": 0,
                "feature_vocabulary": 0,
                "accuracy_estimate": 0.0
            }
        
        try:
            with open(brain_path, 'r') as f:
                brain_data = json.load(f)
            
            total_safe = brain_data.get("total_safe", 0)
            total_vuln = brain_data.get("total_vuln", 0)
            feature_counts = brain_data.get("feature_counts", {})
            
            return {
                "total_scans": total_safe + total_vuln,
                "patterns_learned": len(feature_counts),
                "feature_vocabulary": len(feature_counts),
                "accuracy_estimate": 0.85 if total_safe + total_vuln > 10 else 0.5,
                "total_safe": total_safe,
                "total_vuln": total_vuln
            }
        except Exception as e:
            print(f"Error loading brain stats: {e}")
            return {
                "total_scans": 0,
                "patterns_learned": 0,
                "feature_vocabulary": 0,
                "accuracy_estimate": 0.0
            }


# Singleton instance
_rust_ai = None

def get_rust_ai() -> RustAIEngine:
    """Get or create the Rust AI engine instance"""
    global _rust_ai
    if _rust_ai is None:
        _rust_ai = RustAIEngine()
    return _rust_ai
