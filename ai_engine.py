"""
Sentinel AI Engine - Self-Learning Pattern Recognition System
Implements adaptive Bayesian learning with persistent pattern storage
"""

import json
import os
import pickle
import hashlib
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import re

class PatternDatabase:
    """Persistent storage for learned vulnerability patterns"""
    
    def __init__(self, db_path: str = "ai_patterns.db"):
        self.db_path = db_path
        self.patterns = self._load_patterns()
        self.pattern_stats = defaultdict(lambda: {"seen": 0, "correct": 0, "false_positives": 0})
        
    def _load_patterns(self) -> Dict:
        """Load patterns from disk"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    return pickle.load(f)
            except:
                return self._initialize_patterns()
        return self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict:
        """Initialize with known vulnerability patterns"""
        return {
            "sql_injection": {
                "regex": [
                    r"query\s*\(\s*['\"].*\+.*['\"]",  # String concatenation in queries
                    r"execute\s*\(\s*['\"].*\+",
                    r"SELECT.*FROM.*WHERE.*\+",
                ],
                "keywords": ["query", "execute", "SELECT", "INSERT", "UPDATE", "DELETE"],
                "severity": "high",
                "cvss_base": 8.0,
                "confidence": 0.85
            },
            "xss": {
                "regex": [
                    r"innerHTML\s*=\s*.*\+",
                    r"document\.write\s*\(",
                    r"eval\s*\(",
                    r"dangerouslySetInnerHTML",
                ],
                "keywords": ["innerHTML", "document.write", "eval"],
                "severity": "medium",
                "cvss_base": 6.0,
                "confidence": 0.75
            },
            "rce": {
                "regex": [
                    r"eval\s*\(",
                    r"exec\s*\(",
                    r"system\s*\(",
                    r"shell_exec",
                    r"passthru",
                ],
                "keywords": ["eval", "exec", "system", "shell"],
                "severity": "critical",
                "cvss_base": 9.5,
                "confidence": 0.90
            },
            "hardcoded_secrets": {
                "regex": [
                    r"password\s*=\s*['\"][^'\"]{8,}['\"]",
                    r"api[_-]?key\s*=\s*['\"][^'\"]+['\"]",
                    r"secret\s*=\s*['\"][^'\"]+['\"]",
                    r"token\s*=\s*['\"][^'\"]{20,}['\"]",
                ],
                "keywords": ["password", "api_key", "secret", "token"],
                "severity": "critical",
                "cvss_base": 9.0,
                "confidence": 0.80
            },
            "weak_crypto": {
                "regex": [
                    r"createHash\s*\(\s*['\"]md5['\"]",
                    r"createHash\s*\(\s*['\"]sha1['\"]",
                    r"DES\s*\(",
                ],
                "keywords": ["md5", "sha1", "DES"],
                "severity": "high",
                "cvss_base": 7.5,
                "confidence": 0.85
            },
        }
    
    def save_patterns(self):
        """Persist patterns to disk"""
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.patterns, f)
    
    def add_pattern(self, category: str, regex: str, confidence: float):
        """Add new learned pattern"""
        if category not in self.patterns:
            self.patterns[category] = {
                "regex": [],
                "keywords": [],
                "severity": "medium",
                "cvss_base": 5.0,
                "confidence": confidence
            }
        
        if regex not in self.patterns[category]["regex"]:
            self.patterns[category]["regex"].append(regex)
            self.save_patterns()
    
    def update_confidence(self, category: str, correct: bool):
        """Update pattern confidence based on feedback"""
        if category in self.patterns:
            stats = self.pattern_stats[category]
            stats["seen"] += 1
            if correct:
                stats["correct"] += 1
            else:
                stats["false_positives"] += 1
            
            # Adaptive confidence adjustment
            accuracy = stats["correct"] / stats["seen"] if stats["seen"] > 0 else 0.5
            self.patterns[category]["confidence"] = accuracy
            self.save_patterns()


class AdaptiveBayesClassifier:
    """Self-learning Bayesian classifier with continuous improvement"""
    
    def __init__(self, pattern_db: PatternDatabase):
        self.pattern_db = pattern_db
        self.feature_counts = defaultdict(lambda: defaultdict(int))
        self.class_counts = defaultdict(int)
        self.total_samples = 0
        self.learning_rate = 0.1
        
    def extract_features(self, code: str) -> List[str]:
        """Extract features from code snippet"""
        features = []
        
        # Lexical features
        features.append(f"length_{len(code) // 100}")
        features.append(f"lines_{code.count(chr(10)) // 10}")
        
        # Keyword features
        keywords = ["eval", "exec", "query", "password", "secret", "api_key", 
                   "innerHTML", "document", "system", "shell", "md5", "sha1"]
        for kw in keywords:
            if kw.lower() in code.lower():
                features.append(f"has_{kw}")
        
        # Pattern features
        if re.search(r"['\"].*\+.*['\"]", code):
            features.append("string_concat")
        if re.search(r"=\s*['\"][^'\"]{20,}['\"]", code):
            features.append("long_string_literal")
        if re.search(r"\b(SELECT|INSERT|UPDATE|DELETE)\b", code, re.IGNORECASE):
            features.append("sql_keyword")
        
        return features
    
    def learn(self, features: List[str], is_vulnerable: bool):
        """Incremental learning from new samples"""
        label = "vulnerable" if is_vulnerable else "safe"
        
        self.class_counts[label] += 1
        self.total_samples += 1
        
        for feature in features:
            self.feature_counts[feature][label] += 1
    
    def predict(self, features: List[str]) -> float:
        """Predict vulnerability probability using Naive Bayes"""
        if self.total_samples == 0:
            return 0.5
        
        # Prior probabilities
        p_vuln = (self.class_counts["vulnerable"] + 1) / (self.total_samples + 2)
        p_safe = (self.class_counts["safe"] + 1) / (self.total_samples + 2)
        
        # Likelihood calculation
        log_p_vuln = 0
        log_p_safe = 0
        
        for feature in features:
            vuln_count = self.feature_counts[feature]["vulnerable"] + 1
            safe_count = self.feature_counts[feature]["safe"] + 1
            total_vuln = self.class_counts["vulnerable"] + len(self.feature_counts)
            total_safe = self.class_counts["safe"] + len(self.feature_counts)
            
            log_p_vuln += (vuln_count / total_vuln)
            log_p_safe += (safe_count / total_safe)
        
        # Posterior probability
        score = (p_vuln * log_p_vuln) / ((p_vuln * log_p_vuln) + (p_safe * log_p_safe))
        return max(0.0, min(1.0, score))


class LogicalFlowAnalyzer:
    """Analyzes code logical flow for vulnerabilities"""
    
    def __init__(self):
        self.flow_patterns = {
            "auth_bypass": {
                "pattern": r"if\s*\([^)]*\)\s*{\s*return\s+true",
                "description": "Potential authentication bypass",
                "severity": "critical"
            },
            "missing_validation": {
                "pattern": r"(req\.|request\.|input\.).*(?!.*validate)",
                "description": "Input used without validation",
                "severity": "high"
            },
            "error_exposure": {
                "pattern": r"catch\s*\([^)]*\)\s*{\s*console\.log",
                "description": "Error details exposed in logs",
                "severity": "medium"
            },
        }
    
    def analyze_flow(self, code: str) -> List[Dict]:
        """Analyze logical flow for security issues"""
        findings = []
        
        for flow_type, config in self.flow_patterns.items():
            matches = re.finditer(config["pattern"], code, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                findings.append({
                    "type": flow_type,
                    "description": config["description"],
                    "severity": config["severity"],
                    "location": match.span(),
                    "snippet": match.group(0)
                })
        
        return findings


class SentinelAI:
    """Main AI engine coordinating all analysis components"""
    
    def __init__(self):
        self.pattern_db = PatternDatabase()
        self.classifier = AdaptiveBayesClassifier(self.pattern_db)
        self.flow_analyzer = LogicalFlowAnalyzer()
        self.scan_history = []
        
    def scan_code(self, code: str, filename: str) -> Dict:
        """Comprehensive code analysis"""
        results = {
            "filename": filename,
            "timestamp": datetime.now().isoformat(),
            "vulnerabilities": [],
            "confidence": 0.0,
            "logical_flows": []
        }
        
        # Extract features
        features = self.classifier.extract_features(code)
        
        # Pattern matching
        for category, pattern_config in self.pattern_db.patterns.items():
            for regex in pattern_config["regex"]:
                if re.search(regex, code, re.IGNORECASE):
                    results["vulnerabilities"].append({
                        "category": category,
                        "severity": pattern_config["severity"],
                        "cvss": pattern_config["cvss_base"],
                        "confidence": pattern_config["confidence"],
                        "pattern": regex
                    })
        
        # ML prediction
        ml_confidence = self.classifier.predict(features)
        results["confidence"] = ml_confidence
        
        # Logical flow analysis
        flow_issues = self.flow_analyzer.analyze_flow(code)
        results["logical_flows"] = flow_issues
        
        # Learn from this scan
        is_vulnerable = len(results["vulnerabilities"]) > 0 or len(flow_issues) > 0
        self.classifier.learn(features, is_vulnerable)
        
        # Store scan history
        self.scan_history.append(results)
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get AI learning statistics"""
        return {
            "total_scans": len(self.scan_history),
            "total_samples": self.classifier.total_samples,
            "patterns_learned": len(self.pattern_db.patterns),
            "feature_vocabulary": len(self.classifier.feature_counts),
            "accuracy_estimate": self._estimate_accuracy()
        }
    
    def _estimate_accuracy(self) -> float:
        """Estimate model accuracy from pattern statistics"""
        if not self.pattern_db.pattern_stats:
            return 0.0
        
        total_correct = sum(s["correct"] for s in self.pattern_db.pattern_stats.values())
        total_seen = sum(s["seen"] for s in self.pattern_db.pattern_stats.values())
        
        return total_correct / total_seen if total_seen > 0 else 0.0


# Export for use in demo_runner
__all__ = ['SentinelAI', 'PatternDatabase', 'AdaptiveBayesClassifier', 'LogicalFlowAnalyzer']
