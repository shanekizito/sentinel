# Sentinel: Autonomous Security Governance Platform

**Sentinel** is a large-scale, autonomous security governance platform designed for **Industrial Code Intelligence**. It unifies **Neural Geometries** (Graph Convolutional Networks) with **Symbolic Reasoning** (SMT Solvers) into a single, robust intelligence capable of analyzing massive codebases and proving their security properties in real-time.

## Executive Summary

Sentinel utilizes a dual-layered AI architecture to bridge the gap between probabilistic pattern matching and deterministic verification:

1.  **Neural Layer**: Processes code patterns using **Graph Convolutional Networks (GCNs)** to identify potential vulnerabilities.
2.  **Symbolic Layer**: Validates these findings using **SMT Provers** (Z3, CVC5) to ensure mathematically rigorous security proofs.

This architecture enables the system to operate as an **Autonomous Controller**, capable of detecting, verifying, and patching vulnerabilities without human intervention.

---

## Project Structure

Sentinel is organized into a modular architecture optimized for performance and scalability.

```
sentinel/
├── Core/
│   ├── sentinel-ai/            # Neural Logic & Inference Bridge
│   │   └── src/                # GCN Encoders & Triton gRPC Client
│   ├── sentinel-formal/        # Symbolic Verification Engine
│   │   └── src/                # SMT Solver Interfaces (Z3, CVC5)
│   ├── sentinel-orchestrator/  # Distributed System Management
│   │   └── src/                # Job Scheduling & Mesh Protocols
│   ├── sentinel-cli/           # Command Line Interface
│   │   └── src/                # User Entry Point
│   └── sentinel-reflex/        # Autonomous Patching Engine
├── docs/                       # Detailed Architecture & Specifications
└── rules/                      # Security Rules & Invariants
```

---

## How it Works: The Logic Flow

The engine processes code in a linear, verifiable pipeline:

1.  **Ingestion**: Source code is parsed and converted into a **Code Property Graph (CPG)**, which combines the Abstract Syntax Tree (AST), Control Flow Graph (CFG), and Program Dependence Graph (PDG).
2.  **Analysis (Neural)**: The Neural Layer scans the CPG for "fuzzy" matches—patterns that resemble known vulnerabilities—and generates a set of candidates.
3.  **Verification (Symbolic)**: Each candidate is passed to the Symbolic Layer. SMT solvers race to prove whether the vulnerability is reachable and exploitable.
4.  **Remediation (Reflex)**: If a vulnerability is confirmed, the Reflex Engine generates a patch. This patch is then verified to ensure it fixes the issue without introducing new regressions before being autonomously committed.

---

## Glossary of Key Terms

*   **CPG (Code Property Graph)**: A directed graph representation of code that merges structural (AST), control flow (CFG), and data dependency (PDG) information into a single queryable structure.
*   **Neural-Symbolic AI**: An AI approach that combines neural networks (good at pattern recognition) with symbolic logic (good at reasoning and verification).
*   **SMT Solver**: "Satisfiability Modulo Theories" solver. A tool used for automated theorem proving, allowing Sentinel to mathematically prove code properties.
*   **Reflex Engine**: The subsystem responsible for hypothesizing, generating, and verifying code patches.
*   **Zero-Copy Parsing**: A performance optimization where data is mapped directly from disk to memory (using Mmap) without copying, essential for handling large repositories.

---

## Technical Pillars

### 1. High-Performance Core
Sentinel is engineered for large monorepos, operating with low latency.
*   **Memory-Mapped I/O**: Stores massive graphs on NVMe storage to bypass memory limits.
*   **Multi-Solver Racing**: Runs multiple solvers (Z3, CVC5) in parallel. The first to find a proof terminates the others, significantly speeding up verification.

### 2. Security Grid
*   **Post-Quantum Cryptography (PQC)**: Internal communication is secured using next-generation cryptographic algorithms (Kyber-1024).
*   **Zero-Knowledge Proofs (ZKP)**:  Allows external audits to verify that an analysis was performed correctly without revealing the underlying source code.
*   **Isolated Inference**: Execution occurs within secure enclaves (SGX/TDX) to prevent tampering.

---

## Documentation Hub

For detailed technical specifications, please refer to:

*   **[Getting Started Guide](docs/GETTING_STARTED.md)**: Installation and first-run instructions.
*   **[System Architecture](docs/ARCHITECTURE.md)**: Deep dive into the Neuro-Symbolic design.
*   **[Technical Specification](docs/SPECIFICATION.md)**: Engineering roadmap and implementation details.
*   **[Operational Guide](docs/OPERATIONAL_GUIDE.md)**: Hardware specs and error codes.

---

## Setup & Usage

### Prerequisites
*   **Rust**: Stable toolchain (install via rustup).
*   **Python**: 3.10+ (for AI dependencies).
*   **System**: Linux or Windows (WSL2 recommended for performance).

### Quick Start

1.  **Clone the repository**
    ```bash
    git clone https://github.com/shanekizito/sentinel.git
    cd sentinel
    ```

2.  **Build the Core CLI**
    Navigate to the CLI directory and build the project relative to the root.
    ```bash
    cd Core/sentinel-cli
    cargo build --release
    ```
    *Note: This creates the executable in `target/release/`.*

3.  **Install AI Dependencies**
    Install the required Python packages for the neural engine.
    ```bash
    pip install torch torch-geometric watchdog
    ```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
