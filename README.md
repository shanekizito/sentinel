# Sentinel Sovereign AI: The Omega Singularity

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Sovereignty: Absolute](https://img.shields.io/badge/Sovereignty-Absolute-red.svg)](https://sentinel.ai)
[![Architecture: Hyper-Scale](https://img.shields.io/badge/Architecture-Hyper--Scale-blueviolet.svg)](https://sentinel.ai)

**Sentinel** is a planetary-scale, autonomous security governance platform. It is designed to be the "Sovereign" AI—a system that does not merely assist but *governs* digital infrastructure with mathematical certainty and neural intuition.

It unifies **Neural Geometries (GCN/Transformer)** with **Symbolic Reasoning (Z3/SMT)** into a single, antifragile intelligence capable of ingesting trillions of lines of code and proving their security properties in real-time.

---

## 🚀 Key Features (Hyper-Scale Edition)

### 1. Neural Graph Differential Equations (Neural GDEs)
We have transcended discrete deep learning layers.
- **Continuous Depth**: The `InfinitySovereignGCN` uses **Neural ODEs** (backed by RK4 solvers) to model code evolution as a continuous dynamical system.
- **Infinite Context**: The `InfinitySovereignReflex` Transformer uses **Linear Attention (O(N))** and Dynamic RoPE to handle context windows of 100k+ tokens.
- **Adversarial Hardening**: The training engine features an integrated **FGSM Attack Loop**, forcing the model to learn robust features resistant to perturbation.

### 2. The Sovereign AGI ("Omega")
An autonomous agentic controller that lives on the server.
- **Live OODA Loop**: Real-time FileSystem monitoring (`watchdog`) and Telemetry ingestion.
- **Neuro-Symbolic Toolset**: Physically patches files, executes shell commands, and verifies logic using Z3—all guarded by **PQC Signatures**.
- **Cognitive Engine**: Implements a "Tree of Thoughts" reasoning process to plan complex multi-step security audits.

### 3. Universal Polyglot Ingestion
One engine for the entire stack.
- **15+ Enterprise Languages**: Rust, C, C++, Java, C#, Go, Python, TypeScript, Solidity, SQL, and more.
- **Data Ingestor**: Recursively scans monorepos, auto-detects languages, and compiles them into **Binary V5 Shards** (`.bin`) for high-performance training (`mmap`).

### 4. Production Infrastructure
- **Unified CLI**: A single binary (`sentinel`) to Ingest, Train, and Serve.
- **Elastic FSDP**: Distributed training that survives Spot Instance preemption (SIGTERM) and OOMs.
- **Real Database**: PostgreSQL connection pooling for metadata management.

---

## 🛠️ Installation

### Prerequisites
- Python 3.10+ (PyTorch, PyG, Watchdog)
- Rust 1.75+ (Cargo)
- PostgreSQL (Optional for Metadata)
- CUDA 12.x (For Hyper-Scale Training)

### Quick Start
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/shanekizito/sentinel.git
    cd sentinel
    ```

2.  **Build the Unified CLI**
    ```bash
    cd Core/sentinel-cli
    cargo build --release
    cp target/release/sentinel ../../sentinel.exe
    cd ../..
    ```

3.  **Install Python Dependencies**
    ```bash
    pip install torch torch-geometric torch-scatter watchdog psycopg2-binary
    ```

---

## 💻 Usage

The `sentinel` CLI is your control plane.

### 1. Ingest Data (Create Binary Shards)
Turn a source code directory into training data.
```bash
./sentinel ingest --source ./linux-kernel --output ./data/shards
```

### 2. Train the Brain (Distributed)
Ignite the Hyper-Scale Training Engine.
```bash
# Spawns Python FSDP Kernel
./sentinel train --data ./data/shards --gpus 8
```

### 3. Launch the AGI (Serve)
Start the Autonomous Guardian.
```bash
./sentinel serve --model ./checkpoints/omega_v6.pt --port 8001
```

### 4. Omega Benchmark
Run the synthetic stress test to verify system stability.
```bash
./sentinel bench --files 50000
```

---

## 🏗️ Architecture Overview

```mermaid
graph TD
    User[User / CLI] -->|Ingest| Ingestor[Data Ingestor (Rust)]
    Ingestor -->|Parse| Parser[Tree-Sitter (15 Langs)]
    Parser -->|Build| CPG[Sovereign Graph]
    CPG -->|Serialize| Shards[Binary V5 Shards (.bin)]
    
    Shards -->|Mmap| DataFactory[Data Factory (Python)]
    DataFactory -->|Batch| Trainer[Training Engine (Neural GDEs)]
    Trainer -->|Update| Weights[Model Checkpoints]
    
    Weights -->|Load| AGI[Sovereign AGI (Omega)]
    AGI -->|Reason| Cognition[Reflex Client]
    AGI -->|Act| Tools[Neuro-Symbolic Tools]
```

---

## 🛡️ Security Policy

*   **PQC Signing**: All AGI actions are cryptographically signed.
*   **Safe Mode**: Default sandbox prevents external network calls unless explicitly authorized.
*   **Formal Verification**: Critical logic patches are verified by the Z3 SMT solver before application.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Copyright © 2026 Sentinel Sovereign Systems.**
