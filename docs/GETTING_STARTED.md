# Getting Started with Sentinel

Welcome to the **Sentinel Sovereign Security Platform**. This guide will help you set up the environment, build the CLI, and run your first codebase analysis.

---

## 1. Prerequisites

Before you begin, ensure you have the following installed on your system:

### Required
- **Rust Toolchain (1.75+)**: Used to compile the core engine.
  - [Install Rust](https://www.rust-lang.org/tools/install)
- **Python (3.10+)**: Used for the Neural Training factory.
  - [Install Python](https://www.python.org/downloads/)

### Optional (For Hyper-Scale features)
- **PostgreSQL**: If you plan to use the persistent metadata store.
- **CUDA 12.x**: If you have an NVIDIA GPU and want to enable hardware accelerated training.

---

## 2. Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/shanekizito/sentinel.git
cd sentinel
```

### Step 2: Build the Core Engine
We compile the Rust CLI in release mode for maximum performance.

```bash
# Navigate to the CLI directory
cd Core/sentinel-cli

# Build the binary
cargo build --release

# (Optional) Move the binary to the root for easy access
# Windows (PowerShell)
copy target\release\sentinel.exe ..\..\sentinel.exe
# Linux/Mac
cp target/release/sentinel ../../sentinel
```

### Step 3: Install Python Dependencies
Install the required libraries for the AI components.

```bash
pip install torch torch-geometric torch-scatter watchdog psycopg2-binary
```

---

## 3. Your First Scan

Now that everything is installed, let's run a test ingestion on a local directory.

### 1. Ingest Code (Create Shards)
This command recursively scans a directory, parses the code, and creates "Binary Shards" optimized for AI training.

```bash
# Syntax: ./sentinel ingest --source <path-to-code> --output <path-to-shards>

./sentinel ingest --source ./Frontend --output ./data/shards
```

### 2. Verify Output
Check the `./data/shards` directory. You should see `.bin` files created for your source code.

---

## 4. Next Steps

- **Advanced Usage**: Read the [Operational Guide](OPERATIONAL_GUIDE.md) for distributed deployment.
- **Architecture**: Design deeper into the [System Architecture](ARCHITECTURE.md).
