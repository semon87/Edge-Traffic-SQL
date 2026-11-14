# Track 2: Edge Natural Language & Text-to-SQL Query Engine

This folder contains the complete **Track 2** codebase for translating natural language traffic questions into validated SQLite queries, executing them against edge-extracted vehicle analytics, and generating concise human-readable answers.

Designed for edge deployment on the **NVIDIA Jetson Orin Nano (8GB)** alongside the Track 1 computer vision pipeline.

---

## Architecture Overview

```
User Query ("How many trucks moved down?")
   │
   ▼
[Schema & Enum Prompt Extractor] (< 450 tokens)
   │
   ▼
[Edge SLM: Qwen2.5-Coder-3B / 1.5B (4-bit)] (Ollama / llama-server)
   │
   ▼
[Safe SQL Guard & Parser] (Blocks DROP/DELETE/INSERT)
   │
   ├─► Success: [Read-Only SQLite Engine] (WAL mode)
   │                  │
   │                  ▼
   │          [Result Formatter & Answer Synthesizer]
   │
   └─► Failure: [Self-Correction Feedback Loop] (1-retry syntax recovery)
```

---

## File Structure

| File | Purpose |
| :--- | :--- |
| `sql_engine.py` | Safe read-only SQLite connector, execution timer, and WAL mode optimizer. |
| `schema.py` | Inspects database to dynamically extract DDL and categorical values (`vehicle_type`, `direction`, etc.). |
| `prompts.py` | Token-economical prompt templates, few-shot examples, and self-correction prompt builders. |
| `llm_client.py` | Standard-library-based HTTP client for Ollama, OpenAI-compatible servers, and offline Mock testing. |
| `pipeline.py` | End-to-end query orchestrator with automatic error recovery and answer synthesis. |
| `benchmark_queries.json` | 60 curated traffic benchmark queries across 3 difficulty tiers with ground-truth SQL. |
| `evaluate.py` | Academic benchmark runner calculating **Valid SQL Rate (VSR %)**, **Execution Accuracy (EX %)**, and Latency. |
| `cli.py` | Interactive terminal interface for live traffic querying. |

---

## Quick Start Guide

### 1. Offline / Mock Verification (No installation needed)
You can verify the entire pipeline right now:

```bash
# Run a single test query
python track2_query_engine/cli.py --provider mock --query "How many trucks were detected?" -v

# Verify all 60 ground-truth queries in SQLite
python track2_query_engine/evaluate.py --verify-gt

# Run evaluation benchmark in mock mode
python track2_query_engine/evaluate.py --provider mock
```

---

### 2. Live SLM Inference with Ollama

#### On your Development Laptop (Windows):
1. Download and install [Ollama for Windows](https://ollama.com/download).
2. Open a terminal and pull the recommended 3B model:
   ```bash
   ollama pull qwen2.5-coder:3b
   ```
   *(Or for ultra-low memory: `ollama pull qwen2.5-coder:1.5b`)*
3. Start the interactive CLI:
   ```bash
   python track2_query_engine/cli.py --provider ollama --model qwen2.5-coder:3b
   ```

#### On NVIDIA Jetson Orin Nano (Linux aarch64):
1. Install Ollama:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
2. Pull the model:
   ```bash
   ollama pull qwen2.5-coder:3b
   ```
3. Run the academic benchmark to generate metrics for your paper:
   ```bash
   python track2_query_engine/evaluate.py --provider ollama --model qwen2.5-coder:3b
   ```
   *Reports are saved automatically to `benchmark_reports/benchmark_summary.json` and `benchmark_reports/benchmark_results.csv`.*

---

## Edge Resource Allocation (Jetson Orin Nano 8GB)

| Component | Memory Footprint | Compute Target |
| :--- | :--- | :--- |
| **Track 1 (YOLO11n + ByteTrack)** | ~1.2 GB RAM | GPU (TensorRT FP16) |
| **Track 2 (Qwen2.5-Coder-3B Q4_K_M)** | ~2.6 GB RAM (with 2k ctx) | GPU / CPU Offload |
| **Linux OS & System Services** | ~1.4 GB RAM | ARM CPU |
| **Headroom Buffer** | ~2.2 GB RAM | Safe edge margin |
| **Total System RAM** | **7.4 GB** | **Zero OOM Risk** |

