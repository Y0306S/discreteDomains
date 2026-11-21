# LLMOptimizer - Complete Documentation

## Query and Peptide Campaign Source

Campaign code and sanitized deployment notes for the query and peptide FSBO experiments are collected in [experiments/README.md](experiments/README.md). The bundle includes the active runners, VAE/FSBO rollback implementation, training scripts, and selected shared/upstream source. Model weights, benchmark corpora, SQL databases, and secrets remain external; use the documented templates and preflight checks before deploying.

> Unified documentation for LLM-driven Neural Architecture Search (NAS) and Hyperparameter Optimization (HPO) benchmarks.
> 
> **Last Updated:** 2026-04-21

  ssh your-server
  curl http://127.0.0.1:11434/v1/models  # Should show models
  ps aux | grep ollama  # Should show ollama serve

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Quick Start](#3-quick-start)
4. [Model Configuration](#4-model-configuration)
5. [Benchmarks](#5-benchmarks)
6. [Running Benchmarks](#6-running-benchmarks)
7. [Local Models Setup](#7-local-models-setup)
8. [Results & Verification](#8-results--verification)
9. [Troubleshooting](#9-troubleshooting)
10. [Project Status](#10-project-status)

---


### Two Benchmark Categories

| Category | Benchmarks | Agent | Environment | Python |
|----------|------------|-------|-------------|--------|
| **ML Benchmarks** | NN, RF, XGBOOST, LR, SVM | `hpo_agent.py` | `hpobench` | 3.8 |
| **NAS Benchmarks** | NASCifar10A, FCNet tabular, NASBench-201 | `nas_agent.py` | `hpobench_101` | 3.7 |

**Key Difference:**
- ML benchmarks use `hpo_agent.py` with simpler `recommend_one()` approach (always n=1)
- NAS benchmarks use `nas_agent.py` with batch recommendation support (`recommend_batch()`)

---

## 2. Architecture

### Data Flow

```
Entry Script (run_batch_*.py)
      │
      ▼
benchmark_runner.run_benchmark()
      │
      ├─ Step 1: Init HPOBench benchmark (loads OpenML dataset)
      ├─ Step 2: Extract ConfigSpace (real param names + ranges)
      ├─ Step 3: Profile data (landmarkers)
      ├─ Step 4: LLM call (generate warm-start config)
      ├─ Step 5: Map recommendations → ConfigSpace dict
      ├─ Step 6: Set max fidelity
      ├─ Step 7: Evaluate default config (baseline)
      ├─ Step 8: Evaluate warm-start config
      └─ Step 9: Report comparison metrics
```

### Core Components

| File | Purpose |
|------|---------|
| `benchmark_runner.py` | Central module — all shared logic, data sanitisation, LLM orchestration |
| `hpo_agent.py` | HPO agent — profiler adapter, prompt construction, structured output |
| `nas_agent.py` | NAS agent — batch recommendation support with timeout handling |
| `profiler.py` | DataProfiler — 3 landmarker probes (LR, KNN, DT) |
| `model_registry.py` | Model configurations for API, local, and Ollama models |

---

## 3. Quick Start

### Environment Setup

```bash
# ML Benchmarks (NN, RF, XGBOOST, LR, SVM)
conda activate hpobench
cd /jizhicfs/pkuhetu/bcxu/LLMOptimizer
python scripts/run_batch_ml.py --resume

# NAS-101 + FCNet Tabular
conda activate hpobench_101
python scripts/run_batch_nas_container.py --resume

# NASBench-201
conda activate hpobench
python scripts/run_batch_nas_201.py --resume
```

### Run Specific Models

```bash
# API models only
python scripts/run_batch_ml.py --models "gpt-5.4,gpt-5-mini,gemini-3.1-pro"

# Local models (llama-server auto-managed)
export LOCAL_MODEL_ACTIVE=llama-3.2-3b
python scripts/run_batch_ml.py --models "llama-3.2-3b"

# Ollama vision models (CRITICAL: export first!)
export OLLAMA_ENABLED=1
python scripts/run_batch_nas_container.py --models "llama-3.2-vision-8k"
```

---

## 4. Model Configuration

### API Models (OpenAI-compatible)

| Model | Endpoint | Timeout |
|-------|----------|---------|
| gpt-5.4, gpt-5-mini, gpt-3.5 | `https://api.openai-hk.com/v1` | 180s |
| gemini-3.1-pro, gemini-3-flash | `https://api.openai-hk.com/v1` | 180s |
| claude-sonnet-4-6 | `https://api.openai-hk.com/v1` | 180s |
| deepseek-v3.2, deepseek-r1 | `https://api.openai-hk.com/v1` | 180s |

### Local Models (llama-server)

| Model Key | Model File | GPU Memory |
|-----------|------------|------------|
| `llama-3.2-3b` | Llama-3.2-3B-Instruct-Q4_K_M.gguf | ~4GB |
| `llama-3.2-11b` | Llama-3.2-11B-Vision-Instruct.Q4_K_M.gguf | ~12GB |
| `qwen-3.5-4b` | Qwen3.5-4B-Q4_K_M.gguf | ~5GB |
| `qwen-3.5-9b` | Qwen_Qwen3.5-9B-Q4_K_M.gguf | ~10GB |

**Activation:** `export LOCAL_MODEL_ACTIVE=<model_key>`
**Endpoint:** `http://127.0.0.1:8080/v1`

### Ollama Models (Vision Support)

| Model Key | Ollama Name | Context | GPU Memory |
|-----------|-------------|---------|------------|
| `llama-3.2-vision-8k` | llama3.2-vision-8k | 8192 | ~12GB |
| `llama-3.2-vision` | llama3.2-vision | 128K | ~34GB |

**⚠️ CRITICAL:** `OLLAMA_ENABLED=1` must be exported BEFORE Python import:

```bash
# ✅ CORRECT
export OLLAMA_ENABLED=1
python scripts/run_batch_nas_container.py --models llama-3.2-vision-8k

# ❌ WRONG - module loads before env var is set
OLLAMA_ENABLED=1 python scripts/run_batch_nas_container.py
```

**Why:** `model_registry.py` loads configs at module import time based on `OLLAMA_ENABLED`.

---

## 5. Benchmarks

### ML Benchmarks (hpobench env)

| Benchmark | Model | Task |
|-----------|-------|------|
| NN | Neural Network | OpenML classification |
| RF | Random Forest | OpenML classification |
| XGBOOST | XGBoost | OpenML classification |
| LR | Logistic Regression | OpenML classification |
| SVM | Support Vector Machine | OpenML classification |

### NAS-101 + FCNet Tabular (hpobench_101 env)

| Benchmark | Type | Description |
|-----------|------|-------------|
| NASCifar10A | NASBench-101 | Neural architecture search for CIFAR-10 |
| SliceLocalization | FCNet | Medical imaging |
| ProteinStructure | FCNet | Bioinformatics |
| NavalPropulsion | FCNet | Engineering |
| ParkinsonsTelemonitoring | FCNet | Healthcare |

**NASBench-101 Data:**
```bash
mkdir -p ~/nasbench_full
wget https://storage.googleapis.com/nasbench/nasbench_full.tfrecord \
  -O ~/nasbench_full/nasbench_full.tfrecord
```

### NASBench-201 (hpobench env)

| Benchmark | Datasets |
|-----------|----------|
| Cifar10NasBench201 | CIFAR-10 |
| Cifar100NasBench201 | CIFAR-100 |
| ImageNetNasBench201 | ImageNet16-120 |

### FCNet Configuration Space

| Parameter | Valid Values |
|-----------|-------------|
| `activation_fn_1/2` | relu, tanh |
| `batch_size` | 8, 16, 32, 64 |
| `dropout_1/2` | 0.0, 0.3, 0.6 |
| `init_lr` | 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1 |
| `lr_schedule` | const, cosine |
| `n_units_1/2` | 16, 32, 64, 128, 256, 512 |

**Total configurations:** 62,208 (all pre-computed in HDF5)

---

## 6. Running Benchmarks

### Execution Scripts

| Script | Purpose | Environment |
|--------|---------|-------------|
| `scripts/run_batch_ml.py` | ML benchmarks (NN, RF, XGB, LR, SVM) | hpobench |
| `scripts/run_batch_nas_container.py` | NAS-101 + FCNet tabular | hpobench_101 |
| `scripts/run_batch_nas_201.py` | NASBench-201 | hpobench |

### Common Options

```bash
# Resume incomplete runs
python scripts/run_batch_*.py --resume

# Run specific models
python scripts/run_batch_*.py --models "gpt-5.4,gpt-5-mini"

# Resume with specific models
python scripts/run_batch_*.py --resume --models "llama-3.2-3b"
```

### Complete Examples

```bash
# 1. ML Benchmarks - All API models
conda activate hpobench
python scripts/run_batch_ml.py --resume

# 2. ML Benchmarks - Local model
conda activate hpobench
export LOCAL_MODEL_ACTIVE=llama-3.2-3b
python scripts/run_batch_ml.py --resume --models "llama-3.2-3b"

# 3. NAS-101 - Ollama vision model
conda activate hpobench_101
export OLLAMA_ENABLED=1
python scripts/run_batch_nas_container.py --resume --models "llama-3.2-vision-8k"

# 4. NAS-101 - Specific models
conda activate hpobench_101
python scripts/run_batch_nas_container.py --resume \
  --models "gpt-5.4,llama-3.2-3b,llama-3.2-vision-8k"
```

---

## 7. Local Models Setup

### Ollama (Recommended for Vision)

```bash
# 1. Create 8K context variant (one-time setup)
cat > Modelfile << 'EOF'
FROM llama3.2-vision:latest
PARAMETER num_ctx 8192
EOF
ollama create llama3.2-vision-8k -f Modelfile

# 2. Verify GPU usage
ollama ps
# Should show: llama3.2-vision-8k    100% GPU

# 3. Run benchmarks
export OLLAMA_ENABLED=1
python scripts/run_batch_nas_container.py --resume --models "llama-3.2-vision-8k"
```

**Performance Comparison:**

| Context | GPU Memory | Response Time | Status |
|---------|-----------|---------------|--------|
| 128K | ~34GB | ~105s (CPU) | ❌ Too slow |
| 8K | ~12GB | ~7s (GPU) | ✅ Recommended |

### llama.cpp (Standard Models)

```bash
# Terminal 1: Start llama-server
cd /jizhicfs/pkuhetu/bcxu/llama.cpp
./build/bin/llama-server --host 127.0.0.1 --port 8080 \
  --model ./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  -ngl 99

# Terminal 2: Run benchmarks
cd /jizhicfs/pkuhetu/bcxu/LLMOptimizer
export LOCAL_MODEL_ACTIVE=llama-3.2-3b
python scripts/run_batch_ml.py --resume --models "llama-3.2-3b"
```

---

## 8. Results & Verification

### Results Location

```
experiment_runs/results/
├── <benchmark_name>/
│   └── <model_name>/
│       └── nas_*.json          # Individual run artifacts
├── nas_101_batch_summary_*.json
├── nas_101_best_by_benchmark_*.csv
└── nas_101_llm_scorecard_*.csv
```

### Verification Script

```bash
cd /jizhicfs/pkuhetu/bcxu/LLMOptimizer
python -c "
import os
os.environ['OLLAMA_ENABLED'] = '1'
from src.config.model_registry import MODEL_CONFIGS
print('Configured models:', len(MODEL_CONFIGS))
for m in MODEL_CONFIGS.keys():
    print(f'  - {m}')
"
```

### Expected Results Summary

| Category | Benchmarks | Models | Total Expected |
|----------|------------|--------|----------------|
| ML | 5 (NN, RF, XGB, LR, SVM) | 12 | 300 |
| NAS-101 | 5 (NASCifar10A + 4 FCNet) | 12 | 60 |
| NAS-201 | 3 (Cifar10/100, ImageNet) | 12 | 36 |
| **Total** | | | **396** |

---

## 9. Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `429 Resource Exhausted` | Timeout increased to 180s; retry logic with backoff |
| Ollama not using GPU | Create 8K context variant; check `nvidia-smi` |
| llama-server won't start | Check port 8080: `lsof -i :8080; pkill -f llama-server` |
| TensorFlow slow init | Normal for TF 1.15.0 (~60s), wait for it |
| `OLLAMA_ENABLED` not working | Must export BEFORE running Python |
| Disk quota exceeded | Cannot write new files; use `/tmp/` for logs |

### Verification Commands

```bash
# Check Ollama status
curl http://localhost:11434/api/tags
ollama ps

# Check llama-server
curl http://localhost:8080/v1/models

# Test Ollama API directly
curl http://localhost:11434/v1/chat completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2-vision-8k", "messages": [{"role":"user","content":"hi"}]}'
```

---

## 10. Project Status

### Recently Completed

- ✅ Fixed Gemini timeout: Increased from 60s to 180s
- ✅ Fixed llama-server startup: Added `.gguf` extension
- ✅ Fixed Ollama integration: Documented `OLLAMA_ENABLED` ordering
- ✅ Created 8K context vision model for GPU performance
- ✅ FCNet vision benchmark completed for all 5 benchmarks

### Known Limitations

1. **SVM benchmark** - No results yet (all models)
2. **Cifar10NasBench201** - Entire benchmark missing
3. **llama-3.2-11b** - Missing from all benchmarks
4. **llama-3.2-vision** (128K) - Missing (slow without GPU)

### File Overview

| File | Purpose |
|------|---------|
| `src/benchmarks/benchmark_runner.py` | Central orchestration module |
| `src/agents/hpo_agent.py` | HPO agent with 180s timeout |
| `src/agents/nas_agent.py` | NAS agent with batch recommendations |
| `scripts/profiler.py` | Data profiling with landmarkers |
| `src/config/model_registry.py` | Model configuration registry |
| `scripts/run_batch_ml.py` | ML benchmark runner |
| `scripts/run_batch_nas_container.py` | NAS-101 + FCNet runner |
| `scripts/run_batch_nas_201.py` | NASBench-201 runner |

---

## Appendix: Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_ENABLED` | `0` | Set to `1` to enable Ollama models |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434/v1` | Ollama API endpoint |
| `LOCAL_MODEL_ACTIVE` | `llama-3.2-11b` | Active llama.cpp model |
| `OPENAI_BASE_URL` | `http://127.0.0.1:8080/v1` | llama.cpp server endpoint |
| `OPENAI_HK_API_KEY` | - | API key for OpenAI-HK proxy |

---

---

## 11. Session History & Key Learnings

### Critical Fixes Applied (2026-04-21)

#### 1. Gemini 3.1 Pro Timeout Fix
**Problem:** Gemini API calls were timing out after 60s, causing failures.

**Solution:** Increased timeout to 180s in both agents:
```python
# nas_agent.py and hpo_agent.py
self.client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=180.0,  # Changed from 60.0
)
```

#### 2. llama-server Model Path Fix
**Problem:** llama-server failed to start because model path was missing `.gguf` extension.

**Solution:** Updated model path construction to include extension.

#### 3. Ollama Integration Critical Ordering
**Problem:** Ollama models weren't being recognized even with `OLLAMA_ENABLED=1`.

**Root Cause:** `model_registry.py` loads configurations at **module import time**. Setting the env var on the same command line doesn't work because Python imports the module before the variable is set.

**Solution:**
```bash
# CORRECT - Export in separate command
export OLLAMA_ENABLED=1
python scripts/run_batch_nas_container.py --models llama-3.2-vision-8k

# WRONG - Module imports before env var is active
OLLAMA_ENABLED=1 python scripts/run_batch_nas_container.py
```

### FCNet Vision Benchmark Results (Completed)

Llama 3.2 Vision 8K successfully completed all 5 FCNet benchmarks:

| Benchmark | Best Config | Error | Accuracy |
|-----------|-------------|-------|----------|
| NASCifar10A | Config 2 | 0.1335 | 86.65% |
| SliceLocalization | Default | 0.0027 | 99.73% |
| ProteinStructure | Config 2 | 0.3295 | 67.05% |
| NavalPropulsion | Default | 0.0015 | 99.85% |
| ParkinsonsTelemonitoring | Config 2 | 0.0650 | 93.50% |

**Performance:** ~7s response time with 8K context on GPU (vs ~105s with 128K context on CPU).

#### ML vs NAS Agent Differences
```python
# HPO Agent (ML benchmarks) - Simple approach
class HPOAgent:
    def recommend(self, ...):
        # Always uses n=1, makes 3 separate calls for 3 recommendations
        return self.recommend_one(...)

# NAS Agent (NAS benchmarks) - Batch support
class TabularNASAgent:
    def recommend_batch(self, ..., n=1):
        # Supports n>1 with fallback to sequential calls
        # Has 180s timeout for Gemini models
```

#### FCNet vs NASBench-101 Validity

| Aspect | FCNet Tabular | NASBench-101 |
|--------|--------------|--------------|
| Validity Rules | Simple parameter matching | Complex DAG validation |
| Invalid Configs | Rejected by ConfigSpace | Return "failure" with accuracy=0 |
| Database Size | 62,208 configurations | ~423,000 architectures |
| Lookup Method | Direct HDF5 key lookup | Graph validation + query |

#### Why Ollama for Vision Models
- llama.cpp doesn't support the `mllama` architecture required by Llama 3.2 Vision
- Ollama provides native vision model support
- 8K context variant essential for GPU performance (12GB vs 34GB)

### Disk Space Issue
**Status:** 230T/230T quota exceeded

**Workaround:** Cannot write new documentation files to project directory. Using `/tmp/` for logs and temporary files.

---
