# LLMOptimizer Migration Guide

## Exported Conda Environment Specs

This directory contains exported conda environment specifications for migrating the LLMOptimizer project to a new server.

### Files

| File | Description |
|------|-------------|
| `llamafactory_qwen35_environment.yml` | Conda env spec for LLaMA-Factory training (generation) |
| `llamafactory_qwen35_requirements.txt` | pip freeze for llamafactory_qwen35 |
| `llamafactory_qwen35_repo_environment.yml` | Conda env spec for LLaMA-Factory training (repo version) |
| `llamafactory_qwen35_repo_requirements.txt` | pip freeze for llamafactory_qwen35_repo |
| `openbox_eval_environment.yml` | Conda env spec for evaluation/inference |
| `openbox_eval_requirements.txt` | pip freeze for openbox_eval |
| `hpobench_environment.yml` | Conda env spec for HPOBench ML/NAS benchmarks |
| `hpobench_requirements.txt` | pip freeze for hpobench |
| `hpobench_101_environment.yml` | Conda env spec for NASBench-101 + Tabular |
| `hpobench_101_requirements.txt` | pip freeze for hpobench_101 |
| `taskset_eval_environment.yml` | Conda env spec for TaskSet benchmarks |
| `taskset_eval_requirements.txt` | pip freeze for taskset_eval |

### Important: Editable Installs

The following packages are installed in **editable mode** and depend on local paths:

| Package | Source Path | Used In Envs |
|---------|-------------|--------------|
| `llamafactory` | `/jizhicfs/pkuhetu/bcxu/LlamaFactory` | llamafactory_qwen35, llamafactory_qwen35_repo |
| `hpobench` | `/jizhicfs/pkuhetu/bcxu/HPOBench` | hpobench, hpobench_101 |

On the new server, you must:
1. Clone/copy the source directories to the new location
2. Re-install with `pip install -e <path>` after creating the conda envs

### Reconstructing Envs on New Server

```bash
# 1. Create conda env from spec (repeat for each env)
conda env create -f llamafactory_qwen35_environment.yml

# 2. Activate and install editable packages
conda activate llamafactory_qwen35
pip install -e /path/to/new/LlamaFactory

# 3. For hpobench envs
conda activate hpobench
pip install -e /path/to/new/HPOBench
pip install -e ".[xgboost,rf,lr,nn]"  # from HPOBench directory
```

### Path Replacements Required

All hardcoded paths in the codebase must be updated. Use the provided `migrate_paths.py` script.

Old paths → New paths:
- `/jizhicfs/pkuhetu/bcxu` → `<NEW_PROJECT_ROOT>`
- `/jizhicfs/johnnyslin/anaconda3` → `<NEW_CONDA_ROOT>`
