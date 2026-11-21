"""Default data / output roots for domain transfer. Override with env vars."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(os.environ.get("DOMAIN_DATA_ROOT") or "/root/data")
BOLT_PEPTIDES = Path(os.environ.get("BOLT_PEPTIDES") or DATA_ROOT / "cys/BOLT/optimization/peptides")
APEXGO_INIT = Path(os.environ.get("APEXGO_INIT") or DATA_ROOT / "cys/APEXGo/optimization/apex_oracle/init_data")
QUERY_ROOT = Path(os.environ.get("QUERY_ROOT") or DATA_ROOT / "cys/BOLT/optimization/query_plans")
ZAXIE_ROOT = Path(os.environ.get("ZAXIE_ROOT") or DATA_ROOT / "zaxie")
TASKSET_ROOT = Path(os.environ.get("TASKSET_ROOT") or DATA_ROOT / "cys/taskset_runs")
TASKSET_EVAL_DIR = Path(
    os.environ.get("TASKSET_EVAL_DIR")
    or DATA_ROOT / "cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/scripts/eval"
)
TASKSET_BASELINE_CACHE = Path(
    os.environ.get("TASKSET_BASELINE_CACHE")
    or DATA_ROOT / "cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/eval_runs/dl_test_15/task_baselines.json"
)

HISTORY_ROOT = REPO_ROOT / "history_data"
BENCH_ROOT = REPO_ROOT / "benchmark_data"
BLOCKDATA_ARCHIVE = Path(
    os.environ.get("BLOCKDATA_ARCHIVE") or "/root/data/cys/blockdata_archive"
)


def domain_history(domain: str) -> Path:
    return HISTORY_ROOT / f"domain_{domain}"


def domain_bench(domain: str) -> Path:
    return BENCH_ROOT / f"domain_{domain}"


def protein_bo80_bolt_root() -> Path:
    """Peptide BO outputs (bo80_bolt). Default: benchmark_data symlink -> blockdata_archive."""
    override = os.environ.get("PROTEIN_BO80_BOLT_ROOT")
    if override:
        return Path(override)
    local = domain_bench("protein") / "bo80_bolt"
    if local.exists() or local.is_symlink():
        return local
    return BLOCKDATA_ARCHIVE / "domain_protein" / "bo80_bolt"
