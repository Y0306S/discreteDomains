#!/usr/bin/env python3
"""Build task baseline cache (best_val per task) from TaskSet GCS NPZ learning curves.

Reads local mirror of gs://task_set_data/task_set_data/:
  {gcs_cache}/{task_name}/{family}_1k_10000_replica5.npz

Each NPZ contains ys with shape (num_optimizers, num_replicas, num_steps, 4)
where index 2 is VALID_OUTER loss (paper §3.3 baseline reference).

Per task, best_val = min VALID_OUTER across all grid optimizers, replicas, and steps.
"""

from typing import Dict, Optional, Tuple
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EVAL_DIR))

from precache_datasets_v2 import DEFAULT_TEST_DATA, load_jsonl  # noqa: E402
from run_taskset_dl_eval_optimized import find_task_seed  # noqa: E402
from taskset_paper_scoring import VALID_OUTER_IDX, save_baseline_cache  # noqa: E402

DEFAULT_GCS_CACHE = Path(
    "/root/data/cys/blockdata_archive/shansoround/LLMOptimizer_data_restore/extracted_disk_export_v5/gcs_cache"
)
DEFAULT_OUTPUT = BASE_DIR / "eval_runs/dl_test_15/task_baselines_from_gcs.json"

GRID_FAMILIES = [
    "adam1p_wide_grid",
    "adam4p_wide_grid",
    "adam6p_wide_grid",
    "adam8p_wide_grid",
    "nadamw_grid",
]
NPZ_SUFFIX = "_1k_10000_replica5.npz"


def collect_dl_test_task_names(test_data: Path) -> Dict[str, int]:
    """Map task_name -> first jsonl index in dl_test file."""
    mapping = {}  # type: Dict[str, int]
    for index, record in enumerate(load_jsonl(test_data)):
        user = json.loads(record["messages"][1]["content"])
        task_name = find_task_seed(user["task_family"], user["task"])
        if task_name and task_name not in mapping:
            mapping[task_name] = index
    return mapping


def best_val_from_npz(npz_path: Path) -> Tuple[float, int]:
    """Return (min finite valid_outer, num_optimizers) for one family file.

    Historical NPZ curves can contain -inf/+inf from numerically unstable grid
    runs. Use the best *finite* VALID_OUTER, not np.nanmin (which returns -inf
    if any cell is -inf).
    """
    data = np.load(npz_path, allow_pickle=True)
    ys = data["ys"]
    valid_outer = ys[:, :, :, VALID_OUTER_IDX]
    finite = valid_outer[np.isfinite(valid_outer)]
    if finite.size == 0:
        return float("nan"), int(ys.shape[0])
    positive = finite[finite > 0]
    if positive.size:
        return float(positive.min()), int(ys.shape[0])
    return float(finite.min()), int(ys.shape[0])


def best_val_for_task(task_dir: Path) -> Optional[dict]:
    global_best = float("inf")
    per_family = {}  # type: Dict[str, float]
    num_optimizers = 0
    files_found = 0

    for family in GRID_FAMILIES:
        npz_path = task_dir / f"{family}{NPZ_SUFFIX}"
        if not npz_path.is_file():
            continue
        files_found += 1
        fam_best, n_opt = best_val_from_npz(npz_path)
        if np.isfinite(fam_best):
            per_family[family] = fam_best
            num_optimizers += n_opt

    finite_family = list(per_family.values())
    if files_found == 0 or not finite_family:
        return None
    global_best = float(min(finite_family))

    return {
        "best_val": global_best,
        "num_grid_optimizers": num_optimizers,
        "num_grid_families": files_found,
        "per_family_min_valid_outer": per_family,
        "source": "gcs_cache_npz",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build baseline cache from GCS NPZ runs")
    parser.add_argument("--gcs-cache", type=Path, default=DEFAULT_GCS_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA)
    parser.add_argument(
        "--dl-test-only",
        action="store_true",
        help="Only include task_names referenced in dl_test JSONL (default: all GCS task dirs)",
    )
    args = parser.parse_args()

    if not args.gcs_cache.is_dir():
        raise SystemExit(f"GCS cache not found: {args.gcs_cache}")

    if args.dl_test_only:
        task_names = sorted(collect_dl_test_task_names(args.test_data).keys())
    else:
        task_names = sorted(p.name for p in args.gcs_cache.iterdir() if p.is_dir())

    tasks = {}  # type: Dict[str, dict]
    missing = []  # type: list
    t0 = time.monotonic()

    for i, task_name in enumerate(task_names):
        task_dir = args.gcs_cache / task_name
        if not task_dir.is_dir():
            missing.append(task_name)
            continue
        entry = best_val_for_task(task_dir)
        if entry is None:
            missing.append(task_name)
            continue
        tasks[task_name] = entry
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(task_names)} tasks")

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "gcs_cache": str(args.gcs_cache),
        "test_data": str(args.test_data) if args.dl_test_only else None,
        "dl_test_only": args.dl_test_only,
        "grid_families": GRID_FAMILIES,
        "npz_suffix": NPZ_SUFFIX,
        "valid_outer_index": VALID_OUTER_IDX,
        "tasks_built": len(tasks),
        "tasks_missing": len(missing),
        "elapsed_s": round(time.monotonic() - t0, 2),
    }
    save_baseline_cache(args.output, tasks, meta=meta)

    print(f"Built baseline cache: {args.output}")
    print(f"  tasks: {len(tasks)} ok, {len(missing)} missing")
    print(f"  elapsed: {meta['elapsed_s']}s")
    if missing[:5]:
        print(f"  sample missing: {missing[:5]}")


if __name__ == "__main__":
    main()
