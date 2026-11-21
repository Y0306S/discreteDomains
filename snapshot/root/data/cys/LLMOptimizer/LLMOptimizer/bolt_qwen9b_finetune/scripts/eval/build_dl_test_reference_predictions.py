#!/usr/bin/env python3
"""Build reference predictions JSONL: one shared optimizer config per resolvable dl_test task."""

from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EVAL_DIR))

from precache_datasets_v2 import DEFAULT_TEST_DATA, load_jsonl  # noqa: E402
from run_taskset_dl_eval_optimized import find_task_seed  # noqa: E402

OUT_DIR = BASE_DIR / "eval_runs/dl_test_15"
PRED_PATH = OUT_DIR / "dl_test_15_reference_predictions.jsonl"
MANIFEST_PATH = OUT_DIR / "eval_package_manifest.json"

SHARED_CONFIG = {
    "optimizer_family": "nadamw_grid",
    "hyperparameters": {
        "learning_rate": 0.0005,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-08,
        "use_nesterov": True,
        "adamw_weight_decay": 0.0,
        "l2_weight_decay": 1e-4,
        "warmup_fraction": 0.01,
        "min_learning_rate_mult": 1e-05,
        "constant_fraction": 0.5,
    },
}


def main() -> None:
    predictions = []
    unresolved = []
    seen_indices = set()

    for index, record in enumerate(load_jsonl(DEFAULT_TEST_DATA)):
        user = json.loads(record["messages"][1]["content"])
        task_name = find_task_seed(user["task_family"], user["task"])
        if not task_name:
            unresolved.append(
                {
                    "index": index,
                    "task_family": user.get("task_family"),
                    "reason": "task_not_found_in_task_set_configs",
                }
            )
            continue
        if index in seen_indices:
            continue
        seen_indices.add(index)
        predictions.append(
            {
                "index": index,
                "sample_id": 0,
                "model": "reference_nadamw_grid",
                "config": dict(SHARED_CONFIG),
                "task_name": task_name,
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with PRED_PATH.open("w", encoding="utf-8") as f:
        for row in sorted(predictions, key=lambda r: r["index"]):
            f.write(json.dumps(row) + "\n")

    manifest = {
        "test_data": str(DEFAULT_TEST_DATA),
        "baseline_cache": str(OUT_DIR / "task_baselines.json"),
        "predictions": str(PRED_PATH),
        "optimizer_config": SHARED_CONFIG,
        "tasks_with_predictions": len(predictions),
        "tasks_unresolved": len(unresolved),
        "unresolved_sample": unresolved[:10],
        "eval_command": (
            "python scripts/eval/run_taskset_dl_eval_optimized.py "
            "--test-data data/dl_test_15_modified.jsonl "
            f"--predictions {PRED_PATH.relative_to(BASE_DIR)} "
            f"--baseline-cache {OUT_DIR.joinpath('task_baselines.json').relative_to(BASE_DIR)} "
            "--output-json eval_runs/dl_test_15/my_eval_result.json"
        ),
        "dry_run_command": (
            "python scripts/eval/run_taskset_dl_eval_optimized.py "
            "--test-data data/dl_test_15_modified.jsonl "
            f"--predictions {PRED_PATH.relative_to(BASE_DIR)} "
            f"--baseline-cache {OUT_DIR.joinpath('task_baselines.json').relative_to(BASE_DIR)} "
            "--dry-run --output-json eval_runs/dl_test_15/dl_test_15_dryrun_report.json"
        ),
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(predictions)} predictions -> {PRED_PATH}")
    print(f"Unresolved tasks: {len(unresolved)} (losg/quadratic families)")
    print(f"Manifest -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
