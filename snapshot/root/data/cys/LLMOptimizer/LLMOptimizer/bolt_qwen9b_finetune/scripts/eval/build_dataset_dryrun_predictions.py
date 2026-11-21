#!/usr/bin/env python3
"""Build one-prediction-per-READY-dataset JSONL for dry-run verification."""

from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EVAL_DIR))

from audit_dl_test_datasets import audit_status, discover_dataset_tasks  # noqa: E402
from precache_datasets_v2 import DEFAULT_TEST_DATA, NO_TFDS_FAMILIES, load_jsonl, normalize_dataset_name  # noqa: E402
from run_taskset_dl_eval_optimized import find_task_seed  # noqa: E402

OUT_DIR = BASE_DIR / "eval_runs/dl_test_15/dataset_verify"

# Single fixed optimizer config (from gemini_3.1_pro_preview task 0).
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


def pick_representative_tasks(test_path: Path) -> dict[str, tuple[int, dict]]:
    picked: dict[str, tuple[int, dict]] = {}
    for index, record in enumerate(load_jsonl(test_path)):
        user = json.loads(record["messages"][1]["content"])
        if user.get("task_family") in NO_TFDS_FAMILIES:
            continue
        ds_list = user.get("task", {}).get("dataset")
        if not ds_list or not ds_list[0]:
            continue
        ds = normalize_dataset_name(ds_list[0])
        if ds not in picked:
            picked[ds] = (index, record)
    return picked


def main() -> None:
    test_path = DEFAULT_TEST_DATA
    ds_counts, _ = discover_dataset_tasks(test_path)
    representatives = pick_representative_tasks(test_path)

    predictions: list[dict] = []
    baseline_tasks: dict[str, dict] = {}
    skipped: list[dict] = []

    for ds in sorted(ds_counts.keys()):
        status, _ = audit_status(ds)
        if status != "READY":
            skipped.append({"dataset": ds, "audit_status": status})
            continue
        rep = representatives.get(ds)
        if not rep:
            skipped.append({"dataset": ds, "error": "no representative task"})
            continue
        task_index, record = rep
        user = json.loads(record["messages"][1]["content"])
        task_name = find_task_seed(user["task_family"], user["task"])
        if not task_name:
            skipped.append({"dataset": ds, "task_index": task_index, "error": "task_not_found"})
            continue
        predictions.append(
            {
                "index": task_index,
                "sample_id": 0,
                "model": "dataset_verify_stub",
                "config": dict(SHARED_CONFIG),
            }
        )
        baseline_tasks[task_name] = {
            "best_val": 0.5,
            "num_grid_optimizers": 0,
            "stub": True,
            "dataset": ds,
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = OUT_DIR / "dataset_verify_predictions.jsonl"
    baseline_path = OUT_DIR / "dataset_verify_stub_baselines.json"
    manifest_path = OUT_DIR / "dataset_verify_manifest.json"

    with pred_path.open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row) + "\n")

    with baseline_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "note": "stub baselines for dataset dry-run verify only",
                    "optimizer_config": SHARED_CONFIG,
                },
                "tasks": baseline_tasks,
            },
            f,
            indent=2,
        )

    manifest = {
        "predictions": str(pred_path),
        "baseline_cache": str(baseline_path),
        "ready_datasets": len(predictions),
        "skipped": skipped,
        "task_indices": [p["index"] for p in predictions],
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(predictions)} predictions -> {pred_path}")
    print(f"Wrote stub baselines -> {baseline_path}")
    if skipped:
        print(f"Skipped {len(skipped)} datasets (not READY or no task)")


if __name__ == "__main__":
    main()
