#!/usr/bin/env python3
"""Run a 1-step train_inner smoke test for previously failed dl_test_15 tasks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "scripts" / "eval"))

from run_taskset_dl_eval_optimized import (  # noqa: E402
    CONDA_ACTIVATE,
    CONDA_ENV,
    TASK_SET_ROOT,
    create_optimizer_config,
    find_task_seed,
    normalize_pred_config,
    register_custom_optimizer,
)
from taskset_paper_scoring import get_task_best_val, load_baseline_cache  # noqa: E402

DEFAULT_TEST_DATA = BASE_DIR / "data/dl_test_15_modified.jsonl"
DEFAULT_BASELINE = BASE_DIR / "eval_runs/dl_test_15/task_baselines.json"
DEFAULT_EVAL_JSON = BASE_DIR / "eval_runs/dl_test_15/qwen_base_lora_sft_3_dl_tier5_cpu_eval.json"
DEFAULT_PREDICTIONS = BASE_DIR / "eval_runs/dl_test_15/qwen_base_lora_sft_3_dl_test_predictions.jsonl"
DATA_DIR = "/root/data/cys/tensorflow_datasets"


def load_gold_records(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            records.append(json.loads(line))
    return records


def failed_indices_from_eval(eval_path: Path) -> list[int]:
    from run_taskset_dl_eval_optimized import load_json

    data = load_json(eval_path)
    failed = []
    for row in data.get("results", []):
        if row.get("status") in ("all_failed", "task_not_found"):
            failed.append(int(row["index"]))
    return sorted(failed)


def run_one_step(
    task_name: str,
    optimizer_name: str,
    output_dir: Path,
    *,
    training_steps: int,
    eval_every_n: int,
    timeout_s: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = (
        f"source {CONDA_ACTIVATE} {CONDA_ENV} && "
        f"cd {TASK_SET_ROOT} && "
        f"TFDS_DATA_DIR={DATA_DIR} OMP_NUM_THREADS=2 python -m task_set.train_inner "
        f"--optimizer_name={optimizer_name} "
        f"--task_name={task_name} "
        f"--output_directory={output_dir.resolve()} "
        f"--training_steps={training_steps} "
        f"--eval_every_n={eval_every_n} "
        f"--replica=0"
    )
    t0 = time.monotonic()
    proc = subprocess.run(
        cmd,
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    elapsed = round(time.monotonic() - t0, 2)
    ok = proc.returncode == 0
    return {
        "status": "ok" if ok else "fail",
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "stderr_tail": (proc.stderr or "")[-2000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="1-step dry run for failed dl_test tasks")
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA)
    parser.add_argument("--eval-json", type=Path, default=DEFAULT_EVAL_JSON)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--baseline-cache", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "eval_runs/dl_test_15/failed_task_dryrun")
    parser.add_argument("--task-indices", type=int, nargs="*")
    parser.add_argument("--training-steps", type=int, default=1)
    parser.add_argument("--eval-every-n", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument(
        "--skip-baseline-check",
        action="store_true",
        help="Only verify train_inner runs; do not require baseline cache entries.",
    )
    args = parser.parse_args()

    gold_records = load_gold_records(args.test_data)
    baseline_cache = load_baseline_cache(args.baseline_cache)
    preds_by_index: dict[int, list[dict]] = {}
    with args.predictions.open() as f:
        for line in f:
            row = json.loads(line)
            preds_by_index.setdefault(int(row["index"]), []).append(row)

    indices = args.task_indices or failed_indices_from_eval(args.eval_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    opt_dir = args.output_dir / "custom_optimizers"
    opt_dir.mkdir(parents=True, exist_ok=True)
    train_dir = args.output_dir / "train_outputs"

    results = []
    ok = fail = 0
    for task_index in indices:
        gold = gold_records[task_index]
        user_msg = json.loads(gold["messages"][1]["content"])
        task_family = user_msg["task_family"]
        preds = preds_by_index.get(task_index, [])
        pred = preds[0] if preds else {}
        pred_config = normalize_pred_config(pred)
        entry = {
            "index": task_index,
            "task_family": task_family,
            "status": "setup_fail",
        }

        if not pred_config:
            entry["error"] = "no_prediction"
            results.append(entry)
            fail += 1
            continue

        task_name = find_task_seed(task_family, user_msg["task"])
        if not task_name:
            entry["error"] = "task_not_found"
            results.append(entry)
            fail += 1
            continue

        if not args.skip_baseline_check and get_task_best_val(task_name, baseline_cache) is None:
            entry["error"] = "missing_baseline"
            entry["task_name"] = task_name
            results.append(entry)
            fail += 1
            continue

        opt_seed = task_index * 100
        opt_name, opt_path = create_optimizer_config(
            pred_config["optimizer_family"],
            pred_config["hyperparameters"],
            opt_seed,
            opt_dir,
        )
        try:
            register_custom_optimizer(opt_path, opt_seed, force=True)
        except ValueError as exc:
            entry["error"] = f"optimizer_register_failed: {exc}"
            entry["task_name"] = task_name
            results.append(entry)
            fail += 1
            continue

        print(f"[dryrun] task {task_index}: {task_name} / {opt_name}")
        run_result = run_one_step(
            task_name,
            opt_name,
            train_dir,
            training_steps=args.training_steps,
            eval_every_n=args.eval_every_n,
            timeout_s=args.timeout_s,
        )
        entry.update(
            {
                "task_name": task_name,
                "optimizer_family": pred_config["optimizer_family"],
                "opt_name": opt_name,
                "status": run_result["status"],
                "returncode": run_result.get("returncode"),
                "elapsed_s": run_result.get("elapsed_s"),
            }
        )
        if run_result["status"] != "ok":
            entry["stderr_tail"] = run_result.get("stderr_tail")
            fail += 1
        else:
            ok += 1
        results.append(entry)

    report = {
        "summary": {
            "total": len(indices),
            "ok": ok,
            "fail": fail,
            "training_steps": args.training_steps,
        },
        "results": results,
    }
    out_path = args.output_dir / "dryrun_report.json"
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    print(f"\nDry run complete: {ok}/{len(indices)} ok, {fail} fail")
    print(f"Report: {out_path}")
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
