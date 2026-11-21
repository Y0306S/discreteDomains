#!/usr/bin/env python3
"""Dry-run verification for cached TFDS datasets used by dl_test_15.

For each READY dataset (per audit_dl_test_datasets.py):
  1. TFDS load smoke: read one train example from local cache.
  2. Optional train_inner smoke: run one representative task for a few steps.

Each check runs in a fresh subprocess to avoid TF session conflicts.
Writes JSON under eval_runs/dl_test_15/dataset_verify/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
EVAL_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_SCRIPTS_DIR))

from audit_dl_test_datasets import audit_status, discover_dataset_tasks  # noqa: E402
from precache_datasets_v2 import (  # noqa: E402
    DATA_DIR,
    DEFAULT_TEST_DATA,
    NO_TFDS_FAMILIES,
    load_jsonl,
    normalize_dataset_name,
)

DEFAULT_OUTPUT = BASE_DIR / "eval_runs/dl_test_15/dataset_verify"
DEFAULT_OPTIMIZER = "adam1p_wide_grid_seed0"
CONDA_ACTIVATE = "/root/miniconda3/bin/activate"
CONDA_ENV = "taskset_eval"
TASK_SET_ROOT = Path("/root/data/cys/blockdata_archive/LLMOptimizer/google-research")
LOAD_WORKER = EVAL_SCRIPTS_DIR / "verify_load_worker.py"


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


def find_task_seed(task_family: str, target_task: dict) -> str | None:
    config_path = TASK_SET_ROOT / "task_set/tasks/configs" / f"{task_family}.json"
    if not config_path.exists():
        config_path = TASK_SET_ROOT / "task_set/tasks/configs/char_rnn_language_model.json"
    with config_path.open() as f:
        task_configs = json.load(f)
    for key, cfg_list in task_configs.items():
        if cfg_list[0] == target_task:
            return key
    return None


def run_load_smoke(dataset_name: str) -> dict:
    cmd = (
        f"source {CONDA_ACTIVATE} {CONDA_ENV} && "
        f"OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2 "
        f"python {LOAD_WORKER} {DATA_DIR} {json.dumps(dataset_name)}"
    )
    proc = subprocess.run(cmd, shell=True, executable="/bin/bash", capture_output=True, text=True)
    line = (proc.stdout or "").strip().splitlines()[-1] if proc.stdout else ""
    if line:
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            pass
    return {
        "status": "fail",
        "error": (proc.stderr or proc.stdout or "no output")[-2000:],
        "returncode": proc.returncode,
    }


def run_train_inner_smoke(
    task_name: str,
    *,
    training_steps: int,
    eval_every_n: int,
    optimizer_name: str,
    output_dir: Path,
    timeout_s: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = (
        f"source {CONDA_ACTIVATE} {CONDA_ENV} && "
        f"cd {TASK_SET_ROOT} && "
        f"TFDS_DATA_DIR={DATA_DIR} OMP_NUM_THREADS=4 python -m task_set.train_inner "
        f"--optimizer_name={optimizer_name} "
        f"--task_name={task_name} "
        f"--output_directory={output_dir.resolve()} "
        f"--training_steps={training_steps} "
        f"--eval_every_n={eval_every_n} "
        f"--replica=0"
    )
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env={**dict(**{"TFDS_DATA_DIR": str(DATA_DIR)}), **dict(__import__("os").environ)},
        )
        elapsed = round(time.monotonic() - t0, 2)
        ok = proc.returncode == 0
        return {
            "status": "ok" if ok else "fail",
            "returncode": proc.returncode,
            "elapsed_s": elapsed,
            "stderr_tail": (proc.stderr or "")[-1500:],
            "stdout_tail": (proc.stdout or "")[-1500:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "fail",
            "error": f"timeout after {timeout_s}s",
            "elapsed_s": round(time.monotonic() - t0, 2),
            "stderr_tail": (exc.stderr or "")[-1500:] if exc.stderr else None,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify cached TFDS datasets for dl_test_15")
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-load", action="store_true")
    parser.add_argument("--train-inner", action="store_true")
    parser.add_argument("--training-steps", type=int, default=20)
    parser.add_argument("--eval-every-n", type=int, default=20)
    parser.add_argument("--timeout-s", type=int, default=1200)
    parser.add_argument("--optimizer", default=DEFAULT_OPTIMIZER)
    parser.add_argument("--dataset", action="append", default=None)
    args = parser.parse_args()

    ds_counts, _ = discover_dataset_tasks(args.test_data)
    representatives = pick_representative_tasks(args.test_data)
    datasets = sorted(ds_counts.keys())
    if args.dataset:
        want = {normalize_dataset_name(d) for d in args.dataset}
        datasets = [d for d in datasets if d in want]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_output = args.output_dir / "train_inner_outputs"

    results: list[dict] = []
    load_ok = load_fail = eval_ok = eval_fail = skipped = 0

    print("=" * 72)
    print(f"Dataset verify | cache={DATA_DIR}")
    print(f"load={'on' if not args.skip_load else 'off'} | train_inner={'on' if args.train_inner else 'off'}")
    print("=" * 72)

    for ds in datasets:
        status, path = audit_status(ds)
        entry: dict = {
            "dataset": ds,
            "audit_status": status,
            "cache_path": str(path) if path else None,
            "tasks_in_jsonl": ds_counts[ds],
        }
        rep = representatives.get(ds)
        if rep:
            entry["sample_task_index"] = rep[0]

        if status != "READY":
            entry["overall"] = "skipped"
            skipped += 1
            print(f"[SKIP] {ds} ({status})")
            results.append(entry)
            continue

        print(f"\n[{ds}]")

        if not args.skip_load:
            load_result = run_load_smoke(ds)
            entry["load"] = load_result
            if load_result.get("status") == "ok":
                load_ok += 1
                print(f"  load: OK ({load_result.get('elapsed_s')}s)")
            else:
                load_fail += 1
                entry["overall"] = "fail"
                print(f"  load: FAIL — {load_result.get('error', '')[:200]}")
                results.append(entry)
                continue

        if args.train_inner:
            if not rep:
                entry["eval"] = {"status": "fail", "error": "no representative task"}
                eval_fail += 1
                entry["overall"] = "fail"
                results.append(entry)
                continue
            user = json.loads(rep[1]["messages"][1]["content"])
            task_name = find_task_seed(user["task_family"], user["task"])
            if not task_name:
                entry["eval"] = {"status": "fail", "error": "task_not_found"}
                eval_fail += 1
                entry["overall"] = "fail"
                results.append(entry)
                continue
            eval_result = run_train_inner_smoke(
                task_name,
                training_steps=args.training_steps,
                eval_every_n=args.eval_every_n,
                optimizer_name=args.optimizer,
                output_dir=train_output,
                timeout_s=args.timeout_s,
            )
            eval_result["task_name"] = task_name
            eval_result["task_family"] = user["task_family"]
            entry["eval"] = eval_result
            if eval_result.get("status") == "ok":
                eval_ok += 1
                print(f"  eval: OK task={task_name} ({eval_result.get('elapsed_s')}s)")
            else:
                eval_fail += 1
                print(f"  eval: FAIL — {eval_result.get('error') or eval_result.get('stderr_tail', '')[:200]}")

        entry["overall"] = "ok"
        results.append(entry)

    summary = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "cache_dir": str(DATA_DIR),
        "train_inner": args.train_inner,
        "datasets_total": len(datasets),
        "audit_skipped": skipped,
        "load_ok": load_ok,
        "load_fail": load_fail,
        "eval_ok": eval_ok,
        "eval_fail": eval_fail,
        "overall_ok": sum(1 for r in results if r.get("overall") == "ok"),
    }
    report = {"summary": summary, "results": results}
    latest_path = args.output_dir / "verify_report_latest.json"
    with latest_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 72)
    print(f"Load: {load_ok} ok, {load_fail} fail")
    if args.train_inner:
        print(f"Eval: {eval_ok} ok, {eval_fail} fail")
    print(f"Overall ok: {summary['overall_ok']}/{len(datasets) - skipped}")
    print(f"Report: {latest_path}")
    print("=" * 72)
    if load_fail or eval_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
