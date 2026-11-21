#!/usr/bin/env python3
"""Fix best_val in task_baselines.json and re-score eval results from cached curves.

When GCS baseline cache has -inf for failed grid families, best_val was wrongly set
to -inf. Use min of finite per_family_min_valid_outer instead, then re-score tasks
from saved train_outputs result files (no retraining).
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
BASE_DIR = EVAL_DIR.parents[1]
sys.path.insert(0, str(EVAL_DIR))

from taskset_paper_scoring import (  # noqa: E402
    aggregate_replica_scores,
    finite_best_val,
    get_task_best_val,
    load_json,
    save_json,
    score_replica_curve,
)

DEFAULT_BASELINES = BASE_DIR / "eval_runs/dl_test_15/task_baselines.json"
DEFAULT_TRAINING_STEPS = 10_000

TIER5_MODELS = [
    "qwen_base_lora_sft_3_dl",
    "qwen_base_lora_sft_4_dl",
    "qwen_base_lora_sft_5_dl",
    "qwen_base_lora_sft_7_dl",
    "gemini_3.1_pro_preview",
]


def patch_baseline_cache(path: Path, *, dry_run: bool = False) -> list[tuple[str, float, float]]:
    payload = load_json(path)
    tasks = payload.get("tasks", payload)
    fixes: list[tuple[str, float, float]] = []

    for name, entry in tasks.items():
        if not isinstance(entry, dict):
            continue
        per = entry.get("per_family_min_valid_outer") or {}
        corrected = finite_best_val(per)
        if corrected is None:
            continue
        old = entry.get("best_val")
        if (
            isinstance(old, (int, float))
            and math.isfinite(old)
            and abs(old - corrected) < 1e-15
            and entry.get("best_val_source") == "finite_min_per_family_v2"
        ):
            continue
        fixes.append((name, old, corrected))
        entry["best_val"] = corrected
        entry["best_val_source"] = "finite_min_per_family_v2"

    if fixes and not dry_run:
        if "tasks" in payload:
            payload["tasks"] = tasks
        meta = payload.setdefault("meta", {})
        meta["best_val_finite_patch"] = True
        save_json(path, payload)

    return fixes


def load_replica_curves(
    train_output_dir: Path,
    task_name: str,
    opt_name: str,
    replica: int,
    training_steps: int,
) -> dict[str, list[float]] | None:
    result_file = (
        train_output_dir / task_name / opt_name / str(training_steps) / str(replica) / "result"
    )
    if not result_file.is_file():
        return None
    try:
        curves = load_json(result_file)
    except (OSError, ValueError):
        return None
    if not isinstance(curves, dict) or not curves:
        return None
    return curves


def rescore_task_result(
    result: dict[str, Any],
    baseline_cache: dict[str, dict[str, Any]],
    train_output_dir: Path,
    training_steps: int,
) -> dict[str, Any]:
    if result.get("status") != "ok":
        return result

    task_name = result.get("task_name")
    opt_name = result.get("opt_name")
    if not task_name or not opt_name:
        return result

    best_val = get_task_best_val(task_name, baseline_cache)
    if best_val is None:
        return result

    num_replicas = int(result.get("replicas_total") or 3)
    scored_replicas: list[dict[str, float]] = []
    replica_details: list[dict[str, Any]] = []

    for replica in range(num_replicas):
        curves = load_replica_curves(train_output_dir, task_name, opt_name, replica, training_steps)
        if curves is None:
            replica_details.append({"replica": replica, "status": "missing_cached_result"})
            continue
        try:
            scores = score_replica_curve(curves, best_val)
        except (ValueError, TypeError) as exc:
            replica_details.append(
                {"replica": replica, "status": "score_failed", "error": str(exc)}
            )
            continue
        replica_details.append({"replica": replica, "status": "ok", **scores})
        if math.isfinite(scores["mean_normalized_valid_outer"]):
            scored_replicas.append(scores)

    updated = dict(result)
    updated["best_val_reference"] = best_val
    updated["replica_details"] = replica_details

    if not scored_replicas:
        updated["mean_normalized_valid_outer"] = float("nan")
        return updated

    agg = aggregate_replica_scores(scored_replicas)
    updated.update(
        {
            "status": "ok",
            "metric": "mean_normalized_valid_outer",
            "mean_normalized_valid_outer": agg["mean_normalized_valid_outer"],
            "mean_final_test_loss": agg["final_test_loss"],
            "mean_init_valid_outer": agg["init_valid_outer"],
            "replicas_ok": len(scored_replicas),
        }
    )
    return updated


def summarize_results(results: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    ok = [r for r in results if r.get("status") == "ok"]
    costs = [
        r["mean_normalized_valid_outer"]
        for r in ok
        if r.get("mean_normalized_valid_outer") is not None
        and math.isfinite(r["mean_normalized_valid_outer"])
    ]
    tests = [
        r["mean_final_test_loss"]
        for r in ok
        if r.get("mean_final_test_loss") is not None and math.isfinite(r["mean_final_test_loss"])
    ]

    out = dict(summary)
    out["tasks_ok"] = len(ok)
    out["metric"] = "mean_normalized_valid_outer"
    out["best_val_patch"] = "finite_min_per_family_v2"

    if costs:
        out["mean_of_task_costs"] = statistics.mean(costs)
        out["best_task_cost"] = min(costs)
        out["worst_task_cost"] = max(costs)
        out["tasks_scored"] = len(costs)
        out["tasks_unscored"] = len(ok) - len(costs)
        out["median_task_cost"] = statistics.median(costs)
    if tests:
        out["mean_final_test_loss"] = statistics.mean(tests)
    return out


def rescore_eval_json(
    eval_path: Path,
    baseline_cache: dict[str, dict[str, Any]],
    train_output_dir: Path,
    training_steps: int,
) -> dict[str, Any]:
    payload = load_json(eval_path)
    results = payload.get("results", [])
    rescored = [
        rescore_task_result(r, baseline_cache, train_output_dir, training_steps) for r in results
    ]
    payload["results"] = rescored
    payload["summary"] = summarize_results(rescored, payload.get("summary", {}))
    return payload


def load_baseline_tasks(path: Path) -> dict[str, dict[str, Any]]:
    data = load_json(path)
    if isinstance(data.get("tasks"), dict):
        return data["tasks"]
    return data


def rescore_all_models(baselines: Path, models: list[str], training_steps: int) -> None:
    baseline_cache = load_baseline_tasks(baselines)
    for model in models:
        eval_json = BASE_DIR / f"eval_runs/dl_test_15/{model}_tier5_cpu_eval.json"
        train_dir = BASE_DIR / f"eval_runs/dl_test_15/tier5_cpu/{model}/train_outputs"
        if not eval_json.exists():
            print(f"SKIP {model}: no eval json")
            continue
        payload = rescore_eval_json(eval_json, baseline_cache, train_dir, training_steps)
        save_json(eval_json, payload)
        s = payload["summary"]
        print(
            f"{model}: scored={s.get('tasks_scored')}/{s.get('tasks_ok')} "
            f"mean={s.get('mean_of_task_costs'):.6f} median={s.get('median_task_cost'):.6f} "
            f"unscored={s.get('tasks_unscored')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix baselines and re-score eval JSON from cache")
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--eval-json", type=Path, default=None)
    parser.add_argument("--train-output-dir", type=Path, default=None)
    parser.add_argument("--training-steps", type=int, default=DEFAULT_TRAINING_STEPS)
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    fixes = patch_baseline_cache(args.baselines, dry_run=args.dry_run)
    print(f"Baseline fixes: {len(fixes)}")
    for name, old, new in fixes[:20]:
        print(f"  {name}: {old} -> {new}")
    if len(fixes) > 20:
        print(f"  ... and {len(fixes) - 20} more")

    if args.dry_run:
        return

    if args.all_models:
        rescore_all_models(args.baselines, TIER5_MODELS, args.training_steps)
        return

    eval_json = args.eval_json or (
        BASE_DIR / "eval_runs/dl_test_15/qwen_base_lora_sft_3_dl_tier5_cpu_eval.json"
    )
    train_dir = args.train_output_dir or (
        BASE_DIR / "eval_runs/dl_test_15/tier5_cpu/qwen_base_lora_sft_3_dl/train_outputs"
    )
    baseline_cache = load_baseline_tasks(args.baselines)
    payload = rescore_eval_json(eval_json, baseline_cache, train_dir, args.training_steps)
    save_json(eval_json, payload)
    summary = payload["summary"]
    print(f"\nRe-scored: {eval_json}")
    print(f"  tasks_scored: {summary.get('tasks_scored')}")
    print(f"  mean_of_task_costs: {summary.get('mean_of_task_costs')}")


if __name__ == "__main__":
    main()
