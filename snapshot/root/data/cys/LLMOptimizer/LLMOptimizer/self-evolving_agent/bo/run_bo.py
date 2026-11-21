from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (PACKAGE_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bo.botorch_optimizer import BoTorchOptimizer
from bo.spaces import ConfigSpaceAdapter
from bo.trajectory import summarize_round


def _artifact_root() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_json(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(v) for v in obj]
    return obj


def _load_benchmark(model_name: str, task_id: int):
    try:
        from src.benchmarks.benchmark_runner import BENCHMARK_MAP
    except Exception as exc:
        raise RuntimeError(
            "Could not import HPOBench benchmark helpers. Run this script from "
            "the LLMOptimizer repo inside the hpobench environment."
        ) from exc
    benchmark_cls = BENCHMARK_MAP[model_name.lower()]
    benchmark = benchmark_cls(task_id=task_id, rng=1)
    config_space = benchmark.get_configuration_space(seed=1)
    fidelity_space = benchmark.get_fidelity_space()
    return benchmark, config_space, fidelity_space


def _evaluate_with_metadata(benchmark, config: dict[str, Any], fidelity_config: dict[str, Any]) -> dict[str, Any]:
    from src.benchmarks.benchmark_runner import _evaluate_config

    loss, cost = _evaluate_config(benchmark, config, fidelity_config)
    if loss is None:
        return {
            "loss": None,
            "cost": cost,
            "accuracy": None,
            "status": "failed",
            "error_reason": "evaluation_failed",
        }
    accuracy = (1.0 - float(loss)) * 100.0
    return {
        "loss": float(loss),
        "cost": float(cost) if cost is not None else None,
        "accuracy": accuracy,
        "status": "success",
        "error_reason": None,
    }


def run_bo(
    benchmark_name: str,
    task_id: int,
    max_trials: int = 50,
    n_initial: int = 8,
    batch_size: int = 1,
    patience: int = 10,
    min_delta: float = 1e-4,
    round_size: int = 5,
    seed: int = 1,
    output_dir: str | None = None,
    warm_start_config: dict[str, Any] | None = None,
    use_random_fallback: bool = False,
) -> dict[str, Any]:
    benchmark_name = benchmark_name.lower()
    benchmark, config_space, fidelity_space = _load_benchmark(benchmark_name, task_id)
    adapter = ConfigSpaceAdapter.from_config_space(config_space)

    fidelity_config = {}
    for f in fidelity_space.values():
        if hasattr(f, "upper") and f.upper is not None:
            fidelity_config[f.name] = f.upper
        elif hasattr(f, "sequence"):
            fidelity_config[f.name] = f.sequence[-1]
        else:
            fidelity_config[f.name] = getattr(f, "default_value", None)

    optimizer = BoTorchOptimizer(adapter=adapter, seed=seed, use_random_fallback=use_random_fallback)
    artifact_dir = Path(output_dir) if output_dir else _artifact_root() / "bo_traces" / benchmark_name.upper() / str(task_id)
    _ensure_dir(artifact_dir)

    trials: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    observed_vectors: list[np.ndarray] = []
    observed_losses: list[float] = []
    best_loss = float("inf")
    best_config = None
    best_step = None
    no_improve = 0
    step = 0

    def _record_trial(config: dict[str, Any], source: str) -> dict[str, Any]:
        nonlocal best_loss, best_config, best_step, no_improve
        signature = adapter.signature(config)
        if signature in seen_signatures:
            config = dict(config)
        seen_signatures.add(signature)
        result = _evaluate_with_metadata(benchmark, config, fidelity_config)
        trial = {
            "step": step,
            "config": _safe_json(config),
            "source": source,
            **result,
            "signature": signature,
        }
        if result["loss"] is not None:
            observed_vectors.append(adapter.config_to_vector(config))
            observed_losses.append(result["loss"])
            if result["loss"] + min_delta < best_loss:
                best_loss = result["loss"]
                best_config = dict(config)
                best_step = step
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        trials.append(trial)
        return trial

    if warm_start_config:
        _record_trial(warm_start_config, source="warm_start")
        step += 1

    while step < max_trials:
        if len(observed_vectors) < n_initial:
            vector = optimizer._random_candidate()
            candidate_results = [
                {
                    "config": adapter.vector_to_config(vector),
                    "source": "random",
                }
            ]
        else:
            budget = min(batch_size, max_trials - step)
            candidate_results = optimizer.suggest(
                observed_vectors=observed_vectors,
                observed_losses=observed_losses,
                candidate_budget=budget,
            )

        for candidate in candidate_results:
            if step >= max_trials:
                break
            if isinstance(candidate, dict):
                candidate_config = candidate["config"]
                source = candidate["source"]
            else:
                candidate_config = candidate.config
                source = candidate.source

            candidate_sig = adapter.signature(candidate_config)
            retry = 0
            while candidate_sig in seen_signatures and retry < 50:
                candidate_config = adapter.vector_to_config(optimizer._random_candidate())
                candidate_sig = adapter.signature(candidate_config)
                source = "random_retry"
                retry += 1

            _record_trial(candidate_config, source=source)
            step += 1

            if len(observed_vectors) >= n_initial and no_improve >= patience:
                break

        if len(observed_vectors) >= n_initial and no_improve >= patience:
            break

    round_summaries = []
    for idx in range(0, len(trials), round_size):
        block = trials[: idx + round_size]
        round_summaries.append(summarize_round(block, adapter, len(round_summaries) + 1, round_size))

    trace = {
        "schema_version": 1,
        "artifact_type": "self_evolving_hpo_bo_trace",
        "benchmark": benchmark_name.upper(),
        "task_id": task_id,
        "metric": "loss",
        "best_config": _safe_json(best_config) if best_config is not None else None,
        "best_loss": best_loss if best_loss != float("inf") else None,
        "best_step": best_step,
        "trials": trials,
        "round_summaries": round_summaries,
        "config_space": _safe_json([asdict(spec) for spec in adapter.specs]),
        "fidelity": _safe_json(fidelity_config),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    trace_path = artifact_dir / f"bo_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved BO trace to {trace_path}")
    return trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BO for HPOBench ML tasks.")
    parser.add_argument("--benchmark", required=True, choices=["lr", "rf", "xgboost", "nn"])
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--max-trials", type=int, default=50)
    parser.add_argument("--n-initial", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--round-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--warm-start-json", type=str, default=None)
    parser.add_argument("--use-random-fallback", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warm_start = None
    if args.warm_start_json:
        with open(args.warm_start_json, "r", encoding="utf-8") as f:
            warm_start = json.load(f)
    run_bo(
        benchmark_name=args.benchmark,
        task_id=args.task_id,
        max_trials=args.max_trials,
        n_initial=args.n_initial,
        batch_size=args.batch_size,
        patience=args.patience,
        min_delta=args.min_delta,
        round_size=args.round_size,
        seed=args.seed,
        output_dir=args.output_dir,
        warm_start_config=warm_start,
        use_random_fallback=args.use_random_fallback,
    )


if __name__ == "__main__":
    main()
