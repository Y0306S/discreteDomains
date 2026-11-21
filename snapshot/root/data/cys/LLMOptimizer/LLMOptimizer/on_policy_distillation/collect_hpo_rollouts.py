#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.hpo_agent import FailedRecommendation, HPOAgent
from src.benchmarks.benchmark_runner import (
    BENCHMARK_MAP,
    _coerce_recommended_config,
    _evaluate_config,
    _extract_config_info,
    _extract_fidelity_info,
    _format_config_for_prompt,
)
from src.config.model_registry import MODEL_CONFIGS, require_ollama_server


DEFAULT_TASK_IDS = [359955, 190411, 359977, 359956, 359970]
DEFAULT_TASKS = ["lr", "rf", "xgboost", "nn"]


def parse_csv(value: str, cast=str) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")


def load_completed_rows(path: Path, model_key: str, benchmark: str, task_id: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                row.get("teacher_model_key") == model_key
                and row.get("benchmark") == benchmark.upper()
                and int(row.get("task_id")) == int(task_id)
            ):
                rows.append(row)
    return sorted(rows, key=lambda item: int(item.get("step", 0)))


def previous_results_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    previous = []
    for row in rows:
        config = row.get("evaluated_config")
        loss = row.get("loss")
        accuracy = row.get("accuracy")
        if isinstance(config, dict) and loss is not None:
            previous.append({"config": config, "loss": loss, "accuracy": accuracy})
    return previous


def recommendation_to_json(rec: Any) -> str:
    payload = {
        "reasoning_trace": rec.reasoning_trace,
        "warm_start_config": [
            {"name": param.name, "value": param.value}
            for param in rec.warm_start_config
        ],
    }
    return json.dumps(payload, ensure_ascii=True)


def fidelity_config_from_space(fidelity_space) -> dict[str, Any]:
    fidelity_config = {}
    for f in fidelity_space.values():
        if hasattr(f, "upper") and f.upper is not None:
            fidelity_config[f.name] = f.upper
        elif hasattr(f, "sequence"):
            fidelity_config[f.name] = f.sequence[-1]
        else:
            fidelity_config[f.name] = getattr(f, "default_value", None)
    return fidelity_config


def collect_for_task(
    *,
    model_key: str,
    benchmark_name: str,
    task_id: int,
    steps: int,
    output_path: Path,
    resume: bool,
    include_default_context: bool,
    cot_prompt: str,
) -> None:
    if benchmark_name not in BENCHMARK_MAP:
        raise ValueError(f"Unknown benchmark '{benchmark_name}'. Available: {sorted(BENCHMARK_MAP)}")

    cfg = MODEL_CONFIGS[model_key]
    api_key = cfg.get("api_key_override") or os.getenv("OPENAI_HK_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_HK_API_KEY is not set.")

    completed_rows = load_completed_rows(output_path, model_key, benchmark_name, task_id) if resume else []
    completed_steps = {int(row.get("step", 0)) for row in completed_rows}
    previous_results = previous_results_from_rows(completed_rows)

    benchmark = BENCHMARK_MAP[benchmark_name](task_id=task_id, rng=1)
    agent = HPOAgent(api_key=api_key, model=cfg["id"], base_url=cfg["base_url"])
    cs = benchmark.get_configuration_space(seed=1)
    fidelity_space = benchmark.get_fidelity_space()
    config_params = _extract_config_info(cs)
    fidelity_params = _extract_fidelity_info(fidelity_space)
    config_space_str = _format_config_for_prompt(config_params)
    fidelity_config = fidelity_config_from_space(fidelity_space)

    agent.analyze(
        benchmark_obj=benchmark,
        intent=f"On-policy HPO distillation for {benchmark_name.upper()}",
        task_id=task_id,
    )

    if include_default_context and not previous_results:
        default_config = dict(cs.get_default_configuration())
        default_loss, default_cost = _evaluate_config(
            BENCHMARK_MAP[benchmark_name](task_id=task_id, rng=1),
            default_config,
            fidelity_config,
        )
        default_accuracy = (1 - default_loss) * 100 if default_loss is not None else None
        previous_results.append(
            {
                "config": default_config,
                "loss": default_loss,
                "accuracy": default_accuracy,
                "cost": default_cost,
                "source": "default_baseline",
            }
        )

    print(
        f"[{datetime.now().isoformat(timespec='seconds')}] "
        f"{benchmark_name.upper()} task_id={task_id} teacher={model_key} "
        f"resume_steps={sorted(completed_steps)}"
    )

    for step in range(1, steps + 1):
        if step in completed_steps:
            print(f"  step {step}/{steps}: skip existing")
            continue

        prev_snapshot = list(previous_results)
        recs, token_usage, prompt = agent.recommend_batch(
            model_name=benchmark_name,
            config_space_info=config_space_str,
            n=1,
            previous_results=prev_snapshot,
            cot_prompt=cot_prompt,
        )
        rec = recs[0]

        row: dict[str, Any] = {
            "schema_version": 1,
            "artifact_type": "hpo_on_policy_distillation_step",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "teacher_model_key": model_key,
            "teacher_model_id": cfg["id"],
            "base_url": cfg["base_url"],
            "benchmark": benchmark_name.upper(),
            "task_id": task_id,
            "step": step,
            "prompt": prompt,
            "previous_results": prev_snapshot,
            "config_params": config_params,
            "fidelity_params": fidelity_params,
            "fidelity_config": fidelity_config,
            "token_usage": token_usage,
            "skipped": False,
            "error_reason": None,
            "reasoning_trace": None,
            "recommended_config": None,
            "evaluated_config": None,
            "loss": None,
            "accuracy": None,
            "cost": None,
            "sft_messages": None,
        }

        if isinstance(rec, FailedRecommendation):
            row["skipped"] = True
            row["error_reason"] = rec.error_reason
            row["raw_response"] = rec.raw_response
            append_jsonl(output_path, row)
            print(f"  step {step}/{steps}: failed teacher response: {rec.error_reason}")
            continue

        row["reasoning_trace"] = rec.reasoning_trace
        row["raw_response"] = rec.raw_response
        row["recommended_config"] = [
            {"name": param.name, "value": param.value}
            for param in rec.warm_start_config
        ]

        config_dict, recommended_params, param_errors, invalid_config = _coerce_recommended_config(
            cs=cs,
            warm_start_config=rec.warm_start_config,
        )
        row["recommended_config"] = recommended_params

        if invalid_config:
            row["skipped"] = True
            row["error_reason"] = "; ".join(param_errors)
            append_jsonl(output_path, row)
            print(f"  step {step}/{steps}: invalid config: {row['error_reason']}")
            continue

        loss, cost = _evaluate_config(
            BENCHMARK_MAP[benchmark_name](task_id=task_id, rng=1),
            config_dict,
            fidelity_config,
        )
        accuracy = (1 - loss) * 100 if loss is not None else None
        row["evaluated_config"] = dict(config_dict)
        row["loss"] = loss
        row["accuracy"] = accuracy
        row["cost"] = cost
        row["sft_messages"] = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": recommendation_to_json(rec)},
        ]

        append_jsonl(output_path, row)
        if loss is not None:
            previous_results.append({"config": dict(config_dict), "loss": loss, "accuracy": accuracy})
            print(f"  step {step}/{steps}: loss={loss:.6f} accuracy={accuracy:.3f} cost={cost}")
        else:
            row["skipped"] = True
            row["error_reason"] = "Config validation/evaluation failed"
            print(f"  step {step}/{steps}: evaluation failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect on-policy HPO teacher rollouts for distillation.")
    parser.add_argument("--model-key", default="gemini-3.1-pro", choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--task-ids", default=",".join(str(x) for x in DEFAULT_TASK_IDS))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-default-context", action="store_true")
    parser.add_argument("--cot-prompt", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = parse_csv(args.tasks, str)
    task_ids = parse_csv(args.task_ids, int)
    output = Path(args.output) if args.output else (
        ROOT / "on_policy_distillation" / "runs" / f"hpo_on_policy_{args.model_key}.jsonl"
    )

    require_ollama_server([args.model_key])

    for task_id in task_ids:
        for task in tasks:
            collect_for_task(
                model_key=args.model_key,
                benchmark_name=task,
                task_id=task_id,
                steps=args.steps,
                output_path=output,
                resume=args.resume,
                include_default_context=not args.no_default_context,
                cot_prompt=args.cot_prompt,
            )

    print(f"Saved rollouts to {output}")


if __name__ == "__main__":
    main()
