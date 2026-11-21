#!/usr/bin/env python3
"""Evaluate warm-start configs with the llmoptimizer_benchmark OpenBox objective.

The evaluator follows the benchmark recipe documented in
/jizhicfs/pkuhetu/bcxu/llmoptimizer_benchmark/README.md:
build a ConfigSpace configuration, call the CASH evaluator, and read
result["objectives"][0]. The objective is minimized; classification metrics such
as balanced accuracy are returned as negative scores by Mindware.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import types
from pathlib import Path
from typing import Any


BENCHMARK_ROOT = Path("/jizhicfs/pkuhetu/bcxu/llmoptimizer_benchmark")


def patch_configspace_quantization() -> None:
    """Allow benchmark code written for older ConfigSpace `q=` arguments."""
    try:
        import ConfigSpace.hyperparameters as hp_mod
    except ImportError:
        return
    if getattr(hp_mod.UniformIntegerHyperparameter, "_bolt_q_patch", False):
        return

    original_integer = hp_mod.UniformIntegerHyperparameter
    original_float = hp_mod.UniformFloatHyperparameter

    def uniform_integer(name, lower, upper, default_value=None, q=None, log=False, meta=None, **kwargs):
        _ = kwargs
        meta = dict(meta or {})
        if q is not None:
            meta["q"] = q
        return original_integer(name, lower, upper, default_value=default_value, log=log, meta=meta)

    def uniform_float(name, lower, upper, default_value=None, q=None, log=False, meta=None, **kwargs):
        _ = kwargs
        meta = dict(meta or {})
        if q is not None:
            meta["q"] = q
        return original_float(name, lower, upper, default_value=default_value, log=log, meta=meta)

    uniform_integer._bolt_q_patch = True
    uniform_float._bolt_q_patch = True
    hp_mod.UniformIntegerHyperparameter = uniform_integer
    hp_mod.UniformFloatHyperparameter = uniform_float


def patch_sklearn_version_for_benchmark_configspace() -> None:
    try:
        import sklearn
        import sklearn.metrics._scorer as scorer_mod
    except ImportError:
        return
    if getattr(sklearn, "__version__", "") > "1.3.2":
        sklearn.__version__ = "1.3.2"
    if not hasattr(scorer_mod, "_ThresholdScorer") and hasattr(scorer_mod, "_Scorer"):
        scorer_mod._ThresholdScorer = scorer_mod._Scorer


def install_benchmark_package_stub(benchmark_root: Path) -> None:
    existing = sys.modules.get("mindware")
    if existing is not None and getattr(existing, "__path__", None):
        return
    mindware = types.ModuleType("mindware")
    mindware.__path__ = [str(benchmark_root / "mindware")]
    sys.modules["mindware"] = mindware


def prepare_benchmark_imports(benchmark_root: Path) -> None:
    patch_configspace_quantization()
    patch_sklearn_version_for_benchmark_configspace()
    install_benchmark_package_stub(benchmark_root)
    if str(benchmark_root) not in sys.path:
        sys.path.insert(0, str(benchmark_root))
    examples_dir = benchmark_root / "examples" / "benchmark"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate warm-start HPO configs.")
    parser.add_argument("--benchmark-root", type=Path, default=BENCHMARK_ROOT)
    parser.add_argument("--task-type", choices=["cls", "rgs"])
    parser.add_argument("--dataset")
    parser.add_argument("--algorithm")
    parser.add_argument("--hyperparameters", help="JSON object of hyperparameters.")
    parser.add_argument("--metric", default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--valid-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset-jsonl", type=Path, help="SFT JSONL with metadata fields.")
    parser.add_argument(
        "--predictions-jsonl",
        type=Path,
        help="JSONL with one object per dataset row: {model, index, config|raw_response}.",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--generate",
        choices=["labels", "qwen", "openai"],
        help="Generate prediction JSONL from labels, a Qwen LoRA adapter, or an OpenAI-compatible API.",
    )
    parser.add_argument("--predictions-out", type=Path)
    parser.add_argument("--model-label", default=None)
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("/jizhicfs/pkuhetu/bcxu/models/Qwen3.5-9B-Base"),
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        help="Optional LoRA adapter. Omit for base-model Qwen generation.",
    )
    parser.add_argument("--api-model", default="gemini-3.1-flash")
    parser.add_argument("--api-base-url", default="https://api.openai-hk.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_HK_API_KEY")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--resume", action="store_true", help="Resume generation from an existing JSONL.")
    parser.add_argument("--api-retries", type=int, default=5)
    parser.add_argument("--api-retry-sleep", type=float, default=10.0)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of responses to generate per task. Unique configs are evaluated and the best objective is kept.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for generation. Use 0 for greedy (deterministic) decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability threshold. Set to 1.0 to disable.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling. Set to 0 to disable.",
    )
    return parser.parse_args()


def extract_json_object(text: str) -> dict[str, Any]:
    def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
        if isinstance(payload.get("warm_start_config"), dict):
            payload = payload["warm_start_config"]
        if isinstance(payload.get("config"), dict):
            payload = payload["config"]
        return payload

    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return normalize_payload(payload)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in response.")
    depth = 0
    in_string = False
    escaped = False
    payload_text = None
    for pos in range(start, len(text)):
        char = text[pos]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                payload_text = text[start : pos + 1]
                break
    if payload_text is None:
        raise ValueError("Unterminated JSON object in response.")
    payload = json.loads(payload_text)
    if not isinstance(payload, dict):
        raise ValueError("Extracted JSON is not an object.")
    return normalize_payload(payload)


def load_runtime(benchmark_root: Path):
    prepare_benchmark_imports(benchmark_root)
    from ConfigSpace import Configuration
    from benchmark_utils import get_dataset_info
    from mindware.components.config_space.cs_builder import get_cs_args, get_hpo_cs
    from mindware.components.metrics.metric import get_metric
    from mindware.components.utils.constants import CLASSIFICATION, REGRESSION
    from mindware.modules.cash.cash_evaluator import CASHCLSEvaluator, CASHRGSEvaluator
    from mindware.utils.data_manager import DataManager

    return {
        "Configuration": Configuration,
        "get_dataset_info": get_dataset_info,
        "get_cs_args": get_cs_args,
        "get_hpo_cs": get_hpo_cs,
        "get_metric": get_metric,
        "CLASSIFICATION": CLASSIFICATION,
        "REGRESSION": REGRESSION,
        "CASHCLSEvaluator": CASHCLSEvaluator,
        "CASHRGSEvaluator": CASHRGSEvaluator,
        "DataManager": DataManager,
    }


def coerce_hparams_for_algorithm(hparams: dict[str, Any], algorithm: str) -> dict[str, Any]:
    coerced = {}
    for name, value in hparams.items():
        if name == "algorithm":
            continue
        full_name = name if ":" in name else f"{algorithm}:{name}"
        coerced[full_name] = value
    return coerced


def evaluate_config(
    runtime: dict[str, Any],
    benchmark_root: Path,
    task_type: str,
    dataset: str,
    algorithm: str,
    hparams: dict[str, Any],
    metric: str | None = None,
    test_size: float = 0.2,
    valid_size: float = 0.25,
    seed: int = 1,
) -> dict[str, Any]:
    task_type_id = runtime["CLASSIFICATION"] if task_type == "cls" else runtime["REGRESSION"]
    metric = metric or ("bal_acc" if task_type == "cls" else "mse")

    cwd = Path.cwd()
    try:
        os.chdir(benchmark_root)
        dataset_path, label_column, header, sep = runtime["get_dataset_info"](task_type, dataset)
    finally:
        os.chdir(cwd)
    dataset_path = Path(dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = benchmark_root / dataset_path

    dm = runtime["DataManager"]()
    df = dm.load_csv(str(dataset_path), header=header, sep=sep)
    train_df, _ = dm.split_data(
        df,
        label_col=label_column,
        test_size=test_size,
        random_state=seed,
        task_type=task_type_id,
    )
    train_node = dm.from_train_df(train_df, label_col=label_column)
    train_node = dm.preprocess_fit(train_node, task_type_id)

    cs = runtime["get_hpo_cs"](
        estimator_id=algorithm,
        task_type=task_type_id,
        **runtime["get_cs_args"](
            resampling_params={"test_size": valid_size, "shuffle": False},
            data_node=train_node,
        ),
    )
    config_dict = cs.get_default_configuration().get_dictionary()
    config_dict.update(coerce_hparams_for_algorithm(hparams, algorithm))
    if "algorithm" in {hp.name for hp in cs.get_hyperparameters()}:
        config_dict["algorithm"] = algorithm

    config = runtime["Configuration"](cs, values=config_dict)
    scorer = runtime["get_metric"](metric)
    evaluator_cls = runtime["CASHCLSEvaluator"] if task_type == "cls" else runtime["CASHRGSEvaluator"]
    evaluator = evaluator_cls(
        scorer=scorer,
        data_node=train_node,
        resampling_strategy="holdout",
        resampling_params={"test_size": valid_size, "shuffle": False},
        timestamp=time.time(),
        seed=seed,
    )
    config_out = config.get_dictionary()
    config_out["algorithm"] = algorithm
    result = evaluator(config_out, save=False)
    return {
        "task_type": task_type,
        "dataset": dataset,
        "algorithm": algorithm,
        "metric": metric,
        "objective": float(result["objectives"][0]),
        "objective_direction": "minimize",
        "config": config_out,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def generate_label_predictions(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.dataset_jsonl is None or args.predictions_out is None:
        raise SystemExit("--dataset-jsonl and --predictions-out are required for generation.")
    records = load_jsonl(args.dataset_jsonl)
    if args.max_records is not None:
        records = records[: args.max_records]
    label = args.model_label or "label"
    rows = [
        {
            "model": label,
            "index": index,
            "config": json.loads(record["messages"][2]["content"]),
            "raw_response": record["messages"][2]["content"],
        }
        for index, record in enumerate(records)
    ]
    write_jsonl(args.predictions_out, rows)
    return rows


def generation_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    return [message for message in record["messages"] if message["role"] in {"system", "user"}]


def generate_qwen_predictions(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.dataset_jsonl is None or args.predictions_out is None:
        raise SystemExit("--dataset-jsonl and --predictions-out are required.")
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    records = load_jsonl(args.dataset_jsonl)
    if args.max_records is not None:
        records = records[: args.max_records]
    tokenizer_path = (
        args.adapter_dir
        if args.adapter_dir is not None and (args.adapter_dir / "tokenizer_config.json").exists()
        else args.base_model
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    if args.adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, args.adapter_dir)
        label = args.model_label or args.adapter_dir.name
    else:
        model = base
        label = args.model_label or args.base_model.name
    model.eval()
    rows = []
    for local_index, record in enumerate(records):
        index = record.pop("_global_index", local_index)
        prompt = tokenizer.apply_chat_template(
            generation_messages(record),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        for sample_id in range(args.num_samples):
            with torch.no_grad():
                generate_kwargs = {
                    **inputs,
                    "max_new_tokens": args.max_new_tokens,
                    "pad_token_id": tokenizer.eos_token_id,
                }
                if args.temperature > 0:
                    generate_kwargs["do_sample"] = True
                    generate_kwargs["temperature"] = args.temperature
                    if args.top_p < 1.0:
                        generate_kwargs["top_p"] = args.top_p
                    if args.top_k > 0:
                        generate_kwargs["top_k"] = args.top_k
                else:
                    generate_kwargs["do_sample"] = False
                output_ids = model.generate(**generate_kwargs)
            new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            row = {
                "model": label,
                "index": index,
                "sample_id": sample_id,
                "raw_response": raw,
            }
            try:
                row["config"] = extract_json_object(raw)
            except Exception as exc:
                row["parse_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            write_jsonl(args.predictions_out, rows)
    return rows


def generate_openai_predictions(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.dataset_jsonl is None or args.predictions_out is None:
        raise SystemExit("--dataset-jsonl and --predictions-out are required for generation.")
    from openai import OpenAI

    api_key = args.api_key or os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"No API key found. Pass --api-key or set {args.api_key_env}.")
    records = load_jsonl(args.dataset_jsonl)
    if args.max_records is not None:
        records = records[: args.max_records]
    client = OpenAI(api_key=api_key, base_url=args.api_base_url, timeout=180.0)
    label = args.model_label or args.api_model
    rows = []
    done_indexes = set()
    if args.resume and args.predictions_out.exists():
        rows = load_jsonl(args.predictions_out)
        done_indexes = {
            (int(row["index"]), int(row.get("sample_id", 0)))
            for row in rows
            if row.get("model") == label
        }
    for index, record in enumerate(records):
        for sample_id in range(args.num_samples):
            if (index, sample_id) in done_indexes:
                continue
            response = None
            for attempt in range(args.api_retries + 1):
                try:
                    response = client.chat.completions.create(
                        model=args.api_model,
                        messages=generation_messages(record),
                        temperature=args.temperature,
                        top_p=args.top_p if args.top_p < 1.0 else None,
                        max_tokens=args.max_new_tokens,
                    )
                    break
                except Exception:
                    if attempt >= args.api_retries:
                        raise
                    time.sleep(args.api_retry_sleep * (attempt + 1))
            if response is None:
                raise RuntimeError("API response was not created.")
            raw = response.choices[0].message.content or ""
            row = {
                "model": label,
                "index": index,
                "sample_id": sample_id,
                "raw_response": raw,
            }
            try:
                row["config"] = extract_json_object(raw)
            except Exception as exc:
                row["parse_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            done_indexes.add((index, sample_id))
            write_jsonl(args.predictions_out, rows)
    return rows


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset_jsonl is None or args.predictions_jsonl is None:
        raise SystemExit("--dataset-jsonl and --predictions-jsonl are required for batch mode.")
    runtime = load_runtime(args.benchmark_root)
    records = load_jsonl(args.dataset_jsonl)
    predictions = load_jsonl(args.predictions_jsonl)

    # Group predictions by (index, model) so we can generate multiple samples
    # per task, deduplicate them, evaluate only unique configs, and keep the best.
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for pred in predictions:
        index = int(pred["index"])
        model = str(pred["model"])
        grouped.setdefault((index, model), []).append(pred)

    total_tasks = len(grouped)
    print(f"[OpenBox Eval] Starting evaluation of {total_tasks} tasks...")
    print(f"[OpenBox Eval] Predictions: {args.predictions_jsonl}")
    print(f"[OpenBox Eval] Output: {args.output_json}")
    print("=" * 60)

    by_model: dict[str, list[dict[str, Any]]] = {}
    results = []
    start_time = time.time()

    for task_num, ((index, model), preds) in enumerate(grouped.items(), 1):
        task_start = time.time()
        record = records[index]
        meta = record.get("metadata")
        if not meta:
            raise SystemExit(f"Record {index} is missing metadata.")

        # Deduplicate configs for this task/model.
        unique_configs: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        for pred in preds:
            try:
                config_payload = pred.get("config")
                if config_payload is None:
                    config_payload = extract_json_object(str(pred.get("raw_response", "")))
                config_key = json.dumps(config_payload, sort_keys=True, ensure_ascii=True)
                if config_key not in seen_keys:
                    seen_keys.add(config_key)
                    unique_configs.append(config_payload)
            except Exception:
                continue

        # Evaluate unique configs and keep the one with the best (lowest) objective.
        best_row: dict[str, Any] | None = None
        best_objective: float | None = None
        eval_errors: list[str] = []
        for config_payload in unique_configs:
            try:
                evaluated = evaluate_config(
                    runtime=runtime,
                    benchmark_root=args.benchmark_root,
                    task_type=meta["task_type"],
                    dataset=meta["dataset"],
                    algorithm=meta["algorithm"],
                    hparams=config_payload,
                    metric=meta.get("metric"),
                    test_size=args.test_size,
                    valid_size=args.valid_size,
                    seed=args.seed,
                )
                row = {"model": model, "index": index, "status": "ok", **evaluated}
                obj = float(evaluated["objective"])
                if best_objective is None or obj < best_objective:
                    best_objective = obj
                    best_row = row
            except Exception as exc:
                eval_errors.append(f"{type(exc).__name__}: {exc}")

        if best_row is None:
            # None of the unique configs evaluated successfully.
            row = {
                "model": model,
                "index": index,
                "status": "failed",
                "error": "; ".join(eval_errors) if eval_errors else "No parseable configs",
            }
        else:
            row = best_row

        results.append(row)
        by_model.setdefault(model, []).append(row)

        # Progress logging
        elapsed = time.time() - start_time
        task_elapsed = time.time() - task_start
        avg_time = elapsed / task_num
        remaining = avg_time * (total_tasks - task_num)
        status_icon = "✅" if row["status"] == "ok" else "❌"

        print(f"[{task_num}/{total_tasks}] {status_icon} {meta['dataset']} ({meta['algorithm']}) | "
              f"Task: {task_elapsed:.1f}s | Avg: {avg_time:.1f}s | ETA: {remaining/60:.1f}min")

        # Incremental save: write partial results after each task
        if args.output_json and task_num % 10 == 0:
            _save_progress(args.output_json, by_model, results, partial=True)

    print("=" * 60)
    print(f"[OpenBox Eval] All {total_tasks} tasks complete. Computing normalization...")

    # ---- Per-task normalization ----
    # Gather all valid objectives per task index
    task_objectives: dict[int, list[float]] = {}
    for row in results:
        if row["status"] == "ok":
            idx = int(row["index"])
            task_objectives.setdefault(idx, []).append(float(row["objective"]))

    # For each task compute best (min) and worst (max) valid objective
    task_best: dict[int, float] = {}
    task_worst: dict[int, float] = {}
    for idx, objs in task_objectives.items():
        task_best[idx] = min(objs)
        task_worst[idx] = max(objs)

    num_models = len(by_model)
    # Assign normalized score to every row
    for row in results:
        idx = int(row["index"])
        if row["status"] == "ok":
            obj = float(row["objective"])
            best = task_best[idx]
            worst = task_worst[idx]
            denom = worst - best
            if denom > 1e-12:
                row["normalized_score"] = (obj - best) / denom
            else:
                row["normalized_score"] = 0.0  # all identical
        else:
            # Failure: fixed penalty of 2.0 (worse than any valid normalized score)
            row["normalized_score"] = 2.0

    summary = {}
    for model, rows in by_model.items():
        ok = [row for row in rows if row["status"] == "ok"]
        failed = [row for row in rows if row["status"] == "failed"]
        norm_scores = [row["normalized_score"] for row in rows]
        summary[model] = {
            "total": len(rows),
            "ok": len(ok),
            "failed": len(failed),
            "mean_normalized_score": sum(norm_scores) / len(norm_scores) if norm_scores else None,
            "best_normalized_score": min(norm_scores) if norm_scores else None,
            "worst_normalized_score": max(norm_scores) if norm_scores else None,
            "mean_objective_ok_only": sum(row["objective"] for row in ok) / len(ok) if ok else None,
            "best_objective": min(row["objective"] for row in ok) if ok else None,
        }
    payload = {"summary": summary, "results": results}

    total_elapsed = time.time() - start_time
    print(f"[OpenBox Eval] Complete! Total time: {total_elapsed/60:.1f} minutes")
    for model, s in summary.items():
        print(f"  {model}: {s['ok']}/{s['total']} OK, mean_score={s['mean_normalized_score']:.4f}")

    if args.output_json:
        _save_progress(args.output_json, by_model, results, partial=False)
    return payload


def _save_progress(output_path: Path, by_model: dict, results: list, partial: bool = False) -> None:
    """Save incremental or final results to JSON."""
    # Compute normalization for current results
    task_objectives: dict[int, list[float]] = {}
    for row in results:
        if row["status"] == "ok":
            idx = int(row["index"])
            task_objectives.setdefault(idx, []).append(float(row["objective"]))

    task_best: dict[int, float] = {}
    task_worst: dict[int, float] = {}
    for idx, objs in task_objectives.items():
        task_best[idx] = min(objs)
        task_worst[idx] = max(objs)

    # Update normalized scores
    for row in results:
        idx = int(row["index"])
        if row["status"] == "ok":
            obj = float(row["objective"])
            best = task_best.get(idx, obj)
            worst = task_worst.get(obj, obj)
            denom = worst - best
            if denom > 1e-12:
                row["normalized_score"] = (obj - best) / denom
            else:
                row["normalized_score"] = 0.0
        else:
            row["normalized_score"] = 2.0

    summary = {}
    for model, rows in by_model.items():
        ok = [row for row in rows if row["status"] == "ok"]
        failed = [row for row in rows if row["status"] == "failed"]
        norm_scores = [row["normalized_score"] for row in rows]
        summary[model] = {
            "total": len(rows),
            "ok": len(ok),
            "failed": len(failed),
            "mean_normalized_score": sum(norm_scores) / len(norm_scores) if norm_scores else None,
            "best_normalized_score": min(norm_scores) if norm_scores else None,
            "worst_normalized_score": max(norm_scores) if norm_scores else None,
            "mean_objective_ok_only": sum(row["objective"] for row in ok) / len(ok) if ok else None,
            "best_objective": min(row["objective"] for row in ok) if ok else None,
        }

    payload = {"summary": summary, "results": results}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if partial:
        # Save to a .partial file
        partial_path = output_path.with_suffix(".partial.json")
        with partial_path.open("w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    else:
        # Final save
        with output_path.open("w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        # Clean up partial file if it exists
        partial_path = output_path.with_suffix(".partial.json")
        if partial_path.exists():
            partial_path.unlink()


def main() -> None:
    args = parse_args()
    if args.generate:
        if args.generate == "labels":
            rows = generate_label_predictions(args)
        elif args.generate == "qwen":
            rows = generate_qwen_predictions(args)
        else:
            rows = generate_openai_predictions(args)
        print(json.dumps({"predictions": len(rows), "output": str(args.predictions_out)}, indent=2))
        return
    if args.dataset_jsonl or args.predictions_jsonl:
        print(json.dumps(run_batch(args), indent=2, sort_keys=True))
        return
    required = [args.task_type, args.dataset, args.algorithm, args.hyperparameters]
    if any(value is None for value in required):
        raise SystemExit(
            "Single-config mode requires --task-type, --dataset, --algorithm, and --hyperparameters."
        )
    runtime = load_runtime(args.benchmark_root)
    hparams = json.loads(args.hyperparameters)
    if not isinstance(hparams, dict):
        raise SystemExit("--hyperparameters must be a JSON object")
    print(
        json.dumps(
            evaluate_config(
                runtime=runtime,
                benchmark_root=args.benchmark_root,
                task_type=args.task_type,
                dataset=args.dataset,
                algorithm=args.algorithm,
                hparams=hparams,
                metric=args.metric,
                test_size=args.test_size,
                valid_size=args.valid_size,
                seed=args.seed,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
