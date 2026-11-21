#!/usr/bin/env python3
"""Build trial-1 warm-start presets via lora_recommended4 best-of-3.

Protocol (mltest20 Option B ablation):
  - Sample 3 configs from recommended4 with temp=0.9, top-p=0.9, top-k=50
  - Evaluate all 3 on the same holdout recipe as HPO (valid_size=0.35, test_size=0.25)
  - Keep only the best objective as the trial-1 preset (selection evals are NOT traj rows)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

TASKS = [
    ("yeast", "lightgbm"),
    ("car", "extra_trees"),
    ("PhishingWebsites", "xgboost"),
    ("segment", "catboost"),
    ("boston", "extra_trees"),
    ("tecator", "extra_trees"),
    ("socmob", "extra_trees"),
    ("pol", "xgboost"),
]

DEFAULT_TEST_JSONL = Path(
    "/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/ml_test_20_fixed.jsonl"
)
DEFAULT_ADAPTER = Path(
    "/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_recommended4"
)
DEFAULT_BASE = Path("/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B")
DEFAULT_BENCHMARK = Path("/root/data/cys/blockdata_archive/llmoptimizer_benchmark")
EVAL_CLI = Path(
    "/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/scripts/eval/evaluate_config_cli.py"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_tasks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = set(TASKS)
    selected = []
    for idx, rec in enumerate(records):
        meta = rec.get("metadata") or {}
        key = (meta.get("dataset"), meta.get("algorithm"))
        if key in wanted:
            row = dict(rec)
            row["_global_index"] = idx
            row["_task_key"] = f"{key[0]}:{key[1]}"
            selected.append(row)
    missing = wanted - {(r["metadata"]["dataset"], r["metadata"]["algorithm"]) for r in selected}
    if missing:
        raise SystemExit(f"Missing Option B tasks in JSONL: {sorted(missing)}")
    # Stable Option B order.
    order = {t: i for i, t in enumerate(TASKS)}
    selected.sort(key=lambda r: order[(r["metadata"]["dataset"], r["metadata"]["algorithm"])])
    return selected


def generate_predictions(
    records: list[dict[str, Any]],
    *,
    base_model: Path,
    adapter_dir: Path,
    out_jsonl: Path,
    num_samples: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer_path = (
        adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else base_model
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForImageTextToText.from_pretrained(
        base_model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()

    rows: list[dict[str, Any]] = []
    for local_i, record in enumerate(records):
        index = int(record.get("_global_index", local_i))
        task_key = record.get("_task_key") or (
            f"{record['metadata']['dataset']}:{record['metadata']['algorithm']}"
        )
        messages = [m for m in record["messages"] if m["role"] in {"system", "user"}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        print(f"[gen] {task_key} index={index} samples={num_samples}", flush=True)
        for sample_id in range(num_samples):
            with torch.no_grad():
                gen_kwargs = {
                    **inputs,
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": tokenizer.eos_token_id,
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                }
                output_ids = model.generate(**gen_kwargs)
            new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            row: dict[str, Any] = {
                "model": "lora_recommended4",
                "index": index,
                "sample_id": sample_id,
                "task_key": task_key,
                "raw_response": raw,
            }
            # Lazy import after path setup in main.
            from evaluate_config_cli import extract_json_object

            try:
                row["config"] = extract_json_object(raw)
            except Exception as exc:
                row["parse_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            write_jsonl(out_jsonl, rows)
    return rows


def evaluate_candidates(
    records_by_index: dict[int, dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    benchmark_root: Path,
    test_size: float,
    valid_size: float,
    seed: int,
) -> dict[str, Any]:
    from evaluate_config_cli import evaluate_config, extract_json_object, load_runtime

    runtime = load_runtime(benchmark_root)
    presets: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for pred in predictions:
        grouped.setdefault(str(pred["task_key"]), []).append(pred)

    for task_key, preds in grouped.items():
        rec = records_by_index[int(preds[0]["index"])]
        meta = rec["metadata"]
        candidates = []
        best = None
        for pred in preds:
            entry = {
                "sample_id": pred.get("sample_id"),
                "raw_response": pred.get("raw_response"),
                "parse_error": pred.get("parse_error"),
            }
            try:
                cfg = pred.get("config")
                if cfg is None:
                    cfg = extract_json_object(str(pred.get("raw_response", "")))
                evaluated = evaluate_config(
                    runtime=runtime,
                    benchmark_root=benchmark_root,
                    task_type=meta["task_type"],
                    dataset=meta["dataset"],
                    algorithm=meta["algorithm"],
                    hparams=cfg,
                    metric=meta.get("metric"),
                    test_size=test_size,
                    valid_size=valid_size,
                    seed=seed,
                )
                entry.update(
                    {
                        "config": cfg,
                        "objective": evaluated["objective"],
                        "status": "ok",
                    }
                )
                if best is None or float(evaluated["objective"]) < float(best["objective"]):
                    best = {
                        "config": cfg,
                        "objective": evaluated["objective"],
                        "sample_id": pred.get("sample_id"),
                        "metric": meta.get("metric"),
                        "task_type": meta["task_type"],
                        "dataset": meta["dataset"],
                        "algorithm": meta["algorithm"],
                    }
            except Exception as exc:
                entry["status"] = "failed"
                entry["error"] = f"{type(exc).__name__}: {exc}"
            candidates.append(entry)

        if best is None:
            raise SystemExit(f"No valid candidate for {task_key}: {candidates}")
        presets[task_key] = {
            "config": best["config"],
            "objective": best["objective"],
            "sample_id": best["sample_id"],
            "metric": best["metric"],
            "task_type": best["task_type"],
            "dataset": best["dataset"],
            "algorithm": best["algorithm"],
            "num_candidates": len(candidates),
            "candidates": candidates,
            "selection": {
                "temperature": None,  # filled by caller
                "num_samples": len(preds),
                "valid_size": valid_size,
                "test_size": test_size,
                "seed": seed,
                "note": "best-of-N selection evals are outside the 80-trial budget; "
                "only best_config is injected as trial 1",
            },
        }
        print(
            f"[eval] {task_key}: best_obj={best['objective']:.6f} "
            f"from sample_id={best['sample_id']} "
            f"({sum(1 for c in candidates if c.get('status')=='ok')}/{len(candidates)} ok)",
            flush=True,
        )
    return presets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-jsonl", type=Path, default=DEFAULT_TEST_JSONL)
    p.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    p.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    p.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_BENCHMARK
        / "benchmark_data/real_hpo_v2_mltest20_algoB/warm_start_rec4",
    )
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--valid-size", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--skip-generate", action="store_true")
    p.add_argument("--predictions-jsonl", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(EVAL_CLI.parent))
    from evaluate_config_cli import prepare_benchmark_imports

    prepare_benchmark_imports(args.benchmark_root)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    filtered_jsonl = out_dir / "optionB_8tasks.jsonl"
    predictions_jsonl = args.predictions_jsonl or (out_dir / "rec4_predictions_bo3.jsonl")
    presets_json = out_dir / "presets_best_of_3.json"

    records_all = load_jsonl(args.dataset_jsonl)
    selected = select_tasks(records_all)
    write_jsonl(filtered_jsonl, selected)
    print(f"Wrote {len(selected)} tasks -> {filtered_jsonl}", flush=True)

    if args.skip_generate:
        predictions = load_jsonl(predictions_jsonl)
        print(f"Loaded {len(predictions)} predictions from {predictions_jsonl}", flush=True)
    else:
        predictions = generate_predictions(
            selected,
            base_model=args.base_model,
            adapter_dir=args.adapter_dir,
            out_jsonl=predictions_jsonl,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"Wrote predictions -> {predictions_jsonl}", flush=True)

    by_index = {int(r["_global_index"]): r for r in selected}
    presets = evaluate_candidates(
        by_index,
        predictions,
        benchmark_root=args.benchmark_root,
        test_size=args.test_size,
        valid_size=args.valid_size,
        seed=args.seed,
    )
    for entry in presets.values():
        entry["selection"]["temperature"] = args.temperature
        entry["selection"]["top_p"] = args.top_p
        entry["selection"]["top_k"] = args.top_k
        entry["selection"]["adapter"] = str(args.adapter_dir)

    presets_json.write_text(json.dumps(presets, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote presets -> {presets_json}", flush=True)
    for k, v in presets.items():
        print(f"  {k}: obj={v['objective']:.6f} sample={v['sample_id']}")


if __name__ == "__main__":
    main()
