#!/usr/bin/env python3
"""Compare Gemini against the fine-tuned Qwen warm-start adapter on validation data.

The comparison is schema-based by default: each model is asked for the same
validation-set warm-start prompt, the response JSON is normalized to canonical
algorithm-prefixed keys, and the normalized config is compared with the held-out
best historical config from validation.jsonl.

For true HPO performance, use the JSONL predictions this script writes and run
selected configs through evaluate_config_cli.py or the benchmark runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "validation.jsonl"
DEFAULT_SCHEMA = ROOT / "configs" / "ml_model_schema.json"
DEFAULT_QWEN_BASE = Path("/jizhicfs/pkuhetu/bcxu/models/Qwen3.5-9B-Base")
DEFAULT_QWEN_ADAPTER = ROOT / "outputs" / "llamafactory_qwen35_9b_base_lora_sft"
DEFAULT_OUTPUT = ROOT / "eval_comparison" / "gemini_vs_qwen_validation.jsonl"


@dataclass(frozen=True)
class EvalRecord:
    index: int
    task_type: str
    task_id: str
    dataset: str
    algorithm: str
    system_prompt: str
    user_prompt: str
    gold_payload: dict[str, Any]
    gold_config: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Gemini and fine-tuned Qwen on BOLT warm-start validation prompts."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["qwen", "gemini"],
        default=["qwen", "gemini"],
        help="Which models to query.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-id", action="append", default=None)
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--algorithm", action="append", default=None)
    parser.add_argument("--qwen-base-model", type=Path, default=DEFAULT_QWEN_BASE)
    parser.add_argument("--qwen-adapter-dir", type=Path, default=DEFAULT_QWEN_ADAPTER)
    parser.add_argument("--qwen-device-map", default="auto")
    parser.add_argument("--qwen-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--gemini-model",
        default="gemini-3.1-flash",
        help="Gemini API model id. If this 404s, list models and pass the exact available id.",
    )
    parser.add_argument(
        "--gemini-api-key-env",
        default="GEMINI_API_KEY",
        help="Environment variable containing the Gemini API key.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sleep", type=float, default=0.0, help="Delay between Gemini calls.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip records/models already present in the output JSONL.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call models; only validate data loading and print selected records.",
    )
    return parser.parse_args()


def canonical_task_type(task_type: str) -> str:
    task_type = task_type.strip().lower()
    if task_type in {"cls", "classification"}:
        return "classification"
    if task_type in {"rgs", "regression"}:
        return "regression"
    raise ValueError(f"unknown task_type: {task_type}")


def normalize_algorithm(value: Any, schema: dict[str, Any]) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    aliases = schema.get("algorithm_aliases", {})
    key = raw.lower().replace("-", "_")
    key = re.sub(r"\s+", "_", key)
    return aliases.get(key, raw)


def parse_user_metadata(user_prompt: str) -> dict[str, str]:
    fields = {}
    patterns = {
        "task_type": r"- task_type:\s*(.+)",
        "dataset": r"- dataset_name:\s*(.+)",
        "task_id": r"- task_id:\s*(.+)",
        "algorithm": r"- algorithm:\s*(.+)",
    }
    for name, pattern in patterns.items():
        match = re.search(pattern, user_prompt)
        if match:
            fields[name] = match.group(1).strip()
    return fields


def load_records(path: Path, schema: dict[str, Any]) -> list[EvalRecord]:
    records = []
    with path.open() as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            messages = obj["messages"]
            system_prompt = messages[0]["content"]
            user_prompt = messages[1]["content"]
            gold_payload = json.loads(messages[2]["content"])
            gold_config = gold_payload["warm_start_config"]
            meta = parse_user_metadata(user_prompt)
            task_type = canonical_task_type(meta["task_type"])
            algorithm = normalize_algorithm(meta["algorithm"], schema)
            if algorithm is None:
                raise ValueError(f"{path}:{idx}: missing algorithm")
            records.append(
                EvalRecord(
                    index=idx,
                    task_type=task_type,
                    task_id=meta["task_id"],
                    dataset=meta["dataset"],
                    algorithm=algorithm,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    gold_payload=gold_payload,
                    gold_config=gold_config,
                )
            )
    return records


def filter_records(records: list[EvalRecord], args: argparse.Namespace) -> list[EvalRecord]:
    selected = records
    if args.task_id:
        wanted = set(args.task_id)
        selected = [r for r in selected if r.task_id in wanted]
    if args.dataset:
        wanted = set(args.dataset)
        selected = [r for r in selected if r.dataset in wanted]
    if args.algorithm:
        wanted = {a.lower() for a in args.algorithm}
        selected = [r for r in selected if r.algorithm.lower() in wanted]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def schema_params(schema: dict[str, Any], task_type: str, algorithm: str) -> list[dict[str, str]]:
    return schema["task_types"][task_type]["algorithms"][algorithm]["parameters"]


def coerce_value(value: Any, value_type: str) -> Any:
    if value_type == "int":
        if isinstance(value, bool):
            raise ValueError("bool is not an int hyperparameter")
        return int(round(float(value)))
    if value_type == "float":
        return float(value)
    if value_type == "str":
        if value is None:
            return "None"
        if isinstance(value, bool):
            return str(value)
        return str(value)
    return value


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth = 0
    in_str = False
    esc = False
    for pos in range(start, len(text)):
        ch = text[pos]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                parsed = json.loads(text[start : pos + 1])
                if not isinstance(parsed, dict):
                    raise ValueError("parsed JSON is not an object")
                return parsed
    raise ValueError("unterminated JSON object")


def unwrap_config(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("warm_start_config"), dict):
        return payload["warm_start_config"]
    if isinstance(payload.get("config"), dict):
        return payload["config"]
    return payload


def normalize_config(
    payload_or_config: dict[str, Any],
    record: EvalRecord,
    schema: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_config = unwrap_config(payload_or_config)
    if not isinstance(raw_config, dict):
        raise ValueError("warm_start_config is not an object")

    algorithm = normalize_algorithm(raw_config.get("algorithm"), schema) or record.algorithm
    if algorithm != record.algorithm:
        raise ValueError(f"algorithm mismatch: expected {record.algorithm}, got {algorithm}")

    params = schema_params(schema, record.task_type, algorithm)
    by_short = {p["short_name"].lower(): p for p in params}
    by_name = {p["name"].lower(): p for p in params}

    canonical: dict[str, Any] = {"algorithm": algorithm}
    used_keys = {"algorithm"}
    for key, value in raw_config.items():
        if key == "algorithm":
            continue
        key_l = str(key).lower()
        param = by_name.get(key_l) or by_short.get(key_l)
        if param is None and ":" in key_l:
            param = by_short.get(key_l.split(":", 1)[1])
        if param is None:
            continue
        canonical[param["name"]] = coerce_value(value, param["type"])
        used_keys.add(str(key))

    required = [p["name"] for p in params]
    missing = [name for name in required if name not in canonical]
    extras = [
        str(k)
        for k in raw_config
        if str(k) not in used_keys and str(k).lower() not in by_name and str(k).lower() not in by_short
    ]
    diagnostics = {
        "required_keys": ["algorithm", *required],
        "missing_keys": missing,
        "ignored_extra_keys": extras,
        "schema_ok": not missing,
    }
    return canonical, diagnostics


def value_distance(pred: Any, gold: Any, value_type: str) -> tuple[float | None, bool | None]:
    if pred is None:
        return None, None
    if value_type in {"int", "float"}:
        try:
            p = float(pred)
            g = float(gold)
        except (TypeError, ValueError):
            return None, False
        denom = max(abs(g), 1.0)
        return abs(p - g) / denom, math.isclose(p, g, rel_tol=1e-9, abs_tol=1e-9)
    return None, str(pred) == str(gold)


def score_config(
    pred_config: dict[str, Any] | None,
    gold_config: dict[str, Any],
    record: EvalRecord,
    schema: dict[str, Any],
    parse_ok: bool,
    schema_ok: bool,
) -> dict[str, Any]:
    params = schema_params(schema, record.task_type, record.algorithm)
    required = [p["name"] for p in params]
    if pred_config is None:
        return {
            "parse_ok": parse_ok,
            "schema_ok": False,
            "exact_match": False,
            "key_recall": 0.0,
            "categorical_accuracy": None,
            "numeric_mean_relative_error": None,
            "score": 0.0,
        }

    present = [name for name in required if name in pred_config]
    key_recall = len(present) / len(required) if required else 1.0
    numeric_errors = []
    categorical_hits = []
    for param in params:
        name = param["name"]
        dist, hit = value_distance(pred_config.get(name), gold_config.get(name), param["type"])
        if dist is not None:
            numeric_errors.append(dist)
        if hit is not None and param["type"] == "str":
            categorical_hits.append(hit)
    numeric_error = sum(numeric_errors) / len(numeric_errors) if numeric_errors else None
    categorical_accuracy = (
        sum(1 for x in categorical_hits if x) / len(categorical_hits)
        if categorical_hits
        else None
    )
    exact_match = pred_config == normalize_gold_config(gold_config, record, schema)

    numeric_component = 1.0 / (1.0 + numeric_error) if numeric_error is not None else 1.0
    categorical_component = categorical_accuracy if categorical_accuracy is not None else 1.0
    score = (
        0.25 * float(parse_ok)
        + 0.25 * float(schema_ok)
        + 0.25 * key_recall
        + 0.15 * numeric_component
        + 0.10 * categorical_component
    )
    return {
        "parse_ok": parse_ok,
        "schema_ok": schema_ok,
        "exact_match": exact_match,
        "key_recall": key_recall,
        "categorical_accuracy": categorical_accuracy,
        "numeric_mean_relative_error": numeric_error,
        "score": score,
    }


def normalize_gold_config(
    gold_config: dict[str, Any], record: EvalRecord, schema: dict[str, Any]
) -> dict[str, Any]:
    normalized, _ = normalize_config(gold_config, record, schema)
    return normalized


def build_prompt(record: EvalRecord) -> str:
    return (
        f"{record.system_prompt}\n\n"
        f"{record.user_prompt}\n\n"
        "Return only a JSON object. Do not use markdown fences."
    )


def call_gemini(record: EvalRecord, args: argparse.Namespace) -> str:
    api_key = os.environ.get(args.gemini_api_key_env) or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"Missing Gemini API key. Set {args.gemini_api_key_env} or GOOGLE_API_KEY."
        )
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Install google-genai to query Gemini: pip install google-genai") from exc

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        temperature=args.temperature,
        max_output_tokens=args.max_new_tokens,
        response_mime_type="application/json",
    )
    response = client.models.generate_content(
        model=args.gemini_model,
        contents=build_prompt(record),
        config=config,
    )
    return response.text or ""


class QwenRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForImageTextToText, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Missing Qwen inference dependencies: torch, transformers, peft."
            ) from exc

        dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[args.qwen_dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(args.qwen_adapter_dir),
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForImageTextToText.from_pretrained(
            str(args.qwen_base_model),
            trust_remote_code=True,
            dtype=dtype,
            device_map=args.qwen_device_map,
        )
        self.model = PeftModel.from_pretrained(base, str(args.qwen_adapter_dir))
        self.model.eval()
        self.torch = torch
        self.args = args

    def __call__(self, record: EvalRecord) -> str:
        messages = [
            {"role": "system", "content": record.system_prompt},
            {"role": "user", "content": record.user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = self.model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.args.max_new_tokens,
                do_sample=self.args.temperature > 0,
                temperature=self.args.temperature if self.args.temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[-1]
        response_ids = output[:, prompt_len:]
        return self.tokenizer.batch_decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


def already_done(path: Path) -> set[tuple[int, str]]:
    done = set()
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            done.add((int(obj["index"]), obj["model"]))
    return done


def evaluate_raw_response(
    raw: str,
    record: EvalRecord,
    schema: dict[str, Any],
) -> dict[str, Any]:
    parse_ok = False
    schema_ok = False
    payload = None
    normalized = None
    diagnostics: dict[str, Any] = {}
    error = None
    try:
        payload = extract_json_object(raw)
        parse_ok = True
        normalized, diagnostics = normalize_config(payload, record, schema)
        schema_ok = bool(diagnostics.get("schema_ok"))
    except Exception as exc:  # noqa: BLE001 - diagnostics should capture all model failures.
        error = str(exc)
    metrics = score_config(
        normalized,
        record.gold_config,
        record,
        schema,
        parse_ok=parse_ok,
        schema_ok=schema_ok,
    )
    return {
        "raw_response": raw,
        "parsed_payload": payload,
        "normalized_config": normalized,
        "diagnostics": diagnostics,
        "error": error,
        "metrics": metrics,
    }


def summarize(path: Path) -> dict[str, Any]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    summary = {}
    for model, model_rows in by_model.items():
        metrics = [r["result"]["metrics"] for r in model_rows]
        numeric_errors = [
            m["numeric_mean_relative_error"]
            for m in metrics
            if m["numeric_mean_relative_error"] is not None
        ]
        categorical = [
            m["categorical_accuracy"]
            for m in metrics
            if m["categorical_accuracy"] is not None
        ]
        summary[model] = {
            "records": len(model_rows),
            "parse_rate": sum(m["parse_ok"] for m in metrics) / len(metrics),
            "schema_ok_rate": sum(m["schema_ok"] for m in metrics) / len(metrics),
            "exact_match_rate": sum(m["exact_match"] for m in metrics) / len(metrics),
            "mean_key_recall": sum(m["key_recall"] for m in metrics) / len(metrics),
            "mean_score": sum(m["score"] for m in metrics) / len(metrics),
            "mean_numeric_relative_error": (
                sum(numeric_errors) / len(numeric_errors) if numeric_errors else None
            ),
            "mean_categorical_accuracy": (
                sum(categorical) / len(categorical) if categorical else None
            ),
        }
    return summary


def main() -> None:
    args = parse_args()
    with args.schema.open() as f:
        schema = json.load(f)
    records = filter_records(load_records(args.data, schema), args)
    if args.dry_run:
        print(json.dumps({"selected_records": [r.__dict__ for r in records]}, indent=2))
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(args.output) if args.resume else set()
    qwen = QwenRunner(args) if "qwen" in args.models else None

    with args.output.open("a") as out:
        for record in records:
            for model_name in args.models:
                if (record.index, model_name) in done:
                    continue
                started = time.time()
                if model_name == "qwen":
                    assert qwen is not None
                    raw = qwen(record)
                    model_id = str(args.qwen_adapter_dir)
                else:
                    raw = call_gemini(record, args)
                    model_id = args.gemini_model
                    if args.sleep:
                        time.sleep(args.sleep)

                result = evaluate_raw_response(raw, record, schema)
                row = {
                    "index": record.index,
                    "model": model_name,
                    "model_id": model_id,
                    "task_type": record.task_type,
                    "task_id": record.task_id,
                    "dataset": record.dataset,
                    "algorithm": record.algorithm,
                    "gold_config": normalize_gold_config(record.gold_config, record, schema),
                    "result": result,
                    "latency_sec": round(time.time() - started, 3),
                }
                out.write(json.dumps(row, sort_keys=True) + "\n")
                out.flush()
                print(
                    json.dumps(
                        {
                            "index": row["index"],
                            "model": model_name,
                            "dataset": row["dataset"],
                            "algorithm": row["algorithm"],
                            "metrics": result["metrics"],
                        },
                        sort_keys=True,
                    )
                )

    summary_path = args.output.with_suffix(".summary.json")
    summary = summarize(args.output)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
