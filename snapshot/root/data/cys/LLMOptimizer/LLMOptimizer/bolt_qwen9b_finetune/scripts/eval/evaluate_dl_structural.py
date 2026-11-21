#!/usr/bin/env python3
"""Structural evaluation for DL optimizer warm-start models.

Evaluates generated predictions on:
- JSON parse success rate
- Valid optimizer_family (must exist in search space)
- Exact match with gold answer
- Hyperparameter key overlap (Jaccard)
- Numeric parameter MAE (when family matches)
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
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
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def generation_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    return [message for message in record["messages"] if message["role"] in {"system", "user"}]


def generate_predictions(model, tokenizer, records, max_new_tokens=768) -> list[dict[str, Any]]:
    rows = []
    for index, record in enumerate(records):
        prompt = tokenizer.apply_chat_template(
            generation_messages(record),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        row = {"index": index, "raw_response": raw}
        try:
            row["config"] = extract_json_object(raw)
        except Exception as exc:
            row["parse_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def evaluate_structural(predictions: list[dict], gold_records: list[dict]) -> dict[str, Any]:
    total = len(gold_records)
    parsed = 0
    valid_family = 0
    exact_match = 0
    family_match = 0
    key_overlap_scores = []
    numeric_maes = []
    errors = []

    for pred, gold_record in zip(predictions, gold_records):
        gold_assistant = json.loads(gold_record["messages"][2]["content"])
        gold_family = gold_assistant.get("optimizer_family")
        gold_params = gold_assistant.get("hyperparameters", {})

        user_msg = json.loads(gold_record["messages"][1]["content"])
        search_space = user_msg.get("optimizer_search_space", {})
        valid_families = set(search_space.keys())

        idx = pred["index"]

        if "parse_error" in pred:
            errors.append({"index": idx, "error": pred["parse_error"]})
            continue

        parsed += 1
        config = pred.get("config", {})

        # Handle nested config
        if "warm_start_config" in config:
            config = config["warm_start_config"]
        if "config" in config and isinstance(config["config"], dict):
            config = config["config"]

        pred_family = config.get("optimizer_family")
        pred_params = config.get("hyperparameters", {})

        if pred_family in valid_families:
            valid_family += 1

        if pred_family == gold_family:
            family_match += 1

        if config == gold_assistant:
            exact_match += 1

        # Key overlap
        gold_keys = set(gold_params.keys())
        pred_keys = set(pred_params.keys())
        if gold_keys:
            overlap = len(gold_keys & pred_keys) / len(gold_keys)
            key_overlap_scores.append(overlap)

        # Numeric MAE for matching family
        if pred_family == gold_family and gold_params:
            mae_vals = []
            for key in gold_keys & pred_keys:
                g_val = gold_params[key]
                p_val = pred_params[key]
                if isinstance(g_val, (int, float)) and isinstance(p_val, (int, float)):
                    mae_vals.append(abs(float(g_val) - float(p_val)))
            if mae_vals:
                numeric_maes.append(sum(mae_vals) / len(mae_vals))

    result = {
        "total": total,
        "parsed": parsed,
        "parse_rate": parsed / total if total else 0,
        "valid_family": valid_family,
        "valid_family_rate": valid_family / total if total else 0,
        "family_match": family_match,
        "family_match_rate": family_match / total if total else 0,
        "exact_match": exact_match,
        "exact_match_rate": exact_match / total if total else 0,
        "mean_key_overlap": sum(key_overlap_scores) / len(key_overlap_scores) if key_overlap_scores else 0,
        "mean_numeric_mae": sum(numeric_maes) / len(numeric_maes) if numeric_maes else None,
        "errors": errors[:10],  # cap errors
    }
    return result


def load_model(base_model_path: Path, adapter_dir: Path | None = None):
    tokenizer_path = (
        adapter_dir
        if adapter_dir is not None and (adapter_dir / "tokenizer_config.json").exists()
        else base_model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    if adapter_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, adapter_dir)
    else:
        model = base
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--model-label", type=str, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    args = parser.parse_args()

    records = load_jsonl(args.test_data)
    print(f"[{args.model_label}] Loaded {len(records)} test records", file=sys.stderr)

    model, tokenizer = load_model(args.base_model, args.adapter_dir)
    print(f"[{args.model_label}] Model loaded", file=sys.stderr)

    predictions = generate_predictions(model, tokenizer, records, args.max_new_tokens)
    print(f"[{args.model_label}] Generated {len(predictions)} predictions", file=sys.stderr)

    # Save predictions
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    # Evaluate
    result = evaluate_structural(predictions, records)
    result["model"] = args.model_label

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
