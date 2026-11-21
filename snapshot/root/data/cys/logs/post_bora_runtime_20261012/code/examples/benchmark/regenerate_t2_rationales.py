#!/usr/bin/env python3
"""Regenerate both fixed T2 candidates with a label-blind, frozen local teacher.

prepare -> generate (independent shards) -> finalize. No training or BO runs.
Failed generations are recorded and must be resolved before an export is ready.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import time

try:
    from .p5_response_format import (
        FORMAT_VERSION, RESPONSE_FORMAT, format_response, parse_response, rationale_conversations,
    )
except ImportError:
    from p5_response_format import (
        FORMAT_VERSION, RESPONSE_FORMAT, format_response, parse_response, rationale_conversations,
    )

TEACHER_SYSTEM = (
    "You explain fixed hyperparameter configurations for machine learning experiments. "
    "Use only the supplied task, search space, and prior observations. "
    "Write exactly two concise sentences, at most 80 words total. Explain one or two "
    "actual candidate settings using the supplied candidate-specific facts, then "
    "identify a relevant trade-off or uncertainty requiring validation. "
    "Use exact parameter names and values when mentioning a candidate setting. "
    "Avoid high/low/moderate/strong/weak labels; give the number and explain its mechanism. "
    "Do not claim a setting follows a successful trend or matches top trials. A few "
    "observations cannot establish that a parameter causes better performance. "
    "Do not compare parameter numbers or restate historical trial results. "
    "Use the candidate-specific facts as the authoritative explanation of each setting. "
    "Do not introduce a mechanism that is absent from those facts. "
    "Never assert that a concrete setting provides sufficient capacity, prevents "
    "overfitting or underfitting, or guarantees convergence; those outcomes are unknown. "
    "Do not infer decision-boundary complexity or minority-class behavior from dataset size. "
    "Distinguish evidence from speculation; do not invent an evaluation result. "
    "Do not change the candidate, claim that its result is known, or describe it as "
    "accepted, rejected, preferred, repaired, good, or bad. Do not invent a comparison "
    "candidate. Return only the explanation in plain text, without headers, JSON, "
    "markdown, or thinking tags."
)

PARAMETER_MECHANISMS = (
    "More boosting iterations add capacity and computation; a larger learning rate makes "
    "each boosting update larger and may increase instability. Fewer iterations can underfit. "
    "Increasing max_depth or num_leaves permits more complex trees and can increase variance. "
    "Increasing min_samples_leaf, min_child_samples, min_samples_split, min_child_weight, "
    "gamma, or regularization strength restricts tree fitting, usually reducing variance "
    "at the cost of more bias. A small leaf threshold allows small leaves; it does not "
    "stabilize them. A split threshold cannot override the minimum size of child leaves. "
    "For sklearn forests, max_features is the fraction of features considered at each "
    "split, not a fixed subset per tree. A smaller fraction can decorrelate trees but "
    "also weaken individual splits. Bootstrap samples observations with replacement; "
    "averaging trees can reduce variance. Absolute-error/MAE loss reduces the influence "
    "of large residuals compared with squared error. Poisson criterion models Poisson "
    "deviance for suitable nonnegative targets; it is not an outlier-robust loss. "
    "Criterion=log_loss is a tree split criterion, not the reported validation metric. "
    "LightGBM max_depth also caps effective leaves; row subsampling only operates when "
    "bagging/subsample frequency enables it, which may be unspecified here. "
    "No individual hyperparameter setting guarantees convergence or improvement."
)

PARAMETER_FACTS = {
    "learning_rate": "scales the size of each boosting update; smaller steps can need more iterations, while larger steps can be less stable",
    "n_estimators": "sets the number of boosting iterations and therefore computation and fitting capacity; too few can underfit",
    "iterations": "sets the number of boosting iterations and therefore computation and fitting capacity; too few can underfit",
    "max_depth": "caps tree depth when set to a positive integer; deeper trees permit more complex fits and may overfit",
    "depth": "caps tree depth; deeper trees permit more complex fits and may overfit",
    "num_leaves": "caps the number of leaves per tree, subject to other constraints; more leaves permit more complex fits",
    "min_samples_leaf": "requires at least this many training samples in each leaf; larger thresholds constrain fitting more, while smaller thresholds allow small leaves",
    "min_child_samples": "requires approximately this many training samples in each leaf; larger thresholds constrain fitting more, while smaller thresholds allow small leaves",
    "min_samples_split": "requires at least this many training samples before a node can split; each resulting leaf must still satisfy min_samples_leaf",
    "min_child_weight": "sets a minimum sum of instance Hessian weights for a child; increasing it restricts splitting, and it is not simply a minimum sample count",
    "max_features": "controls the fraction of features considered at each split for a float value; a smaller fraction can decorrelate trees but weaken individual splits",
    "colsample_bytree": "controls the fraction of features sampled per tree; reducing it adds randomness and can weaken individual trees",
    "subsample": "controls sampling of training rows, not features; stochastic sampling can regularize fitting but reduce data per update; for LightGBM its effect also requires a nonzero bagging frequency",
    "bootstrap": "True samples training rows with replacement for each tree, False does not; averaging forest predictions can reduce variance in either case",
    "gamma": "sets the minimum loss reduction required for a split; increasing it restricts splitting",
    "reg_alpha": "sets L1 regularization on leaf weights; increasing it penalizes those weights more strongly",
    "reg_lambda": "sets L2 regularization on leaf weights; increasing it penalizes those weights more strongly",
    "l2_leaf_reg": "sets L2 regularization on leaf weights; increasing it penalizes those weights more strongly",
    "criterion": "specifies the tree split criterion; absolute_error uses absolute deviation, squared_error uses squared deviation, poisson uses Poisson deviance for suitable nonnegative targets, and gini/entropy/log_loss measure classification impurity; it is not the validation metric",
    "loss": "specifies the fitting loss; squared error is more sensitive to large residuals than absolute error, and classification log loss scores probability predictions",
    "loss_function": "specifies the fitting loss; MAE is less sensitive to large residuals than RMSE; neither guarantees improvement on the validation metric",
}


def candidate_facts(config):
    lines = []
    for name, value in sorted(config.items()):
        mechanism = PARAMETER_FACTS.get(name.rsplit(":", 1)[-1])
        if mechanism:
            lines.append(f"- {name} = {canonical(value)}: {mechanism}.")
    return "\n".join(lines)


def metric_context(metadata):
    metric = metadata["metric"]
    if metric == "bal_acc":
        description = "balanced accuracy; higher raw values are better"
    elif metric == "mse":
        description = "mean squared error; lower raw values are better"
    else:
        raise ValueError(f"Unreviewed metric: {metric}")
    return (f"Metric: {description}. Historical rel_score is normalized so higher values "
            "are better; it is not the raw metric and does not give a measured score for this candidate.")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n")
    temporary.replace(path)


def save_jsonl(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")
    temporary.replace(path)


def task_key(metadata):
    return tuple(metadata[name] for name in ("task_type", "dataset", "algorithm"))


def visible_configs(context):
    return [json.loads(line.split(", config=", 1)[1])
            for line in context.splitlines() if ", config=" in line]


def prepare(args):
    if (args.output / "manifest.json").exists():
        raise RuntimeError("Output already prepared; use a new version directory")
    rows = read_jsonl(args.source)
    train = {task_key(row["metadata"]) for row in read_jsonl(args.train_manifest)}
    requests, pairs, quarantine = [], [], []
    for index, row in enumerate(rows):
        meta = row["metadata"]
        if task_key(meta) not in train:
            raise ValueError(f"Non-training task: {task_key(meta)}")
        conversations, context = rationale_conversations(row["conversations"])
        expected_type = "classification" if meta["task_type"] == "cls" else "regression"
        if f"- task_type: {expected_type}\n" not in context:
            raise ValueError(f"Task-type mismatch: {index}")
        visible = {canonical(config) for config in visible_configs(context)}
        exposed = [side for side in ("chosen", "rejected") if canonical(meta[side + "_config"]) in visible]
        if exposed:
            quarantine.append({"source_index": index, "task": task_key(meta),
                               "reason": "candidate_result_visible_in_history", "sides": exposed})
            continue
        pair = {"source_index": index, "conversations": conversations, "request_ids": {}}
        for side in ("chosen", "rejected"):
            config = meta[side + "_config"]
            _, old_config = parse_response(row[side]["value"])
            if canonical(config) != canonical(old_config):
                raise ValueError(f"Source config disagrees with metadata: {index}/{side}")
            messages = [{"role": "system", "content": TEACHER_SYSTEM},
                        {"role": "user", "content": metric_context(meta) + "\n\n" + context + "\n\nFIXED CANDIDATE:\n" + canonical(config) +
                         "\n\nCANDIDATE-SPECIFIC FACTS:\n" + candidate_facts(config)}]
            request_id = hashlib.sha256(canonical(messages).encode()).hexdigest()
            requests.append({"id": request_id, "source_index": index, "side": side,
                             "config": config, "messages": messages})
            pair["request_ids"][side] = request_id
        pairs.append(pair)
    requests.sort(key=lambda row: row["id"])
    pilot_ids, pilot_groups = [], set()
    for pair in pairs:
        meta = rows[pair["source_index"]]["metadata"]
        group = (meta["task_type"], meta["algorithm"])
        if group not in pilot_groups:
            pilot_groups.add(group)
            pilot_ids.extend(pair["request_ids"].values())
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = {"source.jsonl": args.source, "train_manifest.jsonl": args.train_manifest,
              "sft_reference.jsonl": args.sft_reference,
              "regenerate_t2_rationales.py": Path(__file__),
              "p5_response_format.py": Path(__file__).with_name("p5_response_format.py")}
    input_hashes = {}
    for name, source in inputs.items():
        digest = sha256(source)
        shutil.copyfile(source, args.output / name)
        if sha256(args.output / name) != digest or sha256(source) != digest:
            raise RuntimeError(f"Input changed during snapshot: {source}")
        input_hashes[name] = {"source": str(source.resolve()), "sha256": digest}
    for row in read_jsonl(args.sft_reference):
        parse_response(row["messages"][-1]["content"])
    save_jsonl(args.output / "requests.jsonl", requests)
    save_jsonl(args.output / "pairs.jsonl", pairs)
    save_json(args.output / "quarantine.json", quarantine)
    save_json(args.output / "pilot_ids.json", pilot_ids)
    save_json(args.output / "manifest.json", {
        "status": "prepared", "response_format": RESPONSE_FORMAT, "format_version": FORMAT_VERSION,
        "source_pairs": len(rows), "eligible_pairs": len(pairs), "requests": len(requests),
        "quarantined_pairs": len(quarantine), "quarantine_decision": "pending",
        "input_files": input_hashes, "request_sha256": sha256(args.output / "requests.jsonl"),
        "pairs_sha256": sha256(args.output / "pairs.jsonl"),
        "training_tasks": len({task_key(rows[p["source_index"]]["metadata"]) for p in pairs}),
        "teacher_system": TEACHER_SYSTEM, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    print(f"Prepared {len(pairs)} pairs / {len(requests)} requests; quarantined {len(quarantine)}", flush=True)


def rationale_errors(text):
    errors = []
    if not 15 <= len(text.split()) <= 150:
        errors.append("word_count_outside_15_150")
    if re.search(r"(?i)\b(accepted|rejected|preferred|repaired|blamed|problematic)\b", text):
        errors.append("preference_label_wording")
    if any(marker in text for marker in ("<think", "</think", "JSON:", "```", "{", "}")):
        errors.append("unexpected_format")
    if '`' in text:
        errors.append("markdown_in_explanation")
    if re.search(r"(?i)(?:this (?:configuration|candidate)|it) (?:achieved|scored|obtained|improved the best)", text):
        errors.append("claimed_candidate_outcome")
    if "expected to underperform the current best trial" in text.lower():
        errors.append("legacy_fallback_wording")
    if text and text[-1] not in ".!?":
        errors.append("possibly_truncated")
    return errors


def teacher_provenance(model_path):
    files = {path.name: sha256(path) for path in sorted(model_path.iterdir())
             if path.is_file() and path.suffix in (".json", ".safetensors", ".jinja", ".txt")}
    return {"model": str(model_path.resolve()), "adapter": None, "files_sha256": files,
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "tokenizers", "safetensors")}}


def decode_generated_batch(tokenizer, outputs, prompt_width):
    return [tokenizer.decode(output[prompt_width:], skip_special_tokens=True).strip()
            for output in outputs]


def generate(args):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    manifest = json.loads((args.output / "manifest.json").read_text())
    if sha256(args.output / "requests.jsonl") != manifest["request_sha256"]:
        raise RuntimeError("Generation requests changed after preparation")
    requests = read_jsonl(args.output / "requests.jsonl")
    requests = [row for i, row in enumerate(requests) if i % args.shards == args.shard]
    if args.only_ids:
        allowed = set(json.loads(args.only_ids.read_text()))
        requests = [row for row in requests if row["id"] in allowed]
    result_path = args.output / f"generations_{args.shard}.jsonl"
    done = {row["id"] for row in read_jsonl(result_path) if row["valid"]} if result_path.exists() else set()
    pending = [row for row in requests if row["id"] not in done]
    if args.limit:
        pending = pending[:args.limit]
    if not pending:
        print("Nothing pending", flush=True)
        return
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    provenance = teacher_provenance(args.model)
    provenance.update({"seed": args.seed, "temperature": args.temperature, "do_sample": args.temperature > 0,
                       "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size,
                       "enable_thinking": False, "padding_side": "left", "shard": args.shard,
                       "shards": args.shards, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                       "cpu_affinity": sorted(os.sched_getaffinity(0)),
                       "chat_template": tokenizer.chat_template,
                       "generator_sha256": sha256(Path(__file__)),
                       "request_sha256": manifest["request_sha256"]})
    provenance_id = hashlib.sha256(canonical(provenance).encode()).hexdigest()
    save_json(args.output / f"teacher_{provenance_id}.json", provenance)
    print(f"Loading frozen teacher for {len(pending)} requests; provenance={provenance_id}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation="sdpa",
    ).eval()
    model.requires_grad_(False)
    print(f"Loaded {type(model).__name__}; trainable parameters=0", flush=True)
    started = time.monotonic()
    completed = 0
    with result_path.open("a", encoding="utf-8") as stream:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            prompts = [tokenizer.apply_chat_template(row["messages"], tokenize=False,
                       add_generation_prompt=True, enable_thinking=False) for row in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)
            prompt_width = inputs["input_ids"].shape[1]
            if prompt_width + args.max_new_tokens > 8192:
                raise RuntimeError("Prompt exceeds generation context budget; refusing truncation")
            kwargs = {"do_sample": args.temperature > 0}
            if args.temperature > 0:
                kwargs.update(temperature=args.temperature, top_p=0.9)
            t0 = time.monotonic()
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                         pad_token_id=tokenizer.pad_token_id, **kwargs)
            # Generated tokens begin after the common padded input width for every row.
            decoded = decode_generated_batch(tokenizer, outputs, prompt_width)
            for row, output, raw in zip(batch, outputs, decoded):
                tokens = output[prompt_width:].tolist()
                rationale = raw.removeprefix("Rationale:").replace('`', '').strip()
                errors = rationale_errors(rationale)
                record = {"id": row["id"], "source_index": row["source_index"], "side": row["side"],
                          "raw": raw, "rationale": rationale, "valid": not errors, "errors": errors,
                          "generated_token_ids": tokens, "teacher_provenance": provenance_id,
                          "batch_seconds": time.monotonic() - t0,
                          "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                stream.write(canonical(record) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            completed += len(batch)
            print(f"shard={args.shard} generated={completed}/{len(pending)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    del model
    torch.cuda.empty_cache()


def finalize(args):
    manifest = json.loads((args.output / "manifest.json").read_text())
    for name, meta in manifest["input_files"].items():
        if sha256(args.output / name) != meta["sha256"]:
            raise RuntimeError(f"Frozen input changed: {name}")
    for name, key in (("requests.jsonl", "request_sha256"), ("pairs.jsonl", "pairs_sha256")):
        if sha256(args.output / name) != manifest[key]:
            raise RuntimeError(f"Prepared input changed: {name}")
    if manifest["quarantined_pairs"] and not args.exclude_quarantined:
        raise RuntimeError("Quarantined-pair policy needs resolution before export")
    generations = {}
    rejected_generations = []
    for path in sorted(args.output.glob("generations_*.jsonl")):
        for row in read_jsonl(path):
            if row["valid"] and not rationale_errors(row["rationale"]):
                generations[row["id"]] = row
            else:
                rejected_generations.append(row)
    requests = read_jsonl(args.output / "requests.jsonl")
    missing = sorted({row["id"] for row in requests} - set(generations))
    save_json(args.output / "pending_ids.json", missing)
    if missing:
        raise RuntimeError(f"{len(missing)} explanations still missing or invalid")
    source = read_jsonl(args.output / "source.jsonl")
    full, train = [], []
    for pair in read_jsonl(args.output / "pairs.jsonl"):
        old = source[pair["source_index"]]
        meta = dict(old["metadata"])
        meta["source_stage"] = meta.get("stage")
        meta["stage"] = "ORPO_T2"
        meta["source_hamming"] = meta.get("hamming")
        meta["hamming"] = sum(meta["chosen_config"].get(key) != meta["rejected_config"].get(key)
                              for key in set(meta["chosen_config"]) | set(meta["rejected_config"]))
        meta["selection_statistics_scope"] = "original_measured_good_vs_rejected"
        meta["source_good_score"] = meta.pop("chosen_score", None)
        meta["chosen_score"] = None
        meta["chosen_score_status"] = "not_verified_for_repaired_candidate"
        meta["preference_source"] = "synthetic_partial_repair"
        meta["response_format"] = RESPONSE_FORMAT
        meta["source_index"] = pair["source_index"]
        meta["rationale_generation"] = {}
        row = {"conversations": pair["conversations"], "metadata": meta}
        for side in ("chosen", "rejected"):
            generation = generations[pair["request_ids"][side]]
            config = meta[side + "_config"]
            text = format_response(generation["rationale"], config)
            if parse_response(text)[1] != config:
                raise RuntimeError("Configuration changed in export")
            row[side] = {"from": "gpt", "value": text}
            meta["rationale_generation"][side] = {
                "request_id": generation["id"], "teacher_provenance": generation["teacher_provenance"],
            }
        full.append(row)
        train.append({key: row[key] for key in ("conversations", "chosen", "rejected")})
    save_jsonl(args.output / "p5_attr_t2_rationale.jsonl", full)
    save_jsonl(args.output / "p5_attr_t2_rationale_train.jsonl", train)
    save_json(args.output / "dataset_info.json", {"p5_attr_t2_rationale": {
        "file_name": "p5_attr_t2_rationale_train.jsonl", "formatting": "sharegpt", "ranking": True,
        "columns": {"messages": "conversations", "chosen": "chosen", "rejected": "rejected"},
    }})
    manifest.update({"status": "exported_awaiting_quality_review", "exported_pairs": len(full),
                     "quarantine_decision": "excluded" if args.exclude_quarantined else "none",
                     "failed_generation_attempts": len(rejected_generations),
                     "exports_sha256": {name: sha256(args.output / name) for name in
                      ("p5_attr_t2_rationale.jsonl", "p5_attr_t2_rationale_train.jsonl", "dataset_info.json")}})
    shutil.copyfile(Path(__file__), args.output / "exporter_source.py")
    manifest["exporter_code_sha256"] = sha256(args.output / "exporter_source.py")
    save_json(args.output / "manifest.json", manifest)
    print(f"Exported {len(full)} pairs with unchanged configurations and regenerated explanations", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--train-manifest", type=Path, required=True)
    prep.add_argument("--sft-reference", type=Path, required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--model", type=Path, required=True)
    gen.add_argument("--shard", type=int, default=0)
    gen.add_argument("--shards", type=int, default=1)
    gen.add_argument("--batch-size", type=int, default=8)
    gen.add_argument("--max-new-tokens", type=int, default=256)
    gen.add_argument("--temperature", type=float, default=0.0)
    gen.add_argument("--seed", type=int, default=1)
    gen.add_argument("--limit", type=int, default=0)
    gen.add_argument("--only-ids", type=Path)
    fin = sub.add_parser("finalize")
    fin.add_argument("--exclude-quarantined", action="store_true")
    for command in (prep, gen, fin):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate" and (args.shards < 1 or not 0 <= args.shard < args.shards or args.batch_size < 1):
        parser.error("Invalid shard or batch size")
    {"prepare": prepare, "generate": generate, "finalize": finalize}[args.command](args)


if __name__ == "__main__":
    main()
