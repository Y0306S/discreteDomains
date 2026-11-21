#!/usr/bin/env python3
"""Generate three warm starts at temperature 0.9, accepting duplicate proposals."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hf_qwen_client import HuggingFaceQwenClient
from mindware.components.experience.source_gp import configuration, digest, json_digest
from mindware.components.experience.surrogate_landscape import get_algo_cs
from mindware.components.optimizers.ml_checkpoint import atomic_json
from mindware.components.optimizers.pt4opt_llm_optimizer import parse_llm_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-proposals-per-slot", type=int, default=10)
    parser.add_argument("--task-prompts", type=Path, default=Path(
        "/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/ml_test_20_fixed.jsonl"))
    parser.add_argument("--adapter", default="/root/blockdata/model-weights/llamafactory_qwen35_9b_lora_recommended4")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("Invalid shard index/count")
    from transformers import set_seed
    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    original = args.campaign / "warm_start_presets.json"
    presets = json.loads(original.read_text())
    tasks = json.loads((args.campaign / "tasks_132.json").read_text())
    prompts = {f"{r['metadata']['dataset']}:{r['metadata']['algorithm']}": r
               for r in (json.loads(s) for s in args.task_prompts.read_text().splitlines() if s.strip())}
    protocol = json.loads((args.campaign / "protocol.json").read_text())
    client = HuggingFaceQwenClient(protocol["base_model"], args.adapter,
                                  temperature=args.temperature, max_new_tokens=768)
    audit = {"input_sha256": digest(original), "scope": "all", "temperature": args.temperature,
             "seed": args.seed, "adapter": args.adapter, "proposals": [], "replacements": [],
             "shard_index": args.shard_index, "shard_count": args.shard_count, "tasks": []}
    atomic_json(args.output / "original_presets.json", presets)
    generated = {}
    for task_index, task in enumerate(tasks):
        if task_index % args.shard_count != args.shard_index:
            continue
        key = f"{task['dataset']}:{task['algorithm']}"
        cs = get_algo_cs(task["algorithm"], "classification" if task["task_type"] == "cls" else "regression")
        seen, slots = set(), range(3)
        client._ensure_loaded()
        set_seed(args.seed + task_index)
        audit["adapter_verification"] = client.adapter_verification
        messages = prompts[key]["messages"]
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        base_user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
        for i in slots:
            accepted = False
            for attempt in range(args.max_proposals_per_slot):
                user = base_user
                raw_reply = client.complete(system, user)
                record = {"task": key, "slot": i + 1, "attempt": attempt + 1,
                          "system_prompt": system, "user_prompt": user, "raw_reply": raw_reply, "accepted": False}
                try:
                    raw, rationale, _ = parse_llm_json(raw_reply)
                    config = configuration(cs, raw, task["algorithm"])
                    signature = json_digest(config.get_dictionary())
                    flat = {f"{task['algorithm']}:{name}": value for name, value in config.get_dictionary().items()}
                    flat["algorithm"] = task["algorithm"]
                    audit["replacements"].append({"task": key, "slot": i + 1,
                                                  "previous": presets[key][i], "replacement": flat})
                    presets[key][i] = flat
                    record["duplicate_of_earlier_slot"] = signature in seen
                    seen.add(signature)
                    record.update(accepted=True, configuration=flat, rationale=rationale)
                    accepted = True
                except (ValueError, TypeError, RuntimeError) as exc:
                    record["rejection"] = str(exc)
                audit["proposals"].append(record)
                atomic_json(args.output / "audit.json", audit)
                if accepted:
                    print(f"Generated {key} slot {i + 1}; duplicate={record['duplicate_of_earlier_slot']}", flush=True)
                    break
            if not accepted:
                raise RuntimeError(f"No valid configuration for {key} slot {i + 1}")
        generated[key] = presets[key]
        audit["tasks"].append({"task": key, "seed": args.seed + task_index, "distinct_configurations": len(seen)})
        atomic_json(args.output / "presets.json", generated)
        atomic_json(args.output / "audit.json", audit)
    atomic_json(args.output / "presets.json", generated)
    audit["output_sha256"] = digest(args.output / "presets.json")
    audit["complete"] = True
    atomic_json(args.output / "audit.json", audit)
    print(json.dumps({"replacements": len(audit["replacements"]), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
