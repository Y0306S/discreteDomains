#!/usr/bin/env python3
"""Prompt-only preflight for v14 RL collection: no generation, no ML evaluations.

For every allow-listed task and every v14 checkpoint, build the exact collector
prompt and check: Dataset-profile label, recent-trials block, similar-task cards
only at the rerank checkpoints, no card from a test task, and prompt length.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))

import run_p5_rl_bo_prefix_b as collector  # noqa: E402
from experiment_utils import apply_prompt_settings  # noqa: E402

CARD = re.compile(r"Similar task: ([^\n(]+?) \(([a-z_]+)\)")


def collector_args(task, a):
    argv = [
        "--task-type", task["task_type"], "--dataset", task["dataset"], "--algorithm", task["algorithm"],
        "--collect-schedule", "rs_mix_v14", "--output-dir", str(a.output_dir / "unused"),
        "--qwen-adapter-path", str(a.adapter), "--source-gp-cache", str(a.source_gp_cache),
        "--task-allowlist", str(a.allowlist), "--train-manifest", str(a.train_manifest),
        "--test-manifest", str(a.test_manifest),
    ]
    old, sys.argv = sys.argv, ["collector"] + argv
    try:
        args = collector.parse_args()
    finally:
        sys.argv = old
    args.llm_backends = ["qwen-sft6"]
    args.optimizer = ["ptllm_qwen"]
    args.metric = args.metric or ("bal_acc" if args.task_type == "cls" else "mse")
    args.prompt_variant = collector.collect_schedule_steps(args)[0][2]
    apply_prompt_settings(args, args.prompt_variant)
    args.llm_n_initial_samples = 0
    return args


def check_task(task, a, tokenizer, test_tasks):
    args = collector_args(task, a)
    algo = task["algorithm"]
    collector.validate_task_split(args, algo)
    bo_path = collector.bo_history_path(args.task_type, args.dataset, algo, metric=args.metric,
                                        root=Path(args.bo_history_root))
    all_obs = collector.load_bo_observations(bo_path, algorithm=algo, max_n=args.bo_max_n)
    variant_name = collector.expand_optimizer_variants(["ptllm_qwen"], args)[0]
    opt, _ = collector.make_optimizer(args, variant_name, all_obs, a.output_dir / "unused", "preflight")
    rows = []
    for k, scenario, prompt_id in collector.collect_schedule_steps(args):
        messages, meta, _ = collector.build_prefix_prompt(opt, args, all_obs, algo, k, prompt_id,
                                                          use_mix=True, v14=True)
        user = messages[1]["content"]
        cards = CARD.findall(user)
        leaked = [c for c in cards if (task["task_type"], c[0].strip().lower(), c[1]) in test_tasks]
        own = [c for c in cards if c[0].strip().lower() == task["dataset"].lower()]
        n_tokens = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        rows.append({
            "task": f"{task['task_type']}/{task['dataset']}/{algo}", "k": k, "scenario": scenario,
            "prompt_variant": prompt_id, "cards": [f"{d.strip()} ({al})" for d, al in cards],
            "n_cards": len(cards), "leaked_test_cards": leaked, "own_dataset_cards": own,
            "has_recent": '"recent"' in user, "tokens": n_tokens,
            "retrieval_meta": {key: meta[key] for key in meta if "retriev" in key or "rs_" in key},
            "user_prompt": user if a.keep_prompts else None,
        })
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--allowlist", type=Path, default=Path("/root/reports/rl_plan_20261005/train32_tasks.json"))
    p.add_argument("--train-manifest", type=Path, default=Path(
        "/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/ml_train_80_fixed.jsonl"))
    p.add_argument("--test-manifest", type=Path, default=Path(
        "/root/data/cys/llmoptimizer_benchmark/benchmark_data/ml_test_20_fixed_48h_20260925/tasks_132.json"))
    p.add_argument("--adapter", type=Path, default=Path(
        "/root/blockdata/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_orpo_t2_rationale_20261005"))
    p.add_argument("--source-gp-cache", type=Path, default=Path(
        "/root/blockdata/llmoptimizer_benchmark/benchmark_data/ml_test_20_fixed_ckpt32_rgpe_20261002/source_gp_train80_v1"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--only", default=None, help="Run a single task key task_type/dataset/algorithm")
    p.add_argument("--keep-prompts", action="store_true")
    a = p.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(a.adapter))
    test_tasks = {(t["task_type"], t["dataset"].lower(), t["algorithm"])
                  for t in json.loads(a.test_manifest.read_text())}
    tasks = json.loads(a.allowlist.read_text())
    if a.only:
        tasks = [t for t in tasks if f"{t['task_type']}/{t['dataset']}/{t['algorithm']}" == a.only]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    out = a.output_dir / "prompt_preflight.jsonl"
    problems = []
    with out.open("w") as f:
        for task in tasks:
            for row in check_task(task, a, tokenizer, test_tasks):
                f.write(json.dumps(row, default=str) + "\n")
                rerank = row["scenario"] == "rerank"
                if row["leaked_test_cards"] or row["own_dataset_cards"]:
                    problems.append((row["task"], row["k"], "card leak", row["leaked_test_cards"] or row["own_dataset_cards"]))
                if not rerank and row["n_cards"]:
                    problems.append((row["task"], row["k"], "cards outside rerank window", row["n_cards"]))
                if not row["has_recent"]:
                    problems.append((row["task"], row["k"], "missing recent-trials block", None))
                print(f"{row['task']:48s} k={row['k']:3d} {row['scenario']:8s} cards={row['n_cards']} "
                      f"recent={row['has_recent']} tokens={row['tokens']}", flush=True)
    print(json.dumps({"problems": problems, "output": str(out)}, indent=1, default=str))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
