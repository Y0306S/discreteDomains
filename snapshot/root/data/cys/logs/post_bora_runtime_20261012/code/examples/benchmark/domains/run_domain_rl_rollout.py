#!/usr/bin/env python3
"""P5-style RL rollout buffer on a domain search (TUNE tasks only).

Writes JSONL compatible with examples/benchmark/train_p5_reinforce.py:
  messages + completion + reward.reward (Δbest).

Use a distinct LoRA adapter per domain (see lora_adapters.json). Does not
reuse the tabular LoRA as the policy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples" / "benchmark"))

from rl_reward_utils import RewardWeights, compute_trial_reward  # noqa: E402

from mindware.domains import get_adapter
from mindware.domains.paths import domain_bench, domain_history
from mindware.domains.prompts import build_domain_messages
from mindware.domains.search import run_domain_search


def _is_tune(task_id: str, split: dict) -> bool:
    return str(task_id).lower() in {x.lower() for x in split.get("tune") or []}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=["protein", "query", "dl"], required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--n-trials", type=int, default=16)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--optimizer", default="ptllm_cmaes")
    p.add_argument("--split", type=Path, default=None)
    p.add_argument("--allow-lock", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--dummy", action="store_true", default=True)
    args = p.parse_args(argv)

    split_path = args.split or (domain_history(args.domain) / "tune_lock_split.json")
    if split_path.is_file() and not args.allow_lock:
        split = json.loads(split_path.read_text(encoding="utf-8"))
        if not _is_tune(args.task, split):
            raise SystemExit(f"{args.task} is LOCK (or unknown); pass --allow-lock to override")

    adapter = get_adapter(args.domain, task={"task_id": args.task}, dummy=args.dummy)
    out = args.output_dir or (
        domain_bench(args.domain) / "rl_rollouts" / args.task / args.optimizer
    )
    result = run_domain_search(
        adapter,
        n_trials=args.n_trials,
        seed=args.seed,
        optimizer=args.optimizer,
        output_dir=out,
    )
    weights = RewardWeights()
    buffer_path = Path(out) / f"rollouts_{args.domain}_{args.task}.jsonl"
    prior = []
    with buffer_path.open("w", encoding="utf-8") as f:
        for row in result["traj"]:
            perf = -float(row["objective"])  # higher-is-better for reward
            eval_ok = bool(row.get("feasible", 1))
            reward = compute_trial_reward(
                perf=perf,
                prior_perfs=prior,
                eval_ok=eval_ok,
                parse_ok=True,
                duplicate=False,
                weights=weights,
            )
            prior.append(perf)
            msgs = build_domain_messages(adapter, result["traj"][: int(row["trial"]) - 1])
            rec = {
                "dataset": args.task,
                "domain": args.domain,
                "trial": int(row["trial"]),
                "messages": msgs,
                "completion": json.dumps({"source": row["source"], "objective": row["objective"]}),
                "reward": {"reward": float(reward.reward), "delta_best": float(reward.delta_best)},
                "group_id": f"{args.domain}:{args.task}:{row['trial']}",
            }
            f.write(json.dumps(rec) + "\n")
    print(json.dumps({"buffer": str(buffer_path), "n": len(result["traj"]), "best": result["best_objective"]}))
    print("Train with: python examples/benchmark/train_p5_reinforce.py --buffer-root", out)
    print("Use a domain-specific LoRA from examples/benchmark/domains/lora_adapters.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
