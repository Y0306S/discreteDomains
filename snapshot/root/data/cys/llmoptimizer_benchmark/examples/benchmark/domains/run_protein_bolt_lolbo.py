#!/usr/bin/env python3
"""BOLT native LOLBO on benchmark peptides (GP + TuRBO TR + E2E VAE)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains.protein.bolt_seed0_init import BOLT_FULL_ORACLE_BUDGET, CHECKPOINT_ORACLE_CALLS
from mindware.domains.protein.bolt_lolbo_runner import CONSISTENT10, run_consistent10, run_tasks


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n-trials",
        type=int,
        default=500,
        help="Post-init oracle budget per task (after seed_0 pool; BOLT paper uses 200k).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed (BOLT seed_0 omits this; default None).",
    )
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--oracle-mode", default="bolt_native", choices=["bolt_native"])
    p.add_argument("--min-similarity", type=float, default=0.75)
    p.add_argument("--constraint-index", type=int, default=0)
    p.add_argument("--bolt-init-points", type=int, default=1000)
    p.add_argument("--bsz", type=int, default=50)
    p.add_argument(
        "--e2e-freq",
        type=int,
        default=None,
        help="BO rounds without improvement before E2E VAE finetune (default: BOLT seed0=10).",
    )
    p.add_argument("--max-string-length", type=int, default=30)
    p.add_argument("--n-tasks", type=int, default=100)
    p.add_argument("--task-offset", type=int, default=0)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--task-ids", default="", help="Comma-separated pep IDs")
    p.add_argument("--consistent10", action="store_true", help="Run the fixed 10-task subset")
    p.add_argument("--skip-done", action="store_true", default=True)
    p.add_argument("--no-skip-done", action="store_false", dest="skip_done")
    p.add_argument(
        "--run-mode",
        choices=["checkpoint", "continue_bolt"],
        default="checkpoint",
        help="checkpoint: strict 500-call phase; continue_bolt: resume to BOLT 200k with E2E.",
    )
    p.add_argument(
        "--max-oracle-budget",
        type=int,
        default=BOLT_FULL_ORACLE_BUDGET,
        help="Total post-init oracle budget for continue_bolt (BOLT seed_0 uses 200000).",
    )
    p.add_argument(
        "--checkpoint-calls",
        type=int,
        default=CHECKPOINT_ORACLE_CALLS,
        help="Checkpoint size for phase-1 runs.",
    )
    p.add_argument(
        "--init-data-path",
        default=None,
        help="Init sequence file (blocks of --bolt-init-points rows, one per reference). "
        "Defaults to BOLT's single-block seed_0_init.txt.",
    )
    p.add_argument(
        "--init-scores-path",
        default=None,
        help="Init scores file, aligned 1:1 with --init-data-path.",
    )
    p.add_argument(
        "--init-offset-helper",
        type=int,
        default=0,
        help="Index of the FIRST reference present in the init file. BOLT selects the "
        "block as constraint_types[0] - init_offset_helper, so this is a property of the "
        "file, not the task. Use the value in the pool manifest.",
    )
    p.add_argument(
        "--allow-infeasible-init",
        action="store_true",
        help="Do not abort when zero init points satisfy the similarity constraint "
        "(default is to fail loudly; only set this if you know why).",
    )
    p.add_argument(
        "--budget-mode",
        choices=["acquisition", "total"],
        default="acquisition",
        help="acquisition: budget counts only acquisition candidates, so every task gets "
        "exactly max_oracle_budget/bsz rounds (E2E re-scoring reported separately). "
        "total: budget counts every oracle call, which lets one E2E round overshoot.",
    )
    args = p.parse_args()
    tasks = [t.strip() for t in args.task_ids.split(",") if t.strip()] or None
    if args.consistent10:
        rows = run_consistent10(
            n_trials=args.n_trials,
            seed=args.seed,
            device_id=args.device_id,
            task_ids=tasks,
            skip_done=args.skip_done,
            oracle_mode=args.oracle_mode,
            n_init_points=args.bolt_init_points,
            bsz=args.bsz,
            e2e_freq=args.e2e_freq,
            run_mode=args.run_mode,
            max_oracle_budget=args.max_oracle_budget,
            checkpoint_calls=args.checkpoint_calls,
        )
    else:
        rows = run_tasks(
            n_trials=args.n_trials,
            seed=args.seed,
            device_id=args.device_id,
            task_ids=tasks,
            skip_done=args.skip_done,
            oracle_mode=args.oracle_mode,
            min_similarity=args.min_similarity,
            constraint_index=args.constraint_index,
            n_init_points=args.bolt_init_points,
            bsz=args.bsz,
            e2e_freq=args.e2e_freq,
            n_tasks=args.n_tasks,
            task_offset=args.task_offset,
            shard_index=args.shard_index,
            n_shards=args.n_shards,
            run_mode=args.run_mode,
            max_oracle_budget=args.max_oracle_budget,
            checkpoint_calls=args.checkpoint_calls,
            init_data_path=args.init_data_path,
            init_scores_path=args.init_scores_path,
            init_offset_helper=args.init_offset_helper,
            require_feasible_init=not args.allow_infeasible_init,
            budget_mode=args.budget_mode,
        )
    print(json.dumps({"finished": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
