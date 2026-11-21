#!/usr/bin/env python3
"""BOLT native LOLBO (GP + TuRBO + BoTorch) on CEB tasks with execution oracle."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains.queryopt.bolt_lolbo_runner import AUTHOR_CKPT, QUERY_CONSISTENT10, run_tasks

OUT_ROOT = ROOT / "benchmark_data" / "domain_query" / "arm_c_bo_v2" / "bolt_execute"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-trials", type=int, default=80)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--execution-timeout", type=float, default=100.0)
    p.add_argument("--n-tasks", type=int, default=3033)
    p.add_argument("--task-offset", type=int, default=0)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument(
        "--dsn",
        default=os.environ.get(
            "QUERYOPT_PG_DSN",
            "host=127.0.0.1 port=5433 dbname=imdb user=imdb password=imdb",
        ),
    )
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    p.add_argument("--ckpt", type=Path, default=AUTHOR_CKPT)
    p.add_argument("--task-ids", default="")
    p.add_argument("--skip-done", action="store_true", default=True)
    p.add_argument("--no-skip-done", action="store_false", dest="skip_done")
    args = p.parse_args()
    os.environ["QUERYOPT_PG_DSN"] = args.dsn
    tasks = [t.strip() for t in args.task_ids.split(",") if t.strip()] or None
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = run_tasks(
        n_trials=args.n_trials,
        seed=args.seed,
        device=args.device,
        dsn=args.dsn,
        skip_done=args.skip_done,
        task_ids=tasks,
        out_root=args.out_root,
        ckpt=args.ckpt,
        execution_timeout=args.execution_timeout,
        n_tasks=None if tasks else args.n_tasks,
        task_offset=args.task_offset,
        shard_index=args.shard_index,
        n_shards=args.n_shards,
    )
    print(json.dumps({"finished": len(rows), "shard_index": args.shard_index}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
