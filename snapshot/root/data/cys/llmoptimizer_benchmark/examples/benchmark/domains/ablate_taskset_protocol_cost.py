#!/usr/bin/env python3
"""How deep (training steps) and how wide (eval points) can the inner loop go
before wall-clock actually starts to hurt?

Holds the CONFIG fixed (the paper's grid-search winner for each task, from
best_optimizer_per_task.json) and varies only the protocol, so the timing
difference is attributable to the protocol alone and not to which
hyperparameters happened to be sampled.

Two 1-D sweeps rather than a full grid:
  --sweep depth   vary training_steps at a fixed eval density
  --sweep width   vary eval_every_n at fixed training_steps

Note train_inner evaluates when `step % eval_every_n == 0 or step ==
training_steps`, starting at step 0, so eval points = floor(steps/every) + 1.

Usage:
  ablate_taskset_protocol_cost.py --sweep depth --task-ids a,b,c
  ablate_taskset_protocol_cost.py --sweep width --task-ids a,b,c --steps 1000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains.dlhpo import taskset_oracle
from mindware.domains.paths import domain_bench

BEST_PER_TASK = Path("/root/data/cys/taskset_runs/best_optimizer_per_task.json")

DEPTHS = [200, 500, 1000, 2000, 5000, 10000]
WIDTHS = [1000, 500, 200, 100, 50, 20]      # eval_every_n at steps=1000 -> 2,3,6,11,21,51 points


def reference_config(task_id: str) -> Dict[str, Any]:
    data = json.loads(BEST_PER_TASK.read_text())
    entry = data.get(task_id)
    if not entry:
        raise SystemExit(f"{task_id} absent from {BEST_PER_TASK}")
    return {
        "optimizer_family": re.sub(r"_1k$", "", str(entry["optimizer_family_npz"])),
        "hyperparameters": entry["optimizer_hyperparameters"],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", required=True, choices=["depth", "width"])
    p.add_argument("--task-ids", required=True)
    p.add_argument("--steps", type=int, default=1000, help="fixed training_steps for the width sweep")
    p.add_argument("--eval-frac", type=int, default=5,
                   help="depth sweep: eval_every_n = steps//eval-frac, so eval-point count is held constant")
    p.add_argument("--num-replicas", type=int, default=1)
    p.add_argument("--replica-timeout", type=int, default=7200)
    p.add_argument("-o", "--output", type=Path, required=True)
    args = p.parse_args(argv)

    tasks = [t.strip() for t in args.task_ids.split(",") if t.strip()]
    combos: List[Dict[str, int]] = []
    if args.sweep == "depth":
        for s in DEPTHS:
            combos.append({"training_steps": s, "eval_every_n": max(1, s // args.eval_frac)})
    else:
        for e in WIDTHS:
            combos.append({"training_steps": args.steps, "eval_every_n": e})

    print(f"=== {args.sweep} sweep: {len(tasks)} task(s) x {len(combos)} protocol(s), "
          f"{args.num_replicas} replica ===", flush=True)

    rows = []
    for tid in tasks:
        cfg = reference_config(tid)
        for c in combos:
            steps, every = c["training_steps"], c["eval_every_n"]
            n_pts = steps // every + 1
            t0 = time.time()
            try:
                res = taskset_oracle.evaluate_taskset_config(
                    task_id=tid,
                    optimizer_family=cfg["optimizer_family"],
                    hyperparameters=cfg["hyperparameters"],
                    work_dir=domain_bench("taskset") / tid / "_ablate" / f"{steps}_{every}",
                    training_steps=steps,
                    eval_every_n=every,
                    num_replicas=args.num_replicas,
                    objective_metric="mean_normalized_valid_outer",
                    replica_timeout=args.replica_timeout,
                    warm_timeout=args.replica_timeout,
                )
                dt = time.time() - t0
                info = res.info or {}
                row = {
                    "task_id": tid, "training_steps": steps, "eval_every_n": every,
                    "eval_points": n_pts, "seconds": round(dt, 1),
                    "sec_per_1k_steps": round(dt / (steps / 1000.0), 1),
                    "feasible": bool(res.feasible),
                    "mean_normalized_valid_outer": info.get("mean_normalized_valid_outer"),
                    "min_normalized_valid_outer": info.get("min_normalized_valid_outer"),
                    "final_test_loss": info.get("mean_final_test_loss"),
                    "error": info.get("error"),
                }
            except Exception as e:
                dt = time.time() - t0
                row = {"task_id": tid, "training_steps": steps, "eval_every_n": every,
                       "eval_points": n_pts, "seconds": round(dt, 1), "feasible": False,
                       "error": f"{type(e).__name__}: {e}"}
            rows.append(row)
            print(f"  {tid[:38]:<38} steps={steps:<6} every={every:<5} pts={n_pts:<3} "
                  f"{row['seconds']:>7.1f}s  obj={row.get('mean_normalized_valid_outer')}"
                  + (f"  ERR={row['error']}" if row.get("error") else ""), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"sweep": args.sweep, "rows": rows}, indent=2))

    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
