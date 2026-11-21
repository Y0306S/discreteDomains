#!/usr/bin/env python3
"""Re-score TaskSet configs under the full paper protocol (10k / 200 / 5).

Why
---
Paper Appendix A.3 ("Probing short horizon") searches with a truncated inner loop
but **evaluates in the full training regime**: "we test this short horizon learning
by training hyperparameter lists only using some finite amount of training
iterations per task and testing in the full training regieme (10k steps)". It also
warns the truncation bias can be severe (Wu et al. 2016; Metz et al. 2019a).

So a short-horizon BO result is not a result until its winner is re-scored at
10k/200/5. This script does that, and reports the short-horizon bias explicitly
instead of assuming it away.

Two modes
---------
  --reference   Score the paper's own grid-search winner per task
                (best_optimizer_per_task.json). Run this BEFORE the BO sweep: it
                proves the full-protocol path works, measures its cost, and gives
                the reference point the BO winner is compared against.

  --from-run    Score the best config each BO run selected, reading history.json
                under a run directory produced by run_taskset_bo.py.

Examples
--------
  # before the sweep: cost + reference on one task
  rescore_taskset_full_protocol.py --reference --task-ids mlp_family_seed12

  # after the sweep: re-score every BO winner and report the bias
  rescore_taskset_full_protocol.py --from-run /root/data/logs/taskset_bo_train_1000steps
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains.dlhpo import taskset_oracle
from mindware.domains.paths import domain_bench

BEST_PER_TASK = Path("/root/data/cys/taskset_runs/best_optimizer_per_task.json")


def paper_protocol() -> Dict[str, int]:
    """10k steps / eval every 200 / 5 replicas, read from the shared constants."""
    return taskset_oracle.paper_protocol()


def reference_configs(task_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """The paper's grid-search winner per task, as {task_id: {family, hyperparameters}}."""
    if not BEST_PER_TASK.is_file():
        raise SystemExit(f"missing {BEST_PER_TASK}")
    data = json.loads(BEST_PER_TASK.read_text())
    out: Dict[str, Dict[str, Any]] = {}
    for tid in task_ids:
        entry = data.get(tid)
        if not entry:
            print(f"  ! {tid}: absent from best_optimizer_per_task.json", file=sys.stderr)
            continue
        # "nadamw_grid_1k" -> "nadamw_grid"
        family = re.sub(r"_1k$", "", str(entry.get("optimizer_family_npz") or ""))
        hps = entry.get("optimizer_hyperparameters") or {}
        if not family or not hps:
            print(f"  ! {tid}: no family/hyperparameters in reference entry", file=sys.stderr)
            continue
        out[tid] = {
            "optimizer_family": family,
            "hyperparameters": hps,
            "source": "paper_grid_best",
            "reference_valid_inner_loss": entry.get("valid_inner_loss"),
        }
    return out


def bo_winners(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Best config per task from BO history.json files under `run_dir`.

    Layout written by run_taskset_bo.py: <run_dir>/<task_id>/<optimizer>/seed<N>/history.json
    """
    winners: Dict[str, Dict[str, Any]] = {}
    for hist in sorted(run_dir.glob("*/*/*/history.json")):
        task_id = hist.parents[2].name
        try:
            obs = json.loads(hist.read_text()).get("observations") or []
        except (OSError, json.JSONDecodeError) as e:
            print(f"  ! {hist}: {e}", file=sys.stderr)
            continue
        best = None
        for o in obs:
            objs = o.get("objectives") or []
            if not objs:
                continue
            val = objs[0]
            if val is None or not isinstance(val, (int, float)):
                continue
            # objective is minimized; 1e10 is the oracle's infeasible sentinel
            if val >= 1e9:
                continue
            if best is None or val < best[0]:
                best = (val, o)
        if best is None:
            print(f"  ! {task_id}: no feasible observation", file=sys.stderr)
            continue
        val, o = best
        extra = o.get("extra_info") or {}
        art = extra.get("artifact") or {}
        info = extra.get("info") or {}
        if not art.get("optimizer_family"):
            print(f"  ! {task_id}: winning observation has no artifact", file=sys.stderr)
            continue
        winners[task_id] = {
            "optimizer_family": art["optimizer_family"],
            "hyperparameters": art.get("hyperparameters") or {},
            "source": "bo_winner",
            "short_horizon_objective": val,
            "short_horizon_metric": info.get("objective_metric"),
            "short_horizon_protocol": info.get("protocol"),
            "n_observations": len(obs),
        }
    return winners


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--reference", action="store_true", help="score the paper's grid-search winner per task")
    src.add_argument("--from-run", type=Path, default=None, help="score BO winners under this run directory")
    p.add_argument("--task-ids", default=None, help="comma-separated subset (default: all found)")
    p.add_argument("--max-tasks", type=int, default=None, help="cap the number of tasks (use for a timed probe)")
    p.add_argument("--training-steps", type=int, default=None, help="override (default: paper 10000)")
    p.add_argument("--eval-every-n", type=int, default=None, help="override (default: paper 200)")
    p.add_argument("--num-replicas", type=int, default=None, help="override (default: paper 5)")
    p.add_argument(
        "--objective-metric",
        default=taskset_oracle.DEFAULT_OBJECTIVE_METRIC,
        choices=list(taskset_oracle.OBJECTIVE_METRICS),
        help="full-protocol scoring uses the paper §3.3 mean by default",
    )
    p.add_argument("--replica-timeout", type=int, default=7200, help="10k steps needs more than the 2400s default")
    p.add_argument("--warm-timeout", type=int, default=7200)
    p.add_argument("-o", "--output", type=Path, default=None)
    args = p.parse_args(argv)

    proto = paper_protocol()
    steps = args.training_steps if args.training_steps is not None else proto["training_steps"]
    every = args.eval_every_n if args.eval_every_n is not None else proto["eval_every_n"]
    nrep = args.num_replicas if args.num_replicas is not None else proto["num_replicas"]

    if args.reference:
        wanted = [t.strip() for t in (args.task_ids or "").split(",") if t.strip()]
        if not wanted:
            raise SystemExit("--reference needs --task-ids (scoring all 1157 at full protocol is not a default)")
        configs = reference_configs(wanted)
    else:
        run_dir = args.from_run
        if not run_dir.is_dir():
            raise SystemExit(f"not a directory: {run_dir}")
        configs = bo_winners(run_dir)
        if args.task_ids:
            keep = {t.strip() for t in args.task_ids.split(",") if t.strip()}
            configs = {k: v for k, v in configs.items() if k in keep}

    task_ids = sorted(configs)
    if args.max_tasks:
        task_ids = task_ids[: args.max_tasks]

    print(f"=== full-protocol re-scoring: {steps} steps / eval every {every} / {nrep} replicas ===")
    print(f"  source   : {'paper grid best' if args.reference else args.from_run}")
    print(f"  tasks    : {len(task_ids)}")
    print(f"  metric   : {args.objective_metric}")
    print()

    results = []
    t_all = time.time()
    for i, tid in enumerate(task_ids, 1):
        cfg = configs[tid]
        t0 = time.time()
        res = taskset_oracle.evaluate_taskset_config(
            task_id=tid,
            optimizer_family=cfg["optimizer_family"],
            hyperparameters=cfg["hyperparameters"],
            work_dir=domain_bench("taskset") / tid / "_rescore",
            training_steps=steps,
            eval_every_n=every,
            num_replicas=nrep,
            objective_metric=args.objective_metric,
            warm_timeout=args.warm_timeout,
            replica_timeout=args.replica_timeout,
        )
        dt = time.time() - t0
        info = res.info or {}
        row = {
            "task_id": tid,
            "source": cfg["source"],
            "optimizer_family": cfg["optimizer_family"],
            "full_objective": res.objective if res.feasible else None,
            "feasible": bool(res.feasible),
            "full_mean_normalized_valid_outer": info.get("mean_normalized_valid_outer"),
            "full_min_normalized_valid_outer": info.get("min_normalized_valid_outer"),
            "full_final_test_loss": info.get("mean_final_test_loss"),
            "replicas_ok": info.get("replicas_ok"),
            "replicas_total": info.get("replicas_total"),
            "seconds": round(dt, 1),
            "error": info.get("error"),
        }
        if cfg["source"] == "bo_winner":
            row["short_horizon_objective"] = cfg["short_horizon_objective"]
            row["short_horizon_metric"] = cfg["short_horizon_metric"]
            row["short_horizon_protocol"] = cfg["short_horizon_protocol"]
            # Positive bias => the short-horizon score was optimistic relative to full.
            if row["full_min_normalized_valid_outer"] is not None and cfg["short_horizon_metric"] == "min_normalized_valid_outer":
                row["short_horizon_bias"] = round(
                    row["full_min_normalized_valid_outer"] - cfg["short_horizon_objective"], 6
                )
        results.append(row)
        print(f"  [{i}/{len(task_ids)}] {tid}: full={row['full_objective']} "
              f"test={row['full_final_test_loss']} reps={row['replicas_ok']}/{row['replicas_total']} "
              f"({row['seconds']}s)" + (f" ERROR={row['error']}" if row["error"] else ""), flush=True)

    ok = [r for r in results if r["feasible"]]
    payload = {
        "protocol": {"training_steps": steps, "eval_every_n": every, "num_replicas": nrep},
        "objective_metric": args.objective_metric,
        "mode": "reference" if args.reference else "from_run",
        "run_dir": str(args.from_run) if args.from_run else None,
        "n_tasks": len(results),
        "n_feasible": len(ok),
        "wall_seconds": round(time.time() - t_all, 1),
        "seconds_per_task": round((time.time() - t_all) / max(1, len(results)), 1),
        "results": results,
    }
    biases = [r["short_horizon_bias"] for r in results if r.get("short_horizon_bias") is not None]
    if biases:
        import statistics as st
        payload["short_horizon_bias_summary"] = {
            "n": len(biases), "mean": round(st.mean(biases), 6),
            "median": round(st.median(biases), 6),
            "min": round(min(biases), 6), "max": round(max(biases), 6),
        }

    out = args.output or Path(f"/root/data/logs/rescore_full_{'reference' if args.reference else 'bo'}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  feasible {len(ok)}/{len(results)}   wall {payload['wall_seconds']}s "
          f"({payload['seconds_per_task']}s/task)")
    if biases:
        print(f"  short-horizon bias: {payload['short_horizon_bias_summary']}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
