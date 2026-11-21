#!/usr/bin/env python3
"""TaskSet CASH BO: random, TPE, no-VAE LOLBO, or SMAC over optimizer families.

Dummy oracle is a fake loss (fast). Real scoring runs task_set.train_inner in the
taskset_eval env using the BO corpus protocol: 1000 steps x 1 replica,
validation every 1000 steps, minimum normalized validation loss.

  # Fake landscape (CPU, seconds)
  python examples/benchmark/domains/run_taskset_bo.py --optimizer tpe --n-trials 12 --dummy

  # BO corpus evaluation protocol (CPU TF1, 100 configurations)
  python examples/benchmark/domains/run_taskset_bo.py --optimizer tpe --no-dummy --task-ids mlp_family_seed12

  # Real wrapper, short inner loop (wiring / debug)
  python examples/benchmark/domains/run_taskset_bo.py --optimizer tpe --n-trials 1 --smoke --task-ids mlp_family_seed12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains import get_adapter
from mindware.domains.dlhpo.taskset_oracle import DEFAULT_OBJECTIVE_METRIC, bo_protocol, smoke_protocol
from mindware.domains.dlhpo.taskset_space import list_taskset_ids
from mindware.domains.history import history_exists, load_openbox_history
from mindware.domains.taskset_archive import archive_path
from mindware.domains.paths import domain_bench
from mindware.domains.search import run_domain_search


def _load_observations(hist: Path) -> List[dict]:
    if not history_exists(hist):
        return []
    payload = load_openbox_history(hist)
    return list(payload.get("observations") or [])


def _select_tasks(args: argparse.Namespace) -> List[str]:
    if args.task_ids:
        return [t.strip() for t in args.task_ids.split(",") if t.strip()]
    ids = list_taskset_ids()
    start = int(args.offset)
    return ids[start : start + int(args.max_tasks)]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--optimizer", default="tpe", choices=["tpe", "smac", "random", "lolbo"])
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dummy", action="store_true", default=True)
    p.add_argument("--no-dummy", action="store_false", dest="dummy")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Real train_inner with 400 steps x 1 replica (not paper protocol).",
    )
    p.add_argument("--training-steps", type=int, default=None, help="Default: 1000 (400 with --smoke).")
    p.add_argument("--eval-every-n", type=int, default=None, help="Default: 1000 (200 with --smoke).")
    p.add_argument("--num-replicas", type=int, default=None, help="Default: 1.")
    p.add_argument(
        "--objective-metric",
        default=DEFAULT_OBJECTIVE_METRIC,
        choices=["mean_normalized_valid_outer", "min_normalized_valid_outer"],
        help="Which normalized VALID_OUTER aggregation is the BO objective. Default is "
        "min_normalized_valid_outer, matching the BO corpus. The mean "
        "includes the step-0 point, which normalizes to a constant 1.0, so on a 2-point curve "
        "the objective cannot drop below ~0.5. Appendix A.4 validates min as equivalent "
        "(Pearson rho = 0.911).",
    )
    p.add_argument("--warm-timeout", type=int, default=3600)
    p.add_argument("--replica-timeout", type=int, default=2400)
    p.add_argument("--baseline-cache", type=Path, default=None)
    p.add_argument("--lolbo-batch-size", type=int, default=10,
                   help="LOLBO proposal batch size; the total n-trials budget is still enforced.")
    p.add_argument("--lolbo-n-candidates", type=int, default=None,
                   help="LOLBO Thompson candidate pool (default: ML backend's 2000-5000).")
    p.add_argument(
        "--n-startup-trials",
        type=int,
        default=None,
        help="Random trials before model-based search. LOLBO defaults to 10 (minimum 2). "
        "For TPE, run_domain_search otherwise derives it as "
        "max(1, min(8, n_trials//10)) -- i.e. 2 for n_trials=20, NOT the TPESampler default of 10. "
        "The floor is 1 (run_domain_search clamps with max(1, ...)), so a pure-model-based run with "
        "zero random trials is not reachable.",
    )
    p.add_argument("--task-ids", default=None, help="Comma-separated TaskSet task names.")
    p.add_argument("--max-tasks", type=int, default=1)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--skip-done", action="store_true", default=True)
    p.add_argument("--no-skip-done", action="store_false", dest="skip_done")
    args = p.parse_args(argv)
    protocol = smoke_protocol() if args.smoke else bo_protocol()
    for key, default in protocol.items():
        if getattr(args, key) is None:
            setattr(args, key, default)
    protocol = {key: getattr(args, key) for key in protocol}

    tasks = _select_tasks(args)
    if not tasks:
        raise SystemExit("No TaskSet task ids found. Check /root/data/cys/taskset_runs/best_optimizer_per_task.json")

    summaries = []
    for tid in tasks:
        out = args.output_dir
        if out is None:
            out = domain_bench("taskset") / tid / args.optimizer / f"seed{args.seed}"
        else:
            out = Path(out) / tid / args.optimizer / f"seed{args.seed}"
        adapter = get_adapter(
            "taskset",
            task={"task_id": tid},
            dummy=args.dummy,
            smoke=args.smoke,
            training_steps=args.training_steps,
            eval_every_n=args.eval_every_n,
            num_replicas=args.num_replicas,
            objective_metric=args.objective_metric,
            baseline_cache=args.baseline_cache,
            warm_timeout=args.warm_timeout,
            replica_timeout=args.replica_timeout,
            oracle_work_dir=out / "_oracle",
        )
        resume_obs = _load_observations(out / "history.json")
        if args.optimizer == "lolbo" and any(
            (obs.get("extra_info", {}).get("info", {}).get("optimizer_backend")
             != "lolbo_no_vae_cash") for obs in resume_obs
        ):
            raise ValueError(
                "This directory contains an old or unverified LOLBO run. "
                "Old TaskSet lolbo runs used CMA-ES; use a new output directory."
            )
        for obs in resume_obs:
            info = (obs.get("extra_info") or {}).get("info") or {}
            expected_oracle = "dummy_taskset" if adapter.dummy else "train_inner"
            if info.get("oracle") != expected_oracle or (
                not adapter.dummy and (
                    info.get("protocol") != protocol
                    or info.get("objective_metric") != args.objective_metric
                )
            ):
                raise ValueError(
                    "Saved history has a different or missing evaluation protocol. "
                    "Use a fresh output directory for the requested budget and metric."
                )
        if args.skip_done and len(resume_obs) >= int(args.n_trials):
            row = {
                "task_id": tid,
                "domain": "taskset",
                "optimizer": args.optimizer,
                "n_trials": args.n_trials,
                "best_objective": None,
                "output_dir": str(out),
                "skipped": True,
                "n_obs": len(resume_obs),
            }
            summaries.append(row)
            print(json.dumps(row, indent=2), flush=True)
            continue
        if archive_path(out / "history.json").exists():
            raise ValueError("Completed archives are immutable; restore to a separate directory before extending a run.")
        result = run_domain_search(
            adapter,
            n_trials=args.n_trials,
            seed=args.seed,
            optimizer=args.optimizer,
            output_dir=out,
            resume_observations=resume_obs or None,
            n_startup_trials=args.n_startup_trials,
            lolbo_batch_size=args.lolbo_batch_size,
            lolbo_n_candidates=args.lolbo_n_candidates,
        )
        row = {k: result[k] for k in ("task_id", "domain", "optimizer", "n_trials", "best_objective") if k in result}
        row["output_dir"] = str(out)
        row["resumed_from"] = result.get("resumed_from")
        row["dummy"] = bool(adapter.dummy)
        row["smoke"] = bool(args.smoke)
        # Record the inner budget and scoring choice: a "best_objective" is only
        # comparable across runs that used the same protocol and aggregation.
        row["protocol"] = {
            "training_steps": args.training_steps,
            "eval_every_n": args.eval_every_n,
            "num_replicas": args.num_replicas,
        }
        row["objective_metric"] = args.objective_metric
        row["n_startup_trials"] = args.n_startup_trials
        if args.optimizer == "lolbo":
            row["optimizer_backend"] = "lolbo_no_vae_cash"
            row["lolbo_batch_size"] = args.lolbo_batch_size
            row["lolbo_n_candidates"] = args.lolbo_n_candidates
        summaries.append(row)
        print(json.dumps(row, indent=2), flush=True)
    if len(summaries) > 1:
        print(json.dumps({"n_tasks": len(summaries), "runs": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
