#!/usr/bin/env python3
"""Run TaskSet CASH BO over many tasks (resume with --skip-done).

There are 1157 TaskSet tasks. Default here is a small slice; paper-scale inner
evals (train_inner, 10k steps x 5 replicas) should use --no-dummy --max-tasks
explicitly.

  python examples/benchmark/domains/run_taskset_bo_suite.py --optimizer tpe --n-trials 20 --max-tasks 8 --dummy
  python examples/benchmark/domains/run_taskset_bo_suite.py --optimizer smac --n-trials 20 --max-tasks 8 --dummy
  python examples/benchmark/domains/run_taskset_bo_suite.py --optimizer tpe --n-trials 8 --max-tasks 1 --no-dummy --task-ids mlp_family_seed12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.benchmark.domains.run_taskset_bo import main as run_one_batch


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--optimizer", default="tpe", help="tpe|smac|random")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dummy", action="store_true", default=True)
    p.add_argument("--no-dummy", action="store_false", dest="dummy")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--training-steps", type=int, default=None)
    p.add_argument("--eval-every-n", type=int, default=None)
    p.add_argument("--num-replicas", type=int, default=None)
    p.add_argument(
        "--objective-metric",
        default=None,
        choices=["mean_normalized_valid_outer", "min_normalized_valid_outer"],
        help="See run_taskset_bo.py. Prefer min_normalized_valid_outer for short inner runs.",
    )
    p.add_argument("--task-ids", default=None)
    p.add_argument("--max-tasks", type=int, default=8)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--skip-done", action="store_true", default=True)
    p.add_argument("--no-skip-done", action="store_false", dest="skip_done")
    args = p.parse_args(argv)
    argv2 = [
        "--optimizer",
        args.optimizer,
        "--n-trials",
        str(args.n_trials),
        "--seed",
        str(args.seed),
        "--max-tasks",
        str(args.max_tasks),
        "--offset",
        str(args.offset),
    ]
    if args.task_ids:
        argv2.extend(["--task-ids", args.task_ids])
    if args.dummy:
        argv2.append("--dummy")
    else:
        argv2.append("--no-dummy")
    if args.smoke:
        argv2.append("--smoke")
    if args.training_steps is not None:
        argv2.extend(["--training-steps", str(args.training_steps)])
    if args.eval_every_n is not None:
        argv2.extend(["--eval-every-n", str(args.eval_every_n)])
    if args.num_replicas is not None:
        argv2.extend(["--num-replicas", str(args.num_replicas)])
    if args.objective_metric is not None:
        argv2.extend(["--objective-metric", args.objective_metric])
    if args.output_dir is not None:
        argv2.extend(["--output-dir", str(args.output_dir)])
    if args.skip_done:
        argv2.append("--skip-done")
    else:
        argv2.append("--no-skip-done")
    return run_one_batch(argv2)


if __name__ == "__main__":
    raise SystemExit(main())
