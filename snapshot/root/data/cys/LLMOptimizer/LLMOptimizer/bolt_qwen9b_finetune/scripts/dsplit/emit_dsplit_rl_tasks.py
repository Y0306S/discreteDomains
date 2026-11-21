#!/usr/bin/env python3
"""Emit the RL-32 tasks + train-order files for the dataset-level split.

Consumes picked_32_pareto.json from select_train32_pareto_bo_intervals.py and
reproduces the exact two artifacts the GRPO pipeline reads:

  <out>_tasks.json        16 cls (pareto rank order) then 16 rgs, 6 fields each
  <out>_train_order.json  strict cls/rgs interleave of the above

Also writes a verification file asserting no picked dataset is in the test split
-- the property the original run violated for 24/32 of its picks.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

TASK_FIELDS = ["task_type", "dataset", "algorithm", "metric", "late_n_new", "late_pct_sum"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--picked", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True)
    args = ap.parse_args()

    picked = json.loads(args.picked.read_text())["tasks"]
    man = json.loads(args.split_manifest.read_text())
    test_ds = set(man["test_datasets"])
    train_ds = set(man["train_datasets"])

    tasks = [{k: t[k] for k in TASK_FIELDS} for t in picked]

    cls = [t for t in tasks if t["task_type"] == "cls"]
    rgs = [t for t in tasks if t["task_type"] == "rgs"]
    ordered = [t for pair in zip(cls, rgs) for t in pair]
    if len(ordered) != len(tasks):
        raise SystemExit(f"cls/rgs imbalance: cls={len(cls)} rgs={len(rgs)}")

    # ---- invariants ----
    leaked = sorted({t["dataset"] for t in tasks} & test_ds)
    if leaked:
        raise SystemExit(f"RL picks leak into test split: {leaked}")
    unknown = sorted({t["dataset"] for t in tasks} - train_ds)
    if unknown:
        raise SystemExit(f"RL picks not in train split: {unknown}")
    dups = {d: n for d, n in collections.Counter(t["dataset"] for t in tasks).items() if n > 1}

    tasks_path = args.out_prefix.with_name(args.out_prefix.name + "_tasks.json")
    order_path = args.out_prefix.with_name(args.out_prefix.name + "_train_order.json")
    ver_path = args.out_prefix.with_name(args.out_prefix.name + "_split_verification.json")

    tasks_path.write_text(json.dumps(tasks, indent=2) + "\n")
    order_path.write_text(json.dumps(ordered, indent=2) + "\n")
    ver_path.write_text(json.dumps({
        "split": "dataset_level_80_20",
        "split_seed": man["seed"],
        "n_tasks": len(tasks),
        "n_cls": len(cls),
        "n_rgs": len(rgs),
        "all_datasets_in_train_split": True,
        "any_dataset_in_test_split": False,
        "n_datasets_also_appearing_in_test_with_other_algos": 0,
        "distinct_datasets": len({t["dataset"] for t in tasks}),
        "datasets_used_more_than_once": dups,
        "tasks": [
            {"pick_id": i + 1, **{k: t[k] for k in ("dataset", "algorithm", "task_type")},
             "in_train_split": True, "in_test_split": False}
            for i, t in enumerate(tasks)
        ],
    }, indent=2) + "\n")

    print(f"tasks       -> {tasks_path}  ({len(cls)} cls + {len(rgs)} rgs)")
    print(f"train_order -> {order_path}  (strict interleave)")
    print(f"verification-> {ver_path}")
    print(f"distinct datasets: {len({t['dataset'] for t in tasks})}/{len(tasks)}"
          + (f"  duplicates={dups}" if dups else "  (all unique)"))
    print("leakage into test split: NONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
