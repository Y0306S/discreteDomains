#!/usr/bin/env python3
"""Head-to-head X50 vs X50R on unique (task, dataset, algorithm) slots.

find_traj filters by algorithm so two jobs that share a dataset are not
collapsed onto one traj. Decision rule is pre-written in
x50r_vs_x50_eval_spec.json and is not changed after launch.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from plot_mltest20_algo40_avg_rank import find_traj, final_best, load_tasks, task_key, task_ranks

SPEC_PATH = Path(__file__).resolve().parent / "x50r_vs_x50_eval_spec.json"
ALGO40 = ROOT / "benchmark_data" / "real_hpo_v2_mltest20_algo40"
X50_DIR = "ptllm_tpe__P5plusA5T50X50__H2"
X50R_DIR = "ptllm_tpe__P5plusA5T50X50R__H2"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, default=SPEC_PATH)
    p.add_argument("--root", type=Path, default=ALGO40)
    p.add_argument("--n-trials", type=int, default=80)
    p.add_argument(
        "--out",
        type=Path,
        default=ALGO40 / "analysis" / "lodo_similarity" / "x50r_vs_x50_unique_slots.csv",
    )
    args = p.parse_args(argv)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    tasks = load_tasks(ROOT / spec["tasks_json"] if not Path(spec["tasks_json"]).is_absolute() else Path(spec["tasks_json"]))

    rows = []
    x50_scores: dict[str, float] = {}
    x50r_scores: dict[str, float] = {}
    for t in tasks:
        key = task_key(t)
        a = find_traj(args.root, t["task_type"], t["dataset"], X50_DIR, args.n_trials, algorithm=t.get("algorithm"))
        b = find_traj(args.root, t["task_type"], t["dataset"], X50R_DIR, args.n_trials, algorithm=t.get("algorithm"))
        sa = final_best(a) if a else None
        sb = final_best(b) if b else None
        rows.append(
            {
                "slot": key,
                "task_type": t["task_type"],
                "dataset": t["dataset"],
                "algorithm": t["algorithm"],
                "x50_traj": str(a) if a else "",
                "x50r_traj": str(b) if b else "",
                "x50_best": sa,
                "x50r_best": sb,
                "both_complete": int(sa is not None and sb is not None),
                "x50r_wins": int(sa is not None and sb is not None and sb > sa),
                "x50_wins": int(sa is not None and sb is not None and sa > sb),
                "tie": int(sa is not None and sb is not None and abs(sa - sb) < 1e-12),
            }
        )
        if sa is not None and sb is not None:
            x50_scores[key] = sa
            x50r_scores[key] = sb

    unique_declared = len({task_key(t) for t in tasks})
    both = [r for r in rows if r["both_complete"]]
    h2h_r = sum(r["x50r_wins"] for r in both)
    h2h_x = sum(r["x50_wins"] for r in both)
    ties = sum(r["tie"] for r in both)

    mean_rank_x50 = mean_rank_x50r = float("nan")
    if both:
        ranks_x, ranks_r = [], []
        for r in both:
            scores = {"X50": r["x50_best"], "X50R": r["x50r_best"]}
            rk = task_ranks(scores)
            ranks_x.append(rk["X50"])
            ranks_r.append(rk["X50R"])
        mean_rank_x50 = sum(ranks_x) / len(ranks_x)
        mean_rank_x50r = sum(ranks_r) / len(ranks_r)

    adopt = (
        len(both) > 0
        and mean_rank_x50r < mean_rank_x50
        and h2h_r > h2h_x
    )
    decision = "adopt_X50R" if adopt else "keep_X50"
    if not both:
        decision = "no_complete_paired_slots"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["slot"])
        w.writeheader()
        w.writerows(rows)

    summary = {
        "n_declared_slots": unique_declared,
        "n_rows": len(rows),
        "n_both_complete": len(both),
        "mean_rank_x50": mean_rank_x50,
        "mean_rank_x50r": mean_rank_x50r,
        "h2h_x50r_wins": h2h_r,
        "h2h_x50_wins": h2h_x,
        "h2h_ties": ties,
        "decision": decision,
        "decision_rule": spec["decision_rule"],
        "status": spec.get("status"),
        "note": "Live HPO eval was not launched; this reporter is ready for go-ahead results.",
    }
    spath = args.out.with_suffix(".summary.json")
    spath.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {spath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
