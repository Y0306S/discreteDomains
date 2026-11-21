#!/usr/bin/env python3
"""Compare P16/H2 original (real_hpo_v2) vs replication run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from experiment_utils import EVAL8_DATASETS, jobs_from_dataset_names, score_higher_is_better
from plot_pilot_sweep import collect_sweep

VARIANT = "ptllm_tpe__P16__H2"
REPO = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Compare P16/H2 original vs replication.")
    parser.add_argument("--base-dir", default=str(REPO / "benchmark_data" / "real_hpo_v2"))
    parser.add_argument(
        "--rep-dir",
        default="",
        help="Replication base dir (default: benchmark_data/real_hpo_v2_p16_<rep-tag>).",
    )
    parser.add_argument("--rep-tag", default="rep2")
    parser.add_argument("--out-csv", default="")
    args = parser.parse_args()

    base = Path(args.base_dir)
    rep = Path(args.rep_dir) if args.rep_dir else REPO / "benchmark_data" / f"real_hpo_v2_p16_{args.rep_tag}"
    jobs = jobs_from_dataset_names(EVAL8_DATASETS)
    dataset_keys = [f"{t}/{d}" for t, d in jobs]

    _, _, _, summary_orig = collect_sweep(base, jobs, lambda name: name == VARIANT)
    _, _, _, summary_rep = collect_sweep(rep, jobs, lambda name: name == VARIANT)

    rows = []
    for task, ds in jobs:
        key = f"{task}/{ds}"
        higher = score_higher_is_better(task)
        s_orig = summary_orig.get((key, VARIANT))
        s_rep = summary_rep.get((key, VARIANT))
        v_orig = s_orig["best"] if s_orig else float("nan")
        v_rep = s_rep["best"] if s_rep else float("nan")
        if s_orig and s_rep and v_orig == v_orig and v_rep == v_rep:
            delta = v_rep - v_orig
            winner = "rep" if (delta > 0 if higher else delta < 0) else ("orig" if delta != 0 else "tie")
        else:
            delta = float("nan")
            winner = ""
        rows.append(
            {
                "dataset": ds,
                "task": task,
                "orig_best": v_orig,
                "rep_best": v_rep,
                "delta_rep_minus_orig": delta,
                "winner": winner,
            }
        )

    out = Path(args.out_csv) if args.out_csv else rep.parent / "analysis" / "eval8" / f"p16_h2_{args.rep_tag}_vs_orig.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "task", "orig_best", "rep_best", "delta_rep_minus_orig", "winner"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            for k in ("orig_best", "rep_best", "delta_rep_minus_orig"):
                v = row[k]
                row[k] = f"{v:.6g}" if v == v else ""
            w.writerow(row)

    n_comp = sum(1 for r in rows if r["winner"])
    n_rep = sum(1 for r in rows if r["winner"] == "rep")
    n_orig = sum(1 for r in rows if r["winner"] == "orig")
    print(f"Wrote {out}")
    print(f"Per-dataset: rep wins {n_rep}, orig wins {n_orig}, compared {n_comp}/{len(rows)}")
    print(f"Original: {base}")
    print(f"Replication: {rep}")


if __name__ == "__main__":
    main()
