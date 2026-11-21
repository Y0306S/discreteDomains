#!/usr/bin/env python3
"""Progress + health monitor for the corrected STBO peptide runs.

Checks the invariants that the old sweeps silently violated:
  - every task has 1000 feasible init points
  - every reported best satisfies the >=0.75 similarity constraint
  - the acquisition budget is respected exactly
  - no task was truncated
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path

DEFAULT_ROOT = Path("/root/data/cys/blockdata_archive/domain_protein/stbo_bsz10_r100_fixed")
TRAIN = range(0, 900)
TEST = range(900, 1000)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--n-tasks", type=int, default=900)
    a = ap.parse_args()

    idx = a.root / "index.jsonl"
    rows = []
    if idx.is_file():
        for line in idx.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    done = [r for r in rows if r.get("best_feasible_objective") is not None]
    by_task = {r["task_id"]: r for r in done}

    print(f"root      : {a.root}")
    print(f"completed : {len(by_task)} / {a.n_tasks}")

    try:
        alive = subprocess.run(
            ["pgrep", "-fc", "run_protein_bolt_lolbo.py"],
            capture_output=True, text=True,
        ).stdout.strip()
    except OSError:
        alive = "?"
    print(f"workers   : {alive} alive")

    if not by_task:
        print("\nno completed rows yet")
        return 0

    # ---- health invariants ----
    bad_init = [t for t, r in by_task.items() if r.get("n_init_feasible") != 1000]
    bad_sim = [
        t for t, r in by_task.items()
        if (r.get("best_feasible_similarity") or 0) < r.get("min_similarity", 0.75)
    ]
    over = [t for t, r in by_task.items() if not r.get("budget_respected")]
    trunc = [t for t, r in by_task.items() if r.get("truncated")]
    no_feas = [t for t, r in by_task.items() if r.get("best_feasible_objective") is None]

    print("\n--- health ---")
    for label, bad in (
        ("init pool not 1000-feasible", bad_init),
        ("best violates similarity", bad_sim),
        ("budget exceeded", over),
        ("truncated", trunc),
        ("no feasible best", no_feas),
    ):
        flag = "OK " if not bad else "FAIL"
        extra = "" if not bad else f"  e.g. {sorted(bad)[:5]}"
        print(f"  [{flag}] {label:30s} {len(bad)}{extra}")

    # ---- metric ----
    mics = {t: -r["best_feasible_objective"] for t, r in by_task.items()}
    vals = sorted(mics.values())
    print("\n--- best feasible predicted MIC (lower better) ---")
    print(
        f"  n={len(vals)}  min={vals[0]:.2f}  p50={statistics.median(vals):.2f}  "
        f"mean={statistics.mean(vals):.2f}  p90={vals[int(0.9 * len(vals)) - 1]:.2f}  max={vals[-1]:.2f}"
    )

    for name, rng in (("train 000-899", TRAIN), ("test  900-999", TEST)):
        sub = [mics[f"pep_{i:03d}"] for i in rng if f"pep_{i:03d}" in mics]
        if sub:
            print(f"  {name}: n={len(sub)} mean={statistics.mean(sub):.2f} sum={sum(sub):.1f}")

    sub20 = [mics[f"pep_{i:03d}"] for i in range(900, 920) if f"pep_{i:03d}" in mics]
    if len(sub20) == 20:
        print(
            f"\n  paper Table 11 gate (refs 900-919, 20 tasks): sum={sum(sub20):.1f}"
            f"   [published STBO @k=1000 = 625.95; our init pools = 759.9 +/- 67.3]"
        )

    # ---- cost ----
    secs = [r["seconds"] for r in done if r.get("seconds")]
    acq = [r.get("n_acquisition_calls") for r in done if r.get("n_acquisition_calls")]
    e2e = [r.get("n_e2e_calls") for r in done if r.get("n_e2e_calls") is not None]
    if secs:
        print("\n--- cost ---")
        print(f"  seconds/task: mean={statistics.mean(secs):.1f} p50={statistics.median(secs):.1f} max={max(secs):.1f}")
        remaining = a.n_tasks - len(by_task)
        n_alive = int(alive) if str(alive).isdigit() and int(alive) > 0 else 1
        eta_min = remaining * statistics.mean(secs) / max(n_alive, 1) / 60
        print(f"  remaining={remaining}  ETA ~{eta_min:.0f} min at {n_alive} workers")
    if acq:
        print(f"  acquisition calls/task: {statistics.mean(acq):.0f} (target 1000)")
    if e2e:
        print(f"  e2e re-scoring calls/task: mean={statistics.mean(e2e):.0f}  (hidden cost, now counted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
