#!/usr/bin/env python3
"""
Interval-based improvement analysis for algo40 trajectories.

Computes, per task and per method (variant):
  - % improvement over whole run
  - # new-best events over whole run
  - the same stats for intervals:
      0–30, 30–60, 60–80
  - for each new-best event: how much did the best score jump (delta)

Inputs come from the same *_traj.csv learning-curve files that
`plot_mltest20_algo40_avg_rank.py` uses.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO = Path(__file__).resolve().parents[2]
ALGO40 = REPO / "benchmark_data" / "real_hpo_v2_mltest20_algo40"


@dataclass(frozen=True)
class IntervalSpec:
    name: str
    start_iter: int
    end_iter: int  # inclusive


def _fnum(v: object, default: float = float("nan")) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except Exception:
        return default


def compute_pct_improvement(start: float, end: float) -> float:
    # Scores are in "higher is better" space for both cls and rgs in traj CSVs.
    if not math.isfinite(start) or not math.isfinite(end):
        return float("nan")
    denom = abs(start)
    if denom <= 1e-12:
        # Avoid division by ~0: report as relative change to 1.0 scale.
        return (end - start) * 100.0
    return (end - start) / denom * 100.0


def load_traj(traj_path: Path, n_trials: int) -> Tuple[List[int], List[float], List[float]]:
    """
    Returns:
      iter_idx[], best_so_far[], perf[] (perf not strictly needed for this analysis)
    """
    iters: List[int] = []
    bests: List[float] = []
    perfs: List[float] = []
    with traj_path.open("r", encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f)
        for row_i, r in enumerate(rows):
            if row_i >= n_trials:
                break
            ii = r.get("iter_idx")
            iters.append(int(ii) if ii is not None and str(ii).strip() != "" else row_i)
            bests.append(_fnum(r.get("best_so_far")))
            perfs.append(_fnum(r.get("perf")))
    # Keep monotonic safety: sort by iter_idx in case of weird ordering.
    zipped = sorted(zip(iters, bests, perfs), key=lambda t: t[0])
    iters, bests, perfs = [list(x) for x in zip(*zipped)] if zipped else ([], [], [])
    return iters, bests, perfs


def best_at(iters: Sequence[int], bests: Sequence[float], target_iter: int) -> Optional[float]:
    best_val: Optional[float] = None
    for ii, b in zip(iters, bests):
        if ii <= target_iter and math.isfinite(b):
            best_val = b
        if ii > target_iter:
            break
    return best_val


def new_best_events(
    iters: Sequence[int],
    bests: Sequence[float],
    eps: float = 1e-12,
) -> List[Tuple[int, float, float, float]]:
    """
    Returns list of events:
      (iter_idx, best_before, best_after, delta)
    where delta > eps
    """
    events: List[Tuple[int, float, float, float]] = []
    prev_best = float("-inf")
    seen_any = False
    for ii, b in zip(iters, bests):
        if not math.isfinite(b):
            continue
        if not seen_any:
            prev_best = b
            seen_any = True
            continue
        if b > prev_best + eps:
            before = prev_best
            after = b
            events.append((int(ii), before, after, after - before))
            prev_best = b
    return events


def event_interval(intervals: Sequence[IntervalSpec], event_iter: int) -> Optional[str]:
    for it in intervals:
        if it.start_iter <= event_iter <= it.end_iter:
            return it.name
    return None


def main() -> int:
    # Import plot module only for locating trajs on disk.
    sys.path.insert(0, str(REPO))
    from examples.benchmark.plot_mltest20_algo40_avg_rank import (
        build_sources,
        find_traj,
        task_key,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks-json",
        type=Path,
        default=ALGO40 / "tasks_40_hard.json",
        help="List of eval tasks to analyze (dataset×algorithm slots).",
    )
    parser.add_argument("--out-dir", type=Path, default=ALGO40 / "analysis" / "interval_improve")
    parser.add_argument("--n-trials", type=int, default=80)
    parser.add_argument("--seed1-only", action="store_true", help="Exclude seed replications.")
    parser.add_argument(
        "--intervals",
        type=str,
        default="0-30,30-60,60-80",
        help="Comma list of start-end inclusive iter ranges.",
    )
    parser.add_argument(
        "--label-contains",
        type=str,
        default="",
        help="Optional substring filter applied to the leaderboard label (variant key). Empty = all.",
    )
    parser.add_argument(
        "--disk-contains",
        type=str,
        default="",
        help="Optional substring filter applied to the on-disk variant folder name.",
    )
    parser.add_argument(
        "--events-out",
        type=Path,
        default=None,
        help="If set, write per-new-best events CSV (can be large).",
    )

    args = parser.parse_args()

    tasks = __import__("json").loads(args.tasks_json.read_text(encoding="utf-8"))
    task_keys = [task_key(t) for t in tasks]
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    interval_specs: List[IntervalSpec] = []
    parsed = [s.strip() for s in args.intervals.split(",") if s.strip()]
    for seg in parsed:
        if "-" not in seg:
            raise ValueError(f"bad interval segment {seg!r} (expected start-end)")
        a, b = seg.split("-", 1)
        start_i = int(a)
        end_i = int(b)
        name = f"{start_i}-{end_i}"
        interval_specs.append(IntervalSpec(name=name, start_iter=start_i, end_iter=end_i))

    # Whole-run interval is implicit.
    max_iter = args.n_trials - 1
    whole = IntervalSpec(name="0-ALL", start_iter=0, end_iter=max_iter)

    sources = build_sources(include_replications=not args.seed1_only)

    label_contains = args.label_contains.strip()
    disk_contains = args.disk_contains.strip()

    # Header for summary CSV
    summary_path = out_dir / "interval_improve_summary.csv"
    summary_fields = [
        "task_type",
        "dataset",
        "algorithm",
        "task_key",
        "variant_label",
        "variant_disk",
        "interval",
        "start_iter",
        "end_iter",
        "start_best",
        "end_best",
        "pct_improve",
        "n_new_bests",
        "total_delta",
        "mean_delta_new_best",
        "median_delta_new_best",
        "first_new_best_iter",
        "last_new_best_iter",
    ]

    events_rows: List[List[object]] = []

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()

        # Iterate in deterministic order.
        for t in tasks:
            tt, ds, algo = t["task_type"], t["dataset"], t["algorithm"]
            key = task_key(t)
            for variant_label, (root, disk_var) in sorted(sources.items()):
                if label_contains and label_contains not in variant_label:
                    continue
                if disk_contains and disk_contains not in str(disk_var):
                    continue

                traj_path = find_traj(root, tt, ds, disk_var, args.n_trials, algorithm=algo)
                if traj_path is None:
                    continue

                iters, bests, _perfs = load_traj(traj_path, args.n_trials)
                if not iters:
                    continue

                # Compute bests at segment boundaries.
                intervals_all = [whole] + interval_specs
                seg2best: Dict[str, Optional[float]] = {}
                for seg in intervals_all:
                    b0 = best_at(iters, bests, seg.start_iter)
                    b1 = best_at(iters, bests, seg.end_iter)
                    # We'll store start/end separately for event assignment.
                    seg2best[f"{seg.name}:start"] = b0
                    seg2best[f"{seg.name}:end"] = b1

                # New-best events.
                events = new_best_events(iters, bests)
                if events:
                    first_iter = min(e[0] for e in events)
                    last_iter = max(e[0] for e in events)
                else:
                    first_iter = ""
                    last_iter = ""

                # Assign events to intervals.
                seg_events: Dict[str, List[Tuple[int, float, float, float]]] = {seg.name: [] for seg in intervals_all}
                for (e_iter, before, after, delta) in events:
                    segname = event_interval(intervals_all, e_iter)
                    if segname is None:
                        continue
                    seg_events[segname].append((e_iter, before, after, delta))

                # Write one row per segment.
                for seg in intervals_all:
                    start_best = seg2best.get(f"{seg.name}:start")
                    end_best = seg2best.get(f"{seg.name}:end")
                    if start_best is None or end_best is None or (not math.isfinite(start_best) or not math.isfinite(end_best)):
                        continue
                    pct = compute_pct_improvement(float(start_best), float(end_best))

                    deltas = [float(d) for (_it, _b0, _b1, d) in seg_events.get(seg.name, [])]
                    n_new = len(deltas)
                    total_delta = float(sum(deltas)) if deltas else 0.0
                    mean_delta = float(sum(deltas) / n_new) if n_new else float("nan")
                    med_delta = float(median(deltas)) if n_new else float("nan")
                    evs = seg_events.get(seg.name, [])
                    first_ev_iter = min((e[0] for e in evs), default=None)
                    last_ev_iter = max((e[0] for e in evs), default=None)

                    w.writerow(
                        {
                            "task_type": tt,
                            "dataset": ds,
                            "algorithm": algo,
                            "task_key": key,
                            "variant_label": variant_label,
                            "variant_disk": str(disk_var),
                            "interval": seg.name,
                            "start_iter": seg.start_iter,
                            "end_iter": seg.end_iter,
                            "start_best": float(start_best),
                            "end_best": float(end_best),
                            "pct_improve": pct,
                            "n_new_bests": n_new,
                            "total_delta": total_delta,
                            "mean_delta_new_best": mean_delta,
                            "median_delta_new_best": med_delta,
                            "first_new_best_iter": first_ev_iter if first_ev_iter is not None else "",
                            "last_new_best_iter": last_ev_iter if last_ev_iter is not None else "",
                        }
                    )

                # Optional events dump.
                if args.events_out is not None and seg_events:
                    for (e_iter, before, after, delta) in events:
                        segname = event_interval(intervals_all, e_iter)
                        if segname is None:
                            continue
                        events_rows.append(
                            [
                                tt,
                                ds,
                                algo,
                                key,
                                variant_label,
                                str(disk_var),
                                segname,
                                e_iter,
                                before,
                                after,
                                delta,
                            ]
                        )

    if args.events_out is not None:
        fields = [
            "task_type",
            "dataset",
            "algorithm",
            "task_key",
            "variant_label",
            "variant_disk",
            "interval",
            "new_best_iter",
            "best_before",
            "best_after",
            "delta",
        ]
        with args.events_out.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(fields)
            for row in events_rows:
                w.writerow(row)

    print(f"Wrote summary: {summary_path}")
    if args.events_out is not None:
        print(f"Wrote events: {args.events_out} (rows={len(events_rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

