#!/usr/bin/env python3
"""
Interval-based improvement analysis for offline BO histories.

Unlike the algo40 eval trajectories (CASH-* / *_traj.csv), BO histories live at:
  history_data/history_data/{cls,rgs}/{metric}/{dataset}/{algorithm}.json

For each (test) task from --tasks-json, we compute:
  - % improvement over whole run (trial 0 -> trial N-1)
  - # new-best events (running incumbent increases)
  - same stats for 0-30, 30-60, 60-80 (by trial index within the first N trials)
  - optional per-event deltas (how much best-so-far jumped per improvement)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple


REPO = Path(__file__).resolve().parents[2]
HIST_ROOT_DEFAULT = REPO / "history_data" / "history_data"
ALGO40 = REPO / "benchmark_data" / "real_hpo_v2_mltest20_algo40"


@dataclass(frozen=True)
class IntervalSpec:
    name: str
    start_idx: int
    end_idx: int  # inclusive


def _compute_perf(obj_value: float) -> float:
    # OpenBox minimizes objectives; in our P5 history code we treat:
    #   perf = -objective (so higher perf is better).
    return -float(obj_value)


def compute_pct_improvement(start: float, end: float) -> float:
    if not (math.isfinite(start) and math.isfinite(end)):
        return float("nan")
    denom = abs(start)
    if denom <= 1e-12:
        # Fall back to percent change around 1.0 scale (avoid division by ~0).
        return (end - start) * 100.0
    return (end - start) / denom * 100.0


def load_bo_perf_series(history_path: Path, *, n_trials: int) -> List[float]:
    obj = json.loads(history_path.read_text(encoding="utf-8"))
    observations = obj.get("observations") or []
    # Keep only successful BO trials (trial_state==0).
    # For the history JSONs we use, trial order is preserved in the list.
    perfs: List[float] = []
    for o in observations:
        if int(o.get("trial_state", -1)) != 0:
            continue
        objs = o.get("objectives") or []
        if not objs:
            continue
        perfs.append(_compute_perf(float(objs[0])))
        if len(perfs) >= n_trials:
            break
    return perfs


def running_best(perfs: Sequence[float]) -> List[float]:
    out: List[float] = []
    best = float("-inf")
    for p in perfs:
        if math.isfinite(p) and p > best:
            best = p
        out.append(best)
    return out


def new_best_events(iters: Sequence[int], bests: Sequence[float], eps: float = 1e-12) -> List[Tuple[int, float, float, float]]:
    """
    events: (iter_idx, best_before, best_after, delta)
    where best_after is strictly increasing.
    """
    events: List[Tuple[int, float, float, float]] = []
    seen_any = False
    prev_best = float("-inf")
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
            prev_best = after
    return events


def best_at_index(best_series: Sequence[float], idx: int) -> Optional[float]:
    if not best_series:
        return None
    if idx < 0:
        idx = 0
    if idx >= len(best_series):
        # Allow specs like "60-80" when we only loaded first N trials and
        # indices are 0..N-1.
        idx = len(best_series) - 1
    b = best_series[idx]
    return b if math.isfinite(b) else None


def event_interval(intervals: Sequence[IntervalSpec], event_iter: int) -> Optional[str]:
    for it in intervals:
        if it.start_idx <= event_iter <= it.end_idx:
            return it.name
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tasks-json",
        type=Path,
        default=ALGO40 / "tasks_40_hard.json",
        help=(
            "Either a JSON list of task dicts (with keys task_type/dataset/algorithm), "
            "or a JSONL file like ml_test_20_fixed.jsonl where each line has record['metadata'] "
            "containing task_type/dataset/algorithm."
        ),
    )
    p.add_argument("--hist-root", type=Path, default=HIST_ROOT_DEFAULT)
    p.add_argument("--n-trials", type=int, default=80)
    p.add_argument("--out-dir", type=Path, default=ALGO40 / "analysis" / "bo_interval_improve")
    p.add_argument(
        "--intervals",
        type=str,
        default="0-30,30-60,60-80",
        help="Comma list of start-end inclusive intervals by trial index (within first n-trials).",
    )
    p.add_argument("--events-out", type=Path, default=None, help="If set, write per-new-best events CSV.")
    args = p.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks: List[Dict[str, str]] = []
    if str(args.tasks_json).endswith(".jsonl"):
        with args.tasks_json.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                md = rec.get("metadata") or {}
                tt = md.get("task_type")
                ds = md.get("dataset")
                algo = md.get("algorithm")
                if not tt or not ds or not algo:
                    continue
                tasks.append({"task_type": str(tt), "dataset": str(ds), "algorithm": str(algo)})
    else:
        tasks = json.loads(args.tasks_json.read_text(encoding="utf-8"))

    # Parse intervals (within the first n_trials).
    interval_specs: List[IntervalSpec] = []
    parsed = [s.strip() for s in args.intervals.split(",") if s.strip()]
    for seg in parsed:
        a, b = seg.split("-", 1)
        start_i = int(a)
        end_i = int(b)
        interval_specs.append(IntervalSpec(name=f"{start_i}-{end_i}", start_idx=start_i, end_idx=end_i))

    max_idx = args.n_trials - 1
    whole_spec = IntervalSpec(name="0-ALL", start_idx=0, end_idx=max_idx)

    summary_path = out_dir / "bo_interval_improve_summary.csv"
    summary_fields = [
        "task_type",
        "dataset",
        "algorithm",
        "task_key",
        "interval",
        "start_idx",
        "end_idx",
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

        for t in tasks:
            tt = t["task_type"]  # 'cls' or 'rgs'
            ds = t["dataset"]
            algo = t["algorithm"]

            metric = "bal_acc" if tt == "cls" else "mse"
            history_path = args.hist_root / tt / metric / ds / f"{algo}.json"
            if not history_path.is_file():
                continue

            perfs = load_bo_perf_series(history_path, n_trials=args.n_trials)
            if not perfs:
                continue

            bests = running_best(perfs)
            iters = list(range(len(bests)))
            events = new_best_events(iters, bests)

            # Precompute best at boundaries.
            all_intervals = [whole_spec] + interval_specs
            best_at: Dict[str, Optional[float]] = {}
            for seg in all_intervals:
                b0 = best_at_index(bests, seg.start_idx)
                b1 = best_at_index(bests, seg.end_idx)
                best_at[f"{seg.name}:start"] = b0
                best_at[f"{seg.name}:end"] = b1

            seg_events: Dict[str, List[Tuple[int, float, float, float]]] = {seg.name: [] for seg in all_intervals}
            for (e_iter, before, after, delta) in events:
                # Count each new-best event in every interval that contains its index.
                # This makes sub-interval stats (e.g. 0-30) consistent with best-so-far
                # changes computed from the same trial boundaries.
                for seg in all_intervals:
                    if seg.start_idx <= e_iter <= seg.end_idx:
                        seg_events[seg.name].append((e_iter, before, after, delta))

            for seg in all_intervals:
                start_best = best_at.get(f"{seg.name}:start")
                end_best = best_at.get(f"{seg.name}:end")
                if start_best is None or end_best is None:
                    continue

                pct = compute_pct_improvement(float(start_best), float(end_best))

                deltas = [float(e[3]) for e in seg_events.get(seg.name, [])]
                n_new = len(deltas)
                total_delta = float(sum(deltas)) if deltas else 0.0
                mean_delta = float(sum(deltas) / n_new) if n_new else float("nan")
                med_delta = float(median(deltas)) if n_new else float("nan")
                evs = seg_events.get(seg.name, [])
                first_iter = min((e[0] for e in evs), default=None)
                last_iter = max((e[0] for e in evs), default=None)

                w.writerow(
                    {
                        "task_type": tt,
                        "dataset": ds,
                        "algorithm": algo,
                        "task_key": f"{tt}/{ds}/{algo}",
                        "interval": seg.name,
                        "start_idx": seg.start_idx,
                        "end_idx": seg.end_idx,
                        "start_best": float(start_best),
                        "end_best": float(end_best),
                        "pct_improve": pct,
                        "n_new_bests": n_new,
                        "total_delta": total_delta,
                        "mean_delta_new_best": mean_delta,
                        "median_delta_new_best": med_delta,
                        "first_new_best_iter": first_iter if first_iter is not None else "",
                        "last_new_best_iter": last_iter if last_iter is not None else "",
                    }
                )

            if args.events_out is not None:
                for (e_iter, before, after, delta) in events:
                    # Same logic as above: write the event for every containing interval.
                    for seg in all_intervals:
                        if seg.start_idx <= e_iter <= seg.end_idx:
                            events_rows.append(
                                [
                                    tt,
                                    ds,
                                    algo,
                                    f"{tt}/{ds}/{algo}",
                                    seg.name,
                                    e_iter,
                                    before,
                                    after,
                                    delta,
                                ]
                            )

    if args.events_out is not None:
        events_out = args.events_out
        events_out.parent.mkdir(parents=True, exist_ok=True)
        with events_out.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "task_type",
                    "dataset",
                    "algorithm",
                    "task_key",
                    "interval",
                    "new_best_iter",
                    "best_before",
                    "best_after",
                    "delta",
                ]
            )
            for row in events_rows:
                w.writerow(row)
        print(f"Wrote events: {events_out} (rows={len(events_rows)})")

    print(f"Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

