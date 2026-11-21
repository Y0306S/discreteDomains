#!/usr/bin/env python3
"""Select 16 hardest tasks from tasks_40_hard for RL (8 cls + 8 rgs).

Cls ranked by bo_iqr (desc); rgs by bo_growth (desc). Skips slots with < min_bo_trials
successful BO observations (default 100, matching Setup-B prefix grid max prefix).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[2]
DEFAULT_META = (
    REPO
    / "benchmark_data/real_hpo_v2_mltest20_algo40/analysis/task_slice/tasks_40_hard_meta.json"
)
DEFAULT_OUT = REPO / "benchmark_data/rl_task_selection/hard_16.json"
DEFAULT_BO_ROOT = REPO / "history_data/history_data"


def bo_trial_count(
    task_type: str, dataset: str, algorithm: str, *, root: Path
) -> int:
    metric = "bal_acc" if task_type == "cls" else "mse"
    path = root / task_type / metric / dataset / f"{algorithm}.json"
    if not path.is_file():
        return -1
    obs = json.loads(path.read_text(encoding="utf-8")).get("observations") or []
    return sum(1 for o in obs if int(o.get("trial_state", -1)) == 0)


def pick_top8(
    tasks: List[Dict[str, Any]],
    *,
    score_key: str,
    bo_root: Path,
    min_bo_trials: int,
) -> List[Dict[str, Any]]:
    ranked = sorted(tasks, key=lambda t: -float(t[score_key]))
    out: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for t in ranked:
        n = bo_trial_count(t["task_type"], t["dataset"], t["algorithm"], root=bo_root)
        if n < min_bo_trials:
            skipped.append(f"{t['dataset']}/{t['algorithm']} (n={n})")
            continue
        row = {
            "task_type": t["task_type"],
            "dataset": t["dataset"],
            "algorithm": t["algorithm"],
            "metric": "bal_acc" if t["task_type"] == "cls" else "mse",
            "bo_iqr": t.get("bo_iqr"),
            "bo_growth": t.get("bo_growth"),
            "bo_trials": n,
            "hardness_score": float(t[score_key]),
            "hardness_metric": score_key,
        }
        out.append(row)
        if len(out) == 8:
            break
    if len(out) < 8:
        raise SystemExit(
            f"Only found {len(out)}/8 tasks with >={min_bo_trials} BO trials; "
            f"skipped={skipped}"
        )
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Select hard16 from tasks_40_hard meta.")
    p.add_argument("--meta", type=Path, default=DEFAULT_META)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--bo-root", type=Path, default=DEFAULT_BO_ROOT)
    p.add_argument("--min-bo-trials", type=int, default=100)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    tasks = meta["tasks"]
    cls = [t for t in tasks if t["task_type"] == "cls"]
    rgs = [t for t in tasks if t["task_type"] == "rgs"]

    cls8 = pick_top8(
        cls, score_key="bo_iqr", bo_root=args.bo_root, min_bo_trials=args.min_bo_trials
    )
    rgs8 = pick_top8(
        rgs, score_key="bo_growth", bo_root=args.bo_root, min_bo_trials=args.min_bo_trials
    )
    hard16 = cls8 + rgs8

    print(f"hard16: {len(hard16)} tasks (8 cls by bo_iqr, 8 rgs by bo_growth)")
    for t in hard16:
        print(
            f"  {t['task_type']:3s} {t['dataset'][:36]:36s} {t['algorithm']:18s} "
            f"score={t['hardness_score']:.4f} bo_n={t['bo_trials']}"
        )

    if args.dry_run:
        print("dry-run: not writing", args.out)
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(hard16, indent=2) + "\n", encoding="utf-8")
    sidecar = args.out.with_suffix(".meta.json")
    sidecar.write_text(
        json.dumps(
            {
                "source": str(args.meta),
                "selection": {
                    "cls": "top-8 by bo_iqr",
                    "rgs": "top-8 by bo_growth",
                    "min_bo_trials": args.min_bo_trials,
                },
                "tasks": hard16,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out} and {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
