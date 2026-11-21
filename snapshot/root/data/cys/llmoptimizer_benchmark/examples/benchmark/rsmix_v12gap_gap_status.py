#!/usr/bin/env python3
"""Gap-job status for v12 → v12gap (21-point) top-3 offline pipeline.

Gap jobs are the 9 v13−v12 prefix/scenario pairs (shared across RS versions).
Top-3 training uses v3 + v4 buffers only; gap rollouts are version-agnostic.

Usage:
  python examples/benchmark/rsmix_v12gap_gap_status.py
  python examples/benchmark/rsmix_v12gap_gap_status.py --tasks-needing-strict4-collect
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "examples" / "benchmark"))
from p5plus_rl_collect_schedule import V13_GAP_JOBS  # noqa: E402

GAP_JOBS: Set[Tuple[int, str]] = {(int(k), str(sc)) for k, sc in V13_GAP_JOBS}
N_GAP_JOBS = len(GAP_JOBS)


def _task_name(task: dict) -> str:
    return f"{task['task_type']}_{task['dataset']}_{task['algorithm']}"


def load_train_order(path: Path) -> List[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def gap_jobs_in_task_dir(task_dir: Path) -> Set[Tuple[int, str]]:
    have: Set[Tuple[int, str]] = set()
    if not task_dir.is_dir():
        return have
    for rollout in task_dir.rglob("rollouts_*.jsonl"):
        for line in rollout.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            key = (int(rec["prefix_k"]), str(rec.get("scenario") or ""))
            if key in GAP_JOBS:
                have.add(key)
    return have


def gap_jobs_in_merged(task_dir: Path) -> Set[Tuple[int, str]]:
    merged = task_dir / "rollouts_merged.jsonl"
    if not merged.is_file():
        return gap_jobs_in_task_dir(task_dir)
    have: Set[Tuple[int, str]] = set()
    for line in merged.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        key = (int(rec["prefix_k"]), str(rec.get("scenario") or ""))
        if key in GAP_JOBS:
            have.add(key)
    return have


def per_source_gap_counts(task: str, sources: Dict[str, Path]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for name, root in sources.items():
        if not root.is_dir():
            out[name] = 0
            continue
        if name.endswith("_v12gap"):
            out[name] = len(gap_jobs_in_merged(root / task))
        else:
            out[name] = len(gap_jobs_in_task_dir(root / task))
    return out


def gap_union_for_task(task: str, source_order: Iterable[str], sources: Dict[str, Path]) -> Set[Tuple[int, str]]:
    union: Set[Tuple[int, str]] = set()
    for name in source_order:
        root = sources.get(name)
        if root is None or not root.is_dir():
            continue
        if name.endswith("_v12gap"):
            union |= gap_jobs_in_merged(root / task)
        else:
            union |= gap_jobs_in_task_dir(root / task)
    return union


def tasks_needing_strict4_collect(
    order: List[dict],
    strict4_patch: Path,
) -> List[Tuple[dict, int]]:
    missing: List[Tuple[dict, int]] = []
    for task in order:
        name = _task_name(task)
        have = len(gap_jobs_in_task_dir(strict4_patch / name))
        if have < N_GAP_JOBS:
            missing.append((task, have))
    return missing


def tasks_needing_any_gap_collect(
    order: List[dict],
    strict4_patch: Path,
    fallback_order: List[str],
    sources: Dict[str, Path],
) -> List[Tuple[dict, int, int]]:
    """Return (task_row, have_strict4, have_union) for tasks still missing gap after fallbacks."""
    out: List[Tuple[dict, int, int]] = []
    for task in order:
        name = _task_name(task)
        strict4_n = len(gap_jobs_in_task_dir(strict4_patch / name))
        union = gap_union_for_task(name, fallback_order, sources)
        if len(union) < N_GAP_JOBS:
            out.append((task, strict4_n, len(union)))
    return out


def _format_collect_line(task: dict, have: int, *, union: int | None = None) -> str:
    tt, ds, algo = task["task_type"], task["dataset"], task["algorithm"]
    if union is None:
        return f"{tt}|{ds}|{algo}|{have}|{N_GAP_JOBS}"
    return f"{tt}|{ds}|{algo}|{have}|{union}|{N_GAP_JOBS}"


def default_sources(repo: Path) -> Dict[str, Path]:
    roll = repo / "benchmark_data" / "rl_rollouts"
    return {
        "strict4_patch": roll / "B_rsmix_v3_gap_patch_T1p0_pareto32_strict4",
        "v13_partial": roll / "B_rsmix_v3_n4to10_T1p0_pareto32_v13",
        "v3_v12gap": roll / "B_rsmix_v3_n4to10_T1p0_pareto32_v12gap",
        "v4_v12gap": roll / "B_rsmix_v4_n4to10_T1p0_pareto32_v12gap",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--order",
        type=Path,
        default=REPO_ROOT / "benchmark_data/rl_task_selection/picked_32_pareto_train_order.json",
    )
    p.add_argument(
        "--strict4-patch",
        type=Path,
        default=REPO_ROOT / "benchmark_data/rl_rollouts/B_rsmix_v3_gap_patch_T1p0_pareto32_strict4",
    )
    p.add_argument("--tasks-needing-strict4-collect", action="store_true")
    p.add_argument("--tasks-needing-any-gap-collect", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    order = load_train_order(args.order)
    sources = default_sources(REPO_ROOT)
    sources["strict4_patch"] = args.strict4_patch
    fallback_order = ["strict4_patch", "v13_partial", "v3_v12gap"]

    if args.tasks_needing_strict4_collect:
        rows = tasks_needing_strict4_collect(order, args.strict4_patch)
        if args.json:
            print(
                json.dumps(
                    [
                        {
                            "task_type": t["task_type"],
                            "dataset": t["dataset"],
                            "algorithm": t["algorithm"],
                            "have": n,
                            "need": N_GAP_JOBS,
                        }
                        for t, n in rows
                    ],
                    indent=2,
                )
            )
        else:
            for task, have in rows:
                print(_format_collect_line(task, have))
        return 0

    if args.tasks_needing_any_gap_collect:
        rows = tasks_needing_any_gap_collect(order, args.strict4_patch, fallback_order, sources)
        if args.json:
            print(
                json.dumps(
                    [
                        {
                            "task_type": t["task_type"],
                            "dataset": t["dataset"],
                            "algorithm": t["algorithm"],
                            "strict4": s4,
                            "union": u,
                            "need": N_GAP_JOBS,
                        }
                        for t, s4, u in rows
                    ],
                    indent=2,
                )
            )
        else:
            for task, s4, union in rows:
                print(_format_collect_line(task, s4, union=union))
        return 0

    strict4_missing = tasks_needing_strict4_collect(order, args.strict4_patch)
    any_missing = tasks_needing_any_gap_collect(order, args.strict4_patch, fallback_order, sources)
    summary = {
        "n_train_tasks": len(order),
        "n_gap_jobs": N_GAP_JOBS,
        "strict4_patch": str(args.strict4_patch),
        "strict4_collect_needed": len(strict4_missing),
        "any_gap_collect_needed": len(any_missing),
        "strict4_missing_sample": strict4_missing[:5],
        "any_missing_sample": any_missing[:5],
        "fallback_order": fallback_order,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
