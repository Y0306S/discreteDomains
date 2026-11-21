#!/usr/bin/env python3
"""Build v12+gap ablation buffers: v12 (12 jobs) + v13−v12 gap jobs only (9 jobs).

Gap records are RS-version agnostic (normal / explore / exploit) and are taken
from the v3 partial v13 collect.  v12 rs_warm rows stay version-specific.

Usage:
  python examples/benchmark/merge_rsmix_v12gap_buffer.py \\
    --v12-from benchmark_data/rl_rollouts/B_rsmix_v4_n4to10_T1p0_pareto32_v12 \\
    --gap-from benchmark_data/rl_rollouts/B_rsmix_v3_n4to10_T1p0_pareto32_v13 \\
    --out benchmark_data/rl_rollouts/B_rsmix_v4_n4to10_T1p0_pareto32_v12gap
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "examples" / "benchmark"))
from p5plus_rl_collect_schedule import V13_GAP_JOBS  # noqa: E402

GAP_JOBS: Set[Tuple[int, str]] = {(int(k), str(sc)) for k, sc in V13_GAP_JOBS}
JOBS_PER_TASK = 21


def _prefix_k(rec: dict) -> int:
    for key in ("prefix_k", "bo_prefix", "k"):
        if key in rec and rec[key] is not None:
            return int(rec[key])
    raise KeyError(f"record missing prefix key: {list(rec.keys())}")


def _scenario(rec: dict) -> str:
    return str(rec.get("scenario") or "")


def _job_key(rec: dict) -> Tuple[int, str]:
    return (_prefix_k(rec), _scenario(rec))


def _read_lines(path: Path, keep: Set[Tuple[int, str]]) -> List[str]:
    out: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if _job_key(rec) in keep:
            out.append(line.rstrip("\n"))
    return out


def _read_v12_lines(path: Path) -> List[str]:
    out: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if _job_key(rec) not in GAP_JOBS:
            out.append(line.rstrip("\n"))
    return out


def _task_dirs(root: Path) -> Dict[str, Path]:
    if not root.is_dir():
        return {}
    return {
        p.name: p
        for p in sorted(root.iterdir())
        if p.is_dir() and any(p.rglob("rollouts_*.jsonl"))
    }


def _rollout_files(task_dir: Path) -> List[Path]:
    return sorted(task_dir.rglob("rollouts_*.jsonl"))


def merge_buffers(v12_from: Path, gap_from: Path, out: Path, *, dry_run: bool = False) -> dict:
    v12_tasks = _task_dirs(v12_from)
    gap_tasks = _task_dirs(gap_from)
    if not v12_tasks:
        raise SystemExit(f"no task dirs under v12-from: {v12_from}")
    if not gap_tasks:
        raise SystemExit(f"no task dirs under gap-from: {gap_from}")

    if out.exists() and not dry_run:
        shutil.rmtree(out)
    if not dry_run:
        out.mkdir(parents=True, exist_ok=True)

    n_v12 = n_gap = n_out = 0
    per_job: Dict[Tuple[int, str], int] = {}

    for task, v12_dir in sorted(v12_tasks.items()):
        gap_dir = gap_tasks.get(task)
        lines: List[str] = []
        for src in _rollout_files(v12_dir):
            chunk = _read_v12_lines(src)
            n_v12 += len(chunk)
            lines.extend(chunk)
        if gap_dir is not None:
            for src in _rollout_files(gap_dir):
                chunk = _read_lines(src, GAP_JOBS)
                n_gap += len(chunk)
                lines.extend(chunk)
        if not lines:
            raise SystemExit(f"no records merged for task {task}")

        if not dry_run:
            dest_dir = out / task
            dest_dir.mkdir(parents=True, exist_ok=True)
            (dest_dir / "rollouts_merged.jsonl").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
        n_out += len(lines)
        for ln in lines:
            per_job[_job_key(json.loads(ln))] = per_job.get(_job_key(json.loads(ln)), 0) + 1

    stats = {
        "tasks": len(v12_tasks),
        "records": n_out,
        "v12_records": n_v12,
        "gap_records": n_gap,
        "unique_jobs": len(per_job),
        "per_job": {f"k{k}_{sc}": per_job[(k, sc)] for (k, sc) in sorted(per_job)},
        "out": str(out),
    }
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v12-from", type=Path, required=True)
    p.add_argument("--gap-from", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    stats = merge_buffers(args.v12_from, args.gap_from, args.out, dry_run=args.dry_run)
    print(json.dumps(stats, indent=2))
    need = stats["tasks"] * JOBS_PER_TASK * 4
    if stats["records"] < need:
        print(f"WARNING records={stats['records']} < floor {need}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
