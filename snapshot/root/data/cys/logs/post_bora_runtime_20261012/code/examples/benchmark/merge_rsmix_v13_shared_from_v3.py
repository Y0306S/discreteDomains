#!/usr/bin/env python3
"""Build v4/v5 rs_mix v13 buffers by reusing v3 shared prefixes.

Only k in {0,5,10} (rs_warm) are version-specific.  All other prefix/scenario
jobs are identical across RS versions and are copied from the completed v3 v13 buffer.

Usage:
  python examples/benchmark/merge_rsmix_v13_shared_from_v3.py \\
    --shared-from benchmark_data/rl_rollouts/B_rsmix_v3_n4to10_T1p0_pareto32_v13 \\
    --rs-from benchmark_data/rl_rollouts/B_rsmix_v4_rswarm_T1p0_pareto32_v13 \\
    --out benchmark_data/rl_rollouts/B_rsmix_v4_n4to10_T1p0_pareto32_v13 \\
    --rs-version v4
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set

RS_PREFIXES: Set[int] = {0, 5, 10}
V13_JOBS_PER_TASK = 21


def _prefix_k(rec: dict) -> int:
    for key in ("prefix_k", "bo_prefix", "k"):
        if key in rec and rec[key] is not None:
            return int(rec[key])
    raise KeyError(f"record missing prefix key: {rec.keys()}")


def _read_filtered(path: Path, keep: Set[int]) -> List[str]:
    out: List[str] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if _prefix_k(rec) in keep:
            out.append(line.rstrip("\n"))
    return out


def _read_filtered_not(path: Path, drop: Set[int]) -> List[str]:
    out: List[str] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if _prefix_k(rec) not in drop:
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


def merge_buffers(
    shared_from: Path,
    rs_from: Path,
    out: Path,
    *,
    rs_version: str,
    dry_run: bool = False,
) -> dict:
    shared_tasks = _task_dirs(shared_from)
    rs_tasks = _task_dirs(rs_from)
    if not shared_tasks:
        raise SystemExit(f"no task dirs under shared-from: {shared_from}")
    if not rs_tasks:
        raise SystemExit(f"no task dirs under rs-from: {rs_from}")

    missing_rs = sorted(set(shared_tasks) - set(rs_tasks))
    if missing_rs:
        raise SystemExit(
            f"rs-from missing {len(missing_rs)} task dirs present in shared-from, "
            f"e.g. {missing_rs[:3]}"
        )

    if out.exists() and not dry_run:
        shutil.rmtree(out)
    if not dry_run:
        out.mkdir(parents=True, exist_ok=True)

    n_shared = n_rs = n_out = 0
    per_prefix: Dict[int, int] = {}

    for task, shared_dir in sorted(shared_tasks.items()):
        rs_dir = rs_tasks[task]
        lines: List[str] = []
        for src in _rollout_files(shared_dir):
            chunk = _read_filtered_not(src, RS_PREFIXES)
            n_shared += len(chunk)
            lines.extend(chunk)
        for src in _rollout_files(rs_dir):
            chunk = _read_filtered(src, RS_PREFIXES)
            n_rs += len(chunk)
            for ln in chunk:
                rec = json.loads(ln)
                rec["rs_version"] = rs_version
                lines.append(json.dumps(rec, ensure_ascii=False))
        if not lines:
            raise SystemExit(f"no records merged for task {task}")

        dest_dir = out / task
        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / "rollouts_merged.jsonl"
            dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        n_out += len(lines)
        for ln in lines:
            per_prefix[_prefix_k(json.loads(ln))] = per_prefix.get(_prefix_k(json.loads(ln)), 0) + 1

    stats = {
        "tasks": len(shared_tasks),
        "records": n_out,
        "shared_records": n_shared,
        "rs_records": n_rs,
        "per_prefix": dict(sorted(per_prefix.items())),
        "out": str(out),
    }
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shared-from", type=Path, required=True)
    p.add_argument("--rs-from", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rs-version", required=True, choices=["v3", "v4", "v5"])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    stats = merge_buffers(
        args.shared_from,
        args.rs_from,
        args.out,
        rs_version=args.rs_version,
        dry_run=args.dry_run,
    )
    print(json.dumps(stats, indent=2))
    need = stats["tasks"] * V13_JOBS_PER_TASK * 4
    if stats["records"] < need:
        raise SystemExit(f"merged records={stats['records']} < floor {need}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
