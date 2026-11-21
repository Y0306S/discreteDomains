#!/usr/bin/env python3
"""Recompute train_member flags in an rs_mix rollout buffer.

Used when collect used strict_first4 but train should use adaptive 3+1.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _delta_best(record: Dict[str, Any]) -> float:
    rew = record.get("reward") or {}
    try:
        return float(rew.get("delta_best", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _reward_scalar(record: Dict[str, Any]) -> float:
    rew = record.get("reward") or {}
    try:
        return float(rew.get("reward", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _beats_incumbent(record: Dict[str, Any]) -> bool:
    return _delta_best(record) > 0.0


def choose_train_members(
    records: List[Dict[str, Any]],
    *,
    n_base: int,
    rng: random.Random,
    policy: str = "adaptive",
) -> Tuple[List[int], Optional[int], Optional[int]]:
    n = len(records)
    base = list(range(min(n_base, n)))
    if policy == "strict_first4":
        return base, None, None
    if any(_beats_incumbent(records[i]) for i in base):
        return base, None, None
    extras = list(range(n_base, n))
    improvers = [i for i in extras if _beats_incumbent(records[i])]
    if not improvers:
        return base, None, None
    injected = max(improvers, key=lambda i: (_delta_best(records[i]), _reward_scalar(records[i]), i))
    drop = rng.choice(base)
    train = [i for i in base if i != drop] + [injected]
    return train, drop, injected


def retag_file(path: Path, *, seed: int, n_base: int, policy: str) -> int:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return 0

    records = [json.loads(ln) for ln in lines]
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        by_group.setdefault(str(rec["group_id"]), []).append(rec)

    changed = 0
    out_lines: List[str] = []
    for gid in sorted(by_group):
        group = sorted(by_group[gid], key=lambda r: int(r.get("sample_i", 0)))
        k = int(group[0].get("prefix_k", 0))
        rng = random.Random(int(seed) * 10007 + k * 17)
        train_idxs, dropped, injected = choose_train_members(
            group, n_base=n_base, rng=rng, policy=policy
        )
        train_set = set(train_idxs)
        for rec in group:
            old = bool(rec.get("train_member"))
            new = int(rec["sample_i"]) in train_set
            if old != new:
                changed += 1
            rec["train_member"] = new
            rec["train_sample_idxs"] = train_idxs
            rec["dropped_base_idx"] = dropped
            rec["injected_from"] = injected
            out_lines.append(json.dumps(rec, ensure_ascii=False, default=str))

    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return changed


def retag_buffer(root: Path, *, seed: int, n_base: int, policy: str) -> dict:
    total_changed = 0
    n_files = 0
    for path in sorted(root.rglob("rollouts_*.jsonl")):
        total_changed += retag_file(path, seed=seed, n_base=n_base, policy=policy)
        n_files += 1
    return {"root": str(root), "files": n_files, "records_changed": total_changed, "policy": policy}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--buffer-root", type=Path, required=True)
    p.add_argument("--policy", choices=["adaptive", "strict_first4"], default="adaptive")
    p.add_argument("--n-base", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    stats = retag_buffer(args.buffer_root, seed=args.seed, n_base=args.n_base, policy=args.policy)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
