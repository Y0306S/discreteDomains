#!/usr/bin/env python3
"""Offline-reweight Setup B (or any) RL rollout JSONL using stored reward fields.

Uses top-level reward.delta_best / reward.rel_score and parse/dup/eval flags,
so arms can change Δbest vs absolute weights without re-rolling.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_ARMS = {
    "hybrid": {
        "delta_best": 1.0,
        "absolute": 0.1,
        "parse_fail": 0.5,
        "duplicate": 0.1,
        "eval_fail": 1.0,
    },
    "dbest": {
        "delta_best": 1.0,
        "absolute": 0.0,
        "parse_fail": 0.5,
        "duplicate": 0.1,
        "eval_fail": 1.0,
    },
    "abs": {
        "delta_best": 0.0,
        "absolute": 1.0,
        "parse_fail": 0.5,
        "duplicate": 0.1,
        "eval_fail": 1.0,
    },
    "mix_strong": {
        "delta_best": 1.0,
        "absolute": 0.5,
        "parse_fail": 0.5,
        "duplicate": 0.1,
        "eval_fail": 1.0,
    },
}


def load_arms(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return DEFAULT_ARMS
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, float]] = {}
    for name, cfg in data.items():
        out[name] = {
            "delta_best": float(cfg.get("delta_best", 1.0)),
            "absolute": float(cfg.get("absolute", 0.1)),
            "parse_fail": float(cfg.get("parse_fail", 0.5)),
            "duplicate": float(cfg.get("duplicate", 0.1)),
            "eval_fail": float(cfg.get("eval_fail", 1.0)),
        }
    return out


def task_keys_from_order(path: Path | None) -> set[tuple[str, str, str]] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    keys: set[tuple[str, str, str]] = set()
    for row in data:
        keys.add((str(row["task_type"]), str(row["dataset"]), str(row["algorithm"])))
    return keys


def recompute_reward(reward: dict[str, Any], weights: dict[str, float]) -> dict[str, Any]:
    delta = float(reward.get("delta_best", 0.0))
    rel = float(reward.get("rel_score", 0.0))
    parse_ok = bool(reward.get("parse_ok", True))
    duplicate = bool(reward.get("duplicate", False))
    eval_ok = bool(reward.get("eval_ok", True))
    components = {
        "delta_best": weights["delta_best"] * delta,
        "absolute": weights["absolute"] * rel if eval_ok else 0.0,
        "parse_fail": -weights["parse_fail"] if not parse_ok else 0.0,
        "duplicate": -weights["duplicate"] if duplicate else 0.0,
        "eval_fail": -weights["eval_fail"] if not eval_ok else 0.0,
    }
    out = dict(reward)
    out["reward"] = float(sum(components.values()))
    out["reward_components"] = components
    out["reward_weights"] = dict(weights)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Offline reweight RL rollout buffers.")
    p.add_argument("--src-root", type=Path, required=True)
    p.add_argument("--dst-root", type=Path, required=True)
    p.add_argument("--arm", required=True, help="Arm name in arms file / DEFAULT_ARMS")
    p.add_argument("--arms-file", type=Path, default=None)
    p.add_argument(
        "--task-order-file",
        type=Path,
        default=None,
        help="If set, only copy tasks listed in this JSON (task_type/dataset/algorithm).",
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    arms = load_arms(args.arms_file)
    if args.arm not in arms:
        raise SystemExit(f"Unknown arm {args.arm!r}; known={sorted(arms)}")
    weights = arms[args.arm]
    allow = task_keys_from_order(args.task_order_file)

    if args.dst_root.exists():
        if not args.overwrite:
            raise SystemExit(f"dst exists: {args.dst_root} (pass --overwrite)")
        shutil.rmtree(args.dst_root)
    args.dst_root.mkdir(parents=True, exist_ok=True)

    n_in = n_out = 0
    for src in sorted(args.src_root.rglob("rollouts_*.jsonl")):
        rel = src.relative_to(args.src_root)
        # Filter by parent folder task if order given: folders like cls_dataset_algorithm
        parent = src.parent.name
        keep_dir = True
        if allow is not None:
            keep_dir = False
            for tt, ds, algo in allow:
                if parent == f"{tt}_{ds}_{algo}" or parent.endswith(f"_{ds}_{algo}"):
                    keep_dir = True
                    break
            # also accept dataset_algorithm without task_type prefix variants
            if not keep_dir:
                for tt, ds, algo in allow:
                    if f"_{ds}_{algo}" in f"_{parent}" or parent == f"{ds}_{algo}":
                        keep_dir = True
                        break
        if not keep_dir:
            continue

        dst = args.dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                n_in += 1
                row = json.loads(line)
                if allow is not None:
                    key = (
                        str(row.get("task_type") or ""),
                        str(row.get("dataset") or ""),
                        str(row.get("algorithm") or ""),
                    )
                    # Some buffers omit task_type; match on dataset+algorithm
                    ok = key in allow or any(
                        key[1] == ds and key[2] == algo for _, ds, algo in allow
                    )
                    if not ok:
                        continue
                reward = row.get("reward") or {}
                if not isinstance(reward, dict):
                    continue
                row["reward"] = recompute_reward(reward, weights)
                row["reward_arm"] = args.arm
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_out += 1

        # copy sidecar manifests if present
        for side in src.parent.glob("*.manifest.json"):
            shutil.copy2(side, dst.parent / side.name)

    meta = {
        "src_root": str(args.src_root),
        "dst_root": str(args.dst_root),
        "arm": args.arm,
        "weights": weights,
        "task_order_file": str(args.task_order_file) if args.task_order_file else None,
        "n_in_lines": n_in,
        "n_out_lines": n_out,
    }
    (args.dst_root / "reweight_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    if n_out == 0:
        raise SystemExit("No records written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
