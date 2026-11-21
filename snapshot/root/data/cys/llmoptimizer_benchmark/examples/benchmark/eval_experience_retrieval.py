#!/usr/bin/env python3
"""Smoke-test experience retrieval (unit checks + leave-one-out neighbors)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.components.experience.profile import cosine
from mindware.components.experience.retrieve import (
    ExperienceBank,
    format_retrieved_block,
    query_importance_from_trials,
    retrieve_experiences,
    spearman,
)


def _unit_checks() -> None:
    assert abs(cosine([1, 0], [1, 0]) - 1.0) < 1e-9
    assert abs(cosine([1, 0], [0, 1])) < 1e-9
    # Perfect rank agreement
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) < 0
    print("unit checks ok")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bank",
        type=Path,
        default=ROOT / "history_data" / "experience_bank.jsonl",
    )
    p.add_argument("--dataset", default="pc4")
    p.add_argument("--algorithm", default="adaboost")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--unit-only", action="store_true")
    args = p.parse_args(argv)

    _unit_checks()
    if args.unit_only:
        return 0
    if not args.bank.is_file():
        print(f"bank missing: {args.bank} (run build_experience_bank.py)")
        return 2

    bank = ExperienceBank.load(args.bank)
    print(f"bank size={len(bank)} weights={bank.weights}")

    held = None
    for exp in bank.experiences:
        prob = exp.get("problem") or {}
        if (
            str(prob.get("dataset_id") or "").lower() == args.dataset.lower()
            and str(prob.get("algorithm_id") or "").lower() == args.algorithm.lower()
        ):
            held = exp
            break
    if held is None:
        print(f"no held-out experience for {args.dataset}/{args.algorithm}; using first row")
        held = bank.experiences[0]
        args.dataset = (held.get("problem") or {}).get("dataset_id")
        args.algorithm = (held.get("problem") or {}).get("algorithm_id")

    ret = held.get("retrieval") or {}
    trials = ret.get("trials") or []
    query = {
        "dataset_id": args.dataset,
        "algorithm_id": args.algorithm,
        "filters": {
            "domain": "tabular_ml",
            "task_type": (held.get("problem") or {}).get("task_type"),
            "representation": "structured_config",
        },
        "importance_by_role": {},
        "roles": ret.get("roles") or [],
        "profile_vec": ret.get("profile_vec") or {},
    }
    print("\n=== cold start (no local importance) ===")
    cold = retrieve_experiences(bank, query, k=args.k, target_trials=None, rerank_after=8)
    print(format_retrieved_block(cold)[:2000])

    prefix = trials[:8]
    if len(prefix) >= 8:
        query["importance_by_role"] = query_importance_from_trials(prefix)
        print("\n=== after 8 local trials + Spearman rerank ===")
        warm = retrieve_experiences(bank, query, k=args.k, target_trials=prefix, rerank_after=8)
        print(format_retrieved_block(warm)[:2500])
        print("\nJSON hits:")
        for h in warm:
            print(
                json.dumps(
                    {
                        "id": h.exp_id,
                        "score": round(h.score, 4),
                        "spearman": None if h.spearman is None else round(h.spearman, 4),
                        "features": {k: round(v, 4) for k, v in h.features.items()},
                    }
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
