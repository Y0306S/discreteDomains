#!/usr/bin/env python3
"""Honest ceiling for similar-task retrieval.

Rank cascade on trials 1–8. Honest oracle *selects* on trials 9–24 (or first
half of the suffix) and *grades* on trials 25–40 (second half). Naive oracle
selects on the grade slice (biased max). Random / cascade are graded on the
same grade slice. Removes max-over-candidates inflation of the old oracle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.components.experience.retrieve import (
    COLD_WEIGHTS,
    DEFAULT_WEIGHTS,
    ExperienceBank,
    cold_features,
    combine_score,
    query_importance_from_trials,
    spearman_transfer,
)

PREFIX_N = 8
SELECT_N = 16
CASCADE_POOL = 12
GENUINE_RHO = 0.3


def _ds(exp: Dict[str, Any]) -> str:
    return str(((exp.get("problem") or {}).get("dataset_id") or "")).lower()


def _algo(exp: Dict[str, Any]) -> str:
    return str(((exp.get("problem") or {}).get("algorithm_id") or "")).lower()


def _task(exp: Dict[str, Any]) -> str:
    return str(((exp.get("problem") or {}).get("task_type") or "")).lower()


def query_from_prefix(exp: Dict[str, Any], prefix: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ret = exp.get("retrieval") or {}
    prob = exp.get("problem") or {}
    flt = (exp.get("delivery") or {}).get("filters") or {}
    return {
        "dataset_id": prob.get("dataset_id"),
        "algorithm_id": prob.get("algorithm_id"),
        "filters": {
            "domain": flt.get("domain") or "tabular_ml",
            "task_type": prob.get("task_type"),
            "representation": "structured_config",
        },
        "importance_by_role": query_importance_from_trials(prefix),
        "roles": ret.get("roles") or [],
        "profile_vec": ret.get("profile_vec") or {},
    }


def topk_ids(scored: Sequence[Tuple[str, float]], k: int) -> List[str]:
    return [sid for sid, _ in sorted(scored, key=lambda kv: kv[1], reverse=True)[:k]]


def rng_for(exp_id: str) -> random.Random:
    h = hashlib.md5(exp_id.encode("utf-8")).hexdigest()
    return random.Random(int(h[:8], 16))


def mean(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(sum(xs) / len(xs)) if xs else float("nan")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=ROOT / "history_data" / "experience_bank.jsonl")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "benchmark_data" / "real_hpo_v2_mltest20_algo40" / "analysis" / "lodo_similarity",
    )
    p.add_argument("--prefix-n", type=int, default=PREFIX_N)
    p.add_argument("--select-n", type=int, default=SELECT_N)
    args = p.parse_args(argv)

    bank = ExperienceBank.load(args.bank)
    print(f"bank={len(bank)} weights={bank.weights}")
    warm_w = dict(bank.weights) if bank.weights else dict(DEFAULT_WEIGHTS)

    min_len = args.prefix_n + 6  # 3 select + 3 grade
    usable = [
        e
        for e in bank.experiences
        if len((e.get("retrieval") or {}).get("trials") or []) >= min_len
    ]
    by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for exp in usable:
        by_group[(_task(exp), _algo(exp))].append(exp)

    rows: List[Dict[str, Any]] = []
    n_groups = len(by_group)
    for gi, ((task, algo), members) in enumerate(sorted(by_group.items()), 1):
        if len(members) < 3:
            continue
        print(f"  [{gi}/{n_groups}] {task}/{algo} n={len(members)}")
        for tgt in members:
            tgt_id = str(tgt.get("id"))
            trials = (tgt.get("retrieval") or {}).get("trials") or []
            prefix = trials[: args.prefix_n]
            rest = trials[args.prefix_n :]
            n_sel = min(args.select_n, max(3, len(rest) // 2))
            if len(rest) - n_sel < 3:
                n_sel = len(rest) - 3
            if n_sel < 3:
                continue
            select, grade = rest[:n_sel], rest[n_sel:]
            cands = [s for s in members if _ds(s) != _ds(tgt)]
            if len(cands) < 3:
                continue

            q = query_from_prefix(tgt, prefix)
            scored: Dict[str, Dict[str, float]] = {}
            for src in cands:
                sid = str(src.get("id"))
                feats = cold_features(q, src)
                scored[sid] = {
                    "cold": combine_score(feats, COLD_WEIGHTS),
                    "importance": combine_score(feats, warm_w),
                    "prefix_rho": float(spearman_transfer(src, prefix)),
                    "select_rho": float(spearman_transfer(src, select)),
                    "grade_rho": float(spearman_transfer(src, grade)),
                }

            imp_ranked = topk_ids([(sid, scored[sid]["importance"]) for sid in scored], max(CASCADE_POOL, 3))
            casc: List[Tuple[str, float]] = []
            for sid in imp_ranked:
                pre = scored[sid]["prefix_rho"]
                if pre < 0.0:
                    continue
                casc.append((sid, 0.6 * scored[sid]["importance"] + 0.4 * max(0.0, pre)))
            if not casc:
                casc = [(sid, scored[sid]["importance"]) for sid in imp_ranked]
            casc_ids = [sid for sid, _ in sorted(casc, key=lambda kv: kv[1], reverse=True)]

            rng = rng_for(tgt_id)
            rand_ids = list(scored)
            rng.shuffle(rand_ids)

            honest_id = max(scored, key=lambda s: scored[s]["select_rho"])
            naive_id = max(scored, key=lambda s: scored[s]["grade_rho"])
            cold_id = max(scored, key=lambda s: scored[s]["cold"])

            def g(sid: str) -> float:
                return scored[sid]["grade_rho"]

            row = {
                "target_id": tgt_id,
                "dataset": _ds(tgt),
                "algorithm": algo,
                "task_type": task,
                "n_cands": len(cands),
                "n_select": len(select),
                "n_grade": len(grade),
                "random_rho1": g(rand_ids[0]),
                "cold_rho1": g(cold_id),
                "cascade_rho1": g(casc_ids[0]),
                "cascade_prefix_rho": scored[casc_ids[0]]["prefix_rho"],
                "honest_oracle_rho1": g(honest_id),
                "honest_oracle_select_rho": scored[honest_id]["select_rho"],
                "naive_oracle_rho1": g(naive_id),
            }
            rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    qpath = args.out_dir / "honest_ceiling_queries.csv"
    with qpath.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    keys = [
        "random_rho1",
        "cold_rho1",
        "cascade_rho1",
        "honest_oracle_rho1",
        "naive_oracle_rho1",
    ]
    print(f"\n=== Honest ceiling n={len(rows)} ===")
    print(f"{'ranker':<22} {'grade ρ@1':>10} {'ρ≥0.3':>8}")
    summary = []
    for k in keys:
        vals = [float(r[k]) for r in rows]
        ge = mean([1.0 if v >= GENUINE_RHO else 0.0 for v in vals])
        m = mean(vals)
        print(f"{k:<22} {m:10.3f} {ge:8.3f}")
        summary.append({"ranker": k, "n": len(rows), "mean_rho_top1": m, "frac_rho_ge_0.3": ge})

    casc = mean([r["cascade_rho1"] for r in rows])
    hon = mean([r["honest_oracle_rho1"] for r in rows])
    gap = hon - casc
    print(f"\nhonest_oracle - cascade = {gap:+.3f}")
    print("GATE: skip Phase 2 if |gap| <= 0.05 (near ceiling); else proceed if gap > 0.10.")
    spath = args.out_dir / "honest_ceiling_summary.json"
    spath.write_text(
        json.dumps(
            {
                "n": len(rows),
                "cascade": casc,
                "honest_oracle": hon,
                "naive_oracle": mean([r["naive_oracle_rho1"] for r in rows]),
                "random": mean([r["random_rho1"] for r in rows]),
                "cold": mean([r["cold_rho1"] for r in rows]),
                "gap_honest_minus_cascade": gap,
                "skip_phase2": abs(gap) <= 0.05,
                "proceed_phase2": gap > 0.10,
                "rows": summary,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {qpath}")
    print(f"wrote {spath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
