#!/usr/bin/env python3
"""Leave-one-dataset-out test of experience retrieval.

Rank with target trials 1–8. Grade with held-out Spearman on trials 9+ and
hyperparameter direction agreement from full-run factors. Within algorithm only.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from mindware.components.experience.contrastive import FEATURE_NAMES, fit_linear_weights
from mindware.components.experience.retrieve import (
    COLD_WEIGHTS,
    DEFAULT_WEIGHTS,
    ExperienceBank,
    cold_features,
    combine_score,
    infer_role,
    query_importance_from_trials,
    spearman_transfer,
)

from dataset_split import is_lock, is_tune, load_split

RANKERS = ("random_same_algo", "cold", "importance", "cascade", "oracle")
SIGN_ROLES = ("learning_rate", "tree_depth", "regularization", "n_estimators")
GENUINE_RHO = 0.3
PREFIX_N = 8
MIN_SUFFIX = 3
CASCADE_POOL = 12


def _ds(exp: Dict[str, Any]) -> str:
    return str(((exp.get("problem") or {}).get("dataset_id") or "")).lower()


def _algo(exp: Dict[str, Any]) -> str:
    return str(((exp.get("problem") or {}).get("algorithm_id") or "")).lower()


def _task(exp: Dict[str, Any]) -> str:
    return str(((exp.get("problem") or {}).get("task_type") or "")).lower()


def factor_signs(exp: Dict[str, Any]) -> Dict[str, int]:
    """role -> +1 (higher better) / -1 (lower better)."""
    out: Dict[str, int] = {}
    for fac in (exp.get("knowledge") or {}).get("factors") or []:
        role = fac.get("role") or infer_role(str(fac.get("name") or ""))
        if not role or role not in SIGN_ROLES:
            continue
        text = str(fac.get("direction") or "").lower()
        if "higher" in text and "over-represented" in text:
            out[role] = 1
        elif "lower" in text and "over-represented" in text:
            out[role] = -1
    return out


def sign_agree(a: Dict[str, int], b: Dict[str, int], role: str) -> Optional[int]:
    if role not in a or role not in b:
        return None
    return 1 if a[role] == b[role] else 0


def query_from_prefix(exp: Dict[str, Any], prefix: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ret = exp.get("retrieval") or {}
    prob = exp.get("problem") or {}
    flt = (exp.get("delivery") or {}).get("filters") or {}
    hib = bool(prob.get("higher_is_better", True))
    return {
        "dataset_id": prob.get("dataset_id"),
        "algorithm_id": prob.get("algorithm_id"),
        "filters": {
            "domain": flt.get("domain") or "tabular_ml",
            "task_type": prob.get("task_type"),
            "representation": "structured_config",
        },
        "higher_is_better": hib,
        "importance_by_role": query_importance_from_trials(prefix, higher_is_better=hib),
        "roles": ret.get("roles") or [],
        "profile_vec": ret.get("profile_vec") or {},
    }


def topk_ids(scored: Sequence[Tuple[str, float]], k: int) -> List[str]:
    ordered = sorted(scored, key=lambda kv: kv[1], reverse=True)
    return [sid for sid, _ in ordered[:k]]


def rng_for(exp_id: str) -> random.Random:
    h = hashlib.md5(exp_id.encode("utf-8")).hexdigest()
    return random.Random(int(h[:8], 16))


def summarize(rows: List[Dict[str, Any]], ranker: str) -> Dict[str, float]:
    if not rows:
        return {}
    rho1 = [r[f"{ranker}_rho1"] for r in rows if r.get(f"{ranker}_rho1") is not None]
    rho3 = [r[f"{ranker}_rho3"] for r in rows if r.get(f"{ranker}_rho3") is not None]
    lr = [r[f"{ranker}_lr_agree"] for r in rows if r.get(f"{ranker}_lr_agree") is not None]
    gen = [r[f"{ranker}_genuine"] for r in rows if r.get(f"{ranker}_genuine") is not None]
    hit = [r[f"{ranker}_oracle_hit5"] for r in rows if r.get(f"{ranker}_oracle_hit5") is not None]
    n = len(rows)

    def finite(xs):
        return [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]

    def mean(xs):
        xs = finite(xs)
        return float(sum(xs) / len(xs)) if xs else float("nan")

    return {
        "n": float(n),
        "n_lr": float(len(finite(lr))),
        "mean_rho_top1": mean(rho1),
        "mean_rho_top3": mean(rho3),
        "frac_rho_ge_0.3": mean([1.0 if x >= GENUINE_RHO else 0.0 for x in rho1]),
        "frac_rho_gt_0": mean([1.0 if x > 0 else 0.0 for x in rho1]),
        "lr_agree": mean(lr),
        "genuine": mean(gen),
        "oracle_top5_hit": mean(hit),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=ROOT / "history_data" / "experience_bank.jsonl")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "benchmark_data" / "real_hpo_v2_mltest20_algo40" / "analysis" / "lodo_similarity",
    )
    p.add_argument("--prefix-n", type=int, default=PREFIX_N)
    p.add_argument(
        "--split",
        choices=("all", "tune", "lock"),
        default="all",
        help="Restrict target datasets to TUNE / LOCK / all. Candidates stay same-algo LODO.",
    )
    p.add_argument("--queries-name", default="lodo_queries.csv")
    p.add_argument("--summary-name", default="lodo_summary.csv")
    args = p.parse_args(argv)

    bank = ExperienceBank.load(args.bank)
    print(f"bank={len(bank)} weights={bank.weights}")
    split = load_split() if args.split != "all" else None

    usable: List[Dict[str, Any]] = []
    for exp in bank.experiences:
        trials = (exp.get("retrieval") or {}).get("trials") or []
        if len(trials) < args.prefix_n + MIN_SUFFIX:
            continue
        if split is not None:
            ds = _ds(exp)
            if args.split == "tune" and not is_tune(ds, split):
                continue
            if args.split == "lock" and not is_lock(ds, split):
                continue
        usable.append(exp)
    print(f"usable targets (prefix {args.prefix_n} + suffix>={MIN_SUFFIX}): {len(usable)}")

    by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for exp in usable:
        by_group[(_task(exp), _algo(exp))].append(exp)

    warm_w = dict(bank.weights) if bank.weights else dict(DEFAULT_WEIGHTS)
    rho_cache: Dict[Tuple[str, str, str], float] = {}

    def rho(src: Dict[str, Any], trials: Sequence[Dict[str, Any]], tag: str, tgt_id: str) -> float:
        key = (str(src.get("id")), tgt_id, tag)
        if key not in rho_cache:
            rho_cache[key] = float(
                spearman_transfer(
                    src,
                    trials,
                    target_algo=_algo(src),
                    target_higher_is_better=bool((src.get("problem") or {}).get("higher_is_better", True)),
                )
            )
        return rho_cache[key]

    query_rows: List[Dict[str, Any]] = []
    pair_xy: List[List[float]] = []
    pair_y: List[float] = []

    n_groups = len(by_group)
    for gi, ((task, algo), members) in enumerate(sorted(by_group.items()), 1):
        if len(members) < 3:
            continue
        print(f"  [{gi}/{n_groups}] {task}/{algo} n={len(members)}", flush=True)
        for tgt in members:
            tgt_id = str(tgt.get("id"))
            trials = (tgt.get("retrieval") or {}).get("trials") or []
            prefix = trials[: args.prefix_n]
            suffix = trials[args.prefix_n :]
            cands = [s for s in members if _ds(s) != _ds(tgt)]
            if len(cands) < 3:
                continue

            q = query_from_prefix(tgt, prefix)
            tgt_signs = factor_signs(tgt)

            scored: Dict[str, Dict[str, float]] = {}
            for src in cands:
                sid = str(src.get("id"))
                feats = cold_features(q, src)
                held = rho(src, suffix, "suffix", tgt_id)
                pre = rho(src, prefix, "prefix", tgt_id)
                src_signs = factor_signs(src)
                lr = sign_agree(tgt_signs, src_signs, "learning_rate")
                depth = sign_agree(tgt_signs, src_signs, "tree_depth")
                genuine = 1.0 if (held >= GENUINE_RHO and (lr is None or lr == 1)) else 0.0
                scored[sid] = {
                    "cold": combine_score(feats, COLD_WEIGHTS),
                    "importance": combine_score(feats, warm_w),
                    "prefix_rho": pre,
                    "held_rho": held,
                    "lr_agree": float("nan") if lr is None else float(lr),
                    "depth_agree": float("nan") if depth is None else float(depth),
                    "genuine": genuine,
                    **{f"f_{k}": float(feats[k]) for k in FEATURE_NAMES},
                }
                pair_xy.append([float(feats[k]) for k in FEATURE_NAMES])
                pair_y.append(max(0.0, held))

            def list_scores(key: str) -> List[Tuple[str, float]]:
                return [(sid, scored[sid][key]) for sid in scored]

            # cascade: top CASCADE_POOL by importance, drop negative prefix rho, mix
            imp_ranked = topk_ids(list_scores("importance"), max(CASCADE_POOL, 3))
            casc_scored: List[Tuple[str, float]] = []
            for sid in imp_ranked:
                pre = scored[sid]["prefix_rho"]
                if pre < 0.0:
                    continue
                mix = 0.6 * scored[sid]["importance"] + 0.4 * max(0.0, pre)
                casc_scored.append((sid, mix))
            if not casc_scored:
                casc_scored = [(sid, scored[sid]["importance"]) for sid in imp_ranked]

            rng = rng_for(tgt_id)
            rand_ids = [str(s.get("id")) for s in cands]
            rng.shuffle(rand_ids)
            ranker_top = {
                "random_same_algo": rand_ids,
                "cold": topk_ids(list_scores("cold"), 5),
                "importance": topk_ids(list_scores("importance"), 5),
                "cascade": [sid for sid, _ in sorted(casc_scored, key=lambda kv: kv[1], reverse=True)],
                "oracle": topk_ids(list_scores("held_rho"), 5),
            }

            oracle5 = set(ranker_top["oracle"][:5])
            row: Dict[str, Any] = {
                "target_id": tgt_id,
                "dataset": _ds(tgt),
                "algorithm": algo,
                "task_type": task,
                "n_cands": len(cands),
                "n_suffix": len(suffix),
                "oracle_rho1": scored[ranker_top["oracle"][0]]["held_rho"] if ranker_top["oracle"] else None,
            }
            for name in RANKERS:
                ids = ranker_top[name]
                if not ids:
                    continue
                top1 = ids[0]
                top3 = ids[:3]
                rhos = [scored[s]["held_rho"] for s in top3 if s in scored]
                row[f"{name}_top1"] = top1
                row[f"{name}_rho1"] = scored[top1]["held_rho"]
                row[f"{name}_rho3"] = sum(rhos) / len(rhos) if rhos else None
                row[f"{name}_lr_agree"] = scored[top1]["lr_agree"]
                row[f"{name}_depth_agree"] = scored[top1]["depth_agree"]
                row[f"{name}_genuine"] = scored[top1]["genuine"]
                row[f"{name}_oracle_hit5"] = 1.0 if top1 in oracle5 else 0.0
                row[f"{name}_prefix_rho1"] = scored[top1]["prefix_rho"]
            query_rows.append(row)
        # Checkpoint after each algorithm group so a killed run still leaves rows.
        args.out_dir.mkdir(parents=True, exist_ok=True)
        qpath = args.out_dir / args.queries_name
        if query_rows:
            keys = list(query_rows[0].keys())
            with qpath.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(query_rows)
            print(f"    checkpoint {len(query_rows)} queries → {qpath}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    qpath = args.out_dir / args.queries_name
    if query_rows:
        keys = list(query_rows[0].keys())
        with qpath.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(query_rows)

    slices = [("all", query_rows)]
    for task in ("classification", "regression"):
        slices.append((task, [r for r in query_rows if r["task_type"] == task]))

    summary_rows = []
    print("\n=== LODO within-algorithm ===")
    for slice_name, subset in slices:
        if not subset:
            continue
        print(f"\n[{slice_name}] n={len(subset)}")
        print(
            f"{'ranker':<22} {'rho@1':>8} {'rho@3':>8} {'ρ≥0.3':>8} {'ρ>0':>8} "
            f"{'LR agr':>8} {'genuine':>8} {'hit@5':>8}"
        )
        for name in RANKERS:
            s = summarize(subset, name)
            if not s:
                continue
            summary_rows.append({"slice": slice_name, "ranker": name, **s})
            print(
                f"{name:<22} {s['mean_rho_top1']:8.3f} {s['mean_rho_top3']:8.3f} "
                f"{s['frac_rho_ge_0.3']:8.3f} {s['frac_rho_gt_0']:8.3f} "
                f"{s['lr_agree']:8.3f} {s['genuine']:8.3f} {s['oracle_top5_hit']:8.3f}"
            )

    spath = args.out_dir / args.summary_name
    if summary_rows:
        with spath.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)

    lodo_w = fit_linear_weights(pair_xy, pair_y) if len(pair_y) >= 16 else dict(DEFAULT_WEIGHTS)
    wpath = args.out_dir / "lodo_fitted_weights.json"
    wpath.write_text(json.dumps({"heldout_rho_clip01": lodo_w, "deployed": warm_w}, indent=2) + "\n")
    print(f"\nLODO-fitted weights (held-out ρ, clip≥0): {lodo_w}")
    print(f"deployed weights: {warm_w}")
    print(f"wrote {qpath}")
    print(f"wrote {spath}")
    print(f"pairs used for weight fit: {len(pair_y)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
