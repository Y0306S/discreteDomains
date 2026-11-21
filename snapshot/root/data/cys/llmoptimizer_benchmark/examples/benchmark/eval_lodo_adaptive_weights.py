#!/usr/bin/env python3
"""Nested / online weight updates for LODO retrieval — no leak into the query being graded.

For each held-out dataset, linear weights are fit only on *other* datasets' pairs
(y = clip(held-out Spearman, 0, 1)). Ranking that dataset never uses its own label.

Within-algorithm, algo_match is always 1, so it is dropped from the learned formula.
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

from mindware.components.experience.contrastive import fit_linear_weights
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

# Learned on within-algo pairs (algo_match is constant → omitted).
LEARN_KEYS = ("importance_cos", "profile_cos", "role_jaccard")
GENUINE_RHO = 0.3
PREFIX_N = 8
MIN_SUFFIX = 3
CASCADE_POOL = 12
SIGN_ROLES = ("learning_rate", "tree_depth", "regularization", "n_estimators")


def _ds(exp: Dict[str, Any]) -> str:
    return str(((exp.get("problem") or {}).get("dataset_id") or "")).lower()


def _algo(exp: Dict[str, Any]) -> str:
    return str(((exp.get("problem") or {}).get("algorithm_id") or "")).lower()


def _task(exp: Dict[str, Any]) -> str:
    return str(((exp.get("problem") or {}).get("task_type") or "")).lower()


def factor_signs(exp: Dict[str, Any]) -> Dict[str, int]:
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


def pad_weights(three: Dict[str, float]) -> Dict[str, float]:
    """Map 3 learned keys onto the 4-key combine_score interface (algo unused)."""
    return {
        "importance_cos": float(three.get("importance_cos", 0.0)),
        "algo_match": 0.0,
        "profile_cos": float(three.get("profile_cos", 0.0)),
        "role_jaccard": float(three.get("role_jaccard", 0.0)),
    }


def fit_three(X: List[List[float]], y: List[float]) -> Dict[str, float]:
    raw = fit_linear_weights(X, y)
    # fit_linear_weights zips FEATURE_NAMES (4 keys). If we pass 3-col X it will
    # mis-align. Fit with a tiny local ridge instead.
    try:
        import numpy as np
    except ImportError:
        return {"importance_cos": 0.5, "profile_cos": 0.25, "role_jaccard": 0.25}
    if len(X) < 8:
        return {"importance_cos": 0.5, "profile_cos": 0.25, "role_jaccard": 0.25}
    A = np.asarray(X, dtype=float)
    b = np.clip(np.asarray(y, dtype=float), 0.0, 1.0)
    if A.ndim != 2 or A.shape[1] != 3:
        return {"importance_cos": 0.5, "profile_cos": 0.25, "role_jaccard": 0.25}
    gram = A.T @ A + 1e-2 * np.eye(3)
    w = np.linalg.solve(gram, A.T @ b)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 1e-12:
        return {"importance_cos": 0.5, "profile_cos": 0.25, "role_jaccard": 0.25}
    w = w / s
    return {k: float(v) for k, v in zip(LEARN_KEYS, w.tolist())}


def sgd_update(w: Dict[str, float], x3: Sequence[float], y: float, eta: float = 0.05) -> Dict[str, float]:
    pred = sum(w[k] * float(x3[i]) for i, k in enumerate(LEARN_KEYS))
    err = max(0.0, min(1.0, y)) - pred
    out = {}
    for i, k in enumerate(LEARN_KEYS):
        out[k] = max(0.0, w[k] + eta * err * float(x3[i]))
    s = sum(out.values())
    if s <= 1e-12:
        return dict(w)
    return {k: v / s for k, v in out.items()}


def cascade_order(score_by_id: Dict[str, float], prefix_rho: Dict[str, float]) -> List[str]:
    pool = topk_ids(list(score_by_id.items()), max(CASCADE_POOL, 3))
    mixed: List[Tuple[str, float]] = []
    for sid in pool:
        pre = prefix_rho[sid]
        if pre < 0.0:
            continue
        mixed.append((sid, 0.6 * score_by_id[sid] + 0.4 * max(0.0, pre)))
    if not mixed:
        mixed = [(sid, score_by_id[sid]) for sid in pool]
    return [sid for sid, _ in sorted(mixed, key=lambda kv: kv[1], reverse=True)]


def finite_mean(xs: List[Optional[float]]) -> float:
    vals = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def summarize(rows: List[Dict[str, Any]], ranker: str) -> Dict[str, float]:
    rho1 = [r.get(f"{ranker}_rho1") for r in rows]
    rho3 = [r.get(f"{ranker}_rho3") for r in rows]
    gen = [r.get(f"{ranker}_genuine") for r in rows]
    lr = [r.get(f"{ranker}_lr_agree") for r in rows]
    r1 = [x for x in rho1 if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return {
        "n": float(len(rows)),
        "mean_rho_top1": finite_mean(rho1),
        "mean_rho_top3": finite_mean(rho3),
        "frac_rho_ge_0.3": finite_mean([1.0 if (x is not None and x >= GENUINE_RHO) else 0.0 for x in r1]) if r1 else float("nan"),
        "genuine": finite_mean(gen),
        "lr_agree": finite_mean(lr),
    }


def grade_top(ids: List[str], held: Dict[str, float], genuine: Dict[str, float], lr: Dict[str, float]) -> Dict[str, Any]:
    if not ids:
        return {"rho1": None, "rho3": None, "genuine": None, "lr_agree": None, "top1": ""}
    top1 = ids[0]
    top3 = ids[:3]
    rhos = [held[s] for s in top3 if s in held]
    return {
        "top1": top1,
        "rho1": held.get(top1),
        "rho3": sum(rhos) / len(rhos) if rhos else None,
        "genuine": genuine.get(top1),
        "lr_agree": lr.get(top1),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=ROOT / "history_data" / "experience_bank.jsonl")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "benchmark_data" / "real_hpo_v2_mltest20_algo40" / "analysis" / "lodo_similarity",
    )
    args = p.parse_args(argv)

    bank = ExperienceBank.load(args.bank)
    frozen = dict(bank.weights) if bank.weights else dict(DEFAULT_WEIGHTS)
    print(f"bank={len(bank)} frozen_weights={frozen}")

    usable = []
    for exp in bank.experiences:
        trials = (exp.get("retrieval") or {}).get("trials") or []
        if len(trials) >= PREFIX_N + MIN_SUFFIX:
            usable.append(exp)

    by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for exp in usable:
        by_group[(_task(exp), _algo(exp))].append(exp)

    # pair records: training rows for nested / online fits
    # keyed by (task, algo, tgt_ds) → list of dicts
    train_rows: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    query_payloads: List[Dict[str, Any]] = []

    n_groups = len(by_group)
    for gi, ((task, algo), members) in enumerate(sorted(by_group.items()), 1):
        if len(members) < 3:
            continue
        print(f"  [{gi}/{n_groups}] {task}/{algo} n={len(members)}  (pair cache)")
        rho_cache: Dict[Tuple[str, str, str], float] = {}

        def rho(src, trials, tag, tgt_id):
            key = (str(src.get("id")), tgt_id, tag)
            if key not in rho_cache:
                rho_cache[key] = float(spearman_transfer(src, trials))
            return rho_cache[key]

        for tgt in members:
            tgt_id = str(tgt.get("id"))
            tgt_ds = _ds(tgt)
            trials = (tgt.get("retrieval") or {}).get("trials") or []
            prefix = trials[:PREFIX_N]
            suffix = trials[PREFIX_N:]
            cands = [s for s in members if _ds(s) != tgt_ds]
            if len(cands) < 3:
                continue
            q = query_from_prefix(tgt, prefix)
            tgt_signs = factor_signs(tgt)
            cands_scored = []
            for src in cands:
                sid = str(src.get("id"))
                feats = cold_features(q, src)
                held = rho(src, suffix, "suffix", tgt_id)
                pre = rho(src, prefix, "prefix", tgt_id)
                lr = sign_agree(tgt_signs, factor_signs(src), "learning_rate")
                gen = 1.0 if (held >= GENUINE_RHO and (lr is None or lr == 1)) else 0.0
                x3 = [float(feats[k]) for k in LEARN_KEYS]
                rec = {
                    "tgt_id": tgt_id,
                    "tgt_ds": tgt_ds,
                    "src_id": sid,
                    "src_ds": _ds(src),
                    "x3": x3,
                    "feats": feats,
                    "held": held,
                    "prefix": pre,
                    "genuine": gen,
                    "lr": float("nan") if lr is None else float(lr),
                }
                train_rows[(task, algo)].append(rec)
                cands_scored.append(rec)
            query_payloads.append(
                {
                    "tgt_id": tgt_id,
                    "dataset": tgt_ds,
                    "algorithm": algo,
                    "task_type": task,
                    "cands": cands_scored,
                }
            )

    print(f"queries={len(query_payloads)} pair-rows={sum(len(v) for v in train_rows.values())}")

    # Nested: for each query, fit on other datasets in the same (task, algo) group.
    w_nested_by_query: Dict[str, Dict[str, float]] = {}
    nested_w_examples = []
    for q in query_payloads:
        key = (q["task_type"], q["algorithm"])
        X, y = [], []
        for rec in train_rows[key]:
            if rec["tgt_ds"] == q["dataset"] or rec["src_ds"] == q["dataset"]:
                continue
            X.append(rec["x3"])
            y.append(rec["held"])
        w3 = fit_three(X, y)
        w_nested_by_query[q["tgt_id"]] = w3
        nested_w_examples.append({"dataset": q["dataset"], "algorithm": q["algorithm"], "n_train": len(X), **w3})

    # Online walk-forward: datasets sorted by name; fit on previous datasets only, then SGD.
    by_group_q: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for q in query_payloads:
        by_group_q[(q["task_type"], q["algorithm"])].append(q)

    w_online_by_query: Dict[str, Dict[str, float]] = {}
    for (task, algo), qs in by_group_q.items():
        qs_sorted = sorted(qs, key=lambda r: r["dataset"])
        w3 = {"importance_cos": 1.0 / 3, "profile_cos": 1.0 / 3, "role_jaccard": 1.0 / 3}
        seen_ds = []
        for q in qs_sorted:
            # Refit ridge on all previous datasets in this group (stable), then rank, then include this ds.
            X, y = [], []
            seen = set(seen_ds)
            for rec in train_rows[(task, algo)]:
                if rec["tgt_ds"] in seen and rec["src_ds"] in seen:
                    X.append(rec["x3"])
                    y.append(rec["held"])
            if len(X) >= 8:
                w3 = fit_three(X, y)
            w_online_by_query[q["tgt_id"]] = dict(w3)
            # After grading, fold this dataset in (update happens after rank — no leak).
            seen_ds.append(q["dataset"])
            for rec in q["cands"]:
                w3 = sgd_update(w3, rec["x3"], rec["held"], eta=0.02)

    RANKERS = (
        "random_same_algo",
        "cold",
        "frozen_importance",
        "frozen_cascade",
        "nested_importance",
        "nested_cascade",
        "online_importance",
        "online_cascade",
        "oracle",
    )

    rows = []
    for q in query_payloads:
        cands = q["cands"]
        held = {c["src_id"]: c["held"] for c in cands}
        genuine = {c["src_id"]: c["genuine"] for c in cands}
        lr = {c["src_id"]: c["lr"] for c in cands}
        prefix = {c["src_id"]: c["prefix"] for c in cands}

        frozen_sc = {
            c["src_id"]: combine_score(c["feats"], frozen) for c in cands
        }
        cold_sc = {
            c["src_id"]: combine_score(c["feats"], COLD_WEIGHTS) for c in cands
        }
        nw = pad_weights(w_nested_by_query[q["tgt_id"]])
        ow = pad_weights(w_online_by_query[q["tgt_id"]])
        nested_sc = {c["src_id"]: combine_score(c["feats"], nw) for c in cands}
        online_sc = {c["src_id"]: combine_score(c["feats"], ow) for c in cands}

        rng = rng_for(q["tgt_id"])
        rand_ids = [c["src_id"] for c in cands]
        rng.shuffle(rand_ids)

        orders = {
            "random_same_algo": rand_ids,
            "cold": topk_ids(list(cold_sc.items()), 5),
            "frozen_importance": topk_ids(list(frozen_sc.items()), 5),
            "frozen_cascade": cascade_order(frozen_sc, prefix),
            "nested_importance": topk_ids(list(nested_sc.items()), 5),
            "nested_cascade": cascade_order(nested_sc, prefix),
            "online_importance": topk_ids(list(online_sc.items()), 5),
            "online_cascade": cascade_order(online_sc, prefix),
            "oracle": topk_ids(list(held.items()), 5),
        }
        row = {
            "target_id": q["tgt_id"],
            "dataset": q["dataset"],
            "algorithm": q["algorithm"],
            "task_type": q["task_type"],
            "nested_w_imp": w_nested_by_query[q["tgt_id"]]["importance_cos"],
            "nested_w_prof": w_nested_by_query[q["tgt_id"]]["profile_cos"],
            "nested_w_role": w_nested_by_query[q["tgt_id"]]["role_jaccard"],
            "online_w_imp": w_online_by_query[q["tgt_id"]]["importance_cos"],
            "online_w_prof": w_online_by_query[q["tgt_id"]]["profile_cos"],
            "online_w_role": w_online_by_query[q["tgt_id"]]["role_jaccard"],
        }
        for name, ids in orders.items():
            g = grade_top(ids, held, genuine, lr)
            row[f"{name}_top1"] = g["top1"]
            row[f"{name}_rho1"] = g["rho1"]
            row[f"{name}_rho3"] = g["rho3"]
            row[f"{name}_genuine"] = g["genuine"]
            row[f"{name}_lr_agree"] = g["lr_agree"]
        rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    qpath = args.out_dir / "lodo_adaptive_queries.csv"
    with qpath.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== Frozen vs nested vs online (within-algo LODO) ===")
    print(f"{'ranker':<22} {'ρ@1':>8} {'ρ@3':>8} {'ρ≥0.3':>8} {'genuine':>8} {'LR agr':>8}")
    summary = []
    for name in RANKERS:
        s = summarize(rows, name)
        summary.append({"ranker": name, **s})
        print(
            f"{name:<22} {s['mean_rho_top1']:8.3f} {s['mean_rho_top3']:8.3f} "
            f"{s['frac_rho_ge_0.3']:8.3f} {s['genuine']:8.3f} {s['lr_agree']:8.3f}"
        )

    spath = args.out_dir / "lodo_adaptive_summary.csv"
    with spath.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # Mean nested weights across queries
    mean_w = {k: sum(w_nested_by_query[q["tgt_id"]][k] for q in query_payloads) / len(query_payloads) for k in LEARN_KEYS}
    wpath = args.out_dir / "lodo_adaptive_weights.json"
    wpath.write_text(
        json.dumps(
            {
                "note": "Nested weights are per held-out dataset; mean shown. Not written to the production bank.",
                "frozen_deployed": frozen,
                "mean_nested_three": mean_w,
                "protocol": "For each target dataset D, fit 3-d nonnegative ridge on pairs among other datasets in the same (task, algorithm). y=clip(held-out Spearman,0,1). Rank D with those weights. Online: walk datasets in name order, fit on previous datasets only.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nmean nested weights (importance, profile, role): {mean_w}")
    print(f"wrote {qpath}")
    print(f"wrote {spath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
