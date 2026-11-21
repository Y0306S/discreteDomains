#!/usr/bin/env python3
"""Offline gate hyperparameter sweep on LODO TUNE (replay stored fANOVA).

Layer A = per-HP keep, Layer B = per-record usable, Layer C = retrieval k.
fANOVA is not re-run. Precompute candidate cold/importance/Spearman once.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from dataset_split import is_tune, load_split
from mindware.components.experience.bank_gates import (
    DEFAULT_GATE_CONFIG,
    gate_view,
    merge_gate_config,
)
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
RERANK_N = 24
GRADE_START = 24
GRADE_END = 40
CASCADE_POOL = 12
GENUINE_RHO = 0.3
ANALYSIS = (
    ROOT
    / "benchmark_data"
    / "real_hpo_v2_mltest20_algo40"
    / "analysis"
    / "experience_gate_sweep"
)


def _ds(exp: Dict[str, Any]) -> str:
    return str(((exp.get("problem") or {}).get("dataset_id") or "")).lower()


def _algo(exp: Dict[str, Any]) -> str:
    return str(((exp.get("problem") or {}).get("algorithm_id") or "")).lower()


def _task(exp: Dict[str, Any]) -> str:
    return str(((exp.get("problem") or {}).get("task_type") or "")).lower()


def _mean(xs: Sequence[Optional[float]]) -> float:
    vals = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


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


def iter_gate_configs() -> List[Tuple[str, Dict[str, Any]]]:
    base = dict(DEFAULT_GATE_CONFIG)
    base.update({"k_max": 3, "require_usable": True, "spearman_floor": 0.0, "dedup": True})
    out: List[Tuple[str, Dict[str, Any]]] = [("v3_strict_default", dict(base))]

    oat = {
        "keep_floor": [0.01, 0.05, 0.10],
        "keep_cum": [0.40, 0.50, 0.70],
        "keep_topk": [2, 3, 6],
        "keep_min": [1, 3],
        "main_effect_ratio": [0.25, 0.50, 0.65],
        "require_region": [False],
        "max_1d_record": [0.05, 0.08, 0.15],
        "min_keep_hps": [1, 3],
        "interaction_only_reject": [False],
        "k_max": [1, 2],
        "require_usable": [False],
        "spearman_floor": [0.15, 0.30],
        "dedup": [False],
    }
    for key, values in oat.items():
        for val in values:
            cfg = dict(base)
            cfg[key] = val
            out.append((f"oat_{key}={val}", cfg))

    loose = dict(base)
    loose.update(
        {
            "keep_floor": 0.015,
            "keep_cum": 0.30,
            "keep_min": 1,
            "main_effect_ratio": 0.175,
            "max_1d_record": 0.05,
            "min_keep_hps": 1,
            "interaction_only_reject": False,
            "k_max": 3,
        }
    )
    out.append(("v3_loose", loose))

    for floor, ratio, k_max, min_keep in itertools.product(
        (0.01, 0.03, 0.05),
        (0.25, 0.35, 0.50),
        (1, 2, 3),
        (1, 2),
    ):
        cfg = dict(base)
        cfg.update(
            {
                "keep_floor": floor,
                "main_effect_ratio": ratio,
                "k_max": k_max,
                "min_keep_hps": min_keep,
                "keep_min": min_keep,
            }
        )
        out.append((f"grid_f{floor}_r{ratio}_k{k_max}_m{min_keep}", cfg))

    # de-dupe by frozen cfg items
    seen = set()
    unique: List[Tuple[str, Dict[str, Any]]] = []
    for name, cfg in out:
        key = tuple(sorted(cfg.items(), key=lambda kv: kv[0]))
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, cfg))
    return unique


def precompute_lodo(
    bank: ExperienceBank,
    *,
    split: dict,
    warm_w: Dict[str, float],
    prefix_n: int = PREFIX_N,
    rerank_n: int = RERANK_N,
    grade_start: int = GRADE_START,
    grade_end: int = GRADE_END,
    tune_only: bool = True,
) -> List[Dict[str, Any]]:
    usable = [
        e
        for e in bank.experiences
        if len((e.get("retrieval") or {}).get("trials") or []) >= grade_end
    ]
    by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for exp in usable:
        if tune_only and not is_tune(_ds(exp), split):
            continue
        by_group[(_task(exp), _algo(exp))].append(exp)

    queries: List[Dict[str, Any]] = []
    for (task, algo), members in sorted(by_group.items()):
        if len(members) < 3:
            continue
        for tgt in members:
            trials = (tgt.get("retrieval") or {}).get("trials") or []
            prefix = trials[:prefix_n]
            rerank_slice = trials[:rerank_n]
            grade = trials[grade_start:grade_end]
            if len(prefix) < prefix_n or len(grade) < 8:
                continue
            cands = [s for s in members if _ds(s) != _ds(tgt)]
            if len(cands) < 3:
                continue
            q = query_from_prefix(tgt, prefix)
            scored = []
            for src in cands:
                feats = cold_features(q, src)
                scored.append(
                    {
                        "id": str(src.get("id")),
                        "cold": combine_score(feats, COLD_WEIGHTS),
                        "importance": combine_score(feats, warm_w),
                        "prefix_rho": float(spearman_transfer(src, prefix)),
                        "rerank_rho": float(spearman_transfer(src, rerank_slice)),
                        "grade_rho": float(spearman_transfer(src, grade)),
                    }
                )
            queries.append(
                {
                    "target_id": str(tgt.get("id")),
                    "dataset": _ds(tgt),
                    "algorithm": algo,
                    "task_type": task,
                    "n_cands": len(cands),
                    "scored": scored,
                }
            )
    return queries


def cascade_order(scored: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = sorted(scored, key=lambda r: r["importance"], reverse=True)
    pool = ranked[: max(CASCADE_POOL, 3)]
    mixed = []
    for row in pool:
        rho = float(row["rerank_rho"])
        mixed.append((row, 0.6 * row["importance"] + 0.4 * max(0.0, rho)))
    mixed.sort(key=lambda kv: kv[1], reverse=True)
    return [row for row, _ in mixed]


def retrieve_ids(
    scored: List[Dict[str, Any]],
    views: Dict[str, Dict[str, Any]],
    cfg: Dict[str, Any],
    *,
    force_k: Optional[int] = None,
    ignore_usable: bool = False,
) -> List[str]:
    k_max = int(force_k if force_k is not None else cfg.get("k_max", 3))
    require_usable = bool(cfg.get("require_usable", True)) and not ignore_usable
    floor = float(cfg.get("spearman_floor") or 0.0)
    ordered = cascade_order(scored)
    kept: List[str] = []
    seen_body: set = set()
    for row in ordered:
        sid = row["id"]
        view = views.get(sid) or {}
        if require_usable and not view.get("usable_for_prompt"):
            continue
        rho = float(row["rerank_rho"])
        if rho < 0.0:
            continue
        if rho < floor:
            continue
        body = str(view.get("prompt_block") or "")
        hp_lines = "\n".join(ln for ln in body.splitlines() if ln.strip().startswith("- "))
        key = hp_lines or body
        if cfg.get("dedup", True) and key and key in seen_body:
            continue
        if key:
            seen_body.add(key)
        kept.append(sid)
        if len(kept) >= k_max:
            break
    return kept


def eval_config(
    queries: List[Dict[str, Any]],
    by_id: Dict[str, Dict[str, Any]],
    cfg: Dict[str, Any],
    *,
    force_k: Optional[int] = None,
    ignore_usable: bool = False,
    views: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if views is None:
        views = {eid: gate_view(exp, cfg) for eid, exp in by_id.items()}
    rhos_top1: List[Optional[float]] = []
    rhos_all: List[float] = []
    n_empty = 0
    n_ret: List[int] = []
    hist = Counter()
    n_hps: List[int] = []
    for q in queries:
        ids = retrieve_ids(q["scored"], views, cfg, force_k=force_k, ignore_usable=ignore_usable)
        n_ret.append(len(ids))
        hist[len(ids)] += 1
        if not ids:
            n_empty += 1
            rhos_top1.append(None)
            continue
        grade = {row["id"]: row["grade_rho"] for row in q["scored"]}
        rhos_top1.append(grade.get(ids[0]))
        for sid in ids:
            rhos_all.append(float(grade.get(sid) or 0.0))
            n_hps.append(int((views.get(sid) or {}).get("n_keep") or 0))
    usable_n = sum(1 for v in views.values() if v.get("usable_for_prompt"))
    return {
        "n_queries": len(queries),
        "mean_rho_top1": _mean(rhos_top1),
        "mean_rho_at_k": _mean(rhos_all),
        "frac_rho_ge_0.3": (sum(1 for x in rhos_all if x >= GENUINE_RHO) / len(rhos_all)) if rhos_all else 0.0,
        "frac_rho_lt_0": (sum(1 for x in rhos_all if x < 0.0) / len(rhos_all)) if rhos_all else 0.0,
        "pct_empty_retrieve": 100.0 * n_empty / max(len(queries), 1),
        "pct_bank_usable": 100.0 * usable_n / max(len(views), 1),
        "mean_hps_per_card": _mean(n_hps),
        "mean_neighbors_returned": _mean(n_ret),
        "hist_0": hist.get(0, 0),
        "hist_1": hist.get(1, 0),
        "hist_2": hist.get(2, 0),
        "hist_3": hist.get(3, 0),
    }


def pick_winner(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    feas = [
        r
        for r in rows
        if not str(r.get("name") or "").startswith("v2_")
        and float(r.get("pct_empty_retrieve") or 999) <= 30.0
        and float(r.get("mean_hps_per_card") or 0) <= 4.0
        and float(r.get("pct_bank_usable") or 0) >= 40.0
    ]
    if not feas:
        feas = [
            r
            for r in rows
            if not str(r.get("name") or "").startswith("v2_")
            and float(r.get("pct_empty_retrieve") or 999) <= 40.0
        ]
    if not feas:
        feas = [r for r in rows if not str(r.get("name") or "").startswith("v2_")]
    if not feas:
        return None

    def key(r: Dict[str, Any]):
        rho = r.get("mean_rho_top1")
        rho_f = float(rho) if rho is not None and not (isinstance(rho, float) and math.isnan(rho)) else -9.0
        return (
            -rho_f,
            -float(r.get("frac_rho_ge_0.3") or 0.0),
            float(r.get("frac_rho_lt_0") or 0.0),
        )

    feas.sort(key=key)
    return feas[0]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=ROOT / "history_data" / "experience_bank_v3.jsonl")
    p.add_argument("--v2-bank", type=Path, default=ROOT / "history_data" / "experience_bank_v2.jsonl")
    p.add_argument("--out-dir", type=Path, default=ANALYSIS)
    p.add_argument("--max-queries", type=int, default=0, help="0 = all TUNE queries")
    args = p.parse_args(argv)

    if not args.bank.is_file():
        raise SystemExit(f"missing {args.bank}")

    split = load_split()
    bank = ExperienceBank.load(args.bank)
    warm_w = dict(bank.weights) if bank.weights else dict(DEFAULT_WEIGHTS)
    print(f"v3 bank={len(bank)} weights={warm_w}")
    queries = precompute_lodo(bank, split=split, warm_w=warm_w)
    if args.max_queries:
        queries = queries[: args.max_queries]
    print(f"LODO TUNE queries={len(queries)}")
    by_id = {str(e.get("id")): e for e in bank.experiences}

    configs = iter_gate_configs()
    print(f"sweep configs={len(configs)}")
    rows: List[Dict[str, Any]] = []
    for i, (name, cfg) in enumerate(configs, 1):
        metrics = eval_config(queries, by_id, cfg)
        row = {"name": name, **{k: cfg[k] for k in sorted(cfg)}, **metrics}
        rows.append(row)
        if i % 10 == 0 or i == len(configs):
            print(
                f"  {i}/{len(configs)} {name} rho1={metrics['mean_rho_top1']:.3f} "
                f"empty={metrics['pct_empty_retrieve']:.1f}% usable={metrics['pct_bank_usable']:.1f}%"
            )

    if args.v2_bank.is_file():
        v2 = ExperienceBank.load(args.v2_bank)
        v2w = dict(v2.weights) if v2.weights else dict(DEFAULT_WEIGHTS)
        v2_queries = precompute_lodo(v2, split=split, warm_w=v2w)
        v2_by = {str(e.get("id")): e for e in v2.experiences}
        v2_cfg = dict(DEFAULT_GATE_CONFIG)
        v2_cfg.update({"k_max": 3, "require_usable": False, "spearman_floor": 0.0, "dedup": False})
        v2_views = {
            eid: {
                "keep_hps": [],
                "usable_for_prompt": True,
                "prompt_block": str((exp.get("delivery") or {}).get("prompt_block") or eid),
                "n_keep": len((exp.get("knowledge") or {}).get("factors") or []),
                "max_1d": 0.0,
                "max_total": 0.0,
            }
            for eid, exp in v2_by.items()
        }
        v2_metrics = eval_config(
            v2_queries,
            v2_by,
            v2_cfg,
            force_k=3,
            ignore_usable=True,
            views=v2_views,
        )
        rows.append({"name": "v2_always_k3", **v2_cfg, **v2_metrics})
        print(f"v2_always_k3 rho1={v2_metrics['mean_rho_top1']:.3f} empty={v2_metrics['pct_empty_retrieve']:.1f}%")

    v2_rho = next((r["mean_rho_top1"] for r in rows if r["name"] == "v2_always_k3"), float("nan"))
    for r in rows:
        try:
            r["rho_vs_v2_baseline"] = float(r["mean_rho_top1"]) - float(v2_rho)
        except (TypeError, ValueError):
            r["rho_vs_v2_baseline"] = float("nan")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "gate_sweep_results.csv"
    fields = sorted({k for r in rows for k in r.keys()})
    # put name first
    fields = ["name"] + [f for f in fields if f != "name"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    winner = pick_winner(rows)
    payload = {
        "winner": winner,
        "selection": {
            "feasibility": "pct_empty_retrieve<=30, mean_hps_per_card<=4, pct_bank_usable>=40",
            "quality": "max mean_rho_top1; tie-break frac_rho_ge_0.3 then lower frac_rho_lt_0",
        },
        "n_configs": len(rows),
        "n_queries": len(queries),
    }
    (args.out_dir / "gate_sweep_winner.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    if winner:
        rec_cfg = {k: winner[k] for k in DEFAULT_GATE_CONFIG}
        rec_cfg.update(
            {
                "k_max": int(winner.get("k_max") or 3),
                "require_usable": bool(winner.get("require_usable", True)),
                "spearman_floor": float(winner.get("spearman_floor") or 0.0),
                "dedup": bool(winner.get("dedup", True)),
            }
        )
        rec_path = args.out_dir / "gate_config.json"
        rec_path.write_text(json.dumps(rec_cfg, indent=2) + "\n")
        pkg_path = ROOT / "mindware" / "components" / "experience" / "gate_config.json"
        pkg_path.write_text(json.dumps(rec_cfg, indent=2) + "\n")
        print(f"winner={winner['name']} rho1={winner['mean_rho_top1']:.3f} empty={winner['pct_empty_retrieve']:.1f}%")
        print(f"wrote {rec_path} and {pkg_path}")
    print(f"wrote {csv_path}")
    _plot_pareto(rows, args.out_dir / "gate_sweep_pareto.png")
    return 0


def _plot_pareto(rows: List[Dict[str, Any]], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    xs, ys, names = [], [], []
    for r in rows:
        empty = r.get("pct_empty_retrieve")
        rho = r.get("mean_rho_top1")
        if empty is None or rho is None:
            continue
        try:
            if isinstance(rho, float) and math.isnan(rho):
                continue
            xs.append(float(empty))
            ys.append(float(rho))
            names.append(str(r.get("name") or ""))
        except (TypeError, ValueError):
            continue
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d62728" if n.startswith("v2_") else "#1f77b4" for n in names]
    ax.scatter(xs, ys, c=colors, s=28, alpha=0.75)
    for x, y, n in zip(xs, ys, names):
        if n in {"v3_strict_default", "v3_loose", "v2_always_k3"} or n.startswith("grid_") and "k1" in n:
            ax.annotate(n, (x, y), fontsize=7, alpha=0.8)
    ax.axvline(30.0, color="#888", ls="--", lw=0.8, label="empty ≤ 30%")
    ax.set_xlabel("pct_empty_retrieve")
    ax.set_ylabel("mean_rho_top1")
    ax.set_title("Experience-bank gate sweep (LODO TUNE)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
