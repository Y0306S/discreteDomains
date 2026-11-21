#!/usr/bin/env python3
"""LODO funnel with GP surrogate as gate 1, then fANOVA, then take 3.

Gate 1: same algorithm, other dataset; cheap rank → top 12; fit a GP on the
8-trial prefix; Spearman vs cached source GP landscapes; sim >= 0 passes.
Gate 2: fANOVA usable card, only among gate-1 passers.
Prompt: 3 highest mix scores among dual passers. No fill from gate-1 failures.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from dataset_split import is_tune, load_split
from mindware.components.experience.retrieve import (
    COLD_WEIGHTS,
    ExperienceBank,
    cold_features,
    combine_score,
    query_importance_from_trials,
)
from mindware.components.experience.surrogate_landscape import (
    LandscapeCache,
    get_algo_cs,
    predict_landscape,
    sample_query_configs,
    spearman_sim,
)

PREFIX_N = 8
SIMILAR_POOL = 12
K_PROMPT = 3
SIM_FLOOR = 0.0
OUT = (
    ROOT
    / "benchmark_data"
    / "real_hpo_v2_mltest20_algo40"
    / "analysis"
    / "experience_bank_v3"
)


def _ds(exp: Dict[str, Any]) -> str:
    return str(((exp.get("problem") or {}).get("dataset_id") or "")).lower()


def _algo(exp: Dict[str, Any]) -> str:
    return str(((exp.get("problem") or {}).get("algorithm_id") or "")).lower()


def _task(exp: Dict[str, Any]) -> str:
    return str(((exp.get("problem") or {}).get("task_type") or "")).lower()


def fanova_ok(exp: Dict[str, Any]) -> bool:
    d = exp.get("delivery") or {}
    if d.get("usable_for_prompt") is False:
        return False
    return bool(str(d.get("prompt_block") or "").strip())


def query_from_prefix(exp: Dict[str, Any], prefix: List[Dict[str, Any]]) -> Dict[str, Any]:
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


def load_cache(task: str) -> LandscapeCache:
    prefix = "cls" if "class" in task else "rgs"
    meta = ROOT / "history_data" / "meta_resource"
    holdout = meta / f"surrogate_landscape_holdout_{prefix}.pkl"
    full = meta / f"surrogate_landscape_{prefix}.pkl"
    path = holdout if holdout.is_file() else full
    if not path.is_file():
        raise FileNotFoundError(f"missing GP landscape cache: {holdout} and {full}")
    return LandscapeCache(path)


def bucket_hist(c: Counter, n: int, lo: int, hi: int) -> dict:
    return {
        str(k): {"n": int(c[k]), "pct": round(100.0 * c[k] / max(n, 1), 1)}
        for k in range(lo, hi + 1)
    }


def main() -> int:
    bank = ExperienceBank.load(ROOT / "history_data" / "experience_bank_v3.jsonl")
    split = load_split()
    members_all = [
        e
        for e in bank.experiences
        if len((e.get("retrieval") or {}).get("trials") or []) >= PREFIX_N + 3
    ]
    by_group: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for exp in members_all:
        if not is_tune(_ds(exp), split):
            continue
        by_group[(_task(exp), _algo(exp))].append(exp)

    caches = {"classification": load_cache("classification"), "regression": load_cache("regression")}
    cs_qcfg: Dict[tuple, Any] = {}

    hist_g1 = Counter()
    hist_both = Counter()
    hist_prompt = Counter()
    n_pool = 0
    n_g1_fail = 0
    n_g2_fail = 0
    n_both = 0
    n_gp_fail = 0
    rows: List[Dict[str, Any]] = []

    for (task, algo), members in sorted(by_group.items()):
        if len(members) < 3:
            continue
        print(f"{task}/{algo} n={len(members)}", flush=True)
        cache = caches["classification" if "class" in task else "regression"]
        key = (task, algo)
        if key not in cs_qcfg:
            cs = get_algo_cs(algo, task)
            n_query = int((cache.payload.get("hyperparams") or {}).get("n_query_configs") or 25)
            qcfgs = sample_query_configs(cs, n_query) if cs is not None else None
            cs_qcfg[key] = (cs, qcfgs)
        cs, qcfgs = cs_qcfg[key]
        for tgt in members:
            trials = (tgt.get("retrieval") or {}).get("trials") or []
            prefix = trials[:PREFIX_N]
            cands = [s for s in members if _ds(s) != _ds(tgt)]
            if len(cands) < 3:
                continue
            q = query_from_prefix(tgt, prefix)
            qvec: Optional[Any] = None
            if cs is not None and qcfgs is not None:
                qvec = predict_landscape(prefix, cs, algo, qcfgs, min_trials=PREFIX_N)
            if qvec is None:
                n_gp_fail += 1
                hist_g1[0] += 1
                hist_both[0] += 1
                hist_prompt[0] += 1
                rows.append(
                    {
                        "target_id": str(tgt.get("id")),
                        "dataset": _ds(tgt),
                        "algorithm": algo,
                        "task_type": task,
                        "n_cands": len(cands),
                        "n_similar_pool": 0,
                        "n_gate1_similar": 0,
                        "n_gate1_fail": 0,
                        "n_gate2_fanova_fail": 0,
                        "n_pass_both": 0,
                        "n_prompt_cards": 0,
                        "gp_ok": 0,
                        "prompt_datasets": "",
                        "prompt_ids": "",
                        "prompt_sims": "",
                    }
                )
                continue
            scored = []
            for src in cands:
                feats = cold_features(q, src)
                scored.append(
                    {
                        "exp": src,
                        "id": str(src.get("id")),
                        "dataset": _ds(src),
                        "cheap": combine_score(feats, COLD_WEIGHTS),
                    }
                )
            scored.sort(key=lambda r: r["cheap"], reverse=True)
            pool = scored[:SIMILAR_POOL]
            g1 = []
            n_g1_fail_q = 0
            for rec in pool:
                n_pool += 1
                svec = cache.pred(algo, rec["dataset"])
                sim = spearman_sim(qvec, svec) if svec is not None else None
                rec["surrogate_sim"] = sim
                if sim is None or sim < SIM_FLOOR:
                    n_g1_fail += 1
                    n_g1_fail_q += 1
                    continue
                rec["mix"] = 0.6 * rec["cheap"] + 0.4 * max(0.0, sim)
                g1.append(rec)
            both = []
            n_g2_fail_q = 0
            for rec in g1:
                if not fanova_ok(rec["exp"]):
                    n_g2_fail += 1
                    n_g2_fail_q += 1
                    continue
                n_both += 1
                both.append(rec)
            both.sort(key=lambda r: r["mix"], reverse=True)
            prompt = both[:K_PROMPT]
            hist_g1[len(g1)] += 1
            hist_both[len(both)] += 1
            hist_prompt[len(prompt)] += 1
            rows.append(
                {
                    "target_id": str(tgt.get("id")),
                    "dataset": _ds(tgt),
                    "algorithm": algo,
                    "task_type": task,
                    "n_cands": len(cands),
                    "n_similar_pool": len(pool),
                    "n_gate1_similar": len(g1),
                    "n_gate1_fail": n_g1_fail_q,
                    "n_gate2_fanova_fail": n_g2_fail_q,
                    "n_pass_both": len(both),
                    "n_prompt_cards": len(prompt),
                    "gp_ok": 1,
                    "prompt_datasets": "|".join(r["dataset"] for r in prompt),
                    "prompt_ids": "|".join(r["id"] for r in prompt),
                    "prompt_sims": "|".join(f"{r['surrogate_sim']:.3f}" for r in prompt),
                }
            )

    n = len(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    qpath = OUT / "lodo_similar_then_fanova_keep.csv"
    with qpath.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "n_queries": n,
        "gate1": "GP surrogate landscape Spearman on 8-trial prefix vs cached source GPs",
        "similar_pool": SIMILAR_POOL,
        "prompt_k": K_PROMPT,
        "n_query_gp_fail": n_gp_fail,
        "n_similar_pool_slots": n_pool,
        "pct_fail_gate1_of_pool": round(100.0 * n_g1_fail / max(n_pool, 1), 1),
        "pct_fail_gate2_of_gate1": round(100.0 * n_g2_fail / max(n_pool - n_g1_fail, 1), 1),
        "mean_gate1_per_query": (n_pool - n_g1_fail) / max(n, 1),
        "mean_pass_both_per_query": n_both / max(n, 1),
        "hist_gate1_similar": bucket_hist(hist_g1, n, 0, SIMILAR_POOL),
        "hist_pass_both": bucket_hist(hist_both, n, 0, SIMILAR_POOL),
        "hist_prompt_after_take3": bucket_hist(hist_prompt, n, 0, K_PROMPT),
        "mean_prompt_cards": sum(i * hist_prompt[i] for i in range(K_PROMPT + 1)) / max(n, 1),
        "pct_empty_prompt": round(100.0 * hist_prompt[0] / max(n, 1), 1),
        "pct_full3": round(100.0 * hist_prompt[K_PROMPT] / max(n, 1), 1),
        "by_algorithm": {},
    }
    by_algo = defaultdict(list)
    for r in rows:
        by_algo[r["algorithm"]].append(
            (int(r["n_gate1_similar"]), int(r["n_pass_both"]), int(r["n_prompt_cards"]))
        )
    for algo, triples in sorted(by_algo.items()):
        g1s = [t[0] for t in triples]
        boths = [t[1] for t in triples]
        ks = [t[2] for t in triples]
        c = Counter(ks)
        summary["by_algorithm"][algo] = {
            "n": len(ks),
            "mean_gate1": sum(g1s) / len(g1s),
            "mean_pass_both": sum(boths) / len(boths),
            "mean_prompt_cards": sum(ks) / len(ks),
            "pct_empty": round(100.0 * c[0] / len(ks), 1),
            "pct_full3": round(100.0 * c[K_PROMPT] / len(ks), 1),
            "prompt_hist": {str(i): c[i] for i in range(K_PROMPT + 1)},
        }
    (OUT / "lodo_similar_then_fanova_keep.json").write_text(json.dumps(summary, indent=2) + "\n")
    slim = {k: v for k, v in summary.items() if k not in ("hist_gate1_similar", "hist_pass_both")}
    print(json.dumps(slim, indent=2))
    print("hist_gate1_similar:", json.dumps(summary["hist_gate1_similar"]))
    print("hist_pass_both:", json.dumps(summary["hist_pass_both"]))
    print(f"wrote {qpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
