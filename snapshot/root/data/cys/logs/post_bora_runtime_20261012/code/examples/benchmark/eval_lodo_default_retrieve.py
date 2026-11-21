#!/usr/bin/env python3
"""LODO: default retrieve (usable → GP/Spearman backup → floor → top 3 → clash)."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from dataset_split import is_tune, load_split
from mindware.components.experience.retrieve import (
    ExperienceBank,
    format_retrieved_block,
    load_retrieve_knobs,
    query_importance_from_trials,
    retrieve_experiences,
)

PREFIX_N = 8
K_PROMPT = 3
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


def n_clash_cards(block: str) -> int:
    if not str(block or "").strip():
        return 0
    return str(block).count("Similar task:")


def bucket_hist(c: Counter, n: int, lo: int, hi: int) -> dict:
    return {
        str(k): {"n": int(c[k]), "pct": round(100.0 * c[k] / max(n, 1), 1)}
        for k in range(lo, hi + 1)
    }


def main() -> int:
    knobs = load_retrieve_knobs()
    bank = ExperienceBank.load(ROOT / "history_data" / "experience_bank_v3.jsonl")
    split = load_split()
    members_all = [
        e
        for e in bank.experiences
        if len((e.get("retrieval") or {}).get("trials") or []) >= PREFIX_N
    ]
    by_group: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for exp in members_all:
        if not is_tune(_ds(exp), split):
            continue
        by_group[(_task(exp), _algo(exp))].append(exp)

    hist_hits = Counter()
    hist_prompt = Counter()
    rows: List[Dict[str, Any]] = []

    for (task, algo), members in sorted(by_group.items()):
        if len(members) < 3:
            continue
        print(f"{task}/{algo} n={len(members)}", flush=True)
        sub = ExperienceBank(experiences=members, weights=dict(bank.weights))
        for tgt in members:
            trials = (tgt.get("retrieval") or {}).get("trials") or []
            prefix = trials[:PREFIX_N]
            if len([s for s in members if _ds(s) != _ds(tgt)]) < 3:
                continue
            q = query_from_prefix(tgt, prefix)
            hits = retrieve_experiences(
                sub,
                q,
                k=K_PROMPT,
                rerank_after=24,
                target_trials=prefix,
                ranker="surrogate",
                require_usable=True,
            )
            block = format_retrieved_block(hits)
            n_hits = len(hits)
            n_prompt = n_clash_cards(block)
            if n_hits and not n_prompt:
                n_prompt = n_hits
            hist_hits[n_hits] += 1
            hist_prompt[n_prompt] += 1
            rows.append(
                {
                    "target_id": str(tgt.get("id")),
                    "dataset": _ds(tgt),
                    "algorithm": algo,
                    "task_type": task,
                    "n_hits_after_floor": n_hits,
                    "n_prompt_cards": n_prompt,
                    "sims": "|".join(
                        f"{(h.spearman if h.spearman is not None else h.score):.3f}" for h in hits
                    ),
                    "datasets": "|".join(_ds(h.experience) for h in hits),
                }
            )

    n = len(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    qpath = OUT / "lodo_default_floor_keep.csv"
    with qpath.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "n_queries": n,
        "knobs": knobs,
        "hist_after_floor_topk": bucket_hist(hist_hits, n, 0, K_PROMPT),
        "hist_after_clash": bucket_hist(hist_prompt, n, 0, K_PROMPT),
        "mean_hits": sum(i * hist_hits[i] for i in range(K_PROMPT + 1)) / max(n, 1),
        "mean_prompt_cards": sum(i * hist_prompt[i] for i in range(K_PROMPT + 1)) / max(n, 1),
        "pct_empty_after_floor": round(100.0 * hist_hits[0] / max(n, 1), 1),
        "pct_empty_after_clash": round(100.0 * hist_prompt[0] / max(n, 1), 1),
        "pct_full3_after_floor": round(100.0 * hist_hits[K_PROMPT] / max(n, 1), 1),
        "pct_full3_after_clash": round(100.0 * hist_prompt[K_PROMPT] / max(n, 1), 1),
        "by_algorithm": {},
    }
    by_algo = defaultdict(list)
    for r in rows:
        by_algo[r["algorithm"]].append((int(r["n_hits_after_floor"]), int(r["n_prompt_cards"])))
    for algo, pairs in sorted(by_algo.items()):
        hs = [p[0] for p in pairs]
        ps = [p[1] for p in pairs]
        ch = Counter(hs)
        cp = Counter(ps)
        summary["by_algorithm"][algo] = {
            "n": len(hs),
            "mean_hits": sum(hs) / len(hs),
            "mean_prompt": sum(ps) / len(ps),
            "pct_empty_hits": round(100.0 * ch[0] / len(hs), 1),
            "pct_full3_hits": round(100.0 * ch[K_PROMPT] / len(hs), 1),
            "hits_hist": {str(i): ch[i] for i in range(K_PROMPT + 1)},
            "prompt_hist": {str(i): cp[i] for i in range(K_PROMPT + 1)},
        }
    (OUT / "lodo_default_floor_keep.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {qpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
