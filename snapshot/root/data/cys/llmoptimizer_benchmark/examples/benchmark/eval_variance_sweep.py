#!/usr/bin/env python3
"""Phase 2: pre-registered variance knobs. Select on TUNE, measure LOCK once.

Grid (no additions):
  rerank_after ∈ {8, 16, 24}
  shrink k ∈ {0, 5, 10} with coverage-weighted rho_adj
  pool ∈ {cold, importance}
  aggregate ∈ {top1, top3}
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from dataset_split import is_lock, is_tune, load_split
from mindware.components.experience.retrieve import (
    COLD_WEIGHTS,
    DEFAULT_WEIGHTS,
    ExperienceBank,
    cold_features,
    combine_score,
    nearest_source_scores_ex,
    query_importance_from_trials,
    spearman,
    spearman_transfer_stats,
)

from eval_lodo_similarity import _algo, _ds, _task, query_from_prefix, rng_for, topk_ids

RERANK_AFTER = (8, 16, 24)
SHRINK_K = (0, 5, 10)
POOLS = ("cold", "importance")
AGG = ("top1", "top3")
CASCADE_POOL = 12
GRADE_START = 24
MIN_GRADE = 3
GENUINE_RHO = 0.3


def mean(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(sum(xs) / len(xs)) if xs else float("nan")


def ensemble_rho(
    sources: Sequence[Dict[str, Any]],
    grade: Sequence[Dict[str, Any]],
    tgt_algo: str,
    tgt_hib: bool,
) -> float:
    preds: List[List[float]] = []
    obs: List[float] | None = None
    for src in sources:
        packed = nearest_source_scores_ex(
            (src.get("retrieval") or {}).get("trials") or [],
            grade,
            src_algo=str(((src.get("problem") or {}).get("algorithm_id") or "")).lower(),
            tgt_algo=tgt_algo,
            src_higher_is_better=bool((src.get("problem") or {}).get("higher_is_better", True)),
            tgt_higher_is_better=tgt_hib,
        )
        if packed is None:
            continue
        preds.append(packed["pred"])
        obs = packed["obs"]
    if not preds or obs is None:
        return 0.0
    n = min(len(p) for p in preds)
    avg = [sum(p[i] for p in preds) / len(preds) for i in range(n)]
    return float(spearman(avg, obs[:n]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=ROOT / "history_data" / "experience_bank_v2.jsonl")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "benchmark_data" / "real_hpo_v2_mltest20_algo40" / "analysis" / "lodo_similarity",
    )
    args = p.parse_args(argv)

    bank = ExperienceBank.load(args.bank)
    split = load_split()
    warm_w = dict(bank.weights) if bank.weights else dict(DEFAULT_WEIGHTS)
    print(f"bank={len(bank)} weights={warm_w}")

    by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for exp in bank.experiences:
        trials = (exp.get("retrieval") or {}).get("trials") or []
        if len(trials) < GRADE_START + MIN_GRADE:
            continue
        by_group[(_task(exp), _algo(exp))].append(exp)

    configs = [
        {"rerank_after": ra, "shrink_k": k, "pool": pool, "agg": agg}
        for ra, k, pool, agg in product(RERANK_AFTER, SHRINK_K, POOLS, AGG)
    ]
    # per-config lists of grade rhos, keyed by split
    tune_rhos: Dict[str, List[float]] = {json.dumps(c, sort_keys=True): [] for c in configs}
    lock_rhos: Dict[str, List[float]] = {json.dumps(c, sort_keys=True): [] for c in configs}
    baseline_tune: List[float] = []
    baseline_lock: List[float] = []

    n_groups = len(by_group)
    n_tune = n_lock = 0
    for gi, ((task, algo), members) in enumerate(sorted(by_group.items()), 1):
        if len(members) < 3:
            continue
        print(f"  [{gi}/{n_groups}] {task}/{algo} n={len(members)}")
        for tgt in members:
            ds = _ds(tgt)
            if not is_tune(ds, split):
                continue
            trials = (tgt.get("retrieval") or {}).get("trials") or []
            grade = trials[GRADE_START:]
            if len(grade) < MIN_GRADE:
                continue
            cands = [s for s in members if _ds(s) != ds]
            if len(cands) < 3:
                continue
            hib = bool((tgt.get("problem") or {}).get("higher_is_better", True))
            prefixes = {ra: trials[:ra] for ra in RERANK_AFTER}
            queries = {ra: query_from_prefix(tgt, prefixes[ra]) for ra in RERANK_AFTER}

            scored: Dict[str, Dict[str, Any]] = {}
            src_by_id = {str(s.get("id")): s for s in cands}
            for src in cands:
                sid = str(src.get("id"))
                row: Dict[str, Any] = {}
                for ra in RERANK_AFTER:
                    feats = cold_features(queries[ra], src)
                    row[f"cold_{ra}"] = combine_score(feats, COLD_WEIGHTS)
                    row[f"imp_{ra}"] = combine_score(feats, warm_w)
                    st = spearman_transfer_stats(
                        src,
                        prefixes[ra],
                        target_algo=algo,
                        target_higher_is_better=hib,
                    )
                    row[f"pre_{ra}"] = st
                    for k in SHRINK_K:
                        rho_adj = float(st["rho"]) * float(st["n_overlap"]) / (
                            float(st["n_overlap"]) + k
                        )
                        row[f"rho_use_{ra}_{k}"] = rho_adj * (0.5 + 0.5 * float(st["coverage"]))
                st_g = spearman_transfer_stats(
                    src, grade, target_algo=algo, target_higher_is_better=hib
                )
                row["grade_rho"] = float(st_g["rho"])
                scored[sid] = row

            # frozen cascade baseline: importance pool @8, no shrink, top-1
            imp8 = topk_ids([(sid, scored[sid]["imp_8"]) for sid in scored], CASCADE_POOL)
            casc: List[Tuple[str, float]] = []
            for sid in imp8:
                pre = float(scored[sid]["pre_8"]["rho"])
                if pre < 0.0:
                    continue
                casc.append((sid, 0.6 * scored[sid]["imp_8"] + 0.4 * max(0.0, pre)))
            if not casc:
                casc = [(sid, scored[sid]["imp_8"]) for sid in imp8]
            casc.sort(key=lambda kv: kv[1], reverse=True)
            base = scored[casc[0][0]]["grade_rho"] if casc else float("nan")
            baseline_tune.append(base)
            n_tune += 1

            for cfg in configs:
                ck = json.dumps(cfg, sort_keys=True)
                ra, k, pool, agg = cfg["rerank_after"], cfg["shrink_k"], cfg["pool"], cfg["agg"]
                pool_key = f"cold_{ra}" if pool == "cold" else f"imp_{ra}"
                pool_ids = topk_ids([(sid, scored[sid][pool_key]) for sid in scored], CASCADE_POOL)
                ranked: List[Tuple[str, float]] = []
                for sid in pool_ids:
                    pre = float(scored[sid][f"pre_{ra}"]["rho"])
                    if pre < 0.0:
                        continue
                    ranked.append((sid, 0.6 * scored[sid][pool_key] + 0.4 * max(0.0, scored[sid][f"rho_use_{ra}_{k}"])))
                if not ranked:
                    ranked = [(sid, scored[sid][pool_key]) for sid in pool_ids]
                ranked.sort(key=lambda kv: kv[1], reverse=True)
                top_ids = [sid for sid, _ in ranked[: 3 if agg == "top3" else 1]]
                if agg == "top1":
                    val = scored[top_ids[0]]["grade_rho"]
                else:
                    val = ensemble_rho(
                        [src_by_id[sid] for sid in top_ids if sid in src_by_id],
                        grade,
                        algo,
                        hib,
                    )
                tune_rhos[ck].append(val)

    summary_rows = []
    print(f"\n=== Variance sweep TUNE n={n_tune} ===")
    print(f"{'config':<70} {'TUNE ρ':>8}")
    winner_key = None
    winner_tune = -1e9
    for cfg in configs:
        ck = json.dumps(cfg, sort_keys=True)
        tmean = mean(tune_rhos[ck])
        label = f"ra={cfg['rerank_after']} k={cfg['shrink_k']} pool={cfg['pool']} agg={cfg['agg']}"
        summary_rows.append(
            {
                "rerank_after": cfg["rerank_after"],
                "shrink_k": cfg["shrink_k"],
                "pool": cfg["pool"],
                "agg": cfg["agg"],
                "n_tune": len(tune_rhos[ck]),
                "tune_rho": tmean,
                "lock_rho": None,
            }
        )
        if tmean > winner_tune:
            winner_tune = tmean
            winner_key = ck
        print(f"{label:<70} {tmean:8.3f}")

    winner_cfg = json.loads(winner_key)
    print(
        f"\nTUNE winner: ra={winner_cfg['rerank_after']} k={winner_cfg['shrink_k']} "
        f"pool={winner_cfg['pool']} agg={winner_cfg['agg']} TUNE={winner_tune:.3f}"
    )
    print("Measuring LOCK once for the TUNE winner + frozen cascade baseline…")

    n_lock = 0
    winner_lock_vals: List[float] = []
    for (task, algo), members in sorted(by_group.items()):
        if len(members) < 3:
            continue
        for tgt in members:
            ds = _ds(tgt)
            if not is_lock(ds, split):
                continue
            trials = (tgt.get("retrieval") or {}).get("trials") or []
            grade = trials[GRADE_START:]
            if len(grade) < MIN_GRADE:
                continue
            cands = [s for s in members if _ds(s) != ds]
            if len(cands) < 3:
                continue
            hib = bool((tgt.get("problem") or {}).get("higher_is_better", True))
            ra = winner_cfg["rerank_after"]
            prefix = trials[:ra]
            q = query_from_prefix(tgt, prefix)
            q8 = query_from_prefix(tgt, trials[:8])
            scored: Dict[str, Dict[str, Any]] = {}
            src_by_id = {str(s.get("id")): s for s in cands}
            for src in cands:
                sid = str(src.get("id"))
                feats = cold_features(q, src)
                feats8 = cold_features(q8, src)
                st = spearman_transfer_stats(
                    src, prefix, target_algo=algo, target_higher_is_better=hib
                )
                k = winner_cfg["shrink_k"]
                rho_adj = float(st["rho"]) * float(st["n_overlap"]) / (float(st["n_overlap"]) + k)
                rho_use = rho_adj * (0.5 + 0.5 * float(st["coverage"]))
                st8 = spearman_transfer_stats(
                    src, trials[:8], target_algo=algo, target_higher_is_better=hib
                )
                st_g = spearman_transfer_stats(
                    src, grade, target_algo=algo, target_higher_is_better=hib
                )
                scored[sid] = {
                    "cold": combine_score(feats, COLD_WEIGHTS),
                    "imp": combine_score(feats, warm_w),
                    "imp8": combine_score(feats8, warm_w),
                    "pre": st,
                    "pre8": st8,
                    "rho_use": rho_use,
                    "grade_rho": float(st_g["rho"]),
                }
            pool = winner_cfg["pool"]
            pool_key = "cold" if pool == "cold" else "imp"
            pool_ids = topk_ids([(sid, scored[sid][pool_key]) for sid in scored], CASCADE_POOL)
            ranked: List[Tuple[str, float]] = []
            for sid in pool_ids:
                pre = float(scored[sid]["pre"]["rho"])
                if pre < 0.0:
                    continue
                ranked.append((sid, 0.6 * scored[sid][pool_key] + 0.4 * max(0.0, scored[sid]["rho_use"])))
            if not ranked:
                ranked = [(sid, scored[sid][pool_key]) for sid in pool_ids]
            ranked.sort(key=lambda kv: kv[1], reverse=True)
            top_ids = [sid for sid, _ in ranked[: 3 if winner_cfg["agg"] == "top3" else 1]]
            if winner_cfg["agg"] == "top1":
                val = scored[top_ids[0]]["grade_rho"]
            else:
                val = ensemble_rho(
                    [src_by_id[sid] for sid in top_ids if sid in src_by_id],
                    grade,
                    algo,
                    hib,
                )
            winner_lock_vals.append(val)

            imp8 = topk_ids([(sid, scored[sid]["imp8"]) for sid in scored], CASCADE_POOL)
            casc: List[Tuple[str, float]] = []
            for sid in imp8:
                pre = float(scored[sid]["pre8"]["rho"])
                if pre < 0.0:
                    continue
                casc.append((sid, 0.6 * scored[sid]["imp8"] + 0.4 * max(0.0, pre)))
            if not casc:
                casc = [(sid, scored[sid]["imp8"]) for sid in imp8]
            casc.sort(key=lambda kv: kv[1], reverse=True)
            baseline_lock.append(scored[casc[0][0]]["grade_rho"] if casc else float("nan"))
            n_lock += 1

    winner_lock = mean(winner_lock_vals)
    for row in summary_rows:
        if (
            row["rerank_after"] == winner_cfg["rerank_after"]
            and row["shrink_k"] == winner_cfg["shrink_k"]
            and row["pool"] == winner_cfg["pool"]
            and row["agg"] == winner_cfg["agg"]
        ):
            row["lock_rho"] = winner_lock
            row["selected"] = 1
        else:
            row["selected"] = 0
            row["lock_rho"] = ""

    print(
        f"\nTUNE winner: ra={winner_cfg['rerank_after']} k={winner_cfg['shrink_k']} "
        f"pool={winner_cfg['pool']} agg={winner_cfg['agg']} "
        f"TUNE={winner_tune:.3f} LOCK={winner_lock:.3f}"
    )
    print(
        f"frozen cascade baseline (ra=8, importance, top1): "
        f"TUNE={mean(baseline_tune):.3f} LOCK={mean(baseline_lock):.3f}"
    )
    fail_lock = winner_lock < mean(baseline_lock)
    if fail_lock:
        print("LOCK failed to beat frozen cascade; keep rerank_after=8 (no further search).")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    spath = args.out_dir / "variance_sweep_summary.csv"
    with spath.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    payload = {
        "n_tune": n_tune,
        "n_lock": n_lock,
        "winner": winner_cfg,
        "winner_tune_rho": winner_tune,
        "winner_lock_rho": winner_lock,
        "baseline_tune_rho": mean(baseline_tune),
        "baseline_lock_rho": mean(baseline_lock),
        "keep_rerank_after_8": bool(fail_lock),
        "gate_note": (
            "LOCK measured once for the TUNE winner. If LOCK loses, keep rerank_after=8."
        ),
    }
    jpath = args.out_dir / "variance_sweep_winner.json"
    jpath.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {spath}")
    print(f"wrote {jpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Phase 2: pre-registered variance knobs. Select on TUNE, measure LOCK once.

Grid (no additions):
  rerank_after ∈ {8, 16, 24}
  shrink k ∈ {0, 5, 10} with coverage-weighted rho_adj
  pool ∈ {cold, importance}
  aggregate ∈ {top1, top3}
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from dataset_split import is_lock, is_tune, load_split
from mindware.components.experience.retrieve import (
    COLD_WEIGHTS,
    DEFAULT_WEIGHTS,
    ExperienceBank,
    cold_features,
    combine_score,
    nearest_source_scores_ex,
    query_importance_from_trials,
    spearman,
    spearman_transfer_stats,
)

from eval_lodo_similarity import _algo, _ds, _task, query_from_prefix, rng_for, topk_ids

RERANK_AFTER = (8, 16, 24)
SHRINK_K = (0, 5, 10)
POOLS = ("cold", "importance")
AGG = ("top1", "top3")
CASCADE_POOL = 12
GRADE_START = 24
MIN_GRADE = 3
GENUINE_RHO = 0.3


def mean(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(sum(xs) / len(xs)) if xs else float("nan")


def ensemble_rho(
    sources: Sequence[Dict[str, Any]],
    grade: Sequence[Dict[str, Any]],
    tgt_algo: str,
    tgt_hib: bool,
) -> float:
    preds: List[List[float]] = []
    obs: List[float] | None = None
    for src in sources:
        packed = nearest_source_scores_ex(
            (src.get("retrieval") or {}).get("trials") or [],
            grade,
            src_algo=str(((src.get("problem") or {}).get("algorithm_id") or "")).lower(),
            tgt_algo=tgt_algo,
            src_higher_is_better=bool((src.get("problem") or {}).get("higher_is_better", True)),
            tgt_higher_is_better=tgt_hib,
        )
        if packed is None:
            continue
        preds.append(packed["pred"])
        obs = packed["obs"]
    if not preds or obs is None:
        return 0.0
    n = min(len(p) for p in preds)
    avg = [sum(p[i] for p in preds) / len(preds) for i in range(n)]
    return float(spearman(avg, obs[:n]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=ROOT / "history_data" / "experience_bank_v2.jsonl")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "benchmark_data" / "real_hpo_v2_mltest20_algo40" / "analysis" / "lodo_similarity",
    )
    args = p.parse_args(argv)

    bank = ExperienceBank.load(args.bank)
    split = load_split()
    warm_w = dict(bank.weights) if bank.weights else dict(DEFAULT_WEIGHTS)
    print(f"bank={len(bank)} weights={warm_w}")

    by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for exp in bank.experiences:
        trials = (exp.get("retrieval") or {}).get("trials") or []
        if len(trials) < GRADE_START + MIN_GRADE:
            continue
        by_group[(_task(exp), _algo(exp))].append(exp)

    configs = [
        {"rerank_after": ra, "shrink_k": k, "pool": pool, "agg": agg}
        for ra, k, pool, agg in product(RERANK_AFTER, SHRINK_K, POOLS, AGG)
    ]
    # per-config lists of grade rhos, keyed by split
    tune_rhos: Dict[str, List[float]] = {json.dumps(c, sort_keys=True): [] for c in configs}
    lock_rhos: Dict[str, List[float]] = {json.dumps(c, sort_keys=True): [] for c in configs}
    baseline_tune: List[float] = []
    baseline_lock: List[float] = []

    n_groups = len(by_group)
    n_tune = n_lock = 0
    for gi, ((task, algo), members) in enumerate(sorted(by_group.items()), 1):
        if len(members) < 3:
            continue
        print(f"  [{gi}/{n_groups}] {task}/{algo} n={len(members)}")
        for tgt in members:
            ds = _ds(tgt)
            if not is_tune(ds, split):
                continue
            trials = (tgt.get("retrieval") or {}).get("trials") or []
            grade = trials[GRADE_START:]
            if len(grade) < MIN_GRADE:
                continue
            cands = [s for s in members if _ds(s) != ds]
            if len(cands) < 3:
                continue
            hib = bool((tgt.get("problem") or {}).get("higher_is_better", True))
            prefixes = {ra: trials[:ra] for ra in RERANK_AFTER}
            queries = {ra: query_from_prefix(tgt, prefixes[ra]) for ra in RERANK_AFTER}

            scored: Dict[str, Dict[str, Any]] = {}
            src_by_id = {str(s.get("id")): s for s in cands}
            for src in cands:
                sid = str(src.get("id"))
                row: Dict[str, Any] = {}
                for ra in RERANK_AFTER:
                    feats = cold_features(queries[ra], src)
                    row[f"cold_{ra}"] = combine_score(feats, COLD_WEIGHTS)
                    row[f"imp_{ra}"] = combine_score(feats, warm_w)
                    st = spearman_transfer_stats(
                        src,
                        prefixes[ra],
                        target_algo=algo,
                        target_higher_is_better=hib,
                    )
                    row[f"pre_{ra}"] = st
                    for k in SHRINK_K:
                        rho_adj = float(st["rho"]) * float(st["n_overlap"]) / (
                            float(st["n_overlap"]) + k
                        )
                        row[f"rho_use_{ra}_{k}"] = rho_adj * (0.5 + 0.5 * float(st["coverage"]))
                st_g = spearman_transfer_stats(
                    src, grade, target_algo=algo, target_higher_is_better=hib
                )
                row["grade_rho"] = float(st_g["rho"])
                scored[sid] = row

            # frozen cascade baseline: importance pool @8, no shrink, top-1
            imp8 = topk_ids([(sid, scored[sid]["imp_8"]) for sid in scored], CASCADE_POOL)
            casc: List[Tuple[str, float]] = []
            for sid in imp8:
                pre = float(scored[sid]["pre_8"]["rho"])
                if pre < 0.0:
                    continue
                casc.append((sid, 0.6 * scored[sid]["imp_8"] + 0.4 * max(0.0, pre)))
            if not casc:
                casc = [(sid, scored[sid]["imp_8"]) for sid in imp8]
            casc.sort(key=lambda kv: kv[1], reverse=True)
            base = scored[casc[0][0]]["grade_rho"] if casc else float("nan")
            baseline_tune.append(base)
            n_tune += 1

            for cfg in configs:
                ck = json.dumps(cfg, sort_keys=True)
                ra, k, pool, agg = cfg["rerank_after"], cfg["shrink_k"], cfg["pool"], cfg["agg"]
                pool_key = f"cold_{ra}" if pool == "cold" else f"imp_{ra}"
                pool_ids = topk_ids([(sid, scored[sid][pool_key]) for sid in scored], CASCADE_POOL)
                ranked: List[Tuple[str, float]] = []
                for sid in pool_ids:
                    pre = float(scored[sid][f"pre_{ra}"]["rho"])
                    if pre < 0.0:
                        continue
                    ranked.append((sid, 0.6 * scored[sid][pool_key] + 0.4 * max(0.0, scored[sid][f"rho_use_{ra}_{k}"])))
                if not ranked:
                    ranked = [(sid, scored[sid][pool_key]) for sid in pool_ids]
                ranked.sort(key=lambda kv: kv[1], reverse=True)
                top_ids = [sid for sid, _ in ranked[: 3 if agg == "top3" else 1]]
                if agg == "top1":
                    val = scored[top_ids[0]]["grade_rho"]
                else:
                    val = ensemble_rho(
                        [src_by_id[sid] for sid in top_ids if sid in src_by_id],
                        grade,
                        algo,
                        hib,
                    )
                tune_rhos[ck].append(val)

    summary_rows = []
    print(f"\n=== Variance sweep TUNE n={n_tune} ===")
    print(f"{'config':<70} {'TUNE ρ':>8}")
    winner_key = None
    winner_tune = -1e9
    for cfg in configs:
        ck = json.dumps(cfg, sort_keys=True)
        tmean = mean(tune_rhos[ck])
        label = f"ra={cfg['rerank_after']} k={cfg['shrink_k']} pool={cfg['pool']} agg={cfg['agg']}"
        summary_rows.append(
            {
                "rerank_after": cfg["rerank_after"],
                "shrink_k": cfg["shrink_k"],
                "pool": cfg["pool"],
                "agg": cfg["agg"],
                "n_tune": len(tune_rhos[ck]),
                "tune_rho": tmean,
                "lock_rho": None,
            }
        )
        if tmean > winner_tune:
            winner_tune = tmean
            winner_key = ck
        print(f"{label:<70} {tmean:8.3f}")

    winner_cfg = json.loads(winner_key)
    print(
        f"\nTUNE winner: ra={winner_cfg['rerank_after']} k={winner_cfg['shrink_k']} "
        f"pool={winner_cfg['pool']} agg={winner_cfg['agg']} TUNE={winner_tune:.3f}"
    )
    print("Measuring LOCK once for the TUNE winner + frozen cascade baseline…")

    n_lock = 0
    winner_lock_vals: List[float] = []
    for (task, algo), members in sorted(by_group.items()):
        if len(members) < 3:
            continue
        for tgt in members:
            ds = _ds(tgt)
            if not is_lock(ds, split):
                continue
            trials = (tgt.get("retrieval") or {}).get("trials") or []
            grade = trials[GRADE_START:]
            if len(grade) < MIN_GRADE:
                continue
            cands = [s for s in members if _ds(s) != ds]
            if len(cands) < 3:
                continue
            hib = bool((tgt.get("problem") or {}).get("higher_is_better", True))
            ra = winner_cfg["rerank_after"]
            prefix = trials[:ra]
            q = query_from_prefix(tgt, prefix)
            q8 = query_from_prefix(tgt, trials[:8])
            scored: Dict[str, Dict[str, Any]] = {}
            src_by_id = {str(s.get("id")): s for s in cands}
            for src in cands:
                sid = str(src.get("id"))
                feats = cold_features(q, src)
                feats8 = cold_features(q8, src)
                st = spearman_transfer_stats(
                    src, prefix, target_algo=algo, target_higher_is_better=hib
                )
                k = winner_cfg["shrink_k"]
                rho_adj = float(st["rho"]) * float(st["n_overlap"]) / (float(st["n_overlap"]) + k)
                rho_use = rho_adj * (0.5 + 0.5 * float(st["coverage"]))
                st8 = spearman_transfer_stats(
                    src, trials[:8], target_algo=algo, target_higher_is_better=hib
                )
                st_g = spearman_transfer_stats(
                    src, grade, target_algo=algo, target_higher_is_better=hib
                )
                scored[sid] = {
                    "cold": combine_score(feats, COLD_WEIGHTS),
                    "imp": combine_score(feats, warm_w),
                    "imp8": combine_score(feats8, warm_w),
                    "pre": st,
                    "pre8": st8,
                    "rho_use": rho_use,
                    "grade_rho": float(st_g["rho"]),
                }
            pool = winner_cfg["pool"]
            pool_key = "cold" if pool == "cold" else "imp"
            pool_ids = topk_ids([(sid, scored[sid][pool_key]) for sid in scored], CASCADE_POOL)
            ranked: List[Tuple[str, float]] = []
            for sid in pool_ids:
                pre = float(scored[sid]["pre"]["rho"])
                if pre < 0.0:
                    continue
                ranked.append((sid, 0.6 * scored[sid][pool_key] + 0.4 * max(0.0, scored[sid]["rho_use"])))
            if not ranked:
                ranked = [(sid, scored[sid][pool_key]) for sid in pool_ids]
            ranked.sort(key=lambda kv: kv[1], reverse=True)
            top_ids = [sid for sid, _ in ranked[: 3 if winner_cfg["agg"] == "top3" else 1]]
            if winner_cfg["agg"] == "top1":
                val = scored[top_ids[0]]["grade_rho"]
            else:
                val = ensemble_rho(
                    [src_by_id[sid] for sid in top_ids if sid in src_by_id],
                    grade,
                    algo,
                    hib,
                )
            winner_lock_vals.append(val)

            imp8 = topk_ids([(sid, scored[sid]["imp8"]) for sid in scored], CASCADE_POOL)
            casc: List[Tuple[str, float]] = []
            for sid in imp8:
                pre = float(scored[sid]["pre8"]["rho"])
                if pre < 0.0:
                    continue
                casc.append((sid, 0.6 * scored[sid]["imp8"] + 0.4 * max(0.0, pre)))
            if not casc:
                casc = [(sid, scored[sid]["imp8"]) for sid in imp8]
            casc.sort(key=lambda kv: kv[1], reverse=True)
            baseline_lock.append(scored[casc[0][0]]["grade_rho"] if casc else float("nan"))
            n_lock += 1

    winner_lock = mean(winner_lock_vals)
    for row in summary_rows:
        if (
            row["rerank_after"] == winner_cfg["rerank_after"]
            and row["shrink_k"] == winner_cfg["shrink_k"]
            and row["pool"] == winner_cfg["pool"]
            and row["agg"] == winner_cfg["agg"]
        ):
            row["lock_rho"] = winner_lock
            row["selected"] = 1
        else:
            row["selected"] = 0
            row["lock_rho"] = ""

    print(
        f"\nTUNE winner: ra={winner_cfg['rerank_after']} k={winner_cfg['shrink_k']} "
        f"pool={winner_cfg['pool']} agg={winner_cfg['agg']} "
        f"TUNE={winner_tune:.3f} LOCK={winner_lock:.3f}"
    )
    print(
        f"frozen cascade baseline (ra=8, importance, top1): "
        f"TUNE={mean(baseline_tune):.3f} LOCK={mean(baseline_lock):.3f}"
    )
    fail_lock = winner_lock < mean(baseline_lock)
    if fail_lock:
        print("LOCK failed to beat frozen cascade; keep rerank_after=8 (no further search).")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    spath = args.out_dir / "variance_sweep_summary.csv"
    with spath.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    payload = {
        "n_tune": n_tune,
        "n_lock": n_lock,
        "winner": winner_cfg,
        "winner_tune_rho": winner_tune,
        "winner_lock_rho": winner_lock,
        "baseline_tune_rho": mean(baseline_tune),
        "baseline_lock_rho": mean(baseline_lock),
        "keep_rerank_after_8": bool(fail_lock),
        "gate_note": (
            "LOCK measured once for the TUNE winner. If LOCK loses, keep rerank_after=8."
        ),
    }
    jpath = args.out_dir / "variance_sweep_winner.json"
    jpath.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {spath}")
    print(f"wrote {jpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Phase 2: pre-registered variance knobs. Select on TUNE, measure LOCK once.

Grid (no additions):
  rerank_after ∈ {8, 16, 24}
  shrink k ∈ {0, 5, 10} with coverage-weighted rho_adj
  pool ∈ {cold, importance}
  aggregate ∈ {top1, top3}
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from dataset_split import is_lock, is_tune, load_split
from mindware.components.experience.retrieve import (
    COLD_WEIGHTS,
    DEFAULT_WEIGHTS,
    ExperienceBank,
    cold_features,
    combine_score,
    nearest_source_scores_ex,
    query_importance_from_trials,
    spearman,
    spearman_transfer_stats,
)

from eval_lodo_similarity import _algo, _ds, _task, query_from_prefix, rng_for, topk_ids

RERANK_AFTER = (8, 16, 24)
SHRINK_K = (0, 5, 10)
POOLS = ("cold", "importance")
AGG = ("top1", "top3")
CASCADE_POOL = 12
GRADE_START = 24
MIN_GRADE = 3
GENUINE_RHO = 0.3


def mean(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(sum(xs) / len(xs)) if xs else float("nan")


def ensemble_rho(
    sources: Sequence[Dict[str, Any]],
    grade: Sequence[Dict[str, Any]],
    tgt_algo: str,
    tgt_hib: bool,
) -> float:
    preds: List[List[float]] = []
    obs: List[float] | None = None
    for src in sources:
        packed = nearest_source_scores_ex(
            (src.get("retrieval") or {}).get("trials") or [],
            grade,
            src_algo=str(((src.get("problem") or {}).get("algorithm_id") or "")).lower(),
            tgt_algo=tgt_algo,
            src_higher_is_better=bool((src.get("problem") or {}).get("higher_is_better", True)),
            tgt_higher_is_better=tgt_hib,
        )
        if packed is None:
            continue
        preds.append(packed["pred"])
        obs = packed["obs"]
    if not preds or obs is None:
        return 0.0
    n = min(len(p) for p in preds)
    avg = [sum(p[i] for p in preds) / len(preds) for i in range(n)]
    return float(spearman(avg, obs[:n]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=ROOT / "history_data" / "experience_bank_v2.jsonl")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "benchmark_data" / "real_hpo_v2_mltest20_algo40" / "analysis" / "lodo_similarity",
    )
    args = p.parse_args(argv)

    bank = ExperienceBank.load(args.bank)
    split = load_split()
    warm_w = dict(bank.weights) if bank.weights else dict(DEFAULT_WEIGHTS)
    print(f"bank={len(bank)} weights={warm_w}")

    by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for exp in bank.experiences:
        trials = (exp.get("retrieval") or {}).get("trials") or []
        if len(trials) < GRADE_START + MIN_GRADE:
            continue
        by_group[(_task(exp), _algo(exp))].append(exp)

    configs = [
        {"rerank_after": ra, "shrink_k": k, "pool": pool, "agg": agg}
        for ra, k, pool, agg in product(RERANK_AFTER, SHRINK_K, POOLS, AGG)
    ]
    # per-config lists of grade rhos, keyed by split
    tune_rhos: Dict[str, List[float]] = {json.dumps(c, sort_keys=True): [] for c in configs}
    lock_rhos: Dict[str, List[float]] = {json.dumps(c, sort_keys=True): [] for c in configs}
    baseline_tune: List[float] = []
    baseline_lock: List[float] = []

    n_groups = len(by_group)
    n_tune = n_lock = 0
    for gi, ((task, algo), members) in enumerate(sorted(by_group.items()), 1):
        if len(members) < 3:
            continue
        print(f"  [{gi}/{n_groups}] {task}/{algo} n={len(members)}")
        for tgt in members:
            ds = _ds(tgt)
            if not is_tune(ds, split):
                continue
            trials = (tgt.get("retrieval") or {}).get("trials") or []
            grade = trials[GRADE_START:]
            if len(grade) < MIN_GRADE:
                continue
            cands = [s for s in members if _ds(s) != ds]
            if len(cands) < 3:
                continue
            hib = bool((tgt.get("problem") or {}).get("higher_is_better", True))
            prefixes = {ra: trials[:ra] for ra in RERANK_AFTER}
            queries = {ra: query_from_prefix(tgt, prefixes[ra]) for ra in RERANK_AFTER}

            scored: Dict[str, Dict[str, Any]] = {}
            src_by_id = {str(s.get("id")): s for s in cands}
            for src in cands:
                sid = str(src.get("id"))
                row: Dict[str, Any] = {}
                for ra in RERANK_AFTER:
                    feats = cold_features(queries[ra], src)
                    row[f"cold_{ra}"] = combine_score(feats, COLD_WEIGHTS)
                    row[f"imp_{ra}"] = combine_score(feats, warm_w)
                    st = spearman_transfer_stats(
                        src,
                        prefixes[ra],
                        target_algo=algo,
                        target_higher_is_better=hib,
                    )
                    row[f"pre_{ra}"] = st
                    for k in SHRINK_K:
                        rho_adj = float(st["rho"]) * float(st["n_overlap"]) / (
                            float(st["n_overlap"]) + k
                        )
                        row[f"rho_use_{ra}_{k}"] = rho_adj * (0.5 + 0.5 * float(st["coverage"]))
                st_g = spearman_transfer_stats(
                    src, grade, target_algo=algo, target_higher_is_better=hib
                )
                row["grade_rho"] = float(st_g["rho"])
                scored[sid] = row

            # frozen cascade baseline: importance pool @8, no shrink, top-1
            imp8 = topk_ids([(sid, scored[sid]["imp_8"]) for sid in scored], CASCADE_POOL)
            casc: List[Tuple[str, float]] = []
            for sid in imp8:
                pre = float(scored[sid]["pre_8"]["rho"])
                if pre < 0.0:
                    continue
                casc.append((sid, 0.6 * scored[sid]["imp_8"] + 0.4 * max(0.0, pre)))
            if not casc:
                casc = [(sid, scored[sid]["imp_8"]) for sid in imp8]
            casc.sort(key=lambda kv: kv[1], reverse=True)
            base = scored[casc[0][0]]["grade_rho"] if casc else float("nan")
            baseline_tune.append(base)
            n_tune += 1

            for cfg in configs:
                ck = json.dumps(cfg, sort_keys=True)
                ra, k, pool, agg = cfg["rerank_after"], cfg["shrink_k"], cfg["pool"], cfg["agg"]
                pool_key = f"cold_{ra}" if pool == "cold" else f"imp_{ra}"
                pool_ids = topk_ids([(sid, scored[sid][pool_key]) for sid in scored], CASCADE_POOL)
                ranked: List[Tuple[str, float]] = []
                for sid in pool_ids:
                    pre = float(scored[sid][f"pre_{ra}"]["rho"])
                    if pre < 0.0:
                        continue
                    ranked.append((sid, 0.6 * scored[sid][pool_key] + 0.4 * max(0.0, scored[sid][f"rho_use_{ra}_{k}"])))
                if not ranked:
                    ranked = [(sid, scored[sid][pool_key]) for sid in pool_ids]
                ranked.sort(key=lambda kv: kv[1], reverse=True)
                top_ids = [sid for sid, _ in ranked[: 3 if agg == "top3" else 1]]
                if agg == "top1":
                    val = scored[top_ids[0]]["grade_rho"]
                else:
                    val = ensemble_rho(
                        [src_by_id[sid] for sid in top_ids if sid in src_by_id],
                        grade,
                        algo,
                        hib,
                    )
                tune_rhos[ck].append(val)

    summary_rows = []
    print(f"\n=== Variance sweep TUNE n={n_tune} ===")
    print(f"{'config':<70} {'TUNE ρ':>8}")
    winner_key = None
    winner_tune = -1e9
    for cfg in configs:
        ck = json.dumps(cfg, sort_keys=True)
        tmean = mean(tune_rhos[ck])
        label = f"ra={cfg['rerank_after']} k={cfg['shrink_k']} pool={cfg['pool']} agg={cfg['agg']}"
        summary_rows.append(
            {
                "rerank_after": cfg["rerank_after"],
                "shrink_k": cfg["shrink_k"],
                "pool": cfg["pool"],
                "agg": cfg["agg"],
                "n_tune": len(tune_rhos[ck]),
                "tune_rho": tmean,
                "lock_rho": None,
            }
        )
        if tmean > winner_tune:
            winner_tune = tmean
            winner_key = ck
        print(f"{label:<70} {tmean:8.3f}")

    winner_cfg = json.loads(winner_key)
    print(
        f"\nTUNE winner: ra={winner_cfg['rerank_after']} k={winner_cfg['shrink_k']} "
        f"pool={winner_cfg['pool']} agg={winner_cfg['agg']} TUNE={winner_tune:.3f}"
    )
    print("Measuring LOCK once for the TUNE winner + frozen cascade baseline…")

    n_lock = 0
    winner_lock_vals: List[float] = []
    for (task, algo), members in sorted(by_group.items()):
        if len(members) < 3:
            continue
        for tgt in members:
            ds = _ds(tgt)
            if not is_lock(ds, split):
                continue
            trials = (tgt.get("retrieval") or {}).get("trials") or []
            grade = trials[GRADE_START:]
            if len(grade) < MIN_GRADE:
                continue
            cands = [s for s in members if _ds(s) != ds]
            if len(cands) < 3:
                continue
            hib = bool((tgt.get("problem") or {}).get("higher_is_better", True))
            ra = winner_cfg["rerank_after"]
            prefix = trials[:ra]
            q = query_from_prefix(tgt, prefix)
            q8 = query_from_prefix(tgt, trials[:8])
            scored: Dict[str, Dict[str, Any]] = {}
            src_by_id = {str(s.get("id")): s for s in cands}
            for src in cands:
                sid = str(src.get("id"))
                feats = cold_features(q, src)
                feats8 = cold_features(q8, src)
                st = spearman_transfer_stats(
                    src, prefix, target_algo=algo, target_higher_is_better=hib
                )
                k = winner_cfg["shrink_k"]
                rho_adj = float(st["rho"]) * float(st["n_overlap"]) / (float(st["n_overlap"]) + k)
                rho_use = rho_adj * (0.5 + 0.5 * float(st["coverage"]))
                st8 = spearman_transfer_stats(
                    src, trials[:8], target_algo=algo, target_higher_is_better=hib
                )
                st_g = spearman_transfer_stats(
                    src, grade, target_algo=algo, target_higher_is_better=hib
                )
                scored[sid] = {
                    "cold": combine_score(feats, COLD_WEIGHTS),
                    "imp": combine_score(feats, warm_w),
                    "imp8": combine_score(feats8, warm_w),
                    "pre": st,
                    "pre8": st8,
                    "rho_use": rho_use,
                    "grade_rho": float(st_g["rho"]),
                }
            pool = winner_cfg["pool"]
            pool_key = "cold" if pool == "cold" else "imp"
            pool_ids = topk_ids([(sid, scored[sid][pool_key]) for sid in scored], CASCADE_POOL)
            ranked: List[Tuple[str, float]] = []
            for sid in pool_ids:
                pre = float(scored[sid]["pre"]["rho"])
                if pre < 0.0:
                    continue
                ranked.append((sid, 0.6 * scored[sid][pool_key] + 0.4 * max(0.0, scored[sid]["rho_use"])))
            if not ranked:
                ranked = [(sid, scored[sid][pool_key]) for sid in pool_ids]
            ranked.sort(key=lambda kv: kv[1], reverse=True)
            top_ids = [sid for sid, _ in ranked[: 3 if winner_cfg["agg"] == "top3" else 1]]
            if winner_cfg["agg"] == "top1":
                val = scored[top_ids[0]]["grade_rho"]
            else:
                val = ensemble_rho(
                    [src_by_id[sid] for sid in top_ids if sid in src_by_id],
                    grade,
                    algo,
                    hib,
                )
            winner_lock_vals.append(val)

            imp8 = topk_ids([(sid, scored[sid]["imp8"]) for sid in scored], CASCADE_POOL)
            casc: List[Tuple[str, float]] = []
            for sid in imp8:
                pre = float(scored[sid]["pre8"]["rho"])
                if pre < 0.0:
                    continue
                casc.append((sid, 0.6 * scored[sid]["imp8"] + 0.4 * max(0.0, pre)))
            if not casc:
                casc = [(sid, scored[sid]["imp8"]) for sid in imp8]
            casc.sort(key=lambda kv: kv[1], reverse=True)
            baseline_lock.append(scored[casc[0][0]]["grade_rho"] if casc else float("nan"))
            n_lock += 1

    winner_lock = mean(winner_lock_vals)
    for row in summary_rows:
        if (
            row["rerank_after"] == winner_cfg["rerank_after"]
            and row["shrink_k"] == winner_cfg["shrink_k"]
            and row["pool"] == winner_cfg["pool"]
            and row["agg"] == winner_cfg["agg"]
        ):
            row["lock_rho"] = winner_lock
            row["selected"] = 1
        else:
            row["selected"] = 0
            row["lock_rho"] = ""

    print(
        f"\nTUNE winner: ra={winner_cfg['rerank_after']} k={winner_cfg['shrink_k']} "
        f"pool={winner_cfg['pool']} agg={winner_cfg['agg']} "
        f"TUNE={winner_tune:.3f} LOCK={winner_lock:.3f}"
    )
    print(
        f"frozen cascade baseline (ra=8, importance, top1): "
        f"TUNE={mean(baseline_tune):.3f} LOCK={mean(baseline_lock):.3f}"
    )
    fail_lock = winner_lock < mean(baseline_lock)
    if fail_lock:
        print("LOCK failed to beat frozen cascade; keep rerank_after=8 (no further search).")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    spath = args.out_dir / "variance_sweep_summary.csv"
    with spath.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    payload = {
        "n_tune": n_tune,
        "n_lock": n_lock,
        "winner": winner_cfg,
        "winner_tune_rho": winner_tune,
        "winner_lock_rho": winner_lock,
        "baseline_tune_rho": mean(baseline_tune),
        "baseline_lock_rho": mean(baseline_lock),
        "keep_rerank_after_8": bool(fail_lock),
        "gate_note": (
            "LOCK measured once for the TUNE winner. If LOCK loses, keep rerank_after=8."
        ),
    }
    jpath = args.out_dir / "variance_sweep_winner.json"
    jpath.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {spath}")
    print(f"wrote {jpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Phase 2: pre-registered variance knobs. Select on TUNE, measure LOCK once.

Grid (no additions):
  rerank_after ∈ {8, 16, 24}
  shrink k ∈ {0, 5, 10} with coverage-weighted rho_adj
  pool ∈ {cold, importance}
  aggregate ∈ {top1, top3}
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from dataset_split import is_lock, is_tune, load_split
from mindware.components.experience.retrieve import (
    COLD_WEIGHTS,
    DEFAULT_WEIGHTS,
    ExperienceBank,
    cold_features,
    combine_score,
    nearest_source_scores_ex,
    query_importance_from_trials,
    spearman,
    spearman_transfer_stats,
)

from eval_lodo_similarity import _algo, _ds, _task, query_from_prefix, rng_for, topk_ids

RERANK_AFTER = (8, 16, 24)
SHRINK_K = (0, 5, 10)
POOLS = ("cold", "importance")
AGG = ("top1", "top3")
CASCADE_POOL = 12
GRADE_START = 24
MIN_GRADE = 3
GENUINE_RHO = 0.3


def mean(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(sum(xs) / len(xs)) if xs else float("nan")


def ensemble_rho(
    sources: Sequence[Dict[str, Any]],
    grade: Sequence[Dict[str, Any]],
    tgt_algo: str,
    tgt_hib: bool,
) -> float:
    preds: List[List[float]] = []
    obs: List[float] | None = None
    for src in sources:
        packed = nearest_source_scores_ex(
            (src.get("retrieval") or {}).get("trials") or [],
            grade,
            src_algo=str(((src.get("problem") or {}).get("algorithm_id") or "")).lower(),
            tgt_algo=tgt_algo,
            src_higher_is_better=bool((src.get("problem") or {}).get("higher_is_better", True)),
            tgt_higher_is_better=tgt_hib,
        )
        if packed is None:
            continue
        preds.append(packed["pred"])
        obs = packed["obs"]
    if not preds or obs is None:
        return 0.0
    n = min(len(p) for p in preds)
    avg = [sum(p[i] for p in preds) / len(preds) for i in range(n)]
    return float(spearman(avg, obs[:n]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=ROOT / "history_data" / "experience_bank_v2.jsonl")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "benchmark_data" / "real_hpo_v2_mltest20_algo40" / "analysis" / "lodo_similarity",
    )
    args = p.parse_args(argv)

    bank = ExperienceBank.load(args.bank)
    split = load_split()
    warm_w = dict(bank.weights) if bank.weights else dict(DEFAULT_WEIGHTS)
    print(f"bank={len(bank)} weights={warm_w}")

    by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for exp in bank.experiences:
        trials = (exp.get("retrieval") or {}).get("trials") or []
        if len(trials) < GRADE_START + MIN_GRADE:
            continue
        by_group[(_task(exp), _algo(exp))].append(exp)

    configs = [
        {"rerank_after": ra, "shrink_k": k, "pool": pool, "agg": agg}
        for ra, k, pool, agg in product(RERANK_AFTER, SHRINK_K, POOLS, AGG)
    ]
    # per-config lists of grade rhos, keyed by split
    tune_rhos: Dict[str, List[float]] = {json.dumps(c, sort_keys=True): [] for c in configs}
    lock_rhos: Dict[str, List[float]] = {json.dumps(c, sort_keys=True): [] for c in configs}
    baseline_tune: List[float] = []
    baseline_lock: List[float] = []

    n_groups = len(by_group)
    n_tune = n_lock = 0
    for gi, ((task, algo), members) in enumerate(sorted(by_group.items()), 1):
        if len(members) < 3:
            continue
        print(f"  [{gi}/{n_groups}] {task}/{algo} n={len(members)}")
        for tgt in members:
            ds = _ds(tgt)
            if not is_tune(ds, split):
                continue
            trials = (tgt.get("retrieval") or {}).get("trials") or []
            grade = trials[GRADE_START:]
            if len(grade) < MIN_GRADE:
                continue
            cands = [s for s in members if _ds(s) != ds]
            if len(cands) < 3:
                continue
            hib = bool((tgt.get("problem") or {}).get("higher_is_better", True))
            prefixes = {ra: trials[:ra] for ra in RERANK_AFTER}
            queries = {ra: query_from_prefix(tgt, prefixes[ra]) for ra in RERANK_AFTER}

            scored: Dict[str, Dict[str, Any]] = {}
            src_by_id = {str(s.get("id")): s for s in cands}
            for src in cands:
                sid = str(src.get("id"))
                row: Dict[str, Any] = {}
                for ra in RERANK_AFTER:
                    feats = cold_features(queries[ra], src)
                    row[f"cold_{ra}"] = combine_score(feats, COLD_WEIGHTS)
                    row[f"imp_{ra}"] = combine_score(feats, warm_w)
                    st = spearman_transfer_stats(
                        src,
                        prefixes[ra],
                        target_algo=algo,
                        target_higher_is_better=hib,
                    )
                    row[f"pre_{ra}"] = st
                    for k in SHRINK_K:
                        rho_adj = float(st["rho"]) * float(st["n_overlap"]) / (
                            float(st["n_overlap"]) + k
                        )
                        row[f"rho_use_{ra}_{k}"] = rho_adj * (0.5 + 0.5 * float(st["coverage"]))
                st_g = spearman_transfer_stats(
                    src, grade, target_algo=algo, target_higher_is_better=hib
                )
                row["grade_rho"] = float(st_g["rho"])
                scored[sid] = row

            # frozen cascade baseline: importance pool @8, no shrink, top-1
            imp8 = topk_ids([(sid, scored[sid]["imp_8"]) for sid in scored], CASCADE_POOL)
            casc: List[Tuple[str, float]] = []
            for sid in imp8:
                pre = float(scored[sid]["pre_8"]["rho"])
                if pre < 0.0:
                    continue
                casc.append((sid, 0.6 * scored[sid]["imp_8"] + 0.4 * max(0.0, pre)))
            if not casc:
                casc = [(sid, scored[sid]["imp_8"]) for sid in imp8]
            casc.sort(key=lambda kv: kv[1], reverse=True)
            base = scored[casc[0][0]]["grade_rho"] if casc else float("nan")
            baseline_tune.append(base)
            n_tune += 1

            for cfg in configs:
                ck = json.dumps(cfg, sort_keys=True)
                ra, k, pool, agg = cfg["rerank_after"], cfg["shrink_k"], cfg["pool"], cfg["agg"]
                pool_key = f"cold_{ra}" if pool == "cold" else f"imp_{ra}"
                pool_ids = topk_ids([(sid, scored[sid][pool_key]) for sid in scored], CASCADE_POOL)
                ranked: List[Tuple[str, float]] = []
                for sid in pool_ids:
                    pre = float(scored[sid][f"pre_{ra}"]["rho"])
                    if pre < 0.0:
                        continue
                    ranked.append((sid, 0.6 * scored[sid][pool_key] + 0.4 * max(0.0, scored[sid][f"rho_use_{ra}_{k}"])))
                if not ranked:
                    ranked = [(sid, scored[sid][pool_key]) for sid in pool_ids]
                ranked.sort(key=lambda kv: kv[1], reverse=True)
                top_ids = [sid for sid, _ in ranked[: 3 if agg == "top3" else 1]]
                if agg == "top1":
                    val = scored[top_ids[0]]["grade_rho"]
                else:
                    val = ensemble_rho(
                        [src_by_id[sid] for sid in top_ids if sid in src_by_id],
                        grade,
                        algo,
                        hib,
                    )
                tune_rhos[ck].append(val)

    summary_rows = []
    print(f"\n=== Variance sweep TUNE n={n_tune} ===")
    print(f"{'config':<70} {'TUNE ρ':>8}")
    winner_key = None
    winner_tune = -1e9
    for cfg in configs:
        ck = json.dumps(cfg, sort_keys=True)
        tmean = mean(tune_rhos[ck])
        label = f"ra={cfg['rerank_after']} k={cfg['shrink_k']} pool={cfg['pool']} agg={cfg['agg']}"
        summary_rows.append(
            {
                "rerank_after": cfg["rerank_after"],
                "shrink_k": cfg["shrink_k"],
                "pool": cfg["pool"],
                "agg": cfg["agg"],
                "n_tune": len(tune_rhos[ck]),
                "tune_rho": tmean,
                "lock_rho": None,
            }
        )
        if tmean > winner_tune:
            winner_tune = tmean
            winner_key = ck
        print(f"{label:<70} {tmean:8.3f}")

    winner_cfg = json.loads(winner_key)
    print(
        f"\nTUNE winner: ra={winner_cfg['rerank_after']} k={winner_cfg['shrink_k']} "
        f"pool={winner_cfg['pool']} agg={winner_cfg['agg']} TUNE={winner_tune:.3f}"
    )
    print("Measuring LOCK once for the TUNE winner + frozen cascade baseline…")

    n_lock = 0
    winner_lock_vals: List[float] = []
    for (task, algo), members in sorted(by_group.items()):
        if len(members) < 3:
            continue
        for tgt in members:
            ds = _ds(tgt)
            if not is_lock(ds, split):
                continue
            trials = (tgt.get("retrieval") or {}).get("trials") or []
            grade = trials[GRADE_START:]
            if len(grade) < MIN_GRADE:
                continue
            cands = [s for s in members if _ds(s) != ds]
            if len(cands) < 3:
                continue
            hib = bool((tgt.get("problem") or {}).get("higher_is_better", True))
            ra = winner_cfg["rerank_after"]
            prefix = trials[:ra]
            q = query_from_prefix(tgt, prefix)
            q8 = query_from_prefix(tgt, trials[:8])
            scored: Dict[str, Dict[str, Any]] = {}
            src_by_id = {str(s.get("id")): s for s in cands}
            for src in cands:
                sid = str(src.get("id"))
                feats = cold_features(q, src)
                feats8 = cold_features(q8, src)
                st = spearman_transfer_stats(
                    src, prefix, target_algo=algo, target_higher_is_better=hib
                )
                k = winner_cfg["shrink_k"]
                rho_adj = float(st["rho"]) * float(st["n_overlap"]) / (float(st["n_overlap"]) + k)
                rho_use = rho_adj * (0.5 + 0.5 * float(st["coverage"]))
                st8 = spearman_transfer_stats(
                    src, trials[:8], target_algo=algo, target_higher_is_better=hib
                )
                st_g = spearman_transfer_stats(
                    src, grade, target_algo=algo, target_higher_is_better=hib
                )
                scored[sid] = {
                    "cold": combine_score(feats, COLD_WEIGHTS),
                    "imp": combine_score(feats, warm_w),
                    "imp8": combine_score(feats8, warm_w),
                    "pre": st,
                    "pre8": st8,
                    "rho_use": rho_use,
                    "grade_rho": float(st_g["rho"]),
                }
            pool = winner_cfg["pool"]
            pool_key = "cold" if pool == "cold" else "imp"
            pool_ids = topk_ids([(sid, scored[sid][pool_key]) for sid in scored], CASCADE_POOL)
            ranked: List[Tuple[str, float]] = []
            for sid in pool_ids:
                pre = float(scored[sid]["pre"]["rho"])
                if pre < 0.0:
                    continue
                ranked.append((sid, 0.6 * scored[sid][pool_key] + 0.4 * max(0.0, scored[sid]["rho_use"])))
            if not ranked:
                ranked = [(sid, scored[sid][pool_key]) for sid in pool_ids]
            ranked.sort(key=lambda kv: kv[1], reverse=True)
            top_ids = [sid for sid, _ in ranked[: 3 if winner_cfg["agg"] == "top3" else 1]]
            if winner_cfg["agg"] == "top1":
                val = scored[top_ids[0]]["grade_rho"]
            else:
                val = ensemble_rho(
                    [src_by_id[sid] for sid in top_ids if sid in src_by_id],
                    grade,
                    algo,
                    hib,
                )
            winner_lock_vals.append(val)

            imp8 = topk_ids([(sid, scored[sid]["imp8"]) for sid in scored], CASCADE_POOL)
            casc: List[Tuple[str, float]] = []
            for sid in imp8:
                pre = float(scored[sid]["pre8"]["rho"])
                if pre < 0.0:
                    continue
                casc.append((sid, 0.6 * scored[sid]["imp8"] + 0.4 * max(0.0, pre)))
            if not casc:
                casc = [(sid, scored[sid]["imp8"]) for sid in imp8]
            casc.sort(key=lambda kv: kv[1], reverse=True)
            baseline_lock.append(scored[casc[0][0]]["grade_rho"] if casc else float("nan"))
            n_lock += 1

    winner_lock = mean(winner_lock_vals)
    for row in summary_rows:
        if (
            row["rerank_after"] == winner_cfg["rerank_after"]
            and row["shrink_k"] == winner_cfg["shrink_k"]
            and row["pool"] == winner_cfg["pool"]
            and row["agg"] == winner_cfg["agg"]
        ):
            row["lock_rho"] = winner_lock
            row["selected"] = 1
        else:
            row["selected"] = 0
            row["lock_rho"] = ""

    print(
        f"\nTUNE winner: ra={winner_cfg['rerank_after']} k={winner_cfg['shrink_k']} "
        f"pool={winner_cfg['pool']} agg={winner_cfg['agg']} "
        f"TUNE={winner_tune:.3f} LOCK={winner_lock:.3f}"
    )
    print(
        f"frozen cascade baseline (ra=8, importance, top1): "
        f"TUNE={mean(baseline_tune):.3f} LOCK={mean(baseline_lock):.3f}"
    )
    fail_lock = winner_lock < mean(baseline_lock)
    if fail_lock:
        print("LOCK failed to beat frozen cascade; keep rerank_after=8 (no further search).")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    spath = args.out_dir / "variance_sweep_summary.csv"
    with spath.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    payload = {
        "n_tune": n_tune,
        "n_lock": n_lock,
        "winner": winner_cfg,
        "winner_tune_rho": winner_tune,
        "winner_lock_rho": winner_lock,
        "baseline_tune_rho": mean(baseline_tune),
        "baseline_lock_rho": mean(baseline_lock),
        "keep_rerank_after_8": bool(fail_lock),
        "gate_note": (
            "LOCK measured once for the TUNE winner. If LOCK loses, keep rerank_after=8."
        ),
    }
    jpath = args.out_dir / "variance_sweep_winner.json"
    jpath.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {spath}")
    print(f"wrote {jpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
