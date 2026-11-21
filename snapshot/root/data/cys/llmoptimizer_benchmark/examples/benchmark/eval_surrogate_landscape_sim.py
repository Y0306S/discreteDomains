#!/usr/bin/env python3
"""Phase A: TUNE-LODO validation of per-algo GP landscape similarity.

Select surrogate knobs on TUNE (n_query_configs × min_prefix_trials). Measure
the winner once on LOCK. Does not write the production cache (Phase B).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from dataset_split import is_lock, is_tune, load_split
from mindware.components.experience.retrieve import ExperienceBank, spearman_transfer
from mindware.components.experience.surrogate_landscape import (
    get_algo_cs,
    predict_landscape,
    sample_query_configs,
    spearman_sim,
)

from eval_lodo_similarity import _algo, _ds, _task

N_QUERY_CONFIGS = (25, 50, 100)
MIN_PREFIX = (8, 16, 24)
GRADE_START = 24
MIN_GRADE = 3
GENUINE_RHO = 0.3
SEED = 42


def mean(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _usable(exp: Dict[str, Any], min_trials: int) -> bool:
    trials = (exp.get("retrieval") or {}).get("trials") or []
    return len(trials) >= min_trials + MIN_GRADE


def evaluate_cell(
    targets: Sequence[Dict[str, Any]],
    bank_by_group: Dict[Tuple[str, str], List[Dict[str, Any]]],
    *,
    n_query: int,
    min_prefix: int,
    cs_cache: Dict[Tuple[str, str], Any],
    cfg_cache: Dict[Tuple[str, str, int], list],
    pred_cache: Dict[Tuple[str, str, str, int], Optional[Any]],
    max_queries: Optional[int] = None,
) -> Dict[str, float]:
    rhos: List[float] = []
    genuine: List[float] = []
    sim_vs_rho: List[Tuple[float, float]] = []
    n_done = 0
    for tgt in targets:
        if max_queries is not None and n_done >= max_queries:
            break
        trials = (tgt.get("retrieval") or {}).get("trials") or []
        if len(trials) < min_prefix:
            continue
        prefix = trials[:min_prefix]
        grade = trials[GRADE_START:] if len(trials) > GRADE_START else trials[min_prefix:]
        if len(grade) < MIN_GRADE:
            continue
        algo = _algo(tgt)
        task = _task(tgt)
        ds = _ds(tgt)
        key_cs = (task, algo)
        if key_cs not in cs_cache:
            try:
                cs_cache[key_cs] = get_algo_cs(algo, task)
            except Exception:
                continue
        cs = cs_cache[key_cs]
        cfg_key = (task, algo, n_query)
        if cfg_key not in cfg_cache:
            cfg_cache[cfg_key] = sample_query_configs(cs, n_query, seed=SEED)
        qcfgs = cfg_cache[cfg_key]
        qkey = ("query", ds, algo, min_prefix)
        if qkey not in pred_cache:
            pred_cache[qkey] = predict_landscape(prefix, cs, algo, qcfgs, min_trials=min_prefix)
        qvec = pred_cache[qkey]
        if qvec is None:
            continue
        group = [
            e
            for e in bank_by_group.get((task, algo), [])
            if _ds(e) != ds
        ]
        best_src = None
        best_sim = -2.0
        for src in group:
            skey = ("bank", _ds(src), algo, n_query)
            if skey not in pred_cache:
                src_trials = (src.get("retrieval") or {}).get("trials") or []
                pred_cache[skey] = predict_landscape(
                    src_trials, cs, algo, qcfgs, min_trials=min_prefix
                )
            svec = pred_cache[skey]
            sim = spearman_sim(qvec, svec)
            if sim > best_sim:
                best_sim = sim
                best_src = src
        if best_src is None:
            continue
        rho = float(
            spearman_transfer(
                best_src,
                grade,
                target_algo=algo,
                target_higher_is_better=bool((tgt.get("problem") or {}).get("higher_is_better", True)),
            )
        )
        rhos.append(rho)
        genuine.append(1.0 if rho >= GENUINE_RHO else 0.0)
        sim_vs_rho.append((best_sim, rho))
        n_done += 1
    corr = float("nan")
    if len(sim_vs_rho) >= 5:
        xs = [a for a, _ in sim_vs_rho]
        ys = [b for _, b in sim_vs_rho]
        mx, my = mean(xs), mean(ys)
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
        corr = num / den if den > 1e-12 else float("nan")
    return {
        "n": float(len(rhos)),
        "mean_rho_top1": mean(rhos),
        "frac_rho_ge_0.3": mean(genuine),
        "sim_vs_transfer_corr": corr,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=ROOT / "history_data" / "experience_bank_v2.jsonl")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "benchmark_data" / "real_hpo_v2_mltest20_algo40" / "analysis" / "lodo_similarity",
    )
    p.add_argument("--split", choices=("tune", "lock"), default="tune")
    p.add_argument("--smoke", action="store_true", help="Tiny subset: 4 queries, n=50, prefix=24")
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--algorithms", nargs="*", default=None)
    args = p.parse_args(argv)

    bank = ExperienceBank.load(args.bank)
    split = load_split()
    n_query_grid = (50,) if args.smoke else N_QUERY_CONFIGS
    prefix_grid = (24,) if args.smoke else MIN_PREFIX
    max_q = args.max_queries if args.max_queries is not None else (4 if args.smoke else None)

    def keep(exp, which):
        if not _usable(exp, 8):
            return False
        ds = _ds(exp)
        if which == "tune" and not is_tune(ds, split):
            return False
        if which == "lock" and not is_lock(ds, split):
            return False
        if args.algorithms and _algo(exp) not in {a.lower() for a in args.algorithms}:
            return False
        return True

    tune_exps = [e for e in bank.experiences if keep(e, "tune")]
    lock_exps = [e for e in bank.experiences if keep(e, "lock")]
    print(f"bank={len(bank)} tune={len(tune_exps)} lock={len(lock_exps)}")

    by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for exp in bank.experiences:
        if _usable(exp, 8) and is_tune(_ds(exp), split):
            if args.algorithms and _algo(exp) not in {a.lower() for a in args.algorithms}:
                continue
            by_group[(_task(exp), _algo(exp))].append(exp)

    cs_cache: Dict[Tuple[str, str], Any] = {}
    cfg_cache: Dict[Tuple[str, str, int], list] = {}
    pred_cache: Dict[Tuple[str, str, str, int], Optional[Any]] = {}

    targets = tune_exps if args.split == "tune" else lock_exps
    rows = []
    winner = None
    winner_rho = float("-inf")
    for n_query, min_prefix in product(n_query_grid, prefix_grid):
        stats = evaluate_cell(
            targets,
            by_group,
            n_query=n_query,
            min_prefix=min_prefix,
            cs_cache=cs_cache,
            cfg_cache=cfg_cache,
            pred_cache=pred_cache,
            max_queries=max_q,
        )
        row = {"n_query_configs": n_query, "min_prefix_trials": min_prefix, "split": args.split, **stats}
        rows.append(row)
        print(
            f"n={n_query} prefix={min_prefix} queries={stats['n']:.0f} "
            f"rho={stats['mean_rho_top1']:.3f} ge0.3={stats['frac_rho_ge_0.3']:.3f}"
        )
        if args.split == "tune" and stats["n"] > 0 and stats["mean_rho_top1"] > winner_rho:
            winner_rho = stats["mean_rho_top1"]
            winner = row

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / f"surrogate_landscape_{args.split}.json"
    payload: Dict[str, Any] = {"rows": rows, "smoke": bool(args.smoke)}
    if winner:
        payload["winner"] = winner
        if not args.smoke:
            lock_stats = evaluate_cell(
                lock_exps,
                by_group,
                n_query=int(winner["n_query_configs"]),
                min_prefix=int(winner["min_prefix_trials"]),
                cs_cache=cs_cache,
                cfg_cache=cfg_cache,
                pred_cache=pred_cache,
                max_queries=max_q,
            )
            payload["lock"] = lock_stats
            payload["pass_gate"] = bool(
                winner["mean_rho_top1"] == winner["mean_rho_top1"]
                and lock_stats["n"] > 0
                and (lock_stats["mean_rho_top1"] + 0.05) >= winner["mean_rho_top1"]
            )
            print(
                f"LOCK n={lock_stats['n']:.0f} rho={lock_stats['mean_rho_top1']:.3f} "
                f"pass_gate={payload['pass_gate']}"
            )
        winner_path = args.out_dir / "surrogate_landscape_winner.json"
        winner_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {winner_path}")
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
