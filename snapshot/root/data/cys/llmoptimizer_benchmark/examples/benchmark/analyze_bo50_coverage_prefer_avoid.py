#!/usr/bin/env python3
"""Coverage + A5 prefer/avoid richness at BO prefixes vs LLM+BO / classical TPE.

Answers: at k=50, is search-space coverage and good/bad region signal enough
to insert P5+ explore/exploit prompts — and how does it compare to LLM+BO configs?

  python examples/benchmark/analyze_bo50_coverage_prefer_avoid.py
"""

from __future__ import annotations

import ast
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "examples" / "benchmark"))

from bo_history_utils import (  # noqa: E402
    bo_history_path,
    load_bo_observations,
    observations_to_perf_pairs,
)
from mindware.components.models.classification import _classifiers  # noqa: E402
from mindware.components.models.regression import _regressors  # noqa: E402
from mindware.components.optimizers.llm_prompt_blocks import (  # noqa: E402
    build_experience_self_block,
    compute_coverage_stats,
    mean_coverage_pct,
    since_last_improvement,
)

TASKS_JSON = REPO / "benchmark_data/rl_task_selection/picked_32_pareto_tasks.json"
OUT_DIR = REPO / "benchmark_data/rl_task_selection/bo50_coverage_prefer_avoid"
PREFIXES = [10, 20, 30, 40, 50, 60, 70, 80, 100]
LLM_ROOT = REPO / "benchmark_data/real_hpo_v2_mltest20_algo40_p5_s1a_ep7p5"
CLASSICAL_ROOT = REPO / "benchmark_data/real_hpo_v2_mltest20_algo40/classical"


def prefixed_cs(task_type: str, algo: str) -> ConfigurationSpace:
    reg = _classifiers if task_type == "cls" else _regressors
    if algo not in reg:
        raise KeyError(f"unknown algorithm {algo} for {task_type}")
    base = reg[algo].get_hyperparameter_search_space()
    cs = ConfigurationSpace()
    cs.add_hyperparameter(CategoricalHyperparameter("algorithm", [algo]))
    for hp in base.get_hyperparameters():
        name = f"{algo}:{hp.name}"
        if isinstance(hp, UniformFloatHyperparameter):
            cs.add_hyperparameter(
                UniformFloatHyperparameter(
                    name, hp.lower, hp.upper, log=hp.log, default_value=hp.default_value
                )
            )
        elif isinstance(hp, UniformIntegerHyperparameter):
            cs.add_hyperparameter(
                UniformIntegerHyperparameter(
                    name, hp.lower, hp.upper, log=hp.log, default_value=hp.default_value
                )
            )
        elif isinstance(hp, CategoricalHyperparameter):
            cs.add_hyperparameter(
                CategoricalHyperparameter(name, list(hp.choices), default_value=hp.default_value)
            )
    return cs


def to_successes(cfgs: List[dict], perfs: List[float]) -> List[dict]:
    return [{"trial": i + 1, "config": c, "score": float(s)} for i, (c, s) in enumerate(zip(cfgs, perfs))]


def parse_a5_stats(block: str) -> Dict[str, Any]:
    if not block:
        return {
            "has_block": False,
            "n_factor_lines": 0,
            "n_prefer": 0,
            "n_avoid": 0,
            "clear_separation": False,
            "trend": None,
            "stagnation": None,
            "incumbent_updates": None,
        }
    factor_lines = [
        ln for ln in block.splitlines() if ln.startswith("- ") and ("prefer" in ln or "avoid" in ln) and ":" in ln
        and not ln.startswith("- factors")
        and not ln.startswith("- avoid:")
        and not ln.startswith("- mode=")
        and not ln.startswith("- dynamics")
        and not ln.startswith("- baseline")
    ]
    # factor lines look like "- lightgbm:lr: ... prefer ...; avoid ..."
    factors = [
        ln
        for ln in block.splitlines()
        if re.match(r"^- [A-Za-z0-9_]+:", ln)
        and ("prefer" in ln or "avoid" in ln or "marginal" in ln or "over-represented" in ln)
    ]
    # more reliable: lines under factors section until avoid: or Use these
    lines = block.splitlines()
    in_factors = False
    factors = []
    avoid_items = []
    in_avoid = False
    trend = stag = inc = None
    for ln in lines:
        if ln.startswith("- dynamics:"):
            m = re.search(r"trend=([A-Z_]+).*incumbent_updates=(\d+).*stagnation=(\d+)", ln)
            if m:
                trend, inc, stag = m.group(1), int(m.group(2)), int(m.group(3))
        if ln.startswith("- factors"):
            in_factors = True
            in_avoid = False
            continue
        if ln.startswith("- avoid:"):
            in_factors = False
            in_avoid = True
            if "none above" in ln:
                in_avoid = False
            continue
        if ln.startswith("Use these") or ln.startswith("- mode=") or ln.startswith("- baseline"):
            in_factors = False
            in_avoid = False
            continue
        if in_factors and ln.startswith("- "):
            if "no clear separation" in ln:
                continue
            factors.append(ln)
        if in_avoid and ln.startswith("- "):
            avoid_items.append(ln)
    n_prefer = sum(1 for ln in factors if "prefer" in ln)
    n_avoid = sum(1 for ln in factors if "avoid ~" in ln) + len(avoid_items)
    return {
        "has_block": True,
        "n_factor_lines": len(factors),
        "n_prefer": n_prefer,
        "n_avoid": n_avoid,
        "clear_separation": len(factors) > 0,
        "trend": trend,
        "stagnation": stag,
        "incumbent_updates": inc,
    }


def analyze_prefix(successes: List[dict], cs, algo: str) -> Dict[str, Any]:
    n = len(successes)
    cov_stats = compute_coverage_stats(successes, cs, algo)
    mean_cov = mean_coverage_pct(successes, cs, algo) if n else 0.0
    cat = [v["pct"] for v in cov_stats.values() if v["type"] == "categorical"]
    num = [v["pct"] for v in cov_stats.values() if v["type"] == "numeric"]
    stag = since_last_improvement(successes) if n else 0
    explore_phase = n < 15 or stag >= 8 or mean_cov < 40.0
    block = build_experience_self_block(
        successes, mode="marginal_nonoverlap", config_space=cs, min_trials=8
    )
    a5 = parse_a5_stats(block)
    return {
        "n": n,
        "mean_coverage_pct": float(mean_cov),
        "cat_coverage_pct": float(np.mean(cat)) if cat else None,
        "num_coverage_pct": float(np.mean(num)) if num else None,
        "n_hps": len(cov_stats),
        "stagnation": int(stag),
        "explore_refine_says_explore": bool(explore_phase),
        "a5": a5,
        "hp_coverage": {
            k: {"type": v["type"], "pct": float(v["pct"])} for k, v in cov_stats.items()
        },
    }


def load_traj_successes(traj_csv: Path, algo: str, max_n: int) -> List[dict]:
    rows: List[dict] = []
    with traj_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("algorithm") or "") not in ("", algo):
                # allow mismatch only if empty
                if row.get("algorithm") and row.get("algorithm") != algo:
                    continue
            raw = row.get("config_for_prompt") or ""
            if not raw.strip():
                continue
            try:
                cfg = json.loads(raw)
            except json.JSONDecodeError:
                try:
                    cfg = ast.literal_eval(raw)
                except Exception:
                    continue
            if not isinstance(cfg, dict):
                continue
            cfg = dict(cfg)
            cfg.setdefault("algorithm", algo)
            try:
                perf = float(row["perf"])
            except (TypeError, ValueError, KeyError):
                continue
            if not np.isfinite(perf):
                continue
            rows.append({"trial": len(rows) + 1, "config": cfg, "score": perf})
            if len(rows) >= max_n:
                break
    return rows


def find_traj(root: Path, task_type: str, dataset: str, pattern: str) -> Optional[Path]:
    base = root / task_type / dataset
    if not base.exists():
        return None
    cands = sorted(base.rglob("*_traj.csv"))
    # prefer matching pattern in path
    ranked = [p for p in cands if pattern in str(p)] or cands
    return ranked[0] if ranked else None


def main() -> int:
    tasks = json.loads(TASKS_JSON.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_task: List[dict] = []
    curve_rows: List[dict] = []

    for t in tasks:
        tt, ds, algo = t["task_type"], t["dataset"], t["algorithm"]
        metric = t.get("metric") or ("bal_acc" if tt == "cls" else "mse")
        path = bo_history_path(tt, ds, algo, metric=metric)
        obs = load_bo_observations(path, algorithm=algo, max_n=100)
        cfgs, perfs = observations_to_perf_pairs(obs)
        full = to_successes(cfgs, perfs)
        cs = prefixed_cs(tt, algo)
        by_k = {}
        for k in PREFIXES:
            if len(full) < k:
                continue
            stats = analyze_prefix(full[:k], cs, algo)
            by_k[str(k)] = stats
            curve_rows.append(
                {
                    "source": "bo",
                    "task_type": tt,
                    "dataset": ds,
                    "algorithm": algo,
                    "k": k,
                    **{kk: stats[kk] for kk in (
                        "n", "mean_coverage_pct", "cat_coverage_pct", "num_coverage_pct",
                        "stagnation", "explore_refine_says_explore",
                    )},
                    "a5_n_factor_lines": stats["a5"]["n_factor_lines"],
                    "a5_n_prefer": stats["a5"]["n_prefer"],
                    "a5_n_avoid": stats["a5"]["n_avoid"],
                    "a5_clear": stats["a5"]["clear_separation"],
                }
            )

        # Same-dataset LLM/classical trajs often use a *different* algo than pareto32.
        # Compare fairly: take traj algo A, load BO history for (dataset, A), vs traj@50.
        comps = {}
        for label, root, pat in (
            ("llm_p5_s1a", LLM_ROOT, "ptllm_tpe"),
            ("classical_tpe", CLASSICAL_ROOT, "/tpe/"),
        ):
            traj = find_traj(root, tt, ds, pat)
            if traj is None:
                continue
            # peek traj algorithm
            with traj.open(encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                peek = next(rdr, None)
            traj_algo = (peek or {}).get("algorithm") or algo
            try:
                cs_t = prefixed_cs(tt, traj_algo)
            except KeyError:
                continue
            succ50 = load_traj_successes(traj, traj_algo, 50)
            succ80 = load_traj_successes(traj, traj_algo, 80)
            if len(succ50) < 20:
                continue
            # matching BO for same (dataset, traj_algo)
            bo_path_m = bo_history_path(tt, ds, traj_algo, metric=metric)
            bo50_m = None
            if bo_path_m.exists():
                obs_m = load_bo_observations(bo_path_m, algorithm=traj_algo, max_n=50)
                if len(obs_m) >= 40:
                    cfgs_m, perfs_m = observations_to_perf_pairs(obs_m)
                    bo50_m = analyze_prefix(to_successes(cfgs_m, perfs_m)[:50], cs_t, traj_algo)
            comps[label] = {
                "traj": str(traj.relative_to(REPO)),
                "traj_algorithm": traj_algo,
                "train_algorithm": algo,
                "algo_match": traj_algo == algo,
                "k50": analyze_prefix(succ50, cs_t, traj_algo),
                "k80": analyze_prefix(succ80[:80], cs_t, traj_algo) if len(succ80) >= 40 else None,
                "bo50_same_algo": bo50_m,
            }
            st = comps[label]["k50"]
            curve_rows.append(
                {
                    "source": label,
                    "task_type": tt,
                    "dataset": ds,
                    "algorithm": traj_algo,
                    "k": 50,
                    **{kk: st[kk] for kk in (
                        "n", "mean_coverage_pct", "cat_coverage_pct", "num_coverage_pct",
                        "stagnation", "explore_refine_says_explore",
                    )},
                    "a5_n_factor_lines": st["a5"]["n_factor_lines"],
                    "a5_n_prefer": st["a5"]["n_prefer"],
                    "a5_n_avoid": st["a5"]["n_avoid"],
                    "a5_clear": st["a5"]["clear_separation"],
                }
            )
            if bo50_m is not None:
                curve_rows.append(
                    {
                        "source": f"bo_matched_{label}",
                        "task_type": tt,
                        "dataset": ds,
                        "algorithm": traj_algo,
                        "k": 50,
                        **{kk: bo50_m[kk] for kk in (
                            "n", "mean_coverage_pct", "cat_coverage_pct", "num_coverage_pct",
                            "stagnation", "explore_refine_says_explore",
                        )},
                        "a5_n_factor_lines": bo50_m["a5"]["n_factor_lines"],
                        "a5_n_prefer": bo50_m["a5"]["n_prefer"],
                        "a5_n_avoid": bo50_m["a5"]["n_avoid"],
                        "a5_clear": bo50_m["a5"]["clear_separation"],
                    }
                )

        row = {
            "task_type": tt,
            "dataset": ds,
            "algorithm": algo,
            "bo_n": len(full),
            "by_k": by_k,
            "comparisons": comps,
        }
        if "50" in by_k and "100" in by_k:
            row["cov50_vs_100"] = by_k["50"]["mean_coverage_pct"] - by_k["100"]["mean_coverage_pct"]
            row["a5_factors50"] = by_k["50"]["a5"]["n_factor_lines"]
            row["a5_factors100"] = by_k["100"]["a5"]["n_factor_lines"]
        per_task.append(row)
        print(
            f"[ok] {tt}/{ds}/{algo} n={len(full)} "
            f"cov50={by_k.get('50',{}).get('mean_coverage_pct')} "
            f"a5_50={by_k.get('50',{}).get('a5',{}).get('n_factor_lines')} "
            f"comps={list(comps)}"
        )

    # Aggregates
    def mean_at(k: int, field: str, source: str = "bo") -> Optional[float]:
        vals = [
            r[field]
            for r in curve_rows
            if r["source"] == source and r["k"] == k and r.get(field) is not None
        ]
        return float(np.mean(vals)) if vals else None

    summary = {
        "n_tasks": len(per_task),
        "prefixes": PREFIXES,
        "bo_mean_coverage_by_k": {str(k): mean_at(k, "mean_coverage_pct") for k in PREFIXES},
        "bo_mean_a5_factors_by_k": {str(k): mean_at(k, "a5_n_factor_lines") for k in PREFIXES},
        "bo_mean_a5_prefer_by_k": {str(k): mean_at(k, "a5_n_prefer") for k in PREFIXES},
        "bo_mean_a5_avoid_by_k": {str(k): mean_at(k, "a5_n_avoid") for k in PREFIXES},
        "bo_pct_explore_phase_by_k": {
            str(k): mean_at(k, "explore_refine_says_explore") for k in PREFIXES
        },
        "bo_pct_a5_clear_by_k": {
            str(k): float(
                np.mean(
                    [
                        1.0 if r["a5_clear"] else 0.0
                        for r in curve_rows
                        if r["source"] == "bo" and r["k"] == k
                    ]
                )
            )
            for k in PREFIXES
            if any(r["source"] == "bo" and r["k"] == k for r in curve_rows)
        },
        "paired_overlap_tasks": [],
    }

    # Paired BO@50 (same traj algo) vs LLM@50
    paired = []
    for t in per_task:
        c = t["comparisons"].get("llm_p5_s1a")
        if not c or not c.get("bo50_same_algo"):
            continue
        bo50 = c["bo50_same_algo"]
        llm50 = c["k50"]
        paired.append(
            {
                "dataset": t["dataset"],
                "task_type": t["task_type"],
                "traj_algorithm": c["traj_algorithm"],
                "train_algorithm": t["algorithm"],
                "algo_match": c["algo_match"],
                "bo_cov": bo50["mean_coverage_pct"],
                "llm_cov": llm50["mean_coverage_pct"],
                "cov_gap_llm_minus_bo": llm50["mean_coverage_pct"] - bo50["mean_coverage_pct"],
                "bo_a5_factors": bo50["a5"]["n_factor_lines"],
                "llm_a5_factors": llm50["a5"]["n_factor_lines"],
                "bo_a5_prefer": bo50["a5"]["n_prefer"],
                "llm_a5_prefer": llm50["a5"]["n_prefer"],
                "bo_explore_phase": bo50["explore_refine_says_explore"],
                "llm_explore_phase": llm50["explore_refine_says_explore"],
            }
        )
    summary["paired_overlap_tasks"] = paired
    if paired:
        summary["paired_mean_bo_cov"] = float(np.mean([p["bo_cov"] for p in paired]))
        summary["paired_mean_llm_cov"] = float(np.mean([p["llm_cov"] for p in paired]))
        summary["paired_mean_cov_gap"] = float(np.mean([p["cov_gap_llm_minus_bo"] for p in paired]))
        summary["paired_mean_bo_a5"] = float(np.mean([p["bo_a5_factors"] for p in paired]))
        summary["paired_mean_llm_a5"] = float(np.mean([p["llm_a5_factors"] for p in paired]))
        summary["paired_n"] = len(paired)

    paired_tpe = []
    for t in per_task:
        c = t["comparisons"].get("classical_tpe")
        if not c or not c.get("bo50_same_algo"):
            continue
        bo50 = c["bo50_same_algo"]
        tpe50 = c["k50"]
        paired_tpe.append(
            {
                "dataset": t["dataset"],
                "traj_algorithm": c["traj_algorithm"],
                "bo_cov": bo50["mean_coverage_pct"],
                "tpe_cov": tpe50["mean_coverage_pct"],
                "cov_gap_tpe_minus_bo": tpe50["mean_coverage_pct"] - bo50["mean_coverage_pct"],
                "bo_a5_factors": bo50["a5"]["n_factor_lines"],
                "tpe_a5_factors": tpe50["a5"]["n_factor_lines"],
            }
        )
    summary["paired_tpe"] = paired_tpe
    if paired_tpe:
        summary["paired_tpe_n"] = len(paired_tpe)
        summary["paired_mean_tpe_cov"] = float(np.mean([p["tpe_cov"] for p in paired_tpe]))
        summary["paired_mean_tpe_a5"] = float(np.mean([p["tpe_a5_factors"] for p in paired_tpe]))
        summary["paired_mean_tpe_cov_gap"] = float(
            np.mean([p["cov_gap_tpe_minus_bo"] for p in paired_tpe])
        )

    out = {"summary": summary, "tasks": per_task, "curves": curve_rows}
    (OUT_DIR / "bo50_coverage_prefer_avoid.json").write_text(json.dumps(out, indent=2) + "\n")

    import csv as _csv

    with (OUT_DIR / "curves.csv").open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()) if curve_rows else [])
        if curve_rows:
            w.writeheader()
            w.writerows(curve_rows)
    with (OUT_DIR / "paired_llm.csv").open("w", newline="", encoding="utf-8") as f:
        if paired:
            w = _csv.DictWriter(f, fieldnames=list(paired[0].keys()))
            w.writeheader()
            w.writerows(paired)
    with (OUT_DIR / "paired_tpe.csv").open("w", newline="", encoding="utf-8") as f:
        if paired_tpe:
            w = _csv.DictWriter(f, fieldnames=list(paired_tpe[0].keys()))
            w.writeheader()
            w.writerows(paired_tpe)

    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
