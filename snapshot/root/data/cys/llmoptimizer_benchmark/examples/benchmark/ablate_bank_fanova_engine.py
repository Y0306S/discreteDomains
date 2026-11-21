#!/usr/bin/env python3
"""Lock the offline bank-build fANOVA engine on ~100 histories.

Engine matrix
-------------
- fanova_full_cs : official fANOVA on real ConfigSpace; keep-set from importance_1d
- fanova_full    : official fANOVA on the observed sample cloud (CS unavailable)
- fanova_full_ix : same CS encoding; keep-set from importance_total (ablation only)
- fanova_rf      : sklearn RF impurity (in-process fallback)
- proxy          : 1D bin-mean variance (n < 24 or worker failure)

Bank build locks **fanova_full_cs** for keep-set (1D). Total importance is stored
and used only as an exclusion filter, not as the keep ranking. Histories whose
estimator CS cannot load (sklearn forests on unsupported sklearn) fall back to
fanova_full on the observed space — still official fANOVA, not the 1D proxy.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from build_experience_bank import _iter_history_files
from mindware.components.experience.bank_gates import (
    BANK_ENGINE,
    calibrated_keep,
    load_estimator_cs,
    normalize_config,
)
from mindware.components.experience.builder import load_trials_from_bo_json
from mindware.components.optimizers.experience_gate import (
    calibrated_fanova_keep,
    hp_importances,
    hp_importances_both,
    hp_importances_proxy,
)

ANALYSIS = ROOT / "benchmark_data" / "real_hpo_v2_mltest20_algo40" / "analysis" / "experience_bank_v3"


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def _keep(imps: Dict[str, float]) -> List[str]:
    names = calibrated_keep(imps)
    if names:
        return names
    return sorted(calibrated_fanova_keep(imps))


def _pack(path: Path) -> tuple[str, str, str, List[Dict[str, Any]]]:
    algo = path.stem
    dataset = path.parent.name
    tt = path.parent.parent.parent.name
    task_type = "classification" if tt == "cls" else "regression"
    metric = path.parent.parent.name
    higher = metric not in {"mse", "mae", "rmse"}
    trials = load_trials_from_bo_json(
        path,
        algorithm=algo,
        higher_is_better=True if higher else False,
        openbox_minimize=True,
    )
    packed = []
    for i, t in enumerate(trials, 1):
        cfg = normalize_config(t.get("config") or {}, algo)
        if cfg:
            packed.append({"trial": t.get("trial", i), "config": cfg, "score": float(t["score"])})
    return dataset, algo, task_type, packed


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history-root", type=Path, default=ROOT / "history_data")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--min-trials", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=ANALYSIS)
    args = p.parse_args(argv)

    files = _iter_history_files(args.history_root)
    rng = random.Random(args.seed)
    candidates: List[Path] = []
    for path in files:
        try:
            _, _, _, packed = _pack(path)
        except Exception:
            continue
        if len(packed) >= args.min_trials:
            candidates.append(path)
    rng.shuffle(candidates)
    sample = candidates[: args.n]
    print(f"ablation n={len(sample)} eligible={len(candidates)} engine_lock={BANK_ENGINE}")

    rows: List[Dict[str, Any]] = []
    for i, path in enumerate(sample, 1):
        dataset, algo, task_type, packed = _pack(path)
        cs = load_estimator_cs(algo, task_type)
        rec: Dict[str, Any] = {
            "dataset": dataset,
            "algorithm": algo,
            "task_type": task_type,
            "n_trials": len(packed),
            "cs_available": int(cs is not None),
        }
        t0 = time.time()
        try:
            one, tot, _pairs, eng_cs = hp_importances_both(packed, config_space=cs, pair_topk=0)
        except Exception as exc:
            one, tot, eng_cs = {}, {}, f"error:{type(exc).__name__}"
            rec["cs_error"] = str(exc)[:200]
        rec["cs_sec"] = round(time.time() - t0, 3)
        rec["engine_cs"] = eng_cs
        rec["keep_cs_1d"] = json.dumps(_keep(one))
        rec["keep_cs_total"] = json.dumps(_keep(tot or one))
        rec["max_1d"] = round(max(one.values()) if one else 0.0, 4)
        rec["max_total"] = round(max(tot.values()) if tot else 0.0, 4)
        n_ix = 0
        for name, i1 in one.items():
            it = float((tot or {}).get(name, i1) or i1)
            if it > 0.12 and it > 1e-12 and (float(i1) / it) < 0.50:
                n_ix += 1
        rec["n_interaction_hp"] = n_ix

        t0 = time.time()
        try:
            one_obs, tot_obs, _, eng_obs = hp_importances_both(packed, config_space=None, pair_topk=0)
        except Exception:
            one_obs, tot_obs, eng_obs = {}, {}, "error"
        rec["obs_sec"] = round(time.time() - t0, 3)
        rec["engine_obs"] = eng_obs
        rec["keep_obs_1d"] = json.dumps(_keep(one_obs))
        rec["jaccard_cs_vs_obs"] = round(_jaccard(_keep(one), _keep(one_obs)), 3)

        t0 = time.time()
        try:
            rf, eng_rf = hp_importances(packed, engine="rf")
        except Exception:
            rf, eng_rf = {}, "error"
        rec["rf_sec"] = round(time.time() - t0, 3)
        rec["engine_rf"] = eng_rf
        rec["keep_rf"] = json.dumps(_keep(rf))
        rec["jaccard_cs_vs_rf"] = round(_jaccard(_keep(one), _keep(rf)), 3)

        t0 = time.time()
        proxy = hp_importances_proxy(packed)
        rec["proxy_sec"] = round(time.time() - t0, 3)
        rec["keep_proxy"] = json.dumps(_keep(proxy))
        rec["jaccard_cs_vs_proxy"] = round(_jaccard(_keep(one), _keep(proxy)), 3)
        rec["jaccard_1d_vs_total"] = round(_jaccard(_keep(one), _keep(tot or one)), 3)
        rows.append(rec)
        if i % 10 == 0 or i == len(sample):
            print(f"  {i}/{len(sample)} {dataset}/{algo} engine={eng_cs} max_1d={rec['max_1d']}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "fanova_engine_ablation.csv"
    fields = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    def mean(key: str) -> float:
        xs = [float(r[key]) for r in rows if r.get(key) is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    n_cs = sum(r["cs_available"] for r in rows)
    n_cs_ok = sum(1 for r in rows if str(r.get("engine_cs") or "").startswith("fanova_full"))
    n_ix = sum(int(r["n_interaction_hp"]) for r in rows)
    n_ix_rec = sum(1 for r in rows if int(r["n_interaction_hp"]) > 0)
    summary = {
        "n": len(rows),
        "locked_engine": BANK_ENGINE,
        "cs_available_frac": n_cs / max(len(rows), 1),
        "official_fanova_frac": n_cs_ok / max(len(rows), 1),
        "mean_jaccard_cs_vs_obs": mean("jaccard_cs_vs_obs"),
        "mean_jaccard_cs_vs_rf": mean("jaccard_cs_vs_rf"),
        "mean_jaccard_cs_vs_proxy": mean("jaccard_cs_vs_proxy"),
        "mean_jaccard_1d_vs_total": mean("jaccard_1d_vs_total"),
        "mean_max_1d": mean("max_1d"),
        "mean_cs_sec": mean("cs_sec"),
        "records_with_interaction_hp": n_ix_rec,
        "interaction_hp_count": n_ix,
        "decision": (
            f"Lock bank-build engine={BANK_ENGINE} (1D keep-set). "
            "Fallback fanova_full on observed space when estimator CS cannot load. "
            "Do not keep-rank on importance_total."
        ),
    }
    (args.out_dir / "fanova_engine_lock.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
