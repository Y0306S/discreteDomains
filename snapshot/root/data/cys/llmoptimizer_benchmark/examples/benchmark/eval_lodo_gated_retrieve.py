#!/usr/bin/env python3
"""Final LODO calibration: winning gates vs always-k=3.

Adopt gated-k only if mean_rho_top1 AND frac_rho>=0.3 both beat always-k=3.
Writes return-count histogram, usable coverage, and noise vs genuine buckets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from eval_experience_gate_sweep import (
    DEFAULT_GATE_CONFIG,
    GENUINE_RHO,
    eval_config,
    precompute_lodo,
)
from mindware.components.experience.bank_gates import apply_gates, gate_view, merge_gate_config
from mindware.components.experience.retrieve import DEFAULT_WEIGHTS, ExperienceBank

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


def _finite(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return not math.isnan(v)


def load_gate_config(path: Path) -> Dict[str, Any]:
    if path.is_file():
        return merge_gate_config(json.loads(path.read_text(encoding="utf-8")))
    extra = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return merge_gate_config(extra)


def bucket_rows(
    queries: List[Dict[str, Any]],
    views: Dict[str, Dict[str, Any]],
    cfg: Dict[str, Any],
    by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    from eval_experience_gate_sweep import retrieve_ids

    out = []
    for q in queries:
        ids = retrieve_ids(q["scored"], views, cfg)
        grade = {row["id"]: row["grade_rho"] for row in q["scored"]}
        for sid in ids:
            view = views.get(sid) or {}
            rho = float(grade.get(sid) or 0.0)
            max_1d = float(view.get("max_1d") or 0.0)
            if max_1d >= 0.25:
                b1 = "max1d>=0.25"
            elif max_1d >= 0.10:
                b1 = "max1d_0.10-0.25"
            else:
                b1 = "max1d<0.10"
            out.append(
                {
                    "target_id": q["target_id"],
                    "dataset": q["dataset"],
                    "algorithm": q["algorithm"],
                    "task_type": q["task_type"],
                    "source_id": sid,
                    "grade_rho": rho,
                    "genuine": int(rho >= GENUINE_RHO),
                    "negative": int(rho < 0.0),
                    "max_1d_bucket": b1,
                    "usable": int(bool(view.get("usable_for_prompt"))),
                    "n_keep": int(view.get("n_keep") or 0),
                    "n_returned": len(ids),
                }
            )
        if not ids:
            out.append(
                {
                    "target_id": q["target_id"],
                    "dataset": q["dataset"],
                    "algorithm": q["algorithm"],
                    "task_type": q["task_type"],
                    "source_id": "",
                    "grade_rho": "",
                    "genuine": 0,
                    "negative": 0,
                    "max_1d_bucket": "empty",
                    "usable": 0,
                    "n_keep": 0,
                    "n_returned": 0,
                }
            )
    _ = by_id
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=ROOT / "history_data" / "experience_bank_v3.jsonl")
    p.add_argument("--v2-bank", type=Path, default=ROOT / "history_data" / "experience_bank_v2.jsonl")
    p.add_argument("--gate-config", type=Path, default=ANALYSIS / "gate_config.json")
    p.add_argument("--out-dir", type=Path, default=ANALYSIS)
    p.add_argument(
        "--apply-to-bank",
        action="store_true",
        default=True,
        help="Replay winning gates onto v3 jsonl + holdout40.",
    )
    p.add_argument("--no-apply-to-bank", action="store_false", dest="apply_to_bank")
    args = p.parse_args(argv)

    cfg = dict(DEFAULT_GATE_CONFIG)
    cfg.update({"k_max": 3, "require_usable": True, "spearman_floor": 0.0, "dedup": True})
    if args.gate_config.is_file():
        loaded = json.loads(args.gate_config.read_text(encoding="utf-8"))
        cfg = merge_gate_config(loaded)
        for k in ("k_max", "require_usable", "spearman_floor", "dedup"):
            if k in loaded:
                cfg[k] = loaded[k]
    print(f"gate_config={json.dumps(cfg)}")

    split = load_split()
    bank = ExperienceBank.load(args.bank)
    warm_w = dict(bank.weights) if bank.weights else dict(DEFAULT_WEIGHTS)
    queries = precompute_lodo(bank, split=split, warm_w=warm_w)
    by_id = {str(e.get("id")): e for e in bank.experiences}
    print(f"v3 queries={len(queries)}")

    gated = eval_config(queries, by_id, cfg)
    always = eval_config(queries, by_id, cfg, force_k=3, ignore_usable=True)
    views = {eid: gate_view(exp, cfg) for eid, exp in by_id.items()}
    neighbors = bucket_rows(queries, views, cfg, by_id)

    adopt = (
        _finite(gated.get("mean_rho_top1"))
        and _finite(always.get("mean_rho_top1"))
        and float(gated["mean_rho_top1"]) > float(always["mean_rho_top1"])
        and float(gated["frac_rho_ge_0.3"]) > float(always["frac_rho_ge_0.3"])
    )
    if not adopt:
        # Keep v3 cards, but force k_max=3 without dropping unusable at retrieve time.
        live = dict(cfg)
        live["k_max"] = 3
        live["require_usable"] = False
        policy = "always_k3"
    else:
        live = dict(cfg)
        policy = "gated_k"

    hist = {
        "0": int(gated.get("hist_0") or 0),
        "1": int(gated.get("hist_1") or 0),
        "2": int(gated.get("hist_2") or 0),
        "3": int(gated.get("hist_3") or 0),
    }
    by_algo: Dict[str, Counter] = defaultdict(Counter)
    from eval_experience_gate_sweep import retrieve_ids

    for q in queries:
        n = len(retrieve_ids(q["scored"], views, cfg))
        by_algo[q["algorithm"]][n] += 1

    v2_metrics = None
    if args.v2_bank.is_file():
        v2 = ExperienceBank.load(args.v2_bank)
        v2w = dict(v2.weights) if v2.weights else dict(DEFAULT_WEIGHTS)
        v2q = precompute_lodo(v2, split=split, warm_w=v2w)
        v2_by = {str(e.get("id")): e for e in v2.experiences}
        v2_cfg = dict(DEFAULT_GATE_CONFIG)
        v2_cfg.update({"k_max": 3, "require_usable": False, "spearman_floor": 0.0, "dedup": False})
        v2_views = {
            eid: {
                "keep_hps": [],
                "usable_for_prompt": True,
                "prompt_block": str((exp.get("delivery") or {}).get("prompt_block") or eid),
                "n_keep": 3,
                "max_1d": 0.0,
                "max_total": 0.0,
            }
            for eid, exp in v2_by.items()
        }
        v2_metrics = eval_config(v2q, v2_by, v2_cfg, force_k=3, ignore_usable=True, views=v2_views)

    report = {
        "n_queries": len(queries),
        "gated": gated,
        "always_k3": always,
        "v2_always_k3": v2_metrics,
        "adopt_gated_k": adopt,
        "live_policy": policy,
        "live_config": live,
        "return_count_histogram": hist,
        "pct_bank_usable": gated.get("pct_bank_usable"),
        "return_count_by_algorithm": {a: dict(c) for a, c in sorted(by_algo.items())},
        "noise_vs_genuine": {
            "n_returned_neighbors": sum(1 for r in neighbors if r.get("source_id")),
            "frac_genuine_rho_ge_0.3": (
                sum(int(r["genuine"]) for r in neighbors if r.get("source_id"))
                / max(sum(1 for r in neighbors if r.get("source_id")), 1)
            ),
            "frac_negative": (
                sum(int(r["negative"]) for r in neighbors if r.get("source_id"))
                / max(sum(1 for r in neighbors if r.get("source_id")), 1)
            ),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "lodo_gated_vs_always3.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    npath = args.out_dir / "lodo_neighbor_buckets.csv"
    if neighbors:
        with npath.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(neighbors[0].keys()))
            w.writeheader()
            w.writerows(neighbors)

    rec_path = args.out_dir / "gate_config.json"
    rec_path.write_text(json.dumps(live, indent=2) + "\n")
    pkg = ROOT / "mindware" / "components" / "experience" / "gate_config.json"
    pkg.write_text(json.dumps(live, indent=2) + "\n")

    if args.apply_to_bank and args.bank.is_file():
        rewritten = []
        with args.bank.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                exp = json.loads(line)
                apply_gates(exp, live)
                rewritten.append(exp)
        with args.bank.open("w", encoding="utf-8") as f:
            for exp in rewritten:
                f.write(json.dumps(exp, ensure_ascii=False) + "\n")
        hold_path = ROOT / "history_data" / "experience_bank_v3_holdout40.jsonl"
        tasks = json.loads(
            (ROOT / "benchmark_data" / "real_hpo_v2_mltest20_algo40" / "tasks_40.json").read_text()
        )
        eval_ds = {str(t["dataset"]).lower() for t in tasks}
        hold = [e for e in rewritten if _ds(e) not in eval_ds]
        with hold_path.open("w", encoding="utf-8") as f:
            for exp in hold:
                f.write(json.dumps(exp, ensure_ascii=False) + "\n")
        print(f"rewrote {args.bank} n={len(rewritten)} holdout={len(hold)}")

    print(json.dumps({k: report[k] for k in ("adopt_gated_k", "live_policy", "gated", "always_k3")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
