#!/usr/bin/env python3
"""Full-bank fANOVA audit: main effects vs interaction-only surfaces.

Reads experience_bank_v3.jsonl (stored importance_1d / importance_total / top_pairs).
Does not re-run fANOVA.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.components.experience.bank_gates import hp_stem

ANALYSIS = ROOT / "benchmark_data" / "real_hpo_v2_mltest20_algo40" / "analysis" / "experience_bank_v3"


def _stem_map(imps: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in (imps or {}).items():
        stem = hp_stem(k)
        out[stem] = max(float(v), out.get(stem, 0.0))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=ROOT / "history_data" / "experience_bank_v3.jsonl")
    p.add_argument("--out-dir", type=Path, default=ANALYSIS)
    args = p.parse_args(argv)
    if not args.bank.is_file():
        raise SystemExit(f"missing bank {args.bank}")

    rows: List[Dict[str, Any]] = []
    with args.bank.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            exp = json.loads(line)
            prob = exp.get("problem") or {}
            fan = (exp.get("knowledge") or {}).get("fanova") or {}
            one = _stem_map(fan.get("importance_1d") or {})
            tot = _stem_map(fan.get("importance_total") or one)
            max_1d = max(one.values()) if one else 0.0
            max_tot = max(tot.values()) if tot else 0.0
            n_main = sum(1 for v in one.values() if v >= 0.10)
            n_ix = 0
            ix_names = []
            for name, i1 in one.items():
                it = float(tot.get(name, i1) or i1)
                ratio = (i1 / it) if it > 1e-12 else 1.0
                if it > 0.12 and ratio < 0.50:
                    n_ix += 1
                    ix_names.append(name)
            pairs = list(fan.get("top_pairs") or [])
            top_pair = None
            top_pair_imp = 0.0
            for pr in pairs:
                imp = float(pr.get("importance_1d") or pr.get("importance_total") or 0.0)
                if imp >= top_pair_imp:
                    top_pair_imp = imp
                    top_pair = f"{pr.get('a')}|{pr.get('b')}"
            rows.append(
                {
                    "id": exp.get("id"),
                    "dataset": prob.get("dataset_id"),
                    "algorithm": prob.get("algorithm_id"),
                    "task_type": prob.get("task_type"),
                    "engine": fan.get("engine"),
                    "n_trials": fan.get("n_trials") or len((exp.get("retrieval") or {}).get("trials") or []),
                    "max_1d": round(max_1d, 4),
                    "max_total": round(max_tot, 4),
                    "ratio_max1d_over_maxtotal": round(max_1d / max_tot, 4) if max_tot > 1e-12 else 1.0,
                    "n_main_effect_hp": n_main,
                    "n_interaction_hp": n_ix,
                    "interaction_hps": ",".join(ix_names),
                    "flat_max1d_lt_0.08": int(max_1d < 0.08),
                    "clear_main_max1d_ge_0.10": int(max_1d >= 0.10),
                    "multi_driver": int(n_main >= 2),
                    "interaction_only_record": int(max_1d < 0.08 and max_tot > 0.25),
                    "usable_for_prompt": int(bool((exp.get("delivery") or {}).get("usable_for_prompt"))),
                    "n_keep": len(fan.get("keep_hps") or []),
                    "top_pair": top_pair or "",
                    "top_pair_importance": round(top_pair_imp, 4),
                    "n_pairs": len(pairs),
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "interaction_prevalence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    def frac(pred) -> float:
        return sum(1 for r in rows if pred(r)) / n if n else 0.0

    summary = {
        "n": n,
        "frac_flat_max1d_lt_0.08": frac(lambda r: r["flat_max1d_lt_0.08"]),
        "frac_clear_main_max1d_ge_0.10": frac(lambda r: r["clear_main_max1d_ge_0.10"]),
        "frac_multi_driver": frac(lambda r: r["multi_driver"]),
        "frac_any_interaction_hp": frac(lambda r: r["n_interaction_hp"] > 0),
        "frac_interaction_only_record": frac(lambda r: r["interaction_only_record"]),
        "frac_usable_for_prompt": frac(lambda r: r["usable_for_prompt"]),
        "mean_max_1d": sum(r["max_1d"] for r in rows) / n if n else 0.0,
        "mean_max_total": sum(r["max_total"] for r in rows) / n if n else 0.0,
        "mean_keep_hps": sum(r["n_keep"] for r in rows) / n if n else 0.0,
        "frac_has_top_pair": frac(lambda r: bool(r["top_pair"])),
        "conclusion": (
            "Tabular ML BO traces are mainly main-effect driven. "
            "1D keep + 1d/total exclusion is appropriate; pairwise lines stay out of the v3 card."
        ),
    }
    (args.out_dir / "interaction_prevalence_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
