#!/usr/bin/env python3
"""Build eval8-style avg_rank_table.csv for mltest20 Option B.

Aggregates classical + P16 from the shared algoB root and each LoRA P5
adapter root (s0 / s1*), using final best_so_far from 80-trial trajs.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

TASKS = [
    ("cls", "yeast"),
    ("cls", "car"),
    ("cls", "PhishingWebsites"),
    ("cls", "segment"),
    ("rgs", "boston"),
    ("rgs", "tecator"),
    ("rgs", "socmob"),
    ("rgs", "pol"),
]

CLS = {"yeast", "car", "PhishingWebsites", "segment"}
# In this benchmark, larger best_so_far is better for both cls and rgs
# (rgs stores negated holdout MSE).

LORA_MODELS = ["s0", "s1a_ep2p5", "s1b_ep2p5", "s1a_ep7p5", "s1b_ep7p5"]


def fnum(v, default=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def find_traj(root: Path, tt: str, ds: str, variant: str, n_trials: int) -> Path | None:
    """Prefer a complete (>= n_trials) CASH run; never score incomplete aborted runs."""
    d = root / tt / ds / variant
    if not d.is_dir():
        return None
    complete: list[tuple[float, int, Path]] = []
    for run_dir in sorted(p for p in d.glob("CASH-*") if p.is_dir()):
        trajs = sorted(run_dir.glob("*_traj.csv"))
        if not trajs:
            continue
        traj = trajs[-1]
        rows = list(csv.DictReader(traj.open(newline="", encoding="utf-8")))
        if len(rows) < n_trials:
            continue
        bests = [fnum(r.get("best_so_far")) for r in rows]
        bests = [b for b in bests if not math.isnan(b)]
        if not bests:
            continue
        # Higher best_so_far is better; tie-break: more trials, then path name.
        complete.append((max(bests), len(rows), traj))
    if not complete:
        return None
    complete.sort(key=lambda x: (x[0], x[1], str(x[2])))
    return complete[-1][2]


def final_best(traj: Path) -> float | None:
    rows = list(csv.DictReader(traj.open(newline="", encoding="utf-8")))
    if not rows:
        return None
    bests = [fnum(r.get("best_so_far")) for r in rows]
    bests = [b for b in bests if not math.isnan(b) and b != float("-inf")]
    if not bests:
        return None
    # Monotonic for higher-is-better; use max to match plot_pilot_sweep.
    return max(bests)


def average_rank(ordered: list[tuple[str, float]]) -> dict[str, float]:
    """Competition/average ranks for ties (same as plot_real_benchmark)."""
    ranks: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and abs(ordered[j + 1][1] - ordered[i][1]) < 1e-12:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[ordered[k][0]] = avg
        i = j + 1
    return ranks


def task_ranks(scores: dict[str, float], ds: str) -> dict[str, float]:
    # Higher best_so_far is always better (rgs uses negated MSE).
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return average_rank(ordered)


def normalize(scores: dict[str, float], ds: str) -> dict[str, float]:
    if len(scores) < 2:
        return {k: 1.0 for k in scores}
    vals = list(scores.values())
    best, worst = max(vals), min(vals)
    return {k: 1.0 if best == worst else (v - worst) / (best - worst) for k, v in scores.items()}


def display_name(variant: str) -> str:
    mapping = {
        "smac": "smac",
        "smac_2": "smac_2",
        "tpe": "tpe",
        "cmaes": "cmaes",
        "ptllm_tpe__P16__H2": "ptllm_tpe (P16/H2 gpt)",
    }
    if variant in mapping:
        return mapping[variant]
    # ptllm_tpe_qwen_s0__P5__H4 -> qwen s0 P5/H4
    if variant.startswith("ptllm_tpe_qwen_"):
        rest = variant.removeprefix("ptllm_tpe_qwen_")
        model, _, hy = rest.partition("__P5__")
        # hy like "H0", "H4", "H0__2", "H4__warm_start"
        tag, _, extra = hy.partition("__")
        name = f"qwen LoRA {model} (P5/{tag} tpe)"
        if extra:
            name += f"_{extra}"
        return name
    if variant.startswith("ptllm_smac_qwen_"):
        rest = variant.removeprefix("ptllm_smac_qwen_")
        model, _, hy = rest.partition("__P5__")
        tag, _, extra = hy.partition("__")
        name = f"qwen LoRA {model} (P5/{tag} smac)"
        if extra:
            name += f"_{extra}"
        return name
    return variant


def family(variant: str) -> str:
    if variant in ("smac", "smac_2", "tpe", "cmaes"):
        return "classical"
    if variant == "ptllm_tpe__P16__H2":
        return "reference_tuned"
    if "ptllm_smac_qwen" in variant:
        return "qwen_lora_p5_smac"
    if "qwen" in variant:
        return "qwen_lora_p5"
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-root",
        type=Path,
        default=REPO / "benchmark_data" / "real_hpo_v2_mltest20_algoB",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "benchmark_data" / "real_hpo_v2_mltest20_algoB" / "analysis" / "avg_rank",
    )
    parser.add_argument("--n-trials", type=int, default=80)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Only include variants with all 8 datasets",
    )
    args = parser.parse_args()
    base = args.base_root
    if not base.is_absolute():
        base = REPO / base
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # label -> (root, disk_variant_name)
    sources: dict[str, tuple[Path, str]] = {
        "smac": (base, "smac"),
        "smac_2": (base, "smac_2"),
        "tpe": (base, "tpe"),
        "cmaes": (base, "cmaes"),
        "ptllm_tpe__P16__H2": (base, "ptllm_tpe__P16__H2"),
    }
    for model in LORA_MODELS:
        root = base.parent / f"{base.name}_p5_{model}"
        for hy in ("H0", "H4"):
            key = f"ptllm_tpe_qwen_{model}__P5__{hy}"
            sources[key] = (root, f"ptllm_tpe_qwen__P5__{hy}")
        # S1 adapters also have ptllm_smac P5/H0 (BO backend = SMAC).
        if model != "s0":
            key = f"ptllm_smac_qwen_{model}__P5__H0"
            sources[key] = (root, "ptllm_smac_qwen__P5__H0")
            # Optional seed=2 replicate (disk: ...__P5__H0__2).
            if model == "s1b_ep7p5":
                key2 = f"ptllm_smac_qwen_{model}__P5__H0__2"
                sources[key2] = (root, "ptllm_smac_qwen__P5__H0__2")
                # recommended4 best-of-3 warm-start ablations
                sources[f"ptllm_smac_qwen_{model}__P5__H0__warm_start"] = (
                    root,
                    "ptllm_smac_qwen__P5__H0__warm_start",
                )
                sources[f"ptllm_tpe_qwen_{model}__P5__H4__warm_start"] = (
                    root,
                    "ptllm_tpe_qwen__P5__H4__warm_start",
                )
            if model == "s1b_ep2p5":
                sources[f"ptllm_tpe_qwen_{model}__P5__H4__warm_start"] = (
                    root,
                    "ptllm_tpe_qwen__P5__H4__warm_start",
                )
                sources[f"ptllm_tpe_qwen_{model}__P5__H0__warm_start"] = (
                    root,
                    "ptllm_tpe_qwen__P5__H0__warm_start",
                )
                sources[f"ptllm_smac_qwen_{model}__P5__H0__warm_start"] = (
                    root,
                    "ptllm_smac_qwen__P5__H0__warm_start",
                )

    # summary[(tt/ds, variant)] = best
    summary: dict[tuple[str, str], float] = {}
    coverage: dict[str, int] = {v: 0 for v in sources}

    for label, (root, disk_var) in sources.items():
        for tt, ds in TASKS:
            traj = find_traj(root, tt, ds, disk_var, args.n_trials)
            if traj is None:
                continue
            best = final_best(traj)
            if best is None or math.isnan(best) or best == float("-inf"):
                continue
            summary[(f"{tt}/{ds}", label)] = best
            coverage[label] += 1

    variants = list(sources.keys())
    if args.require_complete:
        variants = [v for v in variants if coverage[v] >= 8]
    else:
        variants = [v for v in variants if coverage[v] >= 1]

    dataset_keys = [f"{tt}/{ds}" for tt, ds in TASKS]
    # per-dataset ranks
    per_ds_ranks: dict[str, dict[str, float]] = {}
    per_ds_norm: dict[str, dict[str, float]] = {}
    for tt, ds in TASKS:
        key = f"{tt}/{ds}"
        scores = {v: summary[(key, v)] for v in variants if (key, v) in summary}
        if len(scores) < 2:
            continue
        per_ds_ranks[key] = task_ranks(scores, ds)
        per_ds_norm[key] = normalize(scores, ds)

    avg_rank = {}
    avg_rank_cls = {}
    avg_rank_rgs = {}
    mean_cls_norm = {}
    mean_rgs_norm = {}
    for v in variants:
        ranks_all = [per_ds_ranks[k][v] for k in per_ds_ranks if v in per_ds_ranks[k]]
        ranks_cls = [
            per_ds_ranks[k][v]
            for k in per_ds_ranks
            if v in per_ds_ranks[k] and k.split("/")[1] in CLS
        ]
        ranks_rgs = [
            per_ds_ranks[k][v]
            for k in per_ds_ranks
            if v in per_ds_ranks[k] and k.split("/")[1] not in CLS
        ]
        avg_rank[v] = sum(ranks_all) / len(ranks_all) if ranks_all else float("nan")
        avg_rank_cls[v] = sum(ranks_cls) / len(ranks_cls) if ranks_cls else float("nan")
        avg_rank_rgs[v] = sum(ranks_rgs) / len(ranks_rgs) if ranks_rgs else float("nan")
        cn = [per_ds_norm[k][v] for k in per_ds_norm if v in per_ds_norm[k] and k.split("/")[1] in CLS]
        rn = [per_ds_norm[k][v] for k in per_ds_norm if v in per_ds_norm[k] and k.split("/")[1] not in CLS]
        mean_cls_norm[v] = sum(cn) / len(cn) if cn else float("nan")
        mean_rgs_norm[v] = sum(rn) / len(rn) if rn else float("nan")

    rows = []
    for v in variants:
        cn, rn = mean_cls_norm[v], mean_rgs_norm[v]
        bal = (cn + rn) / 2 if not (math.isnan(cn) or math.isnan(rn)) else float("nan")
        rows.append(
            {
                "display_name": display_name(v),
                "variant": v,
                "family": family(v),
                "datasets": coverage[v],
                "avg_rank": avg_rank[v],
                "avg_rank_cls": avg_rank_cls[v],
                "avg_rank_rgs": avg_rank_rgs[v],
                "mean_cls_norm": cn,
                "mean_rgs_norm": rn,
                "mean_balanced_norm": bal,
            }
        )
    rows.sort(key=lambda r: (math.isnan(r["avg_rank"]), r["avg_rank"]))

    path = out_dir / "avg_rank_table.csv"
    fields = [
        "rank",
        "display_name",
        "variant",
        "family",
        "datasets",
        "avg_rank",
        "avg_rank_cls",
        "avg_rank_rgs",
        "mean_cls_norm",
        "mean_rgs_norm",
        "mean_balanced_norm",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, row in enumerate(rows, start=1):
            out = {"rank": i, "display_name": row["display_name"], "variant": row["variant"], "family": row["family"], "datasets": row["datasets"]}
            for k in ("avg_rank", "avg_rank_cls", "avg_rank_rgs"):
                out[k] = f"{row[k]:.2f}" if not math.isnan(row[k]) else ""
            for k in ("mean_cls_norm", "mean_rgs_norm", "mean_balanced_norm"):
                out[k] = f"{row[k]:.6f}" if not math.isnan(row[k]) else ""
            w.writerow(out)

    # Second table: drop all P5/H4 variants, recompute norms on the reduced pool,
    # and rank by mean_balanced_norm (higher is better).
    no_h4 = [v for v in variants if "__H4" not in v and not v.endswith("/H4")]
    per_ds_ranks_h0: dict[str, dict[str, float]] = {}
    per_ds_norm_h0: dict[str, dict[str, float]] = {}
    for tt, ds in TASKS:
        key = f"{tt}/{ds}"
        scores = {v: summary[(key, v)] for v in no_h4 if (key, v) in summary}
        if len(scores) < 2:
            continue
        per_ds_ranks_h0[key] = task_ranks(scores, ds)
        per_ds_norm_h0[key] = normalize(scores, ds)

    rows_h0 = []
    for v in no_h4:
        ranks_all = [per_ds_ranks_h0[k][v] for k in per_ds_ranks_h0 if v in per_ds_ranks_h0[k]]
        ranks_cls = [
            per_ds_ranks_h0[k][v]
            for k in per_ds_ranks_h0
            if v in per_ds_ranks_h0[k] and k.split("/")[1] in CLS
        ]
        ranks_rgs = [
            per_ds_ranks_h0[k][v]
            for k in per_ds_ranks_h0
            if v in per_ds_ranks_h0[k] and k.split("/")[1] not in CLS
        ]
        cn = [
            per_ds_norm_h0[k][v]
            for k in per_ds_norm_h0
            if v in per_ds_norm_h0[k] and k.split("/")[1] in CLS
        ]
        rn = [
            per_ds_norm_h0[k][v]
            for k in per_ds_norm_h0
            if v in per_ds_norm_h0[k] and k.split("/")[1] not in CLS
        ]
        mean_cn = sum(cn) / len(cn) if cn else float("nan")
        mean_rn = sum(rn) / len(rn) if rn else float("nan")
        bal = (
            (mean_cn + mean_rn) / 2
            if not (math.isnan(mean_cn) or math.isnan(mean_rn))
            else float("nan")
        )
        rows_h0.append(
            {
                "display_name": display_name(v),
                "variant": v,
                "family": family(v),
                "datasets": coverage[v],
                "avg_rank": sum(ranks_all) / len(ranks_all) if ranks_all else float("nan"),
                "avg_rank_cls": sum(ranks_cls) / len(ranks_cls) if ranks_cls else float("nan"),
                "avg_rank_rgs": sum(ranks_rgs) / len(ranks_rgs) if ranks_rgs else float("nan"),
                "mean_cls_norm": mean_cn,
                "mean_rgs_norm": mean_rn,
                "mean_balanced_norm": bal,
            }
        )
    rows_h0.sort(
        key=lambda r: (
            math.isnan(r["mean_balanced_norm"]),
            -r["mean_balanced_norm"] if not math.isnan(r["mean_balanced_norm"]) else 0.0,
        )
    )

    path_h0 = out_dir / "mean_norm_table_no_h4.csv"
    with path_h0.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, row in enumerate(rows_h0, start=1):
            out = {
                "rank": i,
                "display_name": row["display_name"],
                "variant": row["variant"],
                "family": row["family"],
                "datasets": row["datasets"],
            }
            for k in ("avg_rank", "avg_rank_cls", "avg_rank_rgs"):
                out[k] = f"{row[k]:.2f}" if not math.isnan(row[k]) else ""
            for k in ("mean_cls_norm", "mean_rgs_norm", "mean_balanced_norm"):
                out[k] = f"{row[k]:.6f}" if not math.isnan(row[k]) else ""
            w.writerow(out)

    # also write per-dataset best matrix
    matrix_path = out_dir / "valid_best_by_dataset.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as f:
        cols = ["variant", "display_name"] + [f"{tt}/{ds}" for tt, ds in TASKS] + ["coverage"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            v = row["variant"]
            line = {"variant": v, "display_name": row["display_name"], "coverage": coverage[v]}
            for tt, ds in TASKS:
                key = f"{tt}/{ds}"
                if (key, v) in summary:
                    line[key] = f"{summary[(key, v)]:.6f}"
                else:
                    line[key] = ""
            w.writerow(line)

    print(f"Wrote {path}")
    print(f"Wrote {path_h0} (no H4; ranked by mean_balanced_norm)")
    print(f"Wrote {matrix_path}")
    print(f"Variants included: {len(rows)} (8/8 complete: {sum(1 for v in variants if coverage[v] >= 8)})")
    for i, row in enumerate(rows[:15], start=1):
        print(f"  {i:2d}. {row['display_name']:40s} avg_rank={row['avg_rank']:.2f} ds={coverage[row['variant']]}/8")
    print("No-H4 / mean_balanced_norm top:")
    for i, row in enumerate(rows_h0[:10], start=1):
        print(
            f"  {i:2d}. {row['display_name']:40s} "
            f"mean_bal_norm={row['mean_balanced_norm']:.4f} ds={coverage[row['variant']]}/8"
        )
    pending = [v for v in sources if coverage[v] < 8]
    if pending:
        print("Incomplete:")
        for v in pending:
            print(f"  {display_name(v)}: {coverage[v]}/8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
