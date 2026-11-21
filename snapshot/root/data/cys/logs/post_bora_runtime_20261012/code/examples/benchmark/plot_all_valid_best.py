"""Master valid_best table across every optimizer variant found under base-dir (eval8).

Discovers all variant directory names under cls/* and rgs/*, collects trajectory
best-so-far per dataset, and writes valid_best_table.csv with avg_rank.

Example:
  python plot_all_valid_best.py --base-dir benchmark_data/real_hpo_v2
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

from experiment_utils import EVAL8_DATASETS, jobs_from_dataset_names, load_hybrid_variants, parse_variant_tags, score_higher_is_better
from plot_baseline_vs_tuned import append_summary_rows, variant_display_name
from plot_pilot_sweep import (
    _ranks_for_task,
    balanced_meta_score,
    collect_sweep,
    task_normalized_means,
    write_valid_best_table,
)

# Classical + LLM zero-shot references (always listed first when present).
CANONICAL_BASELINES = [
    "smac",
    "tpe",
    "cmaes",
    "ptllm__P0",
    "ptllm_tpe__P0__H0",
]

# Common tuned reference points used across eval8 analysis scripts.
REFERENCE_TUNED = [
    "ptllm__P5",
    "ptllm_tpe__P5__H2",
    "ptllm_tpe__P16__H2",
]

# Replication / external runs: (output variant key, source base dir, source variant dir name).
REPLICATION_VARIANTS: list[tuple[str, str, str]] = [
    ("ptllm_tpe__P16__H2_2", "benchmark_data/real_hpo_v2_p16_rep2", "ptllm_tpe__P16__H2"),
    # P5 SFT LoRA (Qwen3.5-9B) on eval8 — separate output roots from gpt-4o-mini real_hpo_v2.
    ("ptllm_tpe_qwen_s1a__P5__H0", "benchmark_data/real_hpo_v2_p5_sft_s1a", "ptllm_tpe_qwen__P5__H0"),
    ("ptllm_tpe_qwen_s1a__P5__H4", "benchmark_data/real_hpo_v2_p5_sft_s1a", "ptllm_tpe_qwen__P5__H4"),
    ("ptllm_tpe_qwen_s1b__P5__H0", "benchmark_data/real_hpo_v2_p5_sft_s1b", "ptllm_tpe_qwen__P5__H0"),
    ("ptllm_tpe_qwen_s1b__P5__H4", "benchmark_data/real_hpo_v2_p5_sft_s1b", "ptllm_tpe_qwen__P5__H4"),
    # P5/H4 json_only ablation (gpt-4o-mini, stricter parser) vs rationale_json in real_hpo_v2.
    (
        "ptllm_tpe__P5__H4__json_only",
        "benchmark_data/real_hpo_v2_p5h4_json_only",
        "ptllm_tpe__P5__H4__json_only",
    ),
]

PINNED_VARIANTS = CANONICAL_BASELINES + REFERENCE_TUNED + [v[0] for v in REPLICATION_VARIANTS]

PROMPT_ORDER = {f"P{i}": i for i in range(100)}


def _variant_sort_key(name: str) -> tuple:
    rep_suffix = 0
    ablation_suffix = 0
    if name.endswith("__json_only"):
        ablation_suffix = 1
        name = name.removesuffix("__json_only")
    if name.endswith("_2") and name.startswith("ptllm"):
        rep_suffix = 1
        name = name.removesuffix("_2")
    classical = {"smac": 0, "tpe": 1, "cmaes": 2}
    if name in classical:
        return (0, classical[name], "", "", rep_suffix, ablation_suffix)
    if name.startswith("ptllm__"):
        pid = name.split("__", 1)[1]
        return (1, PROMPT_ORDER.get(pid, 99), pid, "", rep_suffix, ablation_suffix)
    if name.startswith("ptllm_tpe_qwen_"):
        parts = name.split("__")
        pid = parts[1] if len(parts) > 1 else ""
        hid = parts[2] if len(parts) > 2 else ""
        tag = parts[0].removeprefix("ptllm_tpe_qwen_")  # s1a / s1b
        return (2, PROMPT_ORDER.get(pid, 99), hid, tag, rep_suffix, ablation_suffix)
    if name.startswith("ptllm_tpe__"):
        parts = name.split("__")
        pid = parts[1] if len(parts) > 1 else ""
        hid = parts[2] if len(parts) > 2 else ""
        return (2, PROMPT_ORDER.get(pid, 99), hid, pid, rep_suffix, ablation_suffix)
    if name.startswith("ptllm_smac__"):
        parts = name.split("__")
        pid = parts[1] if len(parts) > 1 else ""
        hid = parts[2] if len(parts) > 2 else ""
        return (3, PROMPT_ORDER.get(pid, 99), hid, pid, rep_suffix, ablation_suffix)
    if name.startswith("ptllm_cmaes__"):
        parts = name.split("__")
        pid = parts[1] if len(parts) > 1 else ""
        hid = parts[2] if len(parts) > 2 else ""
        return (4, PROMPT_ORDER.get(pid, 99), hid, pid, rep_suffix, ablation_suffix)
    return (9, 0, name, "", rep_suffix, ablation_suffix)


def merge_replication_summary(
    summary: dict,
    jobs: list[tuple[str, str]],
    repo_root: Path,
) -> dict[str, int]:
    """Copy scores from replication base dirs into summary under suffixed variant keys."""
    coverage: dict[str, int] = {}
    for out_variant, rel_base, src_variant in REPLICATION_VARIANTS:
        rep_base = repo_root / rel_base
        if not rep_base.is_dir():
            continue
        _, _, _, rep_summary = collect_sweep(
            rep_base, jobs, lambda name, sv=src_variant: name == sv
        )
        n = 0
        for key, value in rep_summary.items():
            dataset, variant = key
            if variant != src_variant:
                continue
            summary[(dataset, out_variant)] = value
            n += 1
        coverage[out_variant] = n
    return coverage


def discover_variants(base: Path, jobs: list[tuple[str, str]]) -> list[str]:
    found: set[str] = set(PINNED_VARIANTS)
    rep_keys = {v[0] for v in REPLICATION_VARIANTS}
    for task_type, dataset in jobs:
        ds_dir = base / task_type / dataset
        if not ds_dir.is_dir():
            continue
        for child in ds_dir.iterdir():
            if child.is_dir():
                found.add(child.name)
    pinned = [v for v in PINNED_VARIANTS if v in found or v in rep_keys]
    rest = sorted((found - set(PINNED_VARIANTS)), key=_variant_sort_key)
    return pinned + rest


def _write_subset_table(
    out_dir: Path,
    dataset_keys: list[str],
    variants: list[str],
    summary: dict,
    filename: str,
    tag: str,
):
    subset = [v for v in variants if any((d, v) in summary for d in dataset_keys)]
    if len(subset) < 2:
        return None, {}, {}
    tmp = out_dir / tag
    tmp.mkdir(parents=True, exist_ok=True)
    table_path, mean_scores, avg_ranks = write_valid_best_table(tmp, dataset_keys, subset, summary)
    final = out_dir / filename
    shutil.move(str(table_path), str(final))
    shutil.rmtree(tmp, ignore_errors=True)
    return final, mean_scores, avg_ranks


def _task_avg_ranks(
    dataset_keys: list[str],
    variants: list[str],
    summary: dict,
    task_prefix: str,
) -> dict[str, float]:
    keys = [d for d in dataset_keys if d.startswith(f"{task_prefix}/")]
    higher = score_higher_is_better(task_prefix)
    rank_sums = {v: 0.0 for v in variants}
    counts = {v: 0 for v in variants}
    for dataset in keys:
        scores = {v: summary[(dataset, v)] for v in variants if (dataset, v) in summary}
        if len(scores) < 2:
            continue
        for v, rank in _ranks_for_task(scores, higher).items():
            rank_sums[v] += rank
            counts[v] += 1
    return {
        v: (rank_sums[v] / counts[v] if counts[v] else float("nan")) for v in variants
    }


def _raw_task_means(
    dataset_keys: list[str],
    variants: list[str],
    summary: dict,
    task_prefix: str,
) -> dict[str, float]:
    keys = [d for d in dataset_keys if d.startswith(f"{task_prefix}/")]
    out: dict[str, list[float]] = {v: [] for v in variants}
    for dataset in keys:
        for v in variants:
            if (dataset, v) in summary:
                out[v].append(summary[(dataset, v)])
    return {v: (sum(vals) / len(vals) if vals else float("nan")) for v, vals in out.items()}


def _rank_wins(
    dataset_keys: list[str],
    variants: list[str],
    summary: dict,
) -> dict[str, int]:
    wins = {v: 0 for v in variants}
    for dataset in dataset_keys:
        task_type = dataset.split("/", 1)[0]
        higher = score_higher_is_better(task_type)
        scores = {v: summary[(dataset, v)] for v in variants if (dataset, v) in summary}
        if len(scores) < 2:
            continue
        ranks = _ranks_for_task(scores, higher)
        for v, rank in ranks.items():
            if rank == 1.0:
                wins[v] += 1
    return wins


def write_metrics_summary(
    out_dir: Path,
    dataset_keys: list[str],
    variants: list[str],
    summary: dict,
    avg_ranks: dict[str, float],
    mean_scores: dict[str, float],
    coverage: dict[str, int],
    *,
    complete_only: bool = True,
    n_datasets: int = 8,
) -> Path:
    pool = [v for v in variants if coverage.get(v, 0) >= n_datasets] if complete_only else variants
    cls_norm = task_normalized_means(dataset_keys, pool, summary, "cls")
    rgs_norm = task_normalized_means(dataset_keys, pool, summary, "rgs")
    meta_cls, meta_rgs, balanced_meta = balanced_meta_score(cls_norm, rgs_norm)
    rank_cls = _task_avg_ranks(dataset_keys, pool, summary, "cls")
    rank_rgs = _task_avg_ranks(dataset_keys, pool, summary, "rgs")
    raw_cls = _raw_task_means(dataset_keys, pool, summary, "cls")
    raw_rgs = _raw_task_means(dataset_keys, pool, summary, "rgs")
    wins = _rank_wins(dataset_keys, pool, summary)

    smac_scores = {d: summary.get((d, "smac")) for d in dataset_keys if (d, "smac") in summary}
    vs_smac = {v: 0 for v in pool}
    for dataset in dataset_keys:
        if dataset not in smac_scores or smac_scores[dataset] is None:
            continue
        smac_v = smac_scores[dataset]
        for v in pool:
            if v == "smac" or (dataset, v) not in summary:
                continue
            val = summary[(dataset, v)]
            if val > smac_v:
                vs_smac[v] += 1

    rows = []
    for v in pool:
        cn = cls_norm.get(v, float("nan"))
        rn = rgs_norm.get(v, float("nan"))
        balanced = (cn + rn) / 2 if not (math.isnan(cn) or math.isnan(rn)) else float("nan")
        mcn = meta_cls.get(v, float("nan"))
        mrn = meta_rgs.get(v, float("nan"))
        bmeta = balanced_meta.get(v, float("nan"))
        if v in ("smac", "tpe", "cmaes"):
            fam = "classical"
        elif v in CANONICAL_BASELINES:
            fam = "baseline"
        elif v in REFERENCE_TUNED:
            fam = "reference_tuned"
        elif v.endswith("_2"):
            fam = "replication"
        elif v.startswith("ptllm_smac"):
            fam = "ptllm_smac"
        elif v.startswith("ptllm_cmaes"):
            fam = "ptllm_cmaes"
        elif v.startswith("ptllm_tpe"):
            fam = "ptllm_tpe"
        elif v.startswith("ptllm"):
            fam = "ptllm"
        else:
            fam = "other"
        rows.append(
            {
                "variant": v,
                "display_name": variant_display_name(v),
                "family": fam,
                "datasets": coverage.get(v, 0),
                "avg_rank": avg_ranks.get(v, float("nan")),
                "avg_rank_cls": rank_cls.get(v, float("nan")),
                "avg_rank_rgs": rank_rgs.get(v, float("nan")),
                "mean_cls_norm": cn,
                "mean_rgs_norm": rn,
                "mean_balanced_norm": balanced,
                "meta_cls_norm": mcn,
                "meta_rgs_norm": mrn,
                "balanced_score": bmeta,
                "mean_raw_cls": raw_cls.get(v, float("nan")),
                "mean_raw_rgs": raw_rgs.get(v, float("nan")),
                "rank_wins": wins.get(v, 0),
                "beats_smac_count": vs_smac.get(v, 0),
            }
        )

    rows.sort(key=lambda r: (r["avg_rank"], -r["mean_balanced_norm"]))

    path = out_dir / "metrics_summary.csv"
    fields = [
        "rank_by_avg_rank",
        "variant",
        "display_name",
        "family",
        "datasets",
        "avg_rank",
        "avg_rank_cls",
        "avg_rank_rgs",
        "mean_cls_norm",
        "mean_rgs_norm",
        "mean_balanced_norm",
        "meta_cls_norm",
        "meta_rgs_norm",
        "balanced_score",
        "rank_wins",
        "beats_smac_count",
        "mean_raw_cls",
        "mean_raw_rgs",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            row["rank_by_avg_rank"] = i
            for key in list(row):
                if isinstance(row[key], float) and key not in ("rank_wins", "beats_smac_count", "datasets"):
                    row[key] = f"{row[key]:.6f}" if not math.isnan(row[key]) else ""
            writer.writerow(row)

    balanced_path = out_dir / "metrics_by_balanced_norm.csv"
    by_balanced = sorted(
        rows,
        key=lambda r: (
            -float(r["mean_balanced_norm"]) if r["mean_balanced_norm"] != "" else float("-inf"),
            float(r["avg_rank"]) if r["avg_rank"] != "" else float("inf"),
        ),
    )
    with balanced_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for i, row in enumerate(by_balanced, start=1):
            out_row = dict(row)
            out_row["rank_by_avg_rank"] = i
            writer.writerow(out_row)

    meta_path = out_dir / "metrics_by_balanced_score.csv"
    by_meta = sorted(
        rows,
        key=lambda r: (
            -float(r["balanced_score"]) if r["balanced_score"] != "" else float("-inf"),
            float(r["avg_rank"]) if r["avg_rank"] != "" else float("inf"),
        ),
    )
    meta_fields = list(fields)
    meta_fields[0] = "rank_by_balanced_score"
    with meta_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=meta_fields, extrasaction="ignore")
        writer.writeheader()
        for i, row in enumerate(by_meta, start=1):
            out_row = dict(row)
            out_row["rank_by_balanced_score"] = i
            writer.writerow(out_row)

    return path


def write_report_avg_rank(
    out_dir: Path,
    dataset_keys: list[str],
    variants: list[str],
    summary: dict,
    avg_ranks: dict[str, float],
    coverage: dict[str, int],
    *,
    n_datasets: int = 8,
) -> Path:
    """Simplified avg_rank-only table for reporting (complete 8/8 runs)."""
    pool = [v for v in variants if coverage.get(v, 0) >= n_datasets]
    rank_cls = _task_avg_ranks(dataset_keys, pool, summary, "cls")
    rank_rgs = _task_avg_ranks(dataset_keys, pool, summary, "rgs")
    cls_norm = task_normalized_means(dataset_keys, pool, summary, "cls")
    rgs_norm = task_normalized_means(dataset_keys, pool, summary, "rgs")

    rows = []
    for v in pool:
        if v in ("smac", "tpe", "cmaes"):
            fam = "classical"
        elif v in CANONICAL_BASELINES:
            fam = "baseline"
        elif v in REFERENCE_TUNED:
            fam = "reference_tuned"
        elif v.endswith("_2"):
            fam = "replication"
        elif v.startswith("ptllm_smac"):
            fam = "ptllm_smac"
        elif v.startswith("ptllm_cmaes"):
            fam = "ptllm_cmaes"
        elif v.startswith("ptllm_tpe"):
            fam = "ptllm_tpe"
        elif v.startswith("ptllm"):
            fam = "ptllm"
        else:
            fam = "other"
        rows.append(
            {
                "variant": v,
                "display_name": variant_display_name(v),
                "family": fam,
                "avg_rank": avg_ranks.get(v, float("nan")),
                "avg_rank_cls": rank_cls.get(v, float("nan")),
                "avg_rank_rgs": rank_rgs.get(v, float("nan")),
            }
        )
    for row in rows:
        v = row["variant"]
        cn = cls_norm.get(v, float("nan"))
        rn = rgs_norm.get(v, float("nan"))
        row["mean_cls_norm"] = cn
        row["mean_rgs_norm"] = rn
        row["mean_balanced_norm"] = (cn + rn) / 2 if not (math.isnan(cn) or math.isnan(rn)) else float("nan")

    rows.sort(key=lambda r: r["avg_rank"])

    path = out_dir / "avg_rank_table.csv"
    fields = [
        "rank",
        "display_name",
        "variant",
        "family",
        "avg_rank",
        "avg_rank_cls",
        "avg_rank_rgs",
        "mean_cls_norm",
        "mean_rgs_norm",
        "mean_balanced_norm",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            row["rank"] = i
            for key in ("avg_rank", "avg_rank_cls", "avg_rank_rgs"):
                if not math.isnan(row[key]):
                    row[key] = f"{row[key]:.2f}"
                else:
                    row[key] = ""
            for key in ("mean_cls_norm", "mean_rgs_norm", "mean_balanced_norm"):
                if not math.isnan(row[key]):
                    row[key] = f"{row[key]:.6f}"
                else:
                    row[key] = ""
            writer.writerow(row)
    shutil.copy2(path, out_dir / "avg_rank_leaderboard_eval8_complete.csv")
    return path


def write_senior_top5_baselines_report(
    out_dir: Path,
    dataset_keys: list[str],
    variants: list[str],
    summary: dict,
    avg_ranks: dict[str, float],
    coverage: dict[str, int],
    *,
    n_datasets: int = 8,
    top_n: int = 5,
) -> Path:
    """Top-N LLM methods + classical baselines, 8/8 eval8 only (for senior reporting)."""
    classical = ("smac", "tpe", "cmaes")
    complete = [v for v in variants if coverage.get(v, 0) >= n_datasets]
    llm_top = sorted(
        (v for v in complete if v not in classical),
        key=lambda v: avg_ranks.get(v, float("inf")),
    )[:top_n]
    report_variants = [v for v in classical if v in complete] + llm_top
    report_variants = sorted(report_variants, key=lambda v: avg_ranks.get(v, float("inf")))

    rank_cls = _task_avg_ranks(dataset_keys, complete, summary, "cls")
    rank_rgs = _task_avg_ranks(dataset_keys, complete, summary, "rgs")
    cls_norm = task_normalized_means(dataset_keys, complete, summary, "cls")
    rgs_norm = task_normalized_means(dataset_keys, complete, summary, "rgs")

    dataset_short = [d.split("/", 1)[1] for d in dataset_keys]
    fields = [
        "rank",
        "display_name",
        "variant",
        "role",
        "avg_rank",
        "avg_rank_cls",
        "avg_rank_rgs",
        "mean_cls_norm",
        "mean_rgs_norm",
        "mean_balanced_norm",
    ]
    for ds in dataset_short:
        fields.extend([f"score_{ds}", f"rank_{ds}"])

    path = out_dir / "eval8_senior_top5_baselines.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for i, v in enumerate(report_variants, start=1):
            if v in classical:
                role = "classical_baseline"
            elif v.startswith("ptllm_tpe"):
                role = "ptllm_tpe"
            elif v.startswith("ptllm"):
                role = "ptllm"
            else:
                role = "other"
            cn = cls_norm.get(v, float("nan"))
            rn = rgs_norm.get(v, float("nan"))
            balanced = (cn + rn) / 2 if not (math.isnan(cn) or math.isnan(rn)) else float("nan")
            row = {
                "rank": i,
                "display_name": variant_display_name(v),
                "variant": v,
                "role": role,
                "avg_rank": avg_ranks.get(v, float("nan")),
                "avg_rank_cls": rank_cls.get(v, float("nan")),
                "avg_rank_rgs": rank_rgs.get(v, float("nan")),
                "mean_cls_norm": cn,
                "mean_rgs_norm": rn,
                "mean_balanced_norm": balanced,
            }
            for dataset, ds in zip(dataset_keys, dataset_short):
                scores = {
                    rv: summary[(dataset, rv)]
                    for rv in report_variants
                    if (dataset, rv) in summary
                }
                val = summary.get((dataset, v), float("nan"))
                row[f"score_{ds}"] = val
                if len(scores) >= 2 and not math.isnan(val):
                    row[f"rank_{ds}"] = _ranks_for_task(scores, score_higher_is_better(dataset.split("/", 1)[0]))[v]
                else:
                    row[f"rank_{ds}"] = float("nan")
            for key in ("avg_rank", "avg_rank_cls", "avg_rank_rgs"):
                if not math.isnan(row[key]):
                    row[key] = f"{row[key]:.2f}"
                else:
                    row[key] = ""
            for key in ("mean_cls_norm", "mean_rgs_norm", "mean_balanced_norm"):
                if not math.isnan(row[key]):
                    row[key] = f"{row[key]:.6f}"
                else:
                    row[key] = ""
            for ds in dataset_short:
                sk, rk = f"score_{ds}", f"rank_{ds}"
                if isinstance(row[sk], float) and not math.isnan(row[sk]):
                    row[sk] = f"{row[sk]:.6f}"
                else:
                    row[sk] = ""
                if isinstance(row[rk], float) and not math.isnan(row[rk]):
                    row[rk] = f"{row[rk]:.1f}"
                else:
                    row[rk] = ""
            writer.writerow(row)

    # Compact leaderboard (same shape as avg_rank_leaderboard.csv)
    lb_path = out_dir / "eval8_top8_leaderboard.csv"
    with lb_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "variant",
                "display_name",
                "role",
                "avg_rank",
                "avg_rank_cls",
                "avg_rank_rgs",
                "mean_cls_norm",
                "mean_rgs_norm",
                "mean_balanced_norm",
                "datasets_with_data",
            ]
        )
        for i, v in enumerate(report_variants, start=1):
            cn = cls_norm.get(v, float("nan"))
            rn = rgs_norm.get(v, float("nan"))
            balanced = (cn + rn) / 2 if not (math.isnan(cn) or math.isnan(rn)) else float("nan")
            if v in classical:
                role = "classical_baseline"
            elif v.startswith("ptllm_tpe"):
                role = "ptllm_tpe"
            elif v.startswith("ptllm"):
                role = "ptllm"
            else:
                role = "other"
            writer.writerow(
                [
                    i,
                    v,
                    variant_display_name(v),
                    role,
                    f"{avg_ranks[v]:.4f}" if not math.isnan(avg_ranks[v]) else "",
                    f"{rank_cls[v]:.2f}" if not math.isnan(rank_cls.get(v, float("nan"))) else "",
                    f"{rank_rgs[v]:.2f}" if not math.isnan(rank_rgs.get(v, float("nan"))) else "",
                    f"{cn:.6f}" if not math.isnan(cn) else "",
                    f"{rn:.6f}" if not math.isnan(rn) else "",
                    f"{balanced:.6f}" if not math.isnan(balanced) else "",
                    n_datasets,
                ]
            )

    # Per-dataset valid_best (8 methods only)
    vb_path = out_dir / "eval8_top8_valid_best.csv"
    display = [variant_display_name(v) for v in report_variants]
    with vb_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset"] + display)
        for dataset in dataset_keys:
            row = [dataset]
            for v in report_variants:
                val = summary.get((dataset, v), float("nan"))
                row.append("" if math.isnan(val) else f"{val:.6f}")
            writer.writerow(row)
        writer.writerow(
            ["avg_rank"]
            + [
                f"{avg_ranks[v]:.4f}" if not math.isnan(avg_ranks.get(v, float("nan"))) else ""
                for v in report_variants
            ]
        )

    return path


def _hybrid_schedule_label(hybrid_id: str | None) -> str:
    if not hybrid_id:
        return ""
    hybrids = load_hybrid_variants()
    if hybrid_id not in hybrids:
        return hybrid_id
    row = hybrids[hybrid_id]
    phase = int(row.get("llm_phase_trials", 0))
    prob = float(row.get("llm_prob", 0))
    if phase > 0 and prob == 0:
        return f"{phase} LLM + {80 - phase} TPE"
    if phase == 0 and prob > 0:
        return f"mix {int(prob * 100)}% LLM"
    if phase > 0:
        return f"phase{phase} + mix {int(prob * 100)}% LLM"
    return "mix 50% LLM"


def write_p27_phase_sweep_table(
    out_dir: Path,
    dataset_keys: list[str],
    present: list[str],
    summary: dict,
    avg_ranks: dict[str, float],
    coverage: dict[str, int],
    *,
    n_datasets: int = 8,
) -> Path | None:
    """Focused avg_rank table: P27 phase hybrids + key baselines."""
    pool = {v for v in present if coverage.get(v, 0) >= n_datasets}
    rank_cls = _task_avg_ranks(dataset_keys, list(pool), summary, "cls")
    rank_rgs = _task_avg_ranks(dataset_keys, list(pool), summary, "rgs")

    focus_order = [
        "tpe",
        "cmaes",
        "ptllm__P27",
    ]
    focus_order += sorted(
        v for v in pool if v.startswith("ptllm_tpe__P27__") or v in ("ptllm_smac__P27__H2", "ptllm_cmaes__P27__H2")
    )
    seen = set()
    variants = []
    for v in focus_order:
        if v in pool and v not in seen:
            variants.append(v)
            seen.add(v)

    if len(variants) < 2:
        return None

    rows = []
    for v in variants:
        _, _, hid = parse_variant_tags(v)
        rows.append(
            {
                "variant": v,
                "display_name": variant_display_name(v),
                "hybrid": hid or "",
                "schedule": _hybrid_schedule_label(hid),
                "avg_rank": avg_ranks.get(v, float("nan")),
                "avg_rank_cls": rank_cls.get(v, float("nan")),
                "avg_rank_rgs": rank_rgs.get(v, float("nan")),
            }
        )
    rows.sort(key=lambda r: r["avg_rank"])

    path = out_dir / "p27_phase_sweep_avg_rank.csv"
    fields = ["rank", "display_name", "variant", "hybrid", "schedule", "avg_rank", "avg_rank_cls", "avg_rank_rgs"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            row["rank"] = i
            for key in ("avg_rank", "avg_rank_cls", "avg_rank_rgs"):
                val = row[key]
                row[key] = f"{val:.2f}" if isinstance(val, float) and not math.isnan(val) else ""
            writer.writerow(row)
    return path


def write_p16_ablation_avg_rank_table(
    out_dir: Path,
    dataset_keys: list[str],
    present: list[str],
    summary: dict,
    avg_ranks: dict[str, float],
    coverage: dict[str, int],
    *,
    n_datasets: int = 8,
) -> Path | None:
    """Focused avg_rank table: P16 mix sweep, single-agent, and pure-LLM ablations."""
    pool = set(present)
    rank_cls = _task_avg_ranks(dataset_keys, list(pool), summary, "cls")
    rank_rgs = _task_avg_ranks(dataset_keys, list(pool), summary, "rgs")

    ablation_labels = {
        "ptllm_tpe__P16__H2": ("mix70", "P16 two-call + 70% LLM (production)"),
        "ptllm_tpe__P16__H20": ("mix20", "P16 two-call + 20% LLM"),
        "ptllm_tpe__P16__H1": ("mix30", "P16 two-call + 30% LLM"),
        "ptllm_tpe__P16__H40": ("mix40", "P16 two-call + 40% LLM"),
        "ptllm_tpe__P30__H2": ("mix70", "P30 single-call + 70% LLM"),
        "ptllm__P16": ("pure_llm", "ptllm P16 two-call (100% LLM, no TPE)"),
        "ptllm__P5": ("pure_llm", "ptllm P5 (pure LLM reference)"),
        "ptllm_tpe__P5__H2": ("mix70", "P5 + 70% LLM (non-TCS reference)"),
    }
    focus_order = list(ablation_labels.keys())

    variants = [v for v in focus_order if v in pool]
    if len(variants) < 2:
        return None

    rows = []
    for v in variants:
        schedule, notes = ablation_labels[v]
        _, pid, hid = parse_variant_tags(v)
        rows.append(
            {
                "variant": v,
                "display_name": variant_display_name(v),
                "prompt": pid or "",
                "hybrid": hid or "",
                "schedule": schedule,
                "ablation": notes,
                "datasets_with_data": coverage.get(v, 0),
                "avg_rank": avg_ranks.get(v, float("nan")),
                "avg_rank_cls": rank_cls.get(v, float("nan")),
                "avg_rank_rgs": rank_rgs.get(v, float("nan")),
            }
        )
    rows.sort(
        key=lambda r: (
            -(r["datasets_with_data"] >= n_datasets),
            r["avg_rank"] if not math.isnan(r["avg_rank"]) else float("inf"),
        )
    )

    path = out_dir / "p16_ablation_avg_rank.csv"
    fields = [
        "rank",
        "display_name",
        "variant",
        "prompt",
        "hybrid",
        "schedule",
        "ablation",
        "datasets_with_data",
        "avg_rank",
        "avg_rank_cls",
        "avg_rank_rgs",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            row["rank"] = i
            for key in ("avg_rank", "avg_rank_cls", "avg_rank_rgs"):
                val = row[key]
                row[key] = f"{val:.2f}" if isinstance(val, float) and not math.isnan(val) else ""
            writer.writerow(row)
    return path


def main():
    parser = argparse.ArgumentParser(description="Master valid_best CSV for all variants.")
    parser.add_argument("--base-dir", default="benchmark_data/real_hpo_v2")
    parser.add_argument("--include-datasets", default=EVAL8_DATASETS)
    parser.add_argument(
        "--out-dir",
        default="benchmark_data/real_hpo_v2/analysis/eval8/all_valid_best",
    )
    parser.add_argument(
        "--min-datasets",
        type=int,
        default=1,
        help="Only include variants present on at least this many datasets.",
    )
    args = parser.parse_args()

    base = Path(args.base_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    jobs = jobs_from_dataset_names(args.include_datasets)
    dataset_keys = [f"{t}/{d}" for t, d in jobs]
    all_variants = discover_variants(base, jobs)

    _, present_any, _, summary = collect_sweep(base, jobs, lambda name: name in set(all_variants))

    repo_root = Path(__file__).resolve().parents[2]
    rep_coverage = merge_replication_summary(summary, jobs, repo_root)
    for out_variant in rep_coverage:
        if out_variant not in all_variants:
            src = out_variant.removesuffix("_2")
            if src in all_variants:
                all_variants.insert(all_variants.index(src) + 1, out_variant)
            else:
                all_variants.append(out_variant)

    coverage = {v: sum(1 for d in dataset_keys if (d, v) in summary) for v in all_variants}
    rep_keys = {v[0] for v in REPLICATION_VARIANTS}
    present = [
        v
        for v in all_variants
        if v in PINNED_VARIANTS
        or v in rep_keys
        or (coverage[v] >= args.min_datasets and v in present_any)
    ]
    if not present:
        print("No variants found under", base)
        return

    table_path, mean_scores, avg_ranks = write_valid_best_table(out, dataset_keys, present, summary)
    display_names = [variant_display_name(v) for v in present]
    append_summary_rows(table_path, dataset_keys, present, summary, display_names)

    metrics_path = write_metrics_summary(
        out,
        dataset_keys,
        present,
        summary,
        avg_ranks,
        mean_scores,
        coverage,
        complete_only=True,
        n_datasets=len(dataset_keys),
    )
    report_path = write_report_avg_rank(
        out,
        dataset_keys,
        present,
        summary,
        avg_ranks,
        coverage,
        n_datasets=len(dataset_keys),
    )
    senior_path = write_senior_top5_baselines_report(
        out,
        dataset_keys,
        present,
        summary,
        avg_ranks,
        coverage,
        n_datasets=len(dataset_keys),
    )
    p27_path = write_p27_phase_sweep_table(
        out,
        dataset_keys,
        present,
        summary,
        avg_ranks,
        coverage,
        n_datasets=len(dataset_keys),
    )
    p16_ablation_path = write_p16_ablation_avg_rank_table(
        out,
        dataset_keys,
        present,
        summary,
        avg_ranks,
        coverage,
        n_datasets=len(dataset_keys),
    )

    baseline_path, _, baseline_subset_ranks = _write_subset_table(
        out,
        dataset_keys,
        PINNED_VARIANTS,
        summary,
        "valid_best_baselines.csv",
        "_baselines_tmp",
    )

    cls_norm = task_normalized_means(dataset_keys, present, summary, "cls")
    rgs_norm = task_normalized_means(dataset_keys, present, summary, "rgs")

    def _balanced_norm(v: str) -> float:
        cn = cls_norm.get(v, float("nan"))
        rn = rgs_norm.get(v, float("nan"))
        if math.isnan(cn) or math.isnan(rn):
            return float("nan")
        return (cn + rn) / 2

    ranked = sorted(
        present,
        key=lambda v: (
            avg_ranks.get(v, float("inf")),
            -_balanced_norm(v) if not math.isnan(_balanced_norm(v)) else float("inf"),
        ),
    )

    rank_path = out / "avg_rank_leaderboard.csv"
    with rank_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "variant",
                "display_name",
                "avg_rank",
                "mean_cls_norm",
                "mean_rgs_norm",
                "mean_balanced_norm",
                "datasets_with_data",
            ]
        )
        for i, v in enumerate(ranked, start=1):
            cn = cls_norm.get(v, float("nan"))
            rn = rgs_norm.get(v, float("nan"))
            balanced = (cn + rn) / 2 if not (math.isnan(cn) or math.isnan(rn)) else float("nan")
            writer.writerow(
                [
                    i,
                    v,
                    variant_display_name(v),
                    f"{avg_ranks[v]:.4f}" if not math.isnan(avg_ranks[v]) else "",
                    f"{cn:.6f}" if not math.isnan(cn) else "",
                    f"{rn:.6f}" if not math.isnan(rn) else "",
                    f"{balanced:.6f}" if not math.isnan(balanced) else "",
                    coverage[v],
                ]
            )

    n_ds = len(dataset_keys)
    complete = [v for v in ranked if coverage[v] >= n_ds]

    if baseline_path:
        baseline_rank_path = out / "avg_rank_baselines_only.csv"
        bl_complete = [v for v in PINNED_VARIANTS if v in present and coverage[v] >= n_ds]
        bl_ranked = sorted(
            bl_complete,
            key=lambda v: baseline_subset_ranks.get(v, float("inf")),
        )
        with baseline_rank_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "variant", "display_name", "avg_rank", "role"])
            for i, v in enumerate(bl_ranked, start=1):
                if v in ("smac", "tpe", "cmaes"):
                    role = "classical"
                elif v in REFERENCE_TUNED:
                    role = "reference_tuned"
                else:
                    role = "llm_baseline"
                writer.writerow(
                    [
                        i,
                        v,
                        variant_display_name(v),
                        f"{baseline_subset_ranks[v]:.4f}"
                        if v in baseline_subset_ranks and not math.isnan(baseline_subset_ranks[v])
                        else "",
                        role,
                    ]
                )

    (out / "variants.txt").write_text("\n".join(present) + "\n", encoding="utf-8")
    best = ranked[0]
    print(f"[Done] {len(present)} variants, {len(dataset_keys)} datasets")
    print(f"  table: {table_path}")
    print(f"  avg_rank table: {report_path}")
    print(f"  senior report: {senior_path}")
    print(f"  top8 leaderboard: {out / 'eval8_top8_leaderboard.csv'}")
    print(f"  top8 valid_best: {out / 'eval8_top8_valid_best.csv'}")
    if p27_path:
        print(f"  P27 phase sweep: {p27_path}")
    if p16_ablation_path:
        print(f"  P16 ablations: {p16_ablation_path}")
    print(f"  metrics: {metrics_path}")
    print(f"  metrics (by balanced norm): {out / 'metrics_by_balanced_norm.csv'}")
    print(f"  metrics (by balanced_score): {out / 'metrics_by_balanced_score.csv'}")
    if baseline_path:
        print(f"  baselines+refs: {baseline_path}")
        print(f"  baseline ranks: {out / 'avg_rank_baselines_only.csv'}")
    print(f"  leaderboard: {rank_path}")
    print(f"  complete (8/8): {out / 'avg_rank_leaderboard_eval8_complete.csv'}")
    print(f"  best avg_rank (all): {best} ({avg_ranks[best]:.4f})")
    if complete:
        cb = complete[0]
        print(f"  best avg_rank (8/8 complete): {cb} ({avg_ranks[cb]:.4f})")


if __name__ == "__main__":
    main()
