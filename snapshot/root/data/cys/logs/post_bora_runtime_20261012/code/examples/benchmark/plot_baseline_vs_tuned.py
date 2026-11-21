"""Compare baselines vs tuned ptllm/ptllm_tpe on pilot datasets (no covertype).

Writes valid_best_table.csv + plots under analysis/no_covertype/baseline_vs_tuned/.

Default layout:
  baselines: smac, tpe, cmaes, ptllm (P0), ptllm_tpe (P0/H0)
  tuned:     ptllm (P5) + top-3 ptllm_tpe hybrids from the P5 sweep (by avg_rank on eval set)
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from plot_pilot_sweep import (
    _ranks_for_task,
    _variant_colors,
    collect_sweep,
    plot_average_rank_curve,
    plot_dataset_convergence,
    plot_mean_bar,
    plot_avg_rank_bar,
    plot_normalized_task_bars,
    task_normalized_means,
    write_valid_best_table,
)
from experiment_utils import EVAL8_DATASETS, jobs_from_dataset_names, list_v2_dataset_jobs, score_higher_is_better

BASELINE_VARIANTS = [
    "smac",
    "tpe",
    "cmaes",
    "ptllm__P0",
    "ptllm_tpe__P0__H0",
]


def variant_display_name(variant: str) -> str:
    if variant.endswith("_2") and variant.startswith("ptllm"):
        return variant_display_name(variant.removesuffix("_2")) + "_2"
    if variant in ("smac", "tpe", "cmaes"):
        return variant
    # Fine-tuned Qwen LoRA: ptllm_tpe_qwen_s1a__P5__H0 -> ptllm_tpe_qwen_s1a (P5/H0)
    if variant.startswith("ptllm_tpe_qwen_"):
        parts = variant.split("__")
        head = parts[0]  # ptllm_tpe_qwen_s1a
        if len(parts) >= 3:
            return f"{head} ({parts[1]}/{parts[2]})"
        if len(parts) == 2:
            return f"{head} ({parts[1]})"
        return head
    if variant.startswith("ptllm_tpe_qwen__"):
        parts = variant.split("__")
        if len(parts) >= 3:
            return f"ptllm_tpe_qwen ({parts[1]}/{parts[2]})"
    if variant.startswith("ptllm_tpe__"):
        parts = variant.split("__")
        if len(parts) >= 4 and parts[3] == "json_only":
            return f"ptllm_tpe ({parts[1]}/{parts[2]} json_only)"
        if len(parts) >= 3:
            return f"ptllm_tpe ({parts[1]}/{parts[2]})"
    if variant.startswith("ptllm__"):
        return f"ptllm ({variant.split('__', 1)[1]})"
    return variant


def pick_top_tpe_hybrids(
    base: Path,
    jobs: list[tuple[str, str]],
    dataset_keys: list[str],
  *,
    tpe_prompt: str,
    ptllm_prompt: str,
    top_k: int,
) -> list[str]:
    """Return top-k ptllm_tpe variant dir names by avg_rank on dataset_keys."""
    prefix = f"ptllm_tpe__{tpe_prompt}__"
    _, candidates, _, summary = collect_sweep(
        base, jobs, lambda name, p=prefix: name.startswith(p)
    )
    candidates = sorted(candidates)
    if not candidates:
        return []

    rank_pool = BASELINE_VARIANTS + [f"ptllm__{ptllm_prompt}"] + candidates
    _, _, _, pool_summary = collect_sweep(base, jobs, lambda name, s=set(rank_pool): name in s)

    scored: list[tuple[float, str]] = []
    for variant in candidates:
        rank_sum = 0.0
        n = 0
        for dataset in dataset_keys:
            higher = score_higher_is_better(dataset.split("/", 1)[0])
            scores = {
                v: pool_summary[(dataset, v)]
                for v in rank_pool
                if (dataset, v) in pool_summary
            }
            if variant not in scores or len(scores) < 2:
                continue
            ranks = _ranks_for_task(scores, higher)
            rank_sum += ranks[variant]
            n += 1
        if n:
            scored.append((rank_sum / n, variant))

    scored.sort(key=lambda x: x[0])
    return [v for _, v in scored[:top_k]]


def append_summary_rows(
    table_path: Path,
    dataset_keys: list[str],
    table_variants: list[str],
    summary: dict,
    display_names: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    rows = list(csv.reader(table_path.open()))
    cls_keys = [d for d in dataset_keys if d.startswith("cls/")]
    rgs_keys = [d for d in dataset_keys if d.startswith("rgs/")]
    data_rows = {
        r[0]: r
        for r in rows[1:]
        if r[0] not in ("mean_score", "avg_rank", "mean_score_cls", "mean_score_rgs", "avg_rank_cls", "avg_rank_rgs", "mean_score_cls_norm", "mean_score_rgs_norm")
    }

    def row_mean(keys, col_idx):
        vals = [
            float(data_rows[k][col_idx])
            for k in keys
            if k in data_rows and data_rows[k][col_idx]
        ]
        return sum(vals) / len(vals) if vals else float("nan")

    header = rows[0]
    mean_cls = ["mean_score_cls"] + [
        "" if math.isnan(v := row_mean(cls_keys, i)) else f"{v:.6f}" for i in range(1, len(header))
    ]
    mean_rgs = ["mean_score_rgs"] + [
        "" if math.isnan(v := row_mean(rgs_keys, i)) else f"{v:.6f}" for i in range(1, len(header))
    ]

    cls_norm = task_normalized_means(dataset_keys, table_variants, summary, "cls")
    rgs_norm = task_normalized_means(dataset_keys, table_variants, summary, "rgs")
    mean_cls_norm = ["mean_score_cls_norm"] + [
        "" if math.isnan(cls_norm.get(v, float("nan"))) else f"{cls_norm[v]:.6f}" for v in table_variants
    ]
    mean_rgs_norm = ["mean_score_rgs_norm"] + [
        "" if math.isnan(rgs_norm.get(v, float("nan"))) else f"{rgs_norm[v]:.6f}" for v in table_variants
    ]

    rank_sums_cls = {v: 0.0 for v in table_variants}
    for dataset in cls_keys:
        scores = {v: summary[(dataset, v)] for v in table_variants if (dataset, v) in summary}
        if len(scores) < 2:
            continue
        for v, rank in _ranks_for_task(scores, True).items():
            rank_sums_cls[v] += rank
    avg_rank_cls = ["avg_rank_cls"] + [
        "" if not cls_keys else f"{rank_sums_cls[v] / len(cls_keys):.6f}" for v in table_variants
    ]

    rank_sums_rgs = {v: 0.0 for v in table_variants}
    for dataset in rgs_keys:
        scores = {v: summary[(dataset, v)] for v in table_variants if (dataset, v) in summary}
        if len(scores) < 2:
            continue
        for v, rank in _ranks_for_task(scores, False).items():
            rank_sums_rgs[v] += rank
    avg_rank_rgs = ["avg_rank_rgs"] + [
        "" if not rgs_keys else f"{rank_sums_rgs[v] / len(rgs_keys):.6f}" for v in table_variants
    ]

    rows[0] = ["dataset"] + display_names[: len(table_variants)]
    body = [
        r
        for r in rows[1:]
        if r[0]
        not in (
            "mean_score",
            "mean_score_cls",
            "mean_score_rgs",
            "mean_score_cls_norm",
            "mean_score_rgs_norm",
            "avg_rank_cls",
            "avg_rank_rgs",
            "avg_rank",
        )
    ]
    avg_rank_row = next((r for r in rows[1:] if r[0] == "avg_rank"), None)
    out_rows = [rows[0]] + body + [mean_cls, mean_rgs, mean_cls_norm, mean_rgs_norm, avg_rank_cls, avg_rank_rgs]
    if avg_rank_row:
        out_rows.append(avg_rank_row)
    with table_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(out_rows)
    return cls_norm, rgs_norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default="benchmark_data/real_hpo_v2")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--exclude-datasets", default="covertype")
    parser.add_argument(
        "--include-datasets",
        default="",
        help=f"Comma-separated eval datasets (overrides pilot scope). Default eval8: {EVAL8_DATASETS}",
    )
    parser.add_argument("--ptllm-prompt", default="P5", help="Tuned pure-ptllm prompt (default P5).")
    parser.add_argument("--tpe-prompt", default="P5", help="Prompt family for ptllm_tpe hybrid sweep.")
    parser.add_argument(
        "--top-tpe-hybrids",
        type=int,
        default=3,
        help="Number of best ptllm_tpe hybrids to include (default 3).",
    )
    parser.add_argument(
        "--tpe-hybrids",
        default="",
        help="Comma-separated hybrid ids to force (e.g. H2,H0,H6); overrides auto-pick.",
    )
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    if args.include_datasets.strip():
        jobs = jobs_from_dataset_names(args.include_datasets)
    else:
        exclude = {x.strip() for x in args.exclude_datasets.split(",") if x.strip()}
        jobs = [(tt, ds) for tt, ds in list_v2_dataset_jobs("pilot") if ds not in exclude]
    dataset_keys = [f"{tt}/{ds}" for tt, ds in jobs]

    tuned_ptllm = f"ptllm__{args.ptllm_prompt}"
    if args.tpe_hybrids.strip():
        top_tpe = [
            f"ptllm_tpe__{args.tpe_prompt}__{h.strip()}"
            for h in args.tpe_hybrids.split(",")
            if h.strip()
        ]
    else:
        top_tpe = pick_top_tpe_hybrids(
            base,
            jobs,
            dataset_keys,
            tpe_prompt=args.tpe_prompt,
            ptllm_prompt=args.ptllm_prompt,
            top_k=args.top_tpe_hybrids,
        )

    variant_keys = BASELINE_VARIANTS + [tuned_ptllm] + top_tpe
    display_names = [variant_display_name(v) for v in variant_keys]

    out_dir = Path(args.out_dir or base / "analysis" / "no_covertype" / "baseline_vs_tuned")
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    want = set(variant_keys)
    _, _, curves, summary = collect_sweep(base, jobs, lambda name, w=want: name in w)
    table_variants = [v for v in variant_keys if any((d, v) in summary for d in dataset_keys)]

    missing = [v for v in variant_keys if v not in table_variants]
    if missing:
        print(f"[warn] missing variants (skipped): {missing}")

    table_path, mean_scores, avg_ranks = write_valid_best_table(
        out_dir, dataset_keys, table_variants, summary
    )
    cls_norm, rgs_norm = append_summary_rows(
        table_path, dataset_keys, table_variants, summary, display_names
    )

    colors = _variant_colors(table_variants)
    label_map = {v: display_names[variant_keys.index(v)] for v in table_variants}
    title_ds = f"{len(dataset_keys)} datasets"

    written = [table_path]
    for dataset in dataset_keys:
        written.extend(
            plot_dataset_convergence(plots_dir, dataset, table_variants, curves, colors, label_map)
        )
    written.extend(
        plot_normalized_task_bars(
            plots_dir,
            table_variants,
            cls_norm,
            rgs_norm,
            colors,
            f"Normalized mean score — baselines vs tuned ({title_ds})",
            label_map,
        )
    )
    written.extend(
        plot_mean_bar(
            plots_dir,
            table_variants,
            mean_scores,
            colors,
            f"Mean valid_best (raw, all tasks) — {title_ds}",
            label_map,
        )
    )
    written.extend(
        plot_avg_rank_bar(
            plots_dir,
            table_variants,
            avg_ranks,
            colors,
            "Average rank — baselines vs tuned",
            label_map,
        )
    )
    written.extend(
        plot_average_rank_curve(
            plots_dir,
            dataset_keys,
            table_variants,
            curves,
            colors,
            "Baselines vs tuned — avg rank vs trial",
            label_map,
        )
    )

    (out_dir / "selected_variants.txt").write_text(
        "\n".join(
            [
                f"ptllm_prompt={args.ptllm_prompt}",
                f"tpe_prompt={args.tpe_prompt}",
                f"top_tpe_hybrids={args.top_tpe_hybrids}",
                "variants:",
                *[f"  {v} -> {label_map.get(v, v)}" for v in table_variants],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(out_dir / "selected_variants.txt")

    print(f"Table: {table_path}")
    print(f"Tuned ptllm: {tuned_ptllm}")
    print(f"Top {len(top_tpe)} ptllm_tpe: {', '.join(top_tpe)}")
    print("\nSummary (avg_rank, mean_cls_norm):")
    for v in table_variants:
        label = label_map[v]
        ar = avg_ranks.get(v, float("nan"))
        cn = cls_norm.get(v, float("nan"))
        rn = rgs_norm.get(v, float("nan"))
        print(f"  {label:24s}  avg_rank={ar:.3f}  cls_norm={cn:.3f}  rgs_norm={rn:.3f}")

    print("\n[Written]")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
