"""Compare TCS (P16/H2) against eval8 baselines + tuned P5 variants."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from experiment_utils import EVAL8_DATASETS, jobs_from_dataset_names
from plot_baseline_vs_tuned import append_summary_rows, variant_display_name
from plot_pilot_sweep import (
    _variant_colors,
    collect_sweep,
    plot_avg_rank_bar,
    plot_mean_bar,
    plot_normalized_task_bars,
    write_valid_best_table,
)

BASELINE_VARIANTS = [
    "smac",
    "tpe",
    "cmaes",
    "ptllm__P0",
    "ptllm_tpe__P0__H0",
]
TUNED_VARIANTS = [
    "ptllm__P5",
    "ptllm_tpe__P5__H2",
    "ptllm_tpe__P16__H2",
]
ALL_VARIANTS = BASELINE_VARIANTS + TUNED_VARIANTS


def main():
    parser = argparse.ArgumentParser(description="Plot TCS eval vs baselines on eval8.")
    parser.add_argument("--base-dir", default="benchmark_data/real_hpo_v2")
    parser.add_argument("--include-datasets", default=EVAL8_DATASETS)
    parser.add_argument("--out-dir", default="benchmark_data/real_hpo_v2/analysis/eval8/tcs_eval")
    args = parser.parse_args()

    base = Path(args.base_dir)
    out = Path(args.out_dir)
    plots_dir = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    jobs = jobs_from_dataset_names(args.include_datasets)
    dataset_keys = [f"{t}/{d}" for t, d in jobs]
    want = set(ALL_VARIANTS)

    _, _, _, summary = collect_sweep(base, jobs, lambda name, w=want: name in w)
    present = [v for v in ALL_VARIANTS if any((d, v) in summary for d in dataset_keys)]
    missing = [v for v in ALL_VARIANTS if v not in present]
    if missing:
        print(f"[warn] missing variants (skipped): {missing}")

    display_names = [variant_display_name(v) for v in present]
    label_map = {v: variant_display_name(v) for v in present}
    colors = _variant_colors(present)

    table_path, mean_scores, avg_ranks = write_valid_best_table(out, dataset_keys, present, summary)
    cls_norm, rgs_norm = append_summary_rows(
        table_path, dataset_keys, present, summary, display_names
    )

    title_ds = f"{len(dataset_keys)} datasets"
    plot_normalized_task_bars(
        plots_dir,
        present,
        cls_norm,
        rgs_norm,
        colors,
        f"TCS eval — baselines + P5 + P16 ({title_ds})",
        label_map,
    )
    plot_mean_bar(
        plots_dir,
        present,
        mean_scores,
        colors,
        f"Mean valid_best (raw) — baselines + P16 ({title_ds})",
        label_map,
    )
    plot_avg_rank_bar(
        plots_dir,
        present,
        avg_ranks,
        colors,
        "Average rank — baselines + P16 TCS",
        label_map,
    )

    (out / "variants.txt").write_text(
        "\n".join(f"{v}\t{label_map[v]}" for v in present) + "\n",
        encoding="utf-8",
    )

    print(f"[Done] wrote TCS analysis to {out}")
    print(f"  table: {table_path}")
    for v in present:
        print(f"  {label_map[v]}: avg_rank={avg_ranks.get(v, float('nan')):.4f}")


if __name__ == "__main__":
    main()
