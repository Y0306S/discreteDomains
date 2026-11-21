"""Compare P5 vs P17-P25 prompt-block ablations on eval8."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from experiment_utils import EVAL8_DATASETS, jobs_from_dataset_names
from plot_pilot_sweep import (
    _variant_colors,
    collect_sweep,
    plot_avg_rank_bar,
    plot_normalized_task_bars,
    task_normalized_means,
    write_valid_best_table,
)

PTLLM_ABLATIONS = [f"ptllm__P{i}" for i in (17, 18, 19, 20, 22, 23, 24, 26, 27, 28, 29)]
HYBRID_ABLATIONS = [f"ptllm_tpe__P{i}__H2" for i in (21, 25)] + ["ptllm_tpe__P22__H2", "ptllm_smac__P22__H2"]
BASELINES = ["ptllm__P5", "ptllm_tpe__P5__H2"]


def variant_label(name: str) -> str:
    if name == "ptllm__P5":
        return "ptllm (P5 baseline)"
    if name == "ptllm_tpe__P5__H2":
        return "ptllm_tpe (P5/H2 baseline)"
    if name.startswith("ptllm_smac__"):
        parts = name.split("__")
        return f"ptllm_smac ({parts[1]}/{parts[2]})"
    if name.startswith("ptllm_tpe__"):
        parts = name.split("__")
        return f"ptllm_tpe ({parts[1]}/{parts[2]})"
    if name.startswith("ptllm__"):
        return f"ptllm ({name.split('__', 1)[1]})"
    return name


def main():
    parser = argparse.ArgumentParser(description="Plot P5 prompt ablations on eval8.")
    parser.add_argument("--base-dir", default="benchmark_data/real_hpo_v2")
    parser.add_argument("--include-datasets", default=EVAL8_DATASETS)
    parser.add_argument("--out-dir", default="benchmark_data/real_hpo_v2/analysis/eval8/p5_ablations")
    args = parser.parse_args()

    base = Path(args.base_dir)
    out = Path(args.out_dir)
    plots_dir = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    jobs = jobs_from_dataset_names(args.include_datasets)
    dataset_keys = [f"{t}/{d}" for t, d in jobs]
    all_variants = BASELINES + PTLLM_ABLATIONS + HYBRID_ABLATIONS

    _, present, _, summary = collect_sweep(
        base, jobs, lambda name, s=set(all_variants): name in s
    )
    present = [v for v in all_variants if v in present]
    if not present:
        print("No ablation variants found under", base)
        return

    label_map = {v: variant_label(v) for v in present}
    colors = _variant_colors(present)

    table_path, _, avg_ranks = write_valid_best_table(out, dataset_keys, present, summary)
    cls_norm = task_normalized_means(dataset_keys, present, summary, "cls")
    rgs_norm = task_normalized_means(dataset_keys, present, summary, "rgs")
    plot_normalized_task_bars(
        plots_dir,
        present,
        cls_norm,
        rgs_norm,
        colors,
        f"P5 prompt ablations — normalized mean ({len(dataset_keys)} datasets)",
        label_map,
    )
    plot_avg_rank_bar(
        plots_dir,
        present,
        avg_ranks,
        colors,
        f"P5 prompt ablations — average rank ({len(dataset_keys)} datasets)",
        label_map,
    )

    with (out / "variants.txt").open("w", encoding="utf-8") as f:
        for v in present:
            f.write(f"{v}\n")

    print(f"[Done] wrote analysis to {out} ({len(present)} variants, table={table_path})")


if __name__ == "__main__":
    main()
