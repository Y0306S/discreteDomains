"""P27 vs TPE breakdown + convergence plots for current top-5 by avg_rank."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

from experiment_utils import EVAL8_DATASETS, jobs_from_dataset_names, score_higher_is_better
from plot_baseline_vs_tuned import variant_display_name
from plot_pilot_sweep import (
    _ranks_for_task,
    _savefig,
    collect_sweep,
    plot_dataset_convergence,
)

TOP5 = [
    "tpe",
    "cmaes",
    "ptllm__P27",
    "ptllm_smac__P27__H2",
    "ptllm__P22",
]

FOCUS = ["tpe", "ptllm__P27", "cmaes"]


def _better(value_a: float, value_b: float, higher: bool) -> str:
    if math.isnan(value_a) or math.isnan(value_b):
        return "n/a"
    if higher:
        if value_a > value_b:
            return "P27"
        if value_b > value_a:
            return "TPE"
        return "tie"
    if value_a < value_b:
        return "P27"
    if value_b < value_a:
        return "TPE"
    return "tie"


def write_p27_vs_tpe_csv(
    out_dir: Path,
    dataset_keys: list[str],
    pool: list[str],
    summary: dict,
) -> Path:
    path = out_dir / "p27_vs_tpe_per_dataset.csv"
    rows = []
    for dataset in dataset_keys:
        task_type = dataset.split("/", 1)[0]
        higher = score_higher_is_better(task_type)
        tpe_v = summary.get((dataset, "tpe"), float("nan"))
        p27_v = summary.get((dataset, "ptllm__P27"), float("nan"))
        scores = {v: summary[(dataset, v)] for v in pool if (dataset, v) in summary}
        ranks = _ranks_for_task(scores, higher) if len(scores) >= 2 else {}
        rows.append(
            {
                "dataset": dataset,
                "task": task_type,
                "tpe_valid_best": tpe_v,
                "p27_valid_best": p27_v,
                "raw_winner": _better(p27_v, tpe_v, higher),
                "tpe_rank_in_pool": ranks.get("tpe", ""),
                "p27_rank_in_pool": ranks.get("ptllm__P27", ""),
                "pool_size": len(scores),
            }
        )
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for key in ("tpe_valid_best", "p27_valid_best"):
                v = row[key]
                row[key] = f"{v:.6f}" if isinstance(v, float) and not math.isnan(v) else ""
            writer.writerow(row)
    return path


def write_rank_pool_comparison(
    out_dir: Path,
    dataset_keys: list[str],
    pool_new: list[str],
    summary: dict,
) -> Path:
    """Show how avg_rank shifts when the comparison pool grows."""
    hybrids_new = [
        v
        for v in pool_new
        if v.startswith(("ptllm_tpe__P7", "ptllm_tpe__P27", "ptllm_smac__P7", "ptllm_smac__P27", "ptllm_cmaes__"))
    ]
    pool_old = [v for v in pool_new if v not in hybrids_new]

    def avg_rank(pool: list[str], variant: str) -> float:
        rank_sums = 0.0
        counts = 0
        for dataset in dataset_keys:
            task_type = dataset.split("/", 1)[0]
            higher = score_higher_is_better(task_type)
            scores = {v: summary[(dataset, v)] for v in pool if (dataset, v) in summary}
            if variant not in scores or len(scores) < 2:
                continue
            ranks = _ranks_for_task(scores, higher)
            rank_sums += ranks[variant]
            counts += 1
        return rank_sums / counts if counts else float("nan")

    path = out_dir / "rank_pool_comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "pool", "n_methods", "avg_rank"])
        for label, pool in [("before_hybrid_chase", pool_old), ("after_hybrid_chase", pool_new)]:
            for v in FOCUS:
                writer.writerow([v, label, len(pool), f"{avg_rank(pool, v):.4f}"])
    return path


def average_rank_over_trials(
    dataset_keys: list[str],
    pool: list[str],
    curves: dict,
    highlight: list[str],
    *,
    n_trials: int | None = None,
) -> tuple[list[int], dict[str, list[float]]]:
    """Mean rank across eval8 datasets at each trial (ranked within full pool)."""
    highlight_lengths = [
        len(curves[(d, v)])
        for d in dataset_keys
        for v in highlight
        if (d, v) in curves and curves[(d, v)]
    ]
    if not highlight_lengths:
        return [], {v: [] for v in highlight}
    max_trials = min(highlight_lengths)
    if n_trials is None:
        n_trials = max_trials
    else:
        n_trials = min(n_trials, max_trials)

    trials = list(range(1, n_trials + 1))
    mean_rank = {v: [] for v in highlight}

    for t in range(n_trials):
        rank_sums = {v: 0.0 for v in highlight}
        n_sets = 0
        for dataset in dataset_keys:
            task_type = dataset.split("/", 1)[0]
            higher = score_higher_is_better(task_type)
            scores = {}
            for v in pool:
                rows = curves.get((dataset, v))
                if rows and t < len(rows):
                    scores[v] = rows[t]["best_so_far"]
            if len(scores) < 2:
                continue
            ranks = _ranks_for_task(scores, higher)
            n_sets += 1
            for v in highlight:
                if v in ranks:
                    rank_sums[v] += ranks[v]
        for v in highlight:
            mean_rank[v].append(rank_sums[v] / n_sets if n_sets else float("nan"))
    return trials, mean_rank


def write_avg_rank_curve_csv(
    out_dir: Path,
    trials: list[int],
    mean_rank: dict[str, list[float]],
    labels: dict[str, str],
) -> Path:
    path = out_dir / "avg_rank_overall.csv"
    fields = ["trial"] + [labels[v] for v in mean_rank]
    variant_by_label = {labels[v]: v for v in mean_rank}
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, trial in enumerate(trials):
            row = {"trial": trial}
            for label, v in variant_by_label.items():
                val = mean_rank[v][i]
                row[label] = f"{val:.4f}" if val == val else ""
            writer.writerow(row)
    return path


def plot_avg_rank_line(
    plots_dir: Path,
    trials: list[int],
    mean_rank: dict[str, list[float]],
    colors: dict[str, object],
    labels: dict[str, str],
    *,
    n_pool: int,
):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    n_trials = len(trials)
    for v, ranks in mean_rank.items():
        ax.plot(
            trials,
            ranks,
            label=labels[v],
            color=colors[v],
            linewidth=2,
            marker="o",
            markersize=3,
            markevery=max(1, n_trials // 12),
        )
    ax.set_xlabel("Trial")
    ax.set_ylabel("Average rank across 8 datasets (1 = best)")
    ax.set_title(
        f"Overall avg rank over eval8 (top 5 shown, ranked in pool n={n_pool})",
        fontweight="bold",
    )
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = _savefig(fig, plots_dir, "avg_rank_overall")
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="benchmark_data/real_hpo_v2")
    parser.add_argument(
        "--out-dir",
        default="benchmark_data/real_hpo_v2/analysis/eval8/all_valid_best/p27_analysis",
    )
    parser.add_argument(
        "--per-dataset",
        action="store_true",
        help="Also write per-dataset best-so-far convergence plots.",
    )
    args = parser.parse_args()

    base = Path(args.base_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plots_dir = out / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    jobs = jobs_from_dataset_names(EVAL8_DATASETS)
    dataset_keys = [f"{t}/{d}" for t, d in jobs]
    _, _, curves, summary = collect_sweep(base, jobs, lambda _name: True)

    pool = sorted({v for (_, v) in summary})
    complete = [v for v in pool if sum(1 for d in dataset_keys if (d, v) in summary) >= len(dataset_keys)]

    cmp_path = write_p27_vs_tpe_csv(out, dataset_keys, complete, summary)
    pool_path = write_rank_pool_comparison(out, dataset_keys, complete, summary)

    colors = {v: plt.get_cmap("tab10")(i) for i, v in enumerate(TOP5)}
    labels = {v: variant_display_name(v) for v in TOP5}

    trials, mean_rank = average_rank_over_trials(
        dataset_keys, complete, curves, TOP5, n_trials=80
    )
    curve_csv = write_avg_rank_curve_csv(out, trials, mean_rank, labels)
    rank_line = plot_avg_rank_line(plots_dir, trials, mean_rank, colors, labels, n_pool=len(complete))

    # Convenience copy next to avg_rank_table.csv
    parent_plots = base / "analysis/eval8/all_valid_best"
    parent_plots.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        src = plots_dir / f"avg_rank_overall{ext}"
        if src.exists():
            shutil.copy2(src, parent_plots / f"avg_rank_overall_top5{ext}")
        shutil.copy2(out / "avg_rank_overall.csv", parent_plots / "avg_rank_overall_top5.csv")

    written = [rank_line]
    if args.per_dataset:
        for dataset in dataset_keys:
            written.extend(
                plot_dataset_convergence(plots_dir, dataset, TOP5, curves, colors, labels)
            )

    print(f"[Done] analysis -> {out}")
    print(f"  overall plot: {plots_dir}/avg_rank_overall.png")
    print(f"  also copied: {parent_plots}/avg_rank_overall_top5.png")
    print(f"  per-dataset: {cmp_path}")
    print(f"  pool shift:  {pool_path}")
    print(f"  avg_rank csv: {curve_csv}")
    for p in written:
        print(f"  plot: {p}")


if __name__ == "__main__":
    main()
