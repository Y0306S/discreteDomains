"""Aggregate pilot prompt/hybrid sweeps into valid_best_table.csv and plots.

Mirrors ``plot_real_benchmark.py`` handoff layout for a variant sweep:
  - valid_best_table.csv with avg_rank summary row
  - per-dataset convergence curves
  - mean valid-best bar chart
  - average-rank vs trial curve

Example:
  python plot_pilot_sweep.py --mode prompt --exclude-datasets covertype
  python plot_pilot_sweep.py --mode hybrid --prompt P5 --exclude-datasets covertype
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_utils import EVAL8_DATASETS, best_run_dir, jobs_from_dataset_names, list_v2_dataset_jobs, score_higher_is_better
from pick_pilot_winner import avg_rank, collect_variant_scores, prompt_scores_from_ptllm


def fnum(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def traj_label_for_dir(optimizer_dir_name: str) -> str:
    base = optimizer_dir_name.split("__", 1)[0]
    # Dispatch label used in traj CSV (see run_toy_categorical_optimizers.TRAJ_LABELS).
    if base.startswith("ptllm_tpe_qwen"):
        return "ptllm_tpe"
    if base.startswith("ptllm_qwen"):
        return "ptllm"
    return base


def load_trajectory(run_dir: Path, optimizer_dir_name: str):
    label = traj_label_for_dir(optimizer_dir_name)
    traj_files = sorted(run_dir.glob(f"*_opt_{label}_traj.csv"))
    if not traj_files:
        return []
    rows = []
    with traj_files[-1].open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "trial": int(row["iter_idx"]) + 1,
                    "best_so_far": fnum(row.get("best_so_far")),
                }
            )
    rows.sort(key=lambda r: r["trial"])
    return rows


def find_run_dir(dataset_dir: Path, variant: str):
    return best_run_dir(dataset_dir / variant)


def collect_sweep(
    base_dir: Path,
    jobs: list[tuple[str, str]],
    variant_filter,
):
    dataset_keys = [f"{tt}/{ds}" for tt, ds in jobs]
    variants = set()
    curves: dict[tuple[str, str], list] = {}
    summary: dict[tuple[str, str], float] = {}

    for task_type, dataset in jobs:
        key = f"{task_type}/{dataset}"
        ds_dir = base_dir / task_type / dataset
        if not ds_dir.is_dir():
            continue
        higher = score_higher_is_better(task_type)
        for opt_dir in sorted(ds_dir.iterdir()):
            if not opt_dir.is_dir():
                continue
            name = opt_dir.name
            if not variant_filter(name):
                continue
            run_dir = find_run_dir(ds_dir, name)
            if run_dir is None:
                continue
            rows = load_trajectory(run_dir, name)
            if not rows:
                continue
            variants.add(name)
            curves[(key, name)] = rows
            best = max(r["best_so_far"] for r in rows) if higher else min(r["best_so_far"] for r in rows)
            summary[(key, name)] = best

    return dataset_keys, sorted(variants), curves, summary


def _ranks_for_task(scores: dict[str, float], higher_is_better: bool):
    if higher_is_better:
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    else:
        ordered = sorted(scores.items(), key=lambda kv: kv[1])
    ranks = {}
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2
        for k in range(i, j + 1):
            ranks[ordered[k][0]] = avg_rank
        i = j + 1
    return ranks


def normalize_score(value: float, lo: float, hi: float, higher_is_better: bool) -> float:
    """Per-dataset min-max norm in [0, 1]; 1 = best among compared methods."""
    span = hi - lo
    if span <= 0:
        return 1.0
    if higher_is_better:
        return (value - lo) / span
    return (hi - value) / span


def task_normalized_means(
    dataset_keys: list[str],
    variants: list[str],
    summary: dict[tuple[str, str], float],
    task_prefix: str,
) -> dict[str, float]:
    """Mean of per-dataset normalized scores (1 = best on that dataset)."""
    keys = [d for d in dataset_keys if d.startswith(f"{task_prefix}/")]
    higher = score_higher_is_better(task_prefix)
    out: dict[str, list[float]] = {v: [] for v in variants}
    for dataset in keys:
        scores = {v: summary[(dataset, v)] for v in variants if (dataset, v) in summary}
        if len(scores) < 2:
            continue
        lo, hi = min(scores.values()), max(scores.values())
        for v, value in scores.items():
            out[v].append(normalize_score(value, lo, hi, higher))
    return {v: (sum(vals) / len(vals) if vals else float("nan")) for v, vals in out.items()}


def meta_normalize_across_methods(scores: dict[str, float]) -> dict[str, float]:
    """Rescale per-method task means to [0, 1] across the pool; 1 = best in pool."""
    finite = {v: s for v, s in scores.items() if not math.isnan(s)}
    if not finite:
        return {v: float("nan") for v in scores}
    lo, hi = min(finite.values()), max(finite.values())
    span = hi - lo
    out: dict[str, float] = {}
    for v in scores:
        s = scores[v]
        if math.isnan(s):
            out[v] = float("nan")
        elif span <= 0:
            out[v] = 1.0
        else:
            out[v] = (s - lo) / span
    return out


def balanced_meta_score(
    cls_task_norm: dict[str, float],
    rgs_task_norm: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Meta-normalize cls/rgs task means, then average → balanced score in [0, 1]."""
    meta_cls = meta_normalize_across_methods(cls_task_norm)
    meta_rgs = meta_normalize_across_methods(rgs_task_norm)
    balanced: dict[str, float] = {}
    for v in cls_task_norm:
        mc, mr = meta_cls.get(v, float("nan")), meta_rgs.get(v, float("nan"))
        if math.isnan(mc) or math.isnan(mr):
            balanced[v] = float("nan")
        else:
            balanced[v] = (mc + mr) / 2
    return meta_cls, meta_rgs, balanced


def write_valid_best_table(out_dir: Path, dataset_keys, variants, summary):
    table_path = out_dir / "valid_best_table.csv"
    mean_scores = {}
    rank_sums = {v: 0.0 for v in variants}
    rank_counts = {v: 0 for v in variants}

    for variant in variants:
        vals = [summary[(d, variant)] for d in dataset_keys if (d, variant) in summary]
        mean_scores[variant] = sum(vals) / len(vals) if vals else float("nan")

    for dataset in dataset_keys:
        task_type = dataset.split("/", 1)[0]
        higher = score_higher_is_better(task_type)
        scores = {v: summary[(dataset, v)] for v in variants if (dataset, v) in summary}
        if len(scores) < 2:
            continue
        for v, rank in _ranks_for_task(scores, higher).items():
            rank_sums[v] += rank
            rank_counts[v] += 1

    avg_ranks = {
        v: (rank_sums[v] / rank_counts[v] if rank_counts[v] else float("nan")) for v in variants
    }

    with table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset"] + variants)
        for dataset in dataset_keys:
            row = [dataset]
            for variant in variants:
                value = summary.get((dataset, variant), float("nan"))
                row.append("" if math.isnan(value) else f"{value:.6f}")
            writer.writerow(row)
        writer.writerow(
            ["avg_rank"]
            + ["" if math.isnan(avg_ranks[v]) else f"{avg_ranks[v]:.6f}" for v in variants]
        )
    return table_path, mean_scores, avg_ranks


def _variant_colors(variants):
    cmap = plt.get_cmap("tab10")
    if len(variants) > 10:
        cmap = plt.get_cmap("tab20")
    return {v: cmap(i % cmap.N) for i, v in enumerate(variants)}


def _savefig(fig, out_dir: Path, stem: str):
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    return [png, pdf]


def _variant_label(variant: str, labels: dict[str, str] | None = None) -> str:
    if labels and variant in labels:
        return labels[variant]
    if variant.startswith("ptllm_tpe__"):
        parts = variant.split("__")
        if len(parts) >= 3:
            return f"{parts[1]}/{parts[2]}"
    if variant.startswith("ptllm__"):
        parts = variant.split("__", 1)
        return parts[1] if len(parts) > 1 else variant
    return variant


def plot_dataset_convergence(plots_dir: Path, dataset: str, variants, curves, colors, labels=None):
    present = [v for v in variants if curves.get((dataset, v))]
    if not present:
        return []
    task_type = dataset.split("/", 1)[0]
    higher = score_higher_is_better(task_type)
    ylabel = "Best validation score (higher better)"

    fig, ax = plt.subplots(figsize=(10, 6))
    seen_y = []
    for i, variant in enumerate(present):
        rows = curves[(dataset, variant)]
        x = [r["trial"] for r in rows]
        y = [r["best_so_far"] for r in rows]
        seen_y.extend(y)
        label = _variant_label(variant, labels)
        ax.plot(
            x,
            y,
            label=label,
            color=colors[variant],
            linewidth=1.8,
            marker="o",
            markersize=3,
            markevery=max(1, len(x) // 12),
            alpha=0.9,
        )
    if seen_y:
        lo, hi = min(seen_y), max(seen_y)
        pad = max((hi - lo) * 0.08, 1e-6)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(f"{dataset}: best-so-far validation", fontweight="bold")
    ax.set_xlabel("Trial")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    safe = dataset.replace("/", "_")
    out_dir = plots_dir / "per_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = _savefig(fig, out_dir, safe)
    plt.close(fig)
    return out


def plot_mean_bar(plots_dir: Path, variants, mean_scores, colors, title: str, labels=None):
    return _plot_scalar_bar(
        plots_dir,
        variants,
        mean_scores,
        colors,
        title,
        labels,
        ylabel="Mean valid_best across datasets (raw)",
        stem="mean_valid_best_bar",
        fmt="{:.4f}",
    )


def plot_normalized_task_bars(
    plots_dir: Path,
    variants,
    cls_means: dict[str, float],
    rgs_means: dict[str, float],
    colors,
    title: str,
    labels=None,
):
    """Side-by-side normalized mean bars for cls and rgs (1 = best per dataset)."""
    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(variants) * 1.1), 6))
    tick_labels = [_variant_label(v, labels) for v in variants]
    xpos = list(range(len(variants)))

    for ax, means, task, subtitle in [
        (axes[0], cls_means, "cls", "classification"),
        (axes[1], rgs_means, "rgs", "regression"),
    ]:
        vals = [means.get(v, float("nan")) for v in variants]
        bars = ax.bar(xpos, vals, color=[colors[v] for v in variants], alpha=0.9)
        for bar, value in zip(bars, vals):
            if value == value:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_xticks(xpos)
        ax.set_xticklabels(tick_labels, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"Mean normalized score ({task})")
        ax.set_title(subtitle, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(title, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = _savefig(fig, plots_dir, "mean_normalized_cls_rgs_bar")
    plt.close(fig)
    return out


def _plot_scalar_bar(plots_dir, variants, mean_scores, colors, title, labels, ylabel, stem, fmt):
    fig, ax = plt.subplots(figsize=(max(8, len(variants) * 0.9), 6))
    tick_labels = [_variant_label(v, labels) for v in variants]
    vals = [mean_scores.get(v, float("nan")) for v in variants]
    xpos = list(range(len(variants)))
    bars = ax.bar(xpos, vals, color=[colors[v] for v in variants], alpha=0.9)
    for bar, value in zip(bars, vals):
        if value == value:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(xpos)
    ax.set_xticklabels(tick_labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = _savefig(fig, plots_dir, stem)
    plt.close(fig)
    return out


def plot_avg_rank_bar(plots_dir: Path, variants, avg_ranks, colors, title: str, labels=None):
    fig, ax = plt.subplots(figsize=(max(8, len(variants) * 0.9), 6))
    tick_labels = [_variant_label(v, labels) for v in variants]
    vals = [avg_ranks.get(v, float("nan")) for v in variants]
    xpos = list(range(len(variants)))
    bars = ax.bar(xpos, vals, color=[colors[v] for v in variants], alpha=0.9)
    for bar, value in zip(bars, vals):
        if value == value:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(xpos)
    ax.set_xticklabels(tick_labels, rotation=25, ha="right")
    ax.set_ylabel("Average rank (1 = best)")
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = _savefig(fig, plots_dir, "avg_rank_bar")
    plt.close(fig)
    return out


def plot_average_rank_curve(plots_dir: Path, dataset_keys, variants, curves, colors, title: str, labels=None):
    lengths = [len(rows) for rows in curves.values()]
    if not lengths:
        return []
    n_trials = min(lengths)
    trials = list(range(1, n_trials + 1))
    mean_rank = {v: [] for v in variants}

    for t in range(n_trials):
        rank_sums = {v: 0.0 for v in variants}
        n_sets = 0
        for dataset in dataset_keys:
            task_type = dataset.split("/", 1)[0]
            higher = score_higher_is_better(task_type)
            scores = {}
            for v in variants:
                rows = curves.get((dataset, v))
                if rows and t < len(rows):
                    scores[v] = rows[t]["best_so_far"]
            if len(scores) < 2:
                continue
            n_sets += 1
            for v, rank in _ranks_for_task(scores, higher).items():
                rank_sums[v] += rank
        for v in variants:
            mean_rank[v].append(rank_sums[v] / n_sets if n_sets else float("nan"))

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for v in variants:
        label = _variant_label(v, labels)
        ax.plot(
            trials,
            mean_rank[v],
            label=label,
            color=colors[v],
            linewidth=2,
            marker="o",
            markersize=3,
            markevery=max(1, n_trials // 12),
        )
    ax.set_xlabel("Trial")
    ax.set_ylabel("Average rank (1 = best)")
    ax.set_title(title, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = _savefig(fig, plots_dir, "average_rank_summary")
    plt.close(fig)
    return out


def prompt_filter(_name: str) -> bool:
    return _name.startswith("ptllm__") and not _name.startswith("ptllm_tpe")


def hybrid_filter(prompt: str):
    prefix = f"ptllm_tpe__{prompt}__"

    def _fn(name: str) -> bool:
        return name.startswith(prefix)

    return _fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default="benchmark_data/real_hpo_v2")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--mode", choices=["prompt", "hybrid"], required=True)
    parser.add_argument("--prompt", default="P5", help="Fixed prompt for hybrid mode (default P5).")
    parser.add_argument(
        "--exclude-datasets",
        default="covertype",
        help="Comma-separated datasets to exclude (default: covertype).",
    )
    parser.add_argument(
        "--include-datasets",
        default="",
        help=f"Comma-separated datasets (overrides pilot). Eval8: {EVAL8_DATASETS}",
    )
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    if args.include_datasets.strip():
        jobs = jobs_from_dataset_names(args.include_datasets)
        exclude = set()
    else:
        exclude = {x.strip() for x in args.exclude_datasets.split(",") if x.strip()}
        jobs = [(tt, ds) for tt, ds in list_v2_dataset_jobs("pilot") if ds not in exclude]

    if args.mode == "prompt":
        out_dir = Path(args.out_dir or base / "analysis" / "no_covertype" / "stage1_prompts")
        vfilter = prompt_filter
        title_tag = "Stage 1 pilot prompts (ptllm, no covertype)"
    else:
        out_dir = Path(
            args.out_dir or base / "analysis" / "no_covertype" / f"stage2_hybrids_{args.prompt}"
        )
        vfilter = hybrid_filter(args.prompt)
        title_tag = f"Stage 2 pilot hybrids (ptllm_tpe {args.prompt}, no covertype)"

    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataset_keys, variants, curves, summary = collect_sweep(base, jobs, vfilter)
    if not summary:
        raise SystemExit(f"No results for {args.mode} under {base} (exclude={exclude})")

    table_path, mean_scores, avg_ranks = write_valid_best_table(out_dir, dataset_keys, variants, summary)
    colors = _variant_colors(variants)

    written = [table_path]
    for dataset in dataset_keys:
        written.extend(plot_dataset_convergence(plots_dir, dataset, variants, curves, colors))
    written.extend(plot_mean_bar(plots_dir, variants, mean_scores, colors, title_tag))
    written.extend(plot_avg_rank_bar(plots_dir, variants, avg_ranks, colors, f"{title_tag} — avg rank"))
    written.extend(
        plot_average_rank_curve(plots_dir, dataset_keys, variants, curves, colors, title_tag)
    )

    # Winner by avg_rank (same logic as pick_pilot_winner)
    if args.mode == "prompt":
        merged = prompt_scores_from_ptllm(collect_variant_scores(base, "ptllm__", jobs))
        ranks = avg_rank(merged)
        winner = min(ranks.items(), key=lambda kv: kv[1])[0] if ranks else "?"
    else:
        scores = collect_variant_scores(base, "ptllm_tpe__", jobs)
        filtered = {k: v for k, v in scores.items() if k.startswith(f"ptllm_tpe__{args.prompt}__")}
        ranks = avg_rank(filtered)
        winner_full = min(ranks.items(), key=lambda kv: kv[1])[0] if ranks else "?"
        winner = winner_full.split("__")[-1] if winner_full != "?" else "?"

    winner_path = out_dir / "winner.txt"
    winner_path.write_text(
        f"mode={args.mode}\nexclude={','.join(sorted(exclude))}\nwinner={winner}\n",
        encoding="utf-8",
    )
    written.append(winner_path)

    print(f"[{title_tag}]")
    print(f"  datasets: {', '.join(dataset_keys)}")
    print(f"  variants: {len(variants)}")
    print(f"  winner: {winner}")
    print(f"  table: {table_path}")
    print("\n[Written]")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
