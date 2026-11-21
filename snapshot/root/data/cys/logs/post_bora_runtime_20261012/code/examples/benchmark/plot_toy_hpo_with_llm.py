"""Compare the five optimizers on the per-algorithm toy HPO benchmark.

The companion runner ``run_toy_categorical_optimizers.py`` is launched once per
algorithm, and for every algorithm it sweeps five optimizers:

    smac, tpe, cmaes, ptllm, ptllm_tpe

Results land under::

    <base-dir>/<algorithm>/<optimizer>/CASH-*/best_model_info.json
    <base-dir>/<algorithm>/<optimizer>/CASH-*/*_opt_<optimizer>_traj.csv

``best_model_info.json`` carries the held-out ``test_perf`` and an
``opt_trajectory`` summary; the trajectory CSV carries the per-trial
``perf`` / ``best_so_far`` convergence curve. This script reads both directly
(no pre-aggregated CSV needed), writes a tidy long-form table, and renders:

  * best-so-far convergence per algorithm (2x2 grid, five curves each),
  * per-trial validation score per algorithm (2x2 grid),
  * grouped bar charts of the final validation best and held-out test score.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ALGORITHMS = [
    "k_nearest_neighbors",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
]

# Order matters: it drives plot legend order and bar offsets.
OPTIMIZERS = ["smac", "tpe", "cmaes", "ptllm", "ptllm_tpe"]

# Logical optimizer key -> on-disk directory / trajectory-file name. The runner
# names artifacts after the dispatch name (run dir = optimizer_name, traj file =
# *_opt_<optimizer_label>_traj.csv), so for the renamed LLM optimizers this is the
# identity. Override here only if on-disk names ever diverge from the plot keys.
DIRNAMES = {
    "smac": "smac",
    "tpe": "tpe",
    "cmaes": "cmaes",
    "ptllm": "ptllm",
    "ptllm_tpe": "ptllm_tpe",
}

COLORS = {
    "smac": "#2f6fbb",
    "tpe": "#2a9d8f",
    "cmaes": "#e76f51",
    "ptllm": "#7b2cbf",
    "ptllm_tpe": "#e09f3e",
}
MARKERS = {
    "smac": "o",
    "tpe": "s",
    "cmaes": "^",
    "ptllm": "D",
    "ptllm_tpe": "v",
}
LABELS = {
    "smac": "SMAC",
    "tpe": "TPE",
    "cmaes": "CMA-ES",
    "ptllm": "PT4Opt-LLM",
    "ptllm_tpe": "PT4Opt-LLM+TPE",
}


def fnum(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def find_run_dir(base_dir, algorithm, optimizer):
    """Return the most recent CASH-* run dir for one (algorithm, optimizer)."""
    opt_dir = base_dir / algorithm / DIRNAMES[optimizer]
    if not opt_dir.is_dir():
        return None
    candidates = sorted(p for p in opt_dir.glob("CASH-*") if p.is_dir())
    return candidates[-1] if candidates else None


def load_trajectory(run_dir, optimizer):
    """Read the per-trial convergence curve from the optimizer trajectory CSV."""
    traj_files = sorted(run_dir.glob(f"*_opt_{DIRNAMES[optimizer]}_traj.csv"))
    if not traj_files:
        return []
    rows = []
    with traj_files[-1].open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "trial": int(row["iter_idx"]) + 1,
                    "perf": fnum(row.get("perf")),
                    "best_so_far": fnum(row.get("best_so_far")),
                    "wall_elapsed_sec": fnum(row.get("wall_elapsed_sec")),
                }
            )
    rows.sort(key=lambda r: r["trial"])
    return rows


def load_test_perf(run_dir):
    info_path = run_dir / "best_model_info.json"
    if not info_path.is_file():
        return float("nan")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    return fnum(info.get("test_perf"))


def collect_results(base_dir):
    """Return (curves, summary) discovered from the benchmark directory tree.

    curves:  {(algorithm, optimizer): [trial rows]}
    summary: {(algorithm, optimizer): {"valid_best", "test_perf", "n_evals"}}
    """
    curves = {}
    summary = {}
    for algorithm in ALGORITHMS:
        for optimizer in OPTIMIZERS:
            run_dir = find_run_dir(base_dir, algorithm, optimizer)
            if run_dir is None:
                print(f"[skip] no run dir for {algorithm}/{optimizer}")
                continue
            rows = load_trajectory(run_dir, optimizer)
            if not rows:
                print(f"[skip] no trajectory csv for {algorithm}/{optimizer}")
                continue
            curves[(algorithm, optimizer)] = rows
            valid_best = max(r["best_so_far"] for r in rows)
            summary[(algorithm, optimizer)] = {
                "valid_best": valid_best,
                "test_perf": load_test_perf(run_dir),
                "n_evals": len(rows),
            }
    return curves, summary


def write_long_csv(base_dir, curves, summary):
    curve_path = base_dir / "convergence_curves_all.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "optimizer", "trial", "perf", "best_so_far", "wall_elapsed_sec"])
        for (algorithm, optimizer), rows in curves.items():
            for r in rows:
                writer.writerow(
                    [algorithm, optimizer, r["trial"], r["perf"], r["best_so_far"], r["wall_elapsed_sec"]]
                )

    summary_path = base_dir / "summary_table_all.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "optimizer", "valid_best", "test_perf", "n_evals"])
        for (algorithm, optimizer), s in summary.items():
            writer.writerow([algorithm, optimizer, s["valid_best"], s["test_perf"], s["n_evals"]])

    return [curve_path, summary_path]


def _zoom_ylim(ax, all_y):
    """Zoom the y-axis to the spread of the data so small gaps become visible."""
    vals = [v for v in all_y if v == v]  # drop NaN
    if not vals:
        return
    lo, hi = min(vals), max(vals)
    pad = max((hi - lo) * 0.08, 1e-3)
    ax.set_ylim(lo - pad, hi + pad)


def plot_curves(base_dir, curves, metric, out_stem, ylabel, title, markevery=8):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.ravel()
    for ax, algorithm in zip(axes, ALGORITHMS):
        seen_y = []
        for offset, optimizer in enumerate(OPTIMIZERS):
            rows = curves.get((algorithm, optimizer))
            if not rows:
                continue
            x = [r["trial"] for r in rows]
            y = [r[metric] for r in rows]
            seen_y.extend(y)
            ax.plot(
                x,
                y,
                label=LABELS[optimizer],
                color=COLORS[optimizer],
                linestyle="-",
                linewidth=1.8,
                marker=MARKERS[optimizer],
                markersize=5,
                markevery=(offset, markevery),  # stagger markers so they don't overlap
                markeredgecolor="white",
                markeredgewidth=0.5,
                alpha=0.9,
            )
        _zoom_ylim(ax, seen_y)
        ax.set_title(algorithm, fontweight="bold")
        ax.set_xlabel("Trial")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", ncol=len(OPTIMIZERS), frameon=False)
    fig.suptitle(title, y=0.99, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))

    out = _savefig(fig, base_dir, out_stem)
    plt.close(fig)
    return out


def average_rank_over_trials(curves):
    """Mean rank (1=best) of each optimizer at every trial, averaged over algorithms.

    At each trial, optimizers are ranked by best-so-far within an algorithm
    (higher score = better = lower rank), with ties getting the average rank.
    The per-algorithm ranks are then averaged across all algorithms that have a
    curve for every optimizer at that trial.
    """
    # Longest common trial index available for all (algorithm, optimizer) cells.
    lengths = [len(rows) for rows in curves.values()]
    if not lengths:
        return [], {}
    n_trials = min(lengths)

    mean_rank = {opt: [] for opt in OPTIMIZERS}
    trials = list(range(1, n_trials + 1))

    for t in range(n_trials):
        # rank_sums[opt] accumulates this optimizer's rank across algorithms.
        rank_sums = {opt: 0.0 for opt in OPTIMIZERS}
        n_algos = 0
        for algorithm in ALGORITHMS:
            scores = {}
            for opt in OPTIMIZERS:
                rows = curves.get((algorithm, opt))
                if rows and t < len(rows):
                    scores[opt] = rows[t]["best_so_far"]
            if len(scores) != len(OPTIMIZERS):
                continue  # only rank algorithms with all optimizers present
            n_algos += 1
            for opt, rank in _ranks_desc(scores).items():
                rank_sums[opt] += rank
        for opt in OPTIMIZERS:
            mean_rank[opt].append(rank_sums[opt] / n_algos if n_algos else float("nan"))

    return trials, mean_rank


def _ranks_desc(scores):
    """Rank a {key: value} mapping, 1=highest value, average rank for ties."""
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ranks = {}
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2  # positions i..j share the average rank
        for k in range(i, j + 1):
            ranks[ordered[k][0]] = avg_rank
        i = j + 1
    return ranks


def plot_average_rank(base_dir, curves, out_stem):
    trials, mean_rank = average_rank_over_trials(curves)
    if not trials:
        return []

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for optimizer in OPTIMIZERS:
        ax.plot(
            trials,
            mean_rank[optimizer],
            label=LABELS[optimizer],
            color=COLORS[optimizer],
            linestyle="-",
            linewidth=2,
            marker=MARKERS[optimizer],
            markersize=5,
            markevery=6,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )
    ax.set_xlabel("Trial")
    ax.set_ylabel("Average rank (1 = best)")
    ax.set_title(
        f"Average rank across {len(ALGORITHMS)} algorithms vs. trial (lower is better)",
        fontweight="bold",
    )
    ax.invert_yaxis()  # best (rank 1) on top
    ax.set_yticks(range(1, len(OPTIMIZERS) + 1))
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=len(OPTIMIZERS), frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    out = _savefig(fig, base_dir, out_stem)
    plt.close(fig)
    return out


def plot_final_scores(base_dir, summary, key, out_stem, ylabel, title):
    fig, ax = plt.subplots(figsize=(13, 6))
    width = 0.16
    xpos = list(range(len(ALGORITHMS)))
    n = len(OPTIMIZERS)
    offsets = [(i - (n - 1) / 2) * width for i in range(n)]

    for optimizer, offset in zip(OPTIMIZERS, offsets):
        values = [
            summary.get((algorithm, optimizer), {}).get(key, float("nan"))
            for algorithm in ALGORITHMS
        ]
        bars = ax.bar(
            [x + offset for x in xpos],
            values,
            width=width,
            label=LABELS[optimizer],
            color=COLORS[optimizer],
            alpha=0.9,
        )
        for bar, value in zip(bars, values):
            if value == value:  # not NaN
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.003,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    ax.set_xticks(xpos)
    ax.set_xticklabels(ALGORITHMS, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=len(OPTIMIZERS), frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.30))
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    out = _savefig(fig, base_dir, out_stem)
    plt.close(fig)
    return out


def _savefig(fig, base_dir, out_stem):
    png = base_dir / f"{out_stem}.png"
    pdf = base_dir / f"{out_stem}.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    return [png, pdf]


def print_summary_table(summary):
    print("\n[Summary] final validation best / held-out test score")
    header = f"{'algorithm':<22}" + "".join(f"{LABELS[o]:>18}" for o in OPTIMIZERS)
    print(header)
    for algorithm in ALGORITHMS:
        cells = []
        for optimizer in OPTIMIZERS:
            s = summary.get((algorithm, optimizer))
            if s is None:
                cells.append(f"{'-':>18}")
            else:
                cells.append(f"{s['valid_best']:.3f}/{s['test_perf']:.3f}".rjust(18))
        print(f"{algorithm:<22}" + "".join(cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        default="benchmark_data/toy_hpo_per_algorithm_80trials_20260624",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    curves, summary = collect_results(base_dir)
    if not curves:
        raise SystemExit(f"No optimizer trajectories found under {base_dir}")

    print_summary_table(summary)

    written = []
    written.extend(write_long_csv(base_dir, curves, summary))
    written.extend(
        plot_curves(
            base_dir,
            curves,
            "best_so_far",
            "best_so_far_by_algorithm_all",
            "Best validation score so far",
            "Best-so-far convergence by algorithm (5 optimizers)",
        )
    )
    written.extend(
        plot_curves(
            base_dir,
            curves,
            "perf",
            "trial_scores_by_algorithm_all",
            "Validation score",
            "Per-trial validation score by algorithm (5 optimizers)",
        )
    )
    written.extend(
        plot_final_scores(
            base_dir,
            summary,
            "valid_best",
            "final_valid_bar_all",
            "Final best validation score",
            "Final best validation score by algorithm (5 optimizers)",
        )
    )
    written.extend(plot_average_rank(base_dir, curves, "average_rank_summary"))

    print("\n[Written]")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
