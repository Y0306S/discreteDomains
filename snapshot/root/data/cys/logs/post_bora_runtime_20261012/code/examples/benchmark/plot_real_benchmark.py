"""Aggregate Phase 0 real-dataset cls benchmark results and render plots.

Reads per-dataset outputs from ``run_toy_categorical_optimizers.py``:

    <base-dir>/{dataset}/{optimizer}/CASH-*/..._opt_{traj_label}_traj.csv

Writes ``valid_best_table.csv``, convergence plots, and optional handoff package.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_utils import best_run_dir, score_higher_is_better


OPTIMIZERS_BASELINE = ["smac", "tpe", "cmaes", "ptllm", "ptllm_tpe"]
OPTIMIZERS_QWEN = ["ptllm_qwen"]
OPTIMIZERS = OPTIMIZERS_BASELINE + OPTIMIZERS_QWEN

DIRNAMES = {name: name for name in OPTIMIZERS}

# Trajectory CSV suffix uses the dispatch optimizer_label, not the variant dir name.
TRAJ_LABELS = {
    "smac": "smac",
    "tpe": "tpe",
    "cmaes": "cmaes",
    "ptllm": "ptllm",
    "ptllm_tpe": "ptllm_tpe",
    "ptllm_qwen": "ptllm",
    "ptllm_tpe_qwen": "ptllm_tpe",
}

COLORS = {
    "smac": "#2f6fbb",
    "tpe": "#2a9d8f",
    "cmaes": "#e76f51",
    "ptllm": "#7b2cbf",
    "ptllm_tpe": "#e09f3e",
    "ptllm_qwen": "#9b2226",
    "ptllm_tpe_qwen": "#ee9b00",
}
MARKERS = {
    "smac": "o",
    "tpe": "s",
    "cmaes": "^",
    "ptllm": "D",
    "ptllm_tpe": "v",
    "ptllm_qwen": "P",
    "ptllm_tpe_qwen": "X",
}
LABELS = {
    "smac": "SMAC",
    "tpe": "TPE",
    "cmaes": "CMA-ES",
    "ptllm": "PT4Opt-LLM (4o-mini)",
    "ptllm_tpe": "PT4Opt-LLM+TPE (4o-mini)",
    "ptllm_qwen": "PT4Opt-LLM (Qwen SFT6)",
    "ptllm_tpe_qwen": "PT4Opt-LLM+TPE (Qwen SFT6)",
}


def fnum(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def list_datasets(base_dir: Path):
    datasets = []
    for path in sorted(base_dir.iterdir()):
        if path.is_dir() and (path / "summary.json").is_file():
            datasets.append(path.name)
    return datasets


def find_run_dir(base_dir, dataset, optimizer):
    opt_dir = base_dir / dataset / DIRNAMES[optimizer]
    if not opt_dir.is_dir():
        return None
    candidates = sorted(p for p in opt_dir.glob("CASH-*") if p.is_dir())
    return candidates[-1] if candidates else None


def load_trajectory(run_dir, optimizer):
    label = TRAJ_LABELS[optimizer]
    traj_files = sorted(run_dir.glob(f"*_opt_{label}_traj.csv"))
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


def list_datasets_v2(base_dir: Path):
    jobs = []
    for task_type in ("cls", "rgs"):
        task_root = base_dir / task_type
        if not task_root.is_dir():
            continue
        for path in sorted(task_root.iterdir()):
            if path.is_dir() and (path / "summary.json").is_file():
                jobs.append((task_type, path.name, path))
    return jobs


def list_optimizer_dirs(dataset_dir: Path):
    names = []
    for path in sorted(dataset_dir.iterdir()):
        if path.is_dir() and any(path.glob("CASH-*")):
            names.append(path.name)
    return names


def traj_label_for_dir(optimizer_dir_name: str) -> str:
    base = optimizer_dir_name.split("__", 1)[0]
    return TRAJ_LABELS.get(base, base)


def find_run_dir_v2(dataset_dir: Path, optimizer: str):
    return best_run_dir(dataset_dir / optimizer, traj_labels=TRAJ_LABELS)


def load_trajectory_v2(run_dir, optimizer_dir_name: str):
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
                    "perf": fnum(row.get("perf")),
                    "best_so_far": fnum(row.get("best_so_far")),
                    "wall_elapsed_sec": fnum(row.get("wall_elapsed_sec")),
                }
            )
    rows.sort(key=lambda r: r["trial"])
    return rows


def collect_results_v2(base_dir: Path):
    jobs = list_datasets_v2(base_dir)
    curves = {}
    summary = {}
    dataset_keys = []
    optimizers = set()
    for task_type, dataset_name, dataset_dir in jobs:
        key = f"{task_type}/{dataset_name}"
        dataset_keys.append(key)
        higher_is_better = score_higher_is_better(task_type)
        for optimizer in list_optimizer_dirs(dataset_dir):
            optimizers.add(optimizer)
            run_dir = find_run_dir_v2(dataset_dir, optimizer)
            if run_dir is None:
                continue
            rows = load_trajectory_v2(run_dir, optimizer)
            if not rows:
                continue
            curves[(key, optimizer)] = rows
            best = max(r["best_so_far"] for r in rows) if higher_is_better else min(r["best_so_far"] for r in rows)
            summary[(key, optimizer)] = {
                "valid_best": best,
                "n_evals": len(rows),
                "task_type": task_type,
            }
    return dataset_keys, sorted(optimizers), curves, summary


def _ranks_for_task(scores: dict, higher_is_better: bool):
    if higher_is_better:
        return _ranks_desc(scores)
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


def write_valid_best_table_v2(base_dir, dataset_keys, optimizers, summary):
    table_path = base_dir / "valid_best_table.csv"
    mean_scores = {opt: float("nan") for opt in optimizers}
    rank_sums = {opt: 0.0 for opt in optimizers}
    rank_counts = {opt: 0 for opt in optimizers}

    for optimizer in optimizers:
        vals = [
            summary[(dataset, optimizer)]["valid_best"]
            for dataset in dataset_keys
            if (dataset, optimizer) in summary
        ]
        mean_scores[optimizer] = sum(vals) / len(vals) if vals else float("nan")

    for dataset in dataset_keys:
        task_type = dataset.split("/", 1)[0]
        higher = score_higher_is_better(task_type)
        scores = {
            opt: summary[(dataset, opt)]["valid_best"]
            for opt in optimizers
            if (dataset, opt) in summary
        }
        if len(scores) < 2:
            continue
        for opt, rank in _ranks_for_task(scores, higher).items():
            rank_sums[opt] += rank
            rank_counts[opt] += 1

    avg_ranks = {
        opt: (rank_sums[opt] / rank_counts[opt] if rank_counts[opt] else float("nan"))
        for opt in optimizers
    }

    with table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset"] + optimizers)
        for dataset in dataset_keys:
            row = [dataset]
            for optimizer in optimizers:
                value = summary.get((dataset, optimizer), {}).get("valid_best", float("nan"))
                row.append("" if math.isnan(value) else f"{value:.6f}")
            writer.writerow(row)
        writer.writerow(
            ["mean_score"]
            + ["" if math.isnan(mean_scores[o]) else f"{mean_scores[o]:.6f}" for o in optimizers]
        )
        writer.writerow(
            ["avg_rank"]
            + ["" if math.isnan(avg_ranks[o]) else f"{avg_ranks[o]:.6f}" for o in optimizers]
        )
    return table_path


def collect_results(base_dir):
    datasets = list_datasets(base_dir)
    curves = {}
    summary = {}
    for dataset in datasets:
        for optimizer in OPTIMIZERS:
            run_dir = find_run_dir(base_dir, dataset, optimizer)
            if run_dir is None:
                continue
            rows = load_trajectory(run_dir, optimizer)
            if not rows:
                continue
            curves[(dataset, optimizer)] = rows
            summary[(dataset, optimizer)] = {
                "valid_best": max(r["best_so_far"] for r in rows),
                "n_evals": len(rows),
            }
    return datasets, curves, summary


def write_valid_best_table(base_dir, datasets, summary):
    table_path = base_dir / "valid_best_table.csv"
    mean_scores = {}
    rank_sums = {optimizer: 0.0 for optimizer in OPTIMIZERS}
    rank_counts = {optimizer: 0 for optimizer in OPTIMIZERS}

    for optimizer in OPTIMIZERS:
        vals = [
            summary[(dataset, optimizer)]["valid_best"]
            for dataset in datasets
            if (dataset, optimizer) in summary
            and not math.isnan(summary[(dataset, optimizer)]["valid_best"])
        ]
        mean_scores[optimizer] = sum(vals) / len(vals) if vals else float("nan")

    for dataset in datasets:
        scores = {}
        for optimizer in OPTIMIZERS:
            entry = summary.get((dataset, optimizer))
            if not entry:
                continue
            value = entry.get("valid_best", float("nan"))
            if math.isnan(value):
                continue
            scores[optimizer] = value
        if len(scores) < 2:
            continue
        for optimizer, rank in _ranks_desc(scores).items():
            rank_sums[optimizer] += rank
            rank_counts[optimizer] += 1

    avg_ranks = {
        optimizer: (rank_sums[optimizer] / rank_counts[optimizer] if rank_counts[optimizer] else float("nan"))
        for optimizer in OPTIMIZERS
    }

    with table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset"] + OPTIMIZERS)
        for dataset in datasets:
            row = [dataset]
            for optimizer in OPTIMIZERS:
                value = summary.get((dataset, optimizer), {}).get("valid_best", float("nan"))
                row.append("" if math.isnan(value) else f"{value:.6f}")
            writer.writerow(row)
        writer.writerow(
            ["mean_score"]
            + [
                "" if math.isnan(mean_scores[optimizer]) else f"{mean_scores[optimizer]:.6f}"
                for optimizer in OPTIMIZERS
            ]
        )
        writer.writerow(
            ["avg_rank"]
            + [
                "" if math.isnan(avg_ranks[optimizer]) else f"{avg_ranks[optimizer]:.6f}"
                for optimizer in OPTIMIZERS
            ]
        )
    return table_path


def write_long_csv(base_dir, datasets, curves):
    curve_path = base_dir / "convergence_curves_all.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "optimizer", "trial", "perf", "best_so_far", "wall_elapsed_sec"])
        for dataset in datasets:
            for optimizer in OPTIMIZERS:
                rows = curves.get((dataset, optimizer), [])
                for row in rows:
                    writer.writerow(
                        [
                            dataset,
                            optimizer,
                            row["trial"],
                            row["perf"],
                            row["best_so_far"],
                            row["wall_elapsed_sec"],
                        ]
                    )
    return curve_path


def _savefig(fig, base_dir, out_stem):
    png = base_dir / f"{out_stem}.png"
    pdf = base_dir / f"{out_stem}.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    return [png, pdf]


def _ranks_desc(scores):
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
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


def plot_dataset_convergence(base_dir, dataset, curves):
    rows_present = [opt for opt in OPTIMIZERS if curves.get((dataset, opt))]
    if not rows_present:
        return []

    fig, ax = plt.subplots(figsize=(10, 6))
    seen_y = []
    for offset, optimizer in enumerate(OPTIMIZERS):
        rows = curves.get((dataset, optimizer))
        if not rows:
            continue
        x = [r["trial"] for r in rows]
        y = [r["best_so_far"] for r in rows]
        seen_y.extend(y)
        ax.plot(
            x,
            y,
            label=LABELS[optimizer],
            color=COLORS[optimizer],
            linewidth=1.8,
            marker=MARKERS[optimizer],
            markersize=4,
            markevery=(offset, 8),
            alpha=0.9,
        )
    if seen_y:
        lo, hi = min(seen_y), max(seen_y)
        pad = max((hi - lo) * 0.08, 1e-3)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(f"{dataset}: best-so-far validation bal_acc", fontweight="bold")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Best validation score so far")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    out_dir = base_dir / "plots" / "per_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = _savefig(fig, out_dir, dataset)
    plt.close(fig)
    return out


def plot_mean_valid_bar(base_dir, datasets, summary):
    means = []
    for optimizer in OPTIMIZERS:
        vals = [
            summary[(dataset, optimizer)]["valid_best"]
            for dataset in datasets
            if (dataset, optimizer) in summary
        ]
        means.append(sum(vals) / len(vals) if vals else float("nan"))

    fig, ax = plt.subplots(figsize=(12, 6))
    xpos = list(range(len(OPTIMIZERS)))
    bars = ax.bar(
        xpos,
        means,
        color=[COLORS[o] for o in OPTIMIZERS],
        alpha=0.9,
    )
    for bar, value in zip(bars, means):
        if value == value:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.003,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(xpos)
    ax.set_xticklabels([LABELS[o] for o in OPTIMIZERS], rotation=20, ha="right")
    ax.set_ylabel("Mean valid_best across datasets")
    ax.set_title("Mean validation best by optimizer", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = _savefig(fig, base_dir / "plots", "mean_valid_best_bar")
    plt.close(fig)
    return out


def plot_average_rank(base_dir, datasets, curves):
    lengths = [len(rows) for rows in curves.values()]
    if not lengths:
        return []
    n_trials = min(lengths)
    mean_rank = {opt: [] for opt in OPTIMIZERS}
    trials = list(range(1, n_trials + 1))

    for t in range(n_trials):
        rank_sums = {opt: 0.0 for opt in OPTIMIZERS}
        n_sets = 0
        for dataset in datasets:
            scores = {}
            for opt in OPTIMIZERS:
                rows = curves.get((dataset, opt))
                if rows and t < len(rows):
                    scores[opt] = rows[t]["best_so_far"]
            if len(scores) != len(OPTIMIZERS):
                continue
            n_sets += 1
            for opt, rank in _ranks_desc(scores).items():
                rank_sums[opt] += rank
        for opt in OPTIMIZERS:
            mean_rank[opt].append(rank_sums[opt] / n_sets if n_sets else float("nan"))

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for optimizer in OPTIMIZERS:
        ax.plot(
            trials,
            mean_rank[optimizer],
            label=LABELS[optimizer],
            color=COLORS[optimizer],
            linewidth=2,
            marker=MARKERS[optimizer],
            markersize=4,
            markevery=6,
        )
    ax.set_xlabel("Trial")
    ax.set_ylabel("Average rank (1 = best)")
    ax.set_title("Average rank across datasets vs trial", fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = _savefig(fig, base_dir / "plots", "average_rank_summary")
    plt.close(fig)
    return out


def write_handoff(base_dir, table_path, run_config):
    handoff_dir = base_dir / "handoff"
    plots_src = base_dir / "plots"
    handoff_plots = handoff_dir / "plots"
    handoff_dir.mkdir(parents=True, exist_ok=True)
    handoff_plots.mkdir(parents=True, exist_ok=True)

    dest_table = handoff_dir / "valid_best_table.csv"
    dest_table.write_bytes(table_path.read_bytes())

    if plots_src.is_dir():
        for path in plots_src.rglob("*"):
            if path.is_file():
                rel = path.relative_to(plots_src)
                target = handoff_plots / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())

    with (handoff_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    readme = handoff_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Phase 0 Classification HPO Benchmark Handoff",
                "",
                "Primary deliverable: `valid_best_table.csv` (dataset × optimizer, validation best only).",
                "",
                "Optimizers:",
                "- smac, tpe, cmaes",
                "- ptllm, ptllm_tpe (gpt-4o-mini)",
                "- ptllm_qwen, ptllm_tpe_qwen (Qwen SFT6 via Ollama)",
                "",
                "Fixed settings: algorithm=random_forest, metric=bal_acc, holdout validation.",
                "",
                f"Results root: `{base_dir}`",
            ]
        ),
        encoding="utf-8",
    )
    return handoff_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        default="/root/blockdata/llmoptimizer_benchmark/benchmark_data/real_hpo_cls",
    )
    parser.add_argument("--handoff", action="store_true")
    parser.add_argument(
        "--v2-layout",
        action="store_true",
        help="Read cls/ and rgs/ subdirs with dynamic optimizer variant folder names.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if args.v2_layout:
        dataset_keys, optimizers, curves, summary = collect_results_v2(base_dir)
        if not summary:
            raise SystemExit(f"No v2 benchmark results found under {base_dir}")
        table_path = write_valid_best_table_v2(base_dir, dataset_keys, optimizers, summary)
        print(f"[Summary] v2 datasets={len(dataset_keys)} optimizers={len(optimizers)}")
        print(f"[Written] {table_path}")
        if args.handoff:
            handoff_dir = base_dir / "handoff"
            handoff_dir.mkdir(parents=True, exist_ok=True)
            target = handoff_dir / "valid_best_table.csv"
            target.write_bytes(table_path.read_bytes())
            print(f"[Handoff] {handoff_dir}")
        return

    datasets, curves, summary = collect_results(base_dir)
    if not summary:
        raise SystemExit(f"No benchmark results found under {base_dir}")

    table_path = write_valid_best_table(base_dir, datasets, summary)
    curve_path = write_long_csv(base_dir, datasets, curves)

    written = [table_path, curve_path]
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        written.extend(plot_dataset_convergence(base_dir, dataset, curves))
    written.extend(plot_mean_valid_bar(base_dir, datasets, summary))
    written.extend(plot_average_rank(base_dir, datasets, curves))

    run_config = {
        "task_type": "cls",
        "metric": "bal_acc",
        "algorithm": "random_forest",
        "n_datasets": len(datasets),
        "optimizers": OPTIMIZERS,
        "base_dir": str(base_dir),
    }
    if args.handoff:
        handoff_dir = write_handoff(base_dir, table_path, run_config)
        print(f"[Handoff] {handoff_dir}")
        written.append(handoff_dir)

    print(f"[Summary] datasets={len(datasets)} optimizers={len(OPTIMIZERS)}")
    print("\n[Written]")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
