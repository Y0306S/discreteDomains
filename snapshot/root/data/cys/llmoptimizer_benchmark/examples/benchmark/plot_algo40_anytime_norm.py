#!/usr/bin/env python3
"""Anytime normalized performance curves for selected algo40 methods.

Style: step plot of mean normalized best-so-far vs trial index (paper-like).

Per task:
  - load each method's best_so_far trajectory (80 trials)
  - pool ALL trial ``perf`` values from the plotted methods
  - drop worst 20%, scale with min(kept)→0, max(kept)→1
  - map each method's best_so_far[t] through that scale
Then average across tasks (equal weight), and also 50/50 cls/rgs mean.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
ALGO40 = REPO / "benchmark_data" / "real_hpo_v2_mltest20_algo40"
N_TRIALS = 80

# (label, root, disk_variant, style)
SERIES = [
    ("SMAC", ALGO40 / "classical", "smac", "baseline"),
    ("CMA-ES", ALGO40 / "classical", "cmaes", "baseline"),
    ("TPE", ALGO40 / "classical", "tpe", "baseline"),
    (
        "s1a bare (r2)",
        ALGO40.parent / "real_hpo_v2_mltest20_algo40_p5_s1a_ep7p5_r2",
        "ptllm_tpe_qwen__P5__H0",
        "main",
    ),
    (
        "s1a bare (r3)",
        ALGO40.parent / "real_hpo_v2_mltest20_algo40_p5_s1a_ep7p5_r3",
        "ptllm_tpe_qwen__P5__H0",
        "main",
    ),
    (
        "s1b bare",
        ALGO40.parent / "real_hpo_v2_mltest20_algo40_p5_s1b_ep7p5",
        "ptllm_tpe_qwen__P5__H0",
        "main",
    ),
    (
        "s1b DPO-B",
        ALGO40.parent / "real_hpo_v2_mltest20_algo40_p5_dpo_B",
        "ptllm_tpe_qwen__P5__H0",
        "dpo",
    ),
    (
        "s1b DPO-G",
        ALGO40.parent / "real_hpo_v2_mltest20_algo40_p5_dpo_G",
        "ptllm_tpe_qwen__P5__H0",
        "dpo",
    ),
]

# Companion plot: leaderboard ranks #2–#5 + base s1b + classical baselines
# (from mean_norm_table_pooled_drop20pct.csv sorted by mean_balanced_norm)
SERIES_BASELINES = [
    ("SMAC", ALGO40 / "classical", "smac", "baseline"),
    ("CMA-ES", ALGO40 / "classical", "cmaes", "baseline"),
    ("TPE", ALGO40 / "classical", "tpe", "baseline"),
    (
        "s1b bare",
        ALGO40.parent / "real_hpo_v2_mltest20_algo40_p5_s1b_ep7p5",
        "ptllm_tpe_qwen__P5__H0",
        "main",
    ),
    # rank 2
    (
        "s1a bare (r3) #2",
        ALGO40.parent / "real_hpo_v2_mltest20_algo40_p5_s1a_ep7p5_r3",
        "ptllm_tpe_qwen__P5__H0",
        "rank",
    ),
    # rank 3
    (
        "s1b DPO-G #3",
        ALGO40.parent / "real_hpo_v2_mltest20_algo40_p5_dpo_G",
        "ptllm_tpe_qwen__P5__H0",
        "rank",
    ),
    # rank 4
    (
        "s1b rationale-G #4",
        ALGO40.parent / "real_hpo_v2_mltest20_algo40_p5_dpo_rationale_G",
        "ptllm_tpe_qwen__P5__H0",
        "rank",
    ),
    # rank 5
    (
        "s1a D1 #5",
        ALGO40.parent / "real_hpo_v2_mltest20_algo40_p5_s1a_s2_D1",
        "ptllm_tpe_qwen__P5__H0",
        "rank",
    ),
]

# Colors / markers inspired by the reference figure
STYLE = {
    "SMAC": dict(color="#7B2D8E", linestyle="--", marker="^", linewidth=2.0),
    "CMA-ES": dict(color="#2E86AB", linestyle="--", marker="D", linewidth=2.0),
    "TPE": dict(color="#E8871E", linestyle="--", marker="s", linewidth=2.0),
    "s1a bare (r2)": dict(color="#C41E3A", linestyle="-", marker="o", linewidth=2.4),
    "s1a bare (r3)": dict(color="#E76F51", linestyle="-", marker="^", linewidth=2.2),
    "s1b bare": dict(color="#1B9E77", linestyle="-", marker="o", linewidth=2.2),
    "s1b DPO-B": dict(color="#D95F02", linestyle="-.", marker="s", linewidth=2.0),
    "s1b DPO-G": dict(color="#7570B3", linestyle="-.", marker="D", linewidth=2.0),
    "s1a bare (r3) #2": dict(color="#E76F51", linestyle="-", marker="^", linewidth=2.3),
    "s1b DPO-G #3": dict(color="#7570B3", linestyle="-.", marker="D", linewidth=2.1),
    "s1b rationale-G #4": dict(color="#66A61E", linestyle="-.", marker="v", linewidth=2.1),
    "s1a D1 #5": dict(color="#E7298A", linestyle="-", marker="P", linewidth=2.1),
}


def fnum(v, default=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_tasks(path: Path) -> list[dict]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def find_traj(
    root: Path,
    tt: str,
    ds: str,
    variant: str,
    n_trials: int,
    algorithm: str | None = None,
) -> Path | None:
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
        if algorithm:
            algos = [str(r.get("algorithm") or "").strip() for r in rows if str(r.get("algorithm") or "").strip()]
            if not algos:
                continue
            from collections import Counter

            mode = Counter(algos).most_common(1)[0][0]
            if mode != algorithm:
                continue
        bests = [fnum(r.get("best_so_far")) for r in rows]
        bests = [b for b in bests if not math.isnan(b)]
        if not bests:
            continue
        complete.append((max(bests), len(rows), traj))
    if not complete:
        return None
    complete.sort(key=lambda x: (x[0], x[1], str(x[2])))
    return complete[-1][2]


def read_best_curve(traj: Path, n_trials: int) -> list[float] | None:
    rows = list(csv.DictReader(traj.open(newline="", encoding="utf-8")))[:n_trials]
    if len(rows) < n_trials:
        return None
    out: list[float] = []
    for r in rows:
        b = fnum(r.get("best_so_far"))
        if math.isnan(b) or b == float("-inf"):
            return None
        out.append(b)
    # enforce non-decreasing (higher better)
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1])
    return out


def read_perfs(traj: Path, n_trials: int) -> list[float]:
    rows = list(csv.DictReader(traj.open(newline="", encoding="utf-8")))[:n_trials]
    out: list[float] = []
    for r in rows:
        p = fnum(r.get("perf"))
        if not math.isnan(p) and p != float("-inf"):
            out.append(p)
    return out


def pooled_scale(perfs: list[float], drop_worst_frac: float = 0.20) -> tuple[float, float] | None:
    if len(perfs) < 2:
        return None
    xs = sorted(perfs)
    n_drop = int(math.floor(drop_worst_frac * len(xs)))
    kept = xs[n_drop:] if n_drop > 0 else xs
    if not kept or kept[-1] <= kept[0]:
        return None
    return kept[0], kept[-1]


def normalize_curve(curve: list[float], lo: float, hi: float) -> list[float]:
    span = hi - lo
    out = []
    for v in curve:
        if v <= lo:
            out.append(0.0)
        elif v >= hi:
            out.append(1.0)
        else:
            out.append((v - lo) / span)
    return out


def build_curves(
    tasks: list[dict],
    n_trials: int,
    series: list[tuple] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    series = series if series is not None else SERIES
    labels = [s[0] for s in series]
    # per task: method -> curve
    task_curves: list[dict[str, list[float]]] = []
    coverage = {lab: 0 for lab in labels}

    for t in tasks:
        tt, ds = t["task_type"], t["dataset"]
        curves: dict[str, list[float]] = {}
        pool: list[float] = []
        for label, root, variant, _kind in series:
            if not root.is_dir():
                continue
            traj = find_traj(root, tt, ds, variant, n_trials, algorithm=t.get("algorithm"))
            if traj is None:
                continue
            curve = read_best_curve(traj, n_trials)
            if curve is None:
                continue
            curves[label] = curve
            pool.extend(read_perfs(traj, n_trials))
        if len(curves) < 2:
            continue
        scale = pooled_scale(pool)
        if scale is None:
            # fallback: min/max of finals among present methods
            finals = [c[-1] for c in curves.values()]
            lo, hi = min(finals), max(finals)
            if hi <= lo:
                continue
        else:
            lo, hi = scale
        normed = {lab: normalize_curve(c, lo, hi) for lab, c in curves.items()}
        task_curves.append(normed)
        for lab in normed:
            coverage[lab] += 1

    # mean over tasks (nan if missing on a task — skip that task for that method)
    means: dict[str, np.ndarray] = {}
    for lab in labels:
        mats = [tc[lab] for tc in task_curves if lab in tc]
        if not mats:
            continue
        means[lab] = np.mean(np.asarray(mats, dtype=float), axis=0)
    return means, coverage


def build_balanced_curves(
    tasks: list[dict],
    n_trials: int,
    series: list[tuple] | None = None,
) -> dict[str, np.ndarray]:
    """50/50 mean of cls-mean and rgs-mean anytime curves."""
    series = series if series is not None else SERIES
    cls_tasks = [t for t in tasks if t["task_type"] == "cls"]
    rgs_tasks = [t for t in tasks if t["task_type"] == "rgs"]
    cls_m, _ = build_curves(cls_tasks, n_trials, series)
    rgs_m, _ = build_curves(rgs_tasks, n_trials, series)
    out: dict[str, np.ndarray] = {}
    for lab in [s[0] for s in series]:
        if lab in cls_m and lab in rgs_m:
            out[lab] = 0.5 * (cls_m[lab] + rgs_m[lab])
    return out


def plot_anytime(
    means: dict[str, np.ndarray],
    out_path: Path,
    title: str,
    series: list[tuple] | None = None,
    coverage: dict[str, int] | None = None,
    xmin: int = 1,
    xmax: int | None = None,
) -> None:
    series = series if series is not None else SERIES
    xmax = N_TRIALS if xmax is None else xmax
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    x = np.arange(1, N_TRIALS + 1)
    span = max(1, xmax - xmin + 1)
    for i, (label, *_rest) in enumerate(series):
        if label not in means:
            continue
        st = STYLE[label]
        y = means[label]
        markevery = (i * 3, max(4, span // 8))
        ax.step(
            x,
            y,
            where="post",
            label=label,
            color=st["color"],
            linestyle=st["linestyle"],
            linewidth=st["linewidth"],
            marker=st["marker"],
            markersize=6,
            markevery=markevery,
            alpha=0.95,
        )
    ax.set_xlabel("Number of Evaluated Configurations", fontweight="bold")
    ax.set_ylabel("Normalized Performance", fontweight="bold")
    ax.set_title(title)
    ax.set_xlim(xmin, xmax)
    # y-limits from the visible window only
    i0, i1 = xmin - 1, xmax
    ys_list = [v[i0:i1] for v in means.values() if len(v) >= i1]
    ys = np.concatenate(ys_list) if ys_list else np.array([0.0, 1.0])
    y0 = max(0.0, float(np.nanmin(ys)) - 0.05)
    y1 = min(1.02, float(np.nanmax(ys)) + 0.03)
    ax.set_ylim(y0, y1)
    ax.grid(True, linestyle="--", color="0.75", alpha=0.8)
    ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="0.6")
    if coverage:
        cov = ", ".join(f"{k}:{v}" for k, v in coverage.items() if k in means)
        ax.text(
            0.01,
            0.01,
            f"tasks/method: {cov}",
            transform=ax.transAxes,
            fontsize=7,
            color="0.4",
            va="bottom",
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ALGO40 / "analysis" / "avg_rank" / "anytime",
    )
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--xmin", type=int, default=2, help="x-axis min for baselines zoom plot")
    parser.add_argument("--xmax", type=int, default=80, help="x-axis max for baselines zoom plot")
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO / args.out_dir

    tasks = load_tasks(ALGO40 / "tasks_40.json")
    means, coverage = build_curves(tasks, args.n_trials, SERIES)
    bal = build_balanced_curves(tasks, args.n_trials, SERIES)

    # CSV of curves
    csv_path = out_dir / "anytime_mean_norm_curves.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        labs = [s[0] for s in SERIES if s[0] in means]
        w = csv.DictWriter(f, fieldnames=["trial"] + labs)
        w.writeheader()
        for t in range(args.n_trials):
            row = {"trial": t + 1}
            for lab in labs:
                row[lab] = f"{means[lab][t]:.6f}"
            w.writerow(row)

    plot_anytime(
        means,
        out_dir / "anytime_normalized_performance.png",
        "Algo40 anytime (mean over 40 tasks, pooled drop-worst-10% norm)",
        series=SERIES,
        coverage=coverage,
    )
    plot_anytime(
        bal,
        out_dir / "anytime_normalized_performance_balanced.png",
        "Algo40 anytime (balanced cls/rgs mean, pooled drop-worst-10% norm)",
        series=SERIES,
        coverage=coverage,
    )

    # Companion: leaderboard #2–#5 + s1b bare + classical, zoomed to [xmin, xmax]
    base_means, base_cov = build_curves(tasks, args.n_trials, SERIES_BASELINES)
    base_bal = build_balanced_curves(tasks, args.n_trials, SERIES_BASELINES)
    zoom_tag = f"t{args.xmin}_{args.xmax}"
    plot_anytime(
        base_means,
        out_dir / f"anytime_normalized_performance_baselines_{zoom_tag}.png",
        f"Algo40 anytime — ranks #2–#5 + s1b bare + classical (trials {args.xmin}–{args.xmax})",
        series=SERIES_BASELINES,
        coverage=base_cov,
        xmin=args.xmin,
        xmax=args.xmax,
    )
    plot_anytime(
        base_bal,
        out_dir / f"anytime_normalized_performance_baselines_{zoom_tag}_balanced.png",
        f"Algo40 anytime — ranks #2–#5 + s1b + classical, balanced (trials {args.xmin}–{args.xmax})",
        series=SERIES_BASELINES,
        coverage=base_cov,
        xmin=args.xmin,
        xmax=args.xmax,
    )
    base_csv = out_dir / f"anytime_mean_norm_curves_baselines_{zoom_tag}.csv"
    with base_csv.open("w", newline="", encoding="utf-8") as f:
        labs = [s[0] for s in SERIES_BASELINES if s[0] in base_means]
        w = csv.DictWriter(f, fieldnames=["trial"] + labs)
        w.writeheader()
        for t in range(args.n_trials):
            row = {"trial": t + 1}
            for lab in labs:
                row[lab] = f"{base_means[lab][t]:.6f}"
            w.writerow(row)

    print(f"Wrote {csv_path}")
    print(f"Wrote {out_dir / 'anytime_normalized_performance.png'}")
    print(f"Wrote {out_dir / 'anytime_normalized_performance_balanced.png'}")
    print(f"Wrote {out_dir / f'anytime_normalized_performance_baselines_{zoom_tag}.png'}")
    print(f"Wrote {out_dir / f'anytime_normalized_performance_baselines_{zoom_tag}_balanced.png'}")
    print("Coverage:", coverage)
    print("Baselines coverage:", base_cov)
    print("Final mean norms:")
    for lab, *_ in SERIES:
        if lab in means:
            bf = float(bal[lab][-1]) if lab in bal else float("nan")
            print(f"  {lab:16s} final={means[lab][-1]:.4f}  bal_final={bf:.4f}")
    print("Baselines (re-pooled among SMAC/CMA-ES/TPE/s1b) at trial", args.xmax, ":")
    for lab, *_ in SERIES_BASELINES:
        if lab in base_means:
            bf = float(base_bal[lab][args.xmax - 1]) if lab in base_bal else float("nan")
            print(f"  {lab:16s} @t{args.xmax}={base_means[lab][args.xmax - 1]:.4f}  bal={bf:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
