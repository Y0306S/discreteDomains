#!/usr/bin/env python3
"""LLM vs TPE trial placement for s1a run1 (seed 1) vs r2 (seed 2), H0 mix50."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TASKS = json.loads(
    (REPO / "benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40.json").read_text(encoding="utf-8")
)
ROOTS = {
    "run1_seed1": REPO / "benchmark_data/real_hpo_v2_mltest20_algo40_p5_s1a_ep7p5",
    "run2_seed2": REPO / "benchmark_data/real_hpo_v2_mltest20_algo40_p5_s1a_ep7p5_r2",
    "run3_seed3": REPO / "benchmark_data/real_hpo_v2_mltest20_algo40_p5_s1a_ep7p5_r3",
}
VARIANT = "ptllm_tpe_qwen__P5__H0"
N_TRIALS = 80
N_DECILES = 8


def find_traj(root: Path, tt: str, ds: str) -> Path | None:
    d = root / tt / ds / VARIANT
    if not d.is_dir():
        return None
    best = None
    for run_dir in sorted(p for p in d.glob("CASH-*") if p.is_dir()):
        trajs = sorted(run_dir.glob("*_traj.csv"))
        if not trajs:
            continue
        traj = trajs[-1]
        n = sum(1 for _ in csv.DictReader(traj.open(newline="", encoding="utf-8")))
        if n >= N_TRIALS:
            if best is None or str(traj) > str(best):
                best = traj
    return best


def stages_from_traj(traj: Path) -> list[str]:
    rows = list(csv.DictReader(traj.open(newline="", encoding="utf-8")))
    return [r["stage"] for r in rows[:N_TRIALS]]


def bucket_llm(stage: str) -> bool:
    return stage in ("llm", "llm_warm_start")


def decile_counts(stages: list[str]) -> list[dict]:
    size = len(stages)
    chunk = size // N_DECILES
    out = []
    for d in range(N_DECILES):
        lo = d * chunk
        hi = (d + 1) * chunk if d < N_DECILES - 1 else size
        seg = stages[lo:hi]
        llm_n = sum(1 for s in seg if bucket_llm(s))
        tpe_n = sum(1 for s in seg if s == "tpe")
        out.append(
            {
                "decile": d + 1,
                "trial_range": f"{lo}-{hi - 1}",
                "llm": llm_n,
                "tpe": tpe_n,
                "llm_frac": llm_n / len(seg) if seg else 0.0,
            }
        )
    return out


def first_last_window(stages: list[str], window: int = 20) -> dict:
    start = stages[:window]
    end = stages[-window:]
    return {
        f"first_{window}_llm_frac": sum(1 for s in start if bucket_llm(s)) / window,
        f"last_{window}_llm_frac": sum(1 for s in end if bucket_llm(s)) / window,
    }


def main() -> int:
    out_dir = REPO / "benchmark_data/real_hpo_v2_mltest20_algo40/analysis/s1a_hybrid_llm_tpe_spread"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_task = []
    agg_decile: dict[str, dict[int, list[float]]] = {
        run: {d: [] for d in range(1, N_DECILES + 1)} for run in ROOTS
    }
    totals = {run: Counter() for run in ROOTS}

    for t in TASKS:
        tt, ds = t["task_type"], t["dataset"]
        key = f"{tt}/{ds}/{t['algorithm']}"
        row: dict = {"task": key}
        for run, root in ROOTS.items():
            traj = find_traj(root, tt, ds)
            if traj is None:
                row[f"{run}_missing"] = True
                continue
            stages = stages_from_traj(traj)
            c = Counter(stages)
            totals[run]["llm_warm_start"] += c["llm_warm_start"]
            totals[run]["llm"] += c["llm"]
            totals[run]["tpe"] += c["tpe"]
            totals[run]["trials"] += len(stages)

            llm_all = sum(1 for s in stages if bucket_llm(s))
            row[f"{run}_llm_frac"] = llm_all / len(stages)
            row[f"{run}_tpe_frac"] = c["tpe"] / len(stages)
            row[f"{run}_warm_start"] = c["llm_warm_start"]
            fl = first_last_window(stages)
            row.update({f"{run}_{k}": v for k, v in fl.items()})
            decs = decile_counts(stages)
            for d in decs:
                agg_decile[run][d["decile"]].append(d["llm_frac"])
                row[f"{run}_d{d['decile']}_llm_frac"] = d["llm_frac"]
        per_task.append(row)

    import csv as csvmod

    csv_path = out_dir / "per_task_llm_tpe_spread.csv"
    if per_task:
        fields = sorted({k for r in per_task for k in r})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csvmod.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(per_task)

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    summary = {
        "h0_definition": "mix50: 4 llm_warm_start, then each trial 50% coin -> llm else tpe (seed drives coin + TPE)",
        "totals_across_40_tasks": {
            run: {
                "llm_warm_start_trials": totals[run]["llm_warm_start"],
                "llm_after_warmstart": totals[run]["llm"],
                "tpe_trials": totals[run]["tpe"],
                "llm_share_all_trials": (
                    (totals[run]["llm_warm_start"] + totals[run]["llm"]) / totals[run]["trials"]
                    if totals[run]["trials"]
                    else None
                ),
            }
            for run in ROOTS
        },
        "mean_llm_fraction_by_decile": {
            run: {f"D{d}": round(mean(agg_decile[run][d]), 3) for d in range(1, N_DECILES + 1)}
            for run in ROOTS
        },
        "mean_first20_last20_llm_frac": {},
        "interpretation": [],
    }
    for run in ROOTS:
        f20 = [r[f"{run}_first_20_llm_frac"] for r in per_task if f"{run}_first_20_llm_frac" in r]
        l20 = [r[f"{run}_last_20_llm_frac"] for r in per_task if f"{run}_last_20_llm_frac" in r]
        summary["mean_first20_last20_llm_frac"][run] = {
            "first_20": round(mean(f20), 3),
            "last_20": round(mean(l20), 3),
        }

    d1_r1 = summary["mean_llm_fraction_by_decile"]["run1_seed1"]["D1"]
    d8_r1 = summary["mean_llm_fraction_by_decile"]["run1_seed1"]["D8"]
    summary["interpretation"].append(
        f"Run1: LLM share is ~{d1_r1:.0%} in first decile (trials 0–9, mostly warm-start) "
        f"vs ~{d8_r1:.0%} in last decile — after warm-start, mix is ~50/50 spread through the run, not all LLM at end."
    )
    f1 = summary["mean_first20_last20_llm_frac"]["run1_seed1"]
    summary["interpretation"].append(
        f"Run1 avg: first 20 trials ~{f1['first_20']:.0%} LLM (includes 4 warm-start + early coin flips), "
        f"last 20 trials ~{f1['last_20']:.0%} LLM — similar, as H0 expects."
    )
    if summary["mean_first20_last20_llm_frac"]["run2_seed2"]:
        f2 = summary["mean_first20_last20_llm_frac"]["run2_seed2"]
        summary["interpretation"].append(
            f"Seed 2 changes which trials are LLM vs TPE each run, but mean first/last 20 LLM fractions "
            f"stay similar (run2: first {f2['first_20']:.0%}, last {f2['last_20']:.0%})."
        )

    (out_dir / "report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md = out_dir / "README.md"
    md.write_text(
        "# H0 LLM vs TPE spread (s1a seed 1 vs 2)\n\n"
        "## H0\n"
        "4 **LLM warm-start** trials, then **50% random LLM / 50% TPE** per trial (same seed drives those coins).\n\n"
        "## Aggregate (40 tasks × 80 trials)\n"
        + "\n".join(
            f"- **{run}**: warm-start={v['llm_warm_start_trials']}, "
            f"llm={v['llm_after_warmstart']}, tpe={v['tpe_trials']}, "
            f"overall LLM share={v['llm_share_all_trials']:.1%}"
            for run, v in summary["totals_across_40_tasks"].items()
        )
        + "\n\n## Mean LLM fraction by decile (1=early, 8=late)\n\n"
        f"| decile | run1 | run2 |\n|--------|------|------|\n"
        + "\n".join(
            f"| D{d} | {summary['mean_llm_fraction_by_decile']['run1_seed1'][f'D{d}']:.2f} "
            f"| {summary['mean_llm_fraction_by_decile']['run2_seed2'][f'D{d}']:.2f} |"
            for d in range(1, N_DECILES + 1)
        )
        + "\n\n"
        + "\n".join(f"- {x}" for x in summary["interpretation"])
        + f"\n\nDetail: `{csv_path.name}`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
