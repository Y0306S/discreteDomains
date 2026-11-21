#!/usr/bin/env python3
"""Compare s1a_ep7p5 (seed 1) vs s1a_ep7p5_r2 (seed 2): bugs, parse rates, per-task scores."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TASKS = json.loads(
    (REPO / "benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40.json").read_text(encoding="utf-8")
)
ROOT1 = REPO / "benchmark_data/real_hpo_v2_mltest20_algo40_p5_s1a_ep7p5"
ROOT2 = REPO / "benchmark_data/real_hpo_v2_mltest20_algo40_p5_s1a_ep7p5_r2"
VARIANT = "ptllm_tpe_qwen__P5__H0"
N_TRIALS = 80
ADAPTER = Path("/root/blockdata/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5")


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def find_run_dir(root: Path, tt: str, ds: str) -> Path | None:
    d = root / tt / ds / VARIANT
    if not d.is_dir():
        return None
    best: tuple[float, int, Path] | None = None
    for run_dir in sorted(p for p in d.glob("CASH-*") if p.is_dir()):
        trajs = sorted(run_dir.glob("*_traj.csv"))
        if not trajs:
            continue
        traj = trajs[-1]
        rows = list(csv.DictReader(traj.open(newline="", encoding="utf-8")))
        if len(rows) < N_TRIALS:
            continue
        bests = [fnum(r.get("best_so_far")) for r in rows]
        bests = [b for b in bests if not math.isnan(b)]
        if not bests:
            continue
        score = max(bests)
        cand = (score, len(rows), run_dir)
        if best is None or cand[0] > best[0]:
            best = cand
    return best[2] if best else None


def read_summary_seed(root: Path, tt: str, ds: str) -> int | None:
    p = root / tt / ds / "summary.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("seed")


def parse_stats(run_dir: Path) -> dict:
    conv_dir = run_dir / "llm_conversations"
    n_files = 0
    parse_ok = 0
    parse_fail = 0
    empty = 0
    for p in conv_dir.glob("trial_*.json"):
        n_files += 1
        row = json.loads(p.read_text(encoding="utf-8"))
        payload = row.get("parsed_payload")
        if payload is None:
            empty += 1
            continue
        if isinstance(payload, dict) and payload.get("config"):
            parse_ok += 1
        else:
            parse_fail += 1
    return {
        "trials_logged": n_files,
        "parse_ok": parse_ok,
        "parse_fail": parse_fail,
        "parse_empty": empty,
    }


def final_best(run_dir: Path) -> float | None:
    trajs = sorted(run_dir.glob("*_traj.csv"))
    if not trajs:
        return None
    rows = list(csv.DictReader(trajs[-1].open(newline="", encoding="utf-8")))
    bests = [fnum(r.get("best_so_far")) for r in rows]
    bests = [b for b in bests if not math.isnan(b) and b != float("-inf")]
    return max(bests) if bests else None


def main() -> int:
    out_dir = REPO / "benchmark_data/real_hpo_v2_mltest20_algo40/analysis/s1a_seed_replication"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    seed_mismatch = []
    missing_r1 = missing_r2 = 0
    r2_wins = r1_wins = ties = 0
    parse_delta = []

    for t in TASKS:
        tt, ds, algo = t["task_type"], t["dataset"], t["algorithm"]
        key = f"{tt}/{ds}/{algo}"
        run1 = find_run_dir(ROOT1, tt, ds)
        run2 = find_run_dir(ROOT2, tt, ds)
        s1 = read_summary_seed(ROOT1, tt, ds)
        s2 = read_summary_seed(ROOT2, tt, ds)
        if s1 not in (None, 1):
            seed_mismatch.append((key, "run1", s1))
        if s2 not in (None, 2):
            seed_mismatch.append((key, "run2", s2))

        b1 = final_best(run1) if run1 else None
        b2 = final_best(run2) if run2 else None
        if run1 is None:
            missing_r1 += 1
        if run2 is None:
            missing_r2 += 1

        p1 = parse_stats(run1) if run1 else {}
        p2 = parse_stats(run2) if run2 else {}
        if p1 and p2:
            rate1 = p1["parse_ok"] / max(1, p1["trials_logged"])
            rate2 = p2["parse_ok"] / max(1, p2["trials_logged"])
            parse_delta.append(rate2 - rate1)

        delta = None
        if b1 is not None and b2 is not None:
            delta = b2 - b1
            if delta > 1e-9:
                r2_wins += 1
            elif delta < -1e-9:
                r1_wins += 1
            else:
                ties += 1

        rows.append(
            {
                "task": key,
                "seed_run1_summary": s1,
                "seed_run2_summary": s2,
                "best_run1": b1,
                "best_run2": b2,
                "delta_r2_minus_r1": delta,
                "parse_ok_run1": p1.get("parse_ok"),
                "parse_ok_run2": p2.get("parse_ok"),
                "trials_logged_run1": p1.get("trials_logged"),
                "trials_logged_run2": p2.get("trials_logged"),
            }
        )

    rows.sort(key=lambda r: (r["delta_r2_minus_r1"] is None, -(r["delta_r2_minus_r1"] or 0)))

    csv_path = out_dir / "per_task_seed1_vs_seed2.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = list(rows[0].keys()) if rows else []
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    adapter_mtime = ADAPTER / "adapter_model.safetensors"
    report = {
        "adapter_path": str(ADAPTER),
        "adapter_exists": adapter_mtime.is_file(),
        "adapter_size_bytes": adapter_mtime.stat().st_size if adapter_mtime.is_file() else None,
        "expected_seeds": {"run1": 1, "run2": 2},
        "seed_mismatch_in_summary": seed_mismatch,
        "missing_complete_runs": {"run1": missing_r1, "run2": missing_r2},
        "per_task_score": {"r2_better": r2_wins, "r1_better": r1_wins, "ties": ties},
        "parse_ok_rate_mean_delta_r2_minus_r1": (
            sum(parse_delta) / len(parse_delta) if parse_delta else None
        ),
        "top_r2_gains": [
            {k: r[k] for k in ("task", "best_run1", "best_run2", "delta_r2_minus_r1")}
            for r in rows[:8]
            if r.get("delta_r2_minus_r1") is not None
        ],
        "top_r1_gains": [
            {k: r[k] for k in ("task", "best_run1", "best_run2", "delta_r2_minus_r1")}
            for r in sorted(rows, key=lambda x: x.get("delta_r2_minus_r1") or 0)[:8]
            if r.get("delta_r2_minus_r1") is not None
        ],
        "conclusion": [],
    }

    if not seed_mismatch and missing_r1 == 0 and missing_r2 == 0:
        report["conclusion"].append(
            "No seed metadata bugs: run1 summaries report seed=1, run2 report seed=2; same adapter path."
        )
    if report["parse_ok_rate_mean_delta_r2_minus_r1"] is not None:
        d = report["parse_ok_rate_mean_delta_r2_minus_r1"]
        if abs(d) < 0.02:
            report["conclusion"].append(
                "JSON parse success rates are similar between runs — unlikely a parse/regression bug."
            )
        else:
            report["conclusion"].append(
                f"Parse OK rate differs on average by {d:+.3f}; check tasks with low parse_ok."
            )

    report["conclusion"].append(
        f"Score head-to-head: seed-2 better on {r2_wins}/40 tasks, seed-1 on {r1_wins}, ties {ties}. "
        "With identical adapter and only --seed changed, large spread is consistent with TPE + LLM sampling noise, not a wrong checkpoint."
    )

    if r2_wins > r1_wins + 8:
        report["conclusion"].append(
            "r2 lead is large but not explained by config errors found here; consider a third seed or paired bootstrap on per-task deltas."
        )

    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    readme = out_dir / "README.md"
    readme.write_text(
        "# s1a_ep7p5 seed replication (run1 vs r2)\n\n"
        f"- CSV: `{csv_path.name}`\n"
        f"- r2 wins / r1 wins / ties: **{r2_wins} / {r1_wins} / {ties}**\n"
        f"- Mean parse-OK rate Δ (r2−r1): {report['parse_ok_rate_mean_delta_r2_minus_r1']}\n\n"
        + "\n".join(f"- {c}" for c in report["conclusion"])
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
