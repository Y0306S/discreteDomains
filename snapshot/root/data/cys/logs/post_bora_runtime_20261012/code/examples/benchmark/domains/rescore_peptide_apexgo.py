#!/usr/bin/env python3
"""Re-score peptide BO trajectories using the APEXGo paper evaluation protocol.

Paper-aligned evaluation (gram-negative constrained optimization):
  - Objective: ``gramnegonly_mean`` from APEXGo's ``ApexObjective``
  - Constraint: >=75% template similarity via Levenshtein edit distance
  - Selection: greedy diverse top-M set (default M=20, tau=1 edit distance)
  - Improvement: template vs rank-1 diverse peptide on APEXGo score and mean MIC

For each task, the task start sequence is treated as the template (APEXGo runs
one template per optimization job).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains.paths import BOLT_PEPTIDES, domain_bench, protein_bo80_bolt_root
from mindware.domains.protein.apexgo_metrics import (
    _load_apex_objective,
    log2_fold_change,
    passes_similarity_constraint,
    score_sequences,
    select_diverse_set,
    template_similarity,
)

BOLT_APEX = BOLT_PEPTIDES / "apex_oracle"


def _load_start_seqs() -> List[str]:
    path = BOLT_APEX / "init_data/seed_0_init.txt"
    return [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith(">")
    ]


def _parse_idx(task_id: str) -> int:
    return int(str(task_id).strip().split("_")[-1])


def _obs_seq(obs: Dict[str, Any]) -> str:
    extra = obs.get("extra_info") or {}
    info = extra.get("info") or {}
    seq = extra.get("artifact") or info.get("sequence") or ""
    return str(seq).strip()


def _unique_seqs(observations: Sequence[Dict[str, Any]], checkpoint: int) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for obs in observations[: int(checkpoint)]:
        seq = _obs_seq(obs)
        if not seq or seq in seen:
            continue
        seen.add(seq)
        out.append(seq)
    return out


def _eval_task(
    task_id: str,
    *,
    template: str,
    history_path: Path,
    checkpoint: int,
    objective,
    apex_wrapper,
    batch_size: int,
    min_similarity: float,
    diverse_m: int,
    diverse_tau: int,
) -> Dict[str, Any]:
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    observations = list(payload.get("observations") or [])
    n_obs = len(observations)
    if n_obs < 1:
        return {"task_id": task_id, "status": "empty", "n_obs": 0}

    candidates = _unique_seqs(observations, checkpoint)
    if not candidates:
        return {"task_id": task_id, "status": "no_seq", "n_obs": n_obs}

    feasible = [s for s in candidates if passes_similarity_constraint(s, template, min_similarity=min_similarity)]
    to_score = [template] + [s for s in candidates if s != template]
    scored_all = score_sequences(
        to_score,
        objective=objective,
        apex_wrapper=apex_wrapper,
        batch_size=batch_size,
    )
    template_metrics = scored_all[template]

    if not feasible:
        return {
            "task_id": task_id,
            "status": "no_feasible",
            "n_obs": n_obs,
            "checkpoint": int(checkpoint),
            "template": template,
            "n_candidates": len(candidates),
            "n_feasible": 0,
            "template_metrics": template_metrics,
        }

    scored_feasible = {s: scored_all[s]["apexgo_score"] for s in feasible}
    diverse = select_diverse_set(list(scored_feasible.items()), m=diverse_m, tau=diverse_tau)
    best_seq, best_apexgo = diverse[0]
    best_metrics = scored_all[best_seq]

    improved_apexgo = bool(best_apexgo > template_metrics["apexgo_score"] + 1e-9)
    improved_gramneg = bool(best_metrics["gramneg_mean_mic"] < template_metrics["gramneg_mean_mic"] - 1e-9)
    improved_all = bool(best_metrics["all_mean_mic"] < template_metrics["all_mean_mic"] - 1e-9)

    return {
        "task_id": task_id,
        "status": "ok",
        "checkpoint": int(checkpoint),
        "n_obs": n_obs,
        "template": template,
        "n_candidates": len(candidates),
        "n_feasible": len(feasible),
        "template_metrics": template_metrics,
        "best_seq": best_seq,
        "best_metrics": best_metrics,
        "best_similarity": template_similarity(best_seq, template),
        "diverse_set": [{"seq": s, "apexgo_score": sc, **scored_all[s]} for s, sc in diverse],
        "delta_apexgo": best_apexgo - template_metrics["apexgo_score"],
        "delta_gramneg_mean_mic": best_metrics["gramneg_mean_mic"] - template_metrics["gramneg_mean_mic"],
        "delta_all_mean_mic": best_metrics["all_mean_mic"] - template_metrics["all_mean_mic"],
        "log2_fold_gramneg_mean_mic": log2_fold_change(
            best_metrics["gramneg_mean_mic"], template_metrics["gramneg_mean_mic"]
        ),
        "log2_fold_all_mean_mic": log2_fold_change(best_metrics["all_mean_mic"], template_metrics["all_mean_mic"]),
        "improved_apexgo": improved_apexgo,
        "improved_gramneg_mean_mic": improved_gramneg,
        "improved_all_mean_mic": improved_all,
        "eval_protocol": {
            "objective": "gramnegonly_mean",
            "min_similarity": float(min_similarity),
            "diverse_m": int(diverse_m),
            "diverse_tau": int(diverse_tau),
            "selection": "rank1_diverse_set",
        },
    }


def _task_ids(n_tasks: int, offset: int, shard_index: int, n_shards: int) -> List[str]:
    all_ids = [f"pep_{i:03d}" for i in range(int(n_tasks))]
    if int(n_shards) <= 1:
        return all_ids[int(offset) :]
    return [tid for i, tid in enumerate(all_ids) if i % int(n_shards) == int(shard_index)]


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    no_feas = [r for r in rows if r.get("status") == "no_feasible"]
    return {
        "n_tasks": len(rows),
        "n_ok": len(ok),
        "n_no_feasible": len(no_feas),
        "improved_apexgo": sum(1 for r in ok if r.get("improved_apexgo")),
        "improved_gramneg_mean_mic": sum(1 for r in ok if r.get("improved_gramneg_mean_mic")),
        "improved_all_mean_mic": sum(1 for r in ok if r.get("improved_all_mean_mic")),
        "improvement_rate_apexgo": (sum(1 for r in ok if r.get("improved_apexgo")) / len(ok)) if ok else 0.0,
        "improvement_rate_gramneg_mean_mic": (
            sum(1 for r in ok if r.get("improved_gramneg_mean_mic")) / len(ok)
        )
        if ok
        else 0.0,
        "improvement_rate_all_mean_mic": (
            sum(1 for r in ok if r.get("improved_all_mean_mic")) / len(ok)
        )
        if ok
        else 0.0,
        "median_delta_apexgo": float(np.median([r["delta_apexgo"] for r in ok])) if ok else 0.0,
        "median_log2_fold_gramneg_mean_mic": float(
            np.median([r["log2_fold_gramneg_mean_mic"] for r in ok])
        )
        if ok
        else 0.0,
        "median_log2_fold_all_mean_mic": float(np.median([r["log2_fold_all_mean_mic"] for r in ok])) if ok else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="APEXGo paper-faithful re-score for peptide BO trajectories")
    parser.add_argument("--checkpoint", type=int, default=100, help="oracle-call budget to evaluate")
    parser.add_argument("--optimizer", default="tpe")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-tasks", type=int, default=1000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-obs", type=int, default=100, help="skip tasks with fewer observations")
    parser.add_argument("--min-similarity", type=float, default=0.75)
    parser.add_argument("--diverse-m", type=int, default=20)
    parser.add_argument("--diverse-tau", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--summary-name", default=None)
    parser.add_argument(
        "--merge-dir",
        type=Path,
        default=None,
        help="If set, only merge summary_shard*.json under this directory and exit.",
    )
    args = parser.parse_args()

    if args.merge_dir is not None:
        merge_dir = Path(args.merge_dir)
        shard_summaries = sorted(merge_dir.glob("summary_shard*.json"))
        rows: List[Dict[str, Any]] = []
        for p in shard_summaries:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        merged = {
            "shards": rows,
            "aggregate": {
                "improved_apexgo": sum(r.get("improved_apexgo", 0) for r in rows),
                "improved_gramneg_mean_mic": sum(r.get("improved_gramneg_mean_mic", 0) for r in rows),
                "improved_all_mean_mic": sum(r.get("improved_all_mean_mic", 0) for r in rows),
                "n_ok": sum(r.get("n_ok", 0) for r in rows),
                "n_no_feasible": sum(r.get("n_no_feasible", 0) for r in rows),
                "n_tasks": sum(r.get("n_tasks", 0) for r in rows),
            },
        }
        n_ok = merged["aggregate"]["n_ok"]
        if n_ok:
            merged["aggregate"]["improvement_rate_apexgo"] = merged["aggregate"]["improved_apexgo"] / n_ok
            merged["aggregate"]["improvement_rate_gramneg_mean_mic"] = (
                merged["aggregate"]["improved_gramneg_mean_mic"] / n_ok
            )
            merged["aggregate"]["improvement_rate_all_mean_mic"] = (
                merged["aggregate"]["improved_all_mean_mic"] / n_ok
            )
        out = merge_dir / "summary_merged.json"
        out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        print(json.dumps(merged, indent=2))
        return 0

    apex_dir = BOLT_APEX
    if str(apex_dir) not in sys.path:
        sys.path.insert(0, str(apex_dir))
    from APEX_predict import apex_wrapper  # type: ignore

    objective = _load_apex_objective("gramnegonly_mean")
    starts = _load_start_seqs()
    bench_root = protein_bo80_bolt_root()
    out_root = args.out_dir or (domain_bench("protein") / "apexgo_eval" / f"ckpt{int(args.checkpoint)}")
    out_root.mkdir(parents=True, exist_ok=True)

    if args.task_ids:
        tasks = list(args.task_ids)
    else:
        tasks = _task_ids(args.n_tasks, args.offset, args.shard_index, args.n_shards)

    rows: List[Dict[str, Any]] = []
    n_ok = 0
    for task_id in tasks:
        idx = _parse_idx(task_id)
        if idx < 0 or idx >= len(starts):
            print(f"[skip] {task_id}: template idx out of range")
            continue
        hist = bench_root / task_id / str(args.optimizer) / f"seed{int(args.seed)}" / "history.json"
        if not hist.is_file():
            print(f"[skip] {task_id}: missing {hist}")
            continue
        n_obs = len(json.loads(hist.read_text(encoding="utf-8")).get("observations") or [])
        if n_obs < int(args.min_obs):
            print(f"[skip] {task_id}: only {n_obs} obs (<{args.min_obs})")
            continue

        row = _eval_task(
            task_id,
            template=starts[idx],
            history_path=hist,
            checkpoint=int(args.checkpoint),
            objective=objective,
            apex_wrapper=apex_wrapper,
            batch_size=int(args.batch_size),
            min_similarity=float(args.min_similarity),
            diverse_m=int(args.diverse_m),
            diverse_tau=int(args.diverse_tau),
        )
        out_path = out_root / f"{task_id}.json"
        out_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
        if row.get("status") == "ok":
            n_ok += 1
            print(
                f"[ok] {task_id} apexgo {row['template_metrics']['apexgo_score']:.3f}"
                f"->{row['best_metrics']['apexgo_score']:.3f} "
                f"({'+' if row['improved_apexgo'] else '='}) "
                f"feas={row['n_feasible']}/{row['n_candidates']}"
            )
        elif row.get("status") == "no_feasible":
            print(f"[no_feasible] {task_id} candidates={row['n_candidates']}")

    summary = _summarize(rows)
    summary["checkpoint"] = int(args.checkpoint)
    summary["eval_protocol"] = {
        "objective": "gramnegonly_mean",
        "min_similarity": float(args.min_similarity),
        "diverse_m": int(args.diverse_m),
        "diverse_tau": int(args.diverse_tau),
        "selection": "rank1_diverse_set",
    }
    summary_path = out_root / (args.summary_name or f"summary_shard{args.shard_index}.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
