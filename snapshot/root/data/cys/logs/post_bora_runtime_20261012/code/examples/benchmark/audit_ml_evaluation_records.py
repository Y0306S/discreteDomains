"""Verify and export configuration/result pairs without modifying running searches."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import pickle
import tempfile
import sys
import time
import os
import ml_run_archive as run_archive


DEFAULT = Path("/root/data/cys/llmoptimizer_benchmark/benchmark_data/ml_test_20_fixed_48h_20260925")
METHODS = ("ckpt32", "tpe", "lolbo", "random_search")


def config_from_row(row):
    value = row.get("config_json")
    config = json.loads(value) if value else None
    if not isinstance(config, dict) or not config or set(config) == {"repr"}:
        raise ValueError("Missing full configuration")
    return config


def from_topk(rows, items):
    """Keep saved pairs intact; equal scores cannot establish trial order."""
    scores = sorted(float(row["perf"]) for row in rows)
    saved_scores = sorted(float(item[1]) for item in items)
    if len(rows) != len(items) or not all(
            math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(scores, saved_scores)):
        raise ValueError("Saved configuration scores do not cover the trajectory")
    pairs = []
    for config, score, *_ in items:
        if not isinstance(config, dict) or not config or not math.isfinite(float(score)):
            raise ValueError("Invalid saved configuration/result pair")
        pairs.append({"config": config, "performance": float(score), "status": "success",
                      "trial_index": None, "trial_order_known": False, "elapsed_seconds": None})
    return pairs


def from_result(row, result):
    config = config_from_row(row)
    perf = float(row["perf"])
    raw = result.get("result")
    if result["timeout"]:
        outcome = "timeout"
    elif result.get("traceback"):
        outcome = "failed"
    else:
        objectives = raw.get("objectives", raw.get("objective_value")) if isinstance(raw, dict) else raw
        objective = float(objectives[0] if isinstance(objectives, (list, tuple)) else objectives)
        outcome = "success" if math.isfinite(objective) else "nonfinite_result"
        if outcome == "success" and not math.isclose(perf, -objective, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Trajectory score does not match the raw evaluation result")
    if (outcome == "success") != math.isfinite(perf):
        raise ValueError("Trajectory success does not match the raw evaluation outcome")
    if not math.isclose(float(row["eval_time"]), result["elapsed_time"], rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError("Trajectory time does not match the raw evaluation result")
    return {"config": config, "performance": perf if math.isfinite(perf) else None,
            "status": outcome, "trial_index": row["iter_idx"], "trial_order_known": True,
            "elapsed_seconds": result["elapsed_time"], "timeout": bool(result["timeout"]),
            "error": result.get("traceback")}


def write_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def audit(campaign, destination):
    csv.field_size_limit(sys.maxsize)
    summary, errors = {}, []
    for method in run_archive.load_json(campaign / 'protocol.json', {}).get('methods', METHODS):
        counts = Counter()
        records = []
        for task in json.loads((campaign / "reused" / f"{method}.json").read_text()):
            trajectory = Path(task["path"])
            try:
                data = run_archive.read_bytes(trajectory)
                if hashlib.sha256(data).hexdigest() != task["sha256"]:
                    raise ValueError("Selected legacy trajectory changed")
                rows = list(csv.DictReader(data.decode().splitlines()))
                try:
                    pairs = [{"config": config_from_row(row), "performance": float(row["perf"]),
                              "status": "success", "trial_index": row["iter_idx"], "trial_order_known": True,
                              "elapsed_seconds": float(row["eval_time"])} for row in rows]
                    source = trajectory
                except ValueError:
                    files = list(run_archive.iter_files(trajectory.parent, "*topk_config.pkl"))
                    if len(files) != 1:
                        raise ValueError("Expected one run-local saved configuration file")
                    source = files[0]
                    items = pickle.loads(run_archive.read_bytes(source))[task["algorithm"]]
                    pairs = from_topk(rows, items)
                    counts["recovered_pairs_from_run_local_topk"] += len(pairs)
                for pair in pairs:
                    if pair["config"].get("algorithm") != task["algorithm"]:
                        raise ValueError("Saved configuration algorithm does not match the task")
                    records.append({"method": method, "task_type": task["task_type"], "dataset": task["dataset"],
                                    "algorithm": task["algorithm"], "metric": task["metric"], "seed": 1,
                                    "origin": "reused", "trajectory": str(trajectory), "config_result_source": str(source),
                                    **pair})
                counts["reused_tasks"] += 1
                counts["reused_pairs"] += len(pairs)
            except Exception as exc:
                errors.append({"method": method, "path": str(trajectory), "error": str(exc)})
        write_records(destination / "reused" / f"{method}.jsonl", records)
        records = []
        for path in run_archive.iter_files(campaign / "sessions" / method, "progress.json"):
            try:
                progress = run_archive.load_json(path)
                trajectory = Path(progress["trajectory"])
                rows = []
                if run_archive.exists(trajectory):
                    with run_archive.open_text(trajectory, newline="") as stream:
                        rows = list(csv.DictReader(stream))
                committed = progress["attempts"]
                if len(rows) < committed:
                    raise ValueError("Committed evaluation is missing from the trajectory")
                task_type, dataset, algorithm = path.parent.relative_to(campaign / "sessions" / method).parts
                config = run_archive.load_json(trajectory.parent / "config.json")
                for ordinal, row in enumerate(rows[:committed]):
                    folder = path.parent / "evaluations" / str(ordinal)
                    result = pickle.loads(run_archive.read_bytes(folder / "result.pkl"))
                    timing = run_archive.load_json(folder / "execution.json")
                    pair = from_result(row, result)
                    records.append({"method": method, "task_type": task_type, "dataset": dataset,
                                    "algorithm": algorithm, "metric": config["metric"], "seed": config["seed"],
                                    "origin": "current", "trajectory": str(trajectory), "evaluation_id": ordinal,
                                    "raw_result_path": str(folder / "result.pkl"), "execution_path": str(folder / "execution.json"),
                                    "interrupted_executions": timing.get("interrupted_executions", 0), **pair})
                    counts["current_pairs"] += 1
                    counts["current_" + pair["status"]] += 1
                counts["current_sessions"] += 1
            except Exception as exc:
                errors.append({"method": method, "path": str(path), "error": str(exc)})
        write_records(destination / "current_snapshot" / f"{method}.jsonl", records)
        summary[method] = dict(counts)
    report = {"timestamp": time.time(), "campaign": str(campaign), "summary": summary, "errors": errors,
              "note": "Current exports are snapshots; source trajectories and evaluation journals update after each attempt. "
                      "Recovered legacy top-k pairs retain exact configurations and scores, but trial order is unknown. "
                      "Performance uses the optimizer's higher-is-better convention; mse is recorded as negative MSE."}
    write_records(destination / "audit.jsonl", [report])
    print(json.dumps(report, indent=2))
    return not errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if audit(args.campaign, args.output_dir or args.campaign / "record_audit") else 1)
