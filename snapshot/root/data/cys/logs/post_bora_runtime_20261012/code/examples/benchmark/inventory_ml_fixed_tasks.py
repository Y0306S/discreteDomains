"""Inventory clean completion for the fixed 132-pair ML test split."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import os

from ml_runtime_policy import load_policy
import ml_run_archive as run_archive


LOCAL = Path("/root/blockdata/llmoptimizer_benchmark")
SHARED = Path("/root/data/cys/llmoptimizer_benchmark")
CAMPAIGN = SHARED / "benchmark_data/ml_test_20_fixed_recovery_20260925"
MANIFEST = LOCAL / "benchmark_data/ml_test_20_fixed/tasks_132.json"
MANIFEST_HASH = "498029faf93894031bf612c67255be63c73f4b92e0a4d4c31e625febf41f5eec"
METHODS = ("ckpt32", "tpe", "lolbo", "random_search")
LEGACY = {
    "ckpt32": LOCAL / "benchmark_data/ml_test_20_fixed_ckpt32_rec4_ws3",
    "tpe": LOCAL / "benchmark_data/ml_test_20_fixed_tpe100",
    "lolbo": SHARED / "benchmark_data/ml_test_20_fixed_lolbo100",
    "random_search": SHARED / "benchmark_data/ml_test_20_fixed_rs100",
}


def key(task):
    return "/".join(task[name] for name in ("task_type", "dataset", "algorithm"))


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(payload, stream, indent=2)
        temporary = stream.name
    os.replace(temporary, path)


def scan(root, method, source, targets, policy, errors):
    csv.field_size_limit(sys.maxsize)
    found = defaultdict(list)
    for path in run_archive.iter_trajectories(root):
        try:
            with run_archive.open_text(path, newline="") as stream:
                rows = list(csv.DictReader(stream))
            if not rows:
                continue
            algorithms = {row.get("algorithm") for row in rows}
            if len(algorithms) != 1:
                errors.append({"path": str(path), "reason": "Mixed or missing algorithms"})
                continue
            parts = path.relative_to(root).parts
            task_id = "/".join((parts[0], parts[1], next(iter(algorithms))))
            if task_id not in targets:
                continue
            config_file = path.parent / "config.json"
            config = run_archive.load_json(config_file, {})
            if config.get("seed") != 1:
                errors.append({"path": str(path), "reason": "Seed is missing or not 1"})
                continue
            expected_metric = targets[task_id]["metric"]
            if config.get("metric") != expected_metric:
                errors.append({"path": str(path), "reason": "Metric does not match the target task"})
                continue
            perfs = [float(row["perf"]) for row in rows]
            failed = sum(not math.isfinite(value) for value in perfs)
            finite = len(rows) - failed
            unique_trials = len({row.get("iter_idx") for row in rows}) == len(rows)
            verification = (config.get("optimizer_params") or {}).get("adapter_verification")
            method_verified = method != "ckpt32" or bool(verification and verification.get("verified_tensors") == 496)
            failed_durations = [float(row["eval_time"]) for row, perf in zip(rows, perfs)
                                if not math.isfinite(perf) and row.get("eval_time")]
            events_path = path.parent / "evaluation_events.jsonl"
            timeout_events = exception_events = None
            events = []
            if run_archive.exists(events_path):
                events = [json.loads(line) for line in run_archive.read_bytes(events_path).decode().splitlines() if line.strip()]
                timeout_events = sum(bool(event.get("timeout")) for event in events)
                exception_events = sum(bool(event.get("traceback")) for event in events)
            runtime = config.get("runtime_policy") or {}
            policy_match = (
                runtime.get("policy_id") == policy["policy_id"]
                and config.get("per_run_time_limit") == policy["per_evaluation_seconds"]
                and "time_limit" in config and config["time_limit"] is None
                and runtime.get("evaluation_start_method") == "forkserver"
                and runtime.get("attempts_per_search") == 100
                and runtime.get("failed_attempts_consume_budget") is True
                and len(events) == len(rows)
                and all(event.get("timeout_seconds") == policy["per_evaluation_seconds"]
                        and event.get("start_method") == "forkserver" for event in events)
            )
            found[task_id].append({
                "source": source, "path": str(path), "rows": len(rows), "finite": finite,
                "failed": failed, "evaluation_clean": len(rows) == finite == 100 and unique_trials,
                "method_verified": method_verified, "adapter_verification": verification,
                "clean_complete": len(rows) == finite == 100 and unique_trials and method_verified,
                "policy_match": policy_match, "policy_id": runtime.get("policy_id"),
                "per_evaluation_seconds": config.get("per_run_time_limit"),
                "overall_search_seconds": config.get("time_limit"), "seed": config["seed"],
                "timeout_events_in_attempt": timeout_events,
                "exception_events_in_attempt": exception_events,
                "sampler_checkpoint_exists": run_archive.exists(path.with_name(path.name.replace("_traj.csv", "_study.pkl"))),
                "failed_evaluation_seconds": failed_durations,
            })
        except (OSError, ValueError, TypeError, csv.Error, IndexError) as exc:
            errors.append({"path": str(path), "reason": str(exc)})
    return found


def process_matches(state):
    try:
        process = Path("/proc") / str(state["pid"])
        fields = (process / "stat").read_text().rsplit(") ", 1)[1].split()
        if fields[0] == "Z":
            return False
        command = (process / "cmdline").read_bytes().decode().split("\0")
        return all(command[command.index(flag) + 1] == str(value) for flag, value in (
            ("--dataset", state["dataset"]), ("--algorithm", state["algorithm"]),
            ("--optimizer", state["method"]),
            ("--output-dir", CAMPAIGN / "outputs" / state["method"]),
        ))
    except (OSError, ValueError, IndexError):
        return False


def live_workers(errors):
    states = []
    for path in (CAMPAIGN / "state").rglob("*.json"):
        record = json.loads(path.read_text())
        if record.get("status") == "running":
            states.append(record)
    active = {(state["method"], key(state)): state for state in states
              if state.get("host") == socket.gethostname() and process_matches(state)}
    remote = [state for state in states if state.get("host") != socket.gethostname()]
    if remote:
        # Only the selected benchmark PIDs and their task arguments are inspected.
        program = "import json,os\nfrom pathlib import Path\n"
        program += "states = json.loads(" + repr(json.dumps(remote)) + ")\n"
        program += """live = []
for state in states:
    try:
        p = Path('/proc') / str(state['pid'])
        if p.joinpath('stat').read_text().rsplit(') ', 1)[1].split()[0] == 'Z':
            continue
        argv = p.joinpath('cmdline').read_bytes().decode().split('\\0')
        pairs = [('--dataset', state['dataset']), ('--algorithm', state['algorithm']), ('--optimizer', state['method'])]
        if all(argv[argv.index(flag) + 1] == value for flag, value in pairs):
            live.append(state)
    except (OSError, ValueError, IndexError):
        pass
print(json.dumps(live))
"""
        try:
            result = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "LLMOptimizer4",
                 "/root/data/cys/baseline/envs/bl_lolbo_harness_cu118/bin/python", "-"],
                input=program, text=True, capture_output=True, check=True, timeout=30,
            )
            for state in json.loads(result.stdout):
                active[(state["method"], key(state))] = state
        except (subprocess.SubprocessError, ValueError) as exc:
            errors.append({"host": "LLMOptimizer4", "reason": f"Liveness unverified: {exc}"})
            # Conservatively exclude recorded running tasks from new scheduling.
            for state in remote:
                state["liveness_unverified"] = True
                active[(state["method"], key(state))] = state
    return active


def build_inventory(output, policy_path):
    manifest_bytes = MANIFEST.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_HASH:
        raise ValueError("The authoritative 132-task manifest changed")
    tasks = json.loads(manifest_bytes)
    targets = {key(task): task for task in tasks}
    assert len(tasks) == len(targets) == 132
    policy = load_policy(policy_path)
    errors = []
    active = live_workers(errors)
    summaries, records = {}, []
    for method in METHODS:
        candidates = scan(LEGACY[method], method, "legacy", targets, policy, errors)
        fresh = scan(CAMPAIGN / "outputs" / method, method, "recovery_12h", targets, policy, errors)
        for task_id, attempts in fresh.items():
            candidates[task_id].extend(attempts)
        counts = Counter()
        queues = {name: [] for name in ("reuse_clean", "strict_policy", "ready_without_active", "currently_running")}
        for task in tasks:
            task_id = key(task)
            attempts = sorted(candidates[task_id],
                              key=lambda r: (r["clean_complete"], r["policy_match"], r["rows"],
                                             r["finite"], r["source"] != "legacy"), reverse=True)
            chosen = attempts[0] if attempts else None
            clean = bool(chosen and chosen["clean_complete"])
            clean_policy = next((r for r in attempts if r["clean_complete"] and r["policy_match"]), None)
            state = active.get((method, task_id))
            if clean:
                quality = "clean_complete"
            elif chosen and chosen["evaluation_clean"] and not chosen["method_verified"]:
                quality = "adapter_unverified"
            elif chosen and chosen["rows"] == 100:
                quality = "complete_with_failures"
            elif chosen and chosen["rows"] > 100:
                quality = "over_budget"
            elif chosen:
                quality = "incomplete"
            else:
                quality = "no_trajectory"
            counts[quality] += 1
            counts["strict_policy_complete"] += bool(clean_policy)
            counts["currently_running"] += bool(state)
            if not clean:
                queues["reuse_clean"].append(task)
                if state:
                    queues["currently_running"].append(task)
                else:
                    queues["ready_without_active"].append(task)
            if not clean_policy:
                queues["strict_policy"].append(task)
            if clean:
                action = "reuse_clean_result"
            elif state:
                action = "await_current_attempt"
            else:
                action = "run_fresh"
            records.append({"method": method, **task, "task_id": task_id, "quality": quality,
                            "needs_work_if_reuse_clean": not clean,
                            "needs_work_if_strict_policy": not bool(clean_policy),
                            "currently_running": bool(state), "recommended_action": action,
                            "selected_attempt": chosen, "attempts": attempts,
                            "active_worker": {k: state.get(k) for k in ("pid", "host", "log", "liveness_unverified")}
                            if state else None})
        counts["needs_work_if_reuse_clean"] = len(queues["reuse_clean"])
        counts["needs_work_if_strict_policy"] = len(queues["strict_policy"])
        counts["ready_without_active"] = len(queues["ready_without_active"])
        assert counts["clean_complete"] + counts["needs_work_if_reuse_clean"] == 132
        summaries[method] = dict(counts)
        for name, queue in queues.items():
            save_json(output / "queues" / name / f"{method}.json", queue)
    payload = {"snapshot_at": datetime.now(timezone.utc).isoformat(), "manifest": str(MANIFEST),
               "manifest_sha256": MANIFEST_HASH, "tasks_per_method": 132, "methods": list(METHODS),
               "dataset_count": len({task["dataset"] for task in tasks}), "target_policy": policy,
               "success_definition": "One complete 100-attempt trajectory with 100 finite scores; seed 1 and matching algorithm/metric",
               "reuse_decision": "Reuse completely successful existing runs, as requested by the user",
               "summaries": summaries, "tasks": records, "inspection_errors": errors}
    save_json(output / "inventory.json", payload)
    save_json(output / "tasks_132.json", tasks)
    save_json(output / "runtime_policy.json", json.loads(policy_path.read_text()))
    fields = ["method", "task_type", "dataset", "algorithm", "quality", "attempts", "successful", "failed",
              "currently_running", "recommended_action", "source", "per_evaluation_seconds",
              "overall_search_seconds", "policy_match", "trajectory"]
    with (output / "all_tasks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            chosen = record["selected_attempt"] or {}
            writer.writerow({**{name: record[name] for name in fields[:5]},
                             "attempts": chosen.get("rows", 0), "successful": chosen.get("finite", 0),
                             "failed": chosen.get("failed", 0), "currently_running": record["currently_running"],
                             "recommended_action": record["recommended_action"], "source": chosen.get("source"),
                             "per_evaluation_seconds": chosen.get("per_evaluation_seconds"),
                             "overall_search_seconds": chosen.get("overall_search_seconds"),
                             "policy_match": chosen.get("policy_match", False), "trajectory": chosen.get("path")})
    lines = ["# ML Task Readiness", "", f"Snapshot: {payload['snapshot_at']}", "",
             "Exact split: 132 dataset/algorithm tasks across 77 datasets; four methods, 528 required results.", "",
             "Target policy: 48 hours per evaluation, no overall search timeout, 100 attempts including failures.", "",
             "A clean result has exactly 100 finite evaluations in one trajectory. Attempts are never concatenated.", "",
             "ckpt32 additionally requires verified adapter loading. Its historical logs report all 496 adapter tensors missing; those results are not reusable as ckpt32.", "",
             "| Method | Clean results reused | Complete with failures | Incomplete | No trajectory | Still needs work | Active within that work | Not active |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for method, count in summaries.items():
        needs = count["needs_work_if_reuse_clean"]
        ready = count["ready_without_active"]
        lines.append(f"| {method} | {count.get('clean_complete', 0)} | {count.get('complete_with_failures', 0)} | "
                     f"{count.get('incomplete', 0)} | {count.get('no_trajectory', 0)} | {needs} | {needs-ready} | {ready} |")
    lines += ["", "## Queue Files", "",
              "- `queues/reuse_clean/<method>.json`: all tasks still requiring a clean result, including active work.",
              "- `queues/ready_without_active/<method>.json`: the same list excluding currently running tasks.",
              "- `queues/currently_running/<method>.json`: active tasks that still need a clean result.",
              "- `queues/strict_policy/<method>.json`: tasks without a clean result produced under the approved policy.",
              "- `all_tasks.csv`: all 528 task/method records with counts, actions, and selected evidence paths.",
              "- `inventory.json`: all inspected attempts and their original timing provenance.", "",
              "These files are an inventory, not newly launched queues. Recheck active work before launching.",
              "Existing clean results use older timing policies. Old random-search results also have known package-version differences.",
              "The readiness inventory does not establish identical software/hardware or final held-out-row test performance.",
              "The user selected reuse of completely successful existing runs and requested stopping LOLBO/random search on all machines.",
              "Legacy-worker stop audits are in `stopped/`. This inventory inspects legacy/recovery outputs; the new 48-hour campaign has its own live `task_index.json`.", "",
              "## Completed Tasks With Failures", ""]
    for method in METHODS:
        lines += [f"### {method}", ""]
        failures = [r for r in records if r["method"] == method and r["quality"] == "complete_with_failures"]
        for record in failures:
            attempt = record["selected_attempt"]
            lines.append(f"- `{record['task_id']}`: {attempt['finite']}/100 finite; {attempt['failed']} failures.")
        if not failures:
            lines.append("None.")
        lines.append("")
    (output / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"snapshot_at": payload["snapshot_at"], "summaries": summaries,
                      "inspection_errors": errors, "report": str(output / "REPORT.md")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=LOCAL / "configs/ml_runtime_policy.json")
    args = parser.parse_args()
    build_inventory(args.output_dir, args.policy)


if __name__ == "__main__":
    main()
