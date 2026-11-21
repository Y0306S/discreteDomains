#!/usr/bin/env python3
"""Monitor ckpt32 every five minutes until explicitly stopped; never restart workers."""

import argparse
import collections
import csv
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import ml_fixed_campaign as campaign_tools
from ml_run_archive import open_text


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path, default=None):
    return campaign_tools.load(path, default)


def inspect(campaign, previous):
    summary = campaign_tools.status(campaign, quiet=True)
    protocol = read(campaign / "protocol.json")
    issues, workers, ended = [], [], []
    for method, counts in summary.items():
        if method == "ckpt32":
            continue
        for condition in ("incomplete", "complete_with_failures", "liveness_unverified"):
            if counts.get(condition):
                issues.append(f"{method}: {counts[condition]} tasks reported {condition}")
    same_revision = previous.get("ckpt32_revision") == protocol.get("ckpt32_revision")
    old_tasks = {r["task"]: r for r in previous.get("tasks", [])} if same_revision else {}
    rows = []
    for task in read(campaign / "queues/ckpt32.json", []):
        key = campaign_tools.task_key(task)
        state = read(campaign / "state/ckpt32" / f"{key}.json", {})
        progress = read(campaign / "sessions/ckpt32" / key / "progress.json", {})
        record = {"task": key, "attempts": progress.get("attempts", 0),
                  "successful": progress.get("successful", 0), "failed": progress.get("failed", 0),
                  "status": "pending", "pid": state.get("pid"), "gpu": state.get("gpu"),
                  "cpus": state.get("cpus", [])}
        if progress.get("budget_exhausted"):
            record["status"] = "complete" if progress.get("clean_complete") else "complete_with_failures"
        elif state:
            if state.get("host") == "LLMOptimizer2":
                alive = campaign_tools.process_alive(state)
                record["status"] = "running" if alive else "incomplete"
            else:
                record["status"] = "liveness_unverified"
            if record["status"] == "running":
                workers.append(record)
            else:
                issues.append(f"{key}: {record['status']} at {record['attempts']}/100")
        if record["failed"]:
            issues.append(f"{key}: {record['failed']} failed evaluations")
        if record["attempts"] < old_tasks.get(key, {}).get("attempts", 0):
            issues.append(f"{key}: saved attempt count decreased")
        trajectory = progress.get("trajectory")
        if trajectory and record["attempts"]:
            run_dir = Path(trajectory).parent
            config = read(run_dir / "config.json", {})
            params = config.get("optimizer_params", {})
            if (protocol.get("ckpt32_response_format") and
                    params.get("response_format") != protocol["ckpt32_response_format"]):
                issues.append(f"{key}: wrong ckpt32 response format")
            if protocol.get("ckpt32_hybrid_variant") == "HPUCB_arm_p05_retry5":
                for setting, expected in {"llm_duplicate_policy": "history_tpe5", "history_window": 24,
                                          "recent_k": 10, "top_k": 5, "worst_k": 5,
                                          "pucb_p_end": 0.5, "pucb_q": "arm_best"}.items():
                    if params.get(setting) != expected:
                        issues.append(f"{key}: incorrect retry5 setting {setting}")
            verification = params.get("adapter_verification", {})
            if (verification.get("sha256") != protocol.get("adapter_sha256") or
                    verification.get("verified_tensors") != 496):
                issues.append(f"{key}: missing or mismatched ckpt32 adapter verification")
            if params.get("source_gp_manifest_sha256") != protocol.get("source_gp_manifest_sha256"):
                issues.append(f"{key}: source-GP artifact fingerprint mismatch")
            if params.get("preset_warm_start_sha256") != protocol.get("warm_start_sha256"):
                issues.append(f"{key}: warm-start fingerprint mismatch")
            # Full retrieval audits can exceed the CSV reader's default 128 KiB limit.
            csv.field_size_limit(sys.maxsize)
            with open_text(trajectory, newline="") as stream:
                saved_rows = list(csv.DictReader(stream))
            # The trajectory can be one row ahead of its atomic checkpoint while committing.
            committed = saved_rows[:record["attempts"]]
            if len(committed) != record["attempts"] or [int(r["iter_idx"]) for r in committed] != list(range(record["attempts"])):
                issues.append(f"{key}: missing or misnumbered saved evaluation rows")
            for row in committed:
                if not isinstance(json.loads(row["config_json"]), dict) or not row["config_json"]:
                    issues.append(f"{key}: missing evaluated configuration at {row['iter_idx']}")
            for number in range(max(1, old_tasks.get(key, {}).get("attempts", 0) + 1), min(record["attempts"], 9) + 1):
                audit = read(run_dir / "retrieval" / f"trial_{number:04d}.json")
                if audit is None:
                    issues.append(f"{key}: missing retrieval record for evaluation {number}")
                elif any(o["trial"] >= number for o in audit.get("observations", [])):
                    issues.append(f"{key}: future observation in retrieval record {number}")
                elif audit.get("selected_ids") and not (
                    4 <= number <= 8 and committed[number - 1].get(
                        "selected_arm", committed[number - 1].get("stage")) == "llm"
                ):
                    issues.append(f"{key}: RS advice outside eligible early LLM turns at {number}")
        if record["status"] == "complete" and old_tasks.get(key, {}).get("status") != "complete":
            ended.append(key)
        rows.append(record)
    cpu_counts = collections.Counter(cpu for w in workers for cpu in w["cpus"])
    if any(n > 1 for n in cpu_counts.values()):
        issues.append("ckpt32 workers have overlapping CPU assignments")
    gpu_counts = collections.Counter(w["gpu"] for w in workers)
    if any(n > 2 for n in gpu_counts.values()):
        issues.append("More than two ckpt32 workers assigned to a GPU")
    if protocol.get("ckpt32_policy") != "hold" and any(r["status"] == "pending" for r in rows):
        host = read(campaign / "hosts/LLMOptimizer2.json", {})
        if time.time() - host.get("updated_at", 0) > 60:
            issues.append("Controller scheduler heartbeat is stale")
    memory_events = {}
    events_path = Path("/sys/fs/cgroup/memory.events")
    if events_path.exists():
        memory_events = {key: int(value) for key, value in
                         (line.split() for line in events_path.read_text().splitlines())}
        old_oom = previous.get("memory_events", {}).get("oom_kill", memory_events.get("oom_kill", 0))
        if memory_events.get("oom_kill", 0) > old_oom:
            issues.append("New cgroup OOM kills since the previous check")
    return {"checked_at": now(), "ckpt32_revision": protocol.get("ckpt32_revision"),
            "summary": summary, "tasks": rows,
            "issues": sorted(set(issues)), "newly_completed": ended,
            "active_ckpt32": len(workers), "gpu_workers": dict(gpu_counts), "memory_events": memory_events}


def run(args):
    args.loop_dir.mkdir(parents=True, exist_ok=True)
    lock = campaign_tools.claim(args.loop_dir / "monitor.lock")
    if lock is None:
        raise SystemExit("This monitoring loop is already running")
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    initial = time.monotonic()
    due, iteration = initial, 0
    previous = read(args.loop_dir / "latest.json", {})
    state = {"loop_id": "ml-ckpt32-5m", "job": "Check ckpt32 progress, liveness, saved results, and model provenance",
             "campaign": str(args.campaign), "interval_seconds": args.interval, "timezone": "Asia/Shanghai",
             "mechanism": "detached Python process", "pid": os.getpid(), "status": "running",
             "started_at": now(), "start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(") ", 1)[1].split()[19]}
    try:
        while not stopped:
            while not stopped and time.monotonic() < due:
                time.sleep(min(1, due - time.monotonic()))
            if stopped:
                break
            state["last_start"] = now()
            try:
                result = inspect(args.campaign, previous)
            except Exception as exc:
                result = {"checked_at": now(), "issues": [f"Monitoring check failed: {type(exc).__name__}: {exc}"],
                          "previous_successful_result": previous.get("checked_at"), "check_failed": True}
            iteration += 1
            result["iteration"] = iteration
            campaign_tools.save(args.loop_dir / "latest.json", result)
            with (args.loop_dir / "checks.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            fresh = sorted(set(result["issues"]) - set(previous.get("issues", [])))
            if fresh:
                with (args.loop_dir / "alerts.jsonl").open("a") as stream:
                    stream.write(json.dumps({"at": now(), "issues": fresh}) + "\n")
            text = [f"ckpt32 check: {result['checked_at']}",
                    json.dumps(result.get("summary", {}).get("ckpt32", {}), indent=2),
                    "Issues: " + ("; ".join(result["issues"]) or "none detected")]
            (args.loop_dir / "latest.txt").write_text("\n".join(text) + "\n")
            print("\n".join(text), flush=True)
            if not result.get("check_failed"):
                previous = result
            state.update(last_finish=now(), last_result="issues" if result["issues"] else "ok", iteration=iteration)
            due = initial + (int((time.monotonic() - initial) // args.interval) + 1) * args.interval
            state["next_due_unix"] = time.time() + max(0, due - time.monotonic())
            campaign_tools.save(args.loop_dir / "state.json", state)
            if args.once:
                break
    finally:
        state.update(status="stopped", stopped_at=now(), next_due_unix=None)
        campaign_tools.save(args.loop_dir / "state.json", state)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "run", "stop"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--loop-dir", type=Path, default=Path("/root/.codex/loops/ml-ckpt32-5m"))
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("Interval must be positive")
    if args.action == "run":
        run(args)
    elif args.action == "start":
        args.loop_dir.mkdir(parents=True, exist_ok=True)
        with (args.loop_dir / "monitor.log").open("a") as stream:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "run", "--campaign", str(args.campaign),
                                      "--loop-dir", str(args.loop_dir), "--interval", str(args.interval)],
                                     stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"pid": child.pid, "loop_dir": str(args.loop_dir)}))
    else:
        state = read(args.loop_dir / "state.json", {})
        pid = state.get("pid")
        stat = Path(f"/proc/{pid}/stat")
        if state.get("status") == "running" and stat.exists():
            if stat.read_text().rsplit(") ", 1)[1].split()[19] != state.get("start_ticks"):
                raise RuntimeError("Monitor PID was reused")
            os.kill(pid, signal.SIGTERM)
            print(f"Stop requested for monitoring PID {pid}")
        else:
            print("Monitor is not running")


if __name__ == "__main__":
    main()
