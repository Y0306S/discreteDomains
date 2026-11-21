#!/usr/bin/env python3
"""Five-minute backfill checks and a verified, one-time CPU training handoff."""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

LOOP_ID = "taskset-transfer-5m"
CAMPAIGN = Path("/root/data/logs/taskset_major_families_cpu_20260929")
TRAINING = Path("/root/data/logs/taskset_transfer_train838_20261002")
LOOP = Path("/root/.codex/loops") / LOOP_ID
STOP = False


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".%d.tmp" % os.getpid())
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_history(job):
    path = Path(job["output_root"]) / job["task_id"] / job["method"] / "seed1/history.json"
    archive = path.parent.with_name("seed1.tar.gz")
    if archive.exists():
        with tarfile.open(archive, "r:gz") as stream:
            manifest = json.load(stream.extractfile("manifest.json"))
            raw = stream.extractfile("run/history.json").read()
            expected = next(r["sha256"] for r in manifest["entries"] if r["name"] == "run/history.json")
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("History checksum mismatch: " + job["id"])
        return json.loads(raw)
    for candidate in (path, path.with_name("history.prev.json")):
        try:
            return json.loads(candidate.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    if archive.exists():
        return read_history(job)
    raise FileNotFoundError(path)


def remote(action):
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "LLMOptimizer4",
               "/usr/bin/python3", str(TRAINING / "code/taskset_transfer_handoff.py"), action,
               "--training", str(TRAINING)]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=55)
    if result.returncode:
        raise RuntimeError("Remote %s failed: %s" % (action, result.stderr[-2000:]))
    return json.loads(result.stdout)


def iteration():
    jobs = json.loads((CAMPAIGN / "distributed/plan.json").read_text())["jobs"]
    rows = []
    for job in jobs:
        if job["side"] != "train":
            continue
        history = read_history(job)
        observations = history["observations"]
        meta = history["meta_info"]
        if (history["task_id"] != job["task_id"] or meta["domain"] != "taskset"
                or meta["optimizer"] != "tpe" or meta["seed"] != 1 or len(observations) > 100):
            raise ValueError("Backfill identity/budget mismatch")
        rows.append({"task_id": job["task_id"], "observations": len(observations),
                     "successful": sum(o["trial_state"] == 0 and math.isfinite(o["objectives"][0]) for o in observations)})
    if len(rows) != 11:
        raise ValueError("Expected exactly eleven backfills")
    result = {"backfills": rows, "completed": sum(r["observations"] == 100 for r in rows), "total": 11}
    ready = TRAINING / "launch_ready.json"
    if ready.exists():
        result["handoff"] = remote("poll")
    else:
        result["handoff"] = {"status": "waiting_for_verified_training_setup"}
    result["training"] = {}
    for method in ("fsbo", "rgpe"):
        path = TRAINING / method / "status.json"
        result["training"][method] = json.loads(path.read_text()) if path.exists() else {"status": "not_started"}
    return result


def run(lock_fd=None):
    global STOP
    LOOP.mkdir(parents=True, exist_ok=True)
    lock = os.fdopen(lock_fd, "a") if lock_fd is not None else (LOOP / "monitor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def stop(signum, frame):
        global STOP
        STOP = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    interval = 300
    anchor, monotonic_anchor = time.time(), time.monotonic()
    state = {"loop_id": LOOP_ID, "job": "Monitor TaskSet backfills and launch FSBO/RGPE after verified completion",
             "targets": [str(CAMPAIGN), str(TRAINING)], "interval_seconds": interval,
             "timezone": "Asia/Shanghai", "mechanism": "detached Python process; file reports, no automatic chat notifications",
             "schedule_id": "process:%d" % os.getpid(), "pid": os.getpid(), "status": "running",
             "start_ticks": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
             "started_at": anchor, "iteration": 0}
    write(LOOP / "state.json", state)
    try:
        while not STOP:
            state["last_start"] = time.time()
            state["iteration"] += 1
            try:
                report = iteration()
                handoff_status = report['handoff']['status']
                state["last_result"] = ("issues" if handoff_status in ('failed', 'backfill_failed', 'controller_unverified',
                                           'uncertain_controller_launch_requires_inspection') else "ok")
                state.pop("last_error", None)
                state["completed_backfills"] = report["completed"]
                state["handoff_status"] = report["handoff"]["status"]
            except Exception as exc:
                report = {"status": "unverified", "error": repr(exc)}
                state["last_result"] = "error"
                state["last_error"] = repr(exc)
            report["checked_at"] = time.time()
            write(LOOP / "latest.json", report)
            with (LOOP / "history.jsonl").open("a") as stream:
                stream.write(json.dumps(report) + "\n")
            state["last_finish"] = time.time()
            slot = math.floor((time.monotonic() - monotonic_anchor) / interval) + 1
            due = monotonic_anchor + slot * interval
            state["next_due_unix"] = anchor + slot * interval
            write(LOOP / "state.json", state)
            print(json.dumps({k: v for k, v in state.items() if k not in ("targets", "mechanism")}), flush=True)
            while not STOP and time.monotonic() < due:
                time.sleep(min(1, due - time.monotonic()))
    finally:
        if (TRAINING / "handoff.json").exists():
            try:
                state["stop_handoff"] = remote("cancel-drain")
            except Exception as exc:
                state["stop_handoff_error"] = repr(exc)
        state.update(status="stopped", next_due_unix=None, stopped_at=time.time())
        write(LOOP / "state.json", state)
        lock.close()


def start():
    LOOP.mkdir(parents=True, exist_ok=True)
    with (LOOP / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (LOOP / "monitor.log").open("a") as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "run",
                                        "--lock-fd", str(lock.fileno())],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, pass_fds=(lock.fileno(),))
    print(json.dumps({"loop_id": LOOP_ID, "pid": process.pid, "interval_seconds": 300, "state": str(LOOP / "state.json")}))


def stop_loop():
    state = json.loads((LOOP / 'state.json').read_text())
    path = Path('/proc') / str(state['pid']) / 'stat'
    if not path.exists():
        state.update(status='stopped', next_due_unix=None)
        write(LOOP / 'state.json', state)
        return
    if path.read_text().rsplit(')', 1)[1].split()[19] != state['start_ticks']:
        raise RuntimeError('Loop PID identity changed; refusing to signal another process')
    os.kill(state['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        final = json.loads((LOOP / 'state.json').read_text())
        if final['status'] == 'stopped':
            print(json.dumps(final))
            return
        time.sleep(0.5)
    raise RuntimeError('Loop stop is still pending; inspect state.json')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "run", "once", "stop"))
    parser.add_argument("--lock-fd", type=int)
    args = parser.parse_args()
    if args.action == "start":
        start()
    elif args.action == 'stop':
        stop_loop()
    elif args.action == "once":
        print(json.dumps(iteration(), indent=2))
    else:
        run(args.lock_fd)
