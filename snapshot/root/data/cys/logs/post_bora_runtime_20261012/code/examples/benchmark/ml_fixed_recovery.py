"""Prepare, run, and inspect the versioned pair-held-out ML recovery campaign."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time


SOURCE = Path("/root/blockdata/llmoptimizer_benchmark")
SHARED = Path("/root/data/cys/llmoptimizer_benchmark")
DEFAULT_CAMPAIGN = SHARED / "benchmark_data/ml_test_20_fixed_recovery_20260925"
EXPECTED_SHA256 = "498029faf93894031bf612c67255be63c73f4b92e0a4d4c31e625febf41f5eec"
METHODS = ("lolbo", "random_search", "tpe")
PYTHON = "/root/data/cys/baseline/envs/bl_lolbo_harness_cu118/bin/python"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(payload, stream, indent=2)
        temporary = stream.name
    os.replace(temporary, path)


def task_key(task):
    return "/".join(task[k] for k in ("task_type", "dataset", "algorithm"))


def trajectory(path, algorithm):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(row.get("algorithm") != algorithm for row in rows):
        return None
    finite = sum(math.isfinite(float(row["perf"])) for row in rows)
    return {"path": str(path), "rows": len(rows), "finite": finite, "failed": len(rows) - finite}


def attempts(root, task):
    found = []
    for path in (root / task["task_type"] / task["dataset"]).rglob("*_traj.csv"):
        record = trajectory(path, task["algorithm"])
        if record:
            found.append(record)
    return sorted(found, key=lambda row: (row["rows"], row["finite"], row["path"]), reverse=True)


def prepare(campaign):
    manifest = SOURCE / "benchmark_data/ml_test_20_fixed/tasks_132.json"
    assert digest(manifest) == EXPECTED_SHA256, "Target manifest changed"
    tasks = json.loads(manifest.read_text())
    assert len(tasks) == len({task_key(t) for t in tasks}) == 132
    assert len({t["dataset"] for t in tasks}) == 77
    campaign.mkdir(parents=True, exist_ok=False)
    shutil.copy2(manifest, campaign / "tasks_132.json")
    sizes = {}
    for filename in ("classification.csv", "regression.csv"):
        with (SOURCE / "datasets_bundle/dataset_info" / filename).open() as stream:
            for row in csv.DictReader(stream):
                sizes[row["name"]] = float(row.get("n") or 1e12)
    tasks.sort(key=lambda t: (sizes.get(t["dataset"], 1e12), task_key(t)))
    code = campaign / "code"
    code.mkdir()
    data_inventory = {}
    for task in tasks:
        dataset_key = f"{task['task_type']}/{task['dataset']}"
        if dataset_key in data_inventory:
            continue
        def data_path(repo):
            root = repo / "datasets_bundle"
            small = root / f"{task['task_type']}_datasets_downsampled" / f"{task['dataset']}_downsampled.csv"
            return small if small.exists() else root / f"{task['task_type']}_datasets" / f"{task['dataset']}.csv"
        source_path = data_path(SOURCE)
        relative = source_path.relative_to(SOURCE)
        destination = code / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        source_hash = digest(destination)
        shared_path = data_path(SHARED)
        data_inventory[dataset_key] = {"source": str(source_path), "snapshot": str(relative),
                                      "sha256": source_hash,
                                      "shared_data_matches": digest(shared_path) == source_hash}
    shutil.copytree(SOURCE / "datasets_bundle/dataset_info", code / "datasets_bundle/dataset_info")
    write_json(campaign / "data_manifest.json", data_inventory)
    inventory, counts = {}, {}
    roots = {"lolbo": SHARED / "benchmark_data/ml_test_20_fixed_lolbo100",
             "random_search": SHARED / "benchmark_data/ml_test_20_fixed_rs100",
             "tpe": SOURCE / "benchmark_data/ml_test_20_fixed_tpe100"}
    for method in METHODS:
        queue, reused = [], []
        inventory[method] = {}
        for task in tasks:
            previous = attempts(roots[method], task)
            inventory[method][task_key(task)] = previous
            clean = next((p for p in previous if p["rows"] == p["finite"] == 100), None)
            same_data = data_inventory[f"{task['task_type']}/{task['dataset']}"]["shared_data_matches"]
            if method != "tpe" and clean and same_data:
                destination = campaign / "reused" / method / task_key(task) / "trajectory.csv"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(clean["path"], destination)
                reused.append({**task, **clean, "sha256": digest(destination),
                               "snapshot": str(destination), "protocol": "legacy; retain original timing provenance"})
            else:
                queue.append(task)
        queue.sort(key=lambda task: task_key(task) != "rgs/black_friday/extra_trees")
        write_json(campaign / "queues" / f"{method}.json", queue)
        write_json(campaign / "reused" / f"{method}.json", reused)
        counts[method] = {"queued": len(queue), "reused": len(reused)}
    write_json(campaign / "legacy_inventory.json", inventory)
    for name in ("mindware", "examples", "configs"):
        shutil.copytree(SOURCE / name, code / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    files = {str(path.relative_to(code)): digest(path) for path in code.rglob("*")
             if path.is_file() and "datasets_bundle" not in path.parts}
    write_json(campaign / "code_sha256.json", files)
    protocol = {
        "campaign": campaign.name, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "split": "ml_test_20_fixed; dataset-algorithm pairs held out",
        "manifest_sha256": EXPECTED_SHA256, "tasks": 132, "datasets": 77,
        "n_trials": 100, "seed": 1, "evaluation": "holdout", "test_size": 0.25,
        "valid_size": 0.35, "skip_test_eval": True,
        "per_evaluation_seconds": 43200, "task_seconds": 4406400,
        "task_limit_reason": "100 x 12 hours plus 24 hours optimizer/setup headroom",
        "evaluation_start_method": "forkserver", "evaluation_cpus": 4,
        "failure_policy": "Failures count toward 100 attempts; preserve Optuna FAIL handling and search spaces",
        "automatic_retries": 0, "python": PYTHON, "counts": counts,
        "owners": {"lolbo": "controller", "tpe": "controller", "random_search": "LLMOptimizer4"},
        "hostnames": {"lolbo": socket.gethostname(), "tpe": socket.gethostname(),
                      "random_search": "cs-fa498-8793f-server"},
        "parallel_workers_per_method": 4,
        "reuse_limit": "Clean completed legacy runs only. Resource policy differs; do not pool wall-time curves.",
    }
    write_json(campaign / "protocol.json", protocol)
    print(json.dumps(protocol, indent=2), flush=True)


def current(campaign, method, task):
    directory = campaign / "outputs" / method / task_key(task) / method
    records = []
    for path in directory.glob("CASH-*/*_traj.csv"):
        record = trajectory(path, task["algorithm"])
        if record:
            records.append(record)
    return max(records, key=lambda row: row["rows"], default={"rows": 0, "finite": 0, "failed": 0})


def worker(args):
    campaign = args.campaign
    protocol = json.loads((campaign / "protocol.json").read_text())
    assert digest(campaign / "tasks_132.json") == protocol["manifest_sha256"] == EXPECTED_SHA256
    assert socket.gethostname() == protocol["hostnames"][args.method], "Wrong task owner"
    slot_lock_path = campaign / "locks" / f"{args.method}.slot{args.slot}.lock"
    slot_lock_path.parent.mkdir(parents=True, exist_ok=True)
    slot_lock = slot_lock_path.open("a")
    try:
        fcntl.flock(slot_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Worker slot is already active")
    assert len(args.cpus.split(",")) == protocol["evaluation_cpus"]
    cpus = [int(cpu) for cpu in args.cpus.split(",")]
    assert set(cpus) <= os.sched_getaffinity(0)
    queue = json.loads((campaign / "queues" / f"{args.method}.json").read_text())
    env = os.environ.copy()
    code = campaign / "code"
    env.update({"PYTHONPATH": f"{code}/examples/benchmark:{code}",
                "PYTHONUNBUFFERED": "1", "ML_ISOLATED_EVALUATION": "1",
                "CUDA_VISIBLE_DEVICES": args.gpu, "EVAL_N_JOBS": "4",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    for index, task in enumerate(queue):
        if index % args.slots != args.slot:
            continue
        key = task_key(task)
        state_path = campaign / "state" / args.method / f"{key}.json"
        lock_path = campaign / "locks" / args.method / f"{key}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f"LOCKED {args.method} {key}", flush=True)
                continue
            progress = current(campaign, args.method, task)
            if progress["rows"] >= protocol["n_trials"]:
                continue
            # Resume requires the actual sampler state, never a fresh sampler with old row counts.
            if progress["rows"] and not Path(progress["path"].replace("_traj.csv", "_study.pkl")).is_file():
                write_json(state_path, {"status": "blocked_missing_sampler_state", **progress})
                continue
            command = ["taskset", "-c", args.cpus, protocol["python"],
                       str(code / "examples/benchmark/run_toy_categorical_optimizers.py"),
                       "--optimizer", args.method, "--task-type", task["task_type"],
                       "--dataset", task["dataset"], "--algorithm", task["algorithm"],
                       "--metric", task["metric"], "--n-trials", str(protocol["n_trials"]),
                       "--seed", str(protocol["seed"]), "--time-limit", str(protocol["task_seconds"]),
                       "--per-run-time-limit", str(protocol["per_evaluation_seconds"]),
                       "--max-rounds", "1", "--dataset-retries", "1", "--skip-test-eval", "--resume",
                       "--output-dir", str(campaign / "outputs" / args.method),
                       "--history-dir", str(campaign / "history" / args.method)]
            log_path = campaign / "logs" / args.method / f"{key}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            state = {"method": args.method, **task, "status": "running", "host": socket.gethostname(),
                     "supervisor_pid": os.getpid(), "cpus": cpus, "gpu": args.gpu,
                     "started_at": time.time(), "command": command, "log": str(log_path)}
            print(f"START {args.method} {key}", flush=True)
            with log_path.open("a") as log:
                child = subprocess.Popen(command, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT)
                state["pid"] = child.pid
                write_json(state_path, state)
                returncode = child.wait()
            progress = current(campaign, args.method, task)
            complete = progress["rows"] == protocol["n_trials"]
            state.update(progress, returncode=returncode, finished_at=time.time(),
                         status=("complete_with_failures" if progress["failed"] else "complete")
                         if complete else "incomplete")
            write_json(state_path, state)
            print(f"END {args.method} {key} {state['status']} {progress['rows']}/100 rc={returncode}", flush=True)


def status(campaign):
    report = {}
    for method in METHODS:
        queue = json.loads((campaign / "queues" / f"{method}.json").read_text())
        reused = json.loads((campaign / "reused" / f"{method}.json").read_text())
        counts = {"reused": len(reused), "new_complete": 0, "running": 0,
                  "incomplete": 0, "pending": 0, "new_rows": 0, "failed_evaluations": 0}
        active = []
        for task in queue:
            progress = current(campaign, method, task)
            counts["new_rows"] += progress["rows"]
            counts["failed_evaluations"] += progress["failed"]
            path = campaign / "state" / method / f"{task_key(task)}.json"
            state = json.loads(path.read_text()) if path.exists() else {}
            if progress["rows"] == 100:
                counts["new_complete"] += 1
            elif state.get("status") == "running":
                counts["running"] += 1
                active.append({"task": task_key(task), **progress,
                               "host": state["host"], "pid": state["pid"]})
            elif state:
                counts["incomplete"] += 1
            else:
                counts["pending"] += 1
        report[method] = {**counts, "active": active}
    print(json.dumps(report, indent=2))


def start(args):
    protocol = json.loads((args.campaign / "protocol.json").read_text())
    assert socket.gethostname() == protocol["hostnames"][args.method], "Wrong task owner"
    script = args.campaign / "code/examples/benchmark/ml_fixed_recovery.py"
    for slot in range(4):
        lock_path = args.campaign / "locks" / f"{args.method}.slot{slot}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f"Already active: {args.method} slot {slot}", flush=True)
                continue
        first_cpu = {"lolbo": 0, "tpe": 16, "random_search": 80}[args.method] + 4 * slot
        cpus = ",".join(str(i) for i in range(first_cpu, first_cpu + 4))
        command = [sys.executable, str(script), "worker", "--campaign", str(args.campaign),
                   "--method", args.method, "--slot", str(slot), "--slots", "4", "--cpus", cpus]
        if args.method == "lolbo":
            command += ["--gpu", str(slot)]
        log_path = args.campaign / "logs" / f"worker_{args.method}_{slot}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        write_json(args.campaign / "workers" / f"{args.method}_{slot}.json",
                   {"pid": process.pid, "host": socket.gethostname(), "command": command,
                    "log": str(log_path), "started_at": time.time()})
        print(f"Started {args.method} slot={slot} pid={process.pid} cpus={cpus}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "start", "worker", "status"))
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--slots", type=int, default=1)
    parser.add_argument("--cpus", default="0,1,2,3")
    parser.add_argument("--gpu", default="")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.campaign)
    elif args.action == "start":
        if args.method is None:
            parser.error("start requires --method")
        start(args)
    elif args.action == "worker":
        if args.method is None or not 0 <= args.slot < args.slots:
            parser.error("worker requires --method and 0 <= slot < slots")
        worker(args)
    else:
        status(args.campaign)


if __name__ == "__main__":
    main()
