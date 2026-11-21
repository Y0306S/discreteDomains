"""Rerun the ML FSBO test campaign with per-task min-max checkpoints (2026-10-06).

The 2026-09-28 FSBO checkpoints were meta-trained on global-range rescaled labels and
selected against raw-scale validation (classification picked at steps 100-1200). The
2026-10-06 checkpoints use upstream per-task MinMaxScaler labels and min-max targets at
suggest time. Everything else matches ml_transfer_campaign.py.
"""

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from ml_fixed_campaign import claim, digest, load, process_alive, save

SOURCE = Path("/root/blockdata/llmoptimizer_benchmark")
PREVIOUS = Path("/root/data/cys/llmoptimizer_benchmark/benchmark_data/ml_test_20_fixed_48h_20260925")
DEFAULT = PREVIOUS.parent / "ml_test_20_fixed_fsbo_minmax_20261006"
SUPERSEDES = PREVIOUS.parent / "ml_test_20_fixed_fsbo_rgpe_20260928"
FSBO_RUN = PREVIOUS.parent / "ml_fsbo_minmax_train_20261006"
PYTHON = "/root/data/cys/baseline/envs/bl_lolbo_harness_cu118/bin/python"
METHODS = ("fsbo",)
SERIAL = {"adaboost", "gradient_boosting"}
# Shares LLMOptimizer4 with the TaskSet FSBO rerun (physical cores 0-47); stays on 48-79.
HOSTS = {
    "LLMOptimizer4": {"hostname": "cs-fa498-8793f-server", "cpus": list(range(48, 80)),
                      "quota": 64, "memory_gib": 200, "initial_workers": 4, "max_workers": 10,
                      "method_order": ["fsbo"], "method_weights": {"fsbo": 1}},
}


def task_key(task):
    return "/".join(task[name] for name in ("task_type", "dataset", "algorithm"))


def task_paths(campaign, method, task):
    key = task_key(task)
    return (campaign / "sessions" / method / key,
            campaign / "state" / method / (key + ".json"),
            campaign / "locks" / method / (key + ".lock"))


def progress(campaign, method, task):
    session, _, _ = task_paths(campaign, method, task)
    return load(session / "progress.json", {"attempts": 0, "successful": 0, "failed": 0,
                "budget_exhausted": False, "clean_complete": False, "pending": False})


def terminal_state(state, info):
    if state.get("status") in ("incomplete", "complete_with_failures"):
        return state["status"]
    if state.get("status") == "complete":
        if (state.get("run_identity") == info.get("identity") and info.get("identity")
                and info.get("attempts") == info.get("successful") == 100
                and info.get("clean_complete") and not info.get("pending")):
            return "complete"
        return "incomplete"
    return None


def prepare(campaign):
    from mindware.components.optimizers.transfer_hpo.protocol import TRAIN_MANIFEST, HISTORY_ROOT, validate_dump, allowed_task_ids
    assert digest(PREVIOUS / "tasks_132.json") == "498029faf93894031bf612c67255be63c73f4b92e0a4d4c31e625febf41f5eec"
    campaign.mkdir(parents=True, exist_ok=False)
    for name in ("tasks_132.json", "runtime_policy.json", "data_manifest.json"):
        shutil.copy2(PREVIOUS / name, campaign / name)
    code = campaign / "code"
    for name in ("mindware", "examples", "configs"):
        shutil.copytree(SOURCE / name, code / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (code / "datasets_bundle").symlink_to(PREVIOUS / "code/datasets_bundle", target_is_directory=True)
    assets = campaign / "assets"
    assets.mkdir()
    shutil.copy2(TRAIN_MANIFEST, assets / "train_manifest.jsonl")
    targets = load(campaign / "tasks_132.json")
    assert len(targets) == len({task_key(t) for t in targets}) == 132
    sources = set()
    groups = []
    for source in sorted((SOURCE / "benchmark_data/ml_train_80_transfer_v2").glob("*.json")):
        dump = load(source)
        if not isinstance(dump, dict) or "tasks" not in dump:
            continue
        ids = validate_dump(dump)
        assert not set(ids) & {task_key(t) for t in targets}
        sources.update(ids)
        groups.append(source.stem)
        destination = assets / "dumps" / source.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(source, destination)
        for record in dump["tasks"].values():
            origin = Path(record["source_path"])
            destination = assets / "history" / origin.relative_to(HISTORY_ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(origin, destination)
            assert digest(destination) == record["source_sha256"]
        for name in ("weights", "meta.json"):
            origin = FSBO_RUN / "fsbo" / source.stem / name
            destination = assets / "fsbo" / source.stem / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, destination)
        fsbo_meta = load(assets / "fsbo" / source.stem / "meta.json")
        assert fsbo_meta["status"] == "complete" and fsbo_meta.get("target_scaling") == "minmax"
        assert fsbo_meta["dump_sha256"] == digest(assets / "dumps" / source.name)
        for suffix in (".json", ".pkl"):
            origin = SOURCE / "benchmark_data/ml_train_80_rgpe_sources" / (source.stem + "_seed1" + suffix)
            destination = assets / "rgpe" / origin.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(origin, destination)
    assert len(groups) == 16 and sources <= allowed_task_ids()
    costs = {}
    csv.field_size_limit(sys.maxsize)
    for record in load(PREVIOUS / "task_index.json")["tasks"]:
        path = record.get("trajectory")
        if not path or not Path(path).is_file():
            continue
        with open(path, newline="") as stream:
            durations = [float(row.get("eval_time") or 0) for row in csv.DictReader(stream)]
        if durations:
            estimate = sum(durations) / len(durations) * 100
            costs[record["task_id"]] = max(costs.get(record["task_id"], 0), estimate)
    data = load(campaign / "data_manifest.json")
    for task in targets:
        task["estimated_seconds"] = costs.get(task_key(task), 0)
        file = code / data[task["task_type"] + "/" + task["dataset"]]["snapshot"]
        task["memory_reservation_gib"] = max(4.0, min(24.0, 2 + 16 * file.stat().st_size / 2**30))
    targets.sort(key=lambda task: (-task["estimated_seconds"], task_key(task)))
    save(campaign / "queue.json", targets)
    checksums = {str(p.relative_to(campaign)): digest(p)
                 for root in (code, assets) for p in root.rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts}
    for name in ("tasks_132.json", "runtime_policy.json", "data_manifest.json", "queue.json"):
        checksums[name] = digest(campaign / name)
    save(campaign / "checksums.json", checksums)
    protocol = {"created_at": time.time(), "methods": list(METHODS), "tasks_per_method": 132,
                "seed": 1, "hosts": HOSTS, "checksums_sha256": digest(campaign / "checksums.json"),
                "source_groups": groups, "unique_training_sources": len(sources),
                "ei": "unchanged upstream paper implementation", "candidate_pool": 256,
                "startup_trials": 3, "transfer_worker_timeout_seconds": 300,
                "automatic_process_retries": 0, "ramp_seconds": 180,
                "cpu_policy": "1 CPU for AdaBoost/gradient boosting; 4 for other estimators",
                "fresh_start": "All 132 FSBO target searches use new durable sessions",
                "supersedes": str(SUPERSEDES), "fsbo_training_run": str(FSBO_RUN),
                "fsbo_label_scaling": "per_task_minmax_upstream_get_batch", "fsbo_target_scaling": "minmax",
                "diagnostics": "Retain every evaluated configuration, result, and model decision"}
    save(campaign / "protocol.json", protocol)
    print(json.dumps({"campaign": str(campaign), "tasks": 264, "source_groups": len(groups), "training_sources": len(sources)}), flush=True)


def worker_env(campaign, cpus):
    env = os.environ.copy()
    for name in ("ML_SESSION_DIR", "ML_RUN_ID", "ML_EVALUATION_ID"):
        env.pop(name, None)
    env.update({"PYTHONPATH": str(campaign / "code"), "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
                "EVAL_N_JOBS": str(len(cpus)), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                "ML_TRANSFER_HISTORY_MIRROR": str(campaign / "assets/history"),
                "ML_TRANSFER_TRAIN_MANIFEST": str(campaign / "assets/train_manifest.jsonl")})
    return env


def command(campaign, method, task, cpus, root=None, policy=None):
    root = campaign if root is None else root
    session, _, _ = task_paths(root, method, task)
    # Each algorithm owns its output root, preventing dataset-summary collisions.
    output = root / "outputs" / method / task["algorithm"]
    args = ["taskset", "-c", ",".join(map(str, cpus)), PYTHON,
            str(campaign / "code/examples/benchmark/run_toy_categorical_optimizers.py"),
            "--optimizer", method, "--task-type", task["task_type"], "--dataset", task["dataset"],
            "--algorithm", task["algorithm"], "--metric", task["metric"], "--seed", "1",
            "--runtime-policy", str(policy or campaign / "runtime_policy.json"),
            "--session-dir", str(session), "--output-dir", str(output),
            "--history-dir", str(root / "history" / method / task["algorithm"]),
            "--transfer-dump-dir", str(campaign / "assets/dumps"),
            "--fsbo-ckpt-dir", str(campaign / "assets/fsbo"),
            "--rgpe-cache-dir", str(campaign / "assets/rgpe"),
            "--transfer-n-pending", "256", "--transfer-worker-timeout", "300",
            "--tpe-startup-trials", "3", "--max-rounds", "1", "--dataset-retries", "1",
            "--skip-test-eval", "--resume"]
    return args, worker_env(campaign, cpus), session


def verify(campaign, host):
    spec = load(campaign / "protocol.json")["hosts"][host]
    assert socket.gethostname() == spec["hostname"]
    assert set(spec["cpus"]) <= os.sched_getaffinity(0)
    quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
    assert quota != "max" and int(quota) / int(period) >= spec["quota"]
    assert int(Path("/sys/fs/cgroup/memory.max").read_text()) >= spec["memory_gib"] * 2**30
    assert digest(campaign / "checksums.json") == load(campaign / "protocol.json")["checksums_sha256"]
    checksums = load(campaign / "checksums.json")
    for name, expected in checksums.items():
        if digest(campaign / name) != expected:
            raise RuntimeError("Snapshot changed: " + name)
    for record in load(campaign / "data_manifest.json").values():
        if digest(campaign / "code" / record["snapshot"]) != record["sha256"]:
            raise RuntimeError("Dataset snapshot changed: " + record["snapshot"])
    print(json.dumps({"host": host, "verified_files": len(checksums), "datasets_verified": True}), flush=True)


def smoke(campaign, host):
    verify(campaign, host)
    root = campaign / "preflight" / host
    root.mkdir(parents=True, exist_ok=False)
    policy = load(campaign / "runtime_policy.json")
    policy.update(policy_id="fsbo_rgpe_remote_preflight_only", attempts_per_search=4)
    save(root / "policy.json", policy)
    task = {"task_type": "cls", "dataset": "car", "algorithm": "extra_trees", "metric": "bal_acc"}
    results = {}
    for method in METHODS:
        cpus = [HOSTS[host]["cpus"][0]]
        args, env, session = command(campaign, method, task, cpus, root, root / "policy.json")
        with (root / (method + ".log")).open("w") as stream:
            run = subprocess.run(args, cwd=campaign / "code", env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, timeout=600)
        if run.returncode:
            raise RuntimeError("Remote preflight failed; inspect " + str(root / (method + ".log")))
        info = load(session / "progress.json")
        assert info["attempts"] == info["successful"] == 4 and info["clean_complete"]
        decisions = list((session / "decisions/trial-003").glob("attempt-*"))
        assert len(decisions) == 1
        decision = decisions[0]
        response, request = load(decision / "response.json"), load(decision / "request.json")
        metadata = load(decision / "worker.json")
        assert metadata["selected_config"] == request["pending_configs"][response["index"]]
        assert metadata["status"] == "succeeded" and len(response["acquisition"]["values"]) == 256
        if method == "rgpe":
            assert len(response["source_weights"]) == response["n_sources"]
        results[method] = {"successful": 4, "diagnostics_verified": True, "identity": info["identity"]}
        print(json.dumps({"host": host, "method": method, "preflight": results[method]}), flush=True)
    save(root / "passed.json", {"host": host, "checked_at": time.time(), "excluded_from_production": True,
                                "checksums_sha256": digest(campaign / "checksums.json"), "results": results})


def choose_task(spec, tasks, active, unavailable, free, memory, age):
    limit = spec["initial_workers"] if age < 180 else spec["max_workers"]
    if len(active) >= limit or memory >= .85 * spec["memory_gib"] * 2**30:
        return None
    counts = Counter(item["state"]["method"] for item in active.values())
    methods = sorted(spec["method_order"], key=lambda m: counts[m] / spec["method_weights"][m])
    reserved = sum(item["state"].get("memory_reservation_gib", 4) for item in active.values())
    for method in methods:
        for task in tasks:
            key = (method, task_key(task))
            size = 1 if task["algorithm"] in SERIAL else 4
            need = task.get("memory_reservation_gib", 4)
            if key in active or key in unavailable or size > len(free):
                continue
            if reserved + need > .75 * spec["memory_gib"] or memory + need * 2**30 > .9 * spec["memory_gib"] * 2**30:
                continue
            return method, task, free[:size]
    return None


def status(campaign, quiet=False):
    tasks = load(campaign / "queue.json")
    summary, records = {}, []
    for method in METHODS:
        counts = Counter({name: 0 for name in ("complete", "running", "pending", "incomplete", "complete_with_failures", "liveness_unverified")})
        for task in tasks:
            _, state_path, _ = task_paths(campaign, method, task)
            state, info = load(state_path, {}), progress(campaign, method, task)
            value = terminal_state(state, info) or state.get("status", "pending")
            if value == "running":
                heartbeat = load(campaign / "hosts" / (state["host"] + ".json"), {})
                live = any(item.get("pid") == state.get("pid") and item.get("start_ticks") == state.get("start_ticks")
                           for item in heartbeat.get("active", []))
                if not live or time.time() - heartbeat.get("updated_at", 0) > 60:
                    value = "liveness_unverified"
            counts[value] += 1
            records.append({"method": method, "task_id": task_key(task), "status": value,
                            "host": state.get("host"), "pid": state.get("pid"),
                            "attempts": info["attempts"], "successful": info["successful"],
                            "failed": info["failed"], "identity": info.get("identity"),
                            "trajectory": info.get("trajectory")})
        summary[method] = dict(counts)
        summary[method]["successful_evaluations"] = sum(r["successful"] for r in records if r["method"] == method)
    lock = claim(campaign / "report.lock")
    if lock:
        try:
            save(campaign / "task_index.json", {"updated_at": time.time(), "summary": summary, "tasks": records})
        finally:
            lock.close()
    if not quiet:
        print(json.dumps(summary, indent=2), flush=True)
    return summary


def run(campaign, host):
    verify(campaign, host)
    preflight = load(campaign / "preflight" / host / "passed.json", {})
    assert preflight.get("checksums_sha256") == digest(campaign / "checksums.json"), "Preflight required"
    spec = load(campaign / "protocol.json")["hosts"][host]
    host_lock = claim(campaign / "locks" / (host + ".lock"))
    if host_lock is None:
        raise RuntimeError("Scheduler already active on " + host)
    tasks = load(campaign / "queue.json")
    active = {}
    for path in (campaign / "state").rglob("*.json"):
        state = load(path)
        if state.get("host") != host or state.get("status") != "running":
            continue
        key = (state["method"], task_key(state))
        if process_alive(state):
            active[key] = {"state": state, "process": None, "lock": None, "path": path}
        else:
            state.update(status="incomplete", finished_at=time.time(), reason="Previous worker is no longer alive")
            save(path, state)
    started, last_report = time.monotonic(), 0
    while True:
        for key, item in list(active.items()):
            state, child = item["state"], item["process"]
            ended = child.poll() is not None if child else not process_alive(state)
            if not ended:
                continue
            info = progress(campaign, state["method"], state)
            returncode = child.returncode if child else None
            value = "incomplete"
            if returncode in (0, None) and info["budget_exhausted"] and not info.get("pending"):
                value = "complete" if info["clean_complete"] else "complete_with_failures"
            state.update(status=value, finished_at=time.time(), returncode=returncode,
                         run_identity=info.get("identity"), progress=info)
            save(item["path"], state)
            if item["lock"]:
                item["lock"].close()
            del active[key]
            print("END", *key, value, info["attempts"], flush=True)
        unavailable = set()
        for method in METHODS:
            for task in tasks:
                _, state_path, _ = task_paths(campaign, method, task)
                state = load(state_path, {})
                if state.get("status") in ("running", "complete", "incomplete", "complete_with_failures"):
                    unavailable.add((method, task_key(task)))
        occupied = {cpu for item in active.values() for cpu in item["state"]["cpus"]}
        free = [cpu for cpu in spec["cpus"] if cpu not in occupied]
        memory = int(Path("/sys/fs/cgroup/memory.current").read_text())
        candidate = None
        if not (campaign / "STOP_ADMISSION").exists():
            candidate = choose_task(spec, tasks, active, unavailable, free, memory, time.monotonic() - started)
        if candidate:
            method, task, cpus = candidate
            session, state_path, lock_path = task_paths(campaign, method, task)
            lock = claim(lock_path)
            if lock:
                existing = load(state_path, {})
                if existing.get("status") in ("running", "complete", "incomplete", "complete_with_failures"):
                    lock.close()
                else:
                    args, env, session = command(campaign, method, task, cpus)
                    log = campaign / "logs" / method / (task_key(task) + ".log")
                    log.parent.mkdir(parents=True, exist_ok=True)
                    with log.open("a") as stream:
                        child = subprocess.Popen(args, cwd=campaign / "code", env=env, stdin=subprocess.DEVNULL,
                                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                                 pass_fds=(lock.fileno(),))
                    state = {**task, "method": method, "host": host, "status": "running", "pid": child.pid,
                             "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                             "start_ticks": Path("/proc/%d/stat" % child.pid).read_text().rsplit(") ", 1)[1].split()[19],
                             "cpus": cpus, "started_at": time.time(), "session": str(session),
                             "command": args, "log": str(log)}
                    save(state_path, state)
                    active[(method, task_key(task))] = {"state": state, "process": child, "lock": lock, "path": state_path}
                    print("START", method, task_key(task), "cpus", cpus, flush=True)
        save(campaign / "hosts" / (host + ".json"), {"host": host, "pid": os.getpid(), "updated_at": time.time(),
             "active": [item["state"] for item in active.values()], "memory_bytes": memory,
             "cpu_slots_used": sum(len(item["state"]["cpus"]) for item in active.values()),
             "cpu_slots_budget": len(spec["cpus"])})
        if time.monotonic() - last_report > 30:
            status(campaign, quiet=True)
            last_report = time.monotonic()
        if not active and len(unavailable) == 132 * len(METHODS):
            status(campaign, quiet=True)
            break
        time.sleep(5)
    host_lock.close()


def start(campaign, host):
    script = campaign / "code/examples/benchmark" / Path(__file__).name
    path = campaign / "logs" / ("scheduler_" + host + ".log")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        child = subprocess.Popen([PYTHON, str(script), "run", "--campaign", str(campaign), "--host", host],
                                 cwd=campaign / "code", env=worker_env(campaign, [HOSTS[host]["cpus"][0]]),
                                 stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({"host": host, "scheduler_pid": child.pid, "log": str(path)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "smoke", "start", "run", "status"))
    parser.add_argument("--campaign", type=Path, default=DEFAULT)
    parser.add_argument("--host", choices=HOSTS)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.campaign)
    elif args.action == "status":
        status(args.campaign)
    elif not args.host:
        parser.error("--host is required")
    else:
        globals()[args.action](args.campaign, args.host)


if __name__ == "__main__":
    main()
