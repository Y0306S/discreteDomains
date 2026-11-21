#!/usr/bin/env python3
"""Shared, lock-protected test200 queue for multiple CPU evaluation hosts.

GPU endpoints are explicit replicas with one immutable model identity. Each
worker owns its runtime and task state; only dispatch claims are shared.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import threading
import time

from queue_taskset_llm_rgpe import build_plan
from taskset_transfer_common import atomic_json


@contextmanager
def claim_task(directory, task):
    """Hold a kernel lock through the subprocess; process death releases it."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{task}.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield not ((directory / f"{task}.done.json").exists()
                       or (directory / f"{task}.failed.json").exists())
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def resource_limits(config):
    cg = Path("/sys/fs/cgroup")
    quota, period = (cg / "cpu.max").read_text().split()
    cores = len(os.sched_getaffinity(0)) if quota == "max" else int(quota) / int(period)
    reserved = float(config["reserved_cpu_cores"])
    demand = sum(len(w["cpus"]) for w in config["workers"])
    if reserved < 0 or demand + reserved > cores:
        raise ValueError(f"CPU pool {demand} + reserve {reserved} exceeds quota {cores}")
    info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(info["MemAvailable"].split()[0]) * 1024
    maximum = (cg / "memory.max").read_text().strip()
    if maximum != "max":
        available = min(available, int(maximum) - int((cg / "memory.current").read_text()))
    required = float(config["min_available_memory_gib"]) + len(config["workers"]) * float(config["memory_per_worker_gib"])
    if available / 2**30 < required:
        raise ValueError(f"Memory headroom {available / 2**30:.1f} GiB is below pool budget {required:.1f}")
    return {"cpu_quota": cores, "cpu_demand": demand, "reserved_cores": reserved,
            "available_memory_gib": available / 2**30, "required_memory_gib": required}


def execute(config):
    # Legacy planner enforces the frozen split, budget and disjoint CPU sets.
    partitions = build_plan(config)
    items = {item["task"]: item for partition in partitions for item in partition["tasks"]}
    tasks = json.loads(Path(config["split"]).read_text())["test"]
    if not config.get("enabled") or config.get("bo_arm") != "fsbo":
        raise ValueError("An explicitly enabled FSBO configuration is required")
    if socket.gethostname().lower() != config["evaluation_hostname"].lower():
        raise ValueError("Wrong evaluation host")
    if not {cpu for w in config["workers"] for cpu in w["cpus"]} <= os.sched_getaffinity(0):
        raise ValueError("Configured CPU affinity is unavailable")
    limits = resource_limits(config)
    presets = json.loads(Path(config["presets"]).read_text())
    if set(presets) != set(tasks) or any(len(presets[t]) != 3 for t in tasks):
        raise ValueError("Exactly 600 frozen BOLT presets are required")
    from mindware.domains.dlhpo.taskset_llm.inference import SharedQwenClient
    expected = json.loads(Path(config["llm_identity_file"]).read_text())["identity_sha256"]
    endpoints = config["llm_endpoints"]
    if not endpoints or len(set(endpoints)) != len(endpoints):
        raise ValueError("Distinct inference replicas are required")
    for endpoint in endpoints:
        SharedQwenClient(endpoint, config["llm_token_file"], expected_identity=expected)
    campaign = Path(config["campaign"])
    claims, logs = campaign / "dispatch", campaign / "workers"
    claims.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    pool_id = config["pool_id"]
    if not pool_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in pool_id):
        raise ValueError("Invalid pool identifier")
    pool_lock = (claims / f"pool-{pool_id}.lock").open("a")
    fcntl.flock(pool_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", TF_NUM_INTRAOP_THREADS="1",
               TF_NUM_INTEROP_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    status_path = logs / f"{pool_id}.pool.json"
    atomic_json(status_path, {"state": "running", "pid": os.getpid(), "host": socket.gethostname(),
                             "workers": len(partitions), "resources": limits, "started": time.time()})

    def worker(index):
        name = f"{pool_id}-{index:02d}"
        status = logs / f"{name}.json"
        cpus = config["workers"][index]["cpus"]
        endpoint = endpoints[(int(config.get("worker_offset", 0)) + index) % len(endpoints)]
        atomic_json(status, {"state": "starting", "worker": name, "cpus": cpus, "endpoint": endpoint})
        threading.Event().wait(index * float(config.get("launch_stagger_seconds", 0.5)))
        runtime = campaign / "runtimes" / name
        if not runtime.exists():
            shutil.copytree(config["runtime_template"], runtime, symlinks=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        worker_env = dict(env, TASKSET_RUNTIME_ROOT=str(runtime), TASKSET_EVAL_DIR=config["eval_wrapper"],
                          TASKSET_ROOT=config["taskset_data"], TASKSET_EVAL_PYTHON=config["eval_python"],
                          TASKSET_BASELINE_CACHE=config["baseline_cache"], TASKSET_EVAL_THREADS="1")
        offset = (int(config.get("worker_offset", 0)) + index) % len(tasks)
        events = []
        for task in tasks[offset:] + tasks[:offset]:
            with claim_task(claims, task) as acquired:
                if not acquired:
                    continue
                command = list(items[task]["command"])
                command[2] = ",".join(map(str, cpus))
                command[command.index("--llm-endpoint") + 1] = endpoint
                with (logs / f"{name}_{task}.log").open("a") as stream:
                    process = subprocess.Popen(command, env=worker_env, stdin=subprocess.DEVNULL,
                                               stdout=stream, stderr=subprocess.STDOUT)
                    atomic_json(status, {"state": "running", "worker": name, "task": task,
                                         "pid": process.pid, "cpus": cpus, "endpoint": endpoint,
                                         "started": time.time(), "events": events})
                    returncode = process.wait()
                history = campaign / "test" / task / "llm_fsbo" / f"seed{config['seed']}" / "history.json"
                try:
                    evaluated = len(json.loads(history.read_text())["observations"])
                except (OSError, ValueError, KeyError):
                    evaluated = 0
                event = {"task": task, "returncode": returncode, "evaluated": evaluated,
                         "worker": name, "finished": time.time()}
                success = returncode == 0 and evaluated == config["trials"]
                atomic_json(claims / f"{task}.{'done' if success else 'failed'}.json", event)
                events.append(event)
                print(json.dumps(event), flush=True)
        atomic_json(status, {"state": "finished", "worker": name, "cpus": cpus,
                             "endpoint": endpoint, "events": events})
        return events

    try:
        with ThreadPoolExecutor(max_workers=len(partitions)) as pool:
            for future in as_completed([pool.submit(worker, i) for i in range(len(partitions))]):
                future.result()
        failed = len(list(claims.glob("*.failed.json")))
        atomic_json(status_path, {"state": "failed" if failed else "finished", "failed_tasks": failed,
                                 "workers": len(partitions), "finished": time.time(), "resources": limits})
        if failed:
            raise RuntimeError(f"{failed} tasks failed; inspect dispatch markers and logs before retrying")
    finally:
        pool_lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    plan = build_plan(config)
    print(json.dumps({"mode": "execute" if args.execute else "plan", "workers": len(plan),
                      "shared_tasks": sum(len(p["tasks"]) for p in plan), "pool": config["pool_id"]}), flush=True)
    if args.execute:
        execute(config)


if __name__ == "__main__":
    main()
