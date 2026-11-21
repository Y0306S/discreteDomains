#!/usr/bin/env python3
"""Redistribute an existing TaskSet campaign using exclusive physical cores."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def core_key(cpu):
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    return ((path / "physical_package_id").read_text().strip(),
            (path / "core_id").read_text().strip())


def validate(plan, host):
    machine = plan["hosts"][host]
    if socket.gethostname() != machine["hostname"]:
        raise ValueError("Wrong host")
    occupied = set()
    for allocation in machine["reservations"] + machine["workers"]:
        keys = {core_key(cpu) for cpu in allocation["cpus"]}
        if keys & occupied:
            raise ValueError(f"Physical CPU overlap: {allocation['name']}")
        occupied.update(keys)
    quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
    if quota != "max" and len(occupied) > int(quota) / int(period):
        raise ValueError("CPU reservations exceed the container quota")


def prepare(campaign):
    original = read(campaign / "campaign.json")
    revision = campaign / "distributed"
    if (revision / "plan.json").exists():
        raise ValueError("Redistribution plan already exists")
    revision.mkdir(exist_ok=True)
    workers = {host: [] for host in ("LLMOptimizer2", "LLMOptimizer3", "LLMOptimizer4")}

    def add(host, cpus, method, task=None):
        workers[host].append({"name": f"{host}.w{len(workers[host]):02d}",
                              "cpus": cpus, "preferred_method": method,
                              "train_task": task})

    add("LLMOptimizer2", [36, 37], "random")
    add("LLMOptimizer2", [38, 39], "tpe")
    for i, method in enumerate(["lolbo", "random", "tpe", "lolbo", "random", "tpe", "lolbo"]):
        add("LLMOptimizer3", [i * 2, i * 2 + 1], method)
    train = [job for job in original["jobs"] if job["side"] == "train"]
    cpu = 4
    for i, job in enumerate(sorted(train, key=lambda job: ("word_rnn" not in job["task_id"], job["task_id"]))):
        size = 4 if "word_rnn" in job["task_id"] else 2
        add("LLMOptimizer4", list(range(cpu, cpu + size)), ["random", "tpe", "lolbo"][i % 3], job["task_id"])
        cpu += size
    for method in ["random", "tpe", "lolbo"]:
        add("LLMOptimizer4", [cpu, cpu + 1], method)
        cpu += 2
    assert cpu == 48
    hosts = {
        "LLMOptimizer2": {"hostname": "cs-94bac-0693e-server", "headroom_gib": 40,
                          "reservations": [{"name": "ckpt32", "cpus": list(range(24))},
                                           {"name": "BOLT", "cpus": list(range(24, 32))},
                                           {"name": "OptFormer", "cpus": list(range(32, 36))}]},
        "LLMOptimizer3": {"hostname": "cs-0b2cd-18deb-server", "headroom_gib": 10, "reservations": []},
        "LLMOptimizer4": {"hostname": "cs-fa498-8793f-server", "headroom_gib": 32,
                          "reservations": [{"name": "FSBO", "cpus": list(range(4))}]},
    }
    for host in hosts:
        hosts[host]["workers"] = workers[host]
    write(revision / "plan.json", {"created_at": time.time(), "hosts": hosts,
          "campaign": str(campaign), "jobs": original["jobs"], "python": original["python"],
          "split_sha256": original["split_sha256"], "protocol": original["protocol"],
          "evaluations_per_task": 100, "objective_metric": original["objective_metric"],
          "policy": "Exclusive physical cores per independent worker, inherited by its subprocesses; training backfills lend their cores to test baselines after completion."})
    shutil.copy2(campaign / "source_hashes.json", revision / "original_source_hashes.json")
    write(revision / "source_hashes.json", {
        relative: hashlib.sha256((campaign / relative).read_bytes()).hexdigest()
        for relative in read(campaign / "source_hashes.json")})
    shutil.copy2(__file__, revision / "runner.py")
    print(json.dumps({host: len(workers[host]) for host in hosts}))


def load_helpers(campaign):
    sys.path.insert(0, str(campaign / "code"))
    sys.path.insert(0, str(campaign / "code/examples/benchmark/domains"))
    import run_taskset_cpu_campaign
    return run_taskset_cpu_campaign


def work(campaign, host, name):
    plan = read(campaign / "distributed/plan.json")
    validate(plan, host)
    machine = plan["hosts"][host]
    spec = next(worker for worker in machine["workers"] if worker["name"] == name)
    os.sched_setaffinity(0, spec["cpus"])
    helpers = load_helpers(campaign)
    revision = campaign / "distributed"
    state_path = revision / "workers" / f"{name}.json"
    state = {**spec, "host": host, "pid": os.getpid(), "started_at": time.time(),
             "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    worker_lock = (campaign / "locks" / f"worker.{name}.lock").open("a")
    fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    core_locks = []
    for cpu in spec["cpus"]:
        lock = (campaign / "locks" / f"core.{host}.{cpu}.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        core_locks.append(lock)
    jobs = [job for job in plan["jobs"] if job["side"] == "test" or job["task_id"] == spec["train_task"]]
    jobs.sort(key=lambda job: (job["side"] != "train", job["method"] != spec["preferred_method"]))
    try:
        for job in jobs:
            with (campaign / "locks" / f"{job['id']}.lock").open("a") as job_lock:
                try:
                    fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                before = helpers.inspect_history(job, validate=True)
                if before["n_observations"] >= 100:
                    continue
                # Serialize admission and reserve headroom for workers still importing.
                with (campaign / "locks" / f"admission.{host}.lock").open("a") as admission:
                    fcntl.flock(admission, fcntl.LOCK_EX)
                    while helpers.memory_headroom_gib() < machine["headroom_gib"]:
                        write(state_path, {**state, "status": "waiting_for_memory", "job": job, "updated_at": time.time()})
                        time.sleep(10)
                    runtime = campaign / "runtimes" / job["id"]
                    if not runtime.exists():
                        shutil.copytree(campaign / "runtime_template", runtime, symlinks=True)
                    else:
                        shutil.copy2(campaign / "runtime_template/task_set/train_inner.py", runtime / "task_set/train_inner.py")
                    threads = str(len(spec["cpus"]))
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1",
                               OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
                               MKL_NUM_THREADS=threads, NUMEXPR_NUM_THREADS=threads,
                               TASKSET_EVAL_THREADS=threads, TF_NUM_INTRAOP_THREADS=threads,
                               TF_NUM_INTEROP_THREADS="1", TASKSET_RUNTIME_ROOT=str(runtime),
                               TASKSET_EVAL_DIR=str(campaign / "eval_wrapper"),
                               TASKSET_ROOT=str(campaign / "taskset_data"),
                               TASKSET_BASELINE_CACHE=str(campaign / "task_baselines.json"),
                               TASKSET_EVAL_PYTHON="/root/data/cys/conda_envs/taskset_eval/bin/python",
                               PYTHONPATH=str(campaign / "code"))
                    command = helpers.command_for(job, campaign, plan["python"])
                    log = campaign / "job_logs" / f"{job['id']}.log"
                    with log.open("a") as handle:
                        handle.write(f"\nDISTRIBUTED RESUME {host} {name} cpus={spec['cpus']} saved={before['n_observations']}\n")
                        handle.flush()
                        process = subprocess.Popen(command, env=env, cwd=campaign / "code",
                            stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                            pass_fds=tuple(lock.fileno() for lock in [worker_lock, job_lock, *core_locks]))
                    write(state_path, {**state, "status": "running", "job": job, "child_pid": process.pid,
                                       "resumed_from": before["n_observations"], "updated_at": time.time()})
                    time.sleep(2)
                while process.poll() is None:
                    write(state_path, {**state, "status": "running", "job": job, "child_pid": process.pid,
                                       "resumed_from": before["n_observations"],
                                       **helpers.inspect_history(job), "updated_at": time.time()})
                    time.sleep(10)
                after = helpers.inspect_history(job, validate=True)
                result = {"job": job, "host": host, "worker": name, "returncode": process.returncode,
                          **after, "finished_at": time.time()}
                result["status"] = ("complete" if after["n_successful"] == 100 else "complete_with_failures")
                if process.returncode or after["n_observations"] != 100:
                    result["status"] = "incomplete"
                write(campaign / "results" / f"{job['id']}.json", result)
                if result["status"] == "incomplete":
                    raise RuntimeError(f"Job did not complete: {log}")
        write(state_path, {**state, "status": "complete", "updated_at": time.time()})
    except Exception as error:
        write(state_path, {**state, "status": "failed", "error": str(error), "updated_at": time.time()})
        raise


def launch(campaign, host, names=None):
    revision = campaign / "distributed"
    plan = read(revision / "plan.json")
    validate(plan, host)
    if hashlib.sha256((campaign / "split.json").read_bytes()).hexdigest() != plan["split_sha256"]:
        raise ValueError("Split changed")
    for relative, digest in read(revision / "source_hashes.json").items():
        if hashlib.sha256((campaign / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source changed: {relative}")
    workers = plan["hosts"][host]["workers"]
    if names is not None:
        unknown = set(names) - {spec["name"] for spec in workers}
        if unknown or len(names) != len(set(names)):
            raise ValueError(f"Unknown or duplicate worker names: {names}")
        workers = [spec for spec in workers if spec["name"] in names]
    for spec in workers:
        with (campaign / "locks" / f"worker.{spec['name']}.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"Worker is already active: {spec['name']}") from None
    launched = []
    for spec in workers:
        name = spec["name"]
        with (revision / f"{name}.log").open("a") as handle:
            process = subprocess.Popen([plan["python"], str(revision / "runner.py"), "worker",
                "--campaign", str(campaign), "--host", host, "--name", name],
                stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                         MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"), start_new_session=True)
        launched.append({**spec, "pid": process.pid})
        time.sleep(1)
    launch_path = revision / f"launch.{host}.json"
    previous = read(launch_path) if launch_path.exists() else {}
    combined = {spec["name"]: spec for spec in previous.get("workers", [])}
    combined.update({spec["name"]: spec for spec in launched})
    write(launch_path, {"launched_at": previous.get("launched_at", time.time()),
                       "updated_at": time.time(), "workers": list(combined.values()),
                       "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps(launched))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "launch", "worker"])
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--host")
    parser.add_argument("--name")
    parser.add_argument("--names", nargs="+", help="Launch only the named workers.")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.campaign)
    elif args.action == "launch":
        launch(args.campaign, args.host, args.names)
    else:
        work(args.campaign, args.host, args.name)
