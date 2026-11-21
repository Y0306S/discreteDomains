#!/usr/bin/env python3
"""Prepare and run the approved TaskSet train backfill and held-out baselines."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
SPLIT = Path(
    "/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/"
    "taskset_major_families_split_80_20_v1.json"
)
CORPUS = Path("/root/data/logs/taskset_corpus_train_1000s_100t")
PROTOCOL = {"training_steps": 1000, "eval_every_n": 1000, "num_replicas": 1}
METRIC = "min_normalized_valid_outer"
METHODS = ("random", "lolbo", "tpe")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def interleave(tasks):
    families = defaultdict(deque)
    for task in sorted(tasks):
        families[task.rsplit("_seed", 1)[0]].append(task)
    order = sorted(families, key=lambda family: (family != "quadratic_family", family))
    result = []
    while any(families.values()):
        for family in order:
            if families[family]:
                result.append(families[family].popleft())
    return result


def build_jobs(split, missing, campaign):
    train, test = set(split["train"]), set(split["test"])
    if len(train) != 838 or len(test) != 210 or train & test:
        raise ValueError("Expected disjoint 838/210 split")
    if len(split["train"]) != len(train) or len(split["test"]) != len(test):
        raise ValueError("Duplicate split IDs")
    if len(missing) != 11 or len(set(missing)) != 11 or not set(missing) <= train:
        raise ValueError("Expected the approved 11 missing training tasks")
    if (train | test) & set(split["excluded"]):
        raise ValueError("Excluded task in train/test split")
    jobs = []
    for task in interleave(missing):
        jobs.append({"id": f"train.tpe.{task}", "side": "train", "task_id": task,
                     "method": "tpe", "output_root": str(CORPUS)})
    for task in interleave(test):
        for method in METHODS:
            jobs.append({"id": f"test.{method}.{task}", "side": "test", "task_id": task,
                         "method": method, "output_root": str(campaign / "test")})
    return jobs


def command_for(job, campaign, python):
    return [
        python, str(campaign / "code/examples/benchmark/domains/run_taskset_bo.py"),
        "--optimizer", job["method"], "--task-ids", job["task_id"],
        "--seed", "1", "--n-trials", "100", "--no-dummy",
        "--training-steps", "1000", "--eval-every-n", "1000", "--num-replicas", "1",
        "--objective-metric", METRIC,
        "--n-startup-trials", "10" if job["method"] == "lolbo" else "1",
        "--lolbo-batch-size", "10", "--output-dir", job["output_root"],
    ]


def history_path(job):
    return Path(job["output_root"]) / job["task_id"] / job["method"] / "seed1/history.json"


def inspect_history(job, *, validate=False):
    path = history_path(job)
    from mindware.domains.history import history_exists, load_openbox_history

    if not history_exists(path):
        return {"n_observations": 0, "n_successful": 0, "n_gp_proposals": 0}

    history = load_openbox_history(path)
    meta = history.get("meta_info", {})
    if (history.get("task_id") != job["task_id"] or meta.get("optimizer") != job["method"]
            or meta.get("seed") != 1 or meta.get("domain") != "taskset"):
        raise ValueError(f"History identity mismatch: {path}")
    observations = history["observations"]
    if len(observations) > 100:
        raise ValueError(f"Evaluation budget exceeded: {path}")
    if validate:
        from ConfigSpace import Configuration
        from mindware.domains.dlhpo.taskset_space import space_from_taskset

        space = space_from_taskset()
        for obs in observations:
            space.check_configuration(Configuration(space, values=obs["config"]))
            info = obs["extra_info"]["info"]
            if (info.get("oracle") != "train_inner" or info.get("protocol") != PROTOCOL
                    or info.get("objective_metric") != METRIC):
                raise ValueError(f"History evaluation protocol mismatch: {path}")
            if job["method"] == "lolbo" and info.get("optimizer_backend") != "lolbo_no_vae_cash":
                raise ValueError(f"Wrong LOLBO implementation: {path}")
    return {
        "n_observations": len(observations),
        "n_successful": sum(obs["trial_state"] == 0 and math.isfinite(obs["objectives"][0])
                            for obs in observations),
        "n_gp_proposals": sum(obs.get("extra_info", {}).get("info", {}).get("proposal_phase")
                              == "gp_trust_region" for obs in observations),
    }


def prepare(campaign):
    if campaign.exists():
        raise ValueError("Use a fresh campaign directory")
    split = read_json(SPLIT)
    audit = read_json(SPLIT.with_suffix(".audit.json"))
    jobs = build_jobs(split, audit["missing_train_task_ids"], campaign)
    for job in jobs[:11]:
        if history_path(job).exists():
            raise ValueError(f"Missing-history list changed: {history_path(job)}")

    from mindware.domains.dlhpo.taskset_oracle import _load_wrapper, load_task_baseline
    from mindware.domains.paths import TASKSET_BASELINE_CACHE, TASKSET_EVAL_DIR, TASKSET_ROOT

    for task in {job["task_id"] for job in jobs}:
        load_task_baseline(task)
    source_root = _load_wrapper()["mod"].TASK_SET_ROOT
    campaign.mkdir(parents=True)
    shutil.copy2(SPLIT, campaign / "split.json")
    shutil.copy2(TASKSET_BASELINE_CACHE, campaign / "task_baselines.json")
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(ROOT / "mindware", campaign / "code/mindware", ignore=ignored)
    scripts = campaign / "code/examples/benchmark/domains"
    scripts.mkdir(parents=True)
    for filename in ("run_taskset_bo.py", Path(__file__).name):
        shutil.copy2(Path(__file__).with_name(filename), scripts / filename)
    shutil.copytree(TASKSET_EVAL_DIR, campaign / "eval_wrapper", ignore=ignored)
    data = campaign / "taskset_data"
    (data / "scripts").mkdir(parents=True)
    shutil.copy2(TASKSET_ROOT / "best_optimizer_per_task.json", data)
    shutil.copy2(TASKSET_ROOT / "scripts/optimizer_search_spaces.json", data / "scripts")
    template = campaign / "runtime_template"
    shutil.copytree(source_root / "task_set", template / "task_set",
                    ignore=shutil.ignore_patterns("configs", "__pycache__"))
    shutil.copytree(source_root / "task_set/tasks/configs", campaign / "task_definitions")
    (template / "task_set/tasks/configs").symlink_to(campaign / "task_definitions")
    (template / "task_set/optimizers/configs").mkdir()
    hashes = {}
    for folder in ("code", "eval_wrapper", "runtime_template", "task_definitions", "taskset_data"):
        for path in (campaign / folder).rglob("*"):
            if path.is_file() and not path.is_symlink():
                hashes[str(path.relative_to(campaign))] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(campaign / "source_hashes.json", hashes)
    manifest = {
        "created_at": now(), "python": sys.executable, "split_source": str(SPLIT),
        "split_sha256": hashlib.sha256(SPLIT.read_bytes()).hexdigest(),
        "source_repo": str(ROOT), "taskset_source_root": str(source_root),
        "protocol": PROTOCOL, "objective_metric": METRIC, "seed": 1,
        "evaluations_per_task": 100, "total_evaluations": len(jobs) * 100,
        "gpu_enabled": False, "cpu_pool": list(range(32, 40)),
        "storage": {"format": "taskset-run-tar-v1", "cleanup": True},
        "cpu_quota_cores_at_prepare": 40, "minimum_memory_headroom_gib": 20,
        "workers": [
            {"name": "backfill0", "side": "train", "method": "tpe", "cpus": list(range(32, 36)), "nice": 10},
            {"name": "backfill1", "side": "train", "method": "tpe", "cpus": list(range(36, 40)), "nice": 10},
            {"name": "random", "side": "test", "method": "random", "cpus": list(range(32, 36)), "nice": 15},
            {"name": "tpe", "side": "test", "method": "tpe", "cpus": list(range(36, 40)), "nice": 15},
            {"name": "lolbo", "side": "test", "method": "lolbo", "cpus": list(range(32, 40)), "nice": 15},
        ],
        "jobs": jobs,
    }
    write_json(campaign / "campaign.json", manifest)
    print(json.dumps({"campaign": str(campaign), "train_jobs": 11, "test_jobs": 630,
                      "total_evaluations": 64100, "snapshot_files": len(hashes)}, indent=2))


def memory_headroom_gib():
    cgroup = Path("/sys/fs/cgroup")
    limit = (cgroup / "memory.max").read_text().strip()
    if limit == "max":
        return float("inf")
    current = int((cgroup / "memory.current").read_text())
    stat = dict(line.split() for line in (cgroup / "memory.stat").read_text().splitlines())
    # Inactive file cache can be reclaimed; anonymous and kernel memory cannot.
    return (int(limit) - current + int(stat.get("inactive_file", 0))) / 2**30


def worker(campaign, name):
    manifest = read_json(campaign / "campaign.json")
    from mindware.domains.taskset_archive import pack_job

    storage = manifest.get("storage", {})
    spec = next(item for item in manifest["workers"] if item["name"] == name)
    os.sched_setaffinity(0, spec["cpus"])
    os.nice(spec["nice"])
    locks = campaign / "locks"
    locks.mkdir(exist_ok=True)
    state_path = campaign / "workers" / f"{name}.json"
    with (locks / f"worker.{name}.lock").open("a") as worker_lock:
        try:
            fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        state = {"pid": os.getpid(), "started_at": now(), **spec}
        try:
            for job in manifest["jobs"]:
                if job["side"] != spec["side"] or job["method"] != spec["method"]:
                    continue
                with (locks / f"{job['id']}.lock").open("a") as job_lock:
                    try:
                        fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    before = inspect_history(job, validate=True)
                    if before["n_observations"] == 100:
                        continue
                    while memory_headroom_gib() < manifest["minimum_memory_headroom_gib"]:
                        write_json(state_path, {**state, "status": "waiting_for_memory",
                                              "job": job, "updated_at": now()})
                        time.sleep(15)
                    runtime = campaign / "runtimes" / job["id"]
                    if not runtime.exists():
                        shutil.copytree(campaign / "runtime_template", runtime, symlinks=True)
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4",
                               OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4",
                               TASKSET_RUNTIME_ROOT=str(runtime), TASKSET_EVAL_DIR=str(campaign / "eval_wrapper"),
                               TASKSET_ROOT=str(campaign / "taskset_data"),
                               TASKSET_BASELINE_CACHE=str(campaign / "task_baselines.json"),
                               TASKSET_EVAL_PYTHON="/root/data/cys/conda_envs/taskset_eval/bin/python",
                               PYTHONPATH=str(campaign / "code"), PYTHONUNBUFFERED="1")
                    command = command_for(job, campaign, manifest["python"])
                    log = campaign / "job_logs" / f"{job['id']}.log"
                    log.parent.mkdir(exist_ok=True)
                    with log.open("a") as handle:
                        process = subprocess.Popen(command, env=env, cwd=campaign / "code",
                                                   stdout=handle, stderr=subprocess.STDOUT)
                        write_json(state_path, {**state, "status": "running", "job": job,
                                              "child_pid": process.pid, "command": command,
                                              "log": str(log), "updated_at": now()})
                        returncode = process.wait()
                    after = inspect_history(job, validate=True)
                    result = {"job": job, "returncode": returncode, **after, "finished_at": now()}
                    result["status"] = ("complete" if after["n_successful"] == 100 else "complete_with_failures")
                    if returncode != 0 or after["n_observations"] != 100:
                        result["status"] = "incomplete"
                    write_json(campaign / "results" / f"{job['id']}.json", result)
                    if returncode != 0 or after["n_observations"] != 100:
                        raise RuntimeError(f"Job did not complete; inspect {log}")
                    if storage.get("format") == "taskset-run-tar-v1":
                        pack_job(campaign, job, cleanup=storage.get("cleanup", False), lock=job_lock)
            write_json(state_path, {**state, "status": "complete", "updated_at": now()})
        except Exception as exc:
            write_json(state_path, {**state, "status": "failed", "error": str(exc), "updated_at": now()})
            raise


def launch(campaign):
    manifest = read_json(campaign / "campaign.json")
    if hashlib.sha256((campaign / "split.json").read_bytes()).hexdigest() != manifest["split_sha256"]:
        raise ValueError("Campaign split has changed")
    for relative, digest in read_json(campaign / "source_hashes.json").items():
        if hashlib.sha256((campaign / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Snapshot has changed: {relative}")
    if (campaign / "launch.json").exists():
        raise ValueError("Campaign already launched; inspect worker state before resuming")
    launches = []
    for spec in manifest["workers"]:
        log = campaign / f"worker_{spec['name']}.log"
        with log.open("a") as handle:
            process = subprocess.Popen(
                [manifest["python"], str(campaign / "code/examples/benchmark/domains" / Path(__file__).name),
                 "worker", "--campaign", str(campaign), "--name", spec["name"]],
                env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4",
                         OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4"),
                cwd=campaign / "code", stdin=subprocess.DEVNULL, stdout=handle,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
        launches.append({"worker": spec["name"], "pid": process.pid, "log": str(log)})
    write_json(campaign / "launch.json", {"launched_at": now(), "workers": launches})
    print(json.dumps(launches, indent=2))


def status(campaign):
    manifest = read_json(campaign / "campaign.json")
    totals = defaultdict(lambda: {"tasks": 0, "complete": 0, "evaluations": 0, "successful": 0})
    for job in manifest["jobs"]:
        row = inspect_history(job)
        group = totals[f"{job['side']}.{job['method']}"]
        group["tasks"] += 1
        group["complete"] += row["n_observations"] == 100
        group["evaluations"] += row["n_observations"]
        group["successful"] += row["n_successful"]
    worker_dir = campaign / "distributed/workers" if (campaign / "distributed/plan.json").exists() else campaign / "workers"
    workers = [read_json(path) for path in sorted(worker_dir.glob("*.json"))]
    print(json.dumps({"campaign": str(campaign), "totals": totals, "workers": workers,
                      "memory_headroom_gib": round(memory_headroom_gib(), 2)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "worker", "status"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--name")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if args.action == "worker":
        worker(campaign, args.name)
    else:
        {"prepare": prepare, "launch": launch, "status": status}[args.action](campaign)


if __name__ == "__main__":
    main()
