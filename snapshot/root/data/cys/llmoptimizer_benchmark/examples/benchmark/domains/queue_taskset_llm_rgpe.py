#!/usr/bin/env python3
"""Prepare commands by default; --execute runs an explicitly enabled CPU pool.

Run on the evaluation host, with a loopback SSH tunnel to the shared GPU host.
No inference server, tunnel, training or evaluation is started in plan mode.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from taskset_transfer_common import atomic_json, read_split


def build_plan(config):
    parent = read_split()
    split = json.loads(Path(config["split"]).read_text())
    tasks = split["test"]
    if (len(tasks) != 200 or set(split["train"]) != set(parent["train"]) or not set(tasks) <= set(parent["test"])
            or len(set(tasks)) != len(tasks) or not tasks):
        raise ValueError("Invalid frozen TaskSet held-out subset")
    workers = config["workers"]
    cpus = [cpu for worker in workers for cpu in worker["cpus"]]
    if not workers or len(set(cpus)) != len(cpus) or any(not w["cpus"] for w in workers):
        raise ValueError("Worker CPU sets must be nonempty and disjoint")
    if config["trials"] != 100:
        raise ValueError("Campaign requires 100 evaluations per task")
    result = []
    method = config.get("bo_arm", "rgpe")
    if method not in {"fsbo", "rgpe"}:
        raise ValueError("Unknown BO arm")
    for index, worker in enumerate(workers):
        commands = []
        for task in tasks[index::len(workers)]:
            command = ["taskset", "-c", ",".join(map(str, worker["cpus"])), config["python"],
                       str(Path(__file__).with_name(f"run_taskset_llm_{method}.py")), "--task", task]
            backend_keys = ("fsbo_assets", "source_assets") if method == "fsbo" else ("rgpe_campaign",)
            model_key = "base_model" if config.get("base_model") else "adapter"
            if model_key == "base_model" and (method != "fsbo" or config.get("adapter")):
                raise ValueError("Stock Qwen requires FSBO and no adapter")
            for key in ("campaign", model_key, "presets", "bank", "baseline_cache", "split",
                        "trials", "seed", "temperature", "max_new_tokens", "llm_endpoint", "llm_token_file", "llm_identity_file"):
                command += ["--" + key.replace("_", "-"), str(config[key])]
            for key in backend_keys:
                command += ["--" + key.replace("_", "-"), str(config[key])]
            commands.append({"task": task, "command": command})
        result.append({"worker": index, "cpus": worker["cpus"], "tasks": commands})
    return result


def execute(config, plan):
    if not config.get("enabled"):
        raise ValueError("Configuration is disabled; this prepared plan does not authorize a launch")
    if socket.gethostname().lower() != config["evaluation_hostname"].lower():
        raise ValueError("Run the CPU pool on its configured evaluation host")
    assigned = {cpu for w in plan for cpu in w["cpus"]}
    if not assigned <= os.sched_getaffinity(0) or shutil.which("taskset") is None:
        raise ValueError("Configured CPUs or taskset are unavailable")
    meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    if int(meminfo["MemAvailable"].split()[0]) / 1024**2 < config["min_available_memory_gib"]:
        raise ValueError("Insufficient available memory for the configured pool")
    presets = json.loads(Path(config["presets"]).read_text())
    for task in json.loads(Path(config["split"]).read_text())["test"]:
        if len(presets.get(task, [])) != 3:
            raise ValueError(f"Missing three frozen BOLT presets for {task}")
    # Connectivity and immutable model identity are checked before any trial starts.
    from mindware.domains.dlhpo.taskset_llm.inference import SharedQwenClient
    SharedQwenClient(config["llm_endpoint"], config["llm_token_file"],
                     expected_identity=json.loads(Path(config["llm_identity_file"]).read_text())["identity_sha256"])
    logs = Path(config["campaign"]) / "workers"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", TF_NUM_INTRAOP_THREADS="1",
               TF_NUM_INTEROP_THREADS="1")

    def worker(partition):
        runtime = Path(config["campaign"]) / "runtimes" / f"worker{partition['worker']}"
        if not runtime.exists():
            shutil.copytree(config["runtime_template"], runtime, symlinks=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        worker_env = dict(env, TASKSET_RUNTIME_ROOT=str(runtime), TASKSET_EVAL_DIR=config["eval_wrapper"],
                          TASKSET_ROOT=config["taskset_data"], TASKSET_EVAL_PYTHON=config["eval_python"],
                          TASKSET_BASELINE_CACHE=config["baseline_cache"], TASKSET_EVAL_THREADS="1")
        events = []
        for item in partition["tasks"]:
            with (logs / f"worker{partition['worker']}_{item['task']}.log").open("a") as stream:
                process = subprocess.run(item["command"], env=worker_env, stdout=stream, stderr=subprocess.STDOUT)
            event = {"task": item["task"], "returncode": process.returncode}
            events.append(event)
            atomic_json(logs / f"worker{partition['worker']}.json", events)
            if process.returncode:
                raise RuntimeError(f"Task {item['task']} failed; inspect worker log before restarting")
        return events

    with ThreadPoolExecutor(max_workers=len(plan)) as pool:
        for events in pool.map(worker, plan):
            print(json.dumps(events), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    plan = build_plan(config)
    if args.out:
        atomic_json(args.out, {"enabled": config.get("enabled", False), "workers": plan})
    print(json.dumps({"mode": "execute" if args.execute else "plan", "workers": len(plan),
                      "tasks": sum(len(w["tasks"]) for w in plan)}), flush=True)
    if args.execute:
        execute(config, plan)


if __name__ == "__main__":
    main()
