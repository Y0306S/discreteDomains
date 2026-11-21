#!/usr/bin/env python3
"""v14 offline RL: frozen-code rollout collection on pinned workers, then training.

  prepare   freeze code + inputs into RUN (refuses to overwrite)
  pipeline  run the collection workers until all tasks finish, then train
  status    per-task progress
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/root/blockdata/llmoptimizer_benchmark")
RUN = REPO / "benchmark_data/rl_rollouts/v14_orpo_rationale_20261005"
PYTHON = "/root/blockdata/miniconda3/envs/openbox_qwen_eval/bin/python"
ADAPTER = Path("/root/blockdata/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_orpo_t2_rationale_20261005")
ADAPTER_SHA256 = "3f1eeab7d55d56dace3b268ae7eb59d4bebccfc500bd6b907ccee7051601701f"
BASE_MODEL = "/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B"
SOURCE_GP = REPO / "benchmark_data/ml_test_20_fixed_ckpt32_rgpe_20261002/source_gp_train80_v1"
BANK = REPO / "history_data/experience_bank_v3_train80.jsonl"
ALLOWLIST = Path("/root/reports/rl_plan_20261005/train32_tasks.json")
TRAIN_MANIFEST = Path("/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/ml_train_80_fixed.jsonl")
TEST_MANIFEST = Path("/root/data/cys/llmoptimizer_benchmark/benchmark_data/ml_test_20_fixed_48h_20260925/tasks_132.json")
# Agreed 2026-10-05: LLMOptimizer2 GPUs 0-1, three collectors with five pinned cores each.
WORKERS = [
    {"name": "w0", "gpu": "0", "cpus": "0-4"},
    {"name": "w1", "gpu": "0", "cpus": "5,6,7,16,17"},
    {"name": "w2", "gpu": "1", "cpus": "18,19,20,21,23"},
]
SETTINGS = {
    "collect_schedule": "rs_mix_v14", "n_samples": 4, "n_samples_max": 10, "llm_temperature": 1.0,
    "per_run_time_limit": 2400, "penalty_parse_fail": 1.0, "penalty_duplicate": 0.5, "seed": 1,
    "eval_n_jobs": 5,
}
TRAIN = {"gpu": "0,1", "cpus": "0-4", "lr": 1e-6, "kl_beta": 0.05, "warmup_steps": 10, "grad_accum": 8}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def task_key(t) -> str:
    return f"{t['task_type']}_{t['dataset']}_{t['algorithm']}"


def prepare(_args) -> None:
    if (RUN / "inputs.json").exists():
        raise SystemExit(f"{RUN} is already prepared; refusing to overwrite a frozen run")
    if sha256(ADAPTER / "adapter_model.safetensors") != ADAPTER_SHA256:
        raise SystemExit("ORPO adapter hash mismatch")
    gp_manifest = json.loads((SOURCE_GP / "manifest.json").read_text())
    if sha256(BANK) != gp_manifest["bank_sha256"]:
        raise SystemExit("train80 bank does not match the source GP artifact")
    code = RUN / "code"
    code.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "catboost_info")
    for name in ("examples", "mindware", "configs"):
        shutil.copytree(REPO / name, code / name, ignore=ignore)
    (code / "history_data").mkdir()
    shutil.copy2(BANK, code / "history_data" / BANK.name)
    (code / "datasets_bundle").symlink_to((REPO / "datasets_bundle").resolve(), target_is_directory=True)
    files = {str(p.relative_to(code)): sha256(p) for p in sorted(code.rglob("*"))
             if p.is_file() and "datasets_bundle" not in p.relative_to(code).parts}
    write_json(RUN / "code_sha256.json", files)
    snapshot = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    shutil.copy2(ALLOWLIST, RUN / "tasks.json")
    for sub in ("buffer", "claims", "logs", "history"):
        (RUN / sub).mkdir()
    write_json(RUN / "inputs.json", {
        "prepared_at": now(), "code_snapshot_sha256": snapshot, "code_files": len(files),
        "adapter": str(ADAPTER), "adapter_sha256": ADAPTER_SHA256, "base_model": BASE_MODEL,
        "source_gp": str(SOURCE_GP), "source_gp_manifest_sha256": sha256(SOURCE_GP / "manifest.json"),
        "bank_sha256": sha256(BANK), "tasks_sha256": sha256(RUN / "tasks.json"),
        "train_manifest": str(TRAIN_MANIFEST), "train_manifest_sha256": sha256(TRAIN_MANIFEST),
        "test_manifest": str(TEST_MANIFEST), "test_manifest_sha256": sha256(TEST_MANIFEST),
        "workers": WORKERS, "settings": SETTINGS, "train": TRAIN,
    })
    print(json.dumps({"prepared": str(RUN), "code_snapshot_sha256": snapshot, "files": len(files)}))


def collector_cmd(task, worker, inputs) -> list[str]:
    key = task_key(task)
    s = inputs["settings"]
    return [
        "taskset", "-c", worker["cpus"], PYTHON, str(RUN / "code/examples/benchmark/run_p5_rl_bo_prefix_b.py"),
        "--task-type", task["task_type"], "--dataset", task["dataset"], "--algorithm", task["algorithm"],
        "--collect-schedule", s["collect_schedule"], "--output-dir", str(RUN / "buffer"),
        "--buffer-name", f"rollouts_{key}.jsonl", "--run-id", f"v14_{key}",
        "--history-dir", str(RUN / "history"),
        "--qwen-model-path", inputs["base_model"], "--qwen-adapter-path", inputs["adapter"],
        "--source-gp-cache", inputs["source_gp"], "--task-allowlist", str(RUN / "tasks.json"),
        "--train-manifest", inputs["train_manifest"], "--test-manifest", inputs["test_manifest"],
        "--n-samples", str(s["n_samples"]), "--n-samples-max", str(s["n_samples_max"]),
        "--llm-temperature", str(s["llm_temperature"]), "--per-run-time-limit", str(s["per_run_time_limit"]),
        "--penalty-parse-fail", str(s["penalty_parse_fail"]), "--penalty-duplicate", str(s["penalty_duplicate"]),
        "--seed", str(s["seed"]),
    ]


def worker_env(inputs, gpu: str, threads: int) -> dict:
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": gpu, "EVAL_N_JOBS": str(threads), "OMP_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads), "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1",
        "RL_CODE_SNAPSHOT_SHA256": inputs["code_snapshot_sha256"],
    })
    return env


def verify_frozen_code(inputs) -> None:
    files = json.loads((RUN / "code_sha256.json").read_text())
    changed = [p for p, h in files.items() if sha256(RUN / "code" / p) != h]
    if changed:
        raise SystemExit(f"Frozen code changed: {changed[:5]}")


def claim(task, worker_name) -> bool:
    path = RUN / "claims" / f"{task_key(task)}.json"
    if path.exists():
        state = json.loads(path.read_text())
        # A task left "running" by this same worker was interrupted; the collector resumes per group.
        if not (state["status"] == "running" and state["worker"] == worker_name):
            return False
    else:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            return False
    write_json(path, {"task": task, "worker": worker_name, "status": "running", "pid": os.getpid(), "started_at": now()})
    return True


def worker(args) -> None:
    inputs = json.loads((RUN / "inputs.json").read_text())
    w = next(x for x in inputs["workers"] if x["name"] == args.name)
    tasks = json.loads((RUN / "tasks.json").read_text())
    for task in tasks:
        if not claim(task, w["name"]):
            continue
        verify_frozen_code(inputs)
        key = task_key(task)
        t0 = time.time()
        with (RUN / "logs" / f"{key}.log").open("a") as log:
            log.write(f"\n=== {now()} worker={w['name']} gpu={w['gpu']} cpus={w['cpus']}\n")
            log.flush()
            rc = subprocess.call(collector_cmd(task, w, inputs), cwd=RUN / "code", stdout=log,
                                 stderr=subprocess.STDOUT, env=worker_env(inputs, w["gpu"], inputs["settings"]["eval_n_jobs"]))
        write_json(RUN / "claims" / f"{key}.json", {
            "task": task, "worker": w["name"], "status": "complete" if rc == 0 else "failed",
            "returncode": rc, "seconds": time.time() - t0, "finished_at": now()})


def all_complete() -> tuple[bool, dict]:
    tasks = json.loads((RUN / "tasks.json").read_text())
    states = {}
    for t in tasks:
        path = RUN / "claims" / f"{task_key(t)}.json"
        states[task_key(t)] = json.loads(path.read_text())["status"] if path.exists() else "pending"
    return all(v == "complete" for v in states.values()), states


def train() -> int:
    inputs = json.loads((RUN / "inputs.json").read_text())
    verify_frozen_code(inputs)
    out = RUN / "train" / "adapter"
    cmd = [
        "taskset", "-c", TRAIN["cpus"], PYTHON, str(RUN / "code/examples/benchmark/train_p5_reinforce.py"),
        "--buffer-root", str(RUN / "buffer"), "--init-adapter", inputs["adapter"], "--output-dir", str(out),
        "--base-model", inputs["base_model"], "--train-manifest", inputs["train_manifest"],
        "--test-manifest", inputs["test_manifest"], "--task-allowlist", str(RUN / "tasks.json"),
        "--rollout-schema", "v14", "--advantage-mode", "grpo_group", "--kl-mode", "full_vocab",
        "--lr", str(TRAIN["lr"]), "--kl-beta", str(TRAIN["kl_beta"]), "--warmup-steps", str(TRAIN["warmup_steps"]),
        "--grad-accum", str(TRAIN["grad_accum"]), "--seed", "1", "--device", "cuda:0", "--ref-device", "cuda:1",
        "--response-format", "rationale_json",
    ]
    (RUN / "train").mkdir(exist_ok=True)
    with (RUN / "train" / "train.log").open("a") as log:
        log.write(f"\n=== {now()} {' '.join(cmd)}\n")
        log.flush()
        return subprocess.call(cmd, cwd=RUN / "code", stdout=log, stderr=subprocess.STDOUT,
                               env=worker_env(inputs, TRAIN["gpu"], 5))


def pipeline(_args) -> None:
    inputs = json.loads((RUN / "inputs.json").read_text())
    verify_frozen_code(inputs)
    procs = []
    for w in inputs["workers"]:
        log = (RUN / "logs" / f"worker_{w['name']}.log").open("a")
        procs.append(subprocess.Popen([PYTHON, __file__, "worker", "--name", w["name"]], stdout=log,
                                      stderr=subprocess.STDOUT, cwd=RUN))
    write_json(RUN / "pipeline.json", {"started_at": now(), "pid": os.getpid(),
                                       "workers": {w["name"]: p.pid for w, p in zip(inputs["workers"], procs)}})
    for p in procs:
        p.wait()
    done, states = all_complete()
    status = {"collection_finished_at": now(), "all_complete": done, "states": states}
    if done:
        status["train_returncode"] = train()
        status["train_finished_at"] = now()
    write_json(RUN / "pipeline_result.json", status)


# Algorithm-based scheduler (2026-10-05): sklearn AdaBoost / GradientBoosting fit on
# one core, so they get 1; tree ensembles and boosting libraries get 3.
# 40-49 are free physical cores; 64-127 (hyperthread twins of queryopt/postgres cores) are never used.
CORE_POOL = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 23] + list(range(40, 50))
# GPU 2 is shared with OptFormer training, GPU 3 with queryopt LOLBO decoding (approved 2026-10-05).
GPU_SLOTS = {"0": 3, "1": 3, "2": 2, "3": 3}
# Container-wide CPU quota throttling would slow postgres and distort queryopt runtimes:
# launch nothing while the throttle counter is moving.
THROTTLE_QUIET_CHECKS = 3
SINGLE_CORE_ALGORITHMS = {"adaboost", "gradient_boosting"}
COLLECTOR_GPU_MIB = 21000
LOAD_GRACE_SECONDS = 300
MAX_ATTEMPTS = 2
N_GROUPS = 13


def cores_for(task) -> int:
    return 1 if task["algorithm"] in SINGLE_CORE_ALGORITHMS else 3


def bo_cost(task) -> float:
    metric = "bal_acc" if task["task_type"] == "cls" else "mse"
    path = REPO / "history_data/history_data" / task["task_type"] / metric / task["dataset"] / f"{task['algorithm']}.json"
    obs = [o for o in json.loads(path.read_text())["observations"] if "objectives" in o][:100]
    return sum(float(o.get("elapsed_time") or 0) for o in obs)


def task_finished(key: str) -> bool:
    buffer = RUN / "buffer" / f"rollouts_{key}.jsonl"
    if not buffer.exists() or not buffer.with_suffix(".manifest.json").exists():
        return False
    groups = {(r["prefix_k"], r["scenario"]) for r in map(json.loads, buffer.read_text().splitlines()) if r}
    return len(groups) == N_GROUPS


def gpu_free_mib(gpu: str) -> int:
    out = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=memory.total,memory.used",
                                   "--format=csv,noheader,nounits"], text=True)
    total, used = (int(x) for x in out.strip().split(","))
    return total - used


def running_collectors() -> dict:
    """Live v14 collectors, found by command line (lets a restarted scheduler adopt them)."""
    found = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
            if (proc / "stat").read_text().split()[2] == "Z":
                continue
        except OSError:
            continue
        argv = [a.decode() for a in argv if a]
        if not any(a.endswith("run_p5_rl_bo_prefix_b.py") for a in argv) or "--run-id" not in argv:
            continue
        run_id = argv[argv.index("--run-id") + 1]
        if not run_id.startswith("v14_"):
            continue
        env = dict(line.split("=", 1) for line in (proc / "environ").read_bytes().decode(errors="replace").split("\0") if "=" in line)
        found[run_id[len("v14_"):]] = {"pid": int(proc.name), "cpus": sorted(os.sched_getaffinity(int(proc.name))),
                                       "gpu": env.get("CUDA_VISIBLE_DEVICES")}
    return found


def cpu_throttle_events() -> int:
    for line in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines():
        if line.startswith("nr_throttled"):
            return int(line.split()[1])
    return 0


def schedule(_args) -> None:
    inputs = json.loads((RUN / "inputs.json").read_text())
    verify_frozen_code(inputs)
    tasks = sorted(json.loads((RUN / "tasks.json").read_text()), key=bo_cost, reverse=True)
    write_json(RUN / "scheduler.json", {
        "started_at": now(), "pid": os.getpid(), "script": __file__, "script_sha256": sha256(Path(__file__)),
        "core_pool": CORE_POOL, "gpu_slots": GPU_SLOTS, "throttle_quiet_checks": THROTTLE_QUIET_CHECKS, "single_core_algorithms": sorted(SINGLE_CORE_ALGORITHMS),
        "order": [task_key(t) for t in tasks]})
    running: dict = {}
    for key, info in running_collectors().items():
        running[key] = dict(info, proc=None, adopted=True, launched=0.0)
        print(f"[sched] adopted {key} pid={info['pid']} cpus={info['cpus']} gpu={info['gpu']}", flush=True)

    def claim_state(key):
        path = RUN / "claims" / f"{key}.json"
        return json.loads(path.read_text()) if path.exists() else {}

    last_throttle, quiet = cpu_throttle_events(), THROTTLE_QUIET_CHECKS
    while True:
        throttle = cpu_throttle_events()
        if throttle > last_throttle:
            quiet = 0
            print(f"[sched] {now()} CPU quota throttling (+{throttle - last_throttle} events): launches paused", flush=True)
        else:
            quiet += 1
        last_throttle = throttle
        for key in list(running):
            r = running[key]
            alive = r["proc"].poll() is None if r["proc"] else Path(f"/proc/{r['pid']}").exists()
            if alive:
                continue
            done = task_finished(key)
            state = claim_state(key)
            state.update(status="complete" if done else "failed", finished_at=now(),
                         returncode=r["proc"].returncode if r["proc"] else None)
            write_json(RUN / "claims" / f"{key}.json", state)
            print(f"[sched] {key} -> {state['status']}", flush=True)
            del running[key]
        pending = [t for t in tasks if task_key(t) not in running
                   and claim_state(task_key(t)).get("status") != "complete"
                   and int(claim_state(task_key(t)).get("attempts", 0)) < MAX_ATTEMPTS]
        if not pending and not running:
            break
        busy = {c for r in running.values() for c in r["cpus"]}
        free_cores = [c for c in CORE_POOL if c not in busy]
        for task in pending if quiet >= THROTTLE_QUIET_CHECKS else []:
            need = cores_for(task)
            if len(free_cores) < need:
                continue
            gpu = None
            for g, cap in GPU_SLOTS.items():
                on_gpu = [r for r in running.values() if r["gpu"] == g]
                loading = sum(1 for r in on_gpu if time.time() - r["launched"] < LOAD_GRACE_SECONDS)
                if len(on_gpu) < cap and gpu_free_mib(g) - loading * COLLECTOR_GPU_MIB >= COLLECTOR_GPU_MIB:
                    gpu = g
                    break
            if gpu is None:
                break
            cpus, free_cores = free_cores[:need], free_cores[need:]
            key = task_key(task)
            attempts = int(claim_state(key).get("attempts", 0)) + 1
            log = (RUN / "logs" / f"{key}.log").open("a")
            log.write(f"\n=== {now()} scheduler attempt={attempts} gpu={gpu} cpus={cpus}\n")
            log.flush()
            proc = subprocess.Popen(collector_cmd(task, {"cpus": ",".join(map(str, cpus))}, inputs),
                                    cwd=RUN / "code", stdout=log, stderr=subprocess.STDOUT,
                                    env=worker_env(inputs, gpu, need))
            running[key] = {"pid": proc.pid, "proc": proc, "cpus": cpus, "gpu": gpu, "adopted": False,
                            "launched": time.time()}
            write_json(RUN / "claims" / f"{key}.json", {
                "task": task, "status": "running", "host": "LLMOptimizer2", "pid": proc.pid, "gpu": gpu,
                "cpus": cpus, "cores": need, "attempts": attempts, "started_at": now()})
            print(f"[sched] launched {key} pid={proc.pid} gpu={gpu} cpus={cpus}", flush=True)
        time.sleep(20)

    done, states = all_complete()
    result = {"collection_finished_at": now(), "all_complete": done, "states": states}
    if done:
        result["train_returncode"] = train()
        result["train_finished_at"] = now()
    write_json(RUN / "pipeline_result.json", result)


def status(_args) -> None:
    done, states = all_complete()
    groups = {}
    for path in (RUN / "buffer").glob("rollouts_*.jsonl"):
        keys = {(json.loads(line)["prefix_k"]) for line in path.read_text().splitlines() if line.strip()}
        groups[path.stem.removeprefix("rollouts_")] = len(keys)
    for key, st in states.items():
        print(f"{key:55s} {st:9s} groups={groups.get(key, 0)}/13")
    print("all complete:", done)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("pipeline")
    sub.add_parser("status")
    sub.add_parser("schedule")
    w = sub.add_parser("worker")
    w.add_argument("--name", required=True)
    args = p.parse_args()
    {"prepare": prepare, "pipeline": pipeline, "status": status, "worker": worker,
     "schedule": schedule}[args.cmd](args)


if __name__ == "__main__":
    main()
