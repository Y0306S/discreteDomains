"""Schedule the approved ML campaign across three hosts with shared task claims."""

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

import inventory_ml_fixed_tasks as inventory
import ml_run_archive as run_archive


SOURCE = inventory.LOCAL
SHARED = inventory.SHARED
DEFAULT = SHARED / "benchmark_data/ml_test_20_fixed_48h_20260925"
BASELINE_PY = "/root/data/cys/baseline/envs/bl_lolbo_harness_cu118/bin/python"
CKPT_PY = "/root/blockdata/miniconda3/envs/openbox_qwen_eval/bin/python"
SERIAL = {"adaboost", "gradient_boosting"}
HOSTS = {
    "LLMOptimizer2": {"hostname": "cs-94bac-0693e-server", "cpus": list(range(36)),
                      "cpu_quota": 40, "memory_gib": 400, "limits": {"ckpt32": 8, "tpe": 4, "lolbo": 8},
                      "gpu_slots": [0, 0, 1, 1, 2, 2, 3, 3]},
    "LLMOptimizer3": {"hostname": "cs-0b2cd-18deb-server", "cpus": list(range(8, 12)) + list(range(24, 28)) + list(range(16, 20)),
                      "cpu_quota": 14, "memory_gib": 56, "limits": {"lolbo": 6},
                      "gpu_slots": [0, 1, 3, 2, 4, 5]},
    "LLMOptimizer4": {"hostname": "cs-fa498-8793f-server", "cpus": list(range(16)) + list(range(24, 40)),
                      "cpu_quota": 64, "memory_gib": 256, "limits": {"random_search": 16},
                      "gpu_slots": []},
}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def digest(path):
    try:
        stream = Path(path).open("rb")
    except FileNotFoundError:
        return hashlib.sha256(run_archive.read_bytes(path)).hexdigest()
    h = hashlib.sha256()
    with stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path, default=None):
    return run_archive.load_json(path, default)


def code_root(campaign, protocol=None):
    protocol = protocol or load(campaign / "protocol.json")
    return Path(protocol.get("code_root", campaign / "code"))


def task_key(task):
    return inventory.key(task)


def prepare(campaign, ckpt32_policy):
    assert digest(inventory.MANIFEST) == inventory.MANIFEST_HASH
    tasks = load(inventory.MANIFEST)
    assert len(tasks) == len({task_key(task) for task in tasks}) == 132
    policy_path = SOURCE / "configs/ml_runtime_policy.json"
    policy = inventory.load_policy(policy_path)
    campaign.mkdir(parents=True, exist_ok=False)
    shutil.copy2(inventory.MANIFEST, campaign / "tasks_132.json")
    shutil.copy2(policy_path, campaign / "runtime_policy.json")
    errors, counts, costs = [], {}, {}
    selected = []
    for method in inventory.METHODS:
        found = inventory.scan(inventory.LEGACY[method], method, "legacy", {task_key(t): t for t in tasks}, policy, errors)
        recovery = inventory.scan(inventory.CAMPAIGN / "outputs" / method, method, "recovery_12h",
                                  {task_key(t): t for t in tasks}, policy, errors)
        for key, attempts in recovery.items():
            found[key].extend(attempts)
        reused, queue = [], []
        for task in tasks:
            key = task_key(task)
            clean = next((a for a in reversed(found[key]) if a["clean_complete"]), None)
            if clean:
                record = {**task, **clean, "method": method, "sha256": digest(clean["path"])}
                reused.append(record)
                selected.append(record)
            else:
                queue.append(task)
            for attempt in found[key]:
                with run_archive.open_text(attempt["path"], newline="") as stream:
                    rows = list(csv.DictReader(stream))
                durations = [float(row.get("eval_time") or 0) for row in rows]
                estimate = sum(durations) / len(durations) * 100 if durations else 0
                costs[key] = max(costs.get(key, 0), estimate)
        save(campaign / "queues" / f"{method}.json", queue)
        save(campaign / "reused" / f"{method}.json", reused)
        counts[method] = {"fresh": len(queue), "reused_clean": len(reused)}
    if errors:
        save(campaign / "inspection_errors.json", errors)
        raise RuntimeError("Resolve inventory inspection errors before launch")
    for method in inventory.METHODS:
        path = campaign / "queues" / f"{method}.json"
        queue = load(path)
        queue.sort(key=lambda task: (task_key(task) != "rgs/black_friday/extra_trees",
                                     -costs.get(task_key(task), 0), task_key(task)))
        save(path, queue)
    code = campaign / "code"
    for name in ("mindware", "examples", "configs"):
        shutil.copytree(SOURCE / name, code / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # Reuse the already hashed, immutable dataset snapshot for the same split.
    (code / "datasets_bundle").symlink_to(inventory.CAMPAIGN / "code/datasets_bundle", target_is_directory=True)
    for name in ("data_manifest.json",):
        shutil.copy2(inventory.CAMPAIGN / name, campaign / name)
    bank = SOURCE / "history_data/experience_bank_v3_train80.jsonl"
    bank_pairs = {(r["problem"]["dataset_id"], r["problem"]["algorithm_id"])
                  for r in (json.loads(line) for line in bank.read_text().splitlines() if line)}
    assert not bank_pairs & {(t["dataset"], t["algorithm"]) for t in tasks}
    shutil.copy2(bank, campaign / "experience_bank.jsonl")
    shutil.copy2(SOURCE / "benchmark_data/ml_test_20_fixed_ckpt32_rec4_ws3/rec4_ws3_presets.json",
                 campaign / "warm_start_presets.json")
    save(campaign / "code_sha256.json", {str(p.relative_to(code)): digest(p)
                                         for p in code.rglob("*") if p.is_file()})
    protocol = {"campaign": campaign.name, "created_at": time.time(), "manifest_sha256": inventory.MANIFEST_HASH,
                "policy_sha256": digest(campaign / "runtime_policy.json"), "hosts": HOSTS,
                "ckpt32_policy": ckpt32_policy,
                "counts": counts, "seed": 1, "reuse": "Only existing 100-success trajectories",
                "fresh_start": "All remaining tasks start in new sessions; no legacy partials restored",
                "cpu_policy": "1 CPU for AdaBoost/gradient boosting; 4 for other estimators",
                "tasksetbo": "Keep running; random search shares capacity at lower CPU priority",
                "automatic_process_retries": 0, "archive_completed_runs": True,
                "bank_sha256": digest(campaign / "experience_bank.jsonl"),
                "ckpt_python": CKPT_PY, "baseline_python": BASELINE_PY,
                "adapter": "/root/blockdata/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_v4_pareto32_v12/ckpt32",
                "base_model": "/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B"}
    save(campaign / "protocol.json", protocol)
    save(campaign / "selected_legacy.json", selected)
    status(campaign)


def process_alive(state):
    try:
        if state.get("boot_id") != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
            return False
        path = Path("/proc") / str(state["pid"])
        fields = (path / "stat").read_text().rsplit(") ", 1)[1].split()
        if fields[0] == "Z" or fields[19] != state["start_ticks"]:
            return False
        argv = (path / "cmdline").read_bytes().decode().split("\0")
        return "--session-dir" in argv and argv[argv.index("--session-dir") + 1] == state["session"]
    except (OSError, KeyError, ValueError, IndexError):
        return False


def claim(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return stream
    except BlockingIOError:
        stream.close()
        return None


def progress(campaign, method, task):
    return load(campaign / "sessions" / method / task_key(task) / "progress.json",
                {"attempts": 0, "successful": 0, "failed": 0, "clean_complete": False,
                 "budget_exhausted": False, "status": "pending"})


def command(campaign, method, task, cpus, gpu):
    protocol = load(campaign / "protocol.json")
    code = code_root(campaign, protocol)
    session = campaign / "sessions" / method / task_key(task)
    args = ["taskset", "-c", ",".join(map(str, cpus)),
            protocol["ckpt_python"] if method == "ckpt32" else protocol["baseline_python"],
            str(code / "examples/benchmark/run_toy_categorical_optimizers.py"),
            "--task-type", task["task_type"], "--dataset", task["dataset"],
            "--algorithm", task["algorithm"], "--metric", task["metric"], "--seed", "1",
            "--runtime-policy", str(campaign / "runtime_policy.json"), "--session-dir", str(session),
            "--max-rounds", "1", "--dataset-retries", "1", "--skip-test-eval", "--resume",
            "--output-dir", str(campaign / "outputs" / method),
            "--history-dir", str(campaign / "history" / method)]
    if method == "ckpt32":
        args += ["--optimizer", protocol.get("ckpt32_optimizer", "ptllm_tpe_qwen"), "--llm-backends", "qwen-sft6", "--qwen-backend", "hf",
                 "--qwen-model-path", protocol["base_model"], "--qwen-adapter-path", protocol["adapter"],
                 "--llm-max-tokens", "768",
                 "--llm-response-format", protocol.get("ckpt32_response_format", "json_only"),
                 "--llm-n-initial-samples", "3", "--preset-warm-start-json", str(campaign / "warm_start_presets.json"),
                 "--experience-bank", str(campaign / "experience_bank.jsonl"),
                 "--source-gp-cache", str(campaign / "source_gp_train80_v1"),
                 "--prompt-variant", "P5plusA5T50X50RS5f05v4", "--hybrid-variant",
                 protocol.get("ckpt32_hybrid_variant", "HPUCB_arm")]
        if protocol.get("ckpt32_optimizer") == "ptllm_rgpe_qwen":
            assets = protocol["rgpe_assets"]
            args += ["--transfer-dump-dir", str(Path(assets) / "dumps"),
                     "--rgpe-cache-dir", str(Path(assets) / "rgpe"),
                     "--transfer-n-pending", "256", "--transfer-worker-timeout", "300"]
    else:
        args += ["--optimizer", method]
    args += protocol.get("method_args", {}).get(method, [])
    env = os.environ.copy()
    env.update({"PYTHONPATH": f"{code}/examples/benchmark:{code}",
                "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": "" if gpu is None else str(gpu),
                "EVAL_N_JOBS": str(len(cpus)), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"})
    if protocol.get("rgpe_assets"):
        env.update(ML_TRANSFER_HISTORY_MIRROR=str(Path(protocol["rgpe_assets"]) / "history"),
                   ML_TRANSFER_TRAIN_MANIFEST=str(Path(protocol["rgpe_assets"]) / "train_manifest.jsonl"))
    env.update(protocol.get("worker_env", {}))
    if method == "random_search":
        args = ["nice", "-n", "5"] + args
    return args, env, session


def status(campaign, quiet=False):
    summary = {}
    records = []
    protocol = load(campaign / "protocol.json")
    methods = protocol.get("methods", inventory.METHODS)
    for method in methods:
        queue, reused = load(campaign / "queues" / f"{method}.json"), load(campaign / "reused" / f"{method}.json")
        counts = {"reused_clean": len(reused), "new_clean": 0, "complete_with_failures": 0,
                  "running": 0, "incomplete": 0, "pending": 0, "held": 0, "liveness_unverified": 0}
        for task in reused:
            records.append({"method": method, "task_id": task_key(task), "seed": 1,
                            "status": "reused_clean", "attempts": 100, "successful": 100,
                            "failed": 0, "host": None, "trajectory": task["path"],
                            "run_id": "legacy:" + task["sha256"], "policy": "legacy"})
        for task in queue:
            state = load(campaign / "state" / method / f"{task_key(task)}.json", {})
            info = progress(campaign, method, task)
            task_status = info["status"] if info["budget_exhausted"] else state.get("status", "pending")
            host_state = load(campaign / "hosts" / f"{state.get('host')}.json", {})
            if task_status == "running":
                listed = any(item.get("pid") == state.get("pid") and item.get("start_ticks") == state.get("start_ticks")
                             for item in host_state.get("active", []))
                if not listed or time.time() - host_state.get("updated_at", 0) > 30:
                    task_status = "liveness_unverified"
            if method == "ckpt32" and protocol["ckpt32_policy"] == "hold":
                task_status = "held"
            if info["clean_complete"]:
                counts["new_clean"] += 1
            elif info["budget_exhausted"]:
                counts["complete_with_failures"] += 1
            elif task_status in ("held", "liveness_unverified"):
                counts[task_status] += 1
            elif task_status == "running":
                counts["running"] += 1
            elif state and task_status != "pending":
                counts["incomplete"] += 1
            else:
                counts["pending"] += 1
            records.append({"method": method, "task_id": task_key(task), "seed": 1,
                            "status": task_status,
                            "attempts": info["attempts"], "successful": info["successful"], "failed": info["failed"],
                            "host": state.get("host"), "trajectory": info.get("trajectory"),
                            "run_id": info.get("identity"),
                            "policy": info.get("runtime_policy_id") or protocol.get("initial_policy_id", "ml_48h_100_attempts_v1")})
            packed = run_archive.archive_path(campaign, method, task)
            if packed.exists():
                records[-1]["archive"] = str(packed)
        counts["clean_total"] = counts["reused_clean"] + counts["new_clean"]
        counts["remaining_clean"] = 132 - counts["clean_total"]
        summary[method] = counts
    expected_records = int(protocol.get("task_count", 132)) * len(methods)
    assert len(records) == len({(r["method"], r["task_id"], r["seed"]) for r in records}) == expected_records
    lock = claim(campaign / "report.lock")
    if lock:
        try:
            save(campaign / "task_index.json", {"updated_at": time.time(), "manifest_sha256": inventory.MANIFEST_HASH,
                                               "summary": summary, "tasks": records})
            with tempfile.NamedTemporaryFile(mode="w", dir=campaign, delete=False) as stream:
                writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in records for k in row)))
                writer.writeheader()
                writer.writerows(records)
                temporary = stream.name
            os.replace(temporary, campaign / "task_index.csv")
        finally:
            lock.close()
    if not quiet:
        print(json.dumps(summary, indent=2), flush=True)
    return summary


def cpu_throttle_events():
    for line in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines():
        if line.startswith("nr_throttled"):
            return int(line.split()[1])
    return 0


def start_archiver(campaign, host):
    script = code_root(campaign) / "examples/benchmark/archive_ml_campaign.py"
    path = campaign / "logs" / f"archive_{host}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        return subprocess.Popen(["nice", "-n", "19", sys.executable, str(script), "pack",
            "--campaign", str(campaign), "--host", host, "--cleanup"],
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)


def release_remote_runtime(state, strict=False):
    if not state.get("remote_runtime_config"):
        return
    from mindware.components.optimizers.tosfit_remote import RemoteToSFiTRuntime
    from mindware.components.optimizers.runtime_rpc import call
    config = load(state["remote_runtime_config"])
    health = call(config["url"] + "/health", timeout=10)
    if not health.get("run_id"):
        return
    if health.get("session") != str(Path(state["session"]).resolve()):
        if strict:
            raise RuntimeError("The remote slot has a different task owner")
        return
    RemoteToSFiTRuntime(config, state["session"], health["run_id"], 1).detach()


def run(campaign, host, retry_incomplete=False):
    protocol = load(campaign / "protocol.json")
    if not protocol.get("queue_enabled", True):
        raise RuntimeError("This campaign is prepared but disabled")
    spec = protocol["hosts"][host]
    assert socket.gethostname() == spec["hostname"]
    assert digest(campaign / "tasks_132.json") == protocol["manifest_sha256"]
    assert digest(campaign / "runtime_policy.json") == protocol["policy_sha256"]
    host_lock = claim(campaign / "locks" / f"{host}.lock")
    if host_lock is None:
        raise SystemExit("This host already has a scheduler")
    assert set(spec["cpus"]) <= os.sched_getaffinity(0)
    if spec.get("scheduler_cpus"):
        os.sched_setaffinity(0, spec["scheduler_cpus"])
    queues = {method: load(campaign / "queues" / f"{method}.json") for method in spec["limits"]}
    if protocol["ckpt32_policy"] == "hold":
        queues.pop("ckpt32", None)
        if host == "LLMOptimizer2":
            queues.pop("lolbo", None)
    active, attempted = {}, set()
    archive_child, archive_due = None, bool(protocol.get("archive_completed_runs"))
    last_report, last_launch = 0, 0
    # Container CPU-quota throttling slows every process (including PostgreSQL); launch nothing
    # until the throttle counter has stayed flat for this many scheduler passes.
    quiet_needed = int(spec.get("throttle_quiet_passes", 0))
    last_throttle, quiet = cpu_throttle_events(), quiet_needed
    for path in (campaign / "state").rglob("*.json"):
        state = load(path)
        key = (state["method"], task_key(state))
        if state.get("host") != host:
            continue
        if process_alive(state):
            lock = claim(campaign / "locks" / state["method"] / f"{task_key(state)}.lock")
            if lock:
                active[key] = {"state": state, "process": None, "lock": lock, "state_path": path}
        else:
            release_remote_runtime(state)
            info = progress(campaign, state["method"], state)
            state["status"] = info["status"] if info["budget_exhausted"] else ("pending" if retry_incomplete else "incomplete")
            save(path, state)
            if not retry_incomplete:
                attempted.add(key)
    while True:
        for key, item in list(active.items()):
            state, child = item["state"], item["process"]
            ended = child.poll() is not None if child else not process_alive(state)
            if not ended:
                continue
            release_remote_runtime(state, strict=True)
            info = progress(campaign, state["method"], state)
            state.update(finished_at=time.time(), returncode=child.returncode if child else None,
                         status=info["status"] if info["budget_exhausted"] else "incomplete", progress=info)
            save(item["state_path"], state)
            if protocol.get("archive_completed_runs") and info["budget_exhausted"]:
                archive_due = True
            item["lock"].close()
            attempted.add(key)
            del active[key]
            print("END", key, state["status"], info["attempts"], flush=True)
        if archive_child is not None and archive_child.poll() is not None:
            if archive_child.returncode:
                print("ARCHIVE_ERROR", "worker exited", archive_child.returncode, flush=True)
            archive_child = None
        if archive_due and archive_child is None:
            archive_child = start_archiver(campaign, host)
            archive_due = False
        occupied = {cpu for item in active.values() for cpu in item["state"]["cpus"]}
        free = [cpu for cpu in spec["cpus"] if cpu not in occupied]
        gpu_used = [item["state"]["gpu"] for item in active.values() if item["state"]["gpu"] is not None]
        gpu_free = list(spec["gpu_slots"])
        for gpu in gpu_used:
            # After a slot reduction a GPU can briefly run more tasks than it has slots; it just gets no new launches.
            if gpu in gpu_free:
                gpu_free.remove(gpu)
        memory = int(Path("/sys/fs/cgroup/memory.current").read_text())
        working = memory
        if spec.get("memory_guard") == "cgroup_working_set":
            stats = dict(line.split() for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines())
            working -= int(stats.get("inactive_file", 0)) + int(stats.get("slab_reclaimable", 0))
        pressure = working > 0.85 * spec["memory_gib"] * 2**30
        # Start the TPE long task immediately; give ckpt32 the GPU slots before lending them to LOLBO.
        methods = [m for m in protocol.get("methods", ("tpe", "ckpt32", "lolbo", "random_search"))
                   if m in queues]
        candidates = []
        for method in methods:
            for task in queues[method]:
                key = (method, task_key(task))
                if key in attempted or key in active:
                    continue
                info = progress(campaign, method, task)
                state = load(campaign / "state" / method / f"{task_key(task)}.json", {})
                if info["budget_exhausted"] or (state.get("status") in ("incomplete", "complete_with_failures") and not retry_incomplete):
                    attempted.add(key)
                    continue
                candidates.append((method, task, key))
        pending_ckpt = any(m == "ckpt32" for m, _, _ in candidates)
        throttle = cpu_throttle_events()
        if throttle > last_throttle:
            quiet = 0
            print("THROTTLE", throttle - last_throttle, "launches paused", flush=True)
        else:
            quiet += 1
        last_throttle = throttle
        launchable = candidates if quiet >= quiet_needed else []
        from ml_campaign_pool import resources
        from ml_memory_admission import admission
        with resources(campaign, host) as pool, admission(spec) as memory_admission:
            if pool is not None:
                free = [cpu for cpu in free if cpu not in pool.used_cpus]
            for method, task, key in launchable:
                count = sum(item["state"]["method"] == method for item in active.values())
                if count >= spec["limits"][method] or pressure:
                    continue
                if pool is not None and not pool.available(campaign):
                    break
                if host == "LLMOptimizer2" and method == "tpe" and count >= 1 and pending_ckpt:
                    continue
                if host == "LLMOptimizer2" and method == "lolbo" and pending_ckpt:
                    continue
                size = int(spec.get("worker_cores", {}).get(method,
                           spec.get("serial_worker_cores", 1) if task["algorithm"] in SERIAL
                           else spec.get("parallel_worker_cores", 4)))
                if len(free) < size:
                    continue
                if time.monotonic() - last_launch < spec.get("launch_interval_seconds", 0):
                    continue
                estimate = memory_admission.estimate(method, task) if memory_admission else 0
                if memory_admission and not memory_admission.allowed(estimate):
                    continue
                uses_gpu = method in spec.get("gpu_methods", ("ckpt32", "lolbo"))
                gpu = gpu_free[0] if uses_gpu and gpu_free else None
                if uses_gpu and pool is not None:
                    gpu = pool.choose_gpu(method, gpu_free)
                if uses_gpu and gpu is None:
                    continue
                slot = None
                if spec.get("fixed_slots"):
                    slot = next((s for s in spec["fixed_slots"] if set(s["cpus"]) <= set(free)), None)
                    if slot is None:
                        continue
                    gpu = slot.get("gpu")
                    if uses_gpu and gpu not in gpu_free:
                        continue
                lock = claim(campaign / "locks" / method / f"{task_key(task)}.lock")
                if lock is None:
                    continue
                state_path = campaign / "state" / method / f"{task_key(task)}.json"
                existing = load(state_path, {})
                if existing.get("status") == "running" and existing.get("host") != host:
                    # The owning host holds the claim throughout execution; a stale cross-host claim needs inspection.
                    lock.close()
                    continue
                preferred = list(free)
                if host == "LLMOptimizer2" and method == "tpe":
                    preferred.sort(key=lambda cpu: (cpu < 32, cpu))
                cpus = slot["cpus"] if slot else preferred[:size]
                if len(cpus) != size:
                    raise RuntimeError("Fixed slot violates method CPU policy")
                args, env, session = command(campaign, method, task, cpus, gpu)
                env.update(slot.get("env", {}) if slot else {})
                log_path = campaign / "logs" / method / f"{task_key(task)}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a") as stream:
                    child = subprocess.Popen(args, cwd=code_root(campaign, protocol), env=env, stdin=subprocess.DEVNULL,
                                             stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                state = {**task, "method": method, "host": host, "status": "running", "pid": child.pid,
                         "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                         "start_ticks": Path(f"/proc/{child.pid}/stat").read_text().rsplit(") ", 1)[1].split()[19],
                         "cpus": cpus, "gpu": gpu, "started_at": time.time(), "session": str(session),
                         "command": args, "log": str(log_path)}
                if env.get("TOSFIT_REMOTE_CONFIG"):
                    state["remote_runtime_config"] = env["TOSFIT_REMOTE_CONFIG"]
                save(state_path, state)
                if memory_admission:
                    memory_admission.reserve(state, estimate)
                if pool is not None:
                    pool.reserve(method, cpus, gpu)
                active[key] = {"state": state, "process": child, "lock": lock, "state_path": state_path}
                free = [cpu for cpu in free if cpu not in cpus]
                if uses_gpu:
                    gpu_free.remove(gpu)
                last_launch = time.monotonic()
                print("START", method, task_key(task), "cpus", cpus, "gpu", gpu, flush=True)
        save(campaign / "hosts" / f"{host}.json", {"updated_at": time.time(), "host": host,
             "pid": os.getpid(), "active": [item["state"] for item in active.values()],
             "cpu_slots_used": sum(len(item["state"]["cpus"]) for item in active.values()),
             "cpu_slots_budget": len(spec["cpus"]), "memory_bytes": memory})
        if time.monotonic() - last_report >= 30 or (not active and not candidates):
            status(campaign, quiet=True)
            last_report = time.monotonic()
        if not active and not candidates and not archive_due:
            break
        time.sleep(5)
    host_lock.close()


def start(campaign, host, retry_incomplete=False):
    script = code_root(campaign) / "examples/benchmark/ml_fixed_campaign.py"
    protocol = load(campaign / "protocol.json")
    command = [protocol.get("scheduler_python", BASELINE_PY), str(script), "run",
               "--campaign", str(campaign), "--host", host]
    if retry_incomplete:
        command.append("--retry-incomplete")
    path = campaign / "logs" / f"scheduler_{host}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({"host": host, "scheduler_pid": child.pid, "log": str(path)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "start", "run", "status"))
    parser.add_argument("--campaign", type=Path, default=DEFAULT)
    parser.add_argument("--host", choices=HOSTS)
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--ckpt32-policy", choices=("rerun-all", "hold"))
    args = parser.parse_args()
    if args.action == "prepare":
        if not args.ckpt32_policy:
            parser.error("prepare requires an explicit --ckpt32-policy")
        prepare(args.campaign, args.ckpt32_policy)
    elif args.action == "status":
        status(args.campaign)
    elif not args.host:
        parser.error("--host is required")
    elif args.action == "start":
        start(args.campaign, args.host, args.retry_incomplete)
    else:
        run(args.campaign, args.host, args.retry_incomplete)


if __name__ == "__main__":
    main()
