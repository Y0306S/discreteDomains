#!/usr/bin/env python3
"""Benchmark approved OptFormer runtime settings and resume production training."""

import argparse
import csv
import datetime
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mindware.components.optimizers.transfer_hpo.protocol import atomic_json, sha256_directory
from examples.benchmark.launch_full_transfer_training import optformer_environment

CONFIGURATIONS = [
    ("1gpu_b4_log1", "2", 4, 1), ("1gpu_b4_log20", "2", 4, 20),
    ("1gpu_b8_log20", "2", 8, 20), ("1gpu_b16_log20", "2", 16, 20),
    ("2gpu_b4_log20", "0,2", 4, 20), ("2gpu_b8_log20", "0,2", 8, 20),
    ("4gpu_b4_log20", "0,1,2,3", 4, 20), ("4gpu_b8_log20", "0,1,2,3", 8, 20),
]
METRIC_ROW = re.compile(r"^[IWE](\d{4}) (\d\d:\d\d:\d\d\.\d+) .*?\] \[(\d+)\] collection=train (.*)$")
WARMUP_STEPS, TIMED_STEPS = 20, 40
MEMORY_FRACTION, MIN_FREE_MIB = 0.3, 8192


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def process_running(pid):
    path = Path("/proc") / str(pid) / "stat"
    return path.exists() and path.read_text().rsplit(")", 1)[1].split()[0] != "Z"


def checked_process_group(pid):
    command = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
    worker = str(ROOT / "examples/benchmark/optformer_worker.py").encode()
    if worker not in command or b"train" not in command or os.getpgid(pid) != pid:
        raise RuntimeError("Refusing to signal a process other than the isolated OptFormer worker")
    return pid


def stop_group(pid):
    group = checked_process_group(pid)
    os.killpg(group, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            return
        time.sleep(0.25)
    try:
        os.killpg(group, signal.SIGKILL)
    except ProcessLookupError:
        pass


def gpu_memory():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.total,memory.used",
        "--format=csv,noheader,nounits"], text=True, timeout=15)
    return {row[0].strip(): {"total": int(row[1]), "used": int(row[2])}
            for row in csv.reader(io.StringIO(output))}


def latest_checkpoint(directory):
    checkpoints = [path for path in directory.glob("checkpoint_*")
                   if re.fullmatch(r"checkpoint_\d+", path.name) and (path / "checkpoint").is_file()]
    return max(checkpoints, key=lambda path: int(path.name.split("_")[-1]))


def metric_rows(path):
    rows = {}
    year = str(datetime.datetime.fromtimestamp(path.stat().st_mtime).year)
    for line in path.read_text(errors="replace").splitlines():
        match = METRIC_ROW.match(line)
        if not match:
            continue
        values = dict(part.split("=", 1) for part in match[4].split(", ") if "=" in part)
        if "loss" not in values:
            continue
        parsed = {key: float(value) for key, value in values.items()}
        timestamp = datetime.datetime.strptime(year + match[1] + " " + match[2], "%Y%m%d %H:%M:%S.%f")
        rows[int(match[3])] = {"timestamp": timestamp.timestamp(), "metrics": parsed}
    return rows


def summarize(path, start_step):
    rows = metric_rows(path)
    first, last = start_step + WARMUP_STEPS, start_step + WARMUP_STEPS + TIMED_STEPS
    if first not in rows or last not in rows:
        raise ValueError("Missing warm-up boundary or final training metrics")
    if not all(math.isfinite(value) for row in rows.values() for value in row["metrics"].values()):
        raise ValueError("Non-finite training metrics")
    elapsed = rows[last]["timestamp"] - rows[first]["timestamp"]
    if elapsed <= 0:
        raise ValueError("Invalid timing interval")
    return {"timed_steps": TIMED_STEPS, "timed_seconds": elapsed,
            "seconds_per_update": elapsed / TIMED_STEPS,
            "updates_per_second": TIMED_STEPS / elapsed,
            "final_loss_per_token": rows[last]["metrics"]["loss_per_nonpadding_target_token"]}


def choose_configuration(results):
    successful = [result for result in results if result.get("status") == "passed"]
    baseline = next((result for result in successful if result["name"] == CONFIGURATIONS[0][0]), None)
    if not baseline:
        return None
    fastest = min(successful, key=lambda result: result["seconds_per_update"])
    speedup = baseline["seconds_per_update"] / fastest["seconds_per_update"]
    return fastest if speedup >= 1.1 else baseline


def replace_option(command, option, value):
    command = list(command)
    if option in command:
        command[command.index(option) + 1] = str(value)
    else:
        command.extend([option, str(value)])
    return command


def runtime_command(original, model_dir, steps, microbatches, stats_every, benchmark=False):
    command = list(original)
    for option, value in (("--model-dir", model_dir), ("--steps", steps),
                          ("--microbatches", microbatches), ("--stats-every", stats_every)):
        command = replace_option(command, option, value)
    for flag in ("--resume", "--reconfigure-execution"):
        if flag not in command:
            command.append(flag)
    if benchmark:
        command.append("--benchmark")
    return command


def start_worker(command, log_path, gpu_ids, memory_fraction=MEMORY_FRACTION):
    env = optformer_environment(gpu_ids)
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_fraction)
    with log_path.open("x") as log:
        return subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def run_candidate(original, frozen, root, configuration, start_step, deadline):
    name, gpu_ids, per_gpu, stats_every = configuration
    directory = root / name
    directory.mkdir()
    microbatches = 256 // (len(gpu_ids.split(",")) * per_gpu)
    result = {"name": name, "gpu_ids": gpu_ids, "per_gpu_microbatch": per_gpu,
              "microbatches": microbatches, "stats_every": stats_every, "status": "preparing"}
    memory = gpu_memory()
    for gpu in gpu_ids.split(","):
        if memory[gpu]["total"] * (1 - MEMORY_FRACTION) - memory[gpu]["used"] < MIN_FREE_MIB:
            return dict(result, status="skipped_memory_headroom")
    for filename in ("meta.json", "model.gin"):
        shutil.copy2(frozen / filename, directory / filename)
    source_checkpoint = latest_checkpoint(frozen)
    shutil.copytree(source_checkpoint, directory / source_checkpoint.name)
    if sha256_directory(directory / source_checkpoint.name) != sha256_directory(source_checkpoint):
        raise RuntimeError("Copied benchmark checkpoint digest mismatch")
    steps = start_step + WARMUP_STEPS + TIMED_STEPS
    command = runtime_command(original["command"], directory, steps, microbatches, stats_every, benchmark=True)
    log_path = directory / "training.log"
    launched = time.monotonic()
    child = start_worker(command, log_path, gpu_ids)
    result.update(status="running", pid=child.pid, command=command, log=str(log_path), started_utc=utc_now())
    atomic_json(directory / "result.json", result)
    failure = None
    try:
        while child.poll() is None:
            memory = gpu_memory()
            result["peak_total_gpu_used_mib"] = {
                gpu: max(result.get("peak_total_gpu_used_mib", {}).get(gpu, 0), memory[gpu]["used"])
                for gpu in gpu_ids.split(",")}
            if any(memory[gpu]["total"] - memory[gpu]["used"] < MIN_FREE_MIB for gpu in gpu_ids.split(",")):
                failure = "memory_headroom_exceeded"
            elif time.monotonic() - launched >= 12 * 60 or time.monotonic() >= deadline:
                failure = "timeout"
            if failure:
                stop_group(child.pid)
                break
            time.sleep(2)
        child.wait(timeout=30)
        result.update(returncode=child.returncode, elapsed_seconds=time.monotonic() - launched)
        if failure or child.returncode:
            result.update(status="failed", failure=failure or "worker_error")
        else:
            meta = json.loads((directory / "meta.json").read_text())
            if meta.get("status") != "benchmark_complete" or meta.get("steps_completed") != steps:
                raise ValueError("Worker did not verify its complete benchmark checkpoint")
            result.update(summarize(log_path, start_step), status="passed")
    finally:
        if child.poll() is None:
            stop_group(child.pid)
            child.wait(timeout=30)
    atomic_json(directory / "result.json", result)
    return result


def resume_production(campaign, root, original, selected):
    if selected is None or selected["name"] == CONFIGURATIONS[0][0]:
        gpu_ids, microbatches, stats_every, memory_fraction = original.get("gpu", "2"), 64, 1, 0.2
    else:
        gpu_ids, microbatches, stats_every = selected["gpu_ids"], selected["microbatches"], selected["stats_every"]
        memory_fraction = MEMORY_FRACTION
    model_dir = original["command"][original["command"].index("--model-dir") + 1]
    command = runtime_command(original["command"], model_dir, 100000, microbatches, stats_every)
    log_path = root / "production_resume.log"
    child = start_worker(command, log_path, gpu_ids, memory_fraction)
    launch = json.loads((campaign / "launch.json").read_text())
    launch.setdefault("optformer_previous_attempts", []).append(dict(original, status="paused_for_runtime_benchmark"))
    launch["optformer"] = {"pid": child.pid, "command": command, "log": str(log_path),
                           "effective_batch_size": 256, "microbatch_size": 256 // microbatches,
                           "per_gpu_microbatch_size": 256 // microbatches // len(gpu_ids.split(",")),
                           "jax_memory_fraction": memory_fraction, "gpu": gpu_ids, "status": "launched",
                           "restarted_utc": utc_now(), "runtime_benchmark": str(root)}
    atomic_json(campaign / "launch.json", launch)
    return launch["optformer"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    campaign, root = args.campaign.resolve(), args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    launch = json.loads((campaign / "launch.json").read_text())
    original = launch["optformer"]
    checked_process_group(original["pid"])
    model_dir = Path(original["command"][original["command"].index("--model-dir") + 1])
    meta = json.loads((model_dir / "meta.json").read_text())
    if meta["batch_size"] != 256 or meta["requested_steps"] != 100000 or meta["model_size"] != "base":
        raise ValueError("Production settings differ from the approved benchmark plan")
    state = {"status": "waiting_for_checkpoint", "started_utc": utc_now(), "original": original,
             "warmup_steps": WARMUP_STEPS, "timed_steps": TIMED_STEPS,
             "max_benchmark_minutes": 90, "min_free_mib": MIN_FREE_MIB,
             "jax_memory_fraction": MEMORY_FRACTION, "results": []}
    atomic_json(root / "status.json", state)
    code = root / "code"
    code.mkdir()
    for path in (Path(__file__), ROOT / "examples/benchmark/optformer_worker.py",
                 ROOT / "examples/benchmark/optformer_t5x_train.py"):
        shutil.copy2(path, code / path.name)
    initial = latest_checkpoint(model_dir)
    wait_started = time.monotonic()
    print("Waiting for next complete checkpoint after", initial.name, flush=True)
    while latest_checkpoint(model_dir) == initial:
        if not process_running(original["pid"]) or time.monotonic() - wait_started > 20 * 60:
            raise RuntimeError("No new production checkpoint; original run was not stopped")
        time.sleep(1)
    source = latest_checkpoint(model_dir)
    frozen = root / "frozen"
    frozen.mkdir()
    for filename in ("meta.json", "model.gin", "train.gin"):
        shutil.copy2(model_dir / filename, frozen / filename)
    shutil.copytree(source, frozen / source.name)
    digest = sha256_directory(source)
    if digest != sha256_directory(frozen / source.name):
        raise RuntimeError("Preserved checkpoint digest mismatch; original run was not stopped")
    state.update(checkpoint=source.name, checkpoint_sha256=digest)
    stop_group(original["pid"])
    state.update(status="benchmarking", paused_utc=utc_now())
    atomic_json(root / "status.json", state)
    print("Preserved", source.name, "and stopped only OptFormer", flush=True)
    selected = None
    benchmark_started = time.monotonic()
    try:
        start_step = int(source.name.split("_")[-1])
        for configuration in CONFIGURATIONS:
            if time.monotonic() - benchmark_started >= 90 * 60:
                break
            state["current_configuration"] = configuration[0]
            atomic_json(root / "status.json", state)
            print("START", configuration[0], flush=True)
            try:
                result = run_candidate(original, frozen, root, configuration, start_step, benchmark_started + 90 * 60)
            except Exception as exc:
                result = {"name": configuration[0], "status": "failed", "failure": repr(exc)}
            state["results"].append(result)
            atomic_json(root / "status.json", state)
            print(json.dumps(result), flush=True)
        selected = choose_configuration(state["results"])
        state["selected_configuration"] = selected
        state["benchmark_elapsed_seconds"] = time.monotonic() - benchmark_started
    except BaseException as exc:
        state["error"] = repr(exc)
        raise
    finally:
        state["production"] = resume_production(campaign, root, original, selected)
        state.update(status="production_resumed", finished_utc=utc_now())
        atomic_json(root / "status.json", state)
        print("PRODUCTION_RESUMED", json.dumps(state["production"]), flush=True)


if __name__ == "__main__":
    main()
