#!/usr/bin/env python3
"""Launch the full ML-only FSBO and OptFormer source-training campaign."""

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mindware.components.optimizers.transfer_hpo.protocol import (
    TRAIN_MANIFEST, allowed_task_ids, atomic_json, partition_tasks, sha256_file, validate_dump,
)


def optformer_environment(gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu, XLA_PYTHON_CLIENT_MEM_FRACTION="0.2",
               XLA_PYTHON_CLIENT_PREALLOCATE="true", JAX_PLATFORM_NAME="gpu",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               TF_NUM_INTEROP_THREADS="1", TF_NUM_INTRAOP_THREADS="1", TF_CPP_MIN_LOG_LEVEL="2")
    env["PYTHONPATH"] = os.pathsep.join([
        str(ROOT), "/root/data/cys/baseline/repos/optformer-neurips22",
        "/root/data/cys/baseline/repos/t5x-2023-07",
    ])
    return env


def resume_optformer(directory, gpu):
    status = json.loads((directory / "launch.json").read_text())
    previous = status["optformer"]
    process_stat = Path("/proc") / str(previous["pid"]) / "stat"
    if process_stat.exists() and process_stat.read_text().rsplit(")", 1)[1].split()[0] != "Z":
        raise RuntimeError("The previous OptFormer process is still running")
    command = list(previous["command"])
    if "--resume" not in command:
        command.append("--resume")
    attempts = status.setdefault("optformer_previous_attempts", [])
    attempts.append(previous)
    log_path = directory / ("optformer_retry%d.log" % len(attempts))
    snapshots = directory / ("code_optformer_retry%d" % len(attempts))
    snapshots.mkdir()
    for source in [ROOT / "examples/benchmark" / name for name in (
            "optformer_worker.py", "optformer_t5x_train.py", "launch_full_transfer_training.py")]:
        shutil.copy2(source, snapshots / source.name)
    shutil.copy2(ROOT / "mindware/components/optimizers/transfer_hpo/optformer_bridge.py",
                 snapshots / "optformer_bridge.py")
    with log_path.open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=optformer_environment(gpu),
                                   stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    status["optformer"] = dict(previous, pid=process.pid, command=command, log=str(log_path),
                               status="launched", gpu=gpu,
                               restarted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    for key in ("trainer_pid", "verified_gradient_updates", "verified_finite_training_loss",
                "verification_utc", "verified_through_step", "resumed_from_step"):
        status["optformer"].pop(key, None)
    atomic_json(directory / "launch.json", status)
    print(json.dumps(status["optformer"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--fsbo-steps", type=int, default=100000)
    parser.add_argument("--optformer-steps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--resume-optformer", action="store_true")
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    if args.resume_optformer:
        return resume_optformer(directory, args.gpu)
    if directory.exists() and any(directory.iterdir()):
        parser.error("Use a new campaign directory; existing runs are preserved")
    source_root = ROOT / "benchmark_data/ml_train_80_transfer_v2"
    coverage = json.loads((source_root / "coverage.json").read_text())
    source_tasks = set()
    for group in coverage["groups"]:
        dump = json.loads(Path(group["dump"]).read_text())
        source_tasks.update(validate_dump(dump))
    rng = random.Random(args.seed)
    heldout_datasets = set()
    for task_type in ("cls", "rgs"):
        datasets = sorted({task.split("/")[1] for task in source_tasks if task.startswith(task_type + "/")})
        rng.shuffle(datasets)
        heldout_datasets.update((task_type, name) for name in datasets[:math.ceil(len(datasets) * 0.2)])
    validation_ids = sorted(task for task in allowed_task_ids() if tuple(task.split("/")[:2]) in heldout_datasets)
    directory.mkdir(parents=True, exist_ok=True)
    validation_file = directory / "validation_tasks.json"
    atomic_json(validation_file, validation_ids)
    for group in coverage["groups"]:
        dump = json.loads(Path(group["dump"]).read_text())
        partition_tasks(dump["tasks"], validation_file)
    train_ids, valid_ids = partition_tasks(source_tasks, validation_file)
    atomic_json(directory / "split.json", {
        "seed": args.seed, "validation_fraction": 0.2, "unit": "task_type/dataset",
        "train_manifest_sha256": sha256_file(TRAIN_MANIFEST),
        "training_tasks": train_ids, "validation_tasks": valid_ids,
        "validation_datasets": sorted([list(pair) for pair in heldout_datasets]),
    })
    frozen = directory / "sources"
    frozen.mkdir()
    for group in coverage["groups"]:
        source = Path(group["dump"])
        target = frozen / source.name
        shutil.copy2(source, target)
        group["dump"] = str(target)
    atomic_json(frozen / "coverage.json", coverage)
    sequences = frozen / "optformer_sequences.jsonl"
    shutil.copy2(source_root / "optformer/sequences.jsonl", sequences)
    snapshots = directory / "code"
    snapshots.mkdir()
    for source in [ROOT / "examples/benchmark" / name for name in (
            "fsbo_worker.py", "optformer_worker.py", "optformer_t5x_train.py", "run_transfer_training.py")]:
        shutil.copy2(source, snapshots / source.name)
    for source in (ROOT / "mindware/components/optimizers/transfer_hpo").glob("*.py"):
        shutil.copy2(source, snapshots / ("transfer_hpo_" + source.name))
    status = {
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "phase": "production_training", "n_train_tasks": len(train_ids), "n_validation_tasks": len(valid_ids),
        "fsbo_steps_per_space": args.fsbo_steps,
        "fsbo_stop_rule": "fixed update cap; select minimum validation NLL checkpoint",
        "optformer_steps": args.optformer_steps,
        "optformer_schedule_reference": "original release README: 100K updates at batch 256",
        "optformer_model_size": "base", "gpu": args.gpu, "fsbo": {},
    }
    env = optformer_environment(args.gpu)
    command = ["/root/data/cys/baseline/envs/bl_optformer/bin/python",
               str(ROOT / "examples/benchmark/optformer_worker.py"), "train",
               "--sequences", str(sequences), "--validation-tasks", str(validation_file),
               "--model-dir", str(directory / "optformer"), "--model-size", "base",
               "--steps", str(args.optformer_steps), "--seed", str(args.seed),
               "--batch-size", "256", "--microbatches", "64",
               "--eval-batch-size", "4", "--eval-steps", str(math.ceil(len(valid_ids) / 4)),
               "--save-every", "100"]
    with (directory / "optformer.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    status["optformer"] = {"pid": process.pid, "command": command, "log": str(directory / "optformer.log"),
                           "effective_batch_size": 256, "microbatch_size": 4,
                           "jax_memory_fraction": 0.2, "status": "launched"}
    atomic_json(directory / "launch.json", status)
    for group in coverage["groups"]:
        tag = Path(group["dump"]).stem
        command = [sys.executable, str(ROOT / "examples/benchmark/run_transfer_training.py"),
                   "fsbo", "--run-dir", str(directory / "fsbo_runs" / tag),
                   "--dump-dir", str(frozen), "--groups", tag,
                   "--steps", str(args.fsbo_steps), "--validation-tasks", str(validation_file),
                   "--seed", str(args.seed), "--validate-every", "100", "--detach"]
        result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), text=True,
                                capture_output=True, check=True)
        status["fsbo"][tag] = json.loads(result.stdout)
        atomic_json(directory / "launch.json", status)
    print(json.dumps({"run_dir": str(directory), "training_tasks": len(train_ids),
                      "validation_tasks": len(valid_ids), "fsbo_spaces": len(status["fsbo"]),
                      "optformer_pid": process.pid}, indent=2))


if __name__ == "__main__":
    main()
