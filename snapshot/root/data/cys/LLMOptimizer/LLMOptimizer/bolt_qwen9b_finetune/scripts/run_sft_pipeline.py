#!/usr/bin/env python3
"""
Run the Qwen3.5-9B SFT pipeline:

1. Full SFT configs run sequentially, using all 8 GPUs.
2. LoRA configs run in parallel on 4-GPU partitions.
3. When a LoRA job finishes, its GPUs are freed and the next queued
   LoRA job takes over that partition.

Usage:
    # Run everything (foreground)
    python run_sft_pipeline.py

    # Run with nohup (background)
    nohup python run_sft_pipeline.py > pipeline.log 2>&1 &

    # Skip full SFT, only run LoRA jobs
    python run_sft_pipeline.py --skip-full

    # Only run full SFT
    python run_sft_pipeline.py --skip-lora

    # Resume: skip jobs whose output already has a final model.safetensors
    python run_sft_pipeline.py --resume

The shell wrapper activates the llamafactory_qwen35 conda environment.
"""

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path("/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune")
LLAMAFACTORY_DIR = Path("/jizhicfs/pkuhetu/bcxu/LlamaFactory")
CONFIG_DIR = BASE_DIR / "configs" / "ml"
LOG_DIR = BASE_DIR / "logs"

FULL_SFT_CONFIGS = [
    "llamafactory_qwen35_9b_full_sft_recommended.yaml",
    "llamafactory_qwen35_9b_full_sft_4.yaml",
]

LORA_CONFIGS = [
    "llamafactory_qwen35_9b_base_lora_sft_recommended.yaml",
    "llamafactory_qwen35_9b_base_lora_sft_3.yaml",
    "llamafactory_qwen35_9b_base_lora_sft.yaml",
]

# Two 4-GPU partitions on an 8-GPU node.  Adjust if your node layout differs.
GPU_PARTITIONS = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
]

ALL_GPUS = list(range(8))


@dataclass
class Job:
    name: str
    config: Path
    gpus: List[int]
    log_file: Path
    is_full: bool


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    line = f"[{now_str()}] {message}"
    print(line, flush=True)


def run_command(
    cmd: List[str],
    env: dict,
    cwd: Path,
    log_file: Path,
    description: str,
    blocking: bool = True,
) -> Optional[subprocess.Popen]:
    """Run a command either blocking or non-blocking, teeing output to log_file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    log(f"{'[BLOCKING]' if blocking else '[BACKGROUND]'} {description}")
    log(f"  config: {cmd[-1]}")
    log(f"  GPUs  : {env.get('CUDA_VISIBLE_DEVICES', 'unset')}")
    log(f"  log   : {log_file}")

    with open(log_file, "w") as lf:
        lf.write(f"[{now_str()}] Starting: {description}\n")
        lf.write(f"[{now_str()}] Command: {' '.join(cmd)}\n")
        lf.write(f"[{now_str()}] GPUs: {env.get('CUDA_VISIBLE_DEVICES', 'unset')}\n")
        lf.flush()

        if blocking:
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdout=lf,
                stderr=subprocess.STDOUT,
                text=True,
            )
            proc.wait()
            lf.write(f"[{now_str()}] Finished with exit code {proc.returncode}\n")
            log(f"Finished {description} with exit code {proc.returncode}")
            return proc
        else:
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdout=lf,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log(f"Launched {description} (pid={proc.pid})")
            return proc


def build_env(gpus: List[int], is_full: bool) -> dict:
    """Build the environment for a training job."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["FORCE_TORCHRUN"] = "1"
    env["NCCL_P2P_DISABLE"] = "1"
    env["NCCL_IB_DISABLE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    return env


def get_output_dir(config_path: Path) -> Optional[Path]:
    """Extract output_dir from a LLaMA-Factory YAML config."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("output_dir:"):
                    return Path(line.split(":", 1)[1].strip())
    except Exception as e:
        log(f"WARNING: could not parse output_dir from {config_path}: {e}")
    return None


def is_job_completed(config_path: Path) -> bool:
    """Check whether a job already has a final model saved."""
    output_dir = get_output_dir(config_path)
    if output_dir is None:
        return False
    marker = output_dir / "model.safetensors"
    return marker.exists()


def run_full_sft_sequential(skip: bool = False, resume: bool = False) -> bool:
    """Run full SFT jobs one after another, each using all GPUs.

    Continues through failures so that later jobs are not blocked by an
    earlier failure.  Returns True only if all jobs succeeded.
    """
    if skip:
        log("Skipping full SFT stage as requested.")
        return True

    log("=" * 70)
    log("STAGE 1: Full SFT (sequential, all 8 GPUs)")
    log("=" * 70)

    failed_jobs = []

    for config_name in FULL_SFT_CONFIGS:
        config_path = CONFIG_DIR / config_name
        if not config_path.exists():
            log(f"ERROR: config not found: {config_path}")
            failed_jobs.append(config_name)
            continue

        job_name = config_name.replace(".yaml", "")

        if resume and is_job_completed(config_path):
            log(f"[RESUME] Skipping completed Full SFT job: {job_name}")
            continue

        log_file = LOG_DIR / f"{job_name}.log"
        env = build_env(ALL_GPUS, is_full=True)
        cmd = ["llamafactory-cli", "train", str(config_path)]

        proc = run_command(
            cmd=cmd,
            env=env,
            cwd=LLAMAFACTORY_DIR,
            log_file=log_file,
            description=f"Full SFT: {job_name}",
            blocking=True,
        )

        if proc.returncode != 0:
            log(f"ERROR: Full SFT job {job_name} failed with exit code {proc.returncode}")
            failed_jobs.append(job_name)

    if failed_jobs:
        log(f"Full SFT stage finished with failures: {', '.join(failed_jobs)}")
        return False

    log("Full SFT stage completed successfully.")
    return True


def run_lora_parallel(skip: bool = False, resume: bool = False) -> bool:
    """Run LoRA jobs in parallel on 4-GPU partitions, taking over GPUs as jobs finish."""
    if skip:
        log("Skipping LoRA stage as requested.")
        return True

    log("=" * 70)
    log("STAGE 2: LoRA SFT (parallel on 4-GPU partitions)")
    log("=" * 70)

    # Queue of pending LoRA configs; skip already-completed ones when resuming.
    pending = []
    for config_name in LORA_CONFIGS:
        config_path = CONFIG_DIR / config_name
        job_name = config_name.replace(".yaml", "")
        if resume and is_job_completed(config_path):
            log(f"[RESUME] Skipping completed LoRA job: {job_name}")
        else:
            pending.append(config_name)

    # Active jobs: list of (subprocess.Popen, gpu_partition_list, job_name, log_file)
    active = []

    # Free partitions initially
    free_partitions = [list(p) for p in GPU_PARTITIONS]

    failed = False

    while pending or active:
        # Launch jobs on any free partitions
        while pending and free_partitions:
            partition = free_partitions.pop(0)
            config_name = pending.pop(0)
            config_path = CONFIG_DIR / config_name

            if not config_path.exists():
                log(f"ERROR: config not found: {config_path}")
                failed = True
                free_partitions.append(partition)
                continue

            job_name = config_name.replace(".yaml", "")
            log_file = LOG_DIR / f"{job_name}.log"
            env = build_env(partition, is_full=False)
            cmd = ["llamafactory-cli", "train", str(config_path)]

            proc = run_command(
                cmd=cmd,
                env=env,
                cwd=LLAMAFACTORY_DIR,
                log_file=log_file,
                description=f"LoRA: {job_name}",
                blocking=False,
            )

            if proc is None:
                log(f"ERROR: failed to launch {job_name}")
                failed = True
                free_partitions.append(partition)
                continue

            active.append((proc, partition, job_name, log_file))

        if not active and not pending:
            break

        # Wait a bit before checking for finished jobs
        time.sleep(5)

        # Check active jobs for completion
        still_active = []
        for proc, partition, job_name, log_file in active:
            ret = proc.poll()
            if ret is None:
                still_active.append((proc, partition, job_name, log_file))
            else:
                with open(log_file, "a") as lf:
                    lf.write(f"[{now_str()}] Finished with exit code {ret}\n")
                log(f"Finished LoRA: {job_name} (exit code {ret})")
                free_partitions.append(partition)
                if ret != 0:
                    log(f"ERROR: LoRA job {job_name} failed with exit code {ret}")
                    failed = True
        active = still_active

    if failed:
        log("LoRA stage completed with failures.")
        return False

    log("LoRA stage completed successfully.")
    return True


def main() -> int:
    global LOG_DIR

    parser = argparse.ArgumentParser(description="Run the Qwen3.5-9B SFT pipeline.")
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help="Skip the sequential full-SFT stage.",
    )
    parser.add_argument(
        "--skip-lora",
        action="store_true",
        help="Skip the parallel LoRA-SFT stage.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(LOG_DIR),
        help="Directory for per-job logs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip jobs whose output_dir already contains a final model.safetensors.",
    )
    args = parser.parse_args()

    LOG_DIR = Path(args.log_dir)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log("SFT Pipeline started")
    log(f"Log directory: {LOG_DIR}")
    if args.resume:
        log("Resume mode: skipping jobs with existing final model checkpoints.")
    log("=" * 70)

    full_ok = run_full_sft_sequential(skip=args.skip_full, resume=args.resume)
    lora_ok = run_lora_parallel(skip=args.skip_lora, resume=args.resume)

    success = full_ok and lora_ok

    log("=" * 70)
    if success:
        log("All requested SFT jobs completed successfully.")
    else:
        log("Pipeline finished with errors. Check individual logs in:")
        log(f"  {LOG_DIR}")
    log("=" * 70)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
