#!/usr/bin/env python3
"""
Run DL Qwen3.5-9B SFT configs on 4 GPUs.

- Full SFT: sequential, all 4 GPUs per job (DDP).
- LoRA SFT: up to 4 jobs in parallel, one single-GPU job per GPU.

Usage:
    python run_sft_pipeline_dl.py
    python run_sft_pipeline_dl.py --resume
    python run_sft_pipeline_dl.py --skip-full
    python run_sft_pipeline_dl.py --skip-lora
    python run_sft_pipeline_dl.py --dry-run
    python run_sft_pipeline_dl.py --dry-run --resume
    python run_sft_pipeline_dl.py --smoke-test
    python run_sft_pipeline_dl.py --smoke-then-run --resume
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
BLOCKDATA_LLM = Path("/root/data/cys/blockdata_archive/LLMOptimizer")
LLAMAFACTORY_DIR = Path("/root/LLMOptimizer/LlamaFactory")
CONFIG_DIR = BASE_DIR / "configs" / "dl"
DEFAULT_LOG_DIR = BLOCKDATA_LLM / "logs" / "bolt_qwen9b_finetune" / "dl"
LOG_DIR = DEFAULT_LOG_DIR
SMOKE_OUTPUT_ROOT = Path("/root/data/cys/blockdata_archive/model-weights/_smoke_test")

GPUS = [0, 1, 2, 3]

FULL_SFT_CONFIGS = [
    "llamafactory_qwen35_9b_full_sft_4_dl.yaml",
    "llamafactory_qwen35_9b_full_sft_4_dl_rec.yaml",
]

LORA_CONFIGS = [
    "llamafactory_qwen35_9b_base_lora_sft_dl.yaml",
    "llamafactory_qwen35_9b_base_lora_sft_2_dl.yaml",
    "llamafactory_qwen35_9b_base_lora_sft_3_dl.yaml",
    "llamafactory_qwen35_9b_base_lora_sft_4_dl.yaml",
    "llamafactory_qwen35_9b_base_lora_sft_5_dl.yaml",
    "llamafactory_qwen35_9b_base_lora_sft_6_dl.yaml",
    "llamafactory_qwen35_9b_base_lora_sft_7_dl.yaml",
]


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now_str()}] {message}", flush=True)


def get_output_dir(config_path: Path) -> Path | None:
    try:
        with config_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("output_dir:"):
                    return Path(line.split(":", 1)[1].strip())
    except OSError as exc:
        log(f"WARNING: could not parse output_dir from {config_path}: {exc}")
    return None


def is_lora_config(config_path: Path) -> bool:
    try:
        with config_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip() == "finetuning_type: lora":
                    return True
    except OSError:
        return False
    return False


def is_job_completed(config_path: Path) -> bool:
    output_dir = get_output_dir(config_path)
    if output_dir is None:
        return False
    marker = (
        output_dir / "adapter_model.safetensors"
        if is_lora_config(config_path)
        else output_dir / "model.safetensors"
    )
    return marker.exists()


def build_env(*, gpu_id: int | None = None, distributed: bool = False) -> dict:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    if distributed:
        env["FORCE_TORCHRUN"] = "1"
        env["NCCL_P2P_DISABLE"] = "1"
        env["NCCL_IB_DISABLE"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in GPUS)
    else:
        env.pop("FORCE_TORCHRUN", None)
        if gpu_id is None:
            raise ValueError("gpu_id is required for single-GPU jobs")
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    return env


def run_job(
    config_name: str,
    *,
    config_path: Path | None = None,
    gpu_id: int | None = None,
    distributed: bool = False,
    log_suffix: str = "",
) -> int:
    config_path = config_path or (CONFIG_DIR / config_name)
    if not config_path.exists():
        log(f"ERROR: config not found: {config_path}")
        return 1

    job_name = config_name.replace(".yaml", "")
    log_file = LOG_DIR / f"{job_name}{log_suffix}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    cmd = ["llamafactory-cli", "train", str(config_path)]
    env = build_env(gpu_id=gpu_id, distributed=distributed)

    mode = f"GPU {gpu_id}" if gpu_id is not None else f"DDP {env['CUDA_VISIBLE_DEVICES']}"
    log(f"Starting {job_name}")
    log(f"  config: {config_path}")
    log(f"  mode  : {mode}")
    log(f"  log   : {log_file}")

    with log_file.open("w", encoding="utf-8") as lf:
        lf.write(f"[{now_str()}] Command: {' '.join(cmd)}\n")
        lf.write(f"[{now_str()}] Mode: {mode}\n")
        lf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=LLAMAFACTORY_DIR,
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
        )
        proc.wait()
        lf.write(f"[{now_str()}] Finished with exit code {proc.returncode}\n")

    if proc.returncode == 0:
        log(f"Completed {job_name} ({mode})")
    else:
        log(f"ERROR: {job_name} failed with exit code {proc.returncode} ({mode})")
    return proc.returncode


def cleanup_intermediate_checkpoints(config_path: Path) -> None:
    """Remove checkpoint-* dirs after a successful job; keep final model/adapter only."""
    output_dir = get_output_dir(config_path)
    if output_dir is None or not output_dir.exists():
        return
    for path in sorted(output_dir.glob("checkpoint-*")):
        if path.is_dir():
            log(f"Removing intermediate checkpoint: {path}")
            shutil.rmtree(path)


def plan_pending_jobs(config_names: list[str], resume: bool) -> list[tuple[str, str]]:
    """Return (config_name, status) where status is 'run' or 'skip'."""
    planned: list[tuple[str, str]] = []
    for config_name in config_names:
        config_path = CONFIG_DIR / config_name
        if resume and is_job_completed(config_path):
            planned.append((config_name, "skip"))
        else:
            planned.append((config_name, "run"))
    return planned


def plan_lora_batches(config_names: list[str], resume: bool) -> list[list[tuple[str, int]]]:
    """Group pending LoRA jobs into batches of up to len(GPUS), each with a GPU id."""
    pending = [name for name, status in plan_pending_jobs(config_names, resume) if status == "run"]
    batches: list[list[tuple[str, int]]] = []
    for batch_start in range(0, len(pending), len(GPUS)):
        batch = pending[batch_start : batch_start + len(GPUS)]
        batches.append([(config_name, GPUS[idx]) for idx, config_name in enumerate(batch)])
    return batches


def print_dry_run(*, resume: bool, skip_full: bool, skip_lora: bool) -> None:
    log("DRY RUN — no training commands will be launched")
    log("=" * 70)

    if not skip_full:
        log("STAGE 1: Full SFT (sequential, 4-GPU DDP, one job at a time)")
        for config_name, status in plan_pending_jobs(FULL_SFT_CONFIGS, resume):
            job_name = config_name.replace(".yaml", "")
            config_path = CONFIG_DIR / config_name
            output_dir = get_output_dir(config_path)
            if status == "skip":
                log(f"  [SKIP] {job_name}  (checkpoint exists: {output_dir})")
            else:
                log(f"  [RUN ] {job_name}")
                log(f"         config : {config_path}")
                log(f"         GPUs   : {','.join(str(g) for g in GPUS)} (DDP, FORCE_TORCHRUN=1)")
                log(f"         output : {output_dir}")
        log("")

    if not skip_lora:
        log("STAGE 2: LoRA SFT (parallel batches, 1 GPU per job)")
        log("Handover rule: start batch N+1 only after every job in batch N exits.")
        skipped = [n for n, s in plan_pending_jobs(LORA_CONFIGS, resume) if s == "skip"]
        if skipped:
            log(f"Skipped ({len(skipped)}): {', '.join(n.replace('.yaml', '') for n in skipped)}")
        batches = plan_lora_batches(LORA_CONFIGS, resume)
        if not batches:
            log("No pending LoRA jobs.")
        for batch_idx, batch in enumerate(batches, start=1):
            log(f"  Batch {batch_idx}/{len(batches)} — launch together, wait for all to finish:")
            for config_name, gpu_id in batch:
                job_name = config_name.replace(".yaml", "")
                config_path = CONFIG_DIR / config_name
                output_dir = get_output_dir(config_path)
                log(f"    GPU {gpu_id} -> {job_name}")
                log(f"              config : {config_path}")
                log(f"              output : {output_dir}")
                log(f"              env    : CUDA_VISIBLE_DEVICES={gpu_id}, FORCE_TORCHRUN unset")
            if batch_idx < len(batches):
                log(f"  >> barrier: block until all {len(batch)} jobs above complete, then start batch {batch_idx + 1}")
            log("")

    log("=" * 70)
    log("DRY RUN complete")


def make_smoke_config(config_name: str, smoke_dir: Path) -> Path:
    """Write a 1-step smoke-test config; does not modify the original yaml."""
    source = CONFIG_DIR / config_name
    job_name = config_name.replace(".yaml", "")
    dest = smoke_dir / config_name
    override_keys = {
        "max_steps",
        "num_train_epochs",
        "save_steps",
        "overwrite_output_dir",
        "output_dir",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "cutoff_len",
    }
    out: list[str] = []

    for line in source.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and ":" in stripped:
            key = stripped.split(":", 1)[0].strip()
            if key in override_keys:
                continue
        out.append(line)

    out.extend(
        [
            "",
            "output_dir: " + str(SMOKE_OUTPUT_ROOT / job_name),
            "cutoff_len: 512",
            "per_device_train_batch_size: 1",
            "gradient_accumulation_steps: 1",
            "max_steps: 1",
            "num_train_epochs: 1.0",
            "save_steps: 9999",
            "overwrite_output_dir: true",
        ]
    )
    dest.write_text("\n".join(out) + "\n", encoding="utf-8")
    return dest


def run_smoke_tests(*, skip_full: bool, skip_lora: bool) -> bool:
    log("SMOKE TEST — 1 training step per config")
    log("=" * 70)
    if SMOKE_OUTPUT_ROOT.exists():
        shutil.rmtree(SMOKE_OUTPUT_ROOT)
    SMOKE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    ok = True
    with tempfile.TemporaryDirectory(prefix="dl_smoke_configs_") as tmp:
        smoke_dir = Path(tmp)
        if not skip_full:
            log("Smoke: Full SFT (sequential)")
            for config_name in FULL_SFT_CONFIGS:
                smoke_config = make_smoke_config(config_name, smoke_dir)
                job_name = config_name.replace(".yaml", "")
                log(f"  1-step test: {job_name}")
                if (
                    run_job(
                        config_name,
                        config_path=smoke_config,
                        distributed=True,
                        log_suffix="_smoke",
                    )
                    != 0
                ):
                    ok = False

        if not skip_lora:
            batches = plan_lora_batches(LORA_CONFIGS, resume=False)
            for batch_idx, batch in enumerate(batches, start=1):
                log(f"Smoke LoRA batch {batch_idx}/{len(batches)}")
                with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                    futures = []
                    for config_name, gpu_id in batch:
                        smoke_config = make_smoke_config(config_name, smoke_dir)
                        futures.append(
                            pool.submit(
                                run_job,
                                config_name,
                                config_path=smoke_config,
                                gpu_id=gpu_id,
                                distributed=False,
                                log_suffix="_smoke",
                            )
                        )
                    wait(futures)
                    for future in futures:
                        if future.result() != 0:
                            ok = False
                log(f"Smoke batch {batch_idx}/{len(batches)} complete")

    if ok:
        log("SMOKE TEST passed for all configs")
    else:
        log("SMOKE TEST failed — full pipeline will not start")
    log("=" * 70)
    return ok


def run_configs(config_names: list[str], resume: bool, *, distributed: bool) -> bool:
    failed: list[str] = []
    for config_name in config_names:
        config_path = CONFIG_DIR / config_name
        job_name = config_name.replace(".yaml", "")
        if resume and is_job_completed(config_path):
            log(f"[RESUME] Skipping completed job: {job_name}")
            continue
        if run_job(config_name, distributed=distributed) != 0:
            failed.append(job_name)
        else:
            cleanup_intermediate_checkpoints(config_path)
    if failed:
        log(f"Failed jobs: {', '.join(failed)}")
        return False
    return True


def run_lora_parallel(config_names: list[str], resume: bool) -> bool:
    batches = plan_lora_batches(config_names, resume)
    if not batches:
        log("No pending LoRA jobs.")
        return True

    failed: list[str] = []
    for batch_idx, batch in enumerate(batches, start=1):
        names = ", ".join(config_name.replace(".yaml", "") for config_name, _ in batch)
        log(f"LoRA batch {batch_idx}/{len(batches)}: {names}")
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            futures = [
                pool.submit(run_job, config_name, gpu_id=gpu_id, distributed=False)
                for config_name, gpu_id in batch
            ]
            # Barrier: do not leave this block until every job in the batch has exited.
            wait(futures)
            for (config_name, _), future in zip(batch, futures):
                if future.result() != 0:
                    failed.append(config_name.replace(".yaml", ""))
                else:
                    cleanup_intermediate_checkpoints(CONFIG_DIR / config_name)

        log(
            f"Batch {batch_idx}/{len(batches)} complete — "
            f"all {len(batch)} job(s) finished; "
            + ("starting next batch." if batch_idx < len(batches) else "no more batches.")
        )

    if failed:
        log(f"Failed jobs: {', '.join(failed)}")
        return False
    return True


def main() -> int:
    global LOG_DIR

    parser = argparse.ArgumentParser(
        description="Run DL SFT configs: full SFT on 4 GPUs, LoRA one job per GPU."
    )
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--skip-lora", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip jobs whose output_dir already has a final checkpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print execution plan (batches, GPU mapping, barriers) without training.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 1 training step per config as a smoke test, then exit.",
    )
    parser.add_argument(
        "--smoke-then-run",
        action="store_true",
        help="Dry-run plan, smoke-test 1 step per config, then run full pipeline if smoke passes.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(LOG_DIR),
        help="Directory for per-job logs.",
    )
    args = parser.parse_args()

    LOG_DIR = Path(args.log_dir)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print_dry_run(resume=args.resume, skip_full=args.skip_full, skip_lora=args.skip_lora)
        return 0

    if args.smoke_test:
        return 0 if run_smoke_tests(skip_full=args.skip_full, skip_lora=args.skip_lora) else 1

    if args.smoke_then_run:
        print_dry_run(resume=args.resume, skip_full=args.skip_full, skip_lora=args.skip_lora)
        if not run_smoke_tests(skip_full=args.skip_full, skip_lora=args.skip_lora):
            return 1
        log("Smoke tests passed — starting full pipeline")

    log("=" * 70)
    log("DL SFT pipeline started")
    log("  Full SFT: sequential, 4-GPU DDP")
    log("  LoRA SFT: parallel, 1 GPU per job (up to 4 at once)")
    log(f"Config dir: {CONFIG_DIR}")
    log(f"Log dir   : {LOG_DIR}")
    log("=" * 70)

    ok = True
    if not args.skip_full:
        log("STAGE 1: Full SFT")
        ok = run_configs(FULL_SFT_CONFIGS, resume=args.resume, distributed=True) and ok
    if not args.skip_lora:
        log("STAGE 2: LoRA SFT (parallel)")
        ok = run_lora_parallel(LORA_CONFIGS, resume=args.resume) and ok

    log("=" * 70)
    log("DL SFT pipeline finished successfully." if ok else "DL SFT pipeline finished with errors.")
    log("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
