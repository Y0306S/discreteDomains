#!/usr/bin/env python3
"""Run ML Qwen3.5-9B SFT configs on 4 GPUs.

- Full SFT: sequential, all 4 GPUs per job (DeepSpeed ZeRO-3).
- LoRA SFT: up to 4 jobs in parallel, one single-GPU job per GPU.

Outputs go to /root/data/cys/blockdata_archive/model-weights/ with save_only_model (no checkpoints).

Usage:
    python run_sft_pipeline_ml.py --resume
    python run_sft_pipeline_ml.py --skip-full --resume   # LoRA wave only
    python run_sft_pipeline_ml.py --skip-lora --resume   # full SFT only
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
CONDA_BIN = Path("/root/miniconda3/envs/llamafactory_qwen35/bin")
LLAMAFACTORY_CLI = CONDA_BIN / "llamafactory-cli"
CONFIG_DIR = BASE_DIR / "configs" / "ml"
DEFAULT_LOG_DIR = BLOCKDATA_LLM / "logs" / "bolt_qwen9b_finetune" / "ml"
LOG_DIR = DEFAULT_LOG_DIR
SMOKE_OUTPUT_ROOT = Path("/root/data/cys/blockdata_archive/model-weights/_smoke_test_ml")

GPUS = [0, 1, 2, 3]

FULL_SFT_CONFIGS = [
    "llamafactory_qwen35_9b_full_sft_4.yaml",
    "llamafactory_qwen35_9b_full_sft_5.yaml",
    "llamafactory_qwen35_9b_full_sft_6.yaml",
]

LORA_CONFIGS = [
    "llamafactory_qwen35_9b_base_lora_sft_recommended.yaml",
    "llamafactory_qwen35_9b_lora_recommended2.yaml",
    "llamafactory_qwen35_9b_lora_recommended4.yaml",
    "llamafactory_qwen35_9b_lora_recommended5.yaml",
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
    # LlamaFactory DDP invokes `torchrun`; ensure conda bin is on PATH.
    env["PATH"] = f"{CONDA_BIN}{os.pathsep}{env.get('PATH', '')}"

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

    cmd = [str(LLAMAFACTORY_CLI), "train", str(config_path)]
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
    output_dir = get_output_dir(config_path)
    if output_dir is None or not output_dir.exists():
        return
    for path in sorted(output_dir.glob("checkpoint-*")):
        if path.is_dir():
            log(f"Removing intermediate checkpoint: {path}")
            shutil.rmtree(path)


def plan_pending_jobs(config_names: list[str], resume: bool) -> list[tuple[str, str]]:
    planned: list[tuple[str, str]] = []
    for config_name in config_names:
        config_path = CONFIG_DIR / config_name
        if resume and is_job_completed(config_path):
            planned.append((config_name, "skip"))
        else:
            planned.append((config_name, "run"))
    return planned


def plan_lora_batches(config_names: list[str], resume: bool) -> list[list[tuple[str, int]]]:
    pending = [name for name, status in plan_pending_jobs(config_names, resume) if status == "run"]
    batches: list[list[tuple[str, int]]] = []
    for batch_start in range(0, len(pending), len(GPUS)):
        batch = pending[batch_start : batch_start + len(GPUS)]
        batches.append([(config_name, GPUS[idx]) for idx, config_name in enumerate(batch)])
    return batches


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

    parser = argparse.ArgumentParser(description="Run ML SFT configs on 4 GPUs.")
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--skip-lora", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-dir", type=str, default=str(LOG_DIR))
    args = parser.parse_args()

    LOG_DIR = Path(args.log_dir)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log("ML SFT pipeline started")
    log("  Full SFT: sequential, 4-GPU DDP")
    log("  LoRA SFT: parallel, 1 GPU per job (up to 4 at once)")
    log(f"  Weights  : /root/data/cys/blockdata_archive/model-weights/")
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
    log("ML SFT pipeline finished successfully." if ok else "ML SFT pipeline finished with errors.")
    log("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""Run ML Qwen3.5-9B SFT configs on 4 GPUs.

- Full SFT: sequential, all 4 GPUs per job (DeepSpeed ZeRO-3).
- LoRA SFT: up to 4 jobs in parallel, one single-GPU job per GPU.

Outputs go to /root/data/cys/blockdata_archive/model-weights/ with save_only_model (no checkpoints).

Usage:
    python run_sft_pipeline_ml.py --resume
    python run_sft_pipeline_ml.py --skip-full --resume   # LoRA wave only
    python run_sft_pipeline_ml.py --skip-lora --resume   # full SFT only
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
CONDA_BIN = Path("/root/miniconda3/envs/llamafactory_qwen35/bin")
LLAMAFACTORY_CLI = CONDA_BIN / "llamafactory-cli"
CONFIG_DIR = BASE_DIR / "configs" / "ml"
DEFAULT_LOG_DIR = BLOCKDATA_LLM / "logs" / "bolt_qwen9b_finetune" / "ml"
LOG_DIR = DEFAULT_LOG_DIR
SMOKE_OUTPUT_ROOT = Path("/root/data/cys/blockdata_archive/model-weights/_smoke_test_ml")

GPUS = [0, 1, 2, 3]

FULL_SFT_CONFIGS = [
    "llamafactory_qwen35_9b_full_sft_4.yaml",
    "llamafactory_qwen35_9b_full_sft_5.yaml",
    "llamafactory_qwen35_9b_full_sft_6.yaml",
]

LORA_CONFIGS = [
    "llamafactory_qwen35_9b_base_lora_sft_recommended.yaml",
    "llamafactory_qwen35_9b_lora_recommended2.yaml",
    "llamafactory_qwen35_9b_lora_recommended4.yaml",
    "llamafactory_qwen35_9b_lora_recommended5.yaml",
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
    # LlamaFactory DDP invokes `torchrun`; ensure conda bin is on PATH.
    env["PATH"] = f"{CONDA_BIN}{os.pathsep}{env.get('PATH', '')}"

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

    cmd = [str(LLAMAFACTORY_CLI), "train", str(config_path)]
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
    output_dir = get_output_dir(config_path)
    if output_dir is None or not output_dir.exists():
        return
    for path in sorted(output_dir.glob("checkpoint-*")):
        if path.is_dir():
            log(f"Removing intermediate checkpoint: {path}")
            shutil.rmtree(path)


def plan_pending_jobs(config_names: list[str], resume: bool) -> list[tuple[str, str]]:
    planned: list[tuple[str, str]] = []
    for config_name in config_names:
        config_path = CONFIG_DIR / config_name
        if resume and is_job_completed(config_path):
            planned.append((config_name, "skip"))
        else:
            planned.append((config_name, "run"))
    return planned


def plan_lora_batches(config_names: list[str], resume: bool) -> list[list[tuple[str, int]]]:
    pending = [name for name, status in plan_pending_jobs(config_names, resume) if status == "run"]
    batches: list[list[tuple[str, int]]] = []
    for batch_start in range(0, len(pending), len(GPUS)):
        batch = pending[batch_start : batch_start + len(GPUS)]
        batches.append([(config_name, GPUS[idx]) for idx, config_name in enumerate(batch)])
    return batches


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

    parser = argparse.ArgumentParser(description="Run ML SFT configs on 4 GPUs.")
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--skip-lora", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-dir", type=str, default=str(LOG_DIR))
    args = parser.parse_args()

    LOG_DIR = Path(args.log_dir)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log("ML SFT pipeline started")
    log("  Full SFT: sequential, 4-GPU DDP")
    log("  LoRA SFT: parallel, 1 GPU per job (up to 4 at once)")
    log(f"  Weights  : /root/data/cys/blockdata_archive/model-weights/")
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
    log("ML SFT pipeline finished successfully." if ok else "ML SFT pipeline finished with errors.")
    log("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""Run ML Qwen3.5-9B SFT configs on 4 GPUs.

- Full SFT: sequential, all 4 GPUs per job (DeepSpeed ZeRO-3).
- LoRA SFT: up to 4 jobs in parallel, one single-GPU job per GPU.

Outputs go to /root/data/cys/blockdata_archive/model-weights/ with save_only_model (no checkpoints).

Usage:
    python run_sft_pipeline_ml.py --resume
    python run_sft_pipeline_ml.py --skip-full --resume   # LoRA wave only
    python run_sft_pipeline_ml.py --skip-lora --resume   # full SFT only
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
CONDA_BIN = Path("/root/miniconda3/envs/llamafactory_qwen35/bin")
LLAMAFACTORY_CLI = CONDA_BIN / "llamafactory-cli"
CONFIG_DIR = BASE_DIR / "configs" / "ml"
DEFAULT_LOG_DIR = BLOCKDATA_LLM / "logs" / "bolt_qwen9b_finetune" / "ml"
LOG_DIR = DEFAULT_LOG_DIR
SMOKE_OUTPUT_ROOT = Path("/root/data/cys/blockdata_archive/model-weights/_smoke_test_ml")

GPUS = [0, 1, 2, 3]

FULL_SFT_CONFIGS = [
    "llamafactory_qwen35_9b_full_sft_4.yaml",
    "llamafactory_qwen35_9b_full_sft_5.yaml",
    "llamafactory_qwen35_9b_full_sft_6.yaml",
]

LORA_CONFIGS = [
    "llamafactory_qwen35_9b_base_lora_sft_recommended.yaml",
    "llamafactory_qwen35_9b_lora_recommended2.yaml",
    "llamafactory_qwen35_9b_lora_recommended4.yaml",
    "llamafactory_qwen35_9b_lora_recommended5.yaml",
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
    # LlamaFactory DDP invokes `torchrun`; ensure conda bin is on PATH.
    env["PATH"] = f"{CONDA_BIN}{os.pathsep}{env.get('PATH', '')}"

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

    cmd = [str(LLAMAFACTORY_CLI), "train", str(config_path)]
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
    output_dir = get_output_dir(config_path)
    if output_dir is None or not output_dir.exists():
        return
    for path in sorted(output_dir.glob("checkpoint-*")):
        if path.is_dir():
            log(f"Removing intermediate checkpoint: {path}")
            shutil.rmtree(path)


def plan_pending_jobs(config_names: list[str], resume: bool) -> list[tuple[str, str]]:
    planned: list[tuple[str, str]] = []
    for config_name in config_names:
        config_path = CONFIG_DIR / config_name
        if resume and is_job_completed(config_path):
            planned.append((config_name, "skip"))
        else:
            planned.append((config_name, "run"))
    return planned


def plan_lora_batches(config_names: list[str], resume: bool) -> list[list[tuple[str, int]]]:
    pending = [name for name, status in plan_pending_jobs(config_names, resume) if status == "run"]
    batches: list[list[tuple[str, int]]] = []
    for batch_start in range(0, len(pending), len(GPUS)):
        batch = pending[batch_start : batch_start + len(GPUS)]
        batches.append([(config_name, GPUS[idx]) for idx, config_name in enumerate(batch)])
    return batches


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

    parser = argparse.ArgumentParser(description="Run ML SFT configs on 4 GPUs.")
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--skip-lora", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-dir", type=str, default=str(LOG_DIR))
    args = parser.parse_args()

    LOG_DIR = Path(args.log_dir)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log("ML SFT pipeline started")
    log("  Full SFT: sequential, 4-GPU DDP")
    log("  LoRA SFT: parallel, 1 GPU per job (up to 4 at once)")
    log(f"  Weights  : /root/data/cys/blockdata_archive/model-weights/")
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
    log("ML SFT pipeline finished successfully." if ok else "ML SFT pipeline finished with errors.")
    log("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""Run ML Qwen3.5-9B SFT configs on 4 GPUs.

- Full SFT: sequential, all 4 GPUs per job (DeepSpeed ZeRO-3).
- LoRA SFT: up to 4 jobs in parallel, one single-GPU job per GPU.

Outputs go to /root/data/cys/blockdata_archive/model-weights/ with save_only_model (no checkpoints).

Usage:
    python run_sft_pipeline_ml.py --resume
    python run_sft_pipeline_ml.py --skip-full --resume   # LoRA wave only
    python run_sft_pipeline_ml.py --skip-lora --resume   # full SFT only
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
CONDA_BIN = Path("/root/miniconda3/envs/llamafactory_qwen35/bin")
LLAMAFACTORY_CLI = CONDA_BIN / "llamafactory-cli"
CONFIG_DIR = BASE_DIR / "configs" / "ml"
DEFAULT_LOG_DIR = BLOCKDATA_LLM / "logs" / "bolt_qwen9b_finetune" / "ml"
LOG_DIR = DEFAULT_LOG_DIR
SMOKE_OUTPUT_ROOT = Path("/root/data/cys/blockdata_archive/model-weights/_smoke_test_ml")

GPUS = [0, 1, 2, 3]

FULL_SFT_CONFIGS = [
    "llamafactory_qwen35_9b_full_sft_4.yaml",
    "llamafactory_qwen35_9b_full_sft_5.yaml",
    "llamafactory_qwen35_9b_full_sft_6.yaml",
]

LORA_CONFIGS = [
    "llamafactory_qwen35_9b_base_lora_sft_recommended.yaml",
    "llamafactory_qwen35_9b_lora_recommended2.yaml",
    "llamafactory_qwen35_9b_lora_recommended4.yaml",
    "llamafactory_qwen35_9b_lora_recommended5.yaml",
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
    # LlamaFactory DDP invokes `torchrun`; ensure conda bin is on PATH.
    env["PATH"] = f"{CONDA_BIN}{os.pathsep}{env.get('PATH', '')}"

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

    cmd = [str(LLAMAFACTORY_CLI), "train", str(config_path)]
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
    output_dir = get_output_dir(config_path)
    if output_dir is None or not output_dir.exists():
        return
    for path in sorted(output_dir.glob("checkpoint-*")):
        if path.is_dir():
            log(f"Removing intermediate checkpoint: {path}")
            shutil.rmtree(path)


def plan_pending_jobs(config_names: list[str], resume: bool) -> list[tuple[str, str]]:
    planned: list[tuple[str, str]] = []
    for config_name in config_names:
        config_path = CONFIG_DIR / config_name
        if resume and is_job_completed(config_path):
            planned.append((config_name, "skip"))
        else:
            planned.append((config_name, "run"))
    return planned


def plan_lora_batches(config_names: list[str], resume: bool) -> list[list[tuple[str, int]]]:
    pending = [name for name, status in plan_pending_jobs(config_names, resume) if status == "run"]
    batches: list[list[tuple[str, int]]] = []
    for batch_start in range(0, len(pending), len(GPUS)):
        batch = pending[batch_start : batch_start + len(GPUS)]
        batches.append([(config_name, GPUS[idx]) for idx, config_name in enumerate(batch)])
    return batches


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

    parser = argparse.ArgumentParser(description="Run ML SFT configs on 4 GPUs.")
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--skip-lora", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-dir", type=str, default=str(LOG_DIR))
    args = parser.parse_args()

    LOG_DIR = Path(args.log_dir)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log("ML SFT pipeline started")
    log("  Full SFT: sequential, 4-GPU DDP")
    log("  LoRA SFT: parallel, 1 GPU per job (up to 4 at once)")
    log(f"  Weights  : /root/data/cys/blockdata_archive/model-weights/")
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
    log("ML SFT pipeline finished successfully." if ok else "ML SFT pipeline finished with errors.")
    log("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
