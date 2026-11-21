#!/usr/bin/env python3
"""W&B sweep agent entrypoint for REINFORCE tuning (project: llmoptimizer).

Called by `wandb agent`. Initializes the sweep run, then trains in-process so
metrics land on a single W&B run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import wandb

REPO = Path(__file__).resolve().parents[2]
TRAIN_DIR = REPO / "examples" / "benchmark"
sys.path.insert(0, str(TRAIN_DIR))
sys.path.insert(0, str(REPO))

DEFAULT_BUFFER = REPO / "benchmark_data/rl_rollouts/B_rw_dbest_light16"
DEFAULT_ORDER = REPO / "benchmark_data/rl_task_selection/light_16.json"
DEFAULT_INIT = Path(
    os.environ.get(
        "INIT_ADAPTER",
        "/root/blockdata/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5",
    )
)
DEFAULT_WEIGHT_ROOT = Path(
    os.environ.get("WEIGHT_ROOT", "/root/blockdata/model-weights/wandb_tune")
)
PROJECT = os.environ.get("WANDB_PROJECT", "llmoptimizer")
ENTITY = os.environ.get("WANDB_ENTITY") or None
DEVICE = os.environ.get("CUDA_DEVICE", "cuda:0")
SEED = int(os.environ.get("SEED", "1"))


def main() -> int:
    run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        job_type="sweep",
        config={
            "buffer_root": str(DEFAULT_BUFFER),
            "init_adapter": str(DEFAULT_INIT),
            "dataset_order_file": str(DEFAULT_ORDER),
            "advantage_mode": "per_dataset",
            "dataset_sequential": True,
            "strict_dataset_order": True,
            "checkpoint_at": "16",
            "seed": SEED,
        },
    )
    cfg = wandb.config

    out_root = DEFAULT_WEIGHT_ROOT / f"sweep_{run.sweep_id or 'manual'}" / run.id
    ckpt_root = out_root / "checkpoints"
    final_dir = out_root / "final"
    out_root.mkdir(parents=True, exist_ok=True)

    sys.argv = [
        "train_p5_reinforce.py",
        "--buffer-root",
        str(cfg.get("buffer_root", DEFAULT_BUFFER)),
        "--init-adapter",
        str(cfg.get("init_adapter", DEFAULT_INIT)),
        "--output-dir",
        str(final_dir),
        "--dataset-order-file",
        str(cfg.get("dataset_order_file", DEFAULT_ORDER)),
        "--strict-dataset-order",
        "--advantage-mode",
        str(cfg.get("advantage_mode", "per_dataset")),
        "--dataset-sequential",
        "--checkpoint-at",
        str(cfg.get("checkpoint_at", "16")),
        "--checkpoint-root",
        str(ckpt_root),
        "--device",
        DEVICE,
        "--lr",
        str(cfg.lr),
        "--kl-beta",
        str(cfg.kl_beta),
        "--grad-accum",
        str(int(cfg.grad_accum)),
        "--max-grad-norm",
        str(float(cfg.max_grad_norm)),
        "--seed",
        str(int(cfg.get("seed", SEED))),
    ]

    print("[wandb_sweep] starting in-process train", flush=True)
    from train_p5_reinforce import main as train_main

    rc = int(train_main())
    if rc != 0:
        run.summary["train/status"] = "failed"
        wandb.finish(exit_code=rc)
        return rc

    history = run.history(keys=["train/loss"], pandas=False)
    if history:
        last_loss = history[-1].get("train/loss")
        if last_loss is not None:
            run.summary["train/final_loss"] = float(last_loss)

    ckpt16 = ckpt_root / "ckpt16"
    if (ckpt16 / "adapter_model.safetensors").exists():
        run.summary["checkpoint/ckpt16"] = str(ckpt16)
    run.summary["train/status"] = "ok"
    run.summary["output_dir"] = str(final_dir)
    wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
