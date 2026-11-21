#!/usr/bin/env python3
"""Generate P5 S1a/S1b LlamaFactory configs for epoch sweep (2.5 vs 7.5) + tight cutoff.

Trains from base Qwen3.5-9B only (no S0 adapter stack).
Checkpoints every SAVE_STEPS for pause/resume (see run_p5_s1_epoch_sweep.sh).
"""

from __future__ import annotations

from pathlib import Path

CUTOFF_LEN = 6016  # max train example = 5958 tokens (Qwen chat template); round up to 128
SAVE_STEPS = 50
SAVE_TOTAL_LIMIT = 3
BASE = Path(__file__).resolve().parents[1] / "configs" / "ml"
MODEL = "/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B"
DATASET_DIR = "/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/llamafactory_data"

JOBS = [
    ("s1a", "p5_s1a_bc", "config-only BC", 2.5, "ep2p5"),
    ("s1a", "p5_s1a_bc", "config-only BC", 7.5, "ep7p5"),
    ("s1b", "p5_s1b_sft", "rationale+JSON", 2.5, "ep2p5"),
    ("s1b", "p5_s1b_sft", "rationale+JSON", 7.5, "ep7p5"),
]


def render(stage: str, dataset: str, note: str, epochs: float, tag: str) -> str:
    out = f"/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_{stage}_{tag}"
    label = "conservative" if epochs <= 2.5 else "standard"
    return f"""### model
# P5 {stage.upper()} ({note}) from base Qwen3.5-9B; {label} ({epochs} epochs); cutoff_len={CUTOFF_LEN}.
model_name_or_path: {MODEL}
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target: all
loraplus_lr_ratio: 16.0

### dataset
dataset_dir: {DATASET_DIR}
dataset: {dataset}
template: qwen3_5_nothink
cutoff_len: {CUTOFF_LEN}
preprocessing_num_workers: 8
dataloader_num_workers: 2
overwrite_cache: true
packing: false

### output
output_dir: {out}
logging_steps: 1
save_steps: {SAVE_STEPS}
plot_loss: true
overwrite_output_dir: false
save_only_model: false
save_total_limit: {SAVE_TOTAL_LIMIT}
report_to: none

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1.25e-4
num_train_epochs: {epochs}
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
gradient_checkpointing: true
ddp_timeout: 180000000
resume_from_checkpoint: null
seed: 42
neftune_noise_alpha: 5
"""


def main() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    for stage, dataset, note, epochs, tag in JOBS:
        name = f"llamafactory_qwen35_9b_lora_p5_{stage}_{tag}.yaml"
        path = BASE / name
        path.write_text(render(stage, dataset, note, epochs, tag))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
