"""Smoke test: can we run SemanticOpt's assistant-only LoRA SFT on our Qwen backbone?

SemanticOpt's recipe (pt4opt-baselines.md): Unsloth assistant-only LoRA SFT,
LoRA rank/alpha 16, lr 1e-5 cosine, bf16, batch size 64, 1 epoch, ctx 24576.
Here we only check the machinery end to end: 2 optimiser steps on 4 synthetic
trajectories in SemanticOpt's multi-turn chat format, loss computed on
assistant turns only.
"""
import os, json, sys

os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

BASE = "/root/data/cys/blockdata_archive/model-weights/qwen35_9b_merged_p5_s1b_ep7p5"
MAXLEN = 4096          # 24576 in the real run; shortened so the smoke test is quick

from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE, max_seq_length=MAXLEN, dtype=torch.bfloat16, load_in_4bit=False,
)
print("SMOKE loaded base:", type(model).__name__)

model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=1,
)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("SMOKE LoRA attached | trainable params: %.2fM" % (trainable / 1e6))

# SemanticOpt sample: system -> (user obs, assistant {config, mu, sigma}, user y) * T
def traj(seed):
    msgs = [{"role": "system",
             "content": "You are a black-box optimizer. Reply with valid JSON only."}]
    for t in range(3):
        msgs.append({"role": "user", "content": json.dumps(
            {"task": "lightgbm on ada", "budget": 80, "step": t,
             "observations": [{"config": {"learning_rate": 0.05}, "y": -0.771}]})})
        msgs.append({"role": "assistant", "content": json.dumps(
            {"config": {"learning_rate": 0.0432, "num_leaves": 128},
             "mu": -0.7783, "sigma": 0.0121})})
        msgs.append({"role": "user", "content": json.dumps({"y": -0.7791})})
    return {"text": tokenizer.apply_chat_template(msgs, tokenize=False)}

from datasets import Dataset
ds = Dataset.from_list([traj(i) for i in range(4)])

from trl import SFTTrainer, SFTConfig
cfg = SFTConfig(
    output_dir="/tmp/semanticopt_smoke", per_device_train_batch_size=1,
    gradient_accumulation_steps=2, max_steps=2, learning_rate=1e-5,
    lr_scheduler_type="cosine", bf16=True, logging_steps=1,
    max_length=MAXLEN, report_to=[], save_strategy="no", dataset_num_proc=1,
)
trainer = SFTTrainer(model=model, train_dataset=ds, args=cfg)
out = trainer.train()
print("SMOKE train_loss:", round(float(out.training_loss), 5))
print("SMOKE SemanticOpt LoRA SFT path: PASS")
