# Peptide warm-start full SFT

This campaign replaces the queued LoRA run at the user's request. No LoRA training started. The user clarified that the two reserved workers mean **two GPU inference slots**, rather than the two remote CPU evaluation allocations.

Train the full Qwen3.5-9B language model from its clean base with LLaMA-Factory and two DeepSpeed ZeRO-3 ranks on GPUs 2 and 3. Vision and the multimodal projector stay frozen, matching the previous full SFT. No adapters or earlier trained full-model weights are loaded.

The scientific recipe matches the original two-GPU peptide SFT: 8,000 examples, ten measured feasible top peptides from each of 800 training tasks; held-out test200 excluded. Prompts describe the reference, objective and peptide constraints, and the target is one discrete JSON peptide sequence. Five epochs, effective batch 16 (two ranks × microbatch 1 × accumulation 8), 2,500 optimizer steps, learning rate 1e-5, cosine schedule, 20 warmup steps, weight decay 0.01. BF16 GPU computation, CPU-offloaded AdamW with FP32 master weights/states, gradient checkpointing, SDPA and optimized Qwen3.5 linear attention. This is full language-model fine-tuning, not LoRA or 8-bit weight training.

## Admission and coexistence

The queue reserves two GPU inference slots as two training ranks. Each rank has a 21 GiB allocator cap and needs at least 23 GiB free before admission. The prior matching two-GPU recipe measured a 15.92 GiB allocation peak per rank; new-run memory use is recorded independently. SemanticOPT's existing nine inference replicas continue, and peptide ORPO/RL retains GPU 0.

Full SFT also needs at least 240 GiB available under the host cgroup limit and 600 GiB of shared checkpoint storage. Local CPUs 24–26 serve rank 0 and 27–29 serve rank 1. SemanticOPT yields these CPU slots before training admission; they return after completion or failure. Released remote ML CPU allocations return to SemanticOPT after a brief shared-file audit. Running ML evaluations are never stopped; the queue also waits for the final ML inference model to release GPU 3 before full training starts.

All new files and checkpoints are under this campaign on shared storage. The old dataset, full SFT run, and superseded LoRA recipe are preserved. Only SemanticOPT CPU scheduling reservations are amended and rehashed; its scientific recipe, model adapters, GPU replica count and fallback policy stay unchanged.

## Verification and recovery

Preparation verifies all source/dataset hashes, provenance, peptide constraints, ranks and train/test separation. It compares all 8,000 supervised token sequences with LLaMA-Factory preprocessing and checks DeepSpeed batch-size resolution and three CPU AdamW updates against PyTorch. `evidence/preflight.json` records the actual passed gates.

Training checks exactly 8,953,803,264 language-model parameters are trainable, no LoRA/vision parameters are trainable, CPU optimizer identity, finite metrics, expected step budget, and actual updates to three language-model matrix samples by step 2. It compares the saved sample tensors with live gathered ZeRO weights at each checkpoint. Early checkpoint at step 2, regular checkpoints every 100 steps, and final checkpoint at step 2500 include both ranks' optimizer/model partitions and RNG states. Two latest checkpoints are retained. Resume accepts only complete two-rank checkpoints.

Completion requires 2,500 steps and five epochs, a complete final training checkpoint, matching final saved model samples, and SHA256 hashes of all final model shards. `status.json` distinguishes queued, preparing, training, error and completed. `training_status.json` and `rank0_status.json`/`rank1_status.json` show actual training progress. `logs/sft.log` holds both ranks' output.

## Authorized stop at epoch 3

The user requested that the current run stop at three epochs first. The effective
cap is 1,500 optimizer steps; see `stop_policy.json` and `effective_recipe.json`.
The running trainer retains its originally loaded five-epoch cosine schedule and
2,500-step progress denominator. Training is not restarted or rewound. A detached
one-shot controller, `code/stop_at_checkpoint.py`, checks progress once per second
and acts only after both original rank callbacks report that checkpoint-1500 was
saved and compared with live gathered weights. It validates the two-rank model,
optimizer and RNG checkpoint, stops only the recorded SFT queue/launcher/rank
processes, links the checkpoint model files into `model/`, hashes final shards,
publishes three-epoch completion evidence, and releases the SFT CPU reservations.
The complete checkpoint remains available for a later requested continuation.

Controller state: `services/stop_at_epoch3.json`; process identity:
`services/stop_at_epoch3_process.json`; log: `logs/stop_at_epoch3.log`. A controller
error is recorded visibly and must not be interpreted as successful completion.
Five tests cover both-rank readiness, complete checkpoint and epoch validation,
missed targets, process identity, and final export with restart-state retention.
The original recipe and SFT configuration remain preserved as the trajectory's
provenance; the separate stop amendment defines the shorter execution cap.
