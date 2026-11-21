# Peptide warm-start LoRA SFT

Fresh Qwen3.5-9B base-model adapter, separate from the active peptide ORPO → RL campaign.

- LLaMA-Factory SFT; LoRA rank 32, alpha 64, dropout 0.05, all language linear layers. Base weights, vision, embeddings and LM head remain frozen; the trainer checks every trainable parameter name.
- Reuse the audited BOLT top-ten dataset: 8,000 examples, ten distinct feasible measured peptides from each of 800 training tasks. The fixed 200 evaluation tasks are excluded. No new oracle calls.
- The task-only warm-start prompt includes the reference, objective, alphabet, length ≤30, cysteine ≤1, and reference similarity ≥0.75. Output is exactly one JSON sequence. This retains the previous warm-start prompt; it is separate from the Recent50 ORPO/RL prompt.
- Five epochs, effective batch 16 (microbatch 1 × accumulation 16), 2,500 optimizer steps, LR 1e-5, cosine decay, 20 warmup steps. Preserve the previous data, epochs and effective batch; replace full-model 8bit AdamW with FP32 AdamW on the small adapter. Base computation is BF16; gradient checkpointing and SDPA with FlashSDPA enabled, plus optimized Qwen3.5 linear attention.

## Queue and resources

`code/queue_training.py` waits for the remaining ML extra_trees evaluation on LLMOptimizer4 CPUs 0–2 to finish, then rechecks the shared training inputs there against the full audit and tokenizer preflight already completed on the GPU host. The other remaining ML worker (CPUs 3–5) becomes available to SemanticOPT after its existing evaluation finishes. Active ML jobs are never stopped.

After preparation, remote CPUs 0–2 also return to SemanticOPT. Training requires local GPU 3 with at least 36 GiB free, both remaining ML evaluations completed, their final inference model exited, at least 32 GiB host-memory headroom, and SemanticOPT workers yielding local CPUs 24–26. GPU allocation is capped at 32 GiB. SemanticOPT keeps its two existing replicas on GPU 3. The peptide ORPO/RL campaign uses GPU 0.

The queue updates only SemanticOPT CPU scheduling reservations and records the new scheduling hash in its identity. Its dataset, model adapter, scientific recipe and fallback policy are unchanged. Reservations are released when SFT ends or fails.

## Audit and recovery

`evidence/data_audit.json` verifies dataset/source hashes, target feasibility, provenance, per-task ranks and the held-out split. `evidence/preflight.json` verifies all 8,000 assistant-only token sequences against LLaMA-Factory preprocessing, with no truncation or dropping.

`training_status.json` records actual progress, trainable parameters, attention, finite metrics, memory use and evidence of LoRA updates by step 2. Checkpoints at steps 1, 2 and every 100 steps include optimizer/scheduler/RNG state. Latest two are retained; the final adapter is saved separately. Every saved adapter is compared tensor-for-tensor with the live model. Completion requires all 2,500 steps and five epochs. Resume selects only complete checkpoints; an output without a complete checkpoint is rejected.

Launch the durable queue with the CPU Python in `services/queue_process.json`; inspect `status.json`, `logs/queue.log`, and `logs/sft.log`. Training is queued until resource gates pass; preparing the recipe does not imply training has started.
