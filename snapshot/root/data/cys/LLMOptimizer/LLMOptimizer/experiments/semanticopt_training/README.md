SemanticOpt peptide and query training
=====================================

This campaign starts two independent Qwen3.5-9B LoRA SFT adapters from the clean
local base, in order: peptide then query. It does not initialize from ML ORPO/RL.
ML ORPO plus one complete RL cycle is a scheduling dependency only.

Four local LLMOptimizer2 A800s (0,1,2,3), DeepSpeed ZeRO-2, BF16, microbatch one
per GPU, accumulation 16, effective batch 64, one complete epoch. LoRA rank and
alpha 16, zero dropout, LR 1e-5, cosine schedule, seed 42. Existing services are
preserved. Each rank has a 48 GiB PyTorch allocation cap. ZeRO-2 partitions LoRA
optimizer/gradient state; frozen base weights remain replicated. It is not a
four-GPU model-parameter shard. Flash SDPA is enforced, with math/memory-efficient
fallbacks disabled; Qwen3.5 fast linear-attention kernels are required.

The SemanticOpt prompt/loss recipe follows the existing ML implementation:
multi-turn JSON actions plus causal predicted mean/std, original minimize
objective units, four-significant-figure feedback, assistant-only token loss.
Transformers Trainer plus DeepSpeed replaces the ML single-GPU Unsloth runner.
This is a domain adaptation of SemanticOpt, not a reproduction of its synthetic
HEBO dataset. Archived actions are the measured historical LOLBO proposals;
the ten-tree random forest only annotates predictive moments and is fitted to
the preceding selected observations. Frozen-VAE features are offline teacher
inputs only. The LLM outputs discrete sequence strings or alias-plan integers.

Peptide: all 800 fixed training tasks; 50 feasible recorded initialization
examples followed by up to 50 feasible post-initialization distinct proposals,
retained in archive order. This first campaign intentionally filters infeasible
actions and bounds replay size; it does not imitate all 4,053,113 corpus rows.
Similarity is recomputed as 1 - Levenshtein distance / reference length and must
be at least 0.75. Prompts include reference, amino-acid alphabet, maximum length
30, and at most one cysteine. Histories include actual similarity and feasibility.
Two tasks have fewer than 50 feasible post-initialization decisions; their
available decisions are retained. Total 800 conversations and 39,940 decisions.

Query: all 32,790 eligible distinct uncensored plans from 788 train800 tasks,
chronological windows of at most 32 observations. Up to five initial points
per window are context. A one-point trailing fragment is context-only and has
no supervised decision. The 12 corpus exclusions have fewer than two valid
distinct uncensored plans and are explicitly recorded. There are 1,470
conversations and 25,844 decisions. SQL identity, source checksums, canonical
codec round trips, exact decoded SQL equivalence, operators, tables, and aliases
are checked without executing queries. All 200 test tasks are excluded.

Both corpora contain successful measured observations only. Failure-probability
targets are not invented. The shared inference helper validates two candidates
and selects one with success-weighted expected improvement. Deployment and
benchmark evaluation are separate from this training campaign.

Files: configs/*.json, code/prepare.py, code/train.py, code/protocol.py.
Data and preflights: peptide/data/, query/data/. Training progress:
peptide/training_status.json and query/training_status.json. Logs: logs/.
The queue is code/training_queue.py; queue_status.json records its live state. It waits
for ML RL completion, the four-GPU full-model update test, and at least 54 GiB
free per GPU. It stops on a training error. Checkpoints save after steps 1 and
2, every 10 updates, and at completion, retaining two intermediate checkpoints.
The final adapter is compared tensor-for-tensor with the trained model.
