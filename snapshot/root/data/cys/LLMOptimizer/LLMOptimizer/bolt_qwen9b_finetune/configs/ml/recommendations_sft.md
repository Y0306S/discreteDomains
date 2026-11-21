# SFT Hyperparameter Recommendations for BOLT Qwen-9B

## What changed between the best LoRA run and the base LoRA run?

| Parameter | base LoRA (`lora_sft`) | best LoRA (`lora_sft_3`) | Impact |
|---|---|---|---|
| `num_train_epochs` | 3 | 5 | **Biggest observed win** (+8% absolute test OK rate) |
| `per_device_train_batch_size` | 1 | 2 | Minor throughput change |
| `gradient_accumulation_steps` | 16 | 8 | Effective batch size stays 16 (1×16 = 2×8) |
| `warmup_*` | `warmup_ratio: 0.05` (~27 steps) | `warmup_steps: 20` | Minor |
| `packing` | not set (default) | `false` | Minor/safety |

Test results on the 63-task held-out test set:
- `lora_sft` (3 epochs): **34 / 63 OK (54.0%)**
- `lora_sft_3` (5 epochs): **39 / 63 OK (61.9%)**
- `full_sft_4` (full FT, 5 epochs): **39 / 63 OK (61.9%)**

Note: `lora_sft_2`, `lora_sft_3`, and `lora_sft_4` share the same YAML config. The variance between them is run-to-run (seed/init), so fixing the seed and adding regularization is recommended for reproducibility.

## Recommended LoRA config

File: `configs/ml/llamafactory_qwen35_9b_base_lora_sft_recommended.yaml`

Key changes vs. `lora_sft_3`:
1. **More epochs with early stopping** (`num_train_epochs: 10`, `load_best_model_at_end: true`). The jump from 3→5 epochs was the single biggest win; 10 epochs with early stopping gives more room to converge without overfitting.
2. **Higher LoRA capacity** (`lora_rank: 32`, `lora_alpha: 64`). The current rank-8 adapter is small for mapping dataset profiles to precise JSON configs; more capacity should improve generalization.
3. **LoRA dropout** (`lora_dropout: 0.05`) + **NEFTune** (`neftune_noise_alpha: 5`) for regularization and run stability.
4. **LoRA+** (`loraplus_lr_ratio: 16.0`) to train adapter matrices at different rates (often improves convergence).
5. **Lower learning rate** (`1.0e-4` vs `2.0e-4`) to match the higher capacity/dropout and reduce variance between runs.
6. **Fixed seed** (`seed: 42`) for reproducibility.
7. **Best-model checkpointing** (`save_total_limit: 3`, `load_best_model_at_end: true`, `metric_for_best_model: eval_loss`).

## Recommended full fine-tune config

File: `configs/ml/llamafactory_qwen35_9b_full_sft_recommended.yaml`

Key changes vs. `full_sft_4`:
1. **More epochs with early stopping** (`num_train_epochs: 10`, `load_best_model_at_end: true`). `full_sft_4` already ties the best LoRA; more epochs may squeeze out additional gains.
2. **Lower learning rate** (`5.0e-6` vs `1.0e-5`) to avoid catastrophic forgetting and let the model settle closer to the optimum.
3. **Weight decay** (`0.01`) for regularization.
4. **Fixed seed** (`seed: 42`) and **best-model checkpointing**.

## Ranking of expected impact (biggest win first)

| Rank | Change | Expected impact | Rationale |
|---|---|---|---|
| 1 | **Train longer (5→10 epochs) with early stopping** | High | 3→5 epochs already gave +8% absolute; longer training is the clearest headroom. |
| 2 | **Task-profile-level train/val/test split** | High | Already in current data; keep it. Prevents leakage and improves real generalization. |
| 3 | **Increase LoRA rank (8→32) + alpha (16→64)** | Medium-High | Current adapter is under-capacity for the structured output task. |
| 4 | **Add LoRA dropout + NEFTune + weight decay (full)** | Medium | Reduces run-to-run variance and overfitting. |
| 5 | **Lower LoRA LR (2e-4 → 1e-4)** | Medium | Better match for larger adapter + dropout. |
| 6 | **Lower full FT LR (1e-5 → 5e-6)** | Medium | Safer convergence for full fine-tune. |
| 7 | **Fix random seed + best-model checkpointing** | Low-Medium | Makes results reproducible and picks the best epoch. |
| 8 | **Stricter output formatting / constrained decoding** | Medium (inference) | Some runs emit reasoning text instead of JSON. A stronger system prompt or JSON schema/constrained decoding at generation time would fix this. |

## Next steps to run

```bash
# LoRA
llamafactory-cli train configs/ml/llamafactory_qwen35_9b_base_lora_sft_recommended.yaml

# Full FT (requires multi-GPU + DeepSpeed Z3)
FORCE_TORCHRUN=1 llamafactory-cli train configs/ml/llamafactory_qwen35_9b_full_sft_recommended.yaml
```

Both old configs remain untouched:
- `configs/ml/llamafactory_qwen35_9b_base_lora_sft_3.yaml`
- `configs/ml/llamafactory_qwen35_9b_full_sft_4.yaml`
