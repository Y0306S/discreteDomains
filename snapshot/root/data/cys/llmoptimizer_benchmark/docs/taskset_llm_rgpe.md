# TaskSet LLM-RGPE (and the TaskSet ckpt32 pipeline)

The ML LLM-RGPE recipe (ckpt32 + RGPE, `HPUCB_rgpe_p07p03_retry5_llm8_cool20`)
ported to the TaskSet CASH domain (5 optimizer families, train838 / test210 split
`4cf49ac3…`). Built 2026-10-06. **Nothing has been trained or launched.**

## Method

| Part | TaskSet behaviour | Code |
|---|---|---|
| Warm start (trials 1-3) | 3 presets per task from the warm-start adapter (T=0.7) | `gen_taskset_ws3_presets.py` |
| LLM phase (4-8) | forced P5 turns | `taskset_llm/loop.py` |
| Hybrid (9-100) | PUCB llm vs rgpe, LLM prior 0.7→0.3, q=arm_best; cooldown 10-20 | `loop.py` + `hybrid_selector.py` |
| Duplicates | `history_tpe5`: 5 LLM proposals with reminders, then RGPE | `loop.py`, `p5_prompt.retry_*` |
| RGPE arm | the verified 838-source RGPE/TAF worker of the RGPE test210 baseline | `run_taskset_rgpe.suggest` |
| Task rerank (4-8) | same-family sources ranked by Spearman(source-GP prediction, target loss), floor 0.5, top 3, ties by standardized spec cosine | `taskset_llm/experience.py`, `taskset_source_rank_worker.py` |
| P5 history | Top5 / Bottom5 / Recent15 global + Recent3 per optimizer family | `taskset_llm/p5_prompt.py` |
| From trial 50 | ML fifty_fifty coin: explore → per-family exploration counts; exploit → good/bad regions per family (A5 marginal bins on the family's own parameters; the incumbent's values are never marked bad) | `p5_prompt.py` |
| Task profile | task family + TaskSet spec; no optimizer named | `taskset_llm/spec.py` |
| Answer format | flat exact ConfigSpace keys (`family:hp`, `_inv`, `_is_zero`), ML-style lenient repair | `spec.coerce_config` |

Prompts are rendered as LlamaFactory `qwen3_5_nothink` (no `<think>`) in training and
inference (`spec.render_nothink`); the ML warm-start generator appends `<think>\n`,
which this port does not.

## Run order (each step is a command; none has been run except data builds marked ✓)

```bash
cd /root/data/cys/llmoptimizer_benchmark
PY=/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python
FT=/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune

# 1 ✓ warm-start SFT data: top-1 best config per train838 TPE history (ML formula)
$PY examples/benchmark/domains/build_taskset_warmstart_sft.py
# 2   train warm-start adapter (sft_6_dl hyperparameters)
llamafactory-cli train $FT/configs/dl/llamafactory_qwen35_9b_lora_taskset_ws_train838.yaml
# 3   presets for test210
$PY examples/benchmark/domains/gen_taskset_ws3_presets.py \
    --adapter /root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_taskset_ws_train838 --out <campaign>/warm_start_presets.json
# 4 ✓ experience bank (train838 only)
$PY examples/benchmark/domains/build_taskset_experience_bank.py
# 5   TaskSet ckpt32 stage 1+2 data (SFT improvement events, ORPO attribution pairs)
$PY examples/benchmark/domains/build_taskset_p5_training_data.py
llamafactory-cli train $FT/configs/dl/llamafactory_qwen35_9b_lora_taskset_p5_sft_ep7p5.yaml
llamafactory-cli train $FT/configs/dl/llamafactory_qwen35_9b_lora_taskset_p5_sft_ep7p5_orpo_t2.yaml
# 6   TaskSet ckpt32 stage 3 (GRPO): select 32 train tasks, collect real-eval rollouts, train
$PY examples/benchmark/domains/collect_taskset_grpo_rollouts.py select --out <rl>/selection.json
$PY examples/benchmark/domains/collect_taskset_grpo_rollouts.py collect --selection <rl>/selection.json \
    --adapter <orpo adapter> --out <rl>/rollouts
#     train_p5_reinforce.py (v14) with the loader in taskset_reinforce_records.py (see its docstring)
# 7   evaluate: preflight on the dummy oracle, then test210
$PY examples/benchmark/domains/run_taskset_llm_rgpe.py --campaign <campaign> --task <test task> \
    --adapter <TaskSet ckpt32> --presets <campaign>/warm_start_presets.json --trials 12 --check --dummy
```

Data produced so far (in `$FT/data/`, registered in `llamafactory_data/dataset_info.json`):
`taskset_ws_train838.jsonl` (838; best family adam1p 344, nadamw 170, adam4p 119,
adam6p 106, adam8p 99) and `taskset_ws_test210_prompts.jsonl`; bank in
`history_data/domain_taskset/experience_bank_train838.jsonl` (838 cards, 802 with regions).
Dataset names `taskset_p5_sft` / `taskset_p5_orpo_t2` are registered but their files are
written by step 5.

## Notes

- Prompt length: P5 prompts are ~3.7k tokens at trial 9 and up to ~12k at trial 100
  (configs carry 1-14 parameters plus the 34-line CASH space); the SFT/ORPO yamls use
  `cutoff_len: 12288` (ML: 6016).
- Rerank with 3-5 observations: Spearman takes few values (3 points: ±1, ±0.5), so many
  sources tie at 1.0; the spec-cosine tie-break decides among them.
- Rewards for GRPO are real TaskSet evaluations (9-50 s CPU each; ~1.5-4k evaluations
  for 32 tasks × 12 prefixes × 4-10 samples), as in ML.
- Files copied into this repo for the RGPE arm: `examples/benchmark/rgpe_worker.py`,
  `mindware/components/optimizers/transfer_hpo/` (from the RGPE test210 code snapshot)
  and `hybrid_selector.py` (from the live ML repo, adds the `rgpe` arm and attempt counts;
  the previous version is in `.taskset_llm_backups/`).
- Tests: `python -m unittest tests.domains.test_taskset_llm_rgpe` (offline: scripted LLM,
  dummy oracle, fake RGPE).
