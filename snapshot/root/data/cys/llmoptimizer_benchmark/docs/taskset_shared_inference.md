TaskSet LLM/RGPE setup — 31 October 2026

This is the earlier RGPE proposal. The user's subsequent choice is **LLM + FSBO min-max** on test200. See [the current FSBO setup and online checks](taskset_llm_fsbo_preflight.md); retain this page as the earlier setup record.

The prepared topology is one shared Qwen inference service on LLMOptimizer3 GPU 0 and eight CPU evaluation workers on LLMOptimizer4, two CPUs each (48–63). Each worker has its own TaskSet oracle directory, target GP, history and pending proposal. The shared service handles LLM generation only. GPU 1's existing training process is unaffected. No production service, warm-start generation, training or evaluation campaign was launched while preparing this setup.

Prepared files live in `/root/data/cys/baseline/artifacts/taskset_shared_inference_20261031`:

- `workers.disabled.json`: disabled configuration, 100 rounds, seed 1, frozen test200 subset.
- `worker_commands.json`: all 200 commands, partitioned into eight workers with 25 tasks each.
- `bolt_prompts_test200.jsonl`: all 200 inference contexts from the completed BOLT full-SFT prompt builder; no outcomes or assistant targets.
- `before/`: copies of existing files changed during this work.
- `existing_tests.log`, `new_tests.log`, `precision_tests.log`: offline validation.

The 200 tasks come from the SFT/ORPO dataset's `test200.json`, a frozen subset of the original test210 excluding ten slow tasks. Training sources remain train838. Compare methods on this same subset; the old 210-task archives should not be rerun or aggregated against 200-task results without filtering.

The policy is the existing ML `HPUCB_rgpe_p07p03_retry5_llm8_cool20` implementation:

| Round | Proposal source |
| --- | --- |
| 1–3 | Three frozen presets sampled from the completed BOLT full SFT; no online LLM calls |
| 4–8 | LLM, with similar-task retrieval when enough usable history exists |
| 9 | PUCB chooses LLM or RGPE |
| 10–20 | PUCB; force RGPE when the previous round selected LLM, including an LLM turn filled by an RGPE fallback |
| 21–100 | PUCB, without that early cooldown |

PUCB uses `q=arm_best`, `c2=1.4` and an LLM prior decreasing from 0.7 toward 0.3. These are selection weights, not a fixed 70% random allocation. Warm starts do not update the two-arm selector. Every successful observation, including warm starts, is passed to RGPE when it next fits its target GP; it does not fit from only the initial three points forever. The ensemble retains the 838 source GPs.

There are at most five LLM proposals per LLM-selected round. Duplicates are retried with history reminders; five duplicates fall back to RGPE without another objective evaluation. A malformed reply uses the existing random fallback. There is no additional campaign-wide LLM-call cap in ML or TaskSet. Requests to the shared service have no automatic HTTP retries. A service outage now raises a visible error before the objective is evaluated, rather than being scored as a random LLM proposal.

Exploration/exploitation is a prompt policy inside an LLM turn, separate from PUCB's LLM-versus-RGPE decision:

- ML and TaskSet activate it at **round 50**, with 49 completed observations. The user's wording “after 50 rounds” would literally mean round 51; this setup preserves ML's round-50 boundary and reports it explicitly.
- A deterministic 50/50 coin, keyed by seed and round with the same hash recipe as ML, selects one prompt branch. This does not add objective evaluations or extra model calls.
- Exploration lists evaluation counts for **every optimizer family**, including zero-count families and failures, and lists successful counts separately. TaskSet's algorithms here are its five optimizer families, not the source task families used for retrieval.
- Exploitation lists Good/Bad regions for family choice and each family's own active parameters using the existing ML marginal-region helper and log bins where appropriate. It uses successful results only, requires support, protects the incumbent from Bad regions, and reports insufficient evidence rather than inventing regions. These are empirical associations.
- Both branches keep the compact global Top5/Bottom5/Recent15 history and up to Recent3 successes per optimizer family. The count/region computation still uses full committed history.

The early similarity prompt uses up to three same-task-family source cards, ranked by Spearman agreement of source-GP predictions with target losses, with a 0.5 floor and profile similarity as a tiebreaker. It is active on rounds 4–8 only. These cards do not replace the full RGPE source ensemble.

BOLT is the completed **full fine-tuned Qwen3.5-9B**, trained for 2,620 updates on 8,380 examples over five epochs. The generator now accepts `--full-model`, uses its exact training context with disabled Nesterov momentum and the original 10,000-step schedule horizon, and samples three configurations per task (temperature 0.7, top-p 0.9, maximum 768 new tokens, duplicates accepted). The presets are fixed across all 100 rounds; BOLT is not called every round. Sampling and raw draws are logged. Prompt preparation was completed; model generation was not run, so the actual presets file is still absent.

The in-loop model is the TaskSet ORPO adapter. Its actual training dataset used `p5_h0`; similarity cards and the explore/exploit blocks are runtime additions. Do not describe that adapter as having been explicitly trained on the P5+ branches, or claim their quality was validated by wiring tests.

The server uses BF16 base computation with exact FP32 LoRA loading and verification of every saved tensor. A full BOLT model is loaded directly without PEFT. The service hashes model weights and generation settings, serves health only after loading, and serializes generation to avoid concurrent GPU memory spikes. Clients pin that identity and use a request seed derived from the prompt and campaign seed, making sampling independent of request arrival order. This seeded shared service is a new inference identity; use a new campaign directory rather than resume an older local-inference campaign.

The CPU runner has a per-task file lock; a duplicate process cannot concurrently overwrite the same task. Each task retains its existing pending-proposal and history/state checks. A torn history/state commit fails closed and requires inspection; this change does not add recovery for that pre-existing limitation. The dispatcher restricts CPUs, hides GPUs, bounds library threads, checks the evaluation hostname and available memory, and stops a partition on a failed task. Configurations remain disabled by default.

The system Python has an incompatible newer ConfigSpace, and the training-only Qwen environment lacks ConfigSpace. Use the checked environments: CPU workers use `/root/data/cys/conda_envs/hpo_benchmark/bin/python` (ConfigSpace 0.6.1), and model inference/generation uses `/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python` (ConfigSpace 0.6.1, Torch 2.5.1, Transformers 5.6.0, PEFT 0.18.1). Their existing installations were not changed. Both remote hosts and required model/config files were checked read-only.

For later execution, first generate the BOLT presets on LLMOptimizer3 GPU 0, then exit that process and load the ORPO service on that GPU. The following are prepared commands, **not commands already executed**:

```bash
cd /root/data/cys/llmoptimizer_benchmark
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 /root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python examples/benchmark/domains/gen_taskset_ws3_presets.py \
  --full-model /root/data/cys/baseline/artifacts/bolt_taskset_top10_full_sft/model \
  --split benchmark_data/taskset_s1b_s2g_train838_20261029_v2/test200.json \
  --out /root/data/cys/baseline/artifacts/taskset_shared_inference_20261031/bolt_ws3_test200.json
```

Create a bearer token at the configured shared `inference.token` path with owner-only permissions, then start the service on LLMOptimizer3. It verifies all adapter tensors before it accepts requests; full-model GPU loading and generation still need this operational check.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 /root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python examples/benchmark/domains/serve_taskset_llm.py \
  --adapter /root/data/cys/baseline/artifacts/taskset_our_s1b_s2g_20261029/models/orpo_only_orpo \
  --token-file /root/data/cys/baseline/artifacts/taskset_shared_inference_20261031/inference.token \
  --identity-out /root/data/cys/baseline/artifacts/taskset_shared_inference_20261031/inference_identity.json
```

On LLMOptimizer4, an SSH loopback forward reaches that service:

```bash
ssh -N -L 127.0.0.1:18765:127.0.0.1:8765 LLMOptimizer3
```

Review current CPU/memory use again before enabling the prepared config. Plan mode needs no token, GPU, server or generated presets:

```bash
/root/data/cys/conda_envs/hpo_benchmark/bin/python examples/benchmark/domains/queue_taskset_llm_rgpe.py \
  --config /root/data/cys/baseline/artifacts/taskset_shared_inference_20261031/workers.disabled.json \
  --out /root/data/cys/baseline/artifacts/taskset_shared_inference_20261031/worker_commands.json
```

An enabled copy of the configuration plus `--execute` is required for evaluation. CPU workers do not spawn GPU models. No training or existing FSBO/RGPE campaigns are started by these commands in plan mode.
