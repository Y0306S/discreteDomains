# Self-Evolving HPO Agent

This folder implements a three-pillar HPO pipeline inspired by BOLT and Rocket:

1. **Warm start**: generate BOLT-style chat SFT examples from task descriptions to best configurations.
2. **Trajectory exploitation**: run Bayesian optimization, summarize each round, and build trajectory SFT examples where the target is the next configuration plus rationale.
3. **Cross-task transfer**: distill finished optimization traces into textual playbooks for future tasks.

The implementation targets HPOBench ML tasks with numeric ConfigSpace parameters. It removes BOLT's VAE layer and optimizes directly in normalized hyperparameter space.

## Layout

```text
self-evolving_agent/
  configs/ml_tasks.yaml
  bo/
    run_bo.py
    botorch_optimizer.py
    spaces.py
    trajectory.py
  data_generation/
    build_sft_best_config.py
    build_sft_trajectories.py
    smooth_trajectories_with_llm.py
    build_playbooks.py
  llm/
    client.py
    prompts.py
  artifacts/
    bo_traces/
    trajectory_summaries/
    sft_jsonl/
    playbooks/
```

## Typical Pipeline

Run BO on one HPOBench task:

```bash
conda activate hpobench
cd /jizhicfs/pkuhetu/bcxu/LLMOptimizer
python self-evolving_agent/bo/run_bo.py --benchmark lr --task-id 359955 --max-trials 50
```

Create BOLT-style best-configuration SFT data:

```bash
python self-evolving_agent/data_generation/build_sft_best_config.py \
  --trace-dir self-evolving_agent/artifacts/bo_traces \
  --output self-evolving_agent/artifacts/sft_jsonl/best_config.jsonl
```

Smooth real BO trajectories and attach rationales:

```bash
python self-evolving_agent/data_generation/smooth_trajectories_with_llm.py \
  --trace-dir self-evolving_agent/artifacts/bo_traces \
  --output-dir self-evolving_agent/artifacts/trajectory_summaries \
  --offline
```

Create trajectory SFT data:

```bash
python self-evolving_agent/data_generation/build_sft_trajectories.py \
  --trace-dir self-evolving_agent/artifacts/bo_traces \
  --smoothed-dir self-evolving_agent/artifacts/trajectory_summaries \
  --output self-evolving_agent/artifacts/sft_jsonl/trajectory_next_config.jsonl
```

Build cross-task playbooks:

```bash
python self-evolving_agent/data_generation/build_playbooks.py \
  --trace-dir self-evolving_agent/artifacts/bo_traces \
  --output-dir self-evolving_agent/artifacts/playbooks
```

## Notes

- `botorch` and `gpytorch` are required for BO. If they are missing, `run_bo.py` exits with an actionable install message.
- `hpobench` is required only when running BO. SFT/playbook generation operates on saved trace JSON files.
- LLM smoothing may explain or reorder real evaluated configurations, but validation rejects invented configs or modified losses.
