# On-Policy Distillation

This folder collects teacher rollouts for distilling an LLM optimizer.

The collector uses the existing `LLMOptimizer` HPO stack:

- model config from `src/config/model_registry.py`
- OpenAI-compatible API calls through the `openai` Python SDK
- teacher model key `gemini-3.1-pro`
- API base URL `https://api.openai-hk.com/v1`
- API key from `OPENAI_HK_API_KEY`

No separate Gemini SDK is needed because `api.openai-hk.com/v1` is OpenAI-compatible.

## What "on-policy" means here

For each benchmark task, the teacher proposes one configuration, the local HPOBench oracle evaluates it, and that evaluated result is fed back into the next teacher prompt as `previous_results`.

Each step saves:

- the exact teacher prompt
- the teacher JSON response
- the coerced/evaluated configuration
- loss/accuracy/cost
- token usage
- a chat-style `messages` pair ready for SFT distillation

## Example

```bash
cd /jizhicfs/pkuhetu/bcxu/LLMOptimizer
export OPENAI_HK_API_KEY=...

python on_policy_distillation/collect_hpo_rollouts.py \
  --model-key gemini-3.1-pro \
  --tasks lr,rf,xgboost,nn \
  --task-ids 359955,190411,359977,359956,359970 \
  --steps 10 \
  --resume
```

Default output:

```text
on_policy_distillation/runs/hpo_on_policy_gemini-3.1-pro.jsonl
```

