#!/usr/bin/env python3
"""On-policy P5 rollouts with direct holdout-eval rewards for future GRPO/PPO.

Runs ptllm (pure LLM) CASH on a single dataset, hooks PT4OptLLMOptimizer after each
trial, and appends JSONL records:

  messages (state) + completion (action) + reward metadata (Δbest primary).

Example:
  conda run -n openbox_qwen_eval python examples/benchmark/run_p5_rl_rollout_buffer.py \\
    --task-type cls --dataset yeast --algorithm lightgbm \\
    --prompt-variant P5 --qwen-adapter-path /path/to/lora \\
    --llm-response-format json_only --llm-max-tokens 768 \\
    --n-trials 80 --output-dir benchmark_data/rl_rollout_buffers/yeast_lightgbm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "examples" / "benchmark"))

from experiment_utils import apply_prompt_settings, default_llm_max_tokens  # noqa: E402
from mindware import CASH  # noqa: E402
from mindware.components.optimizers.pt4opt_llm_optimizer import PT4OptLLMOptimizer  # noqa: E402
from mindware.components.utils.constants import CLASSIFICATION, REGRESSION, SUCCESS  # noqa: E402
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase  # noqa: E402
from mindware.modules.configs import CONFIGS  # noqa: E402

from rl_reward_utils import RewardWeights, compute_trial_reward  # noqa: E402
from run_toy_categorical_optimizers import (  # noqa: E402
    build_optimizer_params,
    expand_optimizer_variants,
    history_scope_dir,
    load_real_data,
    resolve_variant,
    traj_label_for_variant,
)

MLTEST20_TASKS = [
    ("yeast", "lightgbm", "cls"),
    ("car", "extra_trees", "cls"),
    ("PhishingWebsites", "xgboost", "cls"),
    ("segment", "catboost", "cls"),
    ("boston", "extra_trees", "rgs"),
    ("tecator", "extra_trees", "rgs"),
    ("socmob", "extra_trees", "rgs"),
    ("pol", "xgboost", "rgs"),
]


class RLRolloutBuffer:
  """Append-only JSONL buffer for on-policy RL rollouts."""

  def __init__(
      self,
      path: Path,
      *,
      run_id: str,
      meta: Dict[str, Any],
      weights: RewardWeights,
      resume: bool = False,
  ):
      self.path = path
      self.run_id = run_id
      self.meta = meta
      self.weights = weights
      self.resume = resume
      self._seen_trials: set[int] = set()
      self.path.parent.mkdir(parents=True, exist_ok=True)
      if resume and self.path.is_file():
          self._load_existing_trials()

  def _load_existing_trials(self):
      with self.path.open(encoding="utf-8") as f:
          for line in f:
              line = line.strip()
              if not line:
                  continue
              row = json.loads(line)
              if row.get("run_id") == self.run_id:
                  self._seen_trials.add(int(row["trial"]))

  def on_trial_complete(
      self,
      optimizer: PT4OptLLMOptimizer,
      iter_idx: int,
      prompt_input: Dict[str, Any],
      raw: str,
      payload: Dict[str, Any],
      cfg,
      obs,
      meta: Dict[str, Any],
      note: str,
  ):
      if iter_idx in self._seen_trials:
          return

      success = getattr(obs, "trial_state", None) == SUCCESS
      raw_obj = float(obs.objectives[0]) if getattr(obs, "objectives", None) else np.inf
      eval_ok = success and np.isfinite(raw_obj)
      perf = -raw_obj if eval_ok else float("-inf")

      prior_perfs = [float(p) for p in optimizer.perfs if np.isfinite(p)]
      if eval_ok and len(optimizer.perfs) > 0 and np.isfinite(optimizer.perfs[-1]):
          # iterate() appends to perfs before _dump_conversation on success.
          prior_perfs = prior_perfs[:-1]

      history = optimizer.llm_history[-1] if optimizer.llm_history else {}
      status = str(history.get("status", ""))
      parse_ok = status not in ("FAILED",) and bool(payload)
      duplicate = status == "DUPLICATE" or "duplicate" in note.lower()

      reward = compute_trial_reward(
          perf=perf,
          prior_perfs=prior_perfs,
          parse_ok=parse_ok,
          duplicate=duplicate,
          eval_ok=eval_ok,
          weights=self.weights,
      )

      messages = list(prompt_input.get("messages") or [])
      record = {
          "run_id": self.run_id,
          "trial": int(iter_idx),
          "ts": datetime.now(timezone.utc).isoformat(),
          "dataset": self.meta.get("dataset"),
          "task_type": self.meta.get("task_type"),
          "algorithm": self.meta.get("algorithm"),
          "prompt_variant": self.meta.get("prompt_variant"),
          "variant_name": self.meta.get("variant_name"),
          "adapter_path": self.meta.get("adapter_path"),
          "response_format": self.meta.get("response_format"),
          "messages": messages,
          "completion": raw,
          "parsed_payload": payload,
          "normalized_config": history.get("evaluated_config") or history.get("normalized_config"),
          "note": note,
          "history_meta": meta,
          "reward": reward.to_dict(),
      }
      with self.path.open("a", encoding="utf-8") as f:
          f.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")
      self._seen_trials.add(iter_idx)

  def write_manifest(self, extra: Optional[Dict[str, Any]] = None):
      manifest_path = self.path.with_suffix(".manifest.json")
      payload = {
          "run_id": self.run_id,
          "buffer_path": str(self.path),
          "n_records": len(self._seen_trials),
          "meta": self.meta,
          "weights": {
              "delta_best": self.weights.delta_best,
              "absolute": self.weights.absolute,
              "parse_fail": self.weights.parse_fail,
              "duplicate": self.weights.duplicate,
              "eval_fail": self.weights.eval_fail,
          },
      }
      if extra:
          payload.update(extra)
      manifest_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
      return manifest_path


_HOOK_STATE: Dict[str, Any] = {"installed": False, "orig": None}


def install_rl_hook(buffer: RLRolloutBuffer):
    if _HOOK_STATE["installed"]:
        return
    orig = PT4OptLLMOptimizer._dump_conversation

    def wrapped(self, iter_idx, prompt_input, raw, payload, cfg, obs, meta, note):
        orig(self, iter_idx, prompt_input, raw, payload, cfg, obs, meta, note)
        buffer.on_trial_complete(self, iter_idx, prompt_input, raw, payload, cfg, obs, meta, note)

    PT4OptLLMOptimizer._dump_conversation = wrapped  # type: ignore[method-assign]
    _HOOK_STATE["orig"] = orig
    _HOOK_STATE["installed"] = True


def uninstall_rl_hook():
    if _HOOK_STATE["installed"] and _HOOK_STATE["orig"] is not None:
        PT4OptLLMOptimizer._dump_conversation = _HOOK_STATE["orig"]  # type: ignore[method-assign]
    _HOOK_STATE["installed"] = False
    _HOOK_STATE["orig"] = None


def run_rollout(args) -> Path:
    os.chdir(REPO_ROOT)

    variant_name = expand_optimizer_variants(["ptllm_qwen"], args)[0]
    dispatch, output_name, _backend = resolve_variant(variant_name)
    task_const = CLASSIFICATION if args.task_type == "cls" else REGRESSION
    metric = args.metric

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_dir = Path(args.history_dir).resolve()
    scope_dir = history_scope_dir(history_dir, args.task_type, metric, args.dataset, output_name)
    CONFIGS.ORIGINAL_DATA_DIR = str(history_dir)
    CONFIGS.HISTORY_RUN_SCOPE = str(scope_dir)

    _, train_node, _test_node, dataset_path, dataset_profile = load_real_data(args)
    algorithms = list(args.algorithm)
    optimizer_params = build_optimizer_params(args, variant_name, algorithms, dataset_profile)

    KnowledgeDatabase._instance = None

    run_id = args.run_id or f"{args.dataset}_{algorithms[0]}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    buffer_name = args.buffer_name or f"rollouts_{run_id}.jsonl"
    buffer_path = output_dir / buffer_name

    weights = RewardWeights(
        delta_best=args.reward_delta_best,
        absolute=args.reward_absolute,
        parse_fail=args.penalty_parse_fail,
        duplicate=args.penalty_duplicate,
        eval_fail=args.penalty_eval_fail,
    )
    meta = {
        "dataset": args.dataset,
        "task_type": args.task_type,
        "algorithm": algorithms[0],
        "metric": metric,
        "prompt_variant": args.prompt_variant,
        "variant_name": variant_name,
        "adapter_path": getattr(args, "qwen_adapter_path", None),
        "response_format": args.llm_response_format,
        "n_trials": args.n_trials,
        "seed": args.seed,
        "dataset_path": str(dataset_path),
    }
    buffer = RLRolloutBuffer(buffer_path, run_id=run_id, meta=meta, weights=weights, resume=args.resume)
    install_rl_hook(buffer)

    task_id = f"{args.dataset}_rl_{run_id}"
    print(f"[RL rollout] run_id={run_id}")
    print(f"[RL rollout] variant={variant_name} dataset={args.dataset} algo={algorithms[0]}")
    print(f"[RL rollout] buffer={buffer_path}")

    t0 = time.time()
    try:
        opt = CASH(
            include_algorithms=algorithms,
            task_type=task_const,
            metric=metric,
            data_node=train_node,
            evaluation=args.evaluation,
            resampling_params={"test_size": args.valid_size, "shuffle": False},
            optimizer=dispatch,
            time_limit=args.time_limit,
            amount_of_resource=args.n_trials,
            per_run_time_limit=args.per_run_time_limit,
            inner_iter_num_per_iter=1,
            output_dir=str(output_dir / output_name),
            seed=args.seed,
            n_jobs=1,
            task_id=task_id,
            topk=args.n_trials,
            optimizer_params=optimizer_params,
        )
        best_perf, stats_path = opt.run()
    finally:
        uninstall_rl_hook()

    elapsed = time.time() - t0
    manifest = buffer.write_manifest(
        {
            "elapsed_sec": elapsed,
            "cash_output_dir": str(output_dir / output_name),
            "traj_label": traj_label_for_variant(variant_name),
        }
    )
    print(f"[RL rollout] wrote {len(buffer._seen_trials)} records -> {buffer_path}")
    print(f"[RL rollout] manifest -> {manifest}")
    return buffer_path


def parse_args():
    from run_toy_categorical_optimizers import DEFAULT_QWEN_MODEL_PATH, parse_llm_backends

    parser = argparse.ArgumentParser(description="P5 on-policy RL rollout buffer (direct eval reward).")
    parser.add_argument("--task-type", choices=["cls", "rgs"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--algorithm", nargs="+", required=True)
    parser.add_argument("--metric", default=None, help="Defaults: bal_acc (cls) / mse (rgs).")
    parser.add_argument("--evaluation", default="holdout", choices=["holdout", "cv", "train_test"])
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-trials", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--time-limit", type=int, default=86400)
    parser.add_argument("--per-run-time-limit", type=int, default=600)
    parser.add_argument("--output-dir", default="benchmark_data/rl_rollout_buffers")
    parser.add_argument("--history-dir", default="benchmark_data/bo_history")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--buffer-name", default=None)
    parser.add_argument("--resume", action="store_true", help="Skip trials already present in buffer for run_id.")

    parser.add_argument("--prompt-variant", default="P5")
    parser.add_argument("--hybrid-variant", default="H0")
    parser.add_argument("--llm-backends", default="qwen-sft6")
    parser.add_argument("--qwen-backend", choices=["auto", "hf", "ollama"], default="auto")
    parser.add_argument("--qwen-model-path", default=DEFAULT_QWEN_MODEL_PATH)
    parser.add_argument("--qwen-adapter-path", default=None)
    parser.add_argument("--qwen-enable-thinking", action="store_true")
    parser.add_argument("--llm-mock", action="store_true")
    parser.add_argument("--llm-response", action="append", default=None)
    parser.add_argument("--llm-chat-engine", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--llm-max-tokens", type=int, default=None)
    parser.add_argument("--llm-response-format", default="json_only", choices=["json_only", "rationale_json"])
    parser.add_argument("--llm-timeout", type=int, default=180)
    parser.add_argument("--llm-n-initial-samples", type=int, default=4)
    parser.add_argument("--llm-history-window", type=int, default=20)
    parser.add_argument("--llm-top-k", type=int, default=6)
    parser.add_argument("--llm-recent-k", type=int, default=10)
    parser.add_argument("--llm-worst-k", type=int, default=4)
    parser.add_argument("--llm-run-summary", action="store_true", default=True)
    parser.add_argument("--no-llm-run-summary", action="store_false", dest="llm_run_summary")
    parser.add_argument("--llm-fallback-on-transport-error", action="store_true")
    parser.add_argument("--llm-tpe-llm-prob", type=float, default=0.5)
    parser.add_argument("--llm-tpe-phase-trials", type=int, default=0)
    parser.add_argument("--llm-tpe-tpe-startup-trials", type=int, default=3)
    parser.add_argument("--llm-tpe-tpe-duplicate-retries", type=int, default=5)
    parser.add_argument("--tpe-startup-trials", type=int, default=3)
    parser.add_argument(
        "--preset-warm-start-json",
        default=None,
        help="Optional warm-start JSON for trial-1 slots.",
    )

    parser.add_argument("--reward-delta-best", type=float, default=1.0)
    parser.add_argument("--reward-absolute", type=float, default=0.1)
    parser.add_argument("--penalty-parse-fail", type=float, default=0.5)
    parser.add_argument("--penalty-duplicate", type=float, default=0.1)
    parser.add_argument("--penalty-eval-fail", type=float, default=1.0)

    parser.add_argument(
        "--mltest20-all",
        action="store_true",
        help="Run Option B tasks sequentially (dataset+algorithm from mltest20).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.llm_backends = ["qwen-sft6"]
    args.optimizer = ["ptllm_qwen"]
    if args.metric is None:
        args.metric = "bal_acc" if args.task_type == "cls" else "mse"
    apply_prompt_settings(args, args.prompt_variant)
    if args.hybrid_variant:
        from experiment_utils import apply_hybrid_settings

        apply_hybrid_settings(args, args.hybrid_variant)
    if args.llm_max_tokens is None:
        args.llm_max_tokens = default_llm_max_tokens(args.llm_response_format)

    if args.mltest20_all:
        last = None
        for dataset, algorithm, task_type in MLTEST20_TASKS:
            args.dataset = dataset
            args.algorithm = [algorithm]
            args.task_type = task_type
            args.metric = "bal_acc" if task_type == "cls" else "mse"
            subdir = Path(args.output_dir) / f"{task_type}_{dataset}_{algorithm}"
            args.output_dir = str(subdir)
            last = run_rollout(args)
        return last

    return run_rollout(args)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""On-policy P5 rollouts with direct holdout-eval rewards for future GRPO/PPO.

Runs ptllm (pure LLM) CASH on a single dataset, hooks PT4OptLLMOptimizer after each
trial, and appends JSONL records:

  messages (state) + completion (action) + reward metadata (Δbest primary).

Example:
  conda run -n openbox_qwen_eval python examples/benchmark/run_p5_rl_rollout_buffer.py \\
    --task-type cls --dataset yeast --algorithm lightgbm \\
    --prompt-variant P5 --qwen-adapter-path /path/to/lora \\
    --llm-response-format json_only --llm-max-tokens 768 \\
    --n-trials 80 --output-dir benchmark_data/rl_rollout_buffers/yeast_lightgbm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "examples" / "benchmark"))

from experiment_utils import apply_prompt_settings, default_llm_max_tokens  # noqa: E402
from mindware import CASH  # noqa: E402
from mindware.components.optimizers.pt4opt_llm_optimizer import PT4OptLLMOptimizer  # noqa: E402
from mindware.components.utils.constants import CLASSIFICATION, REGRESSION, SUCCESS  # noqa: E402
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase  # noqa: E402
from mindware.modules.configs import CONFIGS  # noqa: E402

from rl_reward_utils import RewardWeights, compute_trial_reward  # noqa: E402
from run_toy_categorical_optimizers import (  # noqa: E402
    build_optimizer_params,
    expand_optimizer_variants,
    history_scope_dir,
    load_real_data,
    resolve_variant,
    traj_label_for_variant,
)

MLTEST20_TASKS = [
    ("yeast", "lightgbm", "cls"),
    ("car", "extra_trees", "cls"),
    ("PhishingWebsites", "xgboost", "cls"),
    ("segment", "catboost", "cls"),
    ("boston", "extra_trees", "rgs"),
    ("tecator", "extra_trees", "rgs"),
    ("socmob", "extra_trees", "rgs"),
    ("pol", "xgboost", "rgs"),
]


class RLRolloutBuffer:
  """Append-only JSONL buffer for on-policy RL rollouts."""

  def __init__(
      self,
      path: Path,
      *,
      run_id: str,
      meta: Dict[str, Any],
      weights: RewardWeights,
      resume: bool = False,
  ):
      self.path = path
      self.run_id = run_id
      self.meta = meta
      self.weights = weights
      self.resume = resume
      self._seen_trials: set[int] = set()
      self.path.parent.mkdir(parents=True, exist_ok=True)
      if resume and self.path.is_file():
          self._load_existing_trials()

  def _load_existing_trials(self):
      with self.path.open(encoding="utf-8") as f:
          for line in f:
              line = line.strip()
              if not line:
                  continue
              row = json.loads(line)
              if row.get("run_id") == self.run_id:
                  self._seen_trials.add(int(row["trial"]))

  def on_trial_complete(
      self,
      optimizer: PT4OptLLMOptimizer,
      iter_idx: int,
      prompt_input: Dict[str, Any],
      raw: str,
      payload: Dict[str, Any],
      cfg,
      obs,
      meta: Dict[str, Any],
      note: str,
  ):
      if iter_idx in self._seen_trials:
          return

      success = getattr(obs, "trial_state", None) == SUCCESS
      raw_obj = float(obs.objectives[0]) if getattr(obs, "objectives", None) else np.inf
      eval_ok = success and np.isfinite(raw_obj)
      perf = -raw_obj if eval_ok else float("-inf")

      prior_perfs = [float(p) for p in optimizer.perfs if np.isfinite(p)]
      if eval_ok and len(optimizer.perfs) > 0 and np.isfinite(optimizer.perfs[-1]):
          # iterate() appends to perfs before _dump_conversation on success.
          prior_perfs = prior_perfs[:-1]

      history = optimizer.llm_history[-1] if optimizer.llm_history else {}
      status = str(history.get("status", ""))
      parse_ok = status not in ("FAILED",) and bool(payload)
      duplicate = status == "DUPLICATE" or "duplicate" in note.lower()

      reward = compute_trial_reward(
          perf=perf,
          prior_perfs=prior_perfs,
          parse_ok=parse_ok,
          duplicate=duplicate,
          eval_ok=eval_ok,
          weights=self.weights,
      )

      messages = list(prompt_input.get("messages") or [])
      record = {
          "run_id": self.run_id,
          "trial": int(iter_idx),
          "ts": datetime.now(timezone.utc).isoformat(),
          "dataset": self.meta.get("dataset"),
          "task_type": self.meta.get("task_type"),
          "algorithm": self.meta.get("algorithm"),
          "prompt_variant": self.meta.get("prompt_variant"),
          "variant_name": self.meta.get("variant_name"),
          "adapter_path": self.meta.get("adapter_path"),
          "response_format": self.meta.get("response_format"),
          "messages": messages,
          "completion": raw,
          "parsed_payload": payload,
          "normalized_config": history.get("evaluated_config") or history.get("normalized_config"),
          "note": note,
          "history_meta": meta,
          "reward": reward.to_dict(),
      }
      with self.path.open("a", encoding="utf-8") as f:
          f.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")
      self._seen_trials.add(iter_idx)

  def write_manifest(self, extra: Optional[Dict[str, Any]] = None):
      manifest_path = self.path.with_suffix(".manifest.json")
      payload = {
          "run_id": self.run_id,
          "buffer_path": str(self.path),
          "n_records": len(self._seen_trials),
          "meta": self.meta,
          "weights": {
              "delta_best": self.weights.delta_best,
              "absolute": self.weights.absolute,
              "parse_fail": self.weights.parse_fail,
              "duplicate": self.weights.duplicate,
              "eval_fail": self.weights.eval_fail,
          },
      }
      if extra:
          payload.update(extra)
      manifest_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
      return manifest_path


_HOOK_STATE: Dict[str, Any] = {"installed": False, "orig": None}


def install_rl_hook(buffer: RLRolloutBuffer):
    if _HOOK_STATE["installed"]:
        return
    orig = PT4OptLLMOptimizer._dump_conversation

    def wrapped(self, iter_idx, prompt_input, raw, payload, cfg, obs, meta, note):
        orig(self, iter_idx, prompt_input, raw, payload, cfg, obs, meta, note)
        buffer.on_trial_complete(self, iter_idx, prompt_input, raw, payload, cfg, obs, meta, note)

    PT4OptLLMOptimizer._dump_conversation = wrapped  # type: ignore[method-assign]
    _HOOK_STATE["orig"] = orig
    _HOOK_STATE["installed"] = True


def uninstall_rl_hook():
    if _HOOK_STATE["installed"] and _HOOK_STATE["orig"] is not None:
        PT4OptLLMOptimizer._dump_conversation = _HOOK_STATE["orig"]  # type: ignore[method-assign]
    _HOOK_STATE["installed"] = False
    _HOOK_STATE["orig"] = None


def run_rollout(args) -> Path:
    os.chdir(REPO_ROOT)

    variant_name = expand_optimizer_variants(["ptllm_qwen"], args)[0]
    dispatch, output_name, _backend = resolve_variant(variant_name)
    task_const = CLASSIFICATION if args.task_type == "cls" else REGRESSION
    metric = args.metric

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_dir = Path(args.history_dir).resolve()
    scope_dir = history_scope_dir(history_dir, args.task_type, metric, args.dataset, output_name)
    CONFIGS.ORIGINAL_DATA_DIR = str(history_dir)
    CONFIGS.HISTORY_RUN_SCOPE = str(scope_dir)

    _, train_node, _test_node, dataset_path, dataset_profile = load_real_data(args)
    algorithms = list(args.algorithm)
    optimizer_params = build_optimizer_params(args, variant_name, algorithms, dataset_profile)

    KnowledgeDatabase._instance = None

    run_id = args.run_id or f"{args.dataset}_{algorithms[0]}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    buffer_name = args.buffer_name or f"rollouts_{run_id}.jsonl"
    buffer_path = output_dir / buffer_name

    weights = RewardWeights(
        delta_best=args.reward_delta_best,
        absolute=args.reward_absolute,
        parse_fail=args.penalty_parse_fail,
        duplicate=args.penalty_duplicate,
        eval_fail=args.penalty_eval_fail,
    )
    meta = {
        "dataset": args.dataset,
        "task_type": args.task_type,
        "algorithm": algorithms[0],
        "metric": metric,
        "prompt_variant": args.prompt_variant,
        "variant_name": variant_name,
        "adapter_path": getattr(args, "qwen_adapter_path", None),
        "response_format": args.llm_response_format,
        "n_trials": args.n_trials,
        "seed": args.seed,
        "dataset_path": str(dataset_path),
    }
    buffer = RLRolloutBuffer(buffer_path, run_id=run_id, meta=meta, weights=weights, resume=args.resume)
    install_rl_hook(buffer)

    task_id = f"{args.dataset}_rl_{run_id}"
    print(f"[RL rollout] run_id={run_id}")
    print(f"[RL rollout] variant={variant_name} dataset={args.dataset} algo={algorithms[0]}")
    print(f"[RL rollout] buffer={buffer_path}")

    t0 = time.time()
    try:
        opt = CASH(
            include_algorithms=algorithms,
            task_type=task_const,
            metric=metric,
            data_node=train_node,
            evaluation=args.evaluation,
            resampling_params={"test_size": args.valid_size, "shuffle": False},
            optimizer=dispatch,
            time_limit=args.time_limit,
            amount_of_resource=args.n_trials,
            per_run_time_limit=args.per_run_time_limit,
            inner_iter_num_per_iter=1,
            output_dir=str(output_dir / output_name),
            seed=args.seed,
            n_jobs=1,
            task_id=task_id,
            topk=args.n_trials,
            optimizer_params=optimizer_params,
        )
        best_perf, stats_path = opt.run()
    finally:
        uninstall_rl_hook()

    elapsed = time.time() - t0
    manifest = buffer.write_manifest(
        {
            "elapsed_sec": elapsed,
            "cash_output_dir": str(output_dir / output_name),
            "traj_label": traj_label_for_variant(variant_name),
        }
    )
    print(f"[RL rollout] wrote {len(buffer._seen_trials)} records -> {buffer_path}")
    print(f"[RL rollout] manifest -> {manifest}")
    return buffer_path


def parse_args():
    from run_toy_categorical_optimizers import DEFAULT_QWEN_MODEL_PATH, parse_llm_backends

    parser = argparse.ArgumentParser(description="P5 on-policy RL rollout buffer (direct eval reward).")
    parser.add_argument("--task-type", choices=["cls", "rgs"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--algorithm", nargs="+", required=True)
    parser.add_argument("--metric", default=None, help="Defaults: bal_acc (cls) / mse (rgs).")
    parser.add_argument("--evaluation", default="holdout", choices=["holdout", "cv", "train_test"])
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-trials", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--time-limit", type=int, default=86400)
    parser.add_argument("--per-run-time-limit", type=int, default=600)
    parser.add_argument("--output-dir", default="benchmark_data/rl_rollout_buffers")
    parser.add_argument("--history-dir", default="benchmark_data/bo_history")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--buffer-name", default=None)
    parser.add_argument("--resume", action="store_true", help="Skip trials already present in buffer for run_id.")

    parser.add_argument("--prompt-variant", default="P5")
    parser.add_argument("--hybrid-variant", default="H0")
    parser.add_argument("--llm-backends", default="qwen-sft6")
    parser.add_argument("--qwen-backend", choices=["auto", "hf", "ollama"], default="auto")
    parser.add_argument("--qwen-model-path", default=DEFAULT_QWEN_MODEL_PATH)
    parser.add_argument("--qwen-adapter-path", default=None)
    parser.add_argument("--qwen-enable-thinking", action="store_true")
    parser.add_argument("--llm-mock", action="store_true")
    parser.add_argument("--llm-response", action="append", default=None)
    parser.add_argument("--llm-chat-engine", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--llm-max-tokens", type=int, default=None)
    parser.add_argument("--llm-response-format", default="json_only", choices=["json_only", "rationale_json"])
    parser.add_argument("--llm-timeout", type=int, default=180)
    parser.add_argument("--llm-n-initial-samples", type=int, default=4)
    parser.add_argument("--llm-history-window", type=int, default=20)
    parser.add_argument("--llm-top-k", type=int, default=6)
    parser.add_argument("--llm-recent-k", type=int, default=10)
    parser.add_argument("--llm-worst-k", type=int, default=4)
    parser.add_argument("--llm-run-summary", action="store_true", default=True)
    parser.add_argument("--no-llm-run-summary", action="store_false", dest="llm_run_summary")
    parser.add_argument("--llm-fallback-on-transport-error", action="store_true")
    parser.add_argument("--llm-tpe-llm-prob", type=float, default=0.5)
    parser.add_argument("--llm-tpe-phase-trials", type=int, default=0)
    parser.add_argument("--llm-tpe-tpe-startup-trials", type=int, default=3)
    parser.add_argument("--llm-tpe-tpe-duplicate-retries", type=int, default=5)
    parser.add_argument("--tpe-startup-trials", type=int, default=3)
    parser.add_argument(
        "--preset-warm-start-json",
        default=None,
        help="Optional warm-start JSON for trial-1 slots.",
    )

    parser.add_argument("--reward-delta-best", type=float, default=1.0)
    parser.add_argument("--reward-absolute", type=float, default=0.1)
    parser.add_argument("--penalty-parse-fail", type=float, default=0.5)
    parser.add_argument("--penalty-duplicate", type=float, default=0.1)
    parser.add_argument("--penalty-eval-fail", type=float, default=1.0)

    parser.add_argument(
        "--mltest20-all",
        action="store_true",
        help="Run Option B tasks sequentially (dataset+algorithm from mltest20).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.llm_backends = ["qwen-sft6"]
    args.optimizer = ["ptllm_qwen"]
    if args.metric is None:
        args.metric = "bal_acc" if args.task_type == "cls" else "mse"
    apply_prompt_settings(args, args.prompt_variant)
    if args.hybrid_variant:
        from experiment_utils import apply_hybrid_settings

        apply_hybrid_settings(args, args.hybrid_variant)
    if args.llm_max_tokens is None:
        args.llm_max_tokens = default_llm_max_tokens(args.llm_response_format)

    if args.mltest20_all:
        last = None
        for dataset, algorithm, task_type in MLTEST20_TASKS:
            args.dataset = dataset
            args.algorithm = [algorithm]
            args.task_type = task_type
            args.metric = "bal_acc" if task_type == "cls" else "mse"
            subdir = Path(args.output_dir) / f"{task_type}_{dataset}_{algorithm}"
            args.output_dir = str(subdir)
            last = run_rollout(args)
        return last

    return run_rollout(args)


if __name__ == "__main__":
    main()
