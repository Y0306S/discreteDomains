#!/usr/bin/env python3
"""Setup A: BO-seeded P5 RL rollouts (light-16, T=1.0).

Seed PT4Opt history from nested history_data BO JSON (default 100 trials),
set n_initial_samples=0 so the first LLM call already uses P5 history, then
run n_new CASH LLM propose→eval steps (default 4) that append onto the seed.

Init policy: s1a_ep7p5 LoRA (passed via --qwen-adapter-path).
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "examples" / "benchmark"))

from bo_history_utils import (  # noqa: E402
    DEFAULT_BO_HISTORY_ROOT,
    bo_history_path,
    load_bo_observations,
    seed_optimizer_with_bo,
)
from experiment_utils import apply_prompt_settings, default_llm_max_tokens  # noqa: E402
from mindware import CASH  # noqa: E402
from mindware.components.utils.constants import CLASSIFICATION, REGRESSION  # noqa: E402
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase  # noqa: E402
from mindware.modules.configs import CONFIGS  # noqa: E402
from rl_reward_utils import RewardWeights  # noqa: E402
from run_p5_rl_rollout_buffer import (  # noqa: E402
    RLRolloutBuffer,
    install_rl_hook,
    uninstall_rl_hook,
)
from run_toy_categorical_optimizers import (  # noqa: E402
    build_optimizer_params,
    expand_optimizer_variants,
    history_scope_dir,
    load_real_data,
    resolve_variant,
    traj_label_for_variant,
)


def run_one(args) -> Path:
    os.chdir(REPO_ROOT)
    variant_name = expand_optimizer_variants(["ptllm_qwen"], args)[0]
    dispatch, output_name, _backend = resolve_variant(variant_name)
    task_const = CLASSIFICATION if args.task_type == "cls" else REGRESSION
    metric = args.metric or ("bal_acc" if args.task_type == "cls" else "mse")
    algorithms = list(args.algorithm)
    algo = algorithms[0]

    bo_path = Path(args.bo_history_json) if args.bo_history_json else bo_history_path(
        args.task_type, args.dataset, algo, metric=metric, root=Path(args.bo_history_root)
    )
    if not bo_path.is_file():
        raise FileNotFoundError(f"Missing BO history: {bo_path}")
    observations = load_bo_observations(bo_path, algorithm=algo, max_n=args.bo_seed_n)
    if len(observations) < args.bo_seed_n:
        print(
            f"[WARN] only {len(observations)} BO trials (requested {args.bo_seed_n}) from {bo_path}"
        )
    if not observations:
        raise RuntimeError(f"No successful BO observations in {bo_path}")

    n_seed = len(observations)
    n_total = n_seed + args.n_new_trials

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_dir = Path(args.history_dir).resolve()
    scope_dir = history_scope_dir(history_dir, args.task_type, metric, args.dataset, output_name)
    CONFIGS.ORIGINAL_DATA_DIR = str(history_dir)
    CONFIGS.HISTORY_RUN_SCOPE = str(scope_dir)

    _, train_node, _test_node, dataset_path, dataset_profile = load_real_data(args)
    # First LLM call must include BO-seeded history.
    args.llm_n_initial_samples = 0
    optimizer_params = build_optimizer_params(args, variant_name, algorithms, dataset_profile)

    KnowledgeDatabase._instance = None

    run_id = args.run_id or (
        f"{args.dataset}_{algo}_boseed{n_seed}_new{args.n_new_trials}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    )
    buffer_path = output_dir / (args.buffer_name or f"rollouts_{run_id}.jsonl")
    weights = RewardWeights(
        delta_best=args.reward_delta_best,
        absolute=args.reward_absolute,
        parse_fail=args.penalty_parse_fail,
        duplicate=args.penalty_duplicate,
        eval_fail=args.penalty_eval_fail,
    )
    meta = {
        "setup": "A_bo_seeded",
        "dataset": args.dataset,
        "task_type": args.task_type,
        "algorithm": algo,
        "metric": metric,
        "prompt_variant": args.prompt_variant,
        "variant_name": variant_name,
        "adapter_path": getattr(args, "qwen_adapter_path", None),
        "response_format": args.llm_response_format,
        "n_trials_new": args.n_new_trials,
        "bo_seed_n": n_seed,
        "bo_history_path": str(bo_path),
        "n_trials_total_budget": n_total,
        "seed": args.seed,
        "dataset_path": str(dataset_path),
        "llm_temperature": args.llm_temperature,
    }
    buffer = RLRolloutBuffer(buffer_path, run_id=run_id, meta=meta, weights=weights, resume=args.resume)
    install_rl_hook(buffer)

    task_id = f"{args.dataset}_rl_boseed_{run_id}"
    print(f"[SetupA] run_id={run_id}")
    print(f"[SetupA] BO seed={n_seed} from {bo_path}")
    print(f"[SetupA] new LLM trials={args.n_new_trials} total_budget={n_total} T={args.llm_temperature}")

    t0 = time.time()
    try:
        cash = CASH(
            include_algorithms=algorithms,
            task_type=task_const,
            metric=metric,
            data_node=train_node,
            evaluation=args.evaluation,
            resampling_params={"test_size": args.valid_size, "shuffle": False},
            optimizer=dispatch,
            time_limit=args.time_limit,
            amount_of_resource=n_total,
            per_run_time_limit=args.per_run_time_limit,
            inner_iter_num_per_iter=1,
            output_dir=str(output_dir / output_name),
            seed=args.seed,
            n_jobs=1,
            task_id=task_id,
            topk=n_total,
            optimizer_params=optimizer_params,
        )
        n_injected = seed_optimizer_with_bo(cash.optimizer, observations)
        print(f"[SetupA] injected {n_injected} BO trials into optimizer history")
        # Ensure warm-start count is 0 even if params were overridden.
        cash.optimizer.n_no_history_samples = 0
        cash.optimizer.n_initial_samples = 0
        cash.optimizer.evaluation_num_limit = n_total
        best_perf, stats_path = cash.run()
        print(f"[SetupA] done best_perf={best_perf} stats={stats_path}")
    finally:
        uninstall_rl_hook()

    elapsed = time.time() - t0
    manifest = buffer.write_manifest(
        {
            "elapsed_sec": elapsed,
            "cash_output_dir": str(output_dir / output_name),
            "traj_label": traj_label_for_variant(variant_name),
            "n_seeded_bo": n_injected,
            "n_logged_llm": len(buffer._seen_trials),
        }
    )
    print(f"[SetupA] wrote {len(buffer._seen_trials)} LLM records -> {buffer_path}")
    print(f"[SetupA] manifest -> {manifest}")
    return buffer_path


def parse_args():
    from run_toy_categorical_optimizers import DEFAULT_QWEN_MODEL_PATH

    p = argparse.ArgumentParser(description="Setup A: BO-seeded P5 RL rollouts")
    p.add_argument("--task-type", choices=["cls", "rgs"], required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--algorithm", nargs="+", required=True)
    p.add_argument("--metric", default=None)
    p.add_argument("--evaluation", default="holdout", choices=["holdout", "cv", "train_test"])
    p.add_argument("--valid-size", type=float, default=0.2)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--n-new-trials", type=int, default=4, help="LLM propose→eval steps after BO seed")
    p.add_argument("--bo-seed-n", type=int, default=100)
    p.add_argument("--bo-history-root", default=str(DEFAULT_BO_HISTORY_ROOT))
    p.add_argument("--bo-history-json", default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--time-limit", type=int, default=10800)
    p.add_argument("--per-run-time-limit", type=int, default=600)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--history-dir", default="benchmark_data/bo_history")
    p.add_argument("--run-id", default=None)
    p.add_argument("--buffer-name", default=None)
    p.add_argument("--resume", action="store_true")

    p.add_argument("--prompt-variant", default="P5")
    p.add_argument("--hybrid-variant", default="H0")
    p.add_argument("--llm-backends", default="qwen-sft6")
    p.add_argument("--qwen-backend", choices=["auto", "hf", "ollama"], default="hf")
    p.add_argument("--qwen-model-path", default=DEFAULT_QWEN_MODEL_PATH)
    p.add_argument("--qwen-adapter-path", required=True)
    p.add_argument("--qwen-enable-thinking", action="store_true")
    p.add_argument("--llm-mock", action="store_true")
    p.add_argument("--llm-response", action="append", default=None)
    p.add_argument("--llm-chat-engine", default=None)
    p.add_argument("--llm-model", default=None)
    p.add_argument("--llm-temperature", type=float, default=1.0)
    p.add_argument("--llm-max-tokens", type=int, default=768)
    p.add_argument("--llm-response-format", default="json_only", choices=["json_only", "rationale_json"])
    p.add_argument("--llm-timeout", type=int, default=180)
    p.add_argument("--llm-n-initial-samples", type=int, default=0)
    p.add_argument("--llm-history-window", type=int, default=20)
    p.add_argument("--llm-top-k", type=int, default=6)
    p.add_argument("--llm-recent-k", type=int, default=10)
    p.add_argument("--llm-worst-k", type=int, default=4)
    p.add_argument("--llm-run-summary", action="store_true", default=True)
    p.add_argument("--no-llm-run-summary", action="store_false", dest="llm_run_summary")
    p.add_argument("--llm-fallback-on-transport-error", action="store_true")
    p.add_argument("--llm-tpe-llm-prob", type=float, default=0.5)
    p.add_argument("--llm-tpe-phase-trials", type=int, default=0)
    p.add_argument("--llm-tpe-tpe-startup-trials", type=int, default=3)
    p.add_argument("--llm-tpe-tpe-duplicate-retries", type=int, default=5)
    p.add_argument("--tpe-startup-trials", type=int, default=3)
    p.add_argument("--preset-warm-start-json", default=None)

    p.add_argument("--reward-delta-best", type=float, default=1.0)
    p.add_argument("--reward-absolute", type=float, default=0.1)
    p.add_argument("--penalty-parse-fail", type=float, default=0.5)
    p.add_argument("--penalty-duplicate", type=float, default=0.1)
    p.add_argument("--penalty-eval-fail", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    args.llm_backends = ["qwen-sft6"]
    args.optimizer = ["ptllm_qwen"]
    if args.metric is None:
        args.metric = "bal_acc" if args.task_type == "cls" else "mse"
    apply_prompt_settings(args, args.prompt_variant)
    args.llm_n_initial_samples = 0
    if args.llm_max_tokens is None:
        args.llm_max_tokens = default_llm_max_tokens(args.llm_response_format)
    run_one(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Setup A: BO-seeded P5 RL rollouts (light-16, T=1.0).

Seed PT4Opt history from nested history_data BO JSON (default 100 trials),
set n_initial_samples=0 so the first LLM call already uses P5 history, then
run n_new CASH LLM propose→eval steps (default 4) that append onto the seed.

Init policy: s1a_ep7p5 LoRA (passed via --qwen-adapter-path).
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "examples" / "benchmark"))

from bo_history_utils import (  # noqa: E402
    DEFAULT_BO_HISTORY_ROOT,
    bo_history_path,
    load_bo_observations,
    seed_optimizer_with_bo,
)
from experiment_utils import apply_prompt_settings, default_llm_max_tokens  # noqa: E402
from mindware import CASH  # noqa: E402
from mindware.components.utils.constants import CLASSIFICATION, REGRESSION  # noqa: E402
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase  # noqa: E402
from mindware.modules.configs import CONFIGS  # noqa: E402
from rl_reward_utils import RewardWeights  # noqa: E402
from run_p5_rl_rollout_buffer import (  # noqa: E402
    RLRolloutBuffer,
    install_rl_hook,
    uninstall_rl_hook,
)
from run_toy_categorical_optimizers import (  # noqa: E402
    build_optimizer_params,
    expand_optimizer_variants,
    history_scope_dir,
    load_real_data,
    resolve_variant,
    traj_label_for_variant,
)


def run_one(args) -> Path:
    os.chdir(REPO_ROOT)
    variant_name = expand_optimizer_variants(["ptllm_qwen"], args)[0]
    dispatch, output_name, _backend = resolve_variant(variant_name)
    task_const = CLASSIFICATION if args.task_type == "cls" else REGRESSION
    metric = args.metric or ("bal_acc" if args.task_type == "cls" else "mse")
    algorithms = list(args.algorithm)
    algo = algorithms[0]

    bo_path = Path(args.bo_history_json) if args.bo_history_json else bo_history_path(
        args.task_type, args.dataset, algo, metric=metric, root=Path(args.bo_history_root)
    )
    if not bo_path.is_file():
        raise FileNotFoundError(f"Missing BO history: {bo_path}")
    observations = load_bo_observations(bo_path, algorithm=algo, max_n=args.bo_seed_n)
    if len(observations) < args.bo_seed_n:
        print(
            f"[WARN] only {len(observations)} BO trials (requested {args.bo_seed_n}) from {bo_path}"
        )
    if not observations:
        raise RuntimeError(f"No successful BO observations in {bo_path}")

    n_seed = len(observations)
    n_total = n_seed + args.n_new_trials

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_dir = Path(args.history_dir).resolve()
    scope_dir = history_scope_dir(history_dir, args.task_type, metric, args.dataset, output_name)
    CONFIGS.ORIGINAL_DATA_DIR = str(history_dir)
    CONFIGS.HISTORY_RUN_SCOPE = str(scope_dir)

    _, train_node, _test_node, dataset_path, dataset_profile = load_real_data(args)
    # First LLM call must include BO-seeded history.
    args.llm_n_initial_samples = 0
    optimizer_params = build_optimizer_params(args, variant_name, algorithms, dataset_profile)

    KnowledgeDatabase._instance = None

    run_id = args.run_id or (
        f"{args.dataset}_{algo}_boseed{n_seed}_new{args.n_new_trials}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    )
    buffer_path = output_dir / (args.buffer_name or f"rollouts_{run_id}.jsonl")
    weights = RewardWeights(
        delta_best=args.reward_delta_best,
        absolute=args.reward_absolute,
        parse_fail=args.penalty_parse_fail,
        duplicate=args.penalty_duplicate,
        eval_fail=args.penalty_eval_fail,
    )
    meta = {
        "setup": "A_bo_seeded",
        "dataset": args.dataset,
        "task_type": args.task_type,
        "algorithm": algo,
        "metric": metric,
        "prompt_variant": args.prompt_variant,
        "variant_name": variant_name,
        "adapter_path": getattr(args, "qwen_adapter_path", None),
        "response_format": args.llm_response_format,
        "n_trials_new": args.n_new_trials,
        "bo_seed_n": n_seed,
        "bo_history_path": str(bo_path),
        "n_trials_total_budget": n_total,
        "seed": args.seed,
        "dataset_path": str(dataset_path),
        "llm_temperature": args.llm_temperature,
    }
    buffer = RLRolloutBuffer(buffer_path, run_id=run_id, meta=meta, weights=weights, resume=args.resume)
    install_rl_hook(buffer)

    task_id = f"{args.dataset}_rl_boseed_{run_id}"
    print(f"[SetupA] run_id={run_id}")
    print(f"[SetupA] BO seed={n_seed} from {bo_path}")
    print(f"[SetupA] new LLM trials={args.n_new_trials} total_budget={n_total} T={args.llm_temperature}")

    t0 = time.time()
    try:
        cash = CASH(
            include_algorithms=algorithms,
            task_type=task_const,
            metric=metric,
            data_node=train_node,
            evaluation=args.evaluation,
            resampling_params={"test_size": args.valid_size, "shuffle": False},
            optimizer=dispatch,
            time_limit=args.time_limit,
            amount_of_resource=n_total,
            per_run_time_limit=args.per_run_time_limit,
            inner_iter_num_per_iter=1,
            output_dir=str(output_dir / output_name),
            seed=args.seed,
            n_jobs=1,
            task_id=task_id,
            topk=n_total,
            optimizer_params=optimizer_params,
        )
        n_injected = seed_optimizer_with_bo(cash.optimizer, observations)
        print(f"[SetupA] injected {n_injected} BO trials into optimizer history")
        # Ensure warm-start count is 0 even if params were overridden.
        cash.optimizer.n_no_history_samples = 0
        cash.optimizer.n_initial_samples = 0
        cash.optimizer.evaluation_num_limit = n_total
        best_perf, stats_path = cash.run()
        print(f"[SetupA] done best_perf={best_perf} stats={stats_path}")
    finally:
        uninstall_rl_hook()

    elapsed = time.time() - t0
    manifest = buffer.write_manifest(
        {
            "elapsed_sec": elapsed,
            "cash_output_dir": str(output_dir / output_name),
            "traj_label": traj_label_for_variant(variant_name),
            "n_seeded_bo": n_injected,
            "n_logged_llm": len(buffer._seen_trials),
        }
    )
    print(f"[SetupA] wrote {len(buffer._seen_trials)} LLM records -> {buffer_path}")
    print(f"[SetupA] manifest -> {manifest}")
    return buffer_path


def parse_args():
    from run_toy_categorical_optimizers import DEFAULT_QWEN_MODEL_PATH

    p = argparse.ArgumentParser(description="Setup A: BO-seeded P5 RL rollouts")
    p.add_argument("--task-type", choices=["cls", "rgs"], required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--algorithm", nargs="+", required=True)
    p.add_argument("--metric", default=None)
    p.add_argument("--evaluation", default="holdout", choices=["holdout", "cv", "train_test"])
    p.add_argument("--valid-size", type=float, default=0.2)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--n-new-trials", type=int, default=4, help="LLM propose→eval steps after BO seed")
    p.add_argument("--bo-seed-n", type=int, default=100)
    p.add_argument("--bo-history-root", default=str(DEFAULT_BO_HISTORY_ROOT))
    p.add_argument("--bo-history-json", default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--time-limit", type=int, default=10800)
    p.add_argument("--per-run-time-limit", type=int, default=600)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--history-dir", default="benchmark_data/bo_history")
    p.add_argument("--run-id", default=None)
    p.add_argument("--buffer-name", default=None)
    p.add_argument("--resume", action="store_true")

    p.add_argument("--prompt-variant", default="P5")
    p.add_argument("--hybrid-variant", default="H0")
    p.add_argument("--llm-backends", default="qwen-sft6")
    p.add_argument("--qwen-backend", choices=["auto", "hf", "ollama"], default="hf")
    p.add_argument("--qwen-model-path", default=DEFAULT_QWEN_MODEL_PATH)
    p.add_argument("--qwen-adapter-path", required=True)
    p.add_argument("--qwen-enable-thinking", action="store_true")
    p.add_argument("--llm-mock", action="store_true")
    p.add_argument("--llm-response", action="append", default=None)
    p.add_argument("--llm-chat-engine", default=None)
    p.add_argument("--llm-model", default=None)
    p.add_argument("--llm-temperature", type=float, default=1.0)
    p.add_argument("--llm-max-tokens", type=int, default=768)
    p.add_argument("--llm-response-format", default="json_only", choices=["json_only", "rationale_json"])
    p.add_argument("--llm-timeout", type=int, default=180)
    p.add_argument("--llm-n-initial-samples", type=int, default=0)
    p.add_argument("--llm-history-window", type=int, default=20)
    p.add_argument("--llm-top-k", type=int, default=6)
    p.add_argument("--llm-recent-k", type=int, default=10)
    p.add_argument("--llm-worst-k", type=int, default=4)
    p.add_argument("--llm-run-summary", action="store_true", default=True)
    p.add_argument("--no-llm-run-summary", action="store_false", dest="llm_run_summary")
    p.add_argument("--llm-fallback-on-transport-error", action="store_true")
    p.add_argument("--llm-tpe-llm-prob", type=float, default=0.5)
    p.add_argument("--llm-tpe-phase-trials", type=int, default=0)
    p.add_argument("--llm-tpe-tpe-startup-trials", type=int, default=3)
    p.add_argument("--llm-tpe-tpe-duplicate-retries", type=int, default=5)
    p.add_argument("--tpe-startup-trials", type=int, default=3)
    p.add_argument("--preset-warm-start-json", default=None)

    p.add_argument("--reward-delta-best", type=float, default=1.0)
    p.add_argument("--reward-absolute", type=float, default=0.1)
    p.add_argument("--penalty-parse-fail", type=float, default=0.5)
    p.add_argument("--penalty-duplicate", type=float, default=0.1)
    p.add_argument("--penalty-eval-fail", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    args.llm_backends = ["qwen-sft6"]
    args.optimizer = ["ptllm_qwen"]
    if args.metric is None:
        args.metric = "bal_acc" if args.task_type == "cls" else "mse"
    apply_prompt_settings(args, args.prompt_variant)
    args.llm_n_initial_samples = 0
    if args.llm_max_tokens is None:
        args.llm_max_tokens = default_llm_max_tokens(args.llm_response_format)
    run_one(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
