#!/usr/bin/env python3
"""Top up a short shared BO history JSON to N trials using an LLM adapter.

Seeds PT4Opt from existing history_data observations, runs (target_n - n_seed)
LLM propose→eval steps (default: rationale_G P5/H0 ptllm_tpe), then writes the
combined observations back to the same BO JSON path so Setup B prefixes 10..100 work.
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
from run_toy_categorical_optimizers import (  # noqa: E402
    build_optimizer_params,
    expand_optimizer_variants,
    history_scope_dir,
    load_real_data,
    resolve_variant,
)


def _cfg_to_dict(cfg) -> dict:
    if hasattr(cfg, "get_dictionary"):
        return dict(cfg.get_dictionary())
    if isinstance(cfg, dict):
        return dict(cfg)
    raise TypeError(f"unsupported config type: {type(cfg)}")


def write_bo_json(bo_path: Path, original: dict, cash, algo: str, target_n: int) -> int:
    """Overwrite shared BO JSON observations from KD configs/perfs (higher-is-better)."""
    kd = cash.optimizer.KD
    configs = list(kd.configs)
    perfs = list(kd.perfs)
    if len(configs) != len(perfs):
        raise RuntimeError(f"KD configs/perfs length mismatch {len(configs)} vs {len(perfs)}")
    n = min(len(configs), target_n)
    observations = []
    now = datetime.now(timezone.utc).isoformat()
    for i in range(n):
        observations.append(
            {
                "config": _cfg_to_dict(configs[i]),
                "objectives": [-float(perfs[i])],  # OpenBox minimize
                "constraints": None,
                "trial_state": 0,
                "elapsed_time": None,
                "create_time": now,
                "extra_info": {"source": "topup_rationale_G" if i >= len(original.get("observations") or []) else "original_bo"},
            }
        )
    out = dict(original)
    out["observations"] = observations
    out["topup_meta"] = {
        "target_n": target_n,
        "n_written": n,
        "algo": algo,
        "ts": now,
    }
    bo_path.parent.mkdir(parents=True, exist_ok=True)
    bo_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return n


def run_one(args) -> Path:
    os.chdir(REPO_ROOT)
    variant_name = expand_optimizer_variants(["ptllm_tpe_qwen"], args)[0]
    dispatch, output_name, _backend = resolve_variant(variant_name)
    task_const = CLASSIFICATION if args.task_type == "cls" else REGRESSION
    metric = args.metric or ("bal_acc" if args.task_type == "cls" else "mse")
    algo = args.algorithm[0]

    bo_path = Path(args.bo_history_json) if args.bo_history_json else bo_history_path(
        args.task_type, args.dataset, algo, metric=metric, root=Path(args.bo_history_root)
    )
    if not bo_path.is_file():
        raise FileNotFoundError(bo_path)
    original = json.loads(bo_path.read_text(encoding="utf-8"))
    observations = load_bo_observations(bo_path, algorithm=algo)
    n_seed = len(observations)
    if n_seed >= args.target_n:
        print(f"[TopUp] already have {n_seed} >= {args.target_n} in {bo_path} — skip")
        return bo_path
    n_new = args.target_n - n_seed
    n_total = args.target_n

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_dir = Path(args.history_dir).resolve()
    scope_dir = history_scope_dir(history_dir, args.task_type, metric, args.dataset, output_name)
    CONFIGS.ORIGINAL_DATA_DIR = str(history_dir)
    CONFIGS.HISTORY_RUN_SCOPE = str(scope_dir)

    _, train_node, _test_node, dataset_path, dataset_profile = load_real_data(args)
    args.llm_n_initial_samples = 0
    optimizer_params = build_optimizer_params(args, variant_name, [algo], dataset_profile)
    KnowledgeDatabase._instance = None

    run_id = (
        f"{args.dataset}_{algo}_topup{n_seed}to{args.target_n}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    )
    print(f"[TopUp] {args.task_type}/{args.dataset}/{algo}")
    print(f"[TopUp] BO={bo_path} seed={n_seed} need_new={n_new} target={args.target_n}")
    print(f"[TopUp] adapter={args.qwen_adapter_path} format={args.llm_response_format} P{args.prompt_variant}/H{args.hybrid_variant}")

    t0 = time.time()
    cash = CASH(
        include_algorithms=[algo],
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
        task_id=f"{args.dataset}_bo_topup_{run_id}",
        topk=n_total,
        optimizer_params=optimizer_params,
    )
    n_injected = seed_optimizer_with_bo(cash.optimizer, observations, source="bo_history")
    cash.optimizer.n_no_history_samples = 0
    cash.optimizer.n_initial_samples = 0
    cash.optimizer.evaluation_num_limit = n_total
    print(f"[TopUp] injected {n_injected}; running {n_new} LLM trials…")
    best_perf, stats_path = cash.run()
    n_written = write_bo_json(bo_path, original, cash, algo, args.target_n)
    elapsed = time.time() - t0
    print(f"[TopUp] done best={best_perf} wrote_obs={n_written} -> {bo_path} ({elapsed:.0f}s) stats={stats_path}")
    # verify
    check = load_bo_observations(bo_path, algorithm=algo)
    print(f"[TopUp] verify load_bo_observations n={len(check)}")
    if len(check) < args.target_n:
        raise RuntimeError(f"Top-up incomplete: {len(check)} < {args.target_n}")
    return bo_path


def parse_args():
    from run_toy_categorical_optimizers import DEFAULT_QWEN_MODEL_PATH

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-type", choices=["cls", "rgs"], required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--algorithm", nargs="+", required=True)
    p.add_argument("--metric", default=None)
    p.add_argument("--evaluation", default="holdout", choices=["holdout", "cv", "train_test"])
    p.add_argument("--valid-size", type=float, default=0.2)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--target-n", type=int, default=100)
    p.add_argument("--bo-history-root", default=str(DEFAULT_BO_HISTORY_ROOT))
    p.add_argument("--bo-history-json", default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--time-limit", type=int, default=21600)
    p.add_argument("--per-run-time-limit", type=int, default=600)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--history-dir", default="benchmark_data/bo_history_topup")

    p.add_argument("--prompt-variant", default="P5")
    p.add_argument("--hybrid-variant", default="H0")
    p.add_argument("--llm-backends", default="qwen-sft6")
    p.add_argument("--qwen-backend", choices=["auto", "hf", "ollama"], default="hf")
    p.add_argument("--qwen-model-path", default=DEFAULT_QWEN_MODEL_PATH)
    p.add_argument(
        "--qwen-adapter-path",
        default="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_rationale_G",
    )
    p.add_argument("--qwen-enable-thinking", action="store_true")
    p.add_argument("--llm-mock", action="store_true")
    p.add_argument("--llm-response", action="append", default=None)
    p.add_argument("--llm-chat-engine", default=None)
    p.add_argument("--llm-model", default=None)
    p.add_argument("--llm-temperature", type=float, default=0.7)
    p.add_argument("--llm-max-tokens", type=int, default=None)
    p.add_argument("--llm-response-format", default="rationale_json", choices=["json_only", "rationale_json"])
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
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.llm_backends = ["qwen-sft6"]
    args.optimizer = ["ptllm_tpe_qwen"]
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
