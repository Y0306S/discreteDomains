#!/usr/bin/env python3
"""Setup B: BO-prefix prompts × adaptive samples (default 4, up to 10).

Default (collect-schedule=none): one --prompt-variant for every prefix.

Mix schedule (--collect-schedule rs_mix):
  k=0,5,10       rs_warm  (P5plusRS / RSv4 / RSv5 via --rs-version)
  k=20,30,40     normal   (P5)
  k=50,70,90     explore  (P5plusA5Explore)
  k=60,80,100    exploit  (P5plusA5Exploit)

For each prefix k:
  - build history from the first k BO trials only (frozen; LLM evals are NOT appended)
  - sample 4 configs independently at temperature T and evaluate
  - if none beat incumbent (delta_best > 0), continue sampling until 10
  - train group: first 4, or 3 of first 4 + best extra improver (default)
    or strict first 4 only (--train-member-policy strict_first4)
  - log (messages, completion, reward, group_id, train_member)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "examples" / "benchmark"))

from bo_history_utils import (  # noqa: E402
    DEFAULT_BO_HISTORY_ROOT,
    bo_history_path,
    load_bo_observations,
    observations_to_perf_pairs,
    seed_optimizer_with_bo,
)
from experiment_utils import apply_prompt_settings, default_llm_max_tokens  # noqa: E402
from mindware import CASH  # noqa: E402
from mindware.components.optimizers.pt4opt_llm_optimizer import (  # noqa: E402
    PT4OptLLMOptimizer,
    _cfg_dict,
)
from mindware.components.utils.constants import CLASSIFICATION, REGRESSION, SUCCESS  # noqa: E402
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase  # noqa: E402
from mindware.modules.configs import CONFIGS  # noqa: E402
from p5plus_rl_collect_schedule import (  # noqa: E402
    MIX_PREFIXES,
    apply_prompt_to_optimizer,
    mix_jobs,
    mix_jobs_v13,
    mix_jobs_v13_gap,
    prompt_id_for_prefix,
    scenario_for_prefix,
)
from rl_reward_utils import RewardWeights, compute_trial_reward  # noqa: E402
from run_toy_categorical_optimizers import (  # noqa: E402
    build_optimizer_params,
    expand_optimizer_variants,
    history_scope_dir,
    load_real_data,
    resolve_variant,
)


def _append_record(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")


def _delta_best(record: Dict[str, Any]) -> float:
    rew = record.get("reward") or {}
    try:
        return float(rew.get("delta_best", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _reward_scalar(record: Dict[str, Any]) -> float:
    rew = record.get("reward") or {}
    try:
        return float(rew.get("reward", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _beats_incumbent(record: Dict[str, Any]) -> bool:
    return _delta_best(record) > 0.0


def _existing_job_counts(buffer_path: Path) -> Dict[Tuple[int, str], int]:
    counts: Dict[Tuple[int, str], int] = {}
    if not buffer_path.is_file():
        return counts
    for line in buffer_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        key = (int(rec["prefix_k"]), str(rec.get("scenario") or ""))
        counts[key] = counts.get(key, 0) + 1
    return counts


def collect_schedule_steps(args) -> List[Tuple[int, str, str]]:
    """Return (prefix_k, scenario, prompt_id) collect steps for this run."""
    sched = str(getattr(args, "collect_schedule", "none"))
    ver = str(getattr(args, "rs_version", "v3"))
    if sched == "rs_mix_v13":
        return mix_jobs_v13(rs_version=ver)
    if sched == "rs_mix_v13_gap":
        return mix_jobs_v13_gap(rs_version=ver)
    if sched == "rs_mix":
        return mix_jobs(args.prefixes, rs_version=ver)
    prompt_id = str(args.prompt_variant)
    return [(int(k), prompt_id, prompt_id) for k in args.prefixes]


def choose_train_members(
    records: List[Dict[str, Any]],
    *,
    n_base: int,
    rng: random.Random,
    policy: str = "adaptive",
) -> Tuple[List[int], Optional[int], Optional[int]]:
    """Return (train_idxs, dropped_base_idx, injected_extra_idx)."""
    n = len(records)
    base = list(range(min(n_base, n)))
    if policy == "strict_first4":
        return base, None, None
    if any(_beats_incumbent(records[i]) for i in base):
        return base, None, None
    extras = list(range(n_base, n))
    improvers = [i for i in extras if _beats_incumbent(records[i])]
    if not improvers:
        return base, None, None
    injected = max(improvers, key=lambda i: (_delta_best(records[i]), _reward_scalar(records[i]), i))
    drop = rng.choice(base)
    train = [i for i in base if i != drop] + [injected]
    return train, drop, injected


def sample_one(
    *,
    opt: PT4OptLLMOptimizer,
    args,
    sample_i: int,
    k: int,
    group_id: str,
    scenario: str,
    prompt_id: str,
    variant_name: str,
    run_id: str,
    algo: str,
    bo_path: Path,
    messages: List[Dict[str, str]],
    history_meta: Dict[str, Any],
    prior_perfs: List[float],
    weights: RewardWeights,
) -> Dict[str, Any]:
    trial_tag = f"k{k}_s{sample_i}"
    raw, payload, cfg, note = "", {}, None, ""
    parse_ok = False
    duplicate = False
    try:
        cfg, raw, payload, rationale, _cost, attempt = opt._request_config_attempt(
            messages[0]["content"],
            messages[1]["content"],
            attempt_id=sample_i,
        )
        note = rationale or "llm suggested configuration"
        parse_ok = bool(payload) and cfg is not None
        duplicate = bool(attempt.get("duplicate")) or opt._is_duplicate(cfg)
    except Exception as exc:
        note = f"llm_error: {exc}"
        raw = getattr(exc, "raw_response", raw) or str(exc)
        payload = getattr(exc, "parsed_payload", {}) or {}
        parse_ok = False
        cfg = opt._sample_nonduplicate_config()
        duplicate = True

    obs = opt.objective_function(cfg)
    raw_obj = float(obs.objectives[0]) if getattr(obs, "objectives", None) else np.inf
    eval_ok = getattr(obs, "trial_state", None) == SUCCESS and np.isfinite(raw_obj)
    perf = -raw_obj if eval_ok else float("-inf")

    reward = compute_trial_reward(
        perf=perf,
        prior_perfs=prior_perfs,
        parse_ok=parse_ok and not (isinstance(payload, dict) and not payload),
        duplicate=duplicate,
        eval_ok=eval_ok,
        weights=weights,
    )

    return {
        "run_id": run_id,
        "setup": "B_bo_prefix",
        "trial": trial_tag,
        "prefix_k": k,
        "sample_i": sample_i,
        "group_id": group_id,
        "scenario": scenario,
        "collect_schedule": getattr(args, "collect_schedule", "none"),
        "rs_version": getattr(args, "rs_version", None),
        "ts": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "task_type": args.task_type,
        "algorithm": algo,
        "prompt_variant": prompt_id,
        "variant_name": variant_name,
        "adapter_path": getattr(args, "qwen_adapter_path", None),
        "response_format": args.llm_response_format,
        "llm_temperature": args.llm_temperature,
        "bo_history_path": str(bo_path),
        "messages": messages,
        "completion": raw,
        "parsed_payload": payload,
        "normalized_config": _cfg_dict(cfg) if cfg is not None else None,
        "note": note,
        "history_meta": history_meta,
        "reward": reward.to_dict(),
    }


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
    all_obs = load_bo_observations(bo_path, algorithm=algo, max_n=args.bo_max_n)
    if len(all_obs) < max(args.prefixes):
        raise RuntimeError(
            f"Need at least {max(args.prefixes)} BO trials, found {len(all_obs)} in {bo_path}"
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_dir = Path(args.history_dir).resolve()
    scope_dir = history_scope_dir(history_dir, args.task_type, metric, args.dataset, output_name)
    CONFIGS.ORIGINAL_DATA_DIR = str(history_dir)
    CONFIGS.HISTORY_RUN_SCOPE = str(scope_dir)

    _, train_node, _test_node, dataset_path, dataset_profile = load_real_data(args)
    args.llm_n_initial_samples = 0
    optimizer_params = build_optimizer_params(args, variant_name, algorithms, dataset_profile)

    KnowledgeDatabase._instance = None
    run_id = args.run_id or (
        f"{args.dataset}_{algo}_prefixB_"
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

    # Create CASH once for evaluator + LLM client; seed with full BO then we
    # rebuild prompts from frozen prefix slices without appending LLM trials.
    cash = CASH(
        include_algorithms=algorithms,
        task_type=task_const,
        metric=metric,
        data_node=train_node,
        evaluation=args.evaluation,
        resampling_params={"test_size": args.valid_size, "shuffle": False},
        optimizer=dispatch,
        time_limit=args.time_limit,
        amount_of_resource=len(all_obs)
        + max(int(args.n_samples), int(getattr(args, "n_samples_max", args.n_samples)))
        * len(collect_schedule_steps(args))
        + 8,
        per_run_time_limit=args.per_run_time_limit,
        inner_iter_num_per_iter=1,
        output_dir=str(output_dir / output_name),
        seed=args.seed,
        n_jobs=1,
        task_id=f"{args.dataset}_rl_prefixB_{run_id}",
        topk=len(all_obs),
        optimizer_params=optimizer_params,
    )
    opt: PT4OptLLMOptimizer = cash.optimizer  # type: ignore[assignment]
    opt.n_no_history_samples = 0
    opt.n_initial_samples = 0

    print(f"[SetupB] run_id={run_id} dataset={args.dataset} algo={algo}")
    print(
        f"[SetupB] BO={bo_path} n={len(all_obs)} prefixes={args.prefixes} "
        f"samples/prefix={args.n_samples} schedule={getattr(args, 'collect_schedule', 'none')} "
        f"rs_version={getattr(args, 'rs_version', None)}"
    )

    t0 = time.time()
    n_logged = 0
    sched = str(getattr(args, "collect_schedule", "none"))
    use_mix = sched in ("rs_mix", "rs_mix_v13", "rs_mix_v13_gap")
    existing_jobs = _existing_job_counts(buffer_path)
    n_base = int(args.n_samples)
    n_max = int(getattr(args, "n_samples_max", n_base) or n_base)
    if n_max < n_base:
        n_max = n_base
    adaptive = n_max > n_base
    member_policy = str(getattr(args, "train_member_policy", "adaptive"))
    group_metas: List[Dict[str, Any]] = []
    for k, scenario, prompt_id in collect_schedule_steps(args):
        job_key = (int(k), str(scenario))
        if existing_jobs.get(job_key, 0) >= n_base:
            print(
                f"[SetupB] SKIP existing k={k} scenario={scenario} "
                f"n={existing_jobs[job_key]} (need {n_base})"
            )
            continue
        if use_mix:
            apply_prompt_settings(args, prompt_id)
            apply_prompt_to_optimizer(opt, args)
        prefix_obs = all_obs[:k]
        # Reset KD history to exactly this BO prefix (BO-only; no LLM appends).
        kd = opt.KD
        kd.configs.clear()
        kd.perfs.clear()
        if hasattr(kd, "configs_mab") and algo in kd.configs_mab:
            kd.configs_mab[algo].clear()
            kd.perfs_mab[algo].clear()
        opt.seen_signatures.clear()
        opt.seen_objectives.clear()
        opt._seen_config_count = 0
        opt.trial_sources.clear()
        opt.llm_history.clear()
        seed_optimizer_with_bo(opt, prefix_obs, source="bo_prefix")

        prior_perfs = [float(p) for p in opt.perfs if np.isfinite(p)]
        prompt_trial = max(int(k), 1)
        system_prompt, user_prompt = opt._build_history_prompts(
            include_history=True, current_trial=prompt_trial
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        group_id = f"{args.task_type}:{args.dataset}:{algo}:{scenario}:k{k}"
        history_meta = dict(getattr(opt, "_last_history_meta", {}) or {})

        group_records: List[Dict[str, Any]] = []
        for sample_i in range(n_base):
            rec = sample_one(
                opt=opt,
                args=args,
                sample_i=sample_i,
                k=int(k),
                group_id=group_id,
                scenario=scenario,
                prompt_id=prompt_id,
                variant_name=variant_name,
                run_id=run_id,
                algo=algo,
                bo_path=bo_path,
                messages=messages,
                history_meta=history_meta,
                prior_perfs=prior_perfs,
                weights=weights,
            )
            group_records.append(rec)

        extended = False
        if adaptive and not any(_beats_incumbent(r) for r in group_records):
            extended = True
            for sample_i in range(n_base, n_max):
                rec = sample_one(
                    opt=opt,
                    args=args,
                    sample_i=sample_i,
                    k=int(k),
                    group_id=group_id,
                    scenario=scenario,
                    prompt_id=prompt_id,
                    variant_name=variant_name,
                    run_id=run_id,
                    algo=algo,
                    bo_path=bo_path,
                    messages=messages,
                    history_meta=history_meta,
                    prior_perfs=prior_perfs,
                    weights=weights,
                )
                group_records.append(rec)

        rng = random.Random(int(args.seed) * 10007 + int(k) * 17)
        train_idxs, dropped, injected = choose_train_members(
            group_records, n_base=n_base, rng=rng, policy=member_policy
        )
        train_set = set(train_idxs)
        for rec in group_records:
            rec["n_sampled"] = len(group_records)
            rec["extended"] = extended
            rec["train_sample_idxs"] = train_idxs
            rec["train_member"] = int(rec["sample_i"]) in train_set
            rec["injected_from"] = injected
            rec["dropped_base_idx"] = dropped
            _append_record(buffer_path, rec)
            n_logged += 1

        group_metas.append(
            {
                "group_id": group_id,
                "prefix_k": int(k),
                "scenario": scenario,
                "n_sampled": len(group_records),
                "extended": extended,
                "train_sample_idxs": train_idxs,
                "injected_from": injected,
                "dropped_base_idx": dropped,
            }
        )
        print(
            f"[SetupB] finished prefix k={k} scenario={scenario} group={group_id} "
            f"n={len(group_records)} train={train_idxs} extended={extended}"
        )

    elapsed = time.time() - t0
    manifest = {
        "run_id": run_id,
        "setup": "B_bo_prefix",
        "collect_schedule": getattr(args, "collect_schedule", "none"),
        "rs_version": getattr(args, "rs_version", None),
        "buffer_path": str(buffer_path),
        "n_records": n_logged,
        "prefixes": args.prefixes,
        "n_samples": args.n_samples,
        "n_samples_max": getattr(args, "n_samples_max", args.n_samples),
        "groups": group_metas,
        "bo_history_path": str(bo_path),
        "elapsed_sec": elapsed,
        "meta": {
            "dataset": args.dataset,
            "task_type": args.task_type,
            "algorithm": algo,
            "metric": metric,
            "adapter_path": getattr(args, "qwen_adapter_path", None),
            "dataset_path": str(dataset_path),
        },
        "weights": {
            "delta_best": weights.delta_best,
            "absolute": weights.absolute,
            "parse_fail": weights.parse_fail,
            "duplicate": weights.duplicate,
            "eval_fail": weights.eval_fail,
        },
    }
    man_path = buffer_path.with_suffix(".manifest.json")
    man_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"[SetupB] wrote {n_logged} records -> {buffer_path}")
    print(f"[SetupB] manifest -> {man_path}")
    return buffer_path


def parse_args():
    from run_toy_categorical_optimizers import DEFAULT_QWEN_MODEL_PATH

    p = argparse.ArgumentParser(description="Setup B: BO-prefix P5 × 4-sample groups")
    p.add_argument("--task-type", choices=["cls", "rgs"], required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--algorithm", nargs="+", required=True)
    p.add_argument("--metric", default=None)
    p.add_argument("--evaluation", default="holdout", choices=["holdout", "cv", "train_test"])
    p.add_argument("--valid-size", type=float, default=0.2)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--prefixes", type=int, nargs="+", default=None)
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument(
        "--n-samples-max",
        type=int,
        default=10,
        help="If first --n-samples all miss incumbent, continue sampling up to this many.",
    )
    p.add_argument("--bo-max-n", type=int, default=100)
    p.add_argument("--bo-history-root", default=str(DEFAULT_BO_HISTORY_ROOT))
    p.add_argument("--bo-history-json", default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--time-limit", type=int, default=10800)
    p.add_argument("--per-run-time-limit", type=int, default=600)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--history-dir", default="benchmark_data/bo_history")
    p.add_argument("--run-id", default=None)
    p.add_argument("--buffer-name", default=None)

    p.add_argument(
        "--collect-schedule",
        choices=["none", "rs_mix", "rs_mix_v13", "rs_mix_v13_gap"],
        default="none",
    )
    p.add_argument(
        "--train-member-policy",
        choices=["adaptive", "strict_first4"],
        default="adaptive",
        help="How to mark train_member within each prefix group (default: adaptive 3+1).",
    )
    p.add_argument("--rs-version", choices=["v3", "v4", "v5"], default="v3")
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
    args = p.parse_args()
    if args.prefixes is None:
        if args.collect_schedule == "rs_mix_v13":
            args.prefixes = sorted({k for k, _, _ in mix_jobs_v13(rs_version=args.rs_version)})
        elif args.collect_schedule == "rs_mix_v13_gap":
            args.prefixes = sorted({k for k, _, _ in mix_jobs_v13_gap(rs_version=args.rs_version)})
        elif args.collect_schedule == "rs_mix":
            args.prefixes = list(MIX_PREFIXES)
        else:
            args.prefixes = list(range(10, 101, 10))
    need = max(args.prefixes) if args.prefixes else 0
    if need > 0:
        pass  # validated in run_one against BO length
    if args.n_samples_max < args.n_samples:
        args.n_samples_max = args.n_samples
    if args.collect_schedule in ("rs_mix", "rs_mix_v13", "rs_mix_v13_gap"):
        collect_schedule_steps(args)
    return args


def main():
    args = parse_args()
    args.llm_backends = ["qwen-sft6"]
    args.optimizer = ["ptllm_qwen"]
    if args.metric is None:
        args.metric = "bal_acc" if args.task_type == "cls" else "mse"
    if args.collect_schedule in ("rs_mix", "rs_mix_v13"):
        steps = collect_schedule_steps(args)
        args.prompt_variant = steps[0][2] if steps else prompt_id_for_prefix(int(args.prefixes[0]), args.rs_version)
    apply_prompt_settings(args, args.prompt_variant)
    args.llm_n_initial_samples = 0
    if args.llm_max_tokens is None:
        args.llm_max_tokens = default_llm_max_tokens(args.llm_response_format)
    run_one(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
