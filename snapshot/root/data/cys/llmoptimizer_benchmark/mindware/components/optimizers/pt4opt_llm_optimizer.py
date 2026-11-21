import csv
import json
import os
import re
import time
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)
from openbox.utils.constants import SUCCESS

try:
    from ConfigSpace.hyperparameters import OrdinalHyperparameter
except ImportError:
    OrdinalHyperparameter = ()

from mindware.components.optimizers.base_optimizer import BaseOptimizer, MAX_INT
from mindware.components.optimizers.llm_prompt_blocks import (
    build_best_so_far_trajectory_block,
    build_budget_context_block,
    build_coverage_map_block,
    build_explore_refine_summary,
    build_experience_self_block,
    build_hp_marginals_block,
    build_playbook_run_context_block,
    build_rationale_diversity_summary,
    build_tuning_playbook_system_extension,
    format_tpe_hint_block,
    validate_prompt_blocks,
)
from mindware.components.optimizers.tcs_report import (
    _hp_names_from_space,
    build_analysis_agent_prompts,
    build_tcs_report,
    build_tcs_single_agent_prompts,
)
from mindware.components.optimizers.llm_optimizer_utils import wrapper_evaluater
from mindware.components.optimizers.utils import MyConfiguration
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase
from mindware.modules.configs import AVAILABLE_LLMs, LLMCONFIGS, LLM_COST_DICT


def _hp_value(hp):
    return getattr(hp, "value", getattr(hp, "default_value", None))


def _hp_default(hp):
    return getattr(hp, "default_value", getattr(hp, "value", None))


def _hp_choices(hp) -> List[Any]:
    if isinstance(hp, (Constant, UnParametrizedHyperparameter)):
        return [_hp_value(hp)]
    if hasattr(hp, "choices"):
        return list(hp.choices)
    if hasattr(hp, "sequence"):
        return list(hp.sequence)
    return [_hp_default(hp)]


def _cfg_dict(cfg) -> Dict[str, Any]:
    return cfg.get_dictionary().copy() if hasattr(cfg, "get_dictionary") else dict(cfg)


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, default=str)


def _lines(text: str) -> List[str]:
    return (text or "").splitlines()


def _signature(cfg) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((str(k), str(v)) for k, v in _cfg_dict(cfg).items()))


def _config_for_prompt(cfg, fixed_algorithm: str, config_space) -> Dict[str, Any]:
    cfg_dict = _cfg_dict(cfg)
    prompt_config = {}
    for hp in config_space.get_hyperparameters():
        name = hp.name
        if name == "algorithm" or isinstance(hp, (Constant, UnParametrizedHyperparameter)):
            continue
        if name not in cfg_dict:
            continue
        prompt_config[name] = cfg_dict[name]
    return prompt_config


def _allocate_best_worst_counts(n_items: int, top_k: int, worst_k: int) -> Tuple[int, int]:
    top_req = max(0, int(top_k))
    worst_req = max(0, int(worst_k))
    budget = min(max(0, int(n_items)), top_req + worst_req)
    if budget <= 0 or top_req + worst_req == 0:
        return 0, 0
    if top_req == 0:
        return 0, min(worst_req, budget)
    if worst_req == 0:
        return min(top_req, budget), 0
    if budget == 1:
        return (1, 0) if top_req >= worst_req else (0, 1)

    base_top, base_worst = 1, 1
    remaining = budget - 2
    top_remaining_req = top_req - base_top
    worst_remaining_req = worst_req - base_worst
    remaining_req = top_remaining_req + worst_remaining_req
    if remaining <= 0 or remaining_req <= 0:
        return base_top, base_worst

    quotas = [
        ("best", remaining * top_remaining_req / remaining_req, top_remaining_req),
        ("worst", remaining * worst_remaining_req / remaining_req, worst_remaining_req),
    ]
    extra = {name: min(int(quota), cap) for name, quota, cap in quotas}
    left = remaining - sum(extra.values())
    for name, quota, cap in sorted(quotas, key=lambda x: x[1] - int(x[1]), reverse=True):
        if left <= 0:
            break
        if extra[name] < cap:
            extra[name] += 1
            left -= 1
    return base_top + extra["best"], base_worst + extra["worst"]


def build_run_summary(
    successes: List[dict],
    top_trials: set,
    worst_trials: set,
    *,
    higher_is_better: bool = True,
) -> str:
    """Rule-based summary of observed tuning patterns (no extra LLM call)."""
    if not successes:
        return "Run summary: no successful trials yet."

    n = len(successes)
    best_score = max(x["score"] for x in successes) if higher_is_better else min(x["score"] for x in successes)
    lines = [
        "Run summary:",
        f"- Evaluated {n} successful trials; best validation rel_score=1.0 (rank 1).",
    ]

    top_items = [x for x in successes if int(x["trial"]) in top_trials]
    worst_items = [x for x in successes if int(x["trial"]) in worst_trials]
    if not top_items or not worst_items:
        lines.append("- Not enough top/worst separation yet to infer hyperparameter trends.")
        return "\n".join(lines)

    numeric_keys = set()
    for item in top_items + worst_items:
        for key, value in item["config"].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_keys.add(key)

    deltas = []
    for key in sorted(numeric_keys):
        top_vals = [float(item["config"][key]) for item in top_items if key in item["config"]]
        worst_vals = [float(item["config"][key]) for item in worst_items if key in item["config"]]
        if not top_vals or not worst_vals:
            continue
        top_mean = float(np.mean(top_vals))
        worst_mean = float(np.mean(worst_vals))
        denom = max(abs(top_mean), abs(worst_mean), 1e-12)
        rel_gap = abs(top_mean - worst_mean) / denom
        if rel_gap < 0.05:
            continue
        direction = "higher" if top_mean > worst_mean else "lower"
        deltas.append((rel_gap, key, direction, top_mean, worst_mean))

    deltas.sort(reverse=True)
    if deltas:
        lines.append("- Hyperparameters that differ most between top and worst trials:")
        for _, key, direction, top_mean, worst_mean in deltas[:3]:
            lines.append(
                f"  - {key}: top trials tend toward {direction} values "
                f"(top mean={top_mean:.4g}, worst mean={worst_mean:.4g})."
            )
    else:
        lines.append("- Top and worst trials do not show a clear numeric hyperparameter gap yet.")
    return "\n".join(lines)


def _experience_explore_branch(
    *,
    seed: int,
    trial: int,
    mode_tag: str,
    explore_prob: float = 0.5,
) -> str:
    """Deterministic explore/exploit coin flip (reproducible across resumes)."""
    import hashlib

    p = float(explore_prob)
    p = 0.0 if p < 0.0 else (1.0 if p > 1.0 else p)
    key = f"{int(seed)}:{int(trial)}:{mode_tag}:ee50".encode("utf-8")
    u = int(hashlib.md5(key).hexdigest()[:8], 16) / float(0xFFFFFFFF)
    return "explore" if u < p else "exploit"


def build_history_prompt_block(
    configs,
    perfs,
    fixed_algorithm,
    config_space,
    history_window: int = 20,
    top_k: int = 6,
    recent_k: int = 10,
    worst_k: int = 4,
    run_summary: bool = False,
    *,
    prompt_blocks: Optional[Dict[str, bool]] = None,
    budget_total: Optional[int] = None,
    trial_sources: Optional[Dict[int, str]] = None,
    llm_history: Optional[List[dict]] = None,
    experience_self_mode: str = "cohort",
    current_trial: Optional[int] = None,
    experience_self_start_trial: int = 1,
    experience_explore_mode: str = "off",
    experience_explore_prob: float = 0.5,
    experience_blame: Any = None,
    rng_seed: int = 1,
    experience_bank: Any = None,
    experience_retrieve_k: int = 3,
    experience_retrieve_start_trial: int = 1,
    experience_retrieve_rerank_trial: int = 24,
    experience_retrieve_ranker: str = "surrogate",
    experience_retrieve_fade: bool = False,
    experience_retrieve_stop_trial: Optional[int] = None,
    experience_retrieve_spearman_floor: Optional[float] = None,
    experience_retrieve_include_top_bottom: bool = False,
    experience_retrieve_include_prefer: bool = True,
    experience_retrieve_region_style: str = "prefer",
    experience_retrieve_tb_layout: str = "flat",
    experience_retrieve_top_k: int = 1,
    experience_retrieve_bottom_k: int = 1,
    experience_self_gate: str = "clock",
    experience_gate_state: Any = None,
    explore_policy_state: Any = None,
    dataset_id: Optional[str] = None,
    task_type: Optional[str] = None,
    dataset_profile: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Build the reusable history section for the LLM prompt."""
    blocks = validate_prompt_blocks(prompt_blocks)
    ee_mode = str(experience_explore_mode or "off").strip().lower()
    ee_prob = float(experience_explore_prob if experience_explore_prob is not None else 0.5)
    trial_sources = trial_sources or {}
    llm_history = llm_history or []
    successes = []
    for trial, (cfg, perf) in enumerate(zip(configs or [], perfs or []), start=1):
        cfg_dict = _cfg_dict(cfg)
        if cfg_dict.get("algorithm", fixed_algorithm) != fixed_algorithm:
            continue
        score = float(perf)
        if np.isfinite(score):
            successes.append(
                {
                    "trial": trial,
                    "score": score,
                    "config": _config_for_prompt(cfg_dict, fixed_algorithm, config_space),
                }
            )

    # Per-dataset min-max normalization. One prompt only ever holds a single
    # dataset's history, so a single (lo, hi) over the full finite history gives
    # the LLM a consistent [0, 1] scale across datasets (1.0 = best observed),
    # with no raw, dataset-specific magnitudes leaking into SFT data.
    all_scores = [x["score"] for x in successes]
    n_success = len(all_scores)
    hi_score = max(all_scores) if all_scores else 0.0
    lo_score = min(all_scores) if all_scores else 0.0
    span = hi_score - lo_score

    def _rel(score: float) -> float:
        # Degenerate range (all trials tied, e.g. every config scores the same):
        # no signal to convey, so report the top of the scale rather than NaN.
        return 1.0 if span <= 0 else (score - lo_score) / span

    # Competition ranking (1 = best); tied scores share a rank.
    rank_by_trial = {
        int(x["trial"]): 1 + sum(1 for s in all_scores if s > x["score"])
        for x in successes
    }
    rel_by_trial = {int(x["trial"]): _rel(x["score"]) for x in successes}

    best_count, worst_count = _allocate_best_worst_counts(len(successes), top_k, worst_k)
    top = sorted(successes, key=lambda x: x["score"], reverse=True)[:best_count]
    top_trials = {int(x["trial"]) for x in top}
    worst = []
    for item in sorted(successes, key=lambda x: x["score"]):
        if int(item["trial"]) in top_trials:
            continue
        worst.append(item)
        if len(worst) >= worst_count:
            break
    recent = successes[-max(0, recent_k) :] if recent_k > 0 else []

    selected_by_trial = {}
    for items, tag in [(top, "best"), (worst, "worst"), (recent, "recent")]:
        for item in items:
            trial = int(item["trial"])
            selected = selected_by_trial.setdefault(
                trial,
                {
                    "trial": trial,
                    "score": item["score"],
                    "config": item["config"],
                    "tags": [],
                },
            )
            if tag not in selected["tags"]:
                selected["tags"].append(tag)

    selected = [selected_by_trial[k] for k in sorted(selected_by_trial)]
    if len(selected) > history_window:
        priority = {"best": 3, "worst": 2, "recent": 1}

        def trim_key(item):
            tags = item["tags"]
            score_key = -item["score"] if "worst" in tags and "best" not in tags else item["score"]
            return max(priority[tag] for tag in tags), score_key, item["trial"]

        selected = sorted(
            selected,
            key=trim_key,
            reverse=True,
        )[:history_window]
    selected = sorted(selected, key=lambda x: (-x["score"], int(x["trial"])))

    lines = [
        f"Previous validation trials for fixed algorithm {fixed_algorithm}.",
        "rel_score in [0,1] is the validation score min-max normalized within this "
        "dataset (1.0 = best observed, higher is better); rank is the 1-based "
        f"standing among all {n_success} evaluated trials.",
    ]

    extra_sections: List[str] = []
    if blocks.get("budget_context"):
        extra_sections.append(
            build_budget_context_block(
                successes,
                budget_total=budget_total,
                rel_by_trial=rel_by_trial,
                rank_by_trial=rank_by_trial,
            )
        )
    if blocks.get("playbook_run_context"):
        extra_sections.append(
            build_playbook_run_context_block(successes, budget_total=budget_total)
        )
    if blocks.get("best_so_far_trajectory"):
        extra_sections.append(build_best_so_far_trajectory_block(successes, rel_by_trial))
    if extra_sections:
        lines += [""] + extra_sections

    if selected:
        lines += ["", "Selected historical trials:"]
        for i, item in enumerate(selected, start=1):
            trial = int(item["trial"])
            source_tag = ""
            if blocks.get("source_tags"):
                src = trial_sources.get(trial, "unknown")
                source_tag = f"source={src}, "
            lines.append(
                f"{i}. trial={item['trial']}, {source_tag}rel_score={_rel(item['score']):.4f}, "
                f"rank={rank_by_trial[trial]}/{n_success}, "
                f"tags={_json(item['tags'])}, config={_json(item['config'])}"
            )

    if blocks.get("hp_marginals"):
        lines += ["", build_hp_marginals_block(successes, config_space, fixed_algorithm, rel_by_trial)]

    # Static coverage_map only when not using fifty_fifty (that mode owns coverage as explore).
    if blocks.get("coverage_map") and ee_mode != "fifty_fifty":
        lines += ["", build_coverage_map_block(successes, config_space, fixed_algorithm)]

    if run_summary and successes:
        if blocks.get("explore_refine_summary"):
            summary = build_explore_refine_summary(successes, config_space, fixed_algorithm)
        else:
            summary = build_run_summary(successes, top_trials, {int(x["trial"]) for x in worst})
        if blocks.get("rationale_diversity_summary"):
            summary += "\n\n" + build_rationale_diversity_summary(successes, llm_history, rel_by_trial)
        lines += ["", summary]

    experience_branch = None
    experience_self_active = False
    retrieved_hits_meta: List[Dict[str, Any]] = []
    gate_meta: Dict[str, Any] = {}
    if blocks.get("experience_retrieve") and experience_bank is not None:
        trial_no = int(current_trial) if current_trial is not None else 10**9
        try:
            from mindware.components.experience.retrieve import retrieve_for_run
        except Exception:
            retrieve_for_run = None  # type: ignore
        if retrieve_for_run is not None:
            fade_after = None
            if experience_retrieve_fade and blocks.get("experience_self"):
                fade_after = int(experience_self_start_trial or 1)
            stop_trial = None
            if experience_retrieve_stop_trial not in (None, "", 0, "0"):
                stop_trial = int(experience_retrieve_stop_trial)
            ret_block, hits = retrieve_for_run(
                experience_bank,
                dataset_id=dataset_id,
                algorithm_id=fixed_algorithm,
                task_type=task_type,
                dataset_profile=dataset_profile,
                successes=successes,
                config_space=config_space,
                k=int(experience_retrieve_k or 3),
                start_trial=int(experience_retrieve_start_trial or 1),
                rerank_after=int(experience_retrieve_rerank_trial or 24),
                current_trial=trial_no,
                fade_after=fade_after,
                stop_trial=stop_trial,
                ranker=str(experience_retrieve_ranker or "surrogate"),
                spearman_floor=experience_retrieve_spearman_floor,
                include_top_bottom=bool(experience_retrieve_include_top_bottom),
                include_prefer=bool(experience_retrieve_include_prefer),
                region_style=str(experience_retrieve_region_style or "prefer"),
                tb_layout=str(experience_retrieve_tb_layout or "flat"),
                retrieve_top_k=1 if experience_retrieve_top_k is None else max(0, int(experience_retrieve_top_k)),
                retrieve_bottom_k=1 if experience_retrieve_bottom_k is None else max(0, int(experience_retrieve_bottom_k)),
            )
            if ret_block:
                lines += ["", ret_block]
            retrieved_hits_meta = [
                {
                    "id": h.exp_id,
                    "score": h.score,
                    "spearman": h.spearman,
                    "features": h.features,
                }
                for h in hits
            ]

    if blocks.get("experience_self"):
        trial_no = int(current_trial) if current_trial is not None else 10**9
        start_at = max(1, int(experience_self_start_trial or 1))
        gate_mode = str(experience_self_gate or "clock").strip().lower()
        from mindware.components.optimizers.experience_gate import ExperienceGate, apply_strength, filter_fanova

        gate = experience_gate_state
        if gate is None:
            gate = ExperienceGate(mode=gate_mode, cap=start_at)

        exp_block = build_experience_self_block(
            successes,
            min_trials=8,
            mode=experience_self_mode or "cohort",
            config_space=config_space,
            experience_blame=experience_blame,
        )
        active, gate_meta = gate.decide(
            current_trial=trial_no, successes=successes, block=exp_block or ""
        )
        if gate_mode in ("clock", "t50", ""):
            active = trial_no >= start_at

        experience_self_active = bool(active)
        if active:
            if gate_mode in ("fanova", "fanova_full", "fanova_full_cs", "fanova_full_ix", "fanova_rf"):
                engine = {
                    "fanova": "proxy",
                    "fanova_full": "full",
                    "fanova_full_cs": "full_cs",
                    "fanova_full_ix": "full_ix",
                    "fanova_rf": "rf",
                }[gate_mode]
                exp_block = filter_fanova(
                    exp_block or "",
                    successes,
                    gate.fanova_thresh,
                    engine=engine,
                    config_space=config_space,
                )
                from mindware.components.optimizers.experience_gate import _LAST_FANOVA_META

                gate_meta.update(_LAST_FANOVA_META)
            if gate_mode == "strength":
                exp_block = apply_strength(exp_block or "", len(successes), gate.floor, gate.cap)
            if gate_mode == "turbo":
                from mindware.components.optimizers.experience_gate import apply_turbo_width

                exp_block = apply_turbo_width(exp_block or "", gate.turbo_L)
            if ee_mode in ("off", "", "none"):
                experience_branch = "exploit"
            else:
                from mindware.components.optimizers.experience_gate import ExplorePolicy, XOR_EXPLORE_MODES

                policy = explore_policy_state
                if policy is None:
                    policy = ExplorePolicy(
                        mode=ee_mode,
                        seed=int(rng_seed),
                        p_default=ee_prob,
                    )
                if ee_mode in XOR_EXPLORE_MODES or ee_mode == "fifty_fifty":
                    experience_branch = policy.choose(
                        trial=int(trial_no),
                        successes=successes,
                        config_space=config_space,
                        fixed_algorithm=fixed_algorithm,
                        mode_tag=str(experience_self_mode or "cohort"),
                        exp_block=exp_block or "",
                    )
                    ee_prob = float(getattr(policy, "last_p", ee_prob))
                    if getattr(policy, "last_meta", None):
                        gate_meta.update(policy.last_meta)
                else:
                    experience_branch = "exploit"

            if experience_branch == "exploit":
                if exp_block:
                    lines += ["", exp_block]
            elif experience_branch == "explore":
                cov_block = build_coverage_map_block(successes, config_space, fixed_algorithm)
                if cov_block:
                    lines += ["", cov_block]

    meta = {
        "selected_best_trials": [x["trial"] for x in top],
        "selected_recent_trials": [x["trial"] for x in recent],
        "selected_worst_trials": [x["trial"] for x in worst],
        "selected_trials": [{"trial": x["trial"], "tags": x["tags"]} for x in selected],
        "allocated_best_k": best_count,
        "allocated_worst_k": worst_count,
        "history_window": history_window,
        "run_summary": bool(run_summary),
        "prompt_blocks": blocks,
        "experience_self_start_trial": int(experience_self_start_trial or 1),
        "experience_self_active": experience_self_active,
        "experience_explore_mode": ee_mode,
        "experience_explore_prob": ee_prob,
        "experience_explore_branch": experience_branch,
        "experience_retrieve_hits": retrieved_hits_meta,
        "score_normalization": {
            "method": "per_dataset_minmax",
            "lo": lo_score,
            "hi": hi_score,
            "n_success": n_success,
        },
    }
    if gate_meta:
        meta.update(gate_meta)
    return "\n".join(lines), meta


def _payload_looks_like_config(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if isinstance(payload.get("config"), dict):
        return True
    # Bare HP map (no wrapper): keys like "algo:hp".
    return bool(payload) and all(isinstance(k, str) for k in payload) and any(":" in k for k in payload)


def _truncate_leaked_turn_suffix(text: str) -> str:
    """Drop text after a leaked chat turn (common with overfit SFT models)."""
    lowered = text.lower()
    cut = len(text)
    for marker in (
        "\nuser\n",
        "\nassistant\n",
        "\nuser:",
        "\nassistant:",
        "<|im_start|>",
        "",
        "<|endoftext|>",
    ):
        pos = lowered.find(marker)
        if pos > 0:
            cut = min(cut, pos)
    return text[:cut].strip() if cut < len(text) else text


def _iter_json_objects(scan_text: str, start_at: int = 0):
    """Yield (start_index, object) for each decodable ``{...}`` in *scan_text*."""
    decoder = json.JSONDecoder()
    idx = start_at
    last_err: Optional[Exception] = None
    while True:
        start = scan_text.find("{", idx)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(scan_text, start)
        except json.JSONDecodeError as exc:
            last_err = exc
            idx = start + 1
            continue
        yield start, obj
        idx = end
    if last_err is not None:
        raise last_err


def _pick_config_payload(candidates: list[tuple[int, Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Prefer wrapped ``{"config": ...}`` over bare HP maps; else earliest match."""
    if not candidates:
        return None
    wrapped = [obj for _, obj in candidates if isinstance(obj.get("config"), dict)]
    if wrapped:
        return wrapped[0]
    return candidates[0][1]


def parse_llm_json(text: str) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """Parse LLM HPO replies.

    Accepts JSON-only or ``Rationale:`` + ``JSON:`` wrappers. Qwen SFT models
    often continue past the first object into a leaked ``\\nuser\\n`` turn; we
    take the first complete JSON object that looks like a config (not first
    ``{`` through last ``}``, which breaks on multi-object leaks).
    """
    raw_text = _truncate_leaked_turn_suffix((text or "").strip())
    if not raw_text:
        raise ValueError("Empty LLM response.")

    # Prefer fenced blocks when present, else scan the full reply.
    scan_text = raw_text
    fence_prefix = ""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        fence_prefix = raw_text[: match.start()].strip()
        scan_text = match.group(1).strip()

    # If a JSON: label exists, scan from there first (skips braces in rationales).
    label = re.search(r"(?is)\bjson\s*:\s*", scan_text)
    search_bases = [label.end()] if label else []
    search_bases.append(0)

    payload = None
    prefix = fence_prefix
    last_err: Optional[Exception] = None
    for base in search_bases:
        candidates: list[tuple[int, Dict[str, Any]]] = []
        try:
            for start, obj in _iter_json_objects(scan_text, base):
                if isinstance(obj, dict) and _payload_looks_like_config(obj):
                    candidates.append((start, obj))
        except json.JSONDecodeError as exc:
            last_err = exc
            continue
        picked = _pick_config_payload(candidates)
        if picked is not None:
            payload = picked
            if not prefix:
                first_start = candidates[0][0]
                prefix = scan_text[:first_start].strip()
            break

    if payload is None:
        if last_err is not None:
            raise last_err
        raise ValueError("LLM response JSON must be an object with a config.")

    config = payload.get("config", payload)
    if not isinstance(config, dict):
        raise ValueError("LLM response field 'config' must be an object.")
    rationale = str(payload.get("rationale", "")).strip()
    if not rationale and prefix:
        rationale = re.sub(r"(?is)^rationale\s*:\s*", "", prefix).strip()
        rationale = re.sub(r"(?is)\n?\s*json\s*:\s*$", "", rationale).strip()
    return config, rationale, payload


class PT4OptLLMOptimizer(BaseOptimizer):
    optimizer_label = "ptllm"

    def __init__(
        self,
        evaluator,
        config_space,
        name,
        eval_type,
        time_limit=None,
        evaluation_limit=None,
        per_run_time_limit=300,
        per_run_mem_limit=1024,
        inner_iter_num_per_iter=1,
        timestamp=None,
        output_dir="./",
        seed=1,
        n_jobs=1,
        topk=50,
        optimizer_params=None,
        task_id="GenTrans",
        logging_enabled=True,
        **kwargs,
    ):
        super().__init__(
            evaluator=evaluator,
            config_space=config_space,
            name=name,
            eval_type=eval_type,
            task_id=task_id,
            time_limit=time_limit,
            evaluation_limit=evaluation_limit,
            per_run_time_limit=per_run_time_limit,
            per_run_mem_limit=per_run_mem_limit,
            inner_iter_num_per_iter=inner_iter_num_per_iter,
            timestamp=timestamp,
            output_dir=output_dir,
            seed=seed,
            topk=topk,
        )
        self.params = dict(optimizer_params or {})
        self.params.update(kwargs)
        if n_jobs != 1:
            self.logger.warning("PT4OptLLMOptimizer is sequential; n_jobs=%s is ignored.", n_jobs)

        self.KD = KnowledgeDatabase.instance()
        self.configs = self.KD.configs
        self.perfs = self.KD.perfs
        self.objective_function = partial(wrapper_evaluater, evaluator=evaluator, per_run_time_limit=per_run_time_limit)

        self.algorithm_hp = self._get_hp("algorithm")
        self.fixed_algorithm = self._get_single_algorithm()
        self.n_no_history_samples = int(self.params.get("n_no_history_samples", self.params.get("n_initial_samples", 4)))
        self.n_initial_samples = self.n_no_history_samples
        self.history_window = int(self.params.get("history_window", 20))
        self.top_k = int(self.params.get("top_k", 6))
        self.recent_k = int(self.params.get("recent_k", 10))
        self.worst_k = int(self.params.get("worst_k", 4))
        self.run_summary = bool(self.params.get("run_summary", False))
        self.use_tcs_analysis = bool(self.params.get("tcs_analysis", False))
        self.tcs_single_agent = bool(self.params.get("tcs_single_agent", False))
        self.prompt_blocks = validate_prompt_blocks(self.params.get("prompt_blocks"))
        self.experience_self_mode = str(self.params.get("experience_self_mode") or "cohort")
        self.experience_self_start_trial = max(1, int(self.params.get("experience_self_start_trial") or 1))
        self.experience_self_gate = str(self.params.get("experience_self_gate") or "clock").strip().lower()
        from mindware.components.optimizers.experience_gate import ExperienceGate

        self._experience_gate = ExperienceGate(
            mode=self.experience_self_gate,
            cap=self.experience_self_start_trial,
        )
        self.experience_explore_mode = str(self.params.get("experience_explore_mode") or "off").strip().lower()
        self.experience_explore_prob = float(self.params.get("experience_explore_prob") or 0.5)
        from mindware.components.optimizers.experience_gate import ExplorePolicy

        self._explore_policy = ExplorePolicy(
            mode=self.experience_explore_mode,
            seed=int(getattr(self, "seed", 1) or 1),
            p_default=self.experience_explore_prob,
            tpe_min_trial=int(self.params.get("experience_explore_tpe_min_trial") or 0),
        )
        self.experience_blame = dict(self.params.get("experience_blame") or {})
        self.experience_retrieve_k = max(1, int(self.params.get("experience_retrieve_k") or 3))
        self.experience_retrieve_start_trial = max(1, int(self.params.get("experience_retrieve_start_trial") or 1))
        self.experience_retrieve_rerank_trial = max(1, int(self.params.get("experience_retrieve_rerank_trial") or 24))
        self.experience_retrieve_ranker = str(self.params.get("experience_retrieve_ranker") or "surrogate")
        self.experience_retrieve_fade = bool(self.params.get("experience_retrieve_fade", False))
        _stop = self.params.get("experience_retrieve_stop_trial")
        self.experience_retrieve_stop_trial = None if _stop in (None, "", 0, "0") else int(_stop)
        _floor = self.params.get("experience_retrieve_spearman_floor")
        self.experience_retrieve_spearman_floor = None if _floor in (None, "") else float(_floor)
        self.experience_retrieve_include_top_bottom = bool(
            self.params.get("experience_retrieve_include_top_bottom", False)
        )
        self.experience_retrieve_include_prefer = bool(
            self.params.get("experience_retrieve_include_prefer", True)
        )
        self.experience_retrieve_region_style = str(
            self.params.get("experience_retrieve_region_style") or "prefer"
        )
        self.experience_retrieve_tb_layout = str(
            self.params.get("experience_retrieve_tb_layout") or "flat"
        )
        _tk = self.params.get("experience_retrieve_top_k")
        self.experience_retrieve_top_k = 1 if _tk in (None, "") else max(0, int(_tk))
        _bk = self.params.get("experience_retrieve_bottom_k")
        self.experience_retrieve_bottom_k = 1 if _bk in (None, "") else max(0, int(_bk))
        self.dataset_id = self.params.get("dataset_id")
        self.task_type = self.params.get("task_type")
        self._experience_bank = None
        self._experience_bank_warned = False
        self.trial_sources: Dict[int, str] = {}
        self._tpe_hint_config: Optional[Dict[str, Any]] = None
        self.max_duplicate_reprompts = min(1, max(0, int(self.params.get("max_duplicate_reprompts", 1))))
        self.fallback_on_transport_error = bool(self.params.get("fallback_on_transport_error", False))
        self.fallback_on_bad_json = bool(self.params.get("fallback_on_bad_json", True))
        self.fallback_on_invalid_config = bool(self.params.get("fallback_on_invalid_config", True))
        # Match SFT target format: S0/S1a = json_only; S1b/gpt default = rationale_json.
        _fmt = str(
            self.params.get("response_format")
            or self.params.get("llm_response_format")
            or "rationale_json"
        ).strip().lower()
        self.response_format = "json_only" if _fmt in ("json", "json_only", "config_only") else "rationale_json"

        self.llm_client = self.params.get("llm_client")
        self.mock_responses = list(self.params.get("mock_llm_responses") or [])
        self.use_mock_llm = bool(self.params.get("mock_llm") or self.params.get("llm_client_type") == "mock")
        self._setup_openai_client()

        self.llm_history: List[Dict[str, Any]] = []
        self.llm_query_cost: List[float] = []
        self.llm_query_time: List[float] = []
        self.llm_call_count = 0
        self.trial_cnt = 0
        self.exp_output = {}
        self.logging_enabled = logging_enabled
        self.traj_rows: List[Dict[str, Any]] = []
        self.traj_csv_path = os.path.join(self.output_dir, f"{int(self.timestamp)}_opt_{self.optimizer_label}_traj.csv")
        self.conversation_log_dir = os.path.join(self.output_dir, "llm_conversations")
        self.start_wall_time = time.time()
        os.makedirs(self.conversation_log_dir, exist_ok=True)

        self.seen_signatures = set()
        self.seen_objectives: Dict[Tuple[Tuple[str, str], ...], float] = {}
        self._seen_config_count = 0
        self._sync_seen_signatures()
        # Optional FIFO of flat HP dicts used for early llm_warm_start trials
        # (e.g. best-of-N from an external warm-start agent).
        presets = self.params.get("preset_warm_start_configs") or []
        self._preset_warm_start_queue: List[Dict[str, Any]] = [
            dict(p) for p in presets if isinstance(p, dict)
        ]

    def _setup_openai_client(self):
        self.openai_client = None
        if self.use_mock_llm or self.llm_client is not None:
            return
        from openai import OpenAI

        engine = self.params.get("chat_engine", LLMCONFIGS.opt_chat_engine)
        cfg = AVAILABLE_LLMs.get(engine, {})
        self.llm_model = self.params.get("llm_model") or cfg.get("model", engine)
        self.llm_temperature = float(self.params.get("temperature", cfg.get("temperature", 0.2)))
        self.llm_max_tokens = int(self.params.get("max_tokens", 2048))
        self.llm_timeout = int(self.params.get("llm_timeout", 180))
        self.prompt_cost = LLM_COST_DICT.get(engine, {}).get("promt_cost_per_1k", 0.0)
        self.completion_cost = LLM_COST_DICT.get(engine, {}).get("completion_cost_per_1k", 0.0)
        self.openai_client = OpenAI(
            api_key=self.params.get("api_key") or cfg.get("api_key", LLMCONFIGS.OPENAI_KEY),
            base_url=self.params.get("base_url") or cfg.get("base_url", LLMCONFIGS.OPENAI_BASE_URL),
        )

    def run(self):
        while not (self.early_stopped_flag or self.timeout_flag):
            self.iterate()
        return np.max(self.perfs) if self.perfs else -np.inf

    def iterate(self, budget=MAX_INT):
        start = time.time()
        rows = []
        for _ in range(self.inner_iter_num_per_iter):
            if self._quota_or_time_is_up() or time.time() - start > budget:
                break
            iter_idx = self.trial_cnt
            self.trial_cnt += 1
            iter_start = time.time()

            include_history = iter_idx >= self.n_no_history_samples
            source = "llm" if include_history else "llm_warm_start"
            cfg, note, llm_cost, llm_time, raw, payload, meta, rationale, prompt_input = self._suggest_config(
                iter_idx,
                include_history=include_history,
            )

            obs = self.objective_function(cfg)
            raw_obj = float(obs.objectives[0]) if obs.objectives else np.inf
            success = obs.trial_state == SUCCESS and np.isfinite(raw_obj)
            perf = -raw_obj if success else float("-inf")
            eval_time = obs.elapsed_time or 0.0

            if success:
                self.exp_output[time.time()] = (cfg, perf)
                self.update_saver([cfg], [raw_obj])
                self._update_history(obs, source, iter_idx, cfg, rationale)
                self._mark_seen(cfg, raw_obj)
            self._finish_llm_history(cfg, perf, success, obs)
            self._dump_conversation(iter_idx, prompt_input, raw, payload, cfg, obs, meta, note)

            iter_time = time.time() - iter_start
            rows.append({"perf": perf, "config": cfg, "eval_time": eval_time, "llm_time": llm_time, "iter_time": iter_time})
            self._record_event(source, iter_idx, cfg, perf, llm_cost, llm_time, eval_time, iter_time, note)

        if self._quota_or_time_is_up():
            self.timeout_flag = True
        self._update_incumbent()
        return self.incumbent_perf, time.time() - start, self.incumbent_config, rows

    def _suggest_config(self, iter_idx: int, include_history: bool = True):
        t0 = time.time()
        raw, payload, rationale, cost = "", {}, "", 0.0
        attempts = []
        rejected_duplicate_configs = []
        prompt_input = {}
        # External warm-start presets replace the first llm_warm_start suggestions.
        if (not include_history) and self._preset_warm_start_queue:
            flat = dict(self._preset_warm_start_queue.pop(0))
            flat.setdefault("algorithm", self.fixed_algorithm)
            cfg = self.KD._dict_to_cfg(flat)
            note = "preset warm-start configuration (external agent best-of-N)"
            prompt_input = {
                "system_prompt": "",
                "user_prompt": "",
                "include_history": False,
                "preset_warm_start": True,
                "preset_config": _config_for_prompt(cfg, self.fixed_algorithm, self.config_space),
                "messages": [],
            }
            self._append_llm_history(
                iter_idx,
                "PENDING",
                "",
                {"config": flat, "source": "preset_warm_start"},
                cfg,
                note,
                attempts=[],
                rejected_duplicate_configs=[],
            )
            meta = {
                "history_included": False,
                "preset_warm_start": True,
                "selected_best_trials": [],
                "selected_recent_trials": [],
                "selected_worst_trials": [],
                "selected_trials": [],
            }
            self._last_history_meta = meta
            return cfg, note, 0.0, time.time() - t0, "", {"config": flat}, meta, note, prompt_input
        try:
            system_prompt, user_prompt = self._build_prompts(
                include_history=include_history,
                current_trial=iter_idx + 1,
            )
            prompt_input = {
                "system_prompt": system_prompt,
                "system_prompt_lines": _lines(system_prompt),
                "user_prompt": user_prompt,
                "user_prompt_lines": _lines(user_prompt),
                "include_history": include_history,
                "tcs_analysis": bool(include_history and self.use_tcs_analysis),
                "analysis_text": getattr(self, "_last_analysis_text", None),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            prompt_input["attempts"] = attempts
            cfg, raw, payload, rationale, cost, attempt = self._request_config_attempt(
                system_prompt,
                user_prompt,
                attempt_id=0,
            )
            if include_history and self.use_tcs_analysis:
                cost += float(getattr(self, "_last_analysis_cost", 0.0))
            attempts.append(attempt)
            note = "llm suggested configuration" if include_history else "llm warm-start configuration"

            if attempt["duplicate"]:
                rejected_duplicate_configs.append(_config_for_prompt(cfg, self.fixed_algorithm, self.config_space))
                reprompt = self._reprompt_duplicate_if_needed(system_prompt, user_prompt, prompt_input, cfg)
                if reprompt is not None:
                    cfg, raw, payload, rationale, reprompt_cost, attempt = reprompt
                    cost += reprompt_cost
                    attempts.append(attempt)

            if self._is_duplicate(cfg):
                rejected = _config_for_prompt(cfg, self.fixed_algorithm, self.config_space)
                if rejected not in rejected_duplicate_configs:
                    rejected_duplicate_configs.append(rejected)
                cfg, note = self._sample_nonduplicate_config(), self._duplicate_fallback_note(attempts)
                self._append_llm_history(
                    iter_idx,
                    "DUPLICATE",
                    raw,
                    payload,
                    cfg,
                    note,
                    attempts=attempts,
                    rejected_duplicate_configs=rejected_duplicate_configs,
                )
            else:
                if len(attempts) > 1:
                    note = (
                        "llm suggested configuration after duplicate reprompt"
                        if include_history
                        else "llm warm-start configuration after duplicate reprompt"
                    )
                self._append_llm_history(
                    iter_idx,
                    "PENDING",
                    raw,
                    payload,
                    cfg,
                    rationale,
                    attempts=attempts,
                    rejected_duplicate_configs=rejected_duplicate_configs,
                )
            meta = self._last_history_meta
            return cfg, note, cost, time.time() - t0, raw, payload, meta, rationale, prompt_input
        except Exception as exc:
            raw = getattr(exc, "raw_response", raw)
            payload = getattr(exc, "parsed_payload", payload)
            if (not raw and not self.fallback_on_transport_error) or (raw and not self.fallback_on_bad_json):
                raise
            if raw and payload and not self.fallback_on_invalid_config:
                raise
            cfg = self._sample_nonduplicate_config()
            note = f"random fallback after llm error: {exc}"
            self._append_llm_history(
                iter_idx,
                "FAILED",
                raw,
                payload,
                cfg,
                note,
                error=str(exc),
                attempts=attempts,
                rejected_duplicate_configs=rejected_duplicate_configs,
            )
            return cfg, note, cost, time.time() - t0, raw, payload, getattr(self, "_last_history_meta", {}), rationale, prompt_input

    def _request_config_attempt(self, system_prompt: str, user_prompt: str, attempt_id: int):
        raw, cost = self._complete_llm(system_prompt, user_prompt)
        payload = {}
        try:
            config_dict, rationale, payload = parse_llm_json(raw)
            cfg = self.KD._dict_to_cfg(dict(config_dict, algorithm=self.fixed_algorithm))
        except Exception as exc:
            setattr(exc, "raw_response", raw)
            setattr(exc, "parsed_payload", payload)
            raise
        return cfg, raw, payload, rationale, cost, self._make_attempt_record(attempt_id, raw, payload, cfg)

    def _make_attempt_record(self, attempt_id: int, raw: str, payload: Dict[str, Any], cfg) -> Dict[str, Any]:
        return {
            "attempt": attempt_id,
            "raw_response": raw,
            "parsed_payload": payload,
            "normalized_config": _cfg_dict(cfg),
            "duplicate": self._is_duplicate(cfg),
        }

    def _reprompt_duplicate_if_needed(self, system_prompt: str, user_prompt: str, prompt_input: Dict[str, Any], cfg):
        if self.max_duplicate_reprompts <= 0:
            return None
        reprompt_user = self._build_duplicate_reprompt(user_prompt, cfg)
        prompt_input["reprompt_user_prompt"] = reprompt_user
        prompt_input["reprompt_user_prompt_lines"] = _lines(reprompt_user)
        prompt_input["reprompt_messages"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": reprompt_user},
        ]
        return self._request_config_attempt(system_prompt, reprompt_user, attempt_id=1)

    def _duplicate_fallback_note(self, attempts: List[Dict[str, Any]]) -> str:
        if len(attempts) > 1:
            return "random fallback after duplicate llm reprompt"
        return "random fallback for duplicate llm configuration"

    def _complete_llm(self, system_prompt: str, user_prompt: str):
        self.llm_call_count += 1
        if self.llm_client is not None:
            result = self.llm_client.complete(system_prompt, user_prompt)
            if isinstance(result, str):
                return result, 0.0
            return result.content, float(getattr(result, "cost", 0.0))
        if self.use_mock_llm:
            return (self.mock_responses.pop(0) if self.mock_responses else 'Rationale: mock default\n\nJSON:\n{"config": {}}'), 0.0
        resp = self.openai_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
            timeout=self.llm_timeout,
        )
        usage = getattr(resp, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        cost = prompt_tokens / 1000.0 * self.prompt_cost + completion_tokens / 1000.0 * self.completion_cost
        return resp.choices[0].message.content, cost

    def _build_duplicate_reprompt(self, user_prompt: str, duplicate_cfg) -> str:
        rejected = _config_for_prompt(duplicate_cfg, self.fixed_algorithm, self.config_space)
        return "\n\n".join(
            [
                user_prompt,
                "The previous JSON configuration was rejected because it duplicates an already evaluated trial.",
                f"Rejected duplicate configuration: {_json(rejected)}",
                "Return one different configuration. Keep the same output format and use every tunable hyperparameter listed in Search space.",
            ]
        )

    def _build_prompts(self, include_history: bool = True, current_trial: Optional[int] = None):
        if include_history and self.use_tcs_analysis:
            if self.tcs_single_agent:
                return self._build_tcs_single_agent_prompts()
            return self._build_tcs_prompts()
        return self._build_history_prompts(include_history, current_trial=current_trial)

    def _metric_name(self) -> str:
        profile = self.params.get("dataset_profile") or {}
        return str(profile.get("metric", "validation score"))

    def _run_analysis_agent(self, tcs_report: str) -> Tuple[str, float]:
        hp_names = _hp_names_from_space(self.config_space)
        system, user = build_analysis_agent_prompts(
            tcs_report,
            metric_name=self._metric_name(),
            higher_is_better=True,
            hp_names=hp_names,
        )
        return self._complete_llm(system, user)

    def _build_tcs_prompts(self):
        tcs_report, tcs_meta = build_tcs_report(
            self.configs,
            self.perfs,
            self.fixed_algorithm,
            self.config_space,
            metric_name=self._metric_name(),
            higher_is_better=True,
        )
        analysis_text, analysis_cost = self._run_analysis_agent(tcs_report)
        self._last_analysis_cost = float(analysis_cost)
        self._last_analysis_text = analysis_text

        meta = dict(tcs_meta)
        meta.update(
            {
                "history_included": True,
                "tcs_analysis": True,
                "analysis_text": analysis_text,
                "analysis_cost": float(analysis_cost),
            }
        )
        self._last_history_meta = meta

        system = (
            "You are the Optimizer Agent for single-algorithm hyperparameter optimization. "
            f"The algorithm is fixed to '{self.fixed_algorithm}'. "
            "You receive a Trajectory Context Summary (TCS) and an Analysis Agent report. "
            "rel_score is min-max normalized within this dataset (1.0 = best observed, higher is better). "
            "Propose the next configuration to maximize validation score. "
            "First write a short rationale, then write one JSON object containing the configuration. "
            "Do not use markdown."
        )
        user = "\n\n".join(
            [
                self._dataset_text(),
                f"Fixed algorithm: {self.fixed_algorithm}",
                self._search_space_text(),
                tcs_report,
                "Analysis Agent Report:",
                analysis_text,
                'Return exactly this format:\n'
                'Rationale: <short reason for the next configuration>\n\n'
                'JSON:\n{"config": {"<hyperparameter_name>": "<value>"}}\n'
                "Return values for every tunable hyperparameter listed in Search space. "
                "Do not return a configuration listed under EVALUATED CONFIGURATIONS in the TCS. "
                "Use the exact hyperparameter names shown in Search space.",
            ]
        )
        return system, user

    def _build_tcs_single_agent_prompts(self):
        tcs_report, tcs_meta = build_tcs_report(
            self.configs,
            self.perfs,
            self.fixed_algorithm,
            self.config_space,
            metric_name=self._metric_name(),
            higher_is_better=True,
        )
        hp_names = _hp_names_from_space(self.config_space)
        system, user = build_tcs_single_agent_prompts(
            tcs_report,
            fixed_algorithm=self.fixed_algorithm,
            dataset_text=self._dataset_text(),
            search_space_text=self._search_space_text(),
            metric_name=self._metric_name(),
            higher_is_better=True,
            hp_names=hp_names,
        )
        self._last_analysis_cost = 0.0
        self._last_analysis_text = None

        meta = dict(tcs_meta)
        meta.update(
            {
                "history_included": True,
                "tcs_analysis": True,
                "tcs_single_agent": True,
                "analysis_text": None,
                "analysis_cost": 0.0,
            }
        )
        self._last_history_meta = meta
        return system, user

    def _get_experience_bank(self):
        if self._experience_bank is not None:
            return self._experience_bank
        if not self.prompt_blocks.get("experience_retrieve"):
            return None
        path = self.params.get("experience_bank_path")
        if not path:
            if not self._experience_bank_warned:
                self.logger.warning("experience_retrieve is on but experience_bank_path is empty")
                self._experience_bank_warned = True
            return None
        from pathlib import Path as _Path
        from mindware.components.experience.retrieve import ExperienceBank

        p = _Path(path)
        if not p.is_file():
            if not self._experience_bank_warned:
                self.logger.warning("experience bank not found: %s", p)
                self._experience_bank_warned = True
            return None
        try:
            self._experience_bank = ExperienceBank.load(p)
            self.logger.info("Loaded experience bank %s (%s rows)", p, len(self._experience_bank))
        except Exception as exc:
            if not self._experience_bank_warned:
                self.logger.warning("failed to load experience bank %s: %s", p, exc)
                self._experience_bank_warned = True
            return None
        return self._experience_bank

    def _build_history_prompts(self, include_history: bool = True, current_trial: Optional[int] = None):
        if include_history:
            history, meta = build_history_prompt_block(
                self.configs,
                self.perfs,
                self.fixed_algorithm,
                self.config_space,
                self.history_window,
                self.top_k,
                self.recent_k,
                self.worst_k,
                run_summary=self.run_summary,
                prompt_blocks=self.prompt_blocks,
                budget_total=self.evaluation_num_limit,
                trial_sources=self.trial_sources,
                llm_history=self.llm_history,
                experience_self_mode=self.experience_self_mode,
                current_trial=current_trial,
                experience_self_start_trial=self.experience_self_start_trial,
                experience_explore_mode=self.experience_explore_mode,
                experience_explore_prob=self.experience_explore_prob,
                experience_blame=self.experience_blame,
                rng_seed=int(getattr(self, "seed", 1) or 1),
                experience_bank=self._get_experience_bank(),
                experience_retrieve_k=self.experience_retrieve_k,
                experience_retrieve_start_trial=self.experience_retrieve_start_trial,
                experience_retrieve_rerank_trial=self.experience_retrieve_rerank_trial,
                experience_retrieve_ranker=self.experience_retrieve_ranker,
                experience_retrieve_fade=self.experience_retrieve_fade,
                experience_retrieve_stop_trial=self.experience_retrieve_stop_trial,
                experience_retrieve_spearman_floor=self.experience_retrieve_spearman_floor,
                experience_retrieve_include_top_bottom=self.experience_retrieve_include_top_bottom,
                experience_retrieve_include_prefer=self.experience_retrieve_include_prefer,
                experience_retrieve_region_style=self.experience_retrieve_region_style,
                experience_retrieve_tb_layout=self.experience_retrieve_tb_layout,
                experience_retrieve_top_k=self.experience_retrieve_top_k,
                experience_retrieve_bottom_k=self.experience_retrieve_bottom_k,
                experience_self_gate=self.experience_self_gate,
                experience_gate_state=getattr(self, "_experience_gate", None),
                explore_policy_state=getattr(self, "_explore_policy", None),
                dataset_id=self.dataset_id,
                task_type=self.task_type,
                dataset_profile=self.params.get("dataset_profile"),
            )
            meta["history_included"] = True
        else:
            history = "Selected historical trials: omitted for this warm-start query."
            meta = {
                "selected_best_trials": [],
                "selected_recent_trials": [],
                "selected_worst_trials": [],
                "selected_trials": [],
                "allocated_best_k": 0,
                "allocated_worst_k": 0,
                "history_window": self.history_window,
                "history_included": False,
                "prompt_blocks": self.prompt_blocks,
            }
            if self.prompt_blocks.get("experience_retrieve") and self._get_experience_bank() is not None:
                from mindware.components.experience.retrieve import retrieve_for_run

                packed = []
                for i, (cfg, perf) in enumerate(zip(self.configs or [], self.perfs or []), start=1):
                    packed.append(
                        {
                            "trial": i,
                            "config": _cfg_dict(cfg),
                            "score": float(perf),
                        }
                    )
                ret_block, hits = retrieve_for_run(
                    self._get_experience_bank(),
                    dataset_id=self.dataset_id,
                    algorithm_id=self.fixed_algorithm,
                    task_type=self.task_type,
                    dataset_profile=self.params.get("dataset_profile"),
                    successes=packed,
                    config_space=self.config_space,
                    k=self.experience_retrieve_k,
                    start_trial=self.experience_retrieve_start_trial,
                    rerank_after=self.experience_retrieve_rerank_trial,
                    current_trial=int(current_trial) if current_trial is not None else 1,
                    fade_after=None,
                    ranker=self.experience_retrieve_ranker,
                    spearman_floor=self.experience_retrieve_spearman_floor,
                    include_top_bottom=self.experience_retrieve_include_top_bottom,
                    include_prefer=self.experience_retrieve_include_prefer,
                    region_style=self.experience_retrieve_region_style,
                    tb_layout=self.experience_retrieve_tb_layout,
                    retrieve_top_k=self.experience_retrieve_top_k,
                    retrieve_bottom_k=self.experience_retrieve_bottom_k,
                    stop_trial=self.experience_retrieve_stop_trial,
                )
                if ret_block:
                    history = history + "\n\n" + ret_block
                meta["experience_retrieve_hits"] = [
                    {"id": h.exp_id, "score": h.score, "spearman": h.spearman}
                    for h in hits
                ]
        self._last_history_meta = meta
        if self.response_format == "json_only":
            system = (
                "You are a single-algorithm hyperparameter optimization assistant. "
                f"The algorithm is fixed to '{self.fixed_algorithm}'. "
                "Historical trials report rel_score (validation score min-max normalized "
                "within this dataset; 1.0 = best observed, higher is better) and rank "
                "(1 = best). Propose the next configuration to maximize the validation "
                "score. Return only one JSON object containing the configuration. "
                "Do not use markdown."
            )
        else:
            system = (
                "You are a single-algorithm hyperparameter optimization assistant. "
                f"The algorithm is fixed to '{self.fixed_algorithm}'. "
                "Historical trials report rel_score (validation score min-max normalized "
                "within this dataset; 1.0 = best observed, higher is better) and rank "
                "(1 = best). Propose the next configuration to maximize the validation "
                "score. First write a short rationale, then write one JSON object "
                "containing the configuration. Do not use markdown."
            )
        if self.prompt_blocks.get("tuning_playbook_system"):
            system += build_tuning_playbook_system_extension(
                self.fixed_algorithm,
                self.params.get("dataset_profile"),
            )
        duplicate_rule = (
            "Do not return a configuration that duplicates any Selected historical trial. "
            if include_history
            else "No Selected historical trials are provided for this warm-start query. "
        )
        user_parts = [
            self._dataset_text(),
            f"Fixed algorithm: {self.fixed_algorithm}",
            self._search_space_text(),
            history,
        ]
        if include_history and self._tpe_hint_config:
            user_parts.append(format_tpe_hint_block(self._tpe_hint_config))
        if self.response_format == "json_only":
            user_parts.append(
                "Return only this JSON object:\n"
                '{"config": {"<hyperparameter_name>": "<value>"}}\n'
                "Return values for every tunable hyperparameter listed in Search space. "
                f"{duplicate_rule}"
                "Use the exact hyperparameter names shown in Search space. No rationale."
            )
        else:
            user_parts.append(
                "Return exactly this format:\n"
                "Rationale: <short reason for the next configuration>\n\n"
                'JSON:\n{"config": {"<hyperparameter_name>": "<value>"}}\n'
                "Return values for every tunable hyperparameter listed in Search space. "
                f"{duplicate_rule}"
                "Use the exact hyperparameter names shown in Search space."
            )
        user = "\n\n".join(user_parts)
        return system, user

    def _dataset_text(self):
        profile = self.params.get("dataset_profile")
        return "Dataset profile:\n" + _json(profile) if profile else "Dataset profile: not provided."

    def _search_space_text(self):
        rows = []
        for hp in self.config_space.get_hyperparameters():
            name = hp.name
            if name == "algorithm" or isinstance(hp, (Constant, UnParametrizedHyperparameter)):
                continue
            rows.append(f"- {name}: {self._hp_text(hp)}")
        return "Search space:\n" + ("\n".join(rows) if rows else "no tunable hyperparameters")

    def _hp_text(self, hp):
        # Match P5 SFT training Search-space format (range=/q=/log_scale=).
        if isinstance(hp, (Constant, UnParametrizedHyperparameter)):
            return "constant"
        if isinstance(hp, UniformIntegerHyperparameter):
            parts = [f"integer, range=[{int(hp.lower)}, {int(hp.upper)}]"]
            q = getattr(hp, "q", None)
            if q is not None:
                parts.append(f"q={int(q)}")
            if getattr(hp, "log", False):
                parts.append("log_scale=True")
            return ", ".join(parts)
        if isinstance(hp, UniformFloatHyperparameter):
            parts = [f"float, range=[{float(hp.lower)}, {float(hp.upper)}]"]
            q = getattr(hp, "q", None)
            if q is not None:
                parts.append(f"q={float(q)}")
            if getattr(hp, "log", False):
                parts.append("log_scale=True")
            return ", ".join(parts)
        if isinstance(hp, CategoricalHyperparameter):
            return f"categorical, choices={list(hp.choices)!r}"
        if OrdinalHyperparameter and isinstance(hp, OrdinalHyperparameter):
            return f"ordinal, sequence={list(hp.sequence)!r}"
        return f"default={_hp_default(hp)!r}"

    def _update_history(self, obs, source, iter_idx, cfg, rationale):
        is_warm_start = source == "llm_warm_start"
        fallback_text = "LLM warm-start configuration." if is_warm_start else "LLM suggested configuration."
        self.KD.update_history(
            obs,
            source="llm",
            trail_id=f"{iter_idx}_warm_start" if is_warm_start else str(iter_idx),
            detailed_thought=rationale or fallback_text,
            thought_summary=rationale or fallback_text,
            action_config=_cfg_dict(cfg),
            observation_perf=self._raw_observation_perf(obs),
            reflection_summary="No reflection in PT4OptLLMOptimizer.",
        )

    def _append_llm_history(
        self,
        iter_idx,
        status,
        raw,
        payload,
        cfg,
        note,
        error=None,
        attempts=None,
        rejected_duplicate_configs=None,
    ):
        self.llm_history.append(
            {
                "trial": iter_idx,
                "status": status,
                "raw_response": raw,
                "parsed_payload": payload,
                "parsed_config": payload.get("config", payload) if isinstance(payload, dict) else {},
                "normalized_config": _cfg_dict(cfg),
                "rationale": note if status == "PENDING" else "",
                "fallback_reason": note if status != "PENDING" else "",
                "error": error,
                "attempts": attempts or [],
                "rejected_duplicate_configs": rejected_duplicate_configs or [],
            }
        )

    def _finish_llm_history(self, cfg, perf, success, obs):
        if not self.llm_history:
            return
        entry = self.llm_history[-1]
        entry["evaluated_config"] = _cfg_dict(cfg)
        entry["perf"] = perf
        entry["evaluation_status"] = "SUCCESS" if success else "FAILED"
        if entry.get("status") == "PENDING":
            entry["status"] = "SUCCESS" if success else "FAILED"
        if not success:
            entry["error"] = str(getattr(obs, "trial_state", "evaluation_failed"))

    def _dump_conversation(self, iter_idx, prompt_input, raw, payload, cfg, obs, meta, note):
        json_path = os.path.join(self.conversation_log_dir, f"trial_{iter_idx:04d}.json")
        data = {
            "trial": iter_idx,
            "llm_input": prompt_input,
            "llm_output": {
                "raw_response": raw,
                "raw_response_lines": _lines(raw),
                "parsed_payload": payload,
            },
            "raw_response": raw,
            "parsed_payload": payload,
            "normalized_config": _cfg_dict(cfg),
            "llm_attempts": prompt_input.get("attempts", []),
            "objective": obs.objectives[0] if obs.objectives else None,
            "trial_state": getattr(obs, "trial_state", None),
            "history_meta": meta,
            "note": note,
        }
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=True, default=str)
        except IOError as exc:
            self.logger.warning("Failed to write LLM log %s: %s", json_path, exc)

    def _record_event(self, stage, iter_idx, cfg, perf, llm_cost, llm_time, eval_time, iter_time, note):
        if not self.logging_enabled:
            return
        if llm_cost:
            self.llm_query_cost.append(float(llm_cost))
        if llm_time:
            self.llm_query_time.append(float(llm_time))
        row = {
            "ts": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "stage": stage,
            "iter_idx": iter_idx,
            "perf": float(perf),
            "best_so_far": float(max(self.perfs) if self.perfs else perf),
            "algorithm": self.fixed_algorithm,
            "llm_cost": float(llm_cost),
            "cumulative_llm_cost": float(sum(self.llm_query_cost)),
            "llm_time": float(llm_time),
            "eval_time": float(eval_time),
            "iter_time": float(iter_time),
            "wall_elapsed_sec": float(time.time() - self.start_wall_time),
            "note": note,
            "config_for_prompt": _json(_config_for_prompt(cfg, self.fixed_algorithm, self.config_space)),
        }
        self.traj_rows.append(row)
        try:
            exists = os.path.exists(self.traj_csv_path)
            with open(self.traj_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
        except IOError as exc:
            self.logger.warning("Failed to save trajectory to %s: %s", self.traj_csv_path, exc)

    def _get_single_algorithm(self):
        choices = _hp_choices(self.algorithm_hp)
        if len(choices) == 1:
            return choices[0]
        raise ValueError("PT4OptLLMOptimizer only supports single-algorithm HPO.")

    def _get_hp(self, name):
        try:
            return self.config_space[name]
        except Exception:
            return self.config_space.get_hyperparameter(name)

    def _sample_config(self) -> MyConfiguration:
        cfg = self.config_space.sample_configuration()
        if isinstance(cfg, list):
            cfg = cfg[0]
        return cfg if isinstance(cfg, MyConfiguration) else MyConfiguration(cfg)

    def _sample_nonduplicate_config(self, max_attempts: int = 50) -> MyConfiguration:
        cfg = self._sample_config()
        for _ in range(max_attempts):
            if not self._is_duplicate(cfg):
                return cfg
            cfg = self._sample_config()
        return cfg

    def _sync_seen_signatures(self):
        for idx in range(self._seen_config_count, len(self.configs)):
            cfg = self.configs[idx]
            sig = _signature(cfg)
            self.seen_signatures.add(sig)
            if idx < len(self.perfs):
                perf = float(self.perfs[idx])
                if np.isfinite(perf):
                    self.seen_objectives[sig] = -perf
        self._seen_config_count = len(self.configs)

    def _mark_seen(self, cfg, objective=None):
        self._sync_seen_signatures()
        sig = _signature(cfg)
        self.seen_signatures.add(sig)
        if objective is not None and np.isfinite(float(objective)):
            self.seen_objectives[sig] = float(objective)

    def _is_duplicate(self, cfg) -> bool:
        self._sync_seen_signatures()
        return _signature(cfg) in self.seen_signatures

    def _known_objective(self, cfg):
        self._sync_seen_signatures()
        return self.seen_objectives.get(_signature(cfg))

    def _quota_or_time_is_up(self):
        quota = self.evaluation_num_limit is not None and len(self.perfs) >= self.evaluation_num_limit
        timeout = self.time_limit is not None and (time.time() - self.timestamp) > self.time_limit
        return quota or timeout

    def _raw_observation_perf(self, obs):
        raw = float(obs.objectives[0]) if obs.objectives else np.inf
        return raw * (1 if getattr(self.KD, "lower_is_better", False) else -1)

    def _update_incumbent(self):
        if not self.perfs:
            return
        idx = int(np.argmax(self.perfs))
        self.incumbent_config = self.configs[idx].get_dictionary().copy()
        self.incumbent_perf = float(self.perfs[idx])

    def get_opt_trajectory(self):
        return {
            "n_evals": len(self.perfs),
            "final_best": float(max(self.perfs)) if self.perfs else float("-inf"),
            "llm_cost_sum": float(sum(self.llm_query_cost)),
            "llm_calls": int(self.llm_call_count),
            "csv_path": self.traj_csv_path,
            "detail_perfs": ",".join(str(p) for p in self.perfs),
            "history": self.traj_rows[:500],
        }
import csv
import json
import os
import re
import time
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)
from openbox.utils.constants import SUCCESS

try:
    from ConfigSpace.hyperparameters import OrdinalHyperparameter
except ImportError:
    OrdinalHyperparameter = ()

from mindware.components.optimizers.base_optimizer import BaseOptimizer, MAX_INT
from mindware.components.optimizers.llm_prompt_blocks import (
    build_best_so_far_trajectory_block,
    build_budget_context_block,
    build_coverage_map_block,
    build_explore_refine_summary,
    build_experience_self_block,
    build_hp_marginals_block,
    build_playbook_run_context_block,
    build_rationale_diversity_summary,
    build_tuning_playbook_system_extension,
    format_tpe_hint_block,
    validate_prompt_blocks,
)
from mindware.components.optimizers.tcs_report import (
    _hp_names_from_space,
    build_analysis_agent_prompts,
    build_tcs_report,
    build_tcs_single_agent_prompts,
)
from mindware.components.optimizers.llm_optimizer_utils import wrapper_evaluater
from mindware.components.optimizers.utils import MyConfiguration
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase
from mindware.modules.configs import AVAILABLE_LLMs, LLMCONFIGS, LLM_COST_DICT


def _hp_value(hp):
    return getattr(hp, "value", getattr(hp, "default_value", None))


def _hp_default(hp):
    return getattr(hp, "default_value", getattr(hp, "value", None))


def _hp_choices(hp) -> List[Any]:
    if isinstance(hp, (Constant, UnParametrizedHyperparameter)):
        return [_hp_value(hp)]
    if hasattr(hp, "choices"):
        return list(hp.choices)
    if hasattr(hp, "sequence"):
        return list(hp.sequence)
    return [_hp_default(hp)]


def _cfg_dict(cfg) -> Dict[str, Any]:
    return cfg.get_dictionary().copy() if hasattr(cfg, "get_dictionary") else dict(cfg)


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, default=str)


def _lines(text: str) -> List[str]:
    return (text or "").splitlines()


def _signature(cfg) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((str(k), str(v)) for k, v in _cfg_dict(cfg).items()))


def _config_for_prompt(cfg, fixed_algorithm: str, config_space) -> Dict[str, Any]:
    cfg_dict = _cfg_dict(cfg)
    prompt_config = {}
    for hp in config_space.get_hyperparameters():
        name = hp.name
        if name == "algorithm" or isinstance(hp, (Constant, UnParametrizedHyperparameter)):
            continue
        if name not in cfg_dict:
            continue
        prompt_config[name] = cfg_dict[name]
    return prompt_config


def _allocate_best_worst_counts(n_items: int, top_k: int, worst_k: int) -> Tuple[int, int]:
    top_req = max(0, int(top_k))
    worst_req = max(0, int(worst_k))
    budget = min(max(0, int(n_items)), top_req + worst_req)
    if budget <= 0 or top_req + worst_req == 0:
        return 0, 0
    if top_req == 0:
        return 0, min(worst_req, budget)
    if worst_req == 0:
        return min(top_req, budget), 0
    if budget == 1:
        return (1, 0) if top_req >= worst_req else (0, 1)

    base_top, base_worst = 1, 1
    remaining = budget - 2
    top_remaining_req = top_req - base_top
    worst_remaining_req = worst_req - base_worst
    remaining_req = top_remaining_req + worst_remaining_req
    if remaining <= 0 or remaining_req <= 0:
        return base_top, base_worst

    quotas = [
        ("best", remaining * top_remaining_req / remaining_req, top_remaining_req),
        ("worst", remaining * worst_remaining_req / remaining_req, worst_remaining_req),
    ]
    extra = {name: min(int(quota), cap) for name, quota, cap in quotas}
    left = remaining - sum(extra.values())
    for name, quota, cap in sorted(quotas, key=lambda x: x[1] - int(x[1]), reverse=True):
        if left <= 0:
            break
        if extra[name] < cap:
            extra[name] += 1
            left -= 1
    return base_top + extra["best"], base_worst + extra["worst"]


def build_run_summary(
    successes: List[dict],
    top_trials: set,
    worst_trials: set,
    *,
    higher_is_better: bool = True,
) -> str:
    """Rule-based summary of observed tuning patterns (no extra LLM call)."""
    if not successes:
        return "Run summary: no successful trials yet."

    n = len(successes)
    best_score = max(x["score"] for x in successes) if higher_is_better else min(x["score"] for x in successes)
    lines = [
        "Run summary:",
        f"- Evaluated {n} successful trials; best validation rel_score=1.0 (rank 1).",
    ]

    top_items = [x for x in successes if int(x["trial"]) in top_trials]
    worst_items = [x for x in successes if int(x["trial"]) in worst_trials]
    if not top_items or not worst_items:
        lines.append("- Not enough top/worst separation yet to infer hyperparameter trends.")
        return "\n".join(lines)

    numeric_keys = set()
    for item in top_items + worst_items:
        for key, value in item["config"].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_keys.add(key)

    deltas = []
    for key in sorted(numeric_keys):
        top_vals = [float(item["config"][key]) for item in top_items if key in item["config"]]
        worst_vals = [float(item["config"][key]) for item in worst_items if key in item["config"]]
        if not top_vals or not worst_vals:
            continue
        top_mean = float(np.mean(top_vals))
        worst_mean = float(np.mean(worst_vals))
        denom = max(abs(top_mean), abs(worst_mean), 1e-12)
        rel_gap = abs(top_mean - worst_mean) / denom
        if rel_gap < 0.05:
            continue
        direction = "higher" if top_mean > worst_mean else "lower"
        deltas.append((rel_gap, key, direction, top_mean, worst_mean))

    deltas.sort(reverse=True)
    if deltas:
        lines.append("- Hyperparameters that differ most between top and worst trials:")
        for _, key, direction, top_mean, worst_mean in deltas[:3]:
            lines.append(
                f"  - {key}: top trials tend toward {direction} values "
                f"(top mean={top_mean:.4g}, worst mean={worst_mean:.4g})."
            )
    else:
        lines.append("- Top and worst trials do not show a clear numeric hyperparameter gap yet.")
    return "\n".join(lines)


def _experience_explore_branch(
    *,
    seed: int,
    trial: int,
    mode_tag: str,
    explore_prob: float = 0.5,
) -> str:
    """Deterministic explore/exploit coin flip (reproducible across resumes)."""
    import hashlib

    p = float(explore_prob)
    p = 0.0 if p < 0.0 else (1.0 if p > 1.0 else p)
    key = f"{int(seed)}:{int(trial)}:{mode_tag}:ee50".encode("utf-8")
    u = int(hashlib.md5(key).hexdigest()[:8], 16) / float(0xFFFFFFFF)
    return "explore" if u < p else "exploit"


def build_history_prompt_block(
    configs,
    perfs,
    fixed_algorithm,
    config_space,
    history_window: int = 20,
    top_k: int = 6,
    recent_k: int = 10,
    worst_k: int = 4,
    run_summary: bool = False,
    *,
    prompt_blocks: Optional[Dict[str, bool]] = None,
    budget_total: Optional[int] = None,
    trial_sources: Optional[Dict[int, str]] = None,
    llm_history: Optional[List[dict]] = None,
    experience_self_mode: str = "cohort",
    current_trial: Optional[int] = None,
    experience_self_start_trial: int = 1,
    experience_explore_mode: str = "off",
    experience_explore_prob: float = 0.5,
    experience_blame: Any = None,
    rng_seed: int = 1,
    experience_bank: Any = None,
    experience_retrieve_k: int = 3,
    experience_retrieve_start_trial: int = 1,
    experience_retrieve_rerank_trial: int = 24,
    experience_retrieve_ranker: str = "surrogate",
    experience_retrieve_fade: bool = False,
    experience_retrieve_stop_trial: Optional[int] = None,
    experience_retrieve_spearman_floor: Optional[float] = None,
    experience_retrieve_include_top_bottom: bool = False,
    experience_retrieve_include_prefer: bool = True,
    experience_retrieve_region_style: str = "prefer",
    experience_retrieve_tb_layout: str = "flat",
    experience_retrieve_top_k: int = 1,
    experience_retrieve_bottom_k: int = 1,
    experience_self_gate: str = "clock",
    experience_gate_state: Any = None,
    explore_policy_state: Any = None,
    dataset_id: Optional[str] = None,
    task_type: Optional[str] = None,
    dataset_profile: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Build the reusable history section for the LLM prompt."""
    blocks = validate_prompt_blocks(prompt_blocks)
    ee_mode = str(experience_explore_mode or "off").strip().lower()
    ee_prob = float(experience_explore_prob if experience_explore_prob is not None else 0.5)
    trial_sources = trial_sources or {}
    llm_history = llm_history or []
    successes = []
    for trial, (cfg, perf) in enumerate(zip(configs or [], perfs or []), start=1):
        cfg_dict = _cfg_dict(cfg)
        if cfg_dict.get("algorithm", fixed_algorithm) != fixed_algorithm:
            continue
        score = float(perf)
        if np.isfinite(score):
            successes.append(
                {
                    "trial": trial,
                    "score": score,
                    "config": _config_for_prompt(cfg_dict, fixed_algorithm, config_space),
                }
            )

    # Per-dataset min-max normalization. One prompt only ever holds a single
    # dataset's history, so a single (lo, hi) over the full finite history gives
    # the LLM a consistent [0, 1] scale across datasets (1.0 = best observed),
    # with no raw, dataset-specific magnitudes leaking into SFT data.
    all_scores = [x["score"] for x in successes]
    n_success = len(all_scores)
    hi_score = max(all_scores) if all_scores else 0.0
    lo_score = min(all_scores) if all_scores else 0.0
    span = hi_score - lo_score

    def _rel(score: float) -> float:
        # Degenerate range (all trials tied, e.g. every config scores the same):
        # no signal to convey, so report the top of the scale rather than NaN.
        return 1.0 if span <= 0 else (score - lo_score) / span

    # Competition ranking (1 = best); tied scores share a rank.
    rank_by_trial = {
        int(x["trial"]): 1 + sum(1 for s in all_scores if s > x["score"])
        for x in successes
    }
    rel_by_trial = {int(x["trial"]): _rel(x["score"]) for x in successes}

    best_count, worst_count = _allocate_best_worst_counts(len(successes), top_k, worst_k)
    top = sorted(successes, key=lambda x: x["score"], reverse=True)[:best_count]
    top_trials = {int(x["trial"]) for x in top}
    worst = []
    for item in sorted(successes, key=lambda x: x["score"]):
        if int(item["trial"]) in top_trials:
            continue
        worst.append(item)
        if len(worst) >= worst_count:
            break
    recent = successes[-max(0, recent_k) :] if recent_k > 0 else []

    selected_by_trial = {}
    for items, tag in [(top, "best"), (worst, "worst"), (recent, "recent")]:
        for item in items:
            trial = int(item["trial"])
            selected = selected_by_trial.setdefault(
                trial,
                {
                    "trial": trial,
                    "score": item["score"],
                    "config": item["config"],
                    "tags": [],
                },
            )
            if tag not in selected["tags"]:
                selected["tags"].append(tag)

    selected = [selected_by_trial[k] for k in sorted(selected_by_trial)]
    if len(selected) > history_window:
        priority = {"best": 3, "worst": 2, "recent": 1}

        def trim_key(item):
            tags = item["tags"]
            score_key = -item["score"] if "worst" in tags and "best" not in tags else item["score"]
            return max(priority[tag] for tag in tags), score_key, item["trial"]

        selected = sorted(
            selected,
            key=trim_key,
            reverse=True,
        )[:history_window]
    selected = sorted(selected, key=lambda x: (-x["score"], int(x["trial"])))

    lines = [
        f"Previous validation trials for fixed algorithm {fixed_algorithm}.",
        "rel_score in [0,1] is the validation score min-max normalized within this "
        "dataset (1.0 = best observed, higher is better); rank is the 1-based "
        f"standing among all {n_success} evaluated trials.",
    ]

    extra_sections: List[str] = []
    if blocks.get("budget_context"):
        extra_sections.append(
            build_budget_context_block(
                successes,
                budget_total=budget_total,
                rel_by_trial=rel_by_trial,
                rank_by_trial=rank_by_trial,
            )
        )
    if blocks.get("playbook_run_context"):
        extra_sections.append(
            build_playbook_run_context_block(successes, budget_total=budget_total)
        )
    if blocks.get("best_so_far_trajectory"):
        extra_sections.append(build_best_so_far_trajectory_block(successes, rel_by_trial))
    if extra_sections:
        lines += [""] + extra_sections

    if selected:
        lines += ["", "Selected historical trials:"]
        for i, item in enumerate(selected, start=1):
            trial = int(item["trial"])
            source_tag = ""
            if blocks.get("source_tags"):
                src = trial_sources.get(trial, "unknown")
                source_tag = f"source={src}, "
            lines.append(
                f"{i}. trial={item['trial']}, {source_tag}rel_score={_rel(item['score']):.4f}, "
                f"rank={rank_by_trial[trial]}/{n_success}, "
                f"tags={_json(item['tags'])}, config={_json(item['config'])}"
            )

    if blocks.get("hp_marginals"):
        lines += ["", build_hp_marginals_block(successes, config_space, fixed_algorithm, rel_by_trial)]

    # Static coverage_map only when not using fifty_fifty (that mode owns coverage as explore).
    if blocks.get("coverage_map") and ee_mode != "fifty_fifty":
        lines += ["", build_coverage_map_block(successes, config_space, fixed_algorithm)]

    if run_summary and successes:
        if blocks.get("explore_refine_summary"):
            summary = build_explore_refine_summary(successes, config_space, fixed_algorithm)
        else:
            summary = build_run_summary(successes, top_trials, {int(x["trial"]) for x in worst})
        if blocks.get("rationale_diversity_summary"):
            summary += "\n\n" + build_rationale_diversity_summary(successes, llm_history, rel_by_trial)
        lines += ["", summary]

    experience_branch = None
    experience_self_active = False
    retrieved_hits_meta: List[Dict[str, Any]] = []
    gate_meta: Dict[str, Any] = {}
    if blocks.get("experience_retrieve") and experience_bank is not None:
        trial_no = int(current_trial) if current_trial is not None else 10**9
        try:
            from mindware.components.experience.retrieve import retrieve_for_run
        except Exception:
            retrieve_for_run = None  # type: ignore
        if retrieve_for_run is not None:
            fade_after = None
            if experience_retrieve_fade and blocks.get("experience_self"):
                fade_after = int(experience_self_start_trial or 1)
            stop_trial = None
            if experience_retrieve_stop_trial not in (None, "", 0, "0"):
                stop_trial = int(experience_retrieve_stop_trial)
            ret_block, hits = retrieve_for_run(
                experience_bank,
                dataset_id=dataset_id,
                algorithm_id=fixed_algorithm,
                task_type=task_type,
                dataset_profile=dataset_profile,
                successes=successes,
                config_space=config_space,
                k=int(experience_retrieve_k or 3),
                start_trial=int(experience_retrieve_start_trial or 1),
                rerank_after=int(experience_retrieve_rerank_trial or 24),
                current_trial=trial_no,
                fade_after=fade_after,
                stop_trial=stop_trial,
                ranker=str(experience_retrieve_ranker or "surrogate"),
                spearman_floor=experience_retrieve_spearman_floor,
                include_top_bottom=bool(experience_retrieve_include_top_bottom),
                include_prefer=bool(experience_retrieve_include_prefer),
                region_style=str(experience_retrieve_region_style or "prefer"),
                tb_layout=str(experience_retrieve_tb_layout or "flat"),
                retrieve_top_k=1 if experience_retrieve_top_k is None else max(0, int(experience_retrieve_top_k)),
                retrieve_bottom_k=1 if experience_retrieve_bottom_k is None else max(0, int(experience_retrieve_bottom_k)),
            )
            if ret_block:
                lines += ["", ret_block]
            retrieved_hits_meta = [
                {
                    "id": h.exp_id,
                    "score": h.score,
                    "spearman": h.spearman,
                    "features": h.features,
                }
                for h in hits
            ]

    if blocks.get("experience_self"):
        trial_no = int(current_trial) if current_trial is not None else 10**9
        start_at = max(1, int(experience_self_start_trial or 1))
        gate_mode = str(experience_self_gate or "clock").strip().lower()
        from mindware.components.optimizers.experience_gate import ExperienceGate, apply_strength, filter_fanova

        gate = experience_gate_state
        if gate is None:
            gate = ExperienceGate(mode=gate_mode, cap=start_at)

        exp_block = build_experience_self_block(
            successes,
            min_trials=8,
            mode=experience_self_mode or "cohort",
            config_space=config_space,
            experience_blame=experience_blame,
        )
        active, gate_meta = gate.decide(
            current_trial=trial_no, successes=successes, block=exp_block or ""
        )
        if gate_mode in ("clock", "t50", ""):
            active = trial_no >= start_at

        experience_self_active = bool(active)
        if active:
            if gate_mode in ("fanova", "fanova_full", "fanova_full_cs", "fanova_full_ix", "fanova_rf"):
                engine = {
                    "fanova": "proxy",
                    "fanova_full": "full",
                    "fanova_full_cs": "full_cs",
                    "fanova_full_ix": "full_ix",
                    "fanova_rf": "rf",
                }[gate_mode]
                exp_block = filter_fanova(
                    exp_block or "",
                    successes,
                    gate.fanova_thresh,
                    engine=engine,
                    config_space=config_space,
                )
                from mindware.components.optimizers.experience_gate import _LAST_FANOVA_META

                gate_meta.update(_LAST_FANOVA_META)
            if gate_mode == "strength":
                exp_block = apply_strength(exp_block or "", len(successes), gate.floor, gate.cap)
            if gate_mode == "turbo":
                from mindware.components.optimizers.experience_gate import apply_turbo_width

                exp_block = apply_turbo_width(exp_block or "", gate.turbo_L)
            if ee_mode in ("off", "", "none"):
                experience_branch = "exploit"
            else:
                from mindware.components.optimizers.experience_gate import ExplorePolicy, XOR_EXPLORE_MODES

                policy = explore_policy_state
                if policy is None:
                    policy = ExplorePolicy(
                        mode=ee_mode,
                        seed=int(rng_seed),
                        p_default=ee_prob,
                    )
                if ee_mode in XOR_EXPLORE_MODES or ee_mode == "fifty_fifty":
                    experience_branch = policy.choose(
                        trial=int(trial_no),
                        successes=successes,
                        config_space=config_space,
                        fixed_algorithm=fixed_algorithm,
                        mode_tag=str(experience_self_mode or "cohort"),
                        exp_block=exp_block or "",
                    )
                    ee_prob = float(getattr(policy, "last_p", ee_prob))
                    if getattr(policy, "last_meta", None):
                        gate_meta.update(policy.last_meta)
                else:
                    experience_branch = "exploit"

            if experience_branch == "exploit":
                if exp_block:
                    lines += ["", exp_block]
            elif experience_branch == "explore":
                cov_block = build_coverage_map_block(successes, config_space, fixed_algorithm)
                if cov_block:
                    lines += ["", cov_block]

    meta = {
        "selected_best_trials": [x["trial"] for x in top],
        "selected_recent_trials": [x["trial"] for x in recent],
        "selected_worst_trials": [x["trial"] for x in worst],
        "selected_trials": [{"trial": x["trial"], "tags": x["tags"]} for x in selected],
        "allocated_best_k": best_count,
        "allocated_worst_k": worst_count,
        "history_window": history_window,
        "run_summary": bool(run_summary),
        "prompt_blocks": blocks,
        "experience_self_start_trial": int(experience_self_start_trial or 1),
        "experience_self_active": experience_self_active,
        "experience_explore_mode": ee_mode,
        "experience_explore_prob": ee_prob,
        "experience_explore_branch": experience_branch,
        "experience_retrieve_hits": retrieved_hits_meta,
        "score_normalization": {
            "method": "per_dataset_minmax",
            "lo": lo_score,
            "hi": hi_score,
            "n_success": n_success,
        },
    }
    if gate_meta:
        meta.update(gate_meta)
    return "\n".join(lines), meta


def _payload_looks_like_config(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if isinstance(payload.get("config"), dict):
        return True
    # Bare HP map (no wrapper): keys like "algo:hp".
    return bool(payload) and all(isinstance(k, str) for k in payload) and any(":" in k for k in payload)


def _truncate_leaked_turn_suffix(text: str) -> str:
    """Drop text after a leaked chat turn (common with overfit SFT models)."""
    lowered = text.lower()
    cut = len(text)
    for marker in (
        "\nuser\n",
        "\nassistant\n",
        "\nuser:",
        "\nassistant:",
        "<|im_start|>",
        "",
        "<|endoftext|>",
    ):
        pos = lowered.find(marker)
        if pos > 0:
            cut = min(cut, pos)
    return text[:cut].strip() if cut < len(text) else text


def _iter_json_objects(scan_text: str, start_at: int = 0):
    """Yield (start_index, object) for each decodable ``{...}`` in *scan_text*."""
    decoder = json.JSONDecoder()
    idx = start_at
    last_err: Optional[Exception] = None
    while True:
        start = scan_text.find("{", idx)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(scan_text, start)
        except json.JSONDecodeError as exc:
            last_err = exc
            idx = start + 1
            continue
        yield start, obj
        idx = end
    if last_err is not None:
        raise last_err


def _pick_config_payload(candidates: list[tuple[int, Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Prefer wrapped ``{"config": ...}`` over bare HP maps; else earliest match."""
    if not candidates:
        return None
    wrapped = [obj for _, obj in candidates if isinstance(obj.get("config"), dict)]
    if wrapped:
        return wrapped[0]
    return candidates[0][1]


def parse_llm_json(text: str) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """Parse LLM HPO replies.

    Accepts JSON-only or ``Rationale:`` + ``JSON:`` wrappers. Qwen SFT models
    often continue past the first object into a leaked ``\\nuser\\n`` turn; we
    take the first complete JSON object that looks like a config (not first
    ``{`` through last ``}``, which breaks on multi-object leaks).
    """
    raw_text = _truncate_leaked_turn_suffix((text or "").strip())
    if not raw_text:
        raise ValueError("Empty LLM response.")

    # Prefer fenced blocks when present, else scan the full reply.
    scan_text = raw_text
    fence_prefix = ""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        fence_prefix = raw_text[: match.start()].strip()
        scan_text = match.group(1).strip()

    # If a JSON: label exists, scan from there first (skips braces in rationales).
    label = re.search(r"(?is)\bjson\s*:\s*", scan_text)
    search_bases = [label.end()] if label else []
    search_bases.append(0)

    payload = None
    prefix = fence_prefix
    last_err: Optional[Exception] = None
    for base in search_bases:
        candidates: list[tuple[int, Dict[str, Any]]] = []
        try:
            for start, obj in _iter_json_objects(scan_text, base):
                if isinstance(obj, dict) and _payload_looks_like_config(obj):
                    candidates.append((start, obj))
        except json.JSONDecodeError as exc:
            last_err = exc
            continue
        picked = _pick_config_payload(candidates)
        if picked is not None:
            payload = picked
            if not prefix:
                first_start = candidates[0][0]
                prefix = scan_text[:first_start].strip()
            break

    if payload is None:
        if last_err is not None:
            raise last_err
        raise ValueError("LLM response JSON must be an object with a config.")

    config = payload.get("config", payload)
    if not isinstance(config, dict):
        raise ValueError("LLM response field 'config' must be an object.")
    rationale = str(payload.get("rationale", "")).strip()
    if not rationale and prefix:
        rationale = re.sub(r"(?is)^rationale\s*:\s*", "", prefix).strip()
        rationale = re.sub(r"(?is)\n?\s*json\s*:\s*$", "", rationale).strip()
    return config, rationale, payload


class PT4OptLLMOptimizer(BaseOptimizer):
    optimizer_label = "ptllm"

    def __init__(
        self,
        evaluator,
        config_space,
        name,
        eval_type,
        time_limit=None,
        evaluation_limit=None,
        per_run_time_limit=300,
        per_run_mem_limit=1024,
        inner_iter_num_per_iter=1,
        timestamp=None,
        output_dir="./",
        seed=1,
        n_jobs=1,
        topk=50,
        optimizer_params=None,
        task_id="GenTrans",
        logging_enabled=True,
        **kwargs,
    ):
        super().__init__(
            evaluator=evaluator,
            config_space=config_space,
            name=name,
            eval_type=eval_type,
            task_id=task_id,
            time_limit=time_limit,
            evaluation_limit=evaluation_limit,
            per_run_time_limit=per_run_time_limit,
            per_run_mem_limit=per_run_mem_limit,
            inner_iter_num_per_iter=inner_iter_num_per_iter,
            timestamp=timestamp,
            output_dir=output_dir,
            seed=seed,
            topk=topk,
        )
        self.params = dict(optimizer_params or {})
        self.params.update(kwargs)
        if n_jobs != 1:
            self.logger.warning("PT4OptLLMOptimizer is sequential; n_jobs=%s is ignored.", n_jobs)

        self.KD = KnowledgeDatabase.instance()
        self.configs = self.KD.configs
        self.perfs = self.KD.perfs
        self.objective_function = partial(wrapper_evaluater, evaluator=evaluator, per_run_time_limit=per_run_time_limit)

        self.algorithm_hp = self._get_hp("algorithm")
        self.fixed_algorithm = self._get_single_algorithm()
        self.n_no_history_samples = int(self.params.get("n_no_history_samples", self.params.get("n_initial_samples", 4)))
        self.n_initial_samples = self.n_no_history_samples
        self.history_window = int(self.params.get("history_window", 20))
        self.top_k = int(self.params.get("top_k", 6))
        self.recent_k = int(self.params.get("recent_k", 10))
        self.worst_k = int(self.params.get("worst_k", 4))
        self.run_summary = bool(self.params.get("run_summary", False))
        self.use_tcs_analysis = bool(self.params.get("tcs_analysis", False))
        self.tcs_single_agent = bool(self.params.get("tcs_single_agent", False))
        self.prompt_blocks = validate_prompt_blocks(self.params.get("prompt_blocks"))
        self.experience_self_mode = str(self.params.get("experience_self_mode") or "cohort")
        self.experience_self_start_trial = max(1, int(self.params.get("experience_self_start_trial") or 1))
        self.experience_self_gate = str(self.params.get("experience_self_gate") or "clock").strip().lower()
        from mindware.components.optimizers.experience_gate import ExperienceGate

        self._experience_gate = ExperienceGate(
            mode=self.experience_self_gate,
            cap=self.experience_self_start_trial,
        )
        self.experience_explore_mode = str(self.params.get("experience_explore_mode") or "off").strip().lower()
        self.experience_explore_prob = float(self.params.get("experience_explore_prob") or 0.5)
        from mindware.components.optimizers.experience_gate import ExplorePolicy

        self._explore_policy = ExplorePolicy(
            mode=self.experience_explore_mode,
            seed=int(getattr(self, "seed", 1) or 1),
            p_default=self.experience_explore_prob,
            tpe_min_trial=int(self.params.get("experience_explore_tpe_min_trial") or 0),
        )
        self.experience_blame = dict(self.params.get("experience_blame") or {})
        self.experience_retrieve_k = max(1, int(self.params.get("experience_retrieve_k") or 3))
        self.experience_retrieve_start_trial = max(1, int(self.params.get("experience_retrieve_start_trial") or 1))
        self.experience_retrieve_rerank_trial = max(1, int(self.params.get("experience_retrieve_rerank_trial") or 24))
        self.experience_retrieve_ranker = str(self.params.get("experience_retrieve_ranker") or "surrogate")
        self.experience_retrieve_fade = bool(self.params.get("experience_retrieve_fade", False))
        _stop = self.params.get("experience_retrieve_stop_trial")
        self.experience_retrieve_stop_trial = None if _stop in (None, "", 0, "0") else int(_stop)
        _floor = self.params.get("experience_retrieve_spearman_floor")
        self.experience_retrieve_spearman_floor = None if _floor in (None, "") else float(_floor)
        self.experience_retrieve_include_top_bottom = bool(
            self.params.get("experience_retrieve_include_top_bottom", False)
        )
        self.experience_retrieve_include_prefer = bool(
            self.params.get("experience_retrieve_include_prefer", True)
        )
        self.experience_retrieve_region_style = str(
            self.params.get("experience_retrieve_region_style") or "prefer"
        )
        self.experience_retrieve_tb_layout = str(
            self.params.get("experience_retrieve_tb_layout") or "flat"
        )
        _tk = self.params.get("experience_retrieve_top_k")
        self.experience_retrieve_top_k = 1 if _tk in (None, "") else max(0, int(_tk))
        _bk = self.params.get("experience_retrieve_bottom_k")
        self.experience_retrieve_bottom_k = 1 if _bk in (None, "") else max(0, int(_bk))
        self.dataset_id = self.params.get("dataset_id")
        self.task_type = self.params.get("task_type")
        self._experience_bank = None
        self._experience_bank_warned = False
        self.trial_sources: Dict[int, str] = {}
        self._tpe_hint_config: Optional[Dict[str, Any]] = None
        self.max_duplicate_reprompts = min(1, max(0, int(self.params.get("max_duplicate_reprompts", 1))))
        self.fallback_on_transport_error = bool(self.params.get("fallback_on_transport_error", False))
        self.fallback_on_bad_json = bool(self.params.get("fallback_on_bad_json", True))
        self.fallback_on_invalid_config = bool(self.params.get("fallback_on_invalid_config", True))
        # Match SFT target format: S0/S1a = json_only; S1b/gpt default = rationale_json.
        _fmt = str(
            self.params.get("response_format")
            or self.params.get("llm_response_format")
            or "rationale_json"
        ).strip().lower()
        self.response_format = "json_only" if _fmt in ("json", "json_only", "config_only") else "rationale_json"

        self.llm_client = self.params.get("llm_client")
        self.mock_responses = list(self.params.get("mock_llm_responses") or [])
        self.use_mock_llm = bool(self.params.get("mock_llm") or self.params.get("llm_client_type") == "mock")
        self._setup_openai_client()

        self.llm_history: List[Dict[str, Any]] = []
        self.llm_query_cost: List[float] = []
        self.llm_query_time: List[float] = []
        self.llm_call_count = 0
        self.trial_cnt = 0
        self.exp_output = {}
        self.logging_enabled = logging_enabled
        self.traj_rows: List[Dict[str, Any]] = []
        self.traj_csv_path = os.path.join(self.output_dir, f"{int(self.timestamp)}_opt_{self.optimizer_label}_traj.csv")
        self.conversation_log_dir = os.path.join(self.output_dir, "llm_conversations")
        self.start_wall_time = time.time()
        os.makedirs(self.conversation_log_dir, exist_ok=True)

        self.seen_signatures = set()
        self.seen_objectives: Dict[Tuple[Tuple[str, str], ...], float] = {}
        self._seen_config_count = 0
        self._sync_seen_signatures()
        # Optional FIFO of flat HP dicts used for early llm_warm_start trials
        # (e.g. best-of-N from an external warm-start agent).
        presets = self.params.get("preset_warm_start_configs") or []
        self._preset_warm_start_queue: List[Dict[str, Any]] = [
            dict(p) for p in presets if isinstance(p, dict)
        ]

    def _setup_openai_client(self):
        self.openai_client = None
        if self.use_mock_llm or self.llm_client is not None:
            return
        from openai import OpenAI

        engine = self.params.get("chat_engine", LLMCONFIGS.opt_chat_engine)
        cfg = AVAILABLE_LLMs.get(engine, {})
        self.llm_model = self.params.get("llm_model") or cfg.get("model", engine)
        self.llm_temperature = float(self.params.get("temperature", cfg.get("temperature", 0.2)))
        self.llm_max_tokens = int(self.params.get("max_tokens", 2048))
        self.llm_timeout = int(self.params.get("llm_timeout", 180))
        self.prompt_cost = LLM_COST_DICT.get(engine, {}).get("promt_cost_per_1k", 0.0)
        self.completion_cost = LLM_COST_DICT.get(engine, {}).get("completion_cost_per_1k", 0.0)
        self.openai_client = OpenAI(
            api_key=self.params.get("api_key") or cfg.get("api_key", LLMCONFIGS.OPENAI_KEY),
            base_url=self.params.get("base_url") or cfg.get("base_url", LLMCONFIGS.OPENAI_BASE_URL),
        )

    def run(self):
        while not (self.early_stopped_flag or self.timeout_flag):
            self.iterate()
        return np.max(self.perfs) if self.perfs else -np.inf

    def iterate(self, budget=MAX_INT):
        start = time.time()
        rows = []
        for _ in range(self.inner_iter_num_per_iter):
            if self._quota_or_time_is_up() or time.time() - start > budget:
                break
            iter_idx = self.trial_cnt
            self.trial_cnt += 1
            iter_start = time.time()

            include_history = iter_idx >= self.n_no_history_samples
            source = "llm" if include_history else "llm_warm_start"
            cfg, note, llm_cost, llm_time, raw, payload, meta, rationale, prompt_input = self._suggest_config(
                iter_idx,
                include_history=include_history,
            )

            obs = self.objective_function(cfg)
            raw_obj = float(obs.objectives[0]) if obs.objectives else np.inf
            success = obs.trial_state == SUCCESS and np.isfinite(raw_obj)
            perf = -raw_obj if success else float("-inf")
            eval_time = obs.elapsed_time or 0.0

            if success:
                self.exp_output[time.time()] = (cfg, perf)
                self.update_saver([cfg], [raw_obj])
                self._update_history(obs, source, iter_idx, cfg, rationale)
                self._mark_seen(cfg, raw_obj)
            self._finish_llm_history(cfg, perf, success, obs)
            self._dump_conversation(iter_idx, prompt_input, raw, payload, cfg, obs, meta, note)

            iter_time = time.time() - iter_start
            rows.append({"perf": perf, "config": cfg, "eval_time": eval_time, "llm_time": llm_time, "iter_time": iter_time})
            self._record_event(source, iter_idx, cfg, perf, llm_cost, llm_time, eval_time, iter_time, note)

        if self._quota_or_time_is_up():
            self.timeout_flag = True
        self._update_incumbent()
        return self.incumbent_perf, time.time() - start, self.incumbent_config, rows

    def _suggest_config(self, iter_idx: int, include_history: bool = True):
        t0 = time.time()
        raw, payload, rationale, cost = "", {}, "", 0.0
        attempts = []
        rejected_duplicate_configs = []
        prompt_input = {}
        # External warm-start presets replace the first llm_warm_start suggestions.
        if (not include_history) and self._preset_warm_start_queue:
            flat = dict(self._preset_warm_start_queue.pop(0))
            flat.setdefault("algorithm", self.fixed_algorithm)
            cfg = self.KD._dict_to_cfg(flat)
            note = "preset warm-start configuration (external agent best-of-N)"
            prompt_input = {
                "system_prompt": "",
                "user_prompt": "",
                "include_history": False,
                "preset_warm_start": True,
                "preset_config": _config_for_prompt(cfg, self.fixed_algorithm, self.config_space),
                "messages": [],
            }
            self._append_llm_history(
                iter_idx,
                "PENDING",
                "",
                {"config": flat, "source": "preset_warm_start"},
                cfg,
                note,
                attempts=[],
                rejected_duplicate_configs=[],
            )
            meta = {
                "history_included": False,
                "preset_warm_start": True,
                "selected_best_trials": [],
                "selected_recent_trials": [],
                "selected_worst_trials": [],
                "selected_trials": [],
            }
            self._last_history_meta = meta
            return cfg, note, 0.0, time.time() - t0, "", {"config": flat}, meta, note, prompt_input
        try:
            system_prompt, user_prompt = self._build_prompts(
                include_history=include_history,
                current_trial=iter_idx + 1,
            )
            prompt_input = {
                "system_prompt": system_prompt,
                "system_prompt_lines": _lines(system_prompt),
                "user_prompt": user_prompt,
                "user_prompt_lines": _lines(user_prompt),
                "include_history": include_history,
                "tcs_analysis": bool(include_history and self.use_tcs_analysis),
                "analysis_text": getattr(self, "_last_analysis_text", None),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            prompt_input["attempts"] = attempts
            cfg, raw, payload, rationale, cost, attempt = self._request_config_attempt(
                system_prompt,
                user_prompt,
                attempt_id=0,
            )
            if include_history and self.use_tcs_analysis:
                cost += float(getattr(self, "_last_analysis_cost", 0.0))
            attempts.append(attempt)
            note = "llm suggested configuration" if include_history else "llm warm-start configuration"

            if attempt["duplicate"]:
                rejected_duplicate_configs.append(_config_for_prompt(cfg, self.fixed_algorithm, self.config_space))
                reprompt = self._reprompt_duplicate_if_needed(system_prompt, user_prompt, prompt_input, cfg)
                if reprompt is not None:
                    cfg, raw, payload, rationale, reprompt_cost, attempt = reprompt
                    cost += reprompt_cost
                    attempts.append(attempt)

            if self._is_duplicate(cfg):
                rejected = _config_for_prompt(cfg, self.fixed_algorithm, self.config_space)
                if rejected not in rejected_duplicate_configs:
                    rejected_duplicate_configs.append(rejected)
                cfg, note = self._sample_nonduplicate_config(), self._duplicate_fallback_note(attempts)
                self._append_llm_history(
                    iter_idx,
                    "DUPLICATE",
                    raw,
                    payload,
                    cfg,
                    note,
                    attempts=attempts,
                    rejected_duplicate_configs=rejected_duplicate_configs,
                )
            else:
                if len(attempts) > 1:
                    note = (
                        "llm suggested configuration after duplicate reprompt"
                        if include_history
                        else "llm warm-start configuration after duplicate reprompt"
                    )
                self._append_llm_history(
                    iter_idx,
                    "PENDING",
                    raw,
                    payload,
                    cfg,
                    rationale,
                    attempts=attempts,
                    rejected_duplicate_configs=rejected_duplicate_configs,
                )
            meta = self._last_history_meta
            return cfg, note, cost, time.time() - t0, raw, payload, meta, rationale, prompt_input
        except Exception as exc:
            raw = getattr(exc, "raw_response", raw)
            payload = getattr(exc, "parsed_payload", payload)
            if (not raw and not self.fallback_on_transport_error) or (raw and not self.fallback_on_bad_json):
                raise
            if raw and payload and not self.fallback_on_invalid_config:
                raise
            cfg = self._sample_nonduplicate_config()
            note = f"random fallback after llm error: {exc}"
            self._append_llm_history(
                iter_idx,
                "FAILED",
                raw,
                payload,
                cfg,
                note,
                error=str(exc),
                attempts=attempts,
                rejected_duplicate_configs=rejected_duplicate_configs,
            )
            return cfg, note, cost, time.time() - t0, raw, payload, getattr(self, "_last_history_meta", {}), rationale, prompt_input

    def _request_config_attempt(self, system_prompt: str, user_prompt: str, attempt_id: int):
        raw, cost = self._complete_llm(system_prompt, user_prompt)
        payload = {}
        try:
            config_dict, rationale, payload = parse_llm_json(raw)
            cfg = self.KD._dict_to_cfg(dict(config_dict, algorithm=self.fixed_algorithm))
        except Exception as exc:
            setattr(exc, "raw_response", raw)
            setattr(exc, "parsed_payload", payload)
            raise
        return cfg, raw, payload, rationale, cost, self._make_attempt_record(attempt_id, raw, payload, cfg)

    def _make_attempt_record(self, attempt_id: int, raw: str, payload: Dict[str, Any], cfg) -> Dict[str, Any]:
        return {
            "attempt": attempt_id,
            "raw_response": raw,
            "parsed_payload": payload,
            "normalized_config": _cfg_dict(cfg),
            "duplicate": self._is_duplicate(cfg),
        }

    def _reprompt_duplicate_if_needed(self, system_prompt: str, user_prompt: str, prompt_input: Dict[str, Any], cfg):
        if self.max_duplicate_reprompts <= 0:
            return None
        reprompt_user = self._build_duplicate_reprompt(user_prompt, cfg)
        prompt_input["reprompt_user_prompt"] = reprompt_user
        prompt_input["reprompt_user_prompt_lines"] = _lines(reprompt_user)
        prompt_input["reprompt_messages"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": reprompt_user},
        ]
        return self._request_config_attempt(system_prompt, reprompt_user, attempt_id=1)

    def _duplicate_fallback_note(self, attempts: List[Dict[str, Any]]) -> str:
        if len(attempts) > 1:
            return "random fallback after duplicate llm reprompt"
        return "random fallback for duplicate llm configuration"

    def _complete_llm(self, system_prompt: str, user_prompt: str):
        self.llm_call_count += 1
        if self.llm_client is not None:
            result = self.llm_client.complete(system_prompt, user_prompt)
            if isinstance(result, str):
                return result, 0.0
            return result.content, float(getattr(result, "cost", 0.0))
        if self.use_mock_llm:
            return (self.mock_responses.pop(0) if self.mock_responses else 'Rationale: mock default\n\nJSON:\n{"config": {}}'), 0.0
        resp = self.openai_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
            timeout=self.llm_timeout,
        )
        usage = getattr(resp, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        cost = prompt_tokens / 1000.0 * self.prompt_cost + completion_tokens / 1000.0 * self.completion_cost
        return resp.choices[0].message.content, cost

    def _build_duplicate_reprompt(self, user_prompt: str, duplicate_cfg) -> str:
        rejected = _config_for_prompt(duplicate_cfg, self.fixed_algorithm, self.config_space)
        return "\n\n".join(
            [
                user_prompt,
                "The previous JSON configuration was rejected because it duplicates an already evaluated trial.",
                f"Rejected duplicate configuration: {_json(rejected)}",
                "Return one different configuration. Keep the same output format and use every tunable hyperparameter listed in Search space.",
            ]
        )

    def _build_prompts(self, include_history: bool = True, current_trial: Optional[int] = None):
        if include_history and self.use_tcs_analysis:
            if self.tcs_single_agent:
                return self._build_tcs_single_agent_prompts()
            return self._build_tcs_prompts()
        return self._build_history_prompts(include_history, current_trial=current_trial)

    def _metric_name(self) -> str:
        profile = self.params.get("dataset_profile") or {}
        return str(profile.get("metric", "validation score"))

    def _run_analysis_agent(self, tcs_report: str) -> Tuple[str, float]:
        hp_names = _hp_names_from_space(self.config_space)
        system, user = build_analysis_agent_prompts(
            tcs_report,
            metric_name=self._metric_name(),
            higher_is_better=True,
            hp_names=hp_names,
        )
        return self._complete_llm(system, user)

    def _build_tcs_prompts(self):
        tcs_report, tcs_meta = build_tcs_report(
            self.configs,
            self.perfs,
            self.fixed_algorithm,
            self.config_space,
            metric_name=self._metric_name(),
            higher_is_better=True,
        )
        analysis_text, analysis_cost = self._run_analysis_agent(tcs_report)
        self._last_analysis_cost = float(analysis_cost)
        self._last_analysis_text = analysis_text

        meta = dict(tcs_meta)
        meta.update(
            {
                "history_included": True,
                "tcs_analysis": True,
                "analysis_text": analysis_text,
                "analysis_cost": float(analysis_cost),
            }
        )
        self._last_history_meta = meta

        system = (
            "You are the Optimizer Agent for single-algorithm hyperparameter optimization. "
            f"The algorithm is fixed to '{self.fixed_algorithm}'. "
            "You receive a Trajectory Context Summary (TCS) and an Analysis Agent report. "
            "rel_score is min-max normalized within this dataset (1.0 = best observed, higher is better). "
            "Propose the next configuration to maximize validation score. "
            "First write a short rationale, then write one JSON object containing the configuration. "
            "Do not use markdown."
        )
        user = "\n\n".join(
            [
                self._dataset_text(),
                f"Fixed algorithm: {self.fixed_algorithm}",
                self._search_space_text(),
                tcs_report,
                "Analysis Agent Report:",
                analysis_text,
                'Return exactly this format:\n'
                'Rationale: <short reason for the next configuration>\n\n'
                'JSON:\n{"config": {"<hyperparameter_name>": "<value>"}}\n'
                "Return values for every tunable hyperparameter listed in Search space. "
                "Do not return a configuration listed under EVALUATED CONFIGURATIONS in the TCS. "
                "Use the exact hyperparameter names shown in Search space.",
            ]
        )
        return system, user

    def _build_tcs_single_agent_prompts(self):
        tcs_report, tcs_meta = build_tcs_report(
            self.configs,
            self.perfs,
            self.fixed_algorithm,
            self.config_space,
            metric_name=self._metric_name(),
            higher_is_better=True,
        )
        hp_names = _hp_names_from_space(self.config_space)
        system, user = build_tcs_single_agent_prompts(
            tcs_report,
            fixed_algorithm=self.fixed_algorithm,
            dataset_text=self._dataset_text(),
            search_space_text=self._search_space_text(),
            metric_name=self._metric_name(),
            higher_is_better=True,
            hp_names=hp_names,
        )
        self._last_analysis_cost = 0.0
        self._last_analysis_text = None

        meta = dict(tcs_meta)
        meta.update(
            {
                "history_included": True,
                "tcs_analysis": True,
                "tcs_single_agent": True,
                "analysis_text": None,
                "analysis_cost": 0.0,
            }
        )
        self._last_history_meta = meta
        return system, user

    def _get_experience_bank(self):
        if self._experience_bank is not None:
            return self._experience_bank
        if not self.prompt_blocks.get("experience_retrieve"):
            return None
        path = self.params.get("experience_bank_path")
        if not path:
            if not self._experience_bank_warned:
                self.logger.warning("experience_retrieve is on but experience_bank_path is empty")
                self._experience_bank_warned = True
            return None
        from pathlib import Path as _Path
        from mindware.components.experience.retrieve import ExperienceBank

        p = _Path(path)
        if not p.is_file():
            if not self._experience_bank_warned:
                self.logger.warning("experience bank not found: %s", p)
                self._experience_bank_warned = True
            return None
        try:
            self._experience_bank = ExperienceBank.load(p)
            self.logger.info("Loaded experience bank %s (%s rows)", p, len(self._experience_bank))
        except Exception as exc:
            if not self._experience_bank_warned:
                self.logger.warning("failed to load experience bank %s: %s", p, exc)
                self._experience_bank_warned = True
            return None
        return self._experience_bank

    def _build_history_prompts(self, include_history: bool = True, current_trial: Optional[int] = None):
        if include_history:
            history, meta = build_history_prompt_block(
                self.configs,
                self.perfs,
                self.fixed_algorithm,
                self.config_space,
                self.history_window,
                self.top_k,
                self.recent_k,
                self.worst_k,
                run_summary=self.run_summary,
                prompt_blocks=self.prompt_blocks,
                budget_total=self.evaluation_num_limit,
                trial_sources=self.trial_sources,
                llm_history=self.llm_history,
                experience_self_mode=self.experience_self_mode,
                current_trial=current_trial,
                experience_self_start_trial=self.experience_self_start_trial,
                experience_explore_mode=self.experience_explore_mode,
                experience_explore_prob=self.experience_explore_prob,
                experience_blame=self.experience_blame,
                rng_seed=int(getattr(self, "seed", 1) or 1),
                experience_bank=self._get_experience_bank(),
                experience_retrieve_k=self.experience_retrieve_k,
                experience_retrieve_start_trial=self.experience_retrieve_start_trial,
                experience_retrieve_rerank_trial=self.experience_retrieve_rerank_trial,
                experience_retrieve_ranker=self.experience_retrieve_ranker,
                experience_retrieve_fade=self.experience_retrieve_fade,
                experience_retrieve_stop_trial=self.experience_retrieve_stop_trial,
                experience_retrieve_spearman_floor=self.experience_retrieve_spearman_floor,
                experience_retrieve_include_top_bottom=self.experience_retrieve_include_top_bottom,
                experience_retrieve_include_prefer=self.experience_retrieve_include_prefer,
                experience_retrieve_region_style=self.experience_retrieve_region_style,
                experience_retrieve_tb_layout=self.experience_retrieve_tb_layout,
                experience_retrieve_top_k=self.experience_retrieve_top_k,
                experience_retrieve_bottom_k=self.experience_retrieve_bottom_k,
                experience_self_gate=self.experience_self_gate,
                experience_gate_state=getattr(self, "_experience_gate", None),
                explore_policy_state=getattr(self, "_explore_policy", None),
                dataset_id=self.dataset_id,
                task_type=self.task_type,
                dataset_profile=self.params.get("dataset_profile"),
            )
            meta["history_included"] = True
        else:
            history = "Selected historical trials: omitted for this warm-start query."
            meta = {
                "selected_best_trials": [],
                "selected_recent_trials": [],
                "selected_worst_trials": [],
                "selected_trials": [],
                "allocated_best_k": 0,
                "allocated_worst_k": 0,
                "history_window": self.history_window,
                "history_included": False,
                "prompt_blocks": self.prompt_blocks,
            }
            if self.prompt_blocks.get("experience_retrieve") and self._get_experience_bank() is not None:
                from mindware.components.experience.retrieve import retrieve_for_run

                packed = []
                for i, (cfg, perf) in enumerate(zip(self.configs or [], self.perfs or []), start=1):
                    packed.append(
                        {
                            "trial": i,
                            "config": _cfg_dict(cfg),
                            "score": float(perf),
                        }
                    )
                ret_block, hits = retrieve_for_run(
                    self._get_experience_bank(),
                    dataset_id=self.dataset_id,
                    algorithm_id=self.fixed_algorithm,
                    task_type=self.task_type,
                    dataset_profile=self.params.get("dataset_profile"),
                    successes=packed,
                    config_space=self.config_space,
                    k=self.experience_retrieve_k,
                    start_trial=self.experience_retrieve_start_trial,
                    rerank_after=self.experience_retrieve_rerank_trial,
                    current_trial=int(current_trial) if current_trial is not None else 1,
                    fade_after=None,
                    ranker=self.experience_retrieve_ranker,
                    spearman_floor=self.experience_retrieve_spearman_floor,
                    include_top_bottom=self.experience_retrieve_include_top_bottom,
                    include_prefer=self.experience_retrieve_include_prefer,
                    region_style=self.experience_retrieve_region_style,
                    tb_layout=self.experience_retrieve_tb_layout,
                    retrieve_top_k=self.experience_retrieve_top_k,
                    retrieve_bottom_k=self.experience_retrieve_bottom_k,
                    stop_trial=self.experience_retrieve_stop_trial,
                )
                if ret_block:
                    history = history + "\n\n" + ret_block
                meta["experience_retrieve_hits"] = [
                    {"id": h.exp_id, "score": h.score, "spearman": h.spearman}
                    for h in hits
                ]
        self._last_history_meta = meta
        if self.response_format == "json_only":
            system = (
                "You are a single-algorithm hyperparameter optimization assistant. "
                f"The algorithm is fixed to '{self.fixed_algorithm}'. "
                "Historical trials report rel_score (validation score min-max normalized "
                "within this dataset; 1.0 = best observed, higher is better) and rank "
                "(1 = best). Propose the next configuration to maximize the validation "
                "score. Return only one JSON object containing the configuration. "
                "Do not use markdown."
            )
        else:
            system = (
                "You are a single-algorithm hyperparameter optimization assistant. "
                f"The algorithm is fixed to '{self.fixed_algorithm}'. "
                "Historical trials report rel_score (validation score min-max normalized "
                "within this dataset; 1.0 = best observed, higher is better) and rank "
                "(1 = best). Propose the next configuration to maximize the validation "
                "score. First write a short rationale, then write one JSON object "
                "containing the configuration. Do not use markdown."
            )
        if self.prompt_blocks.get("tuning_playbook_system"):
            system += build_tuning_playbook_system_extension(
                self.fixed_algorithm,
                self.params.get("dataset_profile"),
            )
        duplicate_rule = (
            "Do not return a configuration that duplicates any Selected historical trial. "
            if include_history
            else "No Selected historical trials are provided for this warm-start query. "
        )
        user_parts = [
            self._dataset_text(),
            f"Fixed algorithm: {self.fixed_algorithm}",
            self._search_space_text(),
            history,
        ]
        if include_history and self._tpe_hint_config:
            user_parts.append(format_tpe_hint_block(self._tpe_hint_config))
        if self.response_format == "json_only":
            user_parts.append(
                "Return only this JSON object:\n"
                '{"config": {"<hyperparameter_name>": "<value>"}}\n'
                "Return values for every tunable hyperparameter listed in Search space. "
                f"{duplicate_rule}"
                "Use the exact hyperparameter names shown in Search space. No rationale."
            )
        else:
            user_parts.append(
                "Return exactly this format:\n"
                "Rationale: <short reason for the next configuration>\n\n"
                'JSON:\n{"config": {"<hyperparameter_name>": "<value>"}}\n'
                "Return values for every tunable hyperparameter listed in Search space. "
                f"{duplicate_rule}"
                "Use the exact hyperparameter names shown in Search space."
            )
        user = "\n\n".join(user_parts)
        return system, user

    def _dataset_text(self):
        profile = self.params.get("dataset_profile")
        return "Dataset profile:\n" + _json(profile) if profile else "Dataset profile: not provided."

    def _search_space_text(self):
        rows = []
        for hp in self.config_space.get_hyperparameters():
            name = hp.name
            if name == "algorithm" or isinstance(hp, (Constant, UnParametrizedHyperparameter)):
                continue
            rows.append(f"- {name}: {self._hp_text(hp)}")
        return "Search space:\n" + ("\n".join(rows) if rows else "no tunable hyperparameters")

    def _hp_text(self, hp):
        # Match P5 SFT training Search-space format (range=/q=/log_scale=).
        if isinstance(hp, (Constant, UnParametrizedHyperparameter)):
            return "constant"
        if isinstance(hp, UniformIntegerHyperparameter):
            parts = [f"integer, range=[{int(hp.lower)}, {int(hp.upper)}]"]
            q = getattr(hp, "q", None)
            if q is not None:
                parts.append(f"q={int(q)}")
            if getattr(hp, "log", False):
                parts.append("log_scale=True")
            return ", ".join(parts)
        if isinstance(hp, UniformFloatHyperparameter):
            parts = [f"float, range=[{float(hp.lower)}, {float(hp.upper)}]"]
            q = getattr(hp, "q", None)
            if q is not None:
                parts.append(f"q={float(q)}")
            if getattr(hp, "log", False):
                parts.append("log_scale=True")
            return ", ".join(parts)
        if isinstance(hp, CategoricalHyperparameter):
            return f"categorical, choices={list(hp.choices)!r}"
        if OrdinalHyperparameter and isinstance(hp, OrdinalHyperparameter):
            return f"ordinal, sequence={list(hp.sequence)!r}"
        return f"default={_hp_default(hp)!r}"

    def _update_history(self, obs, source, iter_idx, cfg, rationale):
        is_warm_start = source == "llm_warm_start"
        fallback_text = "LLM warm-start configuration." if is_warm_start else "LLM suggested configuration."
        self.KD.update_history(
            obs,
            source="llm",
            trail_id=f"{iter_idx}_warm_start" if is_warm_start else str(iter_idx),
            detailed_thought=rationale or fallback_text,
            thought_summary=rationale or fallback_text,
            action_config=_cfg_dict(cfg),
            observation_perf=self._raw_observation_perf(obs),
            reflection_summary="No reflection in PT4OptLLMOptimizer.",
        )

    def _append_llm_history(
        self,
        iter_idx,
        status,
        raw,
        payload,
        cfg,
        note,
        error=None,
        attempts=None,
        rejected_duplicate_configs=None,
    ):
        self.llm_history.append(
            {
                "trial": iter_idx,
                "status": status,
                "raw_response": raw,
                "parsed_payload": payload,
                "parsed_config": payload.get("config", payload) if isinstance(payload, dict) else {},
                "normalized_config": _cfg_dict(cfg),
                "rationale": note if status == "PENDING" else "",
                "fallback_reason": note if status != "PENDING" else "",
                "error": error,
                "attempts": attempts or [],
                "rejected_duplicate_configs": rejected_duplicate_configs or [],
            }
        )

    def _finish_llm_history(self, cfg, perf, success, obs):
        if not self.llm_history:
            return
        entry = self.llm_history[-1]
        entry["evaluated_config"] = _cfg_dict(cfg)
        entry["perf"] = perf
        entry["evaluation_status"] = "SUCCESS" if success else "FAILED"
        if entry.get("status") == "PENDING":
            entry["status"] = "SUCCESS" if success else "FAILED"
        if not success:
            entry["error"] = str(getattr(obs, "trial_state", "evaluation_failed"))

    def _dump_conversation(self, iter_idx, prompt_input, raw, payload, cfg, obs, meta, note):
        json_path = os.path.join(self.conversation_log_dir, f"trial_{iter_idx:04d}.json")
        data = {
            "trial": iter_idx,
            "llm_input": prompt_input,
            "llm_output": {
                "raw_response": raw,
                "raw_response_lines": _lines(raw),
                "parsed_payload": payload,
            },
            "raw_response": raw,
            "parsed_payload": payload,
            "normalized_config": _cfg_dict(cfg),
            "llm_attempts": prompt_input.get("attempts", []),
            "objective": obs.objectives[0] if obs.objectives else None,
            "trial_state": getattr(obs, "trial_state", None),
            "history_meta": meta,
            "note": note,
        }
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=True, default=str)
        except IOError as exc:
            self.logger.warning("Failed to write LLM log %s: %s", json_path, exc)

    def _record_event(self, stage, iter_idx, cfg, perf, llm_cost, llm_time, eval_time, iter_time, note):
        if not self.logging_enabled:
            return
        if llm_cost:
            self.llm_query_cost.append(float(llm_cost))
        if llm_time:
            self.llm_query_time.append(float(llm_time))
        row = {
            "ts": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "stage": stage,
            "iter_idx": iter_idx,
            "perf": float(perf),
            "best_so_far": float(max(self.perfs) if self.perfs else perf),
            "algorithm": self.fixed_algorithm,
            "llm_cost": float(llm_cost),
            "cumulative_llm_cost": float(sum(self.llm_query_cost)),
            "llm_time": float(llm_time),
            "eval_time": float(eval_time),
            "iter_time": float(iter_time),
            "wall_elapsed_sec": float(time.time() - self.start_wall_time),
            "note": note,
            "config_for_prompt": _json(_config_for_prompt(cfg, self.fixed_algorithm, self.config_space)),
        }
        self.traj_rows.append(row)
        try:
            exists = os.path.exists(self.traj_csv_path)
            with open(self.traj_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
        except IOError as exc:
            self.logger.warning("Failed to save trajectory to %s: %s", self.traj_csv_path, exc)

    def _get_single_algorithm(self):
        choices = _hp_choices(self.algorithm_hp)
        if len(choices) == 1:
            return choices[0]
        raise ValueError("PT4OptLLMOptimizer only supports single-algorithm HPO.")

    def _get_hp(self, name):
        try:
            return self.config_space[name]
        except Exception:
            return self.config_space.get_hyperparameter(name)

    def _sample_config(self) -> MyConfiguration:
        cfg = self.config_space.sample_configuration()
        if isinstance(cfg, list):
            cfg = cfg[0]
        return cfg if isinstance(cfg, MyConfiguration) else MyConfiguration(cfg)

    def _sample_nonduplicate_config(self, max_attempts: int = 50) -> MyConfiguration:
        cfg = self._sample_config()
        for _ in range(max_attempts):
            if not self._is_duplicate(cfg):
                return cfg
            cfg = self._sample_config()
        return cfg

    def _sync_seen_signatures(self):
        for idx in range(self._seen_config_count, len(self.configs)):
            cfg = self.configs[idx]
            sig = _signature(cfg)
            self.seen_signatures.add(sig)
            if idx < len(self.perfs):
                perf = float(self.perfs[idx])
                if np.isfinite(perf):
                    self.seen_objectives[sig] = -perf
        self._seen_config_count = len(self.configs)

    def _mark_seen(self, cfg, objective=None):
        self._sync_seen_signatures()
        sig = _signature(cfg)
        self.seen_signatures.add(sig)
        if objective is not None and np.isfinite(float(objective)):
            self.seen_objectives[sig] = float(objective)

    def _is_duplicate(self, cfg) -> bool:
        self._sync_seen_signatures()
        return _signature(cfg) in self.seen_signatures

    def _known_objective(self, cfg):
        self._sync_seen_signatures()
        return self.seen_objectives.get(_signature(cfg))

    def _quota_or_time_is_up(self):
        quota = self.evaluation_num_limit is not None and len(self.perfs) >= self.evaluation_num_limit
        timeout = self.time_limit is not None and (time.time() - self.timestamp) > self.time_limit
        return quota or timeout

    def _raw_observation_perf(self, obs):
        raw = float(obs.objectives[0]) if obs.objectives else np.inf
        return raw * (1 if getattr(self.KD, "lower_is_better", False) else -1)

    def _update_incumbent(self):
        if not self.perfs:
            return
        idx = int(np.argmax(self.perfs))
        self.incumbent_config = self.configs[idx].get_dictionary().copy()
        self.incumbent_perf = float(self.perfs[idx])

    def get_opt_trajectory(self):
        return {
            "n_evals": len(self.perfs),
            "final_best": float(max(self.perfs)) if self.perfs else float("-inf"),
            "llm_cost_sum": float(sum(self.llm_query_cost)),
            "llm_calls": int(self.llm_call_count),
            "csv_path": self.traj_csv_path,
            "detail_perfs": ",".join(str(p) for p in self.perfs),
            "history": self.traj_rows[:500],
        }
