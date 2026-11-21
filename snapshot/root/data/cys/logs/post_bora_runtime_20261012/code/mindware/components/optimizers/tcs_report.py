"""Paper-faithful Trajectory Context Summarizer (TCS) for trial-level HPO.

Adapted from LLM-TCS-HPT (arXiv:2509.15561) for single-shot RF evaluations:
deterministic TCS block + optional Analysis Agent prompt (6-part diagnosis).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import numpy as np
from ConfigSpace.hyperparameters import Constant, UnParametrizedHyperparameter


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, default=str)


def _cfg_dict(cfg) -> Dict[str, Any]:
    return cfg.get_dictionary().copy() if hasattr(cfg, "get_dictionary") else dict(cfg)


def _config_for_prompt(cfg, fixed_algorithm: str, config_space) -> Dict[str, Any]:
    cfg_dict = _cfg_dict(cfg) if not isinstance(cfg, dict) else cfg
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


def _collect_successes(configs, perfs, fixed_algorithm, config_space) -> List[dict]:
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
    return successes


def _rel_scores(successes: List[dict]) -> Tuple[dict, float, float]:
    scores = [x["score"] for x in successes]
    hi = max(scores) if scores else 0.0
    lo = min(scores) if scores else 0.0
    span = hi - lo

    def rel(score: float) -> float:
        return 1.0 if span <= 0 else (score - lo) / span

    return {int(x["trial"]): rel(x["score"]) for x in successes}, lo, hi


def _rank_by_trial(successes: List[dict]) -> Dict[int, int]:
    all_scores = [x["score"] for x in successes]
    return {int(x["trial"]): 1 + sum(1 for s in all_scores if s > x["score"]) for x in successes}


def _trend_status(successes: List[dict], window: int = 5) -> Tuple[str, float]:
    if len(successes) < 2:
        return "INSUFFICIENT_DATA", 0.0
    recent = successes[-window:]
    rel_map, _, _ = _rel_scores(successes)
    rel_vals = [rel_map[int(x["trial"])] for x in recent]
    change = rel_vals[-1] - rel_vals[-2]
    if len(rel_vals) >= 3:
        std = float(np.std(rel_vals))
        if std < 0.02:
            status = "STAGNANT"
        elif change > 0.02:
            status = "IMPROVING"
        elif change < -0.02:
            status = "DECLINING"
        else:
            status = "FLAT"
    else:
        status = "IMPROVING" if change > 0 else ("DECLINING" if change < 0 else "FLAT")
    return status, change


def _stagnation_trials(successes: List[dict]) -> int:
    if not successes:
        return 0
    best = max(x["score"] for x in successes)
    count = 0
    for item in reversed(successes):
        if item["score"] >= best - 1e-12:
            break
        count += 1
    return count


def _hyperparameter_impact(successes: List[dict], hp_names: List[str]) -> Dict[str, dict]:
    impacts: Dict[str, dict] = {}
    for hp in hp_names:
        by_value: Dict[str, List[float]] = {}
        for item in successes:
            if hp not in item["config"]:
                continue
            key = json.dumps(item["config"][hp], sort_keys=True, default=str)
            by_value.setdefault(key, []).append(float(item["score"]))
        if not by_value:
            impacts[hp] = {"status": "NEVER_TESTED", "unique_values": 0}
            continue
        stats = []
        for val_key, scores in by_value.items():
            stats.append((float(np.mean(scores)), val_key, len(scores)))
        stats.sort(reverse=True)
        best_mean, best_val, best_n = stats[0]
        worst_mean, worst_val, _ = stats[-1]
        impacts[hp] = {
            "status": "TESTED",
            "unique_values": len(by_value),
            "best_value": json.loads(best_val),
            "best_mean_score": best_mean,
            "worst_value": json.loads(worst_val),
            "worst_mean_score": worst_mean,
            "current_value": successes[-1]["config"].get(hp),
            "impact_range": best_mean - worst_mean,
        }
    return impacts


def _hp_names_from_space(config_space) -> List[str]:
    names = []
    for hp in config_space.get_hyperparameters():
        if hp.name == "algorithm" or isinstance(hp, (Constant, UnParametrizedHyperparameter)):
            continue
        names.append(hp.name)
    return names


def build_tcs_report(
    configs,
    perfs,
    fixed_algorithm: str,
    config_space,
    *,
    metric_name: str = "validation score",
    higher_is_better: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """Build deterministic Trajectory Context Summary from trial history."""
    successes = _collect_successes(configs, perfs, fixed_algorithm, config_space)
    if not successes:
        return "TCS: no successful trials yet.", {"n_success": 0, "tcs_sections": []}

    rel_map, lo_score, hi_score = _rel_scores(successes)
    ranks = _rank_by_trial(successes)
    n = len(successes)
    best_item = max(successes, key=lambda x: x["score"])
    latest = successes[-1]
    trend, trend_change = _trend_status(successes)
    stagnation = _stagnation_trials(successes)
    hp_names = _hp_names_from_space(config_space)
    impacts = _hyperparameter_impact(successes, hp_names)

    best_count, worst_count = _allocate_best_worst_counts(n, 5, 5)
    top = sorted(successes, key=lambda x: x["score"], reverse=True)[:best_count]
    top_trials = {int(x["trial"]) for x in top}
    worst = []
    for item in sorted(successes, key=lambda x: x["score"]):
        if int(item["trial"]) in top_trials:
            continue
        worst.append(item)
        if len(worst) >= worst_count:
            break

    lines = [
        "=== TRAJECTORY CONTEXT SUMMARY (TCS) ===",
        "",
        "=== CURRENT SITUATION ===",
        f"Metric: {metric_name} ({'higher is better' if higher_is_better else 'lower is better'})",
        f"Trials evaluated: {n}",
        f"Best validation score: {best_item['score']:.6g} (trial {best_item['trial']}, rel_score={rel_map[int(best_item['trial'])]:.4f}, rank 1/{n})",
        f"Latest validation score: {latest['score']:.6g} (trial {latest['trial']}, rel_score={rel_map[int(latest['trial'])]:.4f}, rank {ranks[int(latest['trial'])]}/{n})",
        f"Score range observed: [{lo_score:.6g}, {hi_score:.6g}]",
        f"Recent trend (rel_score): {trend} (last-step change {trend_change:+.4f})",
        f"Trials since last best improvement: {stagnation}",
        "",
        f"=== LATEST EXPERIMENT (trial {latest['trial']}) ===",
        f"Config: {_json(latest['config'])}",
        f"rel_score: {rel_map[int(latest['trial'])]:.4f}, rank: {ranks[int(latest['trial'])]}/{n}",
    ]

    if len(successes) >= 2:
        prev = successes[-2]
        lines += [
            "",
            "=== PREVIOUS EXPERIMENT COMPARISON ===",
            f"Previous trial {prev['trial']}: rel_score={rel_map[int(prev['trial'])]:.4f}",
            f"Score change: {latest['score'] - prev['score']:+.6g}",
            "Hyperparameter changes:",
        ]
        for hp in hp_names:
            prev_val = prev["config"].get(hp, "NOT_SET")
            curr_val = latest["config"].get(hp, "NOT_SET")
            marker = " (changed)" if prev_val != curr_val else " (unchanged)"
            lines.append(f"  - {hp}: {prev_val} -> {curr_val}{marker}")

    lines += ["", "=== DETAILED HYPERPARAMETER ANALYSIS ==="]
    for hp in hp_names:
        info = impacts[hp]
        lines.append(f"--- {hp} ---")
        if info.get("status") == "NEVER_TESTED":
            lines.append("Status: NEVER TESTED - HIGH PRIORITY FOR EXPLORATION")
            continue
        lines.append(f"Unique values tested: {info['unique_values']}")
        lines.append(
            f"Best mean score at {info['best_value']!r} (mean={info['best_mean_score']:.6g}); "
            f"worst mean at {info['worst_value']!r} (mean={info['worst_mean_score']:.6g})"
        )
        lines.append(f"Current value: {info['current_value']!r}")
        lines.append(f"Impact range (mean score): {info['impact_range']:.6g}")

    lines += ["", "=== TOP TRIALS (reference) ==="]
    for i, item in enumerate(top, start=1):
        lines.append(
            f"{i}. trial={item['trial']}, rel_score={rel_map[int(item['trial'])]:.4f}, config={_json(item['config'])}"
        )
    if worst:
        lines += ["", "=== WORST TRIALS (reference) ==="]
        for i, item in enumerate(worst, start=1):
            lines.append(
                f"{i}. trial={item['trial']}, rel_score={rel_map[int(item['trial'])]:.4f}, config={_json(item['config'])}"
            )

    lines += [
        "",
        "=== EVALUATED CONFIGURATIONS (do not repeat) ===",
    ]
    for item in successes:
        lines.append(f"trial {item['trial']}: {_json(item['config'])}")

    meta = {
        "n_success": n,
        "tcs_sections": [
            "current_situation",
            "latest_experiment",
            "previous_comparison",
            "hyperparameter_analysis",
            "top_worst_reference",
            "evaluated_configs",
        ],
        "best_trial": int(best_item["trial"]),
        "latest_trial": int(latest["trial"]),
        "trend": trend,
        "stagnation_trials": stagnation,
        "score_normalization": {"lo": lo_score, "hi": hi_score, "n_success": n},
    }
    return "\n".join(lines), meta


def build_analysis_agent_prompts(
    tcs_report: str,
    *,
    metric_name: str = "validation score",
    higher_is_better: bool = True,
    hp_names: List[str] | None = None,
) -> Tuple[str, str]:
    """System + user prompts for the Analysis Agent (paper 6-part format)."""
    hp_list = ", ".join(hp_names) if hp_names else "all tunable hyperparameters"
    system = (
        "You are an expert ML hyperparameter optimization analyst. "
        f"Your goal is to improve {metric_name} ({'higher is better' if higher_is_better else 'lower is better'}). "
        f"You must consider ALL available hyperparameters: {hp_list}. "
        "Read the Trajectory Context Summary (TCS) and provide strategic guidance "
        "for the next trial. Do not propose a full configuration yet."
    )
    user = "\n\n".join(
        [
            tcs_report,
            "",
            "ANALYSIS TASK:",
            f"Analyze this tuning situation and recommend how to improve {metric_name} on the next trial.",
            "",
            "REQUIRED ANALYSIS FORMAT (use these exact section headers):",
            "1. PROBLEM DIAGNOSIS: What is preventing further improvement?",
            "2. HYPERPARAMETER IMPACT ASSESSMENT: How would changing each hyperparameter help or hurt?",
            "3. PRIMARY ACTION: Which hyperparameter to change first and why?",
            "4. SPECIFIC RECOMMENDATION: Exact value or direction for the primary change.",
            "5. REASONING: Why this change should beat alternatives.",
            "6. EXPECTED OUTCOME: What rel_score improvement do you expect?",
        ]
    )
    return system, user


def build_tcs_single_agent_prompts(
    tcs_report: str,
    *,
    fixed_algorithm: str,
    dataset_text: str,
    search_space_text: str,
    metric_name: str = "validation score",
    higher_is_better: bool = True,
    hp_names: List[str] | None = None,
) -> Tuple[str, str]:
    """Single-call TCS agent: 6-part analysis + rationale + JSON config in one response."""
    hp_list = ", ".join(hp_names) if hp_names else "all tunable hyperparameters"
    system = (
        "You are an expert single-algorithm hyperparameter optimization agent. "
        f"The algorithm is fixed to '{fixed_algorithm}'. "
        f"Your goal is to improve {metric_name} ({'higher is better' if higher_is_better else 'lower is better'}). "
        f"You must consider ALL tunable hyperparameters: {hp_list}. "
        "Read the Trajectory Context Summary (TCS). rel_score is min-max normalized within this dataset "
        "(1.0 = best observed, higher is better). "
        "First complete the required 6-part analysis, then propose the next configuration. "
        "Do not use markdown."
    )
    user = "\n\n".join(
        [
            dataset_text,
            f"Fixed algorithm: {fixed_algorithm}",
            search_space_text,
            tcs_report,
            "",
            "TASK:",
            f"Analyze this tuning situation and propose the next configuration to improve {metric_name}.",
            "",
            "REQUIRED OUTPUT FORMAT (use these exact section headers in order):",
            "1. PROBLEM DIAGNOSIS: What is preventing further improvement?",
            "2. HYPERPARAMETER IMPACT ASSESSMENT: How would changing each hyperparameter help or hurt?",
            "3. PRIMARY ACTION: Which hyperparameter to change first and why?",
            "4. SPECIFIC RECOMMENDATION: Exact value or direction for the primary change.",
            "5. REASONING: Why this change should beat alternatives.",
            "6. EXPECTED OUTCOME: What rel_score improvement do you expect?",
            "",
            "Rationale: <one short sentence summarizing the next configuration>",
            "",
            'JSON:',
            '{"config": {"<hyperparameter_name>": "<value>"}}',
            "",
            "Return values for every tunable hyperparameter listed in Search space. "
            "Do not return a configuration listed under EVALUATED CONFIGURATIONS in the TCS. "
            "Use the exact hyperparameter names shown in Search space.",
        ]
    )
    return system, user
