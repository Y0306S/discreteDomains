from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from .spaces import ConfigSpaceAdapter


def _loss_key(item: dict[str, Any]) -> float:
    loss = item.get("loss")
    if loss is None:
        return float("inf")
    return float(loss)


def _group_valid_trials(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [trial for trial in trials if trial.get("loss") is not None and not trial.get("failed", False)]


def _round_trials(trials: list[dict[str, Any]], round_size: int) -> list[list[dict[str, Any]]]:
    if round_size <= 0:
        return [trials]
    return [trials[i : i + round_size] for i in range(0, len(trials), round_size)]


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _describe_change(before: dict[str, Any], after: dict[str, Any]) -> dict[str, str]:
    changes = {}
    for key in before.keys():
        if before.get(key) != after.get(key):
            changes[key] = f"{_format_value(before.get(key))} -> {_format_value(after.get(key))}"
    return changes


def summarize_round(
    trials: list[dict[str, Any]],
    adapter: ConfigSpaceAdapter,
    round_index: int,
    round_size: int,
) -> dict[str, Any]:
    valid = _group_valid_trials(trials)
    best_trial = min(valid, key=_loss_key) if valid else None
    top_5 = sorted(valid, key=_loss_key)[:5]
    worst_valid = sorted(valid, key=_loss_key, reverse=True)[:5]
    failed = [trial for trial in trials if trial.get("failed", False) or trial.get("loss") is None]

    cohorts = {"top": [], "mid": [], "bottom": []}
    if valid:
        ordered = sorted(valid, key=_loss_key)
        third = max(1, len(ordered) // 3)
        cohorts["top"] = ordered[:third]
        cohorts["mid"] = ordered[third : 2 * third]
        cohorts["bottom"] = ordered[2 * third :]

    sensitivity_notes = []
    for spec in adapter.numeric_specs():
        if not valid:
            continue
        top_vals = [trial["config"][spec.name] for trial in cohorts["top"]] or [trial["config"][spec.name] for trial in valid]
        bottom_vals = [trial["config"][spec.name] for trial in cohorts["bottom"]] or [trial["config"][spec.name] for trial in valid]
        if not top_vals or not bottom_vals:
            continue
        top_med = statistics.median(top_vals)
        bottom_med = statistics.median(bottom_vals)
        if top_med == bottom_med:
            continue
        sensitivity_notes.append(
            f"{spec.name}: top median {_format_value(top_med)} vs bottom median {_format_value(bottom_med)}"
        )

    good_vs_bad_changes = []
    if best_trial and worst_valid:
        for bad_trial in worst_valid[:3]:
            changes = _describe_change(bad_trial["config"], best_trial["config"])
            if changes:
                good_vs_bad_changes.append(
                    {
                        "from_loss": bad_trial["loss"],
                        "to_loss": best_trial["loss"],
                        "changed": changes,
                    }
                )

    best_so_far = None
    running_best = None
    best_so_far_series = []
    for trial in trials:
        if trial.get("loss") is None:
            continue
        if running_best is None or trial["loss"] < running_best["loss"]:
            running_best = trial
        best_so_far_series.append(
            {
                "step": trial["step"],
                "config": running_best["config"],
                "loss": running_best["loss"],
            }
        )
    if best_so_far_series:
        best_so_far = best_so_far_series[-1]

    losses = [trial["loss"] for trial in valid]
    if len(losses) >= 2:
        slope = losses[-1] - losses[0]
    else:
        slope = 0.0

    unexplored_regions = []
    for spec in adapter.numeric_specs():
        seen = [trial["config"][spec.name] for trial in valid]
        if not seen:
            continue
        low = min(seen)
        high = max(seen)
        span = _format_value(high - low)
        if high < spec.upper:
            unexplored_regions.append(f"{spec.name}: upper side remains open (seen max={_format_value(high)} of {spec.upper})")
        if low > spec.lower:
            unexplored_regions.append(f"{spec.name}: lower side remains open (seen min={_format_value(low)} of {spec.lower})")
        if len(unexplored_regions) >= 5:
            break

    if not valid:
        recommendation = "explore"
        rationale = "No valid trials yet, so continue broad exploration."
    elif slope < -1e-6:
        recommendation = "exploit"
        rationale = "Loss is improving, so stay near the best region and refine locally."
    elif len(unexplored_regions) > len(adapter.numeric_specs()) // 2:
        recommendation = "explore"
        rationale = "Coverage is still narrow, so test uncovered regions."
    else:
        recommendation = "mixed"
        rationale = "There is usable signal, but enough space remains to alternate between refinement and exploration."

    return {
        "round": round_index,
        "steps": [trial["step"] for trial in trials],
        "best_so_far": best_so_far,
        "top_5_configs": [
            {"step": trial["step"], "config": trial["config"], "loss": trial["loss"], "cost": trial.get("cost")}
            for trial in top_5
        ],
        "worst_valid_configs": [
            {"step": trial["step"], "config": trial["config"], "loss": trial["loss"], "cost": trial.get("cost")}
            for trial in worst_valid
        ],
        "failed_or_invalid_configs": [
            {
                "step": trial["step"],
                "config": trial.get("config"),
                "reason": trial.get("error_reason") or trial.get("status") or "invalid",
            }
            for trial in failed
        ],
        "hyperparameter_sensitivity_notes": sensitivity_notes,
        "good_vs_bad_changes": good_vs_bad_changes,
        "learning_curve_signals": [],
        "unexplored_regions": unexplored_regions,
        "exploration_recommendation": recommendation,
        "recommendation_rationale": rationale,
    }


def summarize_trace(
    trace: dict[str, Any],
    adapter: ConfigSpaceAdapter,
    round_size: int = 5,
) -> dict[str, Any]:
    trials = trace.get("trials", [])
    round_blocks = _round_trials(trials, round_size)
    return {
        "task_id": trace["task_id"],
        "benchmark": trace["benchmark"],
        "metric": trace.get("metric", "loss"),
        "round_summaries": [
            summarize_round(block, adapter, idx + 1, round_size)
            for idx, block in enumerate(round_blocks)
        ],
    }


def build_trajectory_prefix(trials: list[dict[str, Any]], index: int) -> list[dict[str, Any]]:
    prefix = []
    for trial in trials[: index + 1]:
        if trial.get("loss") is None:
            continue
        prefix.append(
            {
                "step": trial["step"],
                "config": trial["config"],
                "loss": trial["loss"],
                "cost": trial.get("cost"),
                "source": trial.get("source", "bo"),
            }
        )
    return prefix
