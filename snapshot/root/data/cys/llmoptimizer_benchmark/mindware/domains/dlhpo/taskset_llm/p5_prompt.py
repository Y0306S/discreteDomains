"""P5 prompt for TaskSet CASH: history, per-family recency, explore/exploit blocks.

Layout follows the ML ckpt32 P5 prompt (rel_score/rank, tags, JSON-only reply)
with the TaskSet changes:

* global Top5 / Bottom5 / Recent15 selected trials;
* Recent3 per optimizer family so every family has visible data (fewer if the
  family has fewer successful trials);
* from trial 50 on an LLM turn a deterministic 50/50 coin (same md5 key as the ML
  ``fifty_fifty`` ExplorePolicy) picks
    - explore: per-family exploration counts, or
    - exploit: good/bad regions per family (ML A5 marginal_nonoverlap bins on the
      family's own active parameters, log-scale bins) plus a family-level line.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from mindware.components.optimizers.llm_prompt_blocks import _marginal_prefer_avoid_numeric
from mindware.domains.dlhpo.taskset_llm import spec as S

TOP_K, BOTTOM_K, RECENT_K, FAMILY_RECENT_K = 5, 5, 15, 3
EXPERIENCE_START_TRIAL = 50
EXPLORE_PROB = 0.5
EXPLORE_MODE_TAG = "marginal_nonoverlap"
MIN_REGION_TRIALS = 4

P5_SYSTEM_PROMPT = (
    "You are a CASH hyperparameter optimization assistant for deep-learning training. "
    "Choose an optimizer family and the hyperparameters of that family. Historical trials "
    "report rel_score (validation loss min-max normalized within this task; 1.0 = best "
    "observed (lowest loss), higher is better) and rank (1 = best). Propose the next "
    "configuration to minimize the validation loss. Return only one JSON object containing "
    "the configuration. Do not use markdown."
)

OUTPUT_LINES = [
    "Return only this JSON object:",
    '{"config": {"optimizer_family": "<family>", "<family>:<hyperparameter_name>": "<value>"}}',
    "Return optimizer_family and a value for every parameter of the chosen family. Strictly do "
    "not return a configuration you have previously recommended. Use the exact hyperparameter "
    "names shown in Search space. No rationale.",
]


# -------------------------------------------------------------- observations
def successes_from_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Successful trials with ``score`` = -loss (higher is better), in trial order."""
    return [{"trial": int(r["trial"]), "config": dict(r["config"]), "score": -float(r["loss"]),
             "family": str(r["config"][S.FAMILY_HP])}
            for r in rows if r.get("successful")]


def annotate(successes: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    scores = [s["score"] for s in successes]
    lo, hi = (min(scores), max(scores)) if scores else (0.0, 0.0)
    out = {}
    for s in successes:
        rel = 1.0 if hi == lo else (s["score"] - lo) / (hi - lo)
        out[s["trial"]] = dict(s, rel_score=rel, rank=1 + sum(x > s["score"] for x in scores))
    return out


def _trial_line(item: Dict[str, Any], n: int, tags: Sequence[str]) -> str:
    return (f"trial={item['trial']}, rel_score={item['rel_score']:.4f}, rank={item['rank']}/{n}, "
            f"tags={json.dumps(list(tags))}, config={S.display_config(item['config'])}")


def select_trials(successes: List[Dict[str, Any]]) -> "OrderedDict[int, List[str]]":
    by_score = sorted(successes, key=lambda s: (-s["score"], s["trial"]))
    # Split evenly while the run is short (ML allocates 3/3 at six trials).
    best_k = min(TOP_K, math.ceil(len(successes) / 2))
    worst_k = min(BOTTOM_K, len(successes) - best_k)
    tags: "OrderedDict[int, List[str]]" = OrderedDict()
    for s in by_score[:best_k]:
        tags.setdefault(s["trial"], []).append("best")
    for s in by_score[::-1][:worst_k]:
        tags.setdefault(s["trial"], []).append("worst")
    for s in successes[-RECENT_K:]:
        tags.setdefault(s["trial"], []).append("recent")
    return tags


def history_block(successes: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, Any]]:
    n = len(successes)
    if not n:
        return ["No successful trials yet."], {"selected_trials": []}
    info = annotate(successes)
    tags = select_trials(successes)
    order = sorted(tags, key=lambda t: (-info[t]["rel_score"], t))
    lines = [
        "Previous validation trials.",
        "rel_score in [0,1] is the validation loss min-max normalized within this task (1.0 = best "
        f"observed, higher is better); rank is the 1-based standing among all {n} evaluated trials.",
        "",
        "Selected historical trials:",
    ]
    lines += [f"{i}. " + _trial_line(info[t], n, tags[t]) for i, t in enumerate(order, 1)]
    return lines, {"selected_trials": order, "selected_tags": {str(t): tags[t] for t in order}}


def family_recent_block(successes: List[Dict[str, Any]], families: Sequence[str],
                        shown_above: Sequence[int] = ()) -> Tuple[List[str], Dict[str, Any]]:
    """Recent3 per family; trials already listed above are referenced, not repeated."""
    info = annotate(successes)
    above = set(shown_above)
    n = len(successes)
    lines = [f"Most recent trials per optimizer family (up to {FAMILY_RECENT_K} each):"]
    shown: Dict[str, List[int]] = {}
    for family in families:
        own = [s for s in successes if s["family"] == family]
        recent = own[-FAMILY_RECENT_K:]
        shown[family] = [s["trial"] for s in recent]
        if not recent:
            lines.append(f"- {family}: no successful trials yet")
            continue
        lines.append(f"- {family} ({len(own)} successful trials):")
        for s in recent:
            item = info[s["trial"]]
            if s["trial"] in above:
                lines.append(f"  trial={item['trial']}, rel_score={item['rel_score']:.4f}, "
                             f"rank={item['rank']}/{n}, config shown above")
            else:
                lines.append("  " + _trial_line(item, n, ["family_recent"]))
    return lines, {"family_recent_trials": shown}


def run_summary(rows: Sequence[Dict[str, Any]], successes: List[Dict[str, Any]]) -> List[str]:
    failed = sum(1 for r in rows if not r.get("successful"))
    lines = ["Run summary:", f"- Evaluated {len(successes)} successful trials" + (f" ({failed} failed)." if failed else ".")]
    if successes:
        best = max(successes, key=lambda s: (s["score"], -s["trial"]))
        lines.append(f"- Best so far: trial={best['trial']}, optimizer_family={best['family']}, "
                     f"validation loss={-best['score']:.6g}.")
    return lines


# -------------------------------------------------------- explore / exploit
def explore_branch(seed: int, trial: int, p: float = EXPLORE_PROB) -> str:
    """Byte-identical to the ML ``ExplorePolicy`` fifty_fifty coin."""
    key = f"{int(seed)}:{int(trial)}:{EXPLORE_MODE_TAG}:fifty_fifty".encode("utf-8")
    u = int(hashlib.md5(key).hexdigest()[:8], 16) / float(0xFFFFFFFF)
    return "explore" if u < p else "exploit"


def explore_block(rows: Sequence[Dict[str, Any]], families: Sequence[str]) -> List[str]:
    lines = ["Exploration counts per optimizer family (evaluations so far, including failures):"]
    for family in families:
        tried = [r for r in rows if r["config"].get(S.FAMILY_HP) == family]
        ok = sum(1 for r in tried if r.get("successful"))
        lines.append(f"- {family}: {len(tried)} evaluations ({ok} successful)")
    lines.append("Families with few evaluations are uncertain, not necessarily bad.")
    return lines


def _regions_text(good: Optional[Dict[str, float]], bad: Optional[Dict[str, float]]) -> str:
    parts = [f"{label} [{r['low']:.6g}, {r['high']:.6g}]" for label, r in (("good", good), ("bad", bad)) if r]
    return "; ".join(parts)


def _categorical_regions(own: List[Dict[str, Any]], name: str, choices: Sequence[Any]):
    means = {}
    for choice in choices:
        scores = [s["score"] for s in own if s["config"].get(name) == choice]
        if len(scores) >= 2:
            means[choice] = sum(scores) / len(scores)
    if len(means) < 2:
        return None
    good = max(means, key=means.get)
    bad = min(means, key=means.get)
    return good, bad, means


def family_regions(successes: List[Dict[str, Any]], cs: ConfigurationSpace, family: str,
                   best_config: Optional[Dict[str, Any]] = None) -> List[str]:
    """Good/bad regions on the family's own parameters; a bad region never contains
    the best configuration's value (marginal bins can otherwise contradict it)."""
    own = [s for s in successes if s["family"] == family]
    best_config = best_config or {}
    if len(own) < MIN_REGION_TRIALS:
        return []
    lines = []
    for name in S.family_params(cs, family):
        hp = cs.get_hyperparameter(name)
        if isinstance(hp, CategoricalHyperparameter):
            found = _categorical_regions(own, name, hp.choices)
            if found:
                good, bad, means = found
                if best_config.get(name) == bad:
                    continue
                lines.append(f"  - {name}: good {json.dumps(good)} (mean rel={means[good]:.4g}); "
                             f"bad {json.dumps(bad)} (mean rel={means[bad]:.4g})")
            continue
        _, good, bad = _marginal_prefer_avoid_numeric(
            own, name, spec={"log": bool(getattr(hp, "log", False))}, numeric_log_bins=True)
        value = best_config.get(name)
        if bad and isinstance(value, float) and bad["low"] <= value <= bad["high"]:
            bad = None
        if good or bad:
            lines.append(f"  - {name}: {_regions_text(good, bad)}")
    return lines


def incumbent_family(successes: List[Dict[str, Any]]) -> Optional[str]:
    return max(successes, key=lambda s: (s["score"], -s["trial"]))["family"] if successes else None


def all_family_regions(rel_successes: List[Dict[str, Any]], cs: ConfigurationSpace,
                       protected: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Good/bad regions for the optimizer-family choice and for every family's own parameters.

    Shared by the in-loop exploit block and the experience-bank cards so both describe a
    task the same way. ``protected`` (the incumbent / best config) is never marked bad.
    """
    family_level = None
    fam = _categorical_regions(rel_successes, S.FAMILY_HP, S.families(cs))
    if fam and incumbent_family(rel_successes) != fam[1]:
        good, bad, means = fam
        family_level = (f"- {S.FAMILY_HP}: good {good} (mean rel={means[good]:.4g}); "
                        f"bad {bad} (mean rel={means[bad]:.4g})")
    by_family = {}
    for family in S.families(cs):
        regions = family_regions(rel_successes, cs, family, protected)
        if regions:
            by_family[family] = regions
    return {"family_level": family_level, "by_family": by_family}


def exploit_block(successes: List[Dict[str, Any]], cs: ConfigurationSpace) -> List[str]:
    info = annotate(successes)
    rel = [dict(s, score=info[s["trial"]]["rel_score"]) for s in successes]
    lines = ["Good/Bad regions (from this task's trials so far; associations, not causal laws):"]
    incumbent = max(successes, key=lambda s: (s["score"], -s["trial"]))["config"] if successes else None
    regions = all_family_regions(rel, cs, incumbent)
    if regions["family_level"]:
        lines.append(regions["family_level"])
    for family, family_lines in regions["by_family"].items():
        lines.append(f"- {family}:")
        lines += family_lines
    if len(lines) == 1:
        lines.append("- not enough successful trials per family yet")
    lines.append("Refine inside the good regions; do not copy a past config verbatim if already evaluated.")
    return lines


# ------------------------------------------------------------------- prompt
def space_block(cs: ConfigurationSpace) -> List[str]:
    return ["Search space:", *S.search_space_lines(cs), "", "Search space conditions:",
            *S.condition_lines(cs), "", "Parameter notes:", *S.PARAMETER_NOTES]


def build_p5_prompt(
    task_id: str,
    specs: Dict[str, Dict[str, Any]],
    cs: ConfigurationSpace,
    rows: Sequence[Dict[str, Any]],
    *,
    trial: int,
    seed: int,
    retrieval_lines: Optional[List[str]] = None,
    force_branch: Optional[str] = None,
    prompt_mode: str = "p5plus",
    response_format: str = "json_only",
    protocol_text: Optional[str] = None,
    parameter_notes: Optional[List[str]] = None,
    include_output_instructions: bool = True,
) -> Tuple[str, str, Dict[str, Any]]:
    """Return (system, user, meta) for an in-loop LLM turn at 1-based ``trial``.

    ``force_branch`` ("explore"/"exploit") overrides the coin; GRPO collection uses it.
    """
    if prompt_mode not in {"p5plus", "p5_h0"} or response_format not in {"json_only", "rationale_json"}:
        raise ValueError("Unsupported TaskSet prompt mode or response format")
    if prompt_mode == "p5_h0" and (retrieval_lines or force_branch):
        raise ValueError("P5/H0 does not include P5+ retrieval or explore/exploit blocks")
    successes = successes_from_rows(rows)
    fams = S.families(cs)
    hist, meta = history_block(successes)
    recent, recent_meta = family_recent_block(successes, fams, meta["selected_trials"])
    meta.update(recent_meta)
    lines = ["Task profile:", S.task_text(task_id, specs), "",
             f"Metric: {S.METRIC_TEXT}", f"Training: {protocol_text or S.PROTOCOL_TEXT}", "",
             *space_block(cs), "", *hist, "", *recent, "", *run_summary(rows, successes)]
    if parameter_notes:
        lines += ["", "Additional parameter notes:", *parameter_notes]
    if retrieval_lines:
        lines += ["", *retrieval_lines]
    branch = None
    if prompt_mode == "p5plus" and (force_branch is not None or trial >= EXPERIENCE_START_TRIAL):
        branch = force_branch or explore_branch(seed, trial)
        lines += [""] + (explore_block(rows, fams) if branch == "explore" else exploit_block(successes, cs))
    system = P5_SYSTEM_PROMPT
    if response_format == "rationale_json":
        system = system.replace("Return only one JSON object containing the configuration.",
                                "First write a short rationale, then write one JSON object containing the configuration.")
        output = ["Return exactly this format:", "Rationale: <short reason for the next configuration>",
                  "", "JSON:", OUTPUT_LINES[1],
                  OUTPUT_LINES[2].replace(" No rationale.", "")]
    else:
        output = OUTPUT_LINES
    if include_output_instructions:
        lines += ["", *output]
    meta.update(trial=trial, experience_branch=branch, retrieval_included=bool(retrieval_lines),
                n_successful=len(successes))
    meta.update(prompt_mode=prompt_mode, response_format=response_format)
    return system, "\n".join(lines), meta


def retry_prompt(user_prompt: str, reminders: List[Dict[str, Any]]) -> str:
    """ML ``history_tpe5`` reminder text for duplicate LLM proposals."""
    if not reminders:
        return user_prompt
    lines = [user_prompt, "", "Selected historical trials: temporary recent reminders.",
             "These are previously evaluated configurations, not new evaluations."]
    for index, reminder in enumerate(reminders, 1):
        lines.append(f"Retry reminder {index}: config={S.display_config(reminder['config'])}")
        for result in reminder["previous_evaluations"]:
            if result["successful"]:
                lines.append(f"  trial={result['trial']}, rel_score={result['rel_score']:.4f}, "
                             f"rank={result['rank']}, tags=[\"recent\", \"retry\"]")
            else:
                lines.append(f"  trial={result['trial']}, evaluation_status=FAILED, score=unavailable")
    lines.append("Return one different configuration. Keep the same output format and use every "
                 "parameter of the chosen family listed in Search space.")
    return "\n".join(lines)


def retry_reminder(rows: Sequence[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    successes = successes_from_rows(rows)
    info = annotate(successes)
    key = S.signature(config)
    matches = []
    for r in rows:
        if S.signature(r["config"]) != key:
            continue
        item = info.get(int(r["trial"])) if r.get("successful") else None
        matches.append({"trial": int(r["trial"]), "successful": item is not None,
                        "rel_score": item["rel_score"] if item else None,
                        "rank": item["rank"] if item else None})
    if not matches:
        raise RuntimeError("Duplicate proposal has no recorded evaluation to replay")
    return {"config": dict(config), "previous_evaluations": matches}
