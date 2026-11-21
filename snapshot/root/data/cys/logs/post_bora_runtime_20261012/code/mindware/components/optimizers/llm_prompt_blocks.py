"""Composable deterministic prompt blocks for P5 ablation variants (P17–P25)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)

try:
    from ConfigSpace.hyperparameters import OrdinalHyperparameter as _OrdinalHP
except ImportError:
    _OrdinalHP = ()

PROMPT_BLOCK_KEYS = frozenset(
    {
        "budget_context",
        "explore_refine_summary",
        "best_so_far_trajectory",
        "hp_marginals",
        "source_tags",
        "coverage_map",
        "rationale_diversity_summary",
        "tuning_playbook_system",
        "playbook_run_context",
        "tpe_hint",
        "experience_self",
        "experience_retrieve",
    }
)


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, default=str)


def validate_prompt_blocks(blocks: Optional[Dict[str, bool]]) -> Dict[str, bool]:
    blocks = dict(blocks or {})
    unknown = set(blocks) - PROMPT_BLOCK_KEYS
    if unknown:
        raise ValueError(f"Unknown prompt_blocks keys: {sorted(unknown)}")
    return blocks


def _rel_map(successes: List[dict]) -> Tuple[dict, float, float]:
    scores = [x["score"] for x in successes]
    hi = max(scores) if scores else 0.0
    lo = min(scores) if scores else 0.0
    span = hi - lo

    def rel(score: float) -> float:
        return 1.0 if span <= 0 else (score - lo) / span

    return {int(x["trial"]): rel(x["score"]) for x in successes}, lo, hi


def since_last_improvement(successes: List[dict]) -> int:
    if not successes:
        return 0
    best = max(x["score"] for x in successes)
    count = 0
    for item in reversed(successes):
        if item["score"] >= best - 1e-12:
            break
        count += 1
    return count


def _tunable_hps(config_space, fixed_algorithm: str = "random_forest"):
    rows = []
    for hp in config_space.get_hyperparameters():
        name = hp.name
        if name == "algorithm" or isinstance(hp, (Constant, UnParametrizedHyperparameter)):
            continue
        rows.append(hp)
    return rows


def _hp_short_name(hp_name: str, fixed_algorithm: str) -> str:
    prefix = f"{fixed_algorithm}:"
    return hp_name[len(prefix) :] if hp_name.startswith(prefix) else hp_name


def compute_coverage_stats(successes: List[dict], config_space, fixed_algorithm: str) -> Dict[str, dict]:
    stats = {}
    for hp in _tunable_hps(config_space, fixed_algorithm):
        name = hp.name
        short = _hp_short_name(name, fixed_algorithm)
        if isinstance(hp, CategoricalHyperparameter) or (_OrdinalHP and isinstance(hp, OrdinalHyperparameter)):
            choices = list(hp.choices) if hasattr(hp, "choices") else list(hp.sequence)
            tried = set()
            for item in successes:
                if name in item["config"]:
                    tried.add(str(item["config"][name]))
            untried = [c for c in choices if str(c) not in tried]
            stats[short] = {
                "type": "categorical",
                "tried": len(tried),
                "total": len(choices),
                "pct": 100.0 * len(tried) / len(choices) if choices else 0.0,
                "untried": untried,
            }
        elif isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
            vals = [float(item["config"][name]) for item in successes if name in item["config"]]
            lo_b = float(hp.lower)
            hi_b = float(hp.upper)
            if vals:
                lo_o, hi_o = min(vals), max(vals)
                span = hi_b - lo_b
                pct = 100.0 * (hi_o - lo_o) / span if span > 0 else 0.0
            else:
                lo_o = hi_o = None
                pct = 0.0
            stats[short] = {
                "type": "numeric",
                "observed_lo": lo_o,
                "observed_hi": hi_o,
                "bound_lo": lo_b,
                "bound_hi": hi_b,
                "pct": pct,
            }
    return stats


def mean_coverage_pct(successes: List[dict], config_space, fixed_algorithm: str) -> float:
    cov = compute_coverage_stats(successes, config_space, fixed_algorithm)
    if not cov:
        return 0.0
    return float(np.mean([v["pct"] for v in cov.values()]))


def build_budget_context_block(
    successes: List[dict],
    *,
    budget_total: Optional[int],
    rel_by_trial: Dict[int, float],
    rank_by_trial: Dict[int, int],
) -> str:
    if not successes:
        return "Run budget: no successful trials yet."
    n = len(successes)
    best_item = max(successes, key=lambda x: x["score"])
    stagnation = since_last_improvement(successes)
    remaining = max(0, int(budget_total) - n) if budget_total else None
    budget_line = (
        f"Run budget: {n}/{budget_total} trials used ({remaining} remaining)."
        if budget_total
        else f"Run budget: {n} trials used."
    )
    return "\n".join(
        [
            budget_line,
            f"Trials since last best improvement: {stagnation}.",
            f"Current incumbent rel_score: {rel_by_trial[int(best_item['trial'])]:.4f} "
            f"(rank {rank_by_trial[int(best_item['trial'])]}/{n}).",
        ]
    )


def build_best_so_far_trajectory_block(successes: List[dict], rel_by_trial: Dict[int, float]) -> str:
    if not successes:
        return "Best-so-far trajectory: no successful trials yet."
    lines = ["Best-so-far trajectory (improvement events only):"]
    best_so_far = float("-inf")
    events = []
    for item in successes:
        score = item["score"]
        trial = int(item["trial"])
        rel = rel_by_trial[trial]
        if score > best_so_far + 1e-12:
            delta = rel - (events[-1][2] if events else 0.0)
            events.append((trial, rel, rel if not events else delta))
            best_so_far = score
    events = events[-8:]
    if len(events) < 3:
        lines.append("(Few improvement events; showing recent trial rel_scores for context.)")
        for item in successes[-5:]:
            t = int(item["trial"])
            lines.append(f"  trial {t} → rel={rel_by_trial[t]:.4f}")
    else:
        for i, (trial, rel, delta) in enumerate(events):
            if i == 0:
                lines.append(f"  trial {trial} → rel={rel:.4f}")
            else:
                lines.append(f"  trial {trial} → rel={rel:.4f} ({delta:+.4f})")
    return "\n".join(lines)


def build_explore_refine_summary(successes: List[dict], config_space, fixed_algorithm: str) -> str:
    n = len(successes)
    if n == 0:
        return "Run summary: no successful trials yet."
    stagnation = since_last_improvement(successes)
    coverage = mean_coverage_pct(successes, config_space, fixed_algorithm)
    explore = n < 15 or stagnation >= 8 or coverage < 40.0
    lines = [
        "Run summary:",
        f"- Evaluated {n} successful trials; best validation rel_score=1.0 (rank 1).",
    ]
    if explore:
        lines.append(
            "- Tuning phase: EXPLORE — prioritize under-tested hyperparameter values and diverse configs."
        )
    else:
        lines.append(
            "- Tuning phase: REFINE — make small changes near the incumbent; avoid repeating recent configs."
        )
    return "\n".join(lines)


def build_hp_marginals_block(
    successes: List[dict],
    config_space,
    fixed_algorithm: str,
    rel_by_trial: Dict[int, float],
) -> str:
    if not successes:
        return "Hyperparameter marginals: no successful trials yet."
    hps = _tunable_hps(config_space, fixed_algorithm)
    if len(hps) > 8:
        impacts = []
        for hp in hps:
            name = hp.name
            by_val: Dict[str, List[float]] = {}
            for item in successes:
                if name not in item["config"]:
                    continue
                key = json.dumps(item["config"][name], sort_keys=True, default=str)
                by_val.setdefault(key, []).append(rel_by_trial[int(item["trial"])])
            if len(by_val) < 2:
                continue
            means = [float(np.mean(v)) for v in by_val.values()]
            impacts.append((max(means) - min(means), hp))
        impacts.sort(reverse=True)
        hps = [hp for _, hp in impacts[:6]]

    lines = ["Hyperparameter marginals (mean rel_score per value/bucket):"]
    line_budget = 40
    for hp in hps:
        if len(lines) >= line_budget:
            break
        name = hp.name
        short = _hp_short_name(name, fixed_algorithm)
        lines.append(f"--- {short} ---")
        if isinstance(hp, CategoricalHyperparameter) or (_OrdinalHP and isinstance(hp, OrdinalHyperparameter)):
            choices = list(hp.choices) if hasattr(hp, "choices") else list(hp.sequence)
            by_val: Dict[str, List[float]] = {}
            for item in successes:
                if name not in item["config"]:
                    continue
                key = str(item["config"][name])
                by_val.setdefault(key, []).append(rel_by_trial[int(item["trial"])])
            for choice in choices:
                key = str(choice)
                if key in by_val:
                    lines.append(f"  {choice!r}: n={len(by_val[key])}, mean_rel={float(np.mean(by_val[key])):.4f}")
                else:
                    lines.append(f"  {choice!r}: UNTRIED")
        elif isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
            pts = [(rel_by_trial[int(x["trial"])], float(x["config"][name])) for x in successes if name in x["config"]]
            if len(pts) < 2:
                lines.append("  insufficient data")
                continue
            vals = [p[1] for p in pts]
            rels = [p[0] for p in pts]
            edges = np.quantile(vals, [0.25, 0.5, 0.75]).tolist()
            bounds = [min(vals)] + edges + [max(vals)]
            for i in range(4):
                lo, hi = bounds[i], bounds[i + 1]
                bucket_rels = [r for r, v in zip(rels, vals) if (v >= lo if i == 0 else v > lo) and v <= hi]
                if bucket_rels:
                    lines.append(
                        f"  [{lo:.4g}, {hi:.4g}]: n={len(bucket_rels)}, mean_rel={float(np.mean(bucket_rels)):.4f}"
                    )
    return "\n".join(lines)


def build_coverage_map_block(successes: List[dict], config_space, fixed_algorithm: str) -> str:
    if not successes:
        return "Coverage map: no successful trials yet."
    cov = compute_coverage_stats(successes, config_space, fixed_algorithm)
    lines = ["Coverage map:"]
    for short, info in sorted(cov.items()):
        if info["type"] == "categorical":
            line = f"  {short}: {info['tried']}/{info['total']} values tried ({info['pct']:.0f}%)"
            if info["untried"]:
                line += f"; UNTRIED: {info['untried']}"
            lines.append(line)
        else:
            lo_o = info["observed_lo"]
            hi_o = info["observed_hi"]
            if lo_o is None:
                lines.append(f"  {short}: no values tried yet")
            else:
                lines.append(
                    f"  {short}: observed [{lo_o:.4g}, {hi_o:.4g}] of "
                    f"[{info['bound_lo']:.4g}, {info['bound_hi']:.4g}]"
                )
    return "\n".join(lines)


def build_rationale_diversity_summary(
    successes: List[dict],
    llm_history: List[dict],
    rel_by_trial: Dict[int, float],
) -> str:
    lines = ["Run summary extensions:"]
    if llm_history:
        for entry in reversed(llm_history):
            if entry.get("status") not in ("SUCCESS", "FAILED"):
                continue
            iter_idx = int(entry.get("trial", 0))
            display_trial = iter_idx + 1
            rationale = ""
            payload = entry.get("parsed_payload") or {}
            if isinstance(payload, dict):
                rationale = str(payload.get("rationale", "")).strip()
            if not rationale:
                for attempt in reversed(entry.get("attempts") or []):
                    ap = attempt.get("parsed_payload") or {}
                    if isinstance(ap, dict) and ap.get("rationale"):
                        rationale = str(ap["rationale"]).strip()
                        break
            if not rationale and entry.get("raw_response"):
                raw = str(entry["raw_response"])
                if "Rationale:" in raw:
                    rationale = raw.split("Rationale:", 1)[1].split("JSON:", 1)[0].strip()
            perf = entry.get("perf")
            if perf is not None and successes:
                best = max(x["score"] for x in successes)
                improved = float(perf) >= best - 1e-12
                rel = rel_by_trial.get(display_trial, float("nan"))
                outcome = "improved or matched incumbent" if improved else "did not improve incumbent"
                snippet = rationale[:120] + ("..." if len(rationale) > 120 else "")
                lines.append(f'- Last LLM proposal (trial {display_trial}): rationale="{snippet}"')
                lines.append(f"  → rel_score {rel:.4f}; {outcome}.")
            break
    else:
        lines.append("- Last LLM proposal: none yet.")

    recent = successes[-5:]
    if len(recent) >= 3:
        changed_hps = set()
        all_keys = set()
        for item in recent:
            all_keys.update(item["config"].keys())
        for key in all_keys:
            vals = {json.dumps(item["config"].get(key), sort_keys=True, default=str) for item in recent if key in item["config"]}
            if len(vals) > 1:
                changed_hps.add(key)
        if changed_hps and len(changed_hps) <= 2:
            static = sorted(all_keys - changed_hps)
            lines.append(
                f"- Recent diversity: last {len(recent)} trials changed only {sorted(changed_hps)}."
            )
            if static:
                lines.append(f"  Consider varying: {static[:4]}.")
    return "\n".join(lines)


def build_playbook_run_context_block(
    successes: List[dict],
    *,
    budget_total: Optional[int],
) -> str:
    """Trial position and playbook-aligned action hints (pairs with tuning_playbook_system)."""
    n = len(successes)
    next_trial = n + 1
    stagnation = since_last_improvement(successes)
    budget = int(budget_total) if budget_total else None
    remaining = max(0, budget - n) if budget else None

    if budget:
        pct = 100.0 * n / budget
        run_line = f"Current position: proposing trial {next_trial} of {budget} ({pct:.0f}% budget used"
        if remaining is not None:
            run_line += f", {remaining} remaining"
        run_line += ")."
    else:
        run_line = f"Current position: proposing trial {next_trial} ({n} successful trials so far)."

    if n == 0:
        phase = "LAUNCH"
        actions = [
            "State a causal hypothesis before proposing (playbook #4).",
            "Spread across the search space — try diverse hyperparameter values (playbook #5, #8).",
            "Change few hyperparameters per trial unless deliberately exploring interactions (#1, #9).",
        ]
    elif stagnation >= 8:
        phase = "SATURATION"
        actions = [
            f"No improvement in {stagnation} trials — shift to under-explored hyperparameters (playbook #7).",
            "If best trials sit at a search-space boundary, probe further in that direction (#10).",
            "Avoid configs similar to worst trials (#11); try regularization before capacity (#12).",
        ]
    elif budget and n < max(12, int(0.25 * budget)):
        phase = "EARLY EXPLORE"
        actions = [
            "Prioritize breadth: cover untried categorical values and numeric ranges (playbook #5, #8).",
            "Test interactions between related regularizers, e.g. min_samples_leaf vs min_samples_split (#9).",
            "One narrow goal per trial — avoid changing many hyperparameters at once (#1).",
        ]
    elif budget and n >= int(0.75 * budget):
        phase = "LATE REFINE"
        actions = [
            "Exploit the strong region with small incremental changes near the incumbent (playbook #6).",
            "Only adopt a config if it beats the incumbent validation score (#2).",
            "If plateauing, increase min_samples_* or lower max_features before adding capacity (#12).",
        ]
    else:
        phase = "MID EXPLOIT"
        actions = [
            "Refine around the incumbent with targeted single-hyperparameter moves (playbook #2, #6).",
            "State which hyperparameter should drive improvement and why (#3).",
            f"Trials since last improvement: {stagnation} — if stuck soon, pivot exploration (#7).",
        ]

    lines = [
        "Playbook run context:",
        run_line,
        f"Recommended tuning phase: {phase}.",
        "Suggested actions for this trial:",
    ]
    lines.extend(f"- {a}" for a in actions)
    return "\n".join(lines)


def build_tuning_playbook_system_extension(fixed_algorithm: str, dataset_profile: Optional[dict]) -> str:
    profile = dataset_profile or {}
    n_samples = int(profile.get("n_samples", 0) or 0)
    n_features = int(profile.get("n_features", 0) or 0)
    task_type = str(profile.get("task_type", ""))
    cls_dist = profile.get("class_distribution") or {}
    max_class_frac = 0.0
    if isinstance(cls_dist, dict):
        dist = cls_dist.get("distribution") or cls_dist
        if isinstance(dist, dict) and dist:
            max_class_frac = max(float(v) for v in dist.values())

    lines = [
        "",
        "Tuning principles (adapted from Google Deep Learning Tuning Playbook):",
        "1. One narrow goal per trial — change few hyperparameters unless deliberately exploring.",
        "2. Incremental improvement — only treat a config as better if it beats the incumbent.",
        "3. Avoid accidental wins — state which hyperparameter should drive improvement and why.",
        "4. Launch discipline — adopt a direction only with a causal hypothesis.",
        "5. Exploration = insight — early trials should spread across the search space.",
        "6. Exploitation = refinement — once a strong region is found, adjust nearby.",
        "7. Recognize saturation — shift to under-explored hyperparameters when stuck.",
        "8. All tunables matter — do not fix hyperparameters prematurely.",
        "9. Interactions matter — min_samples_leaf and min_samples_split are related regularizers.",
        "10. Boundary signal — if best trials sit at a search-space edge, probe further in that direction.",
        "11. Avoid infeasible regions — do not copy configs similar to worst trials.",
        "12. Regularize before capacity — increase min_samples_* or lower max_features when plateauing.",
        f"RF guidance for {fixed_algorithm}:",
        "- max_features: capacity/regularization knob; lower helps high-dimensional data.",
        "- min_samples_leaf / min_samples_split: main overfitting controls.",
        "- bootstrap: True adds variance; False fits more tightly.",
    ]
    if task_type == "classification":
        lines.append("- criterion: for imbalanced classes, entropy or log_loss may beat gini.")
    if n_samples and n_samples < 500:
        lines.append(f"- Dataset has n_samples={n_samples}: prefer min_samples_leaf >= 3.")
    if n_features > 50:
        lines.append(f"- Dataset has n_features={n_features}: prioritize exploring max_features.")
    if max_class_frac > 0.7:
        lines.append(f"- Class imbalance detected (max class {max_class_frac:.0%}): criterion choice matters.")
    return "\n".join(lines)


def format_tpe_hint_block(tpe_config: Dict[str, Any]) -> str:
    return "\n".join(
        [
            "TPE sampler suggestion (for reference; you may improve or explore away from it):",
            _json(tpe_config),
            "Do not return this exact configuration if already evaluated.",
        ]
    )


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(float(v))


# Conservative categorical history gate (P5+ is typically injected at trial 40/50).
CAT_MIN_N = 5
CAT_SCORE_DELTA = 0.03
CAT_TOP_FRAC = 0.25
# Numeric blame / factor gap as a fraction of the search-space span.
SPACE_REL_THRESH = 0.15

# Per-variant blame profiles (prompt_variants.json → experience_blame).
DEFAULT_EXPERIENCE_BLAME: Dict[str, Any] = {
    "numeric_log_bins": False,
    "numeric_space_gate": False,
    "cat_history_gate": "off",  # off | conservative | neither
    "cat_min_n": 5,
    "cat_score_delta": 0.03,
    "cat_top_frac": 0.25,
    "numeric_assign": "rank",
}


def parse_experience_blame(raw: Any) -> Dict[str, Any]:
    """Merge variant ``experience_blame`` with defaults (baseline = legacy cat, linear bins)."""
    out = dict(DEFAULT_EXPERIENCE_BLAME)
    if not raw:
        return out
    if not isinstance(raw, dict):
        raise ValueError(f"experience_blame must be a dict, got {type(raw)!r}")
    out.update(raw)
    gate = str(out.get("cat_history_gate") or "off").strip().lower()
    if gate not in ("off", "conservative", "neither"):
        raise ValueError(f"Unknown cat_history_gate {gate!r}")
    out["cat_history_gate"] = gate
    if gate == "neither":
        out.setdefault("cat_min_n", 3)
        out.setdefault("cat_score_delta", 0.02)
    elif gate == "conservative":
        out.setdefault("cat_min_n", CAT_MIN_N)
        out.setdefault("cat_score_delta", CAT_SCORE_DELTA)
    out["numeric_log_bins"] = bool(out.get("numeric_log_bins", False))
    out["numeric_space_gate"] = bool(out.get("numeric_space_gate", False))
    out["cat_min_n"] = int(out.get("cat_min_n", CAT_MIN_N))
    out["cat_score_delta"] = float(out.get("cat_score_delta", CAT_SCORE_DELTA))
    out["cat_top_frac"] = float(out.get("cat_top_frac", CAT_TOP_FRAC))
    return out


def _json_key(v: Any) -> str:
    return json.dumps(v, sort_keys=True, default=str)


def _numeric_hp_specs(config_space) -> Dict[str, Dict[str, Any]]:
    """name -> {lo, hi, log} from ConfigSpace. log follows hp.log, not int vs float."""
    out: Dict[str, Dict[str, Any]] = {}
    if config_space is None:
        return out
    try:
        hps = config_space.get_hyperparameters()
    except Exception:
        return out
    for hp in hps:
        if isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
            out[hp.name] = {
                "lo": float(hp.lower),
                "hi": float(hp.upper),
                "log": bool(getattr(hp, "log", False)),
            }
    return out


def _space_unit(v: float, spec: Optional[Dict[str, Any]]) -> float:
    """Map a numeric HP value onto [0, 1] of the search space (log if flagged)."""
    x = float(v)
    if not spec:
        return x
    lo, hi = float(spec["lo"]), float(spec["hi"])
    if spec.get("log") and x > 0 and lo > 0 and hi > lo:
        den = float(np.log(hi) - np.log(lo)) or 1e-12
        return float((np.log(x) - np.log(lo)) / den)
    if hi > lo:
        return float((x - lo) / (hi - lo))
    return x


def space_rel_diff(a: float, b: float, spec: Optional[Dict[str, Any]]) -> float:
    """|a-b| as a fraction of search-space span. Uses log distance iff spec['log']."""
    if spec:
        return float(min(1.0, abs(_space_unit(a, spec) - _space_unit(b, spec))))
    ga, gb = float(a), float(b)
    scale = max(abs(ga), abs(gb), abs(ga - gb), 1e-12)
    return float(min(1.0, abs(ga - gb) / scale))


def _legacy_categorical_blame(
    successes: List[dict],
    name: str,
    *,
    score_span: float,
) -> Optional[Dict[str, Any]]:
    """Original P5+ cat rule: mean gap vs 5% of score span; always emit avoid."""
    by_cat: Dict[str, List[float]] = {}
    raw_val: Dict[str, Any] = {}
    for item in successes:
        if name not in item["config"]:
            continue
        key = _json_key(item["config"][name])
        by_cat.setdefault(key, []).append(float(item["score"]))
        raw_val[key] = item["config"][name]
    if len(by_cat) < 2:
        return None
    means = {k: float(np.mean(vs)) for k, vs in by_cat.items() if vs}
    if len(means) < 2:
        return None
    best_k = max(means, key=means.get)
    worst_k = min(means, key=means.get)
    delta = max(0.05 * float(score_span), 1e-12)
    if abs(means[best_k] - means[worst_k]) < delta:
        return None
    return {
        "best_val": raw_val[best_k],
        "worst_val": raw_val[worst_k],
        "best_mean": means[best_k],
        "worst_mean": means[worst_k],
        "n_best": len(by_cat[best_k]),
        "n_worst": len(by_cat[worst_k]),
        "emit_avoid": True,
    }


def categorical_history_gate(
    successes: List[dict],
    name: str,
    *,
    min_n: int = CAT_MIN_N,
    score_delta: float = CAT_SCORE_DELTA,
    top_frac: float = CAT_TOP_FRAC,
    avoid_in_top_policy: str = "drop_hp",
) -> Optional[Dict[str, Any]]:
    """Blame a categorical HP only if history is *sure*.

    Keep prefer/avoid only when:
      - both values have n >= min_n trials
      - mean(score|prefer) - mean(score|avoid) >= score_delta
      - prefer value appears in the top ``top_frac`` of trials
      - avoid value does **not** appear there (not an incidental swap), unless
        ``avoid_in_top_policy='prefer_only'`` (cat-neither: keep prefer, drop avoid)
    """
    by_cat: Dict[str, List[float]] = {}
    raw_val: Dict[str, Any] = {}
    for item in successes:
        if name not in item["config"]:
            continue
        key = _json_key(item["config"][name])
        by_cat.setdefault(key, []).append(float(item["score"]))
        raw_val[key] = item["config"][name]
    powered = {k: vs for k, vs in by_cat.items() if len(vs) >= int(min_n)}
    if len(powered) < 2:
        return None
    means = {k: float(np.mean(vs)) for k, vs in powered.items()}
    best_k = max(means, key=means.get)
    worst_k = min(means, key=means.get)
    if best_k == worst_k or (means[best_k] - means[worst_k]) < float(score_delta):
        return None
    ordered = sorted(successes, key=lambda x: float(x["score"]), reverse=True)
    ktop = max(2, int(np.ceil(float(top_frac) * len(ordered))))
    top = ordered[:ktop]

    def _in_top(cat_key: str) -> bool:
        return any(
            name in t["config"] and _json_key(t["config"][name]) == cat_key for t in top
        )

    if not _in_top(best_k):
        return None
    emit_avoid = True
    if _in_top(worst_k):
        if avoid_in_top_policy == "drop_hp":
            return None
        emit_avoid = False
    return {
        "best_val": raw_val[best_k],
        "worst_val": raw_val[worst_k],
        "best_mean": means[best_k],
        "worst_mean": means[worst_k],
        "n_best": len(powered[best_k]),
        "n_worst": len(powered[worst_k]),
        "emit_avoid": emit_avoid,
    }


EXPERIENCE_SELF_MODES = frozenset(
    {
        "cohort",  # A1 / original P5plus (can overlap)
        "cohort_nonoverlap",  # A2
        "marginal_nonoverlap",  # A5
        "marginal_diff",  # A6: prefer=marginal, avoid=nearest-good diffs
    }
)

# Contested prefer∩avoid band: assign to neither (dead zone). Prefer/avoid keep
# only their exclusive parts. Safest for prompts — never say prefer AND avoid.
OVERLAP_POLICY = "neither"


def _interval_overlap(
    a: Optional[Dict[str, float]], b: Optional[Dict[str, float]]
) -> Optional[Tuple[float, float]]:
    if not a or not b:
        return None
    lo = max(float(a["low"]), float(b["low"]))
    hi = min(float(a["high"]), float(b["high"]))
    if lo >= hi - 1e-15:
        return None
    return lo, hi


def _subtract_interval(
    src: Dict[str, float], cut: Tuple[float, float]
) -> List[Dict[str, float]]:
    """Return exclusive pieces of src after removing cut=[c_lo,c_hi]."""
    s_lo, s_hi = float(src["low"]), float(src["high"])
    c_lo, c_hi = cut
    out: List[Dict[str, float]] = []
    if s_lo < c_lo - 1e-15:
        out.append({"low": s_lo, "high": min(s_hi, c_lo)})
    if s_hi > c_hi + 1e-15:
        out.append({"low": max(s_lo, c_hi), "high": s_hi})
    return [p for p in out if p["high"] > p["low"] + 1e-15]


def resolve_prefer_avoid_nonoverlap(
    prefer: Optional[Dict[str, float]],
    avoid: Optional[Dict[str, float]],
    *,
    policy: str = OVERLAP_POLICY,
) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]], str]:
    """Ensure prefer/avoid ranges do not overlap.

    Policy ``neither`` (default): intersection is a dead zone — neither side
    claims it. Each side keeps only exclusive remnants; if a side has multiple
    remnants, keep the widest. If a side becomes empty, that range is dropped
    (direction-only / skip).

    Returns (prefer', avoid', note).
    """
    if policy != "neither":
        raise ValueError(f"Unsupported overlap policy {policy!r}; use 'neither'")
    ov = _interval_overlap(prefer, avoid)
    if ov is None:
        return prefer, avoid, ""
    pref_parts = _subtract_interval(prefer, ov) if prefer else []
    avoid_parts = _subtract_interval(avoid, ov) if avoid else []

    def _widest(parts: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if not parts:
            return None
        return max(parts, key=lambda p: p["high"] - p["low"])

    note = (
        f"overlap[{ov[0]:.4g},{ov[1]:.4g}]→dead_zone(neither); "
        f"prefer/avoid kept exclusive only"
    )
    return _widest(pref_parts), _widest(avoid_parts), note


def _fmt_range(r: Optional[Dict[str, float]]) -> str:
    if not r:
        return ""
    return _json({"low": float(r["low"]), "high": float(r["high"])})


def _race_prefer_avoid_idx(
    scored: List[Tuple[int, float, float, float]],
    bucket: Dict[int, List[float]],
) -> Tuple[set, set]:
    """Prefer/avoid bins whose Hoeffding CIs do not overlap (rel_score in [0,1])."""
    from mindware.components.optimizers.experience_gate import hoeffding_hw

    stats = []
    for i, lo, hi, mean in scored:
        xs = bucket[i]
        hw = hoeffding_hw(len(xs), bound=1.0, delta=0.1)
        stats.append((i, mean - hw, mean + hw, mean))
    if len(stats) < 2:
        return set(), set()
    best = max(stats, key=lambda x: x[3])
    worst = min(stats, key=lambda x: x[3])
    prefer: set = set()
    avoid: set = set()
    if best[1] > worst[2] + 1e-12:
        prefer.add(best[0])
        avoid.add(worst[0])
        for i, lcb, ucb, mean in stats:
            if i == best[0] or i == worst[0]:
                continue
            if lcb > worst[2] + 1e-12 and mean >= best[3] - 1e-12:
                prefer.add(i)
            elif ucb < best[1] - 1e-12 and mean <= worst[3] + 1e-12:
                avoid.add(i)
    return prefer - avoid, avoid - prefer


def _numeric_bins(
    values: List[float], n_bins: int = 5, *, log_scale: bool = False
) -> List[Tuple[float, float]]:
    """Build contiguous bins covering [min,max]; return list of (lo,hi).

    When ``log_scale`` and all values are positive, quantile edges are computed
    in log space then mapped back so learning-rate-style HPs are binned fairly.
    """
    use_log = bool(log_scale) and all(float(v) > 0 for v in values)
    xs = [float(np.log(v)) for v in values] if use_log else [float(v) for v in values]
    lo, hi = float(min(xs)), float(max(xs))
    if hi <= lo + 1e-15:
        a, b = (float(np.exp(lo)), float(np.exp(hi))) if use_log else (lo, hi)
        return [(a, b)]
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = sorted({float(np.quantile(xs, q)) for q in qs})
    if len(edges) < 2:
        a, b = (float(np.exp(lo)), float(np.exp(hi))) if use_log else (lo, hi)
        return [(a, b)]
    bins = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        if b > a + 1e-15:
            if use_log:
                bins.append((float(np.exp(a)), float(np.exp(b))))
            else:
                bins.append((a, b))
    if not bins:
        a, b = (float(np.exp(lo)), float(np.exp(hi))) if use_log else (lo, hi)
        return [(a, b)]
    return bins


def _assign_bin(val: float, bins: List[Tuple[float, float]]) -> int:
    for i, (lo, hi) in enumerate(bins):
        if i == len(bins) - 1:
            if lo - 1e-15 <= val <= hi + 1e-15:
                return i
        elif lo - 1e-15 <= val < hi - 1e-15 or (abs(val - lo) <= 1e-15):
            return i
    # fallback nearest center
    centers = [(lo + hi) / 2 for lo, hi in bins]
    return int(np.argmin([abs(val - c) for c in centers]))


def _marginal_prefer_avoid_numeric(
    successes: List[dict],
    name: str,
    *,
    n_bins: int = 5,
    prefer_frac: float = 0.3,
    avoid_frac: float = 0.3,
    spec: Optional[Dict[str, Any]] = None,
    numeric_log_bins: bool = False,
    numeric_space_gate: bool = False,
    assign_mode: str = "rank",
) -> Tuple[Optional[str], Optional[Dict[str, float]], Optional[Dict[str, float]]]:
    """Prefer/avoid from per-bin mean scores. Each bin is prefer XOR avoid XOR neither."""
    pairs = [
        (float(item["config"][name]), float(item["score"]))
        for item in successes
        if name in item["config"] and _is_number(item["config"][name])
    ]
    if len(pairs) < 4:
        return None, None, None
    assign_mode = str(assign_mode or "rank").strip().lower()
    if assign_mode == "race" and successes:
        all_scores = [float(item["score"]) for item in successes]
        slo, shi = min(all_scores), max(all_scores)
        span = shi - slo
        if span > 1e-12:
            pairs = [(v, (s - slo) / span) for v, s in pairs]
        else:
            pairs = [(v, 0.5) for v, s in pairs]
    vals = [v for v, _ in pairs]
    log_scale = bool(numeric_log_bins and spec and spec.get("log"))
    bins = _numeric_bins(
        vals, n_bins=min(n_bins, max(2, len(set(vals)))), log_scale=log_scale
    )
    if len(bins) < 2:
        return None, None, None
    bucket: Dict[int, List[float]] = {i: [] for i in range(len(bins))}
    for v, s in pairs:
        bucket[_assign_bin(v, bins)].append(s)
    # require at least 2 samples in a bin to score it
    scored = []
    for i, (lo, hi) in enumerate(bins):
        ss = bucket[i]
        if len(ss) < 2:
            continue
        scored.append((i, lo, hi, float(np.mean(ss))))
    if len(scored) < 2:
        return None, None, None
    if assign_mode == "race":
        prefer_idx, avoid_idx = _race_prefer_avoid_idx(scored, bucket)
    else:
        n_pref = max(1, int(np.ceil(prefer_frac * len(scored))))
        n_av = max(1, int(np.ceil(avoid_frac * len(scored))))
        by_mean = sorted(scored, key=lambda x: x[3], reverse=True)
        prefer_idx = {x[0] for x in by_mean[:n_pref]}
        avoid_idx = {x[0] for x in by_mean[-n_av:]}
        contested = prefer_idx & avoid_idx
        prefer_idx -= contested
        avoid_idx -= contested
    if not prefer_idx and not avoid_idx:
        return None, None, None

    def _envelope(idxs: set) -> Optional[Dict[str, float]]:
        if not idxs:
            return None
        los = [bins[i][0] for i in idxs]
        his = [bins[i][1] for i in idxs]
        return {"low": float(min(los)), "high": float(max(his))}

    prefer = _envelope(prefer_idx)
    avoid = _envelope(avoid_idx)
    # If envelope re-introduces overlap (non-contiguous bins), resolve with neither
    prefer, avoid, _ = resolve_prefer_avoid_nonoverlap(prefer, avoid, policy=OVERLAP_POLICY)

    if numeric_space_gate:
        # prefer vs avoid must differ by ≥15% of the search-space span.
        if prefer and avoid:
            p_c = 0.5 * (float(prefer["low"]) + float(prefer["high"]))
            a_c = 0.5 * (float(avoid["low"]) + float(avoid["high"]))
            if space_rel_diff(p_c, a_c, spec) < SPACE_REL_THRESH:
                return None, None, None
        elif prefer:
            if space_rel_diff(float(prefer["low"]), float(prefer["high"]), spec) < SPACE_REL_THRESH:
                prefer = None
        elif avoid:
            if space_rel_diff(float(avoid["low"]), float(avoid["high"]), spec) < SPACE_REL_THRESH:
                avoid = None
        if prefer is None and avoid is None:
            return None, None, None

    pref_mean = float(np.mean([m for i, *_, m in scored if i in prefer_idx])) if prefer_idx else None
    av_mean = float(np.mean([m for i, *_, m in scored if i in avoid_idx])) if avoid_idx else None
    direction = None
    if prefer and avoid and prefer["high"] + prefer["low"] != avoid["high"] + avoid["low"]:
        p_c = 0.5 * (prefer["low"] + prefer["high"])
        a_c = 0.5 * (avoid["low"] + avoid["high"])
        if p_c > a_c:
            direction = (
                f"higher {name} has better marginal score "
                f"(prefer-bin mean={pref_mean:.4g}, avoid-bin mean={av_mean:.4g})"
            )
        else:
            direction = (
                f"lower {name} has better marginal score "
                f"(prefer-bin mean={pref_mean:.4g}, avoid-bin mean={av_mean:.4g})"
            )
    elif prefer and pref_mean is not None:
        direction = f"{name} prefer-bin mean score={pref_mean:.4g}"
    return direction, prefer, avoid


def _nearest_good_diff_avoids(
    successes: List[dict],
    name: str,
    *,
    top_fraction: float = 0.2,
    worst_fraction: float = 0.2,
    rel_diff: float = SPACE_REL_THRESH,
    specs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[Dict[str, float]]:
    """Avoid range from HPs that differ vs nearest good neighbor (innocent HPs skipped).

    Numeric distance is a fraction of the search-space span (log if ``hp.log``).
    """
    specs = specs or {}
    spec = specs.get(name)
    ordered = sorted(successes, key=lambda x: float(x["score"]), reverse=True)
    n = len(ordered)
    k_top = max(2, int(np.ceil(top_fraction * n)))
    k_bad = max(2, int(np.ceil(worst_fraction * n)))
    goods = ordered[:k_top]
    bads = ordered[-k_bad:]
    blamed: List[float] = []

    def _num_cfg(item):
        return {
            k: float(v)
            for k, v in (item.get("config") or {}).items()
            if _is_number(v)
        }

    good_cfgs = [_num_cfg(g) for g in goods]
    if not good_cfgs or name not in good_cfgs[0]:
        good_cfgs = [c for c in good_cfgs if name in c]
    if not good_cfgs:
        return None

    all_vals = [
        float(x["config"][name])
        for x in successes
        if name in x["config"] and _is_number(x["config"][name])
    ]
    if len(all_vals) < 2:
        return None
    obs_span = max(all_vals) - min(all_vals)

    def _dist(bcfg, gc):
        keys = set(bcfg) & set(gc)
        if not keys:
            return 1e18
        diffs = [
            _space_unit(bcfg[k], specs.get(k)) - _space_unit(gc[k], specs.get(k))
            for k in keys
        ]
        return float(np.sqrt(np.mean(np.square(diffs))))

    for bad in bads:
        if name not in bad["config"] or not _is_number(bad["config"][name]):
            continue
        bcfg = _num_cfg(bad)
        if name not in bcfg:
            continue
        gcfg = min(good_cfgs, key=lambda gc: _dist(bcfg, gc))
        if name not in gcfg:
            continue
        if space_rel_diff(bcfg[name], gcfg[name], spec) >= rel_diff:
            blamed.append(bcfg[name])

    if len(blamed) < 2:
        if len(blamed) == 1:
            v = blamed[0]
            pad = max(0.05 * (obs_span if obs_span > 0 else 1.0), 1e-9)
            return {"low": v - pad, "high": v + pad}
        return None
    return {"low": float(min(blamed)), "high": float(max(blamed))}


def build_experience_self_block(
    successes: List[dict],
    *,
    min_trials: int = 8,
    top_fraction: float = 0.2,
    mode: str = "cohort",
    overlap_policy: str = OVERLAP_POLICY,
    config_space=None,
    experience_blame: Any = None,
) -> str:
    """P5+: structured experience fields from *this run's* trajectory only.

    Modes
    -----
    - ``cohort``: original top/worst 20% min-max (may overlap) — baseline A1.
    - ``cohort_nonoverlap``: same cohorts, then ``neither`` dead-zone on overlap — A2.
    - ``marginal_nonoverlap``: per-HP bin mean scores; bins exclusive; neither on
      envelope overlap — A5.
    - ``marginal_diff``: prefer from marginal bins; avoid from nearest-good diffs;
      then neither non-overlap — A6.

    Overlap policy ``neither``: prefer∩avoid is claimed by neither side.
    """
    mode = mode or "cohort"
    if mode not in EXPERIENCE_SELF_MODES:
        raise ValueError(f"Unknown experience_self mode {mode!r}; expected one of {sorted(EXPERIENCE_SELF_MODES)}")

    specs = _numeric_hp_specs(config_space)
    blame = parse_experience_blame(experience_blame)

    n = len(successes)
    if n < int(min_trials):
        return ""

    scores = [float(x["score"]) for x in successes]
    score_lo, score_hi = min(scores), max(scores)
    span = abs(score_hi - score_lo)
    delta = max(0.05 * span, 1e-12)

    # --- dynamics (incumbent curve) ---
    best = scores[0]
    incumbent_trials = [int(successes[0]["trial"])]
    last_improve_idx = 0
    for i, item in enumerate(successes):
        s = float(item["score"])
        if s > best + 1e-12:
            best = s
            incumbent_trials.append(int(item["trial"]))
            last_improve_idx = i
    stagnation = n - 1 - last_improve_idx
    incumbent_updates = len(incumbent_trials)

    window = successes[-5:]
    lo_w, hi_w = score_lo, score_hi

    def _rel(s: float) -> float:
        return 1.0 if hi_w <= lo_w else (s - lo_w) / (hi_w - lo_w)

    if n < 2:
        trend = "INSUFFICIENT_DATA"
    else:
        rels = [_rel(float(x["score"])) for x in window]
        change = rels[-1] - rels[-2]
        if len(rels) >= 3 and float(np.std(rels)) < 0.02:
            trend = "STAGNANT"
        elif change > 0.02:
            trend = "IMPROVING"
        elif change < -0.02:
            trend = "DECLINING"
        else:
            trend = "FLAT"
        if stagnation >= max(5, int(0.15 * n)) and trend != "IMPROVING":
            trend = "STAGNANT"

    warm_end = max(1, int(np.ceil(0.1 * n)))
    phase_note = (
        f"warm 1–{warm_end}; progress with {incumbent_updates} incumbent updates "
        f"until trial {incumbent_trials[-1]}; plateau last {stagnation} trials"
    )

    ordered = sorted(successes, key=lambda x: float(x["score"]), reverse=True)
    k = max(3, int(np.ceil(top_fraction * n)))
    k = min(k, max(1, n // 2))
    top = ordered[:k]
    top_ids = {int(x["trial"]) for x in top}
    worst = [x for x in ordered[-k:] if int(x["trial"]) not in top_ids] or ordered[-k:]

    keys = sorted({kk for item in successes for kk in (item.get("config") or {})})
    factor_lines: List[str] = []
    avoid_lines: List[str] = []
    baseline = float(successes[0]["score"])
    floor = score_lo + 0.2 * span
    enforce_nonoverlap = mode != "cohort"
    factor_label = {
        "cohort": "factors (top-vs-worst cohorts)",
        "cohort_nonoverlap": "factors (cohort + non-overlap/neither)",
        "marginal_nonoverlap": "factors (marginal bins + non-overlap/neither)",
        "marginal_diff": "factors (marginal prefer + nearest-good-diff avoid)",
    }[mode]

    for name in keys:
        vals_all = [item["config"][name] for item in successes if name in item["config"]]
        unique = {json.dumps(v, sort_keys=True, default=str) for v in vals_all}
        if len(unique) < 2:
            continue

        if all(_is_number(v) for v in vals_all):
            direction: Optional[str] = None
            good: Optional[Dict[str, float]] = None
            bad: Optional[Dict[str, float]] = None

            if mode in ("cohort", "cohort_nonoverlap"):
                top_vals = [float(item["config"][name]) for item in top if name in item["config"]]
                worst_vals = [float(item["config"][name]) for item in worst if name in item["config"]]
                if not top_vals or not worst_vals:
                    continue
                top_score_mean = float(np.mean([float(x["score"]) for x in top]))
                worst_score_mean = float(np.mean([float(x["score"]) for x in worst]))
                if abs(top_score_mean - worst_score_mean) < delta:
                    continue
                top_mean = float(np.mean(top_vals))
                worst_mean = float(np.mean(worst_vals))
                spec = specs.get(name)
                if blame["numeric_space_gate"] and space_rel_diff(
                    top_mean, worst_mean, spec
                ) < SPACE_REL_THRESH:
                    continue
                if top_mean > worst_mean:
                    direction = (
                        f"higher {name} over-represented in top trials "
                        f"(top mean={top_mean:.4g}, worst mean={worst_mean:.4g})"
                    )
                else:
                    direction = (
                        f"lower {name} over-represented in top trials "
                        f"(top mean={top_mean:.4g}, worst mean={worst_mean:.4g})"
                    )
                good = {"low": float(min(top_vals)), "high": float(max(top_vals))}
                bad = {"low": float(min(worst_vals)), "high": float(max(worst_vals))}
                emit_avoid = worst_score_mean <= baseline or worst_score_mean <= floor
            elif mode == "marginal_nonoverlap":
                direction, good, bad = _marginal_prefer_avoid_numeric(
                    successes,
                    name,
                    spec=specs.get(name),
                    numeric_log_bins=blame["numeric_log_bins"],
                    numeric_space_gate=blame["numeric_space_gate"],
                    assign_mode=str(blame.get("numeric_assign") or "rank"),
                )
                emit_avoid = bad is not None
                if direction is None and good is None and bad is None:
                    continue
                spec = specs.get(name)
                if blame["numeric_space_gate"] and good and bad and space_rel_diff(
                    0.5 * (good["low"] + good["high"]),
                    0.5 * (bad["low"] + bad["high"]),
                    spec,
                ) < SPACE_REL_THRESH:
                    continue
            else:  # marginal_diff
                direction, good, _marg_bad = _marginal_prefer_avoid_numeric(
                    successes,
                    name,
                    spec=specs.get(name),
                    numeric_log_bins=blame["numeric_log_bins"],
                    numeric_space_gate=blame["numeric_space_gate"],
                    assign_mode=str(blame.get("numeric_assign") or "rank"),
                )
                bad = _nearest_good_diff_avoids(
                    successes, name, top_fraction=top_fraction, specs=specs
                )
                emit_avoid = bad is not None
                if direction is None and good is None and bad is None:
                    continue

            note = ""
            if enforce_nonoverlap:
                good, bad, note = resolve_prefer_avoid_nonoverlap(
                    good, bad, policy=overlap_policy
                )
            else:
                # cohort legacy may keep overlapping ranges
                pass

            bits: List[str] = []
            if direction:
                bits.append(direction)
            if good:
                bits.append(f"prefer ~{_fmt_range(good)}")
            if bad and emit_avoid:
                bits.append(f"avoid ~{_fmt_range(bad)}")
            if note:
                bits.append(f"[{note}]")
            if not bits:
                continue
            factor_lines.append(f"- {name}: " + "; ".join(bits))
            if bad and emit_avoid:
                avoid_lines.append(f"- {name} in {_fmt_range(bad)}")
        else:
            cat_gate = blame["cat_history_gate"]
            if cat_gate == "off":
                gated = _legacy_categorical_blame(successes, name, score_span=span)
            else:
                avoid_policy = "prefer_only" if cat_gate == "neither" else "drop_hp"
                gated = categorical_history_gate(
                    successes,
                    name,
                    min_n=blame["cat_min_n"],
                    score_delta=blame["cat_score_delta"],
                    top_frac=blame["cat_top_frac"],
                    avoid_in_top_policy=avoid_policy,
                )
            if gated is None:
                continue
            best_val = gated["best_val"]
            worst_val = gated["worst_val"]
            factor_lines.append(
                f"- {name}: prefer {best_val!r} over {worst_val!r} "
                f"(mean scores {gated['best_mean']:.4g} vs {gated['worst_mean']:.4g}; "
                f"n={gated['n_best']}/{gated['n_worst']})"
            )
            if gated.get("emit_avoid", True):
                avoid_lines.append(f"- {name}={worst_val!r}")

    lines = [
        "Self-run experience (from this task's trials so far; associations, not causal laws):",
        f"- mode={mode}, overlap_policy={overlap_policy if enforce_nonoverlap else 'allow'}",
        f"- dynamics: trend={trend}, incumbent_updates={incumbent_updates}, "
        f"stagnation={stagnation}, curve: {phase_note}",
        f"- baseline_kind=first_success, baseline_score={baseline:.6g}; "
        f"best_so_far={max(scores):.6g}; "
        f"normalized_gain_vs_first="
        f"{(0.0 if span <= 0 else max(0.0, min(1.0, (max(scores) - baseline) / max(score_hi - baseline, 1e-12)))):.3f}",
    ]
    if factor_lines:
        lines.append(f"- {factor_label}:")
        lines.extend(factor_lines[:6])
    else:
        lines.append("- factors: no clear separation yet.")
    if avoid_lines:
        lines.append("- avoid:")
        lines.extend(avoid_lines[:3])
    else:
        lines.append("- avoid: none above threshold.")
    lines.append(
        "Use these signals with the selected historical trials above; "
        "do not copy a past config verbatim if already evaluated."
    )
    return "\n".join(lines)
