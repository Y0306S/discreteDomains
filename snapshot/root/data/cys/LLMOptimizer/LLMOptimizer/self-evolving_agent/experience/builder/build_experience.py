"""Deterministic Experience v2 builder from BO / HPO trial histories.

Locked rules:
- incumbent_updates = improvement-event count (no separate field)
- phase_note optional, curve template only (not optimizer modes)
- no knowledge.summary — render embed_text / prompt_block from fields
- factor good/bad via top-vs-worst cohort association + δ threshold
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SKIP_KEYS = {"algorithm"}

ROLE_HINTS = (
    ("learning_rate", "learning_rate"),
    ("lr", "learning_rate"),
    ("l2", "regularization"),
    ("reg_lambda", "regularization"),
    ("reg_alpha", "regularization"),
    ("reg", "regularization"),
    ("alpha", "regularization"),
    ("C", "regularization"),
    ("min_child_weight", "regularization"),
    ("num_leaves", "tree_depth"),
    ("max_depth", "tree_depth"),
    ("depth", "tree_depth"),
    ("n_estimators", "n_estimators"),
    ("estimators", "n_estimators"),
    ("colsample", "feature_fraction"),
    ("max_features", "feature_fraction"),
    ("n_neighbors", "n_neighbors"),
    ("min_samples_leaf", "leaf_size"),
    ("min_samples_split", "leaf_size"),
    ("subsample", "sampling"),
    ("batch", "batch_or_budget"),
    ("work_mem", "batch_or_budget"),
    ("bootstrap", "sampling"),
    ("criterion", "optimizer_choice"),
    ("algorithm", "optimizer_choice"),
    ("weights", "distance_weight"),
    ("p", "minkowski_p"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _jsonable(v: Any) -> Any:
    if isinstance(v, float):
        return float(v)
    if isinstance(v, (int, str, bool)) or v is None:
        return v
    return json.loads(json.dumps(v, default=str))


def _stable_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _infer_role(name: str) -> Optional[str]:
    raw = name.split(":")[-1] if ":" in name else name
    low = raw.lower()
    for needle, role in sorted(ROLE_HINTS, key=lambda kv: -len(kv[0])):
        n = needle.lower()
        if n == "c":
            if raw == "C" or low in {"c", "c_"} or low.endswith("_c") or low.startswith("c_"):
                return role
            continue
        if len(n) == 1:
            if low == n:
                return role
            continue
        if n in low:
            return role
    return None


def _tunable_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in cfg.items() if k not in SKIP_KEYS}


# ---------------------------------------------------------------------------
# Load histories
# ---------------------------------------------------------------------------


def load_trials_from_bo_json(
    path: Path,
    *,
    algorithm: Optional[str] = None,
    higher_is_better: bool = True,
    openbox_minimize: bool = True,
) -> List[Dict[str, Any]]:
    """Load OpenBox-style history into trial dicts with higher-is-better scores."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    observations = list(obj.get("observations") or [])
    trials: List[Dict[str, Any]] = []
    for i, row in enumerate(observations, start=1):
        state = row.get("trial_state", 0)
        if state is not None and int(state) != 0:
            continue
        cfg = dict(row.get("config") or {})
        if algorithm and cfg.get("algorithm") not in (None, algorithm):
            continue
        objectives = row.get("objectives") or []
        if not objectives:
            continue
        raw = float(objectives[0])
        if not math.isfinite(raw):
            continue
        # OpenBox stores minimize objectives; MindWare uses higher-is-better metric.
        score = (-raw if openbox_minimize else raw)
        if not higher_is_better:
            # If caller says lower-is-better and openbox already minimizes, score=raw.
            score = raw if openbox_minimize else -raw
        trials.append(
            {
                "trial": i,
                "config": _tunable_config(cfg),
                "full_config": cfg,
                "score": float(score),
                "raw_objective": raw,
            }
        )
    # Re-number by successful order for stable trial ids in experience
    for j, t in enumerate(trials, start=1):
        t["trial"] = j
    return trials


def load_trials_from_pairs(
    configs: Sequence[Dict[str, Any]],
    perfs: Sequence[float],
) -> List[Dict[str, Any]]:
    trials: List[Dict[str, Any]] = []
    for i, (cfg, perf) in enumerate(zip(configs, perfs), start=1):
        score = float(perf)
        if not math.isfinite(score):
            continue
        trials.append(
            {
                "trial": i,
                "config": _tunable_config(dict(cfg)),
                "full_config": dict(cfg),
                "score": score,
            }
        )
    return trials


# ---------------------------------------------------------------------------
# Result: baseline, gain, dynamics
# ---------------------------------------------------------------------------


def compute_baseline(
    trials: Sequence[Dict[str, Any]],
    *,
    baseline_score: Optional[float] = None,
    baseline_kind: Optional[str] = None,
    default_config: Optional[Dict[str, Any]] = None,
) -> Tuple[float, str]:
    if baseline_score is not None:
        return float(baseline_score), baseline_kind or "provided"
    if default_config is not None:
        target = _tunable_config(default_config)
        for t in trials:
            if t["config"] == target:
                return float(t["score"]), "default_config"
        # Default known but not evaluated in this run → fall back
    if not trials:
        raise ValueError("No successful trials for baseline")
    return float(trials[0]["score"]), "first_success"


def normalized_gain(
    best: float,
    baseline: float,
    score_lo: float,
    score_hi: float,
    *,
    higher_is_better: bool = True,
) -> float:
    eps = 1e-12
    if higher_is_better:
        raw = best - baseline
        denom = max(score_hi - baseline, eps)
    else:
        raw = baseline - best
        denom = max(baseline - score_lo, eps)
    return _clip01(raw / denom)


def compute_dynamics(
    trials: Sequence[Dict[str, Any]],
    *,
    higher_is_better: bool = True,
    trend_window: int = 5,
    include_phase_note: bool = True,
) -> Dict[str, Any]:
    if not trials:
        return {
            "trend": "INSUFFICIENT_DATA",
            "stagnation": 0,
            "incumbent_updates": 0,
            "phase_note": None,
        }

    def better(a: float, b: float) -> bool:
        return a > b + 1e-12 if higher_is_better else a < b - 1e-12

    # Incumbent series
    best = trials[0]["score"]
    incumbent_update_trials = [int(trials[0]["trial"])]
    best_so_far: List[float] = []
    for t in trials:
        if better(t["score"], best):
            best = t["score"]
            incumbent_update_trials.append(int(t["trial"]))
        best_so_far.append(best)

    # Stagnation: trials after last improvement
    last_improve_idx = 0
    running = trials[0]["score"]
    for i, t in enumerate(trials):
        if better(t["score"], running):
            running = t["score"]
            last_improve_idx = i
    stagnation = len(trials) - 1 - last_improve_idx

    # Trend on last window (rel to full-run min-max)
    scores = [t["score"] for t in trials]
    lo, hi = min(scores), max(scores)
    span = hi - lo

    def rel(s: float) -> float:
        return 1.0 if span <= 0 else ((s - lo) / span if higher_is_better else (hi - s) / span)

    if len(trials) < 2:
        trend = "INSUFFICIENT_DATA"
    else:
        recent = trials[-trend_window:]
        rels = [rel(t["score"]) for t in recent]
        change = rels[-1] - rels[-2]
        if len(rels) >= 3 and statistics.pstdev(rels) < 0.02:
            trend = "STAGNANT"
        elif change > 0.02:
            trend = "IMPROVING"
        elif change < -0.02:
            trend = "DECLINING"
        else:
            trend = "FLAT"
        if stagnation >= max(5, int(0.15 * len(trials))) and trend != "IMPROVING":
            trend = "STAGNANT"

    phase_note = None
    if include_phase_note:
        n = len(trials)
        warm_end = max(1, int(math.ceil(0.1 * n)))
        last_prog = incumbent_update_trials[-1]
        plateau_len = stagnation
        phase_note = (
            f"warm 1–{warm_end}; progress with {len(incumbent_update_trials)} incumbent "
            f"updates until trial {last_prog}; plateau last {plateau_len} trials"
        )

    return {
        "trend": trend,
        "stagnation": int(stagnation),
        "incumbent_updates": len(incumbent_update_trials),
        "phase_note": phase_note,
        "_incumbent_update_trials": incumbent_update_trials,
    }


# ---------------------------------------------------------------------------
# Knowledge: factors, evidence, takeaways, avoid
# ---------------------------------------------------------------------------


def _cohorts(
    trials: Sequence[Dict[str, Any]],
    *,
    higher_is_better: bool,
    fraction: float = 0.2,
    min_k: int = 3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ordered = sorted(
        trials,
        key=lambda t: t["score"],
        reverse=higher_is_better,
    )
    k = max(min_k, int(math.ceil(fraction * len(ordered))))
    k = min(k, max(1, len(ordered) // 2))
    top = ordered[:k]
    worst = ordered[-k:]
    # Ensure disjoint when possible
    top_ids = {t["trial"] for t in top}
    worst = [t for t in worst if t["trial"] not in top_ids] or ordered[-k:]
    return top, worst


def _effect_threshold(score_lo: float, score_hi: float) -> float:
    span = abs(score_hi - score_lo)
    return max(0.05 * span, 1e-12)


def _factor_kind(values: Sequence[Any]) -> str:
    if all(_is_number(v) for v in values):
        return "numeric_knob"
    return "categorical_knob"


def analyze_factors(
    trials: Sequence[Dict[str, Any]],
    *,
    higher_is_better: bool = True,
    top_fraction: float = 0.2,
) -> Tuple[List[Dict[str, Any]], float]:
    if len(trials) < 4:
        return [], 0.0
    scores = [t["score"] for t in trials]
    score_lo, score_hi = min(scores), max(scores)
    delta = _effect_threshold(score_lo, score_hi)
    top, worst = _cohorts(trials, higher_is_better=higher_is_better, fraction=top_fraction)

    keys: List[str] = sorted({k for t in trials for k in t["config"]})
    raw_rows: List[Dict[str, Any]] = []

    for name in keys:
        vals_all = [t["config"][name] for t in trials if name in t["config"]]
        unique = {json.dumps(v, sort_keys=True, default=str) for v in vals_all}
        if len(unique) == 0:
            status = "NEVER_TESTED"
        elif len(unique) == 1:
            status = "CONSTANT"
        elif len(unique) < 2:
            status = "UNDER_TESTED"
        else:
            status = "TESTED"

        kind = _factor_kind(vals_all) if vals_all else "other"
        role = _infer_role(name)

        top_vals = [t["config"][name] for t in top if name in t["config"]]
        worst_vals = [t["config"][name] for t in worst if name in t["config"]]

        importance_raw = 0.0
        direction = "Insufficient contrast between top and worst cohorts."
        good_region: Any = None
        bad_region: Any = None

        if status == "TESTED" and top_vals and worst_vals:
            if kind == "numeric_knob":
                top_mean = statistics.mean(float(v) for v in top_vals)
                worst_mean = statistics.mean(float(v) for v in worst_vals)
                # Score gap between cohorts (not HP gap)
                top_score_mean = statistics.mean(t["score"] for t in top)
                worst_score_mean = statistics.mean(t["score"] for t in worst)
                score_gap = abs(top_score_mean - worst_score_mean)
                hp_gap = abs(top_mean - worst_mean)
                # Importance mixes score separation with HP separation (normalized later)
                importance_raw = score_gap * (1.0 + hp_gap / (abs(top_mean) + abs(worst_mean) + 1e-12))

                if score_gap >= delta and hp_gap > 0:
                    if top_mean > worst_mean:
                        direction = (
                            f"In this run, higher {name} was over-represented in top trials "
                            f"(top mean={top_mean:.4g}, worst mean={worst_mean:.4g})."
                        )
                        good_region = {"low": float(min(top_vals)), "high": float(max(top_vals))}
                        bad_region = {"low": float(min(worst_vals)), "high": float(max(worst_vals))}
                    else:
                        direction = (
                            f"In this run, lower {name} was over-represented in top trials "
                            f"(top mean={top_mean:.4g}, worst mean={worst_mean:.4g})."
                        )
                        good_region = {"low": float(min(top_vals)), "high": float(max(top_vals))}
                        bad_region = {"low": float(min(worst_vals)), "high": float(max(worst_vals))}
                else:
                    direction = f"No clear top-vs-worst separation for {name} (score gap < δ)."
                    importance_raw = 0.0
                    good_region = bad_region = None
            else:
                top_c = Counter(json.dumps(v, sort_keys=True, default=str) for v in top_vals)
                worst_c = Counter(json.dumps(v, sort_keys=True, default=str) for v in worst_vals)
                # Score means per category over all trials
                by_cat: Dict[str, List[float]] = defaultdict(list)
                for t in trials:
                    if name not in t["config"]:
                        continue
                    key = json.dumps(t["config"][name], sort_keys=True, default=str)
                    by_cat[key].append(float(t["score"]))
                cat_means = {k: statistics.mean(vs) for k, vs in by_cat.items() if vs}
                if len(cat_means) >= 2:
                    best_cat = max(cat_means, key=cat_means.get) if higher_is_better else min(cat_means, key=cat_means.get)
                    worst_cat = min(cat_means, key=cat_means.get) if higher_is_better else max(cat_means, key=cat_means.get)
                    score_gap = abs(cat_means[best_cat] - cat_means[worst_cat])
                    importance_raw = score_gap
                    if score_gap >= delta:
                        best_val = json.loads(best_cat)
                        worst_val = json.loads(worst_cat)
                        direction = (
                            f"In this run, {name}={best_val!r} associated with better scores "
                            f"than {worst_val!r}."
                        )
                        good_region = [best_val]
                        bad_region = [worst_val]
                    else:
                        direction = f"No clear categorical separation for {name}."
                        importance_raw = 0.0

        raw_rows.append(
            {
                "name": name,
                "kind": kind,
                "role": role,
                "status": status,
                "importance_raw": importance_raw,
                "method": "top_worst_gap",
                "direction": direction,
                "good_region": good_region,
                "bad_region": bad_region,
                "evidence_ids": [],
                "_top_trials": [int(t["trial"]) for t in top],
                "_worst_trials": [int(t["trial"]) for t in worst],
                "_score_gap": abs(
                    statistics.mean(t["score"] for t in top) - statistics.mean(t["score"] for t in worst)
                )
                if top and worst
                else 0.0,
            }
        )

    total = sum(r["importance_raw"] for r in raw_rows) or 1.0
    factors: List[Dict[str, Any]] = []
    for r in sorted(raw_rows, key=lambda x: x["importance_raw"], reverse=True):
        factors.append(
            {
                "name": r["name"],
                "kind": r["kind"],
                "role": r["role"],
                "status": r["status"],
                "importance": round(r["importance_raw"] / total, 4),
                "method": r["method"],
                "direction": r["direction"],
                "good_region": r["good_region"],
                "bad_region": r["bad_region"],
                "evidence_ids": [],
                "_top_trials": r["_top_trials"],
                "_worst_trials": r["_worst_trials"],
                "_score_gap": r["_score_gap"],
            }
        )
    return factors, delta


def build_evidence(
    trials: Sequence[Dict[str, Any]],
    factors: Sequence[Dict[str, Any]],
    dynamics: Dict[str, Any],
    *,
    higher_is_better: bool,
    delta: float,
) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    eid = 0

    def next_id() -> str:
        nonlocal eid
        eid += 1
        return f"ev_{eid:03d}"

    # Incumbent jumps
    update_trials = list(dynamics.get("_incumbent_update_trials") or [])
    by_id = {int(t["trial"]): t for t in trials}
    for a, b in zip(update_trials, update_trials[1:]):
        if a not in by_id or b not in by_id:
            continue
        prev, curr = by_id[a], by_id[b]
        changed = [
            k
            for k in sorted(set(prev["config"]) | set(curr["config"]))
            if prev["config"].get(k) != curr["config"].get(k)
        ]
        if not changed:
            continue
        dscore = float(curr["score"] - prev["score"])
        if not higher_is_better:
            dscore = -dscore
        ev_id = next_id()
        evidence.append(
            {
                "id": ev_id,
                "claim": (
                    f"Incumbent jump trial {a}→{b}: changed {', '.join(changed)}; "
                    f"score {prev['score']:.4g}→{curr['score']:.4g}."
                ),
                "kind": "incumbent_jump",
                "factors": changed,
                "trials": [a, b],
                "delta": round(dscore, 6),
                "confidence": _clip01(0.55 + min(0.35, abs(dscore) / (abs(delta) * 5 + 1e-12))),
            }
        )

    # Top vs worst per strong factor
    for fac in factors:
        if fac.get("status") != "TESTED" or fac.get("importance", 0) <= 0:
            continue
        if fac.get("good_region") is None and fac.get("bad_region") is None:
            continue
        if fac.get("_score_gap", 0) < delta:
            continue
        ev_id = next_id()
        evidence.append(
            {
                "id": ev_id,
                "claim": fac["direction"],
                "kind": "top_vs_worst",
                "factors": [fac["name"]],
                "trials": sorted(set(fac.get("_top_trials", [])[:5] + fac.get("_worst_trials", [])[:3])),
                "delta": round(float(fac.get("_score_gap", 0.0)), 6),
                "confidence": _clip01(0.5 + min(0.4, fac["importance"])),
            }
        )
        fac["evidence_ids"].append(ev_id)

    return evidence


def build_takeaways(
    factors: Sequence[Dict[str, Any]],
    evidence: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    by_id = {e["id"]: e for e in evidence}
    worked: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    transfer: List[Dict[str, Any]] = []

    for fac in factors:
        if fac.get("status") != "TESTED" or not fac.get("evidence_ids"):
            continue
        confs = [by_id[i]["confidence"] for i in fac["evidence_ids"] if i in by_id]
        conf = statistics.mean(confs) if confs else 0.0
        if conf < 0.6 or fac.get("importance", 0) <= 0:
            continue
        eids = list(fac["evidence_ids"])
        if fac.get("good_region") is not None:
            worked.append(
                {
                    "text": f"Prefer {fac['name']} in {fac['good_region']!s} ({fac['direction']})",
                    "confidence": round(conf, 3),
                    "evidence_ids": eids,
                    "scope": "same_algorithm",
                }
            )
        if fac.get("bad_region") is not None:
            failed.append(
                {
                    "text": f"Avoid {fac['name']} in {fac['bad_region']!s} on this run.",
                    "confidence": round(conf, 3),
                    "evidence_ids": eids,
                    "scope": "same_algorithm",
                }
            )
        # Transfer only for strong, role-labeled, non-tiny factors
        if fac.get("role") and fac.get("importance", 0) >= 0.15 and conf >= 0.65:
            transfer.append(
                {
                    "text": (
                        f"Role `{fac['role']}` ({fac['name']}) showed top-vs-worst association "
                        f"in this run; treat as a candidate priority knob on similar tasks."
                    ),
                    "confidence": round(conf * 0.9, 3),
                    "evidence_ids": eids,
                    "scope": "same_domain",
                }
            )

    return {
        "worked": worked[:5],
        "failed": failed[:5],
        "transfer": transfer[:3],
    }


def build_avoid(
    trials: Sequence[Dict[str, Any]],
    factors: Sequence[Dict[str, Any]],
    *,
    baseline_score: float,
    score_lo: float,
    score_hi: float,
    delta: float,
    higher_is_better: bool,
    max_items: int = 3,
) -> List[Dict[str, Any]]:
    span = abs(score_hi - score_lo)
    poor_cut = (
        baseline_score
        if higher_is_better
        else baseline_score
    )
    # Also allow "near floor"
    floor = score_lo + 0.2 * span if higher_is_better else score_hi - 0.2 * span

    def is_poor(mean_score: float) -> bool:
        if higher_is_better:
            return mean_score <= poor_cut or mean_score <= floor
        return mean_score >= poor_cut or mean_score >= floor

    avoid: List[Dict[str, Any]] = []
    for fac in factors:
        if fac.get("bad_region") is None or fac.get("status") != "TESTED":
            continue
        if fac.get("_score_gap", 0) < delta:
            continue
        worst_ids = set(fac.get("_worst_trials") or [])
        if len(worst_ids) < 2:
            continue
        worst_trials = [t for t in trials if t["trial"] in worst_ids]
        mean_score = statistics.mean(t["score"] for t in worst_trials)
        if not is_poor(mean_score):
            continue
        avoid.append(
            {
                "text": f"Harmful association for {fac['name']}: {fac['bad_region']!s}",
                "region": {fac["name"]: fac["bad_region"]},
                "trials": sorted(worst_ids)[:8],
            }
        )
        if len(avoid) >= max_items:
            break
    return avoid


# ---------------------------------------------------------------------------
# Quality + render
# ---------------------------------------------------------------------------


def compute_quality(
    trials: Sequence[Dict[str, Any]],
    factors: Sequence[Dict[str, Any]],
    evidence: Sequence[Dict[str, Any]],
    dynamics: Dict[str, Any],
    n_total_including_failed: Optional[int] = None,
) -> Dict[str, Any]:
    n = len(trials)
    n_tot = n_total_including_failed or n
    success_rate = n / max(n_tot, 1)
    tested = [f for f in factors if f.get("status") == "TESTED"]
    coverage = len(tested) / max(len(factors), 1) if factors else 0.0
    log_n = _clip01(math.log1p(n) / math.log1p(100))
    stag = dynamics.get("stagnation", 0)
    stag_pen = _clip01(stag / max(n, 1))
    fidelity = _clip01(0.35 * success_rate + 0.25 * coverage + 0.20 * log_n + 0.20 * (1 - stag_pen))

    supported = sum(1 for e in evidence if e.get("trials"))
    total_e = max(len(evidence), 1)
    effect = statistics.mean([abs(e.get("delta") or 0) for e in evidence]) if evidence else 0.0
    # Normalize effect loosely
    effect_score = _clip01(effect * 10) if evidence else 0.0
    conf = _clip01(0.40 * (supported / total_e) + 0.30 * effect_score + 0.20 * 0.7 + 0.10 * coverage)
    overall = round(0.5 * fidelity + 0.5 * conf, 4)
    return {
        "fidelity": round(fidelity, 4),
        "confidence": round(conf, 4),
        "overall": overall,
        "note": f"{n}/{n_tot} success; {len(tested)}/{len(factors)} factors TESTED; stagnation={stag}",
    }


def render_embed_text(problem: Dict[str, Any], knowledge: Dict[str, Any], result: Dict[str, Any]) -> str:
    top = sorted(knowledge.get("factors") or [], key=lambda f: f.get("importance", 0), reverse=True)[:3]
    fac_bits = "; ".join(f"{f['name']}({f['importance']:.2f})" for f in top) or "none"
    avoid = knowledge.get("avoid") or []
    avoid_bit = avoid[0]["text"] if avoid else "none"
    dyn = result.get("dynamics") or {}
    return (
        f"{problem.get('context', '')} "
        f"gain={result.get('normalized_gain')}; "
        f"incumbent_updates={dyn.get('incumbent_updates')}; "
        f"trend={dyn.get('trend')}; stagnant={dyn.get('stagnation')}. "
        f"Top factors: {fac_bits}. Avoid: {avoid_bit}."
    ).strip()


def render_prompt_block(
    problem: Dict[str, Any],
    knowledge: Dict[str, Any],
    result: Dict[str, Any],
    quality: Dict[str, Any],
) -> str:
    top = sorted(knowledge.get("factors") or [], key=lambda f: f.get("importance", 0), reverse=True)[:3]
    lines = [
        f"EXPERIENCE ({problem.get('dataset_id')}/{problem.get('algorithm_id')}, q={quality.get('overall')}):",
        f"- gain={result.get('normalized_gain')} over baseline ({result.get('baseline_kind')}); "
        f"{result.get('dynamics', {}).get('incumbent_updates')} incumbent updates; "
        f"ended {result.get('dynamics', {}).get('trend')}, stagnant {result.get('dynamics', {}).get('stagnation')}.",
    ]
    if result.get("dynamics", {}).get("phase_note"):
        lines.append(f"- curve: {result['dynamics']['phase_note']}")
    for f in top:
        if f.get("importance", 0) <= 0:
            continue
        lines.append(f"- factor {f['name']}: {f['direction']}")
    for a in (knowledge.get("avoid") or [])[:2]:
        lines.append(f"- avoid: {a['text']}")
    return "\n".join(lines)


def _strip_private(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_private(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [_strip_private(x) for x in obj]
    return obj


def _solution_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "config",
        "config": _jsonable(cfg),
        "vector": None,
        "sequence": None,
        "artifacts": None,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_experience_from_trials(
    trials: Sequence[Dict[str, Any]],
    *,
    run_uri: str,
    optimizer: str = "unknown",
    domain: str = "tabular_ml",
    task_type: str = "classification",
    dataset_id: Optional[str] = None,
    algorithm_id: Optional[str] = None,
    metric: str = "bal_acc",
    higher_is_better: bool = True,
    context: Optional[str] = None,
    profile: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    baseline_score: Optional[float] = None,
    baseline_kind: Optional[str] = None,
    default_config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    include_phase_note: bool = True,
    n_failed: int = 0,
    experience_id: Optional[str] = None,
) -> Dict[str, Any]:
    if not trials:
        raise ValueError("Need at least one successful trial")

    scores = [t["score"] for t in trials]
    score_lo, score_hi = min(scores), max(scores)
    best_trial = max(trials, key=lambda t: t["score"]) if higher_is_better else min(trials, key=lambda t: t["score"])

    base_score, base_kind = compute_baseline(
        trials,
        baseline_score=baseline_score,
        baseline_kind=baseline_kind,
        default_config=default_config,
    )
    gain = normalized_gain(
        best_trial["score"], base_score, score_lo, score_hi, higher_is_better=higher_is_better
    )
    dynamics = compute_dynamics(trials, higher_is_better=higher_is_better, include_phase_note=include_phase_note)

    factors, delta = analyze_factors(trials, higher_is_better=higher_is_better)
    evidence = build_evidence(trials, factors, dynamics, higher_is_better=higher_is_better, delta=delta)
    takeaways = build_takeaways(factors, evidence)
    avoid = build_avoid(
        trials,
        factors,
        baseline_score=base_score,
        score_lo=score_lo,
        score_hi=score_hi,
        delta=delta,
        higher_is_better=higher_is_better,
    )
    quality = compute_quality(trials, factors, evidence, dynamics, n_total_including_failed=len(trials) + n_failed)

    # Space fingerprint from observed keys
    keys = sorted({k for t in trials for k in t["config"]})
    space_factors = []
    for name in keys:
        vals = [t["config"][name] for t in trials if name in t["config"]]
        kind = _factor_kind(vals) if vals else "other"
        space_factors.append(
            {
                "name": name,
                "kind": kind,
                "role": _infer_role(name),
                "type": "float" if kind == "numeric_knob" else "categorical",
                "log_scale": None,
                "bounds": None,
            }
        )
    space_hash = _stable_hash({"keys": keys, "algorithm": algorithm_id})

    algo = algorithm_id or (trials[0]["full_config"].get("algorithm") if trials[0].get("full_config") else None)
    ds = dataset_id or "unknown"
    ctx = context or (
        f"{task_type} on {ds}"
        + (f" with {algo}" if algo else "")
        + f"; optimize {metric} ({'higher' if higher_is_better else 'lower'} is better)."
    )

    problem = {
        "domain": domain,
        "task_type": task_type,
        "dataset_id": ds,
        "algorithm_id": algo,
        "context": ctx,
        "metric": metric,
        "higher_is_better": higher_is_better,
        "profile": profile or {},
        "tags": tags or [],
        "representation": "structured_config",
        "space": {
            "space_hash": space_hash,
            "dim": len(keys),
            "decoder": None,
            "bounds": None,
            "factors": space_factors,
            "space_text": ", ".join(keys),
        },
    }

    result = {
        "n_trials": len(trials) + n_failed,
        "n_success": len(trials),
        "n_failed": n_failed,
        "score_range": {"lo": score_lo, "hi": score_hi},
        "best": {
            "trial": int(best_trial["trial"]),
            "score": float(best_trial["score"]),
            "solution": _solution_from_config(best_trial["config"]),
        },
        "baseline_score": float(base_score),
        "baseline_kind": base_kind,
        "normalized_gain": round(gain, 4),
        "dynamics": {
            "trend": dynamics["trend"],
            "stagnation": dynamics["stagnation"],
            "incumbent_updates": dynamics["incumbent_updates"],
            "phase_note": dynamics.get("phase_note"),
        },
    }

    knowledge = {
        "factors": [_strip_private(f) for f in factors if f.get("importance", 0) > 0 or f.get("status") == "TESTED"],
        "evidence": evidence,
        "takeaways": takeaways,
        "avoid": avoid,
    }
    # Keep top factors only in delivery render; store all TESTED with signal
    knowledge["factors"] = [
        f for f in knowledge["factors"] if f.get("importance", 0) > 0 or f.get("good_region") is not None
    ][:12]

    embed = render_embed_text(problem, knowledge, result)
    prompt = render_prompt_block(problem, knowledge, result, quality)

    # Train good/bad from top/worst configs
    top, worst = _cohorts(trials, higher_is_better=higher_is_better)
    train = {
        "best": _solution_from_config(best_trial["config"]),
        "good": [_solution_from_config(t["config"]) for t in top[:5]],
        "bad": [_solution_from_config(t["config"]) for t in worst[:5]],
    }

    exp_id = experience_id or f"exp_{ds}_{algo or 'algo'}_{space_hash}"
    return {
        "schema_version": "experience.v2",
        "id": exp_id,
        "created_at": _utc_now(),
        "meta": {
            "run_uri": run_uri,
            "optimizer": optimizer,
            "seed": seed,
            "budget_trials": len(trials) + n_failed,
            "summarizer": {
                "facts": "top_worst_gap+incumbent_jumps+curve_dynamics",
                "llm": None,
                "template": "experience_builder.v2",
            },
        },
        "problem": problem,
        "result": result,
        "knowledge": knowledge,
        "delivery": {
            "quality": quality,
            "embed_text": embed,
            "filters": {
                "domain": domain,
                "task_type": task_type,
                "representation": "structured_config",
                "algorithm_id": algo,
                "space_hash": space_hash,
            },
            "prompt_block": prompt,
            "train": train,
        },
    }


def build_experience_from_bo_json(path: Path, **kwargs: Any) -> Dict[str, Any]:
    path = Path(path)
    # Infer ids from .../cls/bal_acc/{dataset}/{algo}.json
    parts = path.parts
    dataset_id = kwargs.pop("dataset_id", None)
    algorithm_id = kwargs.pop("algorithm_id", None)
    task_type = kwargs.get("task_type", "classification")
    metric = kwargs.get("metric", "bal_acc")
    domain = kwargs.get("domain", "tabular_ml")
    if dataset_id is None or algorithm_id is None:
        try:
            algorithm_id = algorithm_id or path.stem
            dataset_id = dataset_id or path.parent.name
            metric = path.parent.parent.name
            tt = path.parent.parent.parent.name
            task_type = "classification" if tt == "cls" else ("regression" if tt == "rgs" else task_type)
            kwargs["metric"] = metric
            kwargs["task_type"] = task_type
        except Exception:
            pass

    higher_is_better = kwargs.pop("higher_is_better", metric not in {"mse", "mae", "rmse", "latency_ms"})
    openbox_minimize = kwargs.pop("openbox_minimize", True)
    algorithm = kwargs.get("algorithm_id") or algorithm_id

    # Count failures
    raw = json.loads(path.read_text(encoding="utf-8"))
    n_failed = sum(1 for row in (raw.get("observations") or []) if int(row.get("trial_state", 0)) != 0)

    trials = load_trials_from_bo_json(
        path,
        algorithm=algorithm,
        higher_is_better=True if higher_is_better else False,
        openbox_minimize=openbox_minimize,
    )
    # If lower-is-better metric stored as minimize already, load_trials with higher_is_better False
    # uses raw objective; for mse openbox may store positive mse to minimize — handle via flag.
    if not higher_is_better:
        trials = load_trials_from_bo_json(
            path, algorithm=algorithm, higher_is_better=False, openbox_minimize=True
        )

    return build_experience_from_trials(
        trials,
        run_uri=str(path),
        dataset_id=dataset_id,
        algorithm_id=algorithm_id,
        higher_is_better=higher_is_better,
        n_failed=n_failed,
        domain=domain,
        **kwargs,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build Experience v2 JSON from a BO history file")
    p.add_argument("history_json", type=Path, help="OpenBox-style observations JSON")
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--optimizer", default="bo_history")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--baseline-score", type=float, default=None)
    p.add_argument("--baseline-kind", default=None)
    p.add_argument("--no-phase-note", action="store_true")
    p.add_argument("--validate", action="store_true", help="Validate against experience.v2 schema")
    args = p.parse_args(argv)

    exp = build_experience_from_bo_json(
        args.history_json,
        optimizer=args.optimizer,
        seed=args.seed,
        baseline_score=args.baseline_score,
        baseline_kind=args.baseline_kind,
        include_phase_note=not args.no_phase_note,
    )

    if args.validate:
        try:
            import jsonschema
        except ImportError as e:
            raise SystemExit("jsonschema not installed") from e
        schema_path = Path(__file__).resolve().parents[1] / "schema" / "experience.v2.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.validate(exp, schema)

    text = json.dumps(exp, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Deterministic Experience v2 builder from BO / HPO trial histories.

Locked rules:
- incumbent_updates = improvement-event count (no separate field)
- phase_note optional, curve template only (not optimizer modes)
- no knowledge.summary — render embed_text / prompt_block from fields
- factor good/bad via top-vs-worst cohort association + δ threshold
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SKIP_KEYS = {"algorithm"}

ROLE_HINTS = (
    ("learning_rate", "learning_rate"),
    ("lr", "learning_rate"),
    ("l2", "regularization"),
    ("reg_lambda", "regularization"),
    ("reg_alpha", "regularization"),
    ("reg", "regularization"),
    ("alpha", "regularization"),
    ("C", "regularization"),
    ("min_child_weight", "regularization"),
    ("num_leaves", "tree_depth"),
    ("max_depth", "tree_depth"),
    ("depth", "tree_depth"),
    ("n_estimators", "n_estimators"),
    ("estimators", "n_estimators"),
    ("colsample", "feature_fraction"),
    ("max_features", "feature_fraction"),
    ("n_neighbors", "n_neighbors"),
    ("min_samples_leaf", "leaf_size"),
    ("min_samples_split", "leaf_size"),
    ("subsample", "sampling"),
    ("batch", "batch_or_budget"),
    ("work_mem", "batch_or_budget"),
    ("bootstrap", "sampling"),
    ("criterion", "optimizer_choice"),
    ("algorithm", "optimizer_choice"),
    ("weights", "distance_weight"),
    ("p", "minkowski_p"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _jsonable(v: Any) -> Any:
    if isinstance(v, float):
        return float(v)
    if isinstance(v, (int, str, bool)) or v is None:
        return v
    return json.loads(json.dumps(v, default=str))


def _stable_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _infer_role(name: str) -> Optional[str]:
    raw = name.split(":")[-1] if ":" in name else name
    low = raw.lower()
    for needle, role in sorted(ROLE_HINTS, key=lambda kv: -len(kv[0])):
        n = needle.lower()
        if n == "c":
            if raw == "C" or low in {"c", "c_"} or low.endswith("_c") or low.startswith("c_"):
                return role
            continue
        if len(n) == 1:
            if low == n:
                return role
            continue
        if n in low:
            return role
    return None


def _tunable_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in cfg.items() if k not in SKIP_KEYS}


# ---------------------------------------------------------------------------
# Load histories
# ---------------------------------------------------------------------------


def load_trials_from_bo_json(
    path: Path,
    *,
    algorithm: Optional[str] = None,
    higher_is_better: bool = True,
    openbox_minimize: bool = True,
) -> List[Dict[str, Any]]:
    """Load OpenBox-style history into trial dicts with higher-is-better scores."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    observations = list(obj.get("observations") or [])
    trials: List[Dict[str, Any]] = []
    for i, row in enumerate(observations, start=1):
        state = row.get("trial_state", 0)
        if state is not None and int(state) != 0:
            continue
        cfg = dict(row.get("config") or {})
        if algorithm and cfg.get("algorithm") not in (None, algorithm):
            continue
        objectives = row.get("objectives") or []
        if not objectives:
            continue
        raw = float(objectives[0])
        if not math.isfinite(raw):
            continue
        # OpenBox stores minimize objectives; MindWare uses higher-is-better metric.
        score = (-raw if openbox_minimize else raw)
        if not higher_is_better:
            # If caller says lower-is-better and openbox already minimizes, score=raw.
            score = raw if openbox_minimize else -raw
        trials.append(
            {
                "trial": i,
                "config": _tunable_config(cfg),
                "full_config": cfg,
                "score": float(score),
                "raw_objective": raw,
            }
        )
    # Re-number by successful order for stable trial ids in experience
    for j, t in enumerate(trials, start=1):
        t["trial"] = j
    return trials


def load_trials_from_pairs(
    configs: Sequence[Dict[str, Any]],
    perfs: Sequence[float],
) -> List[Dict[str, Any]]:
    trials: List[Dict[str, Any]] = []
    for i, (cfg, perf) in enumerate(zip(configs, perfs), start=1):
        score = float(perf)
        if not math.isfinite(score):
            continue
        trials.append(
            {
                "trial": i,
                "config": _tunable_config(dict(cfg)),
                "full_config": dict(cfg),
                "score": score,
            }
        )
    return trials


# ---------------------------------------------------------------------------
# Result: baseline, gain, dynamics
# ---------------------------------------------------------------------------


def compute_baseline(
    trials: Sequence[Dict[str, Any]],
    *,
    baseline_score: Optional[float] = None,
    baseline_kind: Optional[str] = None,
    default_config: Optional[Dict[str, Any]] = None,
) -> Tuple[float, str]:
    if baseline_score is not None:
        return float(baseline_score), baseline_kind or "provided"
    if default_config is not None:
        target = _tunable_config(default_config)
        for t in trials:
            if t["config"] == target:
                return float(t["score"]), "default_config"
        # Default known but not evaluated in this run → fall back
    if not trials:
        raise ValueError("No successful trials for baseline")
    return float(trials[0]["score"]), "first_success"


def normalized_gain(
    best: float,
    baseline: float,
    score_lo: float,
    score_hi: float,
    *,
    higher_is_better: bool = True,
) -> float:
    eps = 1e-12
    if higher_is_better:
        raw = best - baseline
        denom = max(score_hi - baseline, eps)
    else:
        raw = baseline - best
        denom = max(baseline - score_lo, eps)
    return _clip01(raw / denom)


def compute_dynamics(
    trials: Sequence[Dict[str, Any]],
    *,
    higher_is_better: bool = True,
    trend_window: int = 5,
    include_phase_note: bool = True,
) -> Dict[str, Any]:
    if not trials:
        return {
            "trend": "INSUFFICIENT_DATA",
            "stagnation": 0,
            "incumbent_updates": 0,
            "phase_note": None,
        }

    def better(a: float, b: float) -> bool:
        return a > b + 1e-12 if higher_is_better else a < b - 1e-12

    # Incumbent series
    best = trials[0]["score"]
    incumbent_update_trials = [int(trials[0]["trial"])]
    best_so_far: List[float] = []
    for t in trials:
        if better(t["score"], best):
            best = t["score"]
            incumbent_update_trials.append(int(t["trial"]))
        best_so_far.append(best)

    # Stagnation: trials after last improvement
    last_improve_idx = 0
    running = trials[0]["score"]
    for i, t in enumerate(trials):
        if better(t["score"], running):
            running = t["score"]
            last_improve_idx = i
    stagnation = len(trials) - 1 - last_improve_idx

    # Trend on last window (rel to full-run min-max)
    scores = [t["score"] for t in trials]
    lo, hi = min(scores), max(scores)
    span = hi - lo

    def rel(s: float) -> float:
        return 1.0 if span <= 0 else ((s - lo) / span if higher_is_better else (hi - s) / span)

    if len(trials) < 2:
        trend = "INSUFFICIENT_DATA"
    else:
        recent = trials[-trend_window:]
        rels = [rel(t["score"]) for t in recent]
        change = rels[-1] - rels[-2]
        if len(rels) >= 3 and statistics.pstdev(rels) < 0.02:
            trend = "STAGNANT"
        elif change > 0.02:
            trend = "IMPROVING"
        elif change < -0.02:
            trend = "DECLINING"
        else:
            trend = "FLAT"
        if stagnation >= max(5, int(0.15 * len(trials))) and trend != "IMPROVING":
            trend = "STAGNANT"

    phase_note = None
    if include_phase_note:
        n = len(trials)
        warm_end = max(1, int(math.ceil(0.1 * n)))
        last_prog = incumbent_update_trials[-1]
        plateau_len = stagnation
        phase_note = (
            f"warm 1–{warm_end}; progress with {len(incumbent_update_trials)} incumbent "
            f"updates until trial {last_prog}; plateau last {plateau_len} trials"
        )

    return {
        "trend": trend,
        "stagnation": int(stagnation),
        "incumbent_updates": len(incumbent_update_trials),
        "phase_note": phase_note,
        "_incumbent_update_trials": incumbent_update_trials,
    }


# ---------------------------------------------------------------------------
# Knowledge: factors, evidence, takeaways, avoid
# ---------------------------------------------------------------------------


def _cohorts(
    trials: Sequence[Dict[str, Any]],
    *,
    higher_is_better: bool,
    fraction: float = 0.2,
    min_k: int = 3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ordered = sorted(
        trials,
        key=lambda t: t["score"],
        reverse=higher_is_better,
    )
    k = max(min_k, int(math.ceil(fraction * len(ordered))))
    k = min(k, max(1, len(ordered) // 2))
    top = ordered[:k]
    worst = ordered[-k:]
    # Ensure disjoint when possible
    top_ids = {t["trial"] for t in top}
    worst = [t for t in worst if t["trial"] not in top_ids] or ordered[-k:]
    return top, worst


def _effect_threshold(score_lo: float, score_hi: float) -> float:
    span = abs(score_hi - score_lo)
    return max(0.05 * span, 1e-12)


def _factor_kind(values: Sequence[Any]) -> str:
    if all(_is_number(v) for v in values):
        return "numeric_knob"
    return "categorical_knob"


def analyze_factors(
    trials: Sequence[Dict[str, Any]],
    *,
    higher_is_better: bool = True,
    top_fraction: float = 0.2,
) -> Tuple[List[Dict[str, Any]], float]:
    if len(trials) < 4:
        return [], 0.0
    scores = [t["score"] for t in trials]
    score_lo, score_hi = min(scores), max(scores)
    delta = _effect_threshold(score_lo, score_hi)
    top, worst = _cohorts(trials, higher_is_better=higher_is_better, fraction=top_fraction)

    keys: List[str] = sorted({k for t in trials for k in t["config"]})
    raw_rows: List[Dict[str, Any]] = []

    for name in keys:
        vals_all = [t["config"][name] for t in trials if name in t["config"]]
        unique = {json.dumps(v, sort_keys=True, default=str) for v in vals_all}
        if len(unique) == 0:
            status = "NEVER_TESTED"
        elif len(unique) == 1:
            status = "CONSTANT"
        elif len(unique) < 2:
            status = "UNDER_TESTED"
        else:
            status = "TESTED"

        kind = _factor_kind(vals_all) if vals_all else "other"
        role = _infer_role(name)

        top_vals = [t["config"][name] for t in top if name in t["config"]]
        worst_vals = [t["config"][name] for t in worst if name in t["config"]]

        importance_raw = 0.0
        direction = "Insufficient contrast between top and worst cohorts."
        good_region: Any = None
        bad_region: Any = None

        if status == "TESTED" and top_vals and worst_vals:
            if kind == "numeric_knob":
                top_mean = statistics.mean(float(v) for v in top_vals)
                worst_mean = statistics.mean(float(v) for v in worst_vals)
                # Score gap between cohorts (not HP gap)
                top_score_mean = statistics.mean(t["score"] for t in top)
                worst_score_mean = statistics.mean(t["score"] for t in worst)
                score_gap = abs(top_score_mean - worst_score_mean)
                hp_gap = abs(top_mean - worst_mean)
                # Importance mixes score separation with HP separation (normalized later)
                importance_raw = score_gap * (1.0 + hp_gap / (abs(top_mean) + abs(worst_mean) + 1e-12))

                if score_gap >= delta and hp_gap > 0:
                    if top_mean > worst_mean:
                        direction = (
                            f"In this run, higher {name} was over-represented in top trials "
                            f"(top mean={top_mean:.4g}, worst mean={worst_mean:.4g})."
                        )
                        good_region = {"low": float(min(top_vals)), "high": float(max(top_vals))}
                        bad_region = {"low": float(min(worst_vals)), "high": float(max(worst_vals))}
                    else:
                        direction = (
                            f"In this run, lower {name} was over-represented in top trials "
                            f"(top mean={top_mean:.4g}, worst mean={worst_mean:.4g})."
                        )
                        good_region = {"low": float(min(top_vals)), "high": float(max(top_vals))}
                        bad_region = {"low": float(min(worst_vals)), "high": float(max(worst_vals))}
                else:
                    direction = f"No clear top-vs-worst separation for {name} (score gap < δ)."
                    importance_raw = 0.0
                    good_region = bad_region = None
            else:
                top_c = Counter(json.dumps(v, sort_keys=True, default=str) for v in top_vals)
                worst_c = Counter(json.dumps(v, sort_keys=True, default=str) for v in worst_vals)
                # Score means per category over all trials
                by_cat: Dict[str, List[float]] = defaultdict(list)
                for t in trials:
                    if name not in t["config"]:
                        continue
                    key = json.dumps(t["config"][name], sort_keys=True, default=str)
                    by_cat[key].append(float(t["score"]))
                cat_means = {k: statistics.mean(vs) for k, vs in by_cat.items() if vs}
                if len(cat_means) >= 2:
                    best_cat = max(cat_means, key=cat_means.get) if higher_is_better else min(cat_means, key=cat_means.get)
                    worst_cat = min(cat_means, key=cat_means.get) if higher_is_better else max(cat_means, key=cat_means.get)
                    score_gap = abs(cat_means[best_cat] - cat_means[worst_cat])
                    importance_raw = score_gap
                    if score_gap >= delta:
                        best_val = json.loads(best_cat)
                        worst_val = json.loads(worst_cat)
                        direction = (
                            f"In this run, {name}={best_val!r} associated with better scores "
                            f"than {worst_val!r}."
                        )
                        good_region = [best_val]
                        bad_region = [worst_val]
                    else:
                        direction = f"No clear categorical separation for {name}."
                        importance_raw = 0.0

        raw_rows.append(
            {
                "name": name,
                "kind": kind,
                "role": role,
                "status": status,
                "importance_raw": importance_raw,
                "method": "top_worst_gap",
                "direction": direction,
                "good_region": good_region,
                "bad_region": bad_region,
                "evidence_ids": [],
                "_top_trials": [int(t["trial"]) for t in top],
                "_worst_trials": [int(t["trial"]) for t in worst],
                "_score_gap": abs(
                    statistics.mean(t["score"] for t in top) - statistics.mean(t["score"] for t in worst)
                )
                if top and worst
                else 0.0,
            }
        )

    total = sum(r["importance_raw"] for r in raw_rows) or 1.0
    factors: List[Dict[str, Any]] = []
    for r in sorted(raw_rows, key=lambda x: x["importance_raw"], reverse=True):
        factors.append(
            {
                "name": r["name"],
                "kind": r["kind"],
                "role": r["role"],
                "status": r["status"],
                "importance": round(r["importance_raw"] / total, 4),
                "method": r["method"],
                "direction": r["direction"],
                "good_region": r["good_region"],
                "bad_region": r["bad_region"],
                "evidence_ids": [],
                "_top_trials": r["_top_trials"],
                "_worst_trials": r["_worst_trials"],
                "_score_gap": r["_score_gap"],
            }
        )
    return factors, delta


def build_evidence(
    trials: Sequence[Dict[str, Any]],
    factors: Sequence[Dict[str, Any]],
    dynamics: Dict[str, Any],
    *,
    higher_is_better: bool,
    delta: float,
) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    eid = 0

    def next_id() -> str:
        nonlocal eid
        eid += 1
        return f"ev_{eid:03d}"

    # Incumbent jumps
    update_trials = list(dynamics.get("_incumbent_update_trials") or [])
    by_id = {int(t["trial"]): t for t in trials}
    for a, b in zip(update_trials, update_trials[1:]):
        if a not in by_id or b not in by_id:
            continue
        prev, curr = by_id[a], by_id[b]
        changed = [
            k
            for k in sorted(set(prev["config"]) | set(curr["config"]))
            if prev["config"].get(k) != curr["config"].get(k)
        ]
        if not changed:
            continue
        dscore = float(curr["score"] - prev["score"])
        if not higher_is_better:
            dscore = -dscore
        ev_id = next_id()
        evidence.append(
            {
                "id": ev_id,
                "claim": (
                    f"Incumbent jump trial {a}→{b}: changed {', '.join(changed)}; "
                    f"score {prev['score']:.4g}→{curr['score']:.4g}."
                ),
                "kind": "incumbent_jump",
                "factors": changed,
                "trials": [a, b],
                "delta": round(dscore, 6),
                "confidence": _clip01(0.55 + min(0.35, abs(dscore) / (abs(delta) * 5 + 1e-12))),
            }
        )

    # Top vs worst per strong factor
    for fac in factors:
        if fac.get("status") != "TESTED" or fac.get("importance", 0) <= 0:
            continue
        if fac.get("good_region") is None and fac.get("bad_region") is None:
            continue
        if fac.get("_score_gap", 0) < delta:
            continue
        ev_id = next_id()
        evidence.append(
            {
                "id": ev_id,
                "claim": fac["direction"],
                "kind": "top_vs_worst",
                "factors": [fac["name"]],
                "trials": sorted(set(fac.get("_top_trials", [])[:5] + fac.get("_worst_trials", [])[:3])),
                "delta": round(float(fac.get("_score_gap", 0.0)), 6),
                "confidence": _clip01(0.5 + min(0.4, fac["importance"])),
            }
        )
        fac["evidence_ids"].append(ev_id)

    return evidence


def build_takeaways(
    factors: Sequence[Dict[str, Any]],
    evidence: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    by_id = {e["id"]: e for e in evidence}
    worked: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    transfer: List[Dict[str, Any]] = []

    for fac in factors:
        if fac.get("status") != "TESTED" or not fac.get("evidence_ids"):
            continue
        confs = [by_id[i]["confidence"] for i in fac["evidence_ids"] if i in by_id]
        conf = statistics.mean(confs) if confs else 0.0
        if conf < 0.6 or fac.get("importance", 0) <= 0:
            continue
        eids = list(fac["evidence_ids"])
        if fac.get("good_region") is not None:
            worked.append(
                {
                    "text": f"Prefer {fac['name']} in {fac['good_region']!s} ({fac['direction']})",
                    "confidence": round(conf, 3),
                    "evidence_ids": eids,
                    "scope": "same_algorithm",
                }
            )
        if fac.get("bad_region") is not None:
            failed.append(
                {
                    "text": f"Avoid {fac['name']} in {fac['bad_region']!s} on this run.",
                    "confidence": round(conf, 3),
                    "evidence_ids": eids,
                    "scope": "same_algorithm",
                }
            )
        # Transfer only for strong, role-labeled, non-tiny factors
        if fac.get("role") and fac.get("importance", 0) >= 0.15 and conf >= 0.65:
            transfer.append(
                {
                    "text": (
                        f"Role `{fac['role']}` ({fac['name']}) showed top-vs-worst association "
                        f"in this run; treat as a candidate priority knob on similar tasks."
                    ),
                    "confidence": round(conf * 0.9, 3),
                    "evidence_ids": eids,
                    "scope": "same_domain",
                }
            )

    return {
        "worked": worked[:5],
        "failed": failed[:5],
        "transfer": transfer[:3],
    }


def build_avoid(
    trials: Sequence[Dict[str, Any]],
    factors: Sequence[Dict[str, Any]],
    *,
    baseline_score: float,
    score_lo: float,
    score_hi: float,
    delta: float,
    higher_is_better: bool,
    max_items: int = 3,
) -> List[Dict[str, Any]]:
    span = abs(score_hi - score_lo)
    poor_cut = (
        baseline_score
        if higher_is_better
        else baseline_score
    )
    # Also allow "near floor"
    floor = score_lo + 0.2 * span if higher_is_better else score_hi - 0.2 * span

    def is_poor(mean_score: float) -> bool:
        if higher_is_better:
            return mean_score <= poor_cut or mean_score <= floor
        return mean_score >= poor_cut or mean_score >= floor

    avoid: List[Dict[str, Any]] = []
    for fac in factors:
        if fac.get("bad_region") is None or fac.get("status") != "TESTED":
            continue
        if fac.get("_score_gap", 0) < delta:
            continue
        worst_ids = set(fac.get("_worst_trials") or [])
        if len(worst_ids) < 2:
            continue
        worst_trials = [t for t in trials if t["trial"] in worst_ids]
        mean_score = statistics.mean(t["score"] for t in worst_trials)
        if not is_poor(mean_score):
            continue
        avoid.append(
            {
                "text": f"Harmful association for {fac['name']}: {fac['bad_region']!s}",
                "region": {fac["name"]: fac["bad_region"]},
                "trials": sorted(worst_ids)[:8],
            }
        )
        if len(avoid) >= max_items:
            break
    return avoid


# ---------------------------------------------------------------------------
# Quality + render
# ---------------------------------------------------------------------------


def compute_quality(
    trials: Sequence[Dict[str, Any]],
    factors: Sequence[Dict[str, Any]],
    evidence: Sequence[Dict[str, Any]],
    dynamics: Dict[str, Any],
    n_total_including_failed: Optional[int] = None,
) -> Dict[str, Any]:
    n = len(trials)
    n_tot = n_total_including_failed or n
    success_rate = n / max(n_tot, 1)
    tested = [f for f in factors if f.get("status") == "TESTED"]
    coverage = len(tested) / max(len(factors), 1) if factors else 0.0
    log_n = _clip01(math.log1p(n) / math.log1p(100))
    stag = dynamics.get("stagnation", 0)
    stag_pen = _clip01(stag / max(n, 1))
    fidelity = _clip01(0.35 * success_rate + 0.25 * coverage + 0.20 * log_n + 0.20 * (1 - stag_pen))

    supported = sum(1 for e in evidence if e.get("trials"))
    total_e = max(len(evidence), 1)
    effect = statistics.mean([abs(e.get("delta") or 0) for e in evidence]) if evidence else 0.0
    # Normalize effect loosely
    effect_score = _clip01(effect * 10) if evidence else 0.0
    conf = _clip01(0.40 * (supported / total_e) + 0.30 * effect_score + 0.20 * 0.7 + 0.10 * coverage)
    overall = round(0.5 * fidelity + 0.5 * conf, 4)
    return {
        "fidelity": round(fidelity, 4),
        "confidence": round(conf, 4),
        "overall": overall,
        "note": f"{n}/{n_tot} success; {len(tested)}/{len(factors)} factors TESTED; stagnation={stag}",
    }


def render_embed_text(problem: Dict[str, Any], knowledge: Dict[str, Any], result: Dict[str, Any]) -> str:
    top = sorted(knowledge.get("factors") or [], key=lambda f: f.get("importance", 0), reverse=True)[:3]
    fac_bits = "; ".join(f"{f['name']}({f['importance']:.2f})" for f in top) or "none"
    avoid = knowledge.get("avoid") or []
    avoid_bit = avoid[0]["text"] if avoid else "none"
    dyn = result.get("dynamics") or {}
    return (
        f"{problem.get('context', '')} "
        f"gain={result.get('normalized_gain')}; "
        f"incumbent_updates={dyn.get('incumbent_updates')}; "
        f"trend={dyn.get('trend')}; stagnant={dyn.get('stagnation')}. "
        f"Top factors: {fac_bits}. Avoid: {avoid_bit}."
    ).strip()


def render_prompt_block(
    problem: Dict[str, Any],
    knowledge: Dict[str, Any],
    result: Dict[str, Any],
    quality: Dict[str, Any],
) -> str:
    top = sorted(knowledge.get("factors") or [], key=lambda f: f.get("importance", 0), reverse=True)[:3]
    lines = [
        f"EXPERIENCE ({problem.get('dataset_id')}/{problem.get('algorithm_id')}, q={quality.get('overall')}):",
        f"- gain={result.get('normalized_gain')} over baseline ({result.get('baseline_kind')}); "
        f"{result.get('dynamics', {}).get('incumbent_updates')} incumbent updates; "
        f"ended {result.get('dynamics', {}).get('trend')}, stagnant {result.get('dynamics', {}).get('stagnation')}.",
    ]
    if result.get("dynamics", {}).get("phase_note"):
        lines.append(f"- curve: {result['dynamics']['phase_note']}")
    for f in top:
        if f.get("importance", 0) <= 0:
            continue
        lines.append(f"- factor {f['name']}: {f['direction']}")
    for a in (knowledge.get("avoid") or [])[:2]:
        lines.append(f"- avoid: {a['text']}")
    return "\n".join(lines)


def _strip_private(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_private(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [_strip_private(x) for x in obj]
    return obj


def _solution_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "config",
        "config": _jsonable(cfg),
        "vector": None,
        "sequence": None,
        "artifacts": None,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_experience_from_trials(
    trials: Sequence[Dict[str, Any]],
    *,
    run_uri: str,
    optimizer: str = "unknown",
    domain: str = "tabular_ml",
    task_type: str = "classification",
    dataset_id: Optional[str] = None,
    algorithm_id: Optional[str] = None,
    metric: str = "bal_acc",
    higher_is_better: bool = True,
    context: Optional[str] = None,
    profile: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    baseline_score: Optional[float] = None,
    baseline_kind: Optional[str] = None,
    default_config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    include_phase_note: bool = True,
    n_failed: int = 0,
    experience_id: Optional[str] = None,
) -> Dict[str, Any]:
    if not trials:
        raise ValueError("Need at least one successful trial")

    scores = [t["score"] for t in trials]
    score_lo, score_hi = min(scores), max(scores)
    best_trial = max(trials, key=lambda t: t["score"]) if higher_is_better else min(trials, key=lambda t: t["score"])

    base_score, base_kind = compute_baseline(
        trials,
        baseline_score=baseline_score,
        baseline_kind=baseline_kind,
        default_config=default_config,
    )
    gain = normalized_gain(
        best_trial["score"], base_score, score_lo, score_hi, higher_is_better=higher_is_better
    )
    dynamics = compute_dynamics(trials, higher_is_better=higher_is_better, include_phase_note=include_phase_note)

    factors, delta = analyze_factors(trials, higher_is_better=higher_is_better)
    evidence = build_evidence(trials, factors, dynamics, higher_is_better=higher_is_better, delta=delta)
    takeaways = build_takeaways(factors, evidence)
    avoid = build_avoid(
        trials,
        factors,
        baseline_score=base_score,
        score_lo=score_lo,
        score_hi=score_hi,
        delta=delta,
        higher_is_better=higher_is_better,
    )
    quality = compute_quality(trials, factors, evidence, dynamics, n_total_including_failed=len(trials) + n_failed)

    # Space fingerprint from observed keys
    keys = sorted({k for t in trials for k in t["config"]})
    space_factors = []
    for name in keys:
        vals = [t["config"][name] for t in trials if name in t["config"]]
        kind = _factor_kind(vals) if vals else "other"
        space_factors.append(
            {
                "name": name,
                "kind": kind,
                "role": _infer_role(name),
                "type": "float" if kind == "numeric_knob" else "categorical",
                "log_scale": None,
                "bounds": None,
            }
        )
    space_hash = _stable_hash({"keys": keys, "algorithm": algorithm_id})

    algo = algorithm_id or (trials[0]["full_config"].get("algorithm") if trials[0].get("full_config") else None)
    ds = dataset_id or "unknown"
    ctx = context or (
        f"{task_type} on {ds}"
        + (f" with {algo}" if algo else "")
        + f"; optimize {metric} ({'higher' if higher_is_better else 'lower'} is better)."
    )

    problem = {
        "domain": domain,
        "task_type": task_type,
        "dataset_id": ds,
        "algorithm_id": algo,
        "context": ctx,
        "metric": metric,
        "higher_is_better": higher_is_better,
        "profile": profile or {},
        "tags": tags or [],
        "representation": "structured_config",
        "space": {
            "space_hash": space_hash,
            "dim": len(keys),
            "decoder": None,
            "bounds": None,
            "factors": space_factors,
            "space_text": ", ".join(keys),
        },
    }

    result = {
        "n_trials": len(trials) + n_failed,
        "n_success": len(trials),
        "n_failed": n_failed,
        "score_range": {"lo": score_lo, "hi": score_hi},
        "best": {
            "trial": int(best_trial["trial"]),
            "score": float(best_trial["score"]),
            "solution": _solution_from_config(best_trial["config"]),
        },
        "baseline_score": float(base_score),
        "baseline_kind": base_kind,
        "normalized_gain": round(gain, 4),
        "dynamics": {
            "trend": dynamics["trend"],
            "stagnation": dynamics["stagnation"],
            "incumbent_updates": dynamics["incumbent_updates"],
            "phase_note": dynamics.get("phase_note"),
        },
    }

    knowledge = {
        "factors": [_strip_private(f) for f in factors if f.get("importance", 0) > 0 or f.get("status") == "TESTED"],
        "evidence": evidence,
        "takeaways": takeaways,
        "avoid": avoid,
    }
    # Keep top factors only in delivery render; store all TESTED with signal
    knowledge["factors"] = [
        f for f in knowledge["factors"] if f.get("importance", 0) > 0 or f.get("good_region") is not None
    ][:12]

    embed = render_embed_text(problem, knowledge, result)
    prompt = render_prompt_block(problem, knowledge, result, quality)

    # Train good/bad from top/worst configs
    top, worst = _cohorts(trials, higher_is_better=higher_is_better)
    train = {
        "best": _solution_from_config(best_trial["config"]),
        "good": [_solution_from_config(t["config"]) for t in top[:5]],
        "bad": [_solution_from_config(t["config"]) for t in worst[:5]],
    }

    exp_id = experience_id or f"exp_{ds}_{algo or 'algo'}_{space_hash}"
    return {
        "schema_version": "experience.v2",
        "id": exp_id,
        "created_at": _utc_now(),
        "meta": {
            "run_uri": run_uri,
            "optimizer": optimizer,
            "seed": seed,
            "budget_trials": len(trials) + n_failed,
            "summarizer": {
                "facts": "top_worst_gap+incumbent_jumps+curve_dynamics",
                "llm": None,
                "template": "experience_builder.v2",
            },
        },
        "problem": problem,
        "result": result,
        "knowledge": knowledge,
        "delivery": {
            "quality": quality,
            "embed_text": embed,
            "filters": {
                "domain": domain,
                "task_type": task_type,
                "representation": "structured_config",
                "algorithm_id": algo,
                "space_hash": space_hash,
            },
            "prompt_block": prompt,
            "train": train,
        },
    }


def build_experience_from_bo_json(path: Path, **kwargs: Any) -> Dict[str, Any]:
    path = Path(path)
    # Infer ids from .../cls/bal_acc/{dataset}/{algo}.json
    parts = path.parts
    dataset_id = kwargs.pop("dataset_id", None)
    algorithm_id = kwargs.pop("algorithm_id", None)
    task_type = kwargs.get("task_type", "classification")
    metric = kwargs.get("metric", "bal_acc")
    domain = kwargs.get("domain", "tabular_ml")
    if dataset_id is None or algorithm_id is None:
        try:
            algorithm_id = algorithm_id or path.stem
            dataset_id = dataset_id or path.parent.name
            metric = path.parent.parent.name
            tt = path.parent.parent.parent.name
            task_type = "classification" if tt == "cls" else ("regression" if tt == "rgs" else task_type)
            kwargs["metric"] = metric
            kwargs["task_type"] = task_type
        except Exception:
            pass

    higher_is_better = kwargs.pop("higher_is_better", metric not in {"mse", "mae", "rmse", "latency_ms"})
    openbox_minimize = kwargs.pop("openbox_minimize", True)
    algorithm = kwargs.get("algorithm_id") or algorithm_id

    # Count failures
    raw = json.loads(path.read_text(encoding="utf-8"))
    n_failed = sum(1 for row in (raw.get("observations") or []) if int(row.get("trial_state", 0)) != 0)

    trials = load_trials_from_bo_json(
        path,
        algorithm=algorithm,
        higher_is_better=True if higher_is_better else False,
        openbox_minimize=openbox_minimize,
    )
    # If lower-is-better metric stored as minimize already, load_trials with higher_is_better False
    # uses raw objective; for mse openbox may store positive mse to minimize — handle via flag.
    if not higher_is_better:
        trials = load_trials_from_bo_json(
            path, algorithm=algorithm, higher_is_better=False, openbox_minimize=True
        )

    return build_experience_from_trials(
        trials,
        run_uri=str(path),
        dataset_id=dataset_id,
        algorithm_id=algorithm_id,
        higher_is_better=higher_is_better,
        n_failed=n_failed,
        domain=domain,
        **kwargs,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build Experience v2 JSON from a BO history file")
    p.add_argument("history_json", type=Path, help="OpenBox-style observations JSON")
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--optimizer", default="bo_history")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--baseline-score", type=float, default=None)
    p.add_argument("--baseline-kind", default=None)
    p.add_argument("--no-phase-note", action="store_true")
    p.add_argument("--validate", action="store_true", help="Validate against experience.v2 schema")
    args = p.parse_args(argv)

    exp = build_experience_from_bo_json(
        args.history_json,
        optimizer=args.optimizer,
        seed=args.seed,
        baseline_score=args.baseline_score,
        baseline_kind=args.baseline_kind,
        include_phase_note=not args.no_phase_note,
    )

    if args.validate:
        try:
            import jsonschema
        except ImportError as e:
            raise SystemExit("jsonschema not installed") from e
        schema_path = Path(__file__).resolve().parents[1] / "schema" / "experience.v2.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.validate(exp, schema)

    text = json.dumps(exp, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Deterministic Experience v2 builder from BO / HPO trial histories.

Locked rules:
- incumbent_updates = improvement-event count (no separate field)
- phase_note optional, curve template only (not optimizer modes)
- no knowledge.summary — render embed_text / prompt_block from fields
- factor good/bad via top-vs-worst cohort association + δ threshold
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SKIP_KEYS = {"algorithm"}

ROLE_HINTS = (
    ("learning_rate", "learning_rate"),
    ("lr", "learning_rate"),
    ("l2", "regularization"),
    ("reg_lambda", "regularization"),
    ("reg_alpha", "regularization"),
    ("reg", "regularization"),
    ("alpha", "regularization"),
    ("C", "regularization"),
    ("min_child_weight", "regularization"),
    ("num_leaves", "tree_depth"),
    ("max_depth", "tree_depth"),
    ("depth", "tree_depth"),
    ("n_estimators", "n_estimators"),
    ("estimators", "n_estimators"),
    ("colsample", "feature_fraction"),
    ("max_features", "feature_fraction"),
    ("n_neighbors", "n_neighbors"),
    ("min_samples_leaf", "leaf_size"),
    ("min_samples_split", "leaf_size"),
    ("subsample", "sampling"),
    ("batch", "batch_or_budget"),
    ("work_mem", "batch_or_budget"),
    ("bootstrap", "sampling"),
    ("criterion", "optimizer_choice"),
    ("algorithm", "optimizer_choice"),
    ("weights", "distance_weight"),
    ("p", "minkowski_p"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _jsonable(v: Any) -> Any:
    if isinstance(v, float):
        return float(v)
    if isinstance(v, (int, str, bool)) or v is None:
        return v
    return json.loads(json.dumps(v, default=str))


def _stable_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _infer_role(name: str) -> Optional[str]:
    raw = name.split(":")[-1] if ":" in name else name
    low = raw.lower()
    for needle, role in sorted(ROLE_HINTS, key=lambda kv: -len(kv[0])):
        n = needle.lower()
        if n == "c":
            if raw == "C" or low in {"c", "c_"} or low.endswith("_c") or low.startswith("c_"):
                return role
            continue
        if len(n) == 1:
            if low == n:
                return role
            continue
        if n in low:
            return role
    return None


def _tunable_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in cfg.items() if k not in SKIP_KEYS}


# ---------------------------------------------------------------------------
# Load histories
# ---------------------------------------------------------------------------


def load_trials_from_bo_json(
    path: Path,
    *,
    algorithm: Optional[str] = None,
    higher_is_better: bool = True,
    openbox_minimize: bool = True,
) -> List[Dict[str, Any]]:
    """Load OpenBox-style history into trial dicts with higher-is-better scores."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    observations = list(obj.get("observations") or [])
    trials: List[Dict[str, Any]] = []
    for i, row in enumerate(observations, start=1):
        state = row.get("trial_state", 0)
        if state is not None and int(state) != 0:
            continue
        cfg = dict(row.get("config") or {})
        if algorithm and cfg.get("algorithm") not in (None, algorithm):
            continue
        objectives = row.get("objectives") or []
        if not objectives:
            continue
        raw = float(objectives[0])
        if not math.isfinite(raw):
            continue
        # OpenBox stores minimize objectives; MindWare uses higher-is-better metric.
        score = (-raw if openbox_minimize else raw)
        if not higher_is_better:
            # If caller says lower-is-better and openbox already minimizes, score=raw.
            score = raw if openbox_minimize else -raw
        trials.append(
            {
                "trial": i,
                "config": _tunable_config(cfg),
                "full_config": cfg,
                "score": float(score),
                "raw_objective": raw,
            }
        )
    # Re-number by successful order for stable trial ids in experience
    for j, t in enumerate(trials, start=1):
        t["trial"] = j
    return trials


def load_trials_from_pairs(
    configs: Sequence[Dict[str, Any]],
    perfs: Sequence[float],
) -> List[Dict[str, Any]]:
    trials: List[Dict[str, Any]] = []
    for i, (cfg, perf) in enumerate(zip(configs, perfs), start=1):
        score = float(perf)
        if not math.isfinite(score):
            continue
        trials.append(
            {
                "trial": i,
                "config": _tunable_config(dict(cfg)),
                "full_config": dict(cfg),
                "score": score,
            }
        )
    return trials


# ---------------------------------------------------------------------------
# Result: baseline, gain, dynamics
# ---------------------------------------------------------------------------


def compute_baseline(
    trials: Sequence[Dict[str, Any]],
    *,
    baseline_score: Optional[float] = None,
    baseline_kind: Optional[str] = None,
    default_config: Optional[Dict[str, Any]] = None,
) -> Tuple[float, str]:
    if baseline_score is not None:
        return float(baseline_score), baseline_kind or "provided"
    if default_config is not None:
        target = _tunable_config(default_config)
        for t in trials:
            if t["config"] == target:
                return float(t["score"]), "default_config"
        # Default known but not evaluated in this run → fall back
    if not trials:
        raise ValueError("No successful trials for baseline")
    return float(trials[0]["score"]), "first_success"


def normalized_gain(
    best: float,
    baseline: float,
    score_lo: float,
    score_hi: float,
    *,
    higher_is_better: bool = True,
) -> float:
    eps = 1e-12
    if higher_is_better:
        raw = best - baseline
        denom = max(score_hi - baseline, eps)
    else:
        raw = baseline - best
        denom = max(baseline - score_lo, eps)
    return _clip01(raw / denom)


def compute_dynamics(
    trials: Sequence[Dict[str, Any]],
    *,
    higher_is_better: bool = True,
    trend_window: int = 5,
    include_phase_note: bool = True,
) -> Dict[str, Any]:
    if not trials:
        return {
            "trend": "INSUFFICIENT_DATA",
            "stagnation": 0,
            "incumbent_updates": 0,
            "phase_note": None,
        }

    def better(a: float, b: float) -> bool:
        return a > b + 1e-12 if higher_is_better else a < b - 1e-12

    # Incumbent series
    best = trials[0]["score"]
    incumbent_update_trials = [int(trials[0]["trial"])]
    best_so_far: List[float] = []
    for t in trials:
        if better(t["score"], best):
            best = t["score"]
            incumbent_update_trials.append(int(t["trial"]))
        best_so_far.append(best)

    # Stagnation: trials after last improvement
    last_improve_idx = 0
    running = trials[0]["score"]
    for i, t in enumerate(trials):
        if better(t["score"], running):
            running = t["score"]
            last_improve_idx = i
    stagnation = len(trials) - 1 - last_improve_idx

    # Trend on last window (rel to full-run min-max)
    scores = [t["score"] for t in trials]
    lo, hi = min(scores), max(scores)
    span = hi - lo

    def rel(s: float) -> float:
        return 1.0 if span <= 0 else ((s - lo) / span if higher_is_better else (hi - s) / span)

    if len(trials) < 2:
        trend = "INSUFFICIENT_DATA"
    else:
        recent = trials[-trend_window:]
        rels = [rel(t["score"]) for t in recent]
        change = rels[-1] - rels[-2]
        if len(rels) >= 3 and statistics.pstdev(rels) < 0.02:
            trend = "STAGNANT"
        elif change > 0.02:
            trend = "IMPROVING"
        elif change < -0.02:
            trend = "DECLINING"
        else:
            trend = "FLAT"
        if stagnation >= max(5, int(0.15 * len(trials))) and trend != "IMPROVING":
            trend = "STAGNANT"

    phase_note = None
    if include_phase_note:
        n = len(trials)
        warm_end = max(1, int(math.ceil(0.1 * n)))
        last_prog = incumbent_update_trials[-1]
        plateau_len = stagnation
        phase_note = (
            f"warm 1–{warm_end}; progress with {len(incumbent_update_trials)} incumbent "
            f"updates until trial {last_prog}; plateau last {plateau_len} trials"
        )

    return {
        "trend": trend,
        "stagnation": int(stagnation),
        "incumbent_updates": len(incumbent_update_trials),
        "phase_note": phase_note,
        "_incumbent_update_trials": incumbent_update_trials,
    }


# ---------------------------------------------------------------------------
# Knowledge: factors, evidence, takeaways, avoid
# ---------------------------------------------------------------------------


def _cohorts(
    trials: Sequence[Dict[str, Any]],
    *,
    higher_is_better: bool,
    fraction: float = 0.2,
    min_k: int = 3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ordered = sorted(
        trials,
        key=lambda t: t["score"],
        reverse=higher_is_better,
    )
    k = max(min_k, int(math.ceil(fraction * len(ordered))))
    k = min(k, max(1, len(ordered) // 2))
    top = ordered[:k]
    worst = ordered[-k:]
    # Ensure disjoint when possible
    top_ids = {t["trial"] for t in top}
    worst = [t for t in worst if t["trial"] not in top_ids] or ordered[-k:]
    return top, worst


def _effect_threshold(score_lo: float, score_hi: float) -> float:
    span = abs(score_hi - score_lo)
    return max(0.05 * span, 1e-12)


def _factor_kind(values: Sequence[Any]) -> str:
    if all(_is_number(v) for v in values):
        return "numeric_knob"
    return "categorical_knob"


def analyze_factors(
    trials: Sequence[Dict[str, Any]],
    *,
    higher_is_better: bool = True,
    top_fraction: float = 0.2,
) -> Tuple[List[Dict[str, Any]], float]:
    if len(trials) < 4:
        return [], 0.0
    scores = [t["score"] for t in trials]
    score_lo, score_hi = min(scores), max(scores)
    delta = _effect_threshold(score_lo, score_hi)
    top, worst = _cohorts(trials, higher_is_better=higher_is_better, fraction=top_fraction)

    keys: List[str] = sorted({k for t in trials for k in t["config"]})
    raw_rows: List[Dict[str, Any]] = []

    for name in keys:
        vals_all = [t["config"][name] for t in trials if name in t["config"]]
        unique = {json.dumps(v, sort_keys=True, default=str) for v in vals_all}
        if len(unique) == 0:
            status = "NEVER_TESTED"
        elif len(unique) == 1:
            status = "CONSTANT"
        elif len(unique) < 2:
            status = "UNDER_TESTED"
        else:
            status = "TESTED"

        kind = _factor_kind(vals_all) if vals_all else "other"
        role = _infer_role(name)

        top_vals = [t["config"][name] for t in top if name in t["config"]]
        worst_vals = [t["config"][name] for t in worst if name in t["config"]]

        importance_raw = 0.0
        direction = "Insufficient contrast between top and worst cohorts."
        good_region: Any = None
        bad_region: Any = None

        if status == "TESTED" and top_vals and worst_vals:
            if kind == "numeric_knob":
                top_mean = statistics.mean(float(v) for v in top_vals)
                worst_mean = statistics.mean(float(v) for v in worst_vals)
                # Score gap between cohorts (not HP gap)
                top_score_mean = statistics.mean(t["score"] for t in top)
                worst_score_mean = statistics.mean(t["score"] for t in worst)
                score_gap = abs(top_score_mean - worst_score_mean)
                hp_gap = abs(top_mean - worst_mean)
                # Importance mixes score separation with HP separation (normalized later)
                importance_raw = score_gap * (1.0 + hp_gap / (abs(top_mean) + abs(worst_mean) + 1e-12))

                if score_gap >= delta and hp_gap > 0:
                    if top_mean > worst_mean:
                        direction = (
                            f"In this run, higher {name} was over-represented in top trials "
                            f"(top mean={top_mean:.4g}, worst mean={worst_mean:.4g})."
                        )
                        good_region = {"low": float(min(top_vals)), "high": float(max(top_vals))}
                        bad_region = {"low": float(min(worst_vals)), "high": float(max(worst_vals))}
                    else:
                        direction = (
                            f"In this run, lower {name} was over-represented in top trials "
                            f"(top mean={top_mean:.4g}, worst mean={worst_mean:.4g})."
                        )
                        good_region = {"low": float(min(top_vals)), "high": float(max(top_vals))}
                        bad_region = {"low": float(min(worst_vals)), "high": float(max(worst_vals))}
                else:
                    direction = f"No clear top-vs-worst separation for {name} (score gap < δ)."
                    importance_raw = 0.0
                    good_region = bad_region = None
            else:
                top_c = Counter(json.dumps(v, sort_keys=True, default=str) for v in top_vals)
                worst_c = Counter(json.dumps(v, sort_keys=True, default=str) for v in worst_vals)
                # Score means per category over all trials
                by_cat: Dict[str, List[float]] = defaultdict(list)
                for t in trials:
                    if name not in t["config"]:
                        continue
                    key = json.dumps(t["config"][name], sort_keys=True, default=str)
                    by_cat[key].append(float(t["score"]))
                cat_means = {k: statistics.mean(vs) for k, vs in by_cat.items() if vs}
                if len(cat_means) >= 2:
                    best_cat = max(cat_means, key=cat_means.get) if higher_is_better else min(cat_means, key=cat_means.get)
                    worst_cat = min(cat_means, key=cat_means.get) if higher_is_better else max(cat_means, key=cat_means.get)
                    score_gap = abs(cat_means[best_cat] - cat_means[worst_cat])
                    importance_raw = score_gap
                    if score_gap >= delta:
                        best_val = json.loads(best_cat)
                        worst_val = json.loads(worst_cat)
                        direction = (
                            f"In this run, {name}={best_val!r} associated with better scores "
                            f"than {worst_val!r}."
                        )
                        good_region = [best_val]
                        bad_region = [worst_val]
                    else:
                        direction = f"No clear categorical separation for {name}."
                        importance_raw = 0.0

        raw_rows.append(
            {
                "name": name,
                "kind": kind,
                "role": role,
                "status": status,
                "importance_raw": importance_raw,
                "method": "top_worst_gap",
                "direction": direction,
                "good_region": good_region,
                "bad_region": bad_region,
                "evidence_ids": [],
                "_top_trials": [int(t["trial"]) for t in top],
                "_worst_trials": [int(t["trial"]) for t in worst],
                "_score_gap": abs(
                    statistics.mean(t["score"] for t in top) - statistics.mean(t["score"] for t in worst)
                )
                if top and worst
                else 0.0,
            }
        )

    total = sum(r["importance_raw"] for r in raw_rows) or 1.0
    factors: List[Dict[str, Any]] = []
    for r in sorted(raw_rows, key=lambda x: x["importance_raw"], reverse=True):
        factors.append(
            {
                "name": r["name"],
                "kind": r["kind"],
                "role": r["role"],
                "status": r["status"],
                "importance": round(r["importance_raw"] / total, 4),
                "method": r["method"],
                "direction": r["direction"],
                "good_region": r["good_region"],
                "bad_region": r["bad_region"],
                "evidence_ids": [],
                "_top_trials": r["_top_trials"],
                "_worst_trials": r["_worst_trials"],
                "_score_gap": r["_score_gap"],
            }
        )
    return factors, delta


def build_evidence(
    trials: Sequence[Dict[str, Any]],
    factors: Sequence[Dict[str, Any]],
    dynamics: Dict[str, Any],
    *,
    higher_is_better: bool,
    delta: float,
) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    eid = 0

    def next_id() -> str:
        nonlocal eid
        eid += 1
        return f"ev_{eid:03d}"

    # Incumbent jumps
    update_trials = list(dynamics.get("_incumbent_update_trials") or [])
    by_id = {int(t["trial"]): t for t in trials}
    for a, b in zip(update_trials, update_trials[1:]):
        if a not in by_id or b not in by_id:
            continue
        prev, curr = by_id[a], by_id[b]
        changed = [
            k
            for k in sorted(set(prev["config"]) | set(curr["config"]))
            if prev["config"].get(k) != curr["config"].get(k)
        ]
        if not changed:
            continue
        dscore = float(curr["score"] - prev["score"])
        if not higher_is_better:
            dscore = -dscore
        ev_id = next_id()
        evidence.append(
            {
                "id": ev_id,
                "claim": (
                    f"Incumbent jump trial {a}→{b}: changed {', '.join(changed)}; "
                    f"score {prev['score']:.4g}→{curr['score']:.4g}."
                ),
                "kind": "incumbent_jump",
                "factors": changed,
                "trials": [a, b],
                "delta": round(dscore, 6),
                "confidence": _clip01(0.55 + min(0.35, abs(dscore) / (abs(delta) * 5 + 1e-12))),
            }
        )

    # Top vs worst per strong factor
    for fac in factors:
        if fac.get("status") != "TESTED" or fac.get("importance", 0) <= 0:
            continue
        if fac.get("good_region") is None and fac.get("bad_region") is None:
            continue
        if fac.get("_score_gap", 0) < delta:
            continue
        ev_id = next_id()
        evidence.append(
            {
                "id": ev_id,
                "claim": fac["direction"],
                "kind": "top_vs_worst",
                "factors": [fac["name"]],
                "trials": sorted(set(fac.get("_top_trials", [])[:5] + fac.get("_worst_trials", [])[:3])),
                "delta": round(float(fac.get("_score_gap", 0.0)), 6),
                "confidence": _clip01(0.5 + min(0.4, fac["importance"])),
            }
        )
        fac["evidence_ids"].append(ev_id)

    return evidence


def build_takeaways(
    factors: Sequence[Dict[str, Any]],
    evidence: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    by_id = {e["id"]: e for e in evidence}
    worked: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    transfer: List[Dict[str, Any]] = []

    for fac in factors:
        if fac.get("status") != "TESTED" or not fac.get("evidence_ids"):
            continue
        confs = [by_id[i]["confidence"] for i in fac["evidence_ids"] if i in by_id]
        conf = statistics.mean(confs) if confs else 0.0
        if conf < 0.6 or fac.get("importance", 0) <= 0:
            continue
        eids = list(fac["evidence_ids"])
        if fac.get("good_region") is not None:
            worked.append(
                {
                    "text": f"Prefer {fac['name']} in {fac['good_region']!s} ({fac['direction']})",
                    "confidence": round(conf, 3),
                    "evidence_ids": eids,
                    "scope": "same_algorithm",
                }
            )
        if fac.get("bad_region") is not None:
            failed.append(
                {
                    "text": f"Avoid {fac['name']} in {fac['bad_region']!s} on this run.",
                    "confidence": round(conf, 3),
                    "evidence_ids": eids,
                    "scope": "same_algorithm",
                }
            )
        # Transfer only for strong, role-labeled, non-tiny factors
        if fac.get("role") and fac.get("importance", 0) >= 0.15 and conf >= 0.65:
            transfer.append(
                {
                    "text": (
                        f"Role `{fac['role']}` ({fac['name']}) showed top-vs-worst association "
                        f"in this run; treat as a candidate priority knob on similar tasks."
                    ),
                    "confidence": round(conf * 0.9, 3),
                    "evidence_ids": eids,
                    "scope": "same_domain",
                }
            )

    return {
        "worked": worked[:5],
        "failed": failed[:5],
        "transfer": transfer[:3],
    }


def build_avoid(
    trials: Sequence[Dict[str, Any]],
    factors: Sequence[Dict[str, Any]],
    *,
    baseline_score: float,
    score_lo: float,
    score_hi: float,
    delta: float,
    higher_is_better: bool,
    max_items: int = 3,
) -> List[Dict[str, Any]]:
    span = abs(score_hi - score_lo)
    poor_cut = (
        baseline_score
        if higher_is_better
        else baseline_score
    )
    # Also allow "near floor"
    floor = score_lo + 0.2 * span if higher_is_better else score_hi - 0.2 * span

    def is_poor(mean_score: float) -> bool:
        if higher_is_better:
            return mean_score <= poor_cut or mean_score <= floor
        return mean_score >= poor_cut or mean_score >= floor

    avoid: List[Dict[str, Any]] = []
    for fac in factors:
        if fac.get("bad_region") is None or fac.get("status") != "TESTED":
            continue
        if fac.get("_score_gap", 0) < delta:
            continue
        worst_ids = set(fac.get("_worst_trials") or [])
        if len(worst_ids) < 2:
            continue
        worst_trials = [t for t in trials if t["trial"] in worst_ids]
        mean_score = statistics.mean(t["score"] for t in worst_trials)
        if not is_poor(mean_score):
            continue
        avoid.append(
            {
                "text": f"Harmful association for {fac['name']}: {fac['bad_region']!s}",
                "region": {fac["name"]: fac["bad_region"]},
                "trials": sorted(worst_ids)[:8],
            }
        )
        if len(avoid) >= max_items:
            break
    return avoid


# ---------------------------------------------------------------------------
# Quality + render
# ---------------------------------------------------------------------------


def compute_quality(
    trials: Sequence[Dict[str, Any]],
    factors: Sequence[Dict[str, Any]],
    evidence: Sequence[Dict[str, Any]],
    dynamics: Dict[str, Any],
    n_total_including_failed: Optional[int] = None,
) -> Dict[str, Any]:
    n = len(trials)
    n_tot = n_total_including_failed or n
    success_rate = n / max(n_tot, 1)
    tested = [f for f in factors if f.get("status") == "TESTED"]
    coverage = len(tested) / max(len(factors), 1) if factors else 0.0
    log_n = _clip01(math.log1p(n) / math.log1p(100))
    stag = dynamics.get("stagnation", 0)
    stag_pen = _clip01(stag / max(n, 1))
    fidelity = _clip01(0.35 * success_rate + 0.25 * coverage + 0.20 * log_n + 0.20 * (1 - stag_pen))

    supported = sum(1 for e in evidence if e.get("trials"))
    total_e = max(len(evidence), 1)
    effect = statistics.mean([abs(e.get("delta") or 0) for e in evidence]) if evidence else 0.0
    # Normalize effect loosely
    effect_score = _clip01(effect * 10) if evidence else 0.0
    conf = _clip01(0.40 * (supported / total_e) + 0.30 * effect_score + 0.20 * 0.7 + 0.10 * coverage)
    overall = round(0.5 * fidelity + 0.5 * conf, 4)
    return {
        "fidelity": round(fidelity, 4),
        "confidence": round(conf, 4),
        "overall": overall,
        "note": f"{n}/{n_tot} success; {len(tested)}/{len(factors)} factors TESTED; stagnation={stag}",
    }


def render_embed_text(problem: Dict[str, Any], knowledge: Dict[str, Any], result: Dict[str, Any]) -> str:
    top = sorted(knowledge.get("factors") or [], key=lambda f: f.get("importance", 0), reverse=True)[:3]
    fac_bits = "; ".join(f"{f['name']}({f['importance']:.2f})" for f in top) or "none"
    avoid = knowledge.get("avoid") or []
    avoid_bit = avoid[0]["text"] if avoid else "none"
    dyn = result.get("dynamics") or {}
    return (
        f"{problem.get('context', '')} "
        f"gain={result.get('normalized_gain')}; "
        f"incumbent_updates={dyn.get('incumbent_updates')}; "
        f"trend={dyn.get('trend')}; stagnant={dyn.get('stagnation')}. "
        f"Top factors: {fac_bits}. Avoid: {avoid_bit}."
    ).strip()


def render_prompt_block(
    problem: Dict[str, Any],
    knowledge: Dict[str, Any],
    result: Dict[str, Any],
    quality: Dict[str, Any],
) -> str:
    top = sorted(knowledge.get("factors") or [], key=lambda f: f.get("importance", 0), reverse=True)[:3]
    lines = [
        f"EXPERIENCE ({problem.get('dataset_id')}/{problem.get('algorithm_id')}, q={quality.get('overall')}):",
        f"- gain={result.get('normalized_gain')} over baseline ({result.get('baseline_kind')}); "
        f"{result.get('dynamics', {}).get('incumbent_updates')} incumbent updates; "
        f"ended {result.get('dynamics', {}).get('trend')}, stagnant {result.get('dynamics', {}).get('stagnation')}.",
    ]
    if result.get("dynamics", {}).get("phase_note"):
        lines.append(f"- curve: {result['dynamics']['phase_note']}")
    for f in top:
        if f.get("importance", 0) <= 0:
            continue
        lines.append(f"- factor {f['name']}: {f['direction']}")
    for a in (knowledge.get("avoid") or [])[:2]:
        lines.append(f"- avoid: {a['text']}")
    return "\n".join(lines)


def _strip_private(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_private(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [_strip_private(x) for x in obj]
    return obj


def _solution_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "config",
        "config": _jsonable(cfg),
        "vector": None,
        "sequence": None,
        "artifacts": None,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_experience_from_trials(
    trials: Sequence[Dict[str, Any]],
    *,
    run_uri: str,
    optimizer: str = "unknown",
    domain: str = "tabular_ml",
    task_type: str = "classification",
    dataset_id: Optional[str] = None,
    algorithm_id: Optional[str] = None,
    metric: str = "bal_acc",
    higher_is_better: bool = True,
    context: Optional[str] = None,
    profile: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    baseline_score: Optional[float] = None,
    baseline_kind: Optional[str] = None,
    default_config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    include_phase_note: bool = True,
    n_failed: int = 0,
    experience_id: Optional[str] = None,
) -> Dict[str, Any]:
    if not trials:
        raise ValueError("Need at least one successful trial")

    scores = [t["score"] for t in trials]
    score_lo, score_hi = min(scores), max(scores)
    best_trial = max(trials, key=lambda t: t["score"]) if higher_is_better else min(trials, key=lambda t: t["score"])

    base_score, base_kind = compute_baseline(
        trials,
        baseline_score=baseline_score,
        baseline_kind=baseline_kind,
        default_config=default_config,
    )
    gain = normalized_gain(
        best_trial["score"], base_score, score_lo, score_hi, higher_is_better=higher_is_better
    )
    dynamics = compute_dynamics(trials, higher_is_better=higher_is_better, include_phase_note=include_phase_note)

    factors, delta = analyze_factors(trials, higher_is_better=higher_is_better)
    evidence = build_evidence(trials, factors, dynamics, higher_is_better=higher_is_better, delta=delta)
    takeaways = build_takeaways(factors, evidence)
    avoid = build_avoid(
        trials,
        factors,
        baseline_score=base_score,
        score_lo=score_lo,
        score_hi=score_hi,
        delta=delta,
        higher_is_better=higher_is_better,
    )
    quality = compute_quality(trials, factors, evidence, dynamics, n_total_including_failed=len(trials) + n_failed)

    # Space fingerprint from observed keys
    keys = sorted({k for t in trials for k in t["config"]})
    space_factors = []
    for name in keys:
        vals = [t["config"][name] for t in trials if name in t["config"]]
        kind = _factor_kind(vals) if vals else "other"
        space_factors.append(
            {
                "name": name,
                "kind": kind,
                "role": _infer_role(name),
                "type": "float" if kind == "numeric_knob" else "categorical",
                "log_scale": None,
                "bounds": None,
            }
        )
    space_hash = _stable_hash({"keys": keys, "algorithm": algorithm_id})

    algo = algorithm_id or (trials[0]["full_config"].get("algorithm") if trials[0].get("full_config") else None)
    ds = dataset_id or "unknown"
    ctx = context or (
        f"{task_type} on {ds}"
        + (f" with {algo}" if algo else "")
        + f"; optimize {metric} ({'higher' if higher_is_better else 'lower'} is better)."
    )

    problem = {
        "domain": domain,
        "task_type": task_type,
        "dataset_id": ds,
        "algorithm_id": algo,
        "context": ctx,
        "metric": metric,
        "higher_is_better": higher_is_better,
        "profile": profile or {},
        "tags": tags or [],
        "representation": "structured_config",
        "space": {
            "space_hash": space_hash,
            "dim": len(keys),
            "decoder": None,
            "bounds": None,
            "factors": space_factors,
            "space_text": ", ".join(keys),
        },
    }

    result = {
        "n_trials": len(trials) + n_failed,
        "n_success": len(trials),
        "n_failed": n_failed,
        "score_range": {"lo": score_lo, "hi": score_hi},
        "best": {
            "trial": int(best_trial["trial"]),
            "score": float(best_trial["score"]),
            "solution": _solution_from_config(best_trial["config"]),
        },
        "baseline_score": float(base_score),
        "baseline_kind": base_kind,
        "normalized_gain": round(gain, 4),
        "dynamics": {
            "trend": dynamics["trend"],
            "stagnation": dynamics["stagnation"],
            "incumbent_updates": dynamics["incumbent_updates"],
            "phase_note": dynamics.get("phase_note"),
        },
    }

    knowledge = {
        "factors": [_strip_private(f) for f in factors if f.get("importance", 0) > 0 or f.get("status") == "TESTED"],
        "evidence": evidence,
        "takeaways": takeaways,
        "avoid": avoid,
    }
    # Keep top factors only in delivery render; store all TESTED with signal
    knowledge["factors"] = [
        f for f in knowledge["factors"] if f.get("importance", 0) > 0 or f.get("good_region") is not None
    ][:12]

    embed = render_embed_text(problem, knowledge, result)
    prompt = render_prompt_block(problem, knowledge, result, quality)

    # Train good/bad from top/worst configs
    top, worst = _cohorts(trials, higher_is_better=higher_is_better)
    train = {
        "best": _solution_from_config(best_trial["config"]),
        "good": [_solution_from_config(t["config"]) for t in top[:5]],
        "bad": [_solution_from_config(t["config"]) for t in worst[:5]],
    }

    exp_id = experience_id or f"exp_{ds}_{algo or 'algo'}_{space_hash}"
    return {
        "schema_version": "experience.v2",
        "id": exp_id,
        "created_at": _utc_now(),
        "meta": {
            "run_uri": run_uri,
            "optimizer": optimizer,
            "seed": seed,
            "budget_trials": len(trials) + n_failed,
            "summarizer": {
                "facts": "top_worst_gap+incumbent_jumps+curve_dynamics",
                "llm": None,
                "template": "experience_builder.v2",
            },
        },
        "problem": problem,
        "result": result,
        "knowledge": knowledge,
        "delivery": {
            "quality": quality,
            "embed_text": embed,
            "filters": {
                "domain": domain,
                "task_type": task_type,
                "representation": "structured_config",
                "algorithm_id": algo,
                "space_hash": space_hash,
            },
            "prompt_block": prompt,
            "train": train,
        },
    }


def build_experience_from_bo_json(path: Path, **kwargs: Any) -> Dict[str, Any]:
    path = Path(path)
    # Infer ids from .../cls/bal_acc/{dataset}/{algo}.json
    parts = path.parts
    dataset_id = kwargs.pop("dataset_id", None)
    algorithm_id = kwargs.pop("algorithm_id", None)
    task_type = kwargs.get("task_type", "classification")
    metric = kwargs.get("metric", "bal_acc")
    domain = kwargs.get("domain", "tabular_ml")
    if dataset_id is None or algorithm_id is None:
        try:
            algorithm_id = algorithm_id or path.stem
            dataset_id = dataset_id or path.parent.name
            metric = path.parent.parent.name
            tt = path.parent.parent.parent.name
            task_type = "classification" if tt == "cls" else ("regression" if tt == "rgs" else task_type)
            kwargs["metric"] = metric
            kwargs["task_type"] = task_type
        except Exception:
            pass

    higher_is_better = kwargs.pop("higher_is_better", metric not in {"mse", "mae", "rmse", "latency_ms"})
    openbox_minimize = kwargs.pop("openbox_minimize", True)
    algorithm = kwargs.get("algorithm_id") or algorithm_id

    # Count failures
    raw = json.loads(path.read_text(encoding="utf-8"))
    n_failed = sum(1 for row in (raw.get("observations") or []) if int(row.get("trial_state", 0)) != 0)

    trials = load_trials_from_bo_json(
        path,
        algorithm=algorithm,
        higher_is_better=True if higher_is_better else False,
        openbox_minimize=openbox_minimize,
    )
    # If lower-is-better metric stored as minimize already, load_trials with higher_is_better False
    # uses raw objective; for mse openbox may store positive mse to minimize — handle via flag.
    if not higher_is_better:
        trials = load_trials_from_bo_json(
            path, algorithm=algorithm, higher_is_better=False, openbox_minimize=True
        )

    return build_experience_from_trials(
        trials,
        run_uri=str(path),
        dataset_id=dataset_id,
        algorithm_id=algorithm_id,
        higher_is_better=higher_is_better,
        n_failed=n_failed,
        domain=domain,
        **kwargs,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build Experience v2 JSON from a BO history file")
    p.add_argument("history_json", type=Path, help="OpenBox-style observations JSON")
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--optimizer", default="bo_history")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--baseline-score", type=float, default=None)
    p.add_argument("--baseline-kind", default=None)
    p.add_argument("--no-phase-note", action="store_true")
    p.add_argument("--validate", action="store_true", help="Validate against experience.v2 schema")
    args = p.parse_args(argv)

    exp = build_experience_from_bo_json(
        args.history_json,
        optimizer=args.optimizer,
        seed=args.seed,
        baseline_score=args.baseline_score,
        baseline_kind=args.baseline_kind,
        include_phase_note=not args.no_phase_note,
    )

    if args.validate:
        try:
            import jsonschema
        except ImportError as e:
            raise SystemExit("jsonschema not installed") from e
        schema_path = Path(__file__).resolve().parents[1] / "schema" / "experience.v2.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.validate(exp, schema)

    text = json.dumps(exp, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Deterministic Experience v2 builder from BO / HPO trial histories.

Locked rules:
- incumbent_updates = improvement-event count (no separate field)
- phase_note optional, curve template only (not optimizer modes)
- no knowledge.summary — render embed_text / prompt_block from fields
- factor good/bad via top-vs-worst cohort association + δ threshold
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SKIP_KEYS = {"algorithm"}

ROLE_HINTS = (
    ("learning_rate", "learning_rate"),
    ("lr", "learning_rate"),
    ("l2", "regularization"),
    ("reg_lambda", "regularization"),
    ("reg_alpha", "regularization"),
    ("reg", "regularization"),
    ("alpha", "regularization"),
    ("C", "regularization"),
    ("min_child_weight", "regularization"),
    ("num_leaves", "tree_depth"),
    ("max_depth", "tree_depth"),
    ("depth", "tree_depth"),
    ("n_estimators", "n_estimators"),
    ("estimators", "n_estimators"),
    ("colsample", "feature_fraction"),
    ("max_features", "feature_fraction"),
    ("n_neighbors", "n_neighbors"),
    ("min_samples_leaf", "leaf_size"),
    ("min_samples_split", "leaf_size"),
    ("subsample", "sampling"),
    ("batch", "batch_or_budget"),
    ("work_mem", "batch_or_budget"),
    ("bootstrap", "sampling"),
    ("criterion", "optimizer_choice"),
    ("algorithm", "optimizer_choice"),
    ("weights", "distance_weight"),
    ("p", "minkowski_p"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _jsonable(v: Any) -> Any:
    if isinstance(v, float):
        return float(v)
    if isinstance(v, (int, str, bool)) or v is None:
        return v
    return json.loads(json.dumps(v, default=str))


def _stable_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _infer_role(name: str) -> Optional[str]:
    raw = name.split(":")[-1] if ":" in name else name
    low = raw.lower()
    for needle, role in sorted(ROLE_HINTS, key=lambda kv: -len(kv[0])):
        n = needle.lower()
        if n == "c":
            if raw == "C" or low in {"c", "c_"} or low.endswith("_c") or low.startswith("c_"):
                return role
            continue
        if len(n) == 1:
            if low == n:
                return role
            continue
        if n in low:
            return role
    return None


def _tunable_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in cfg.items() if k not in SKIP_KEYS}


# ---------------------------------------------------------------------------
# Load histories
# ---------------------------------------------------------------------------


def load_trials_from_bo_json(
    path: Path,
    *,
    algorithm: Optional[str] = None,
    higher_is_better: bool = True,
    openbox_minimize: bool = True,
) -> List[Dict[str, Any]]:
    """Load OpenBox-style history into trial dicts with higher-is-better scores."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    observations = list(obj.get("observations") or [])
    trials: List[Dict[str, Any]] = []
    for i, row in enumerate(observations, start=1):
        state = row.get("trial_state", 0)
        if state is not None and int(state) != 0:
            continue
        cfg = dict(row.get("config") or {})
        if algorithm and cfg.get("algorithm") not in (None, algorithm):
            continue
        objectives = row.get("objectives") or []
        if not objectives:
            continue
        raw = float(objectives[0])
        if not math.isfinite(raw):
            continue
        # OpenBox stores minimize objectives; MindWare uses higher-is-better metric.
        score = (-raw if openbox_minimize else raw)
        if not higher_is_better:
            # If caller says lower-is-better and openbox already minimizes, score=raw.
            score = raw if openbox_minimize else -raw
        trials.append(
            {
                "trial": i,
                "config": _tunable_config(cfg),
                "full_config": cfg,
                "score": float(score),
                "raw_objective": raw,
            }
        )
    # Re-number by successful order for stable trial ids in experience
    for j, t in enumerate(trials, start=1):
        t["trial"] = j
    return trials


def load_trials_from_pairs(
    configs: Sequence[Dict[str, Any]],
    perfs: Sequence[float],
) -> List[Dict[str, Any]]:
    trials: List[Dict[str, Any]] = []
    for i, (cfg, perf) in enumerate(zip(configs, perfs), start=1):
        score = float(perf)
        if not math.isfinite(score):
            continue
        trials.append(
            {
                "trial": i,
                "config": _tunable_config(dict(cfg)),
                "full_config": dict(cfg),
                "score": score,
            }
        )
    return trials


# ---------------------------------------------------------------------------
# Result: baseline, gain, dynamics
# ---------------------------------------------------------------------------


def compute_baseline(
    trials: Sequence[Dict[str, Any]],
    *,
    baseline_score: Optional[float] = None,
    baseline_kind: Optional[str] = None,
    default_config: Optional[Dict[str, Any]] = None,
) -> Tuple[float, str]:
    if baseline_score is not None:
        return float(baseline_score), baseline_kind or "provided"
    if default_config is not None:
        target = _tunable_config(default_config)
        for t in trials:
            if t["config"] == target:
                return float(t["score"]), "default_config"
        # Default known but not evaluated in this run → fall back
    if not trials:
        raise ValueError("No successful trials for baseline")
    return float(trials[0]["score"]), "first_success"


def normalized_gain(
    best: float,
    baseline: float,
    score_lo: float,
    score_hi: float,
    *,
    higher_is_better: bool = True,
) -> float:
    eps = 1e-12
    if higher_is_better:
        raw = best - baseline
        denom = max(score_hi - baseline, eps)
    else:
        raw = baseline - best
        denom = max(baseline - score_lo, eps)
    return _clip01(raw / denom)


def compute_dynamics(
    trials: Sequence[Dict[str, Any]],
    *,
    higher_is_better: bool = True,
    trend_window: int = 5,
    include_phase_note: bool = True,
) -> Dict[str, Any]:
    if not trials:
        return {
            "trend": "INSUFFICIENT_DATA",
            "stagnation": 0,
            "incumbent_updates": 0,
            "phase_note": None,
        }

    def better(a: float, b: float) -> bool:
        return a > b + 1e-12 if higher_is_better else a < b - 1e-12

    # Incumbent series
    best = trials[0]["score"]
    incumbent_update_trials = [int(trials[0]["trial"])]
    best_so_far: List[float] = []
    for t in trials:
        if better(t["score"], best):
            best = t["score"]
            incumbent_update_trials.append(int(t["trial"]))
        best_so_far.append(best)

    # Stagnation: trials after last improvement
    last_improve_idx = 0
    running = trials[0]["score"]
    for i, t in enumerate(trials):
        if better(t["score"], running):
            running = t["score"]
            last_improve_idx = i
    stagnation = len(trials) - 1 - last_improve_idx

    # Trend on last window (rel to full-run min-max)
    scores = [t["score"] for t in trials]
    lo, hi = min(scores), max(scores)
    span = hi - lo

    def rel(s: float) -> float:
        return 1.0 if span <= 0 else ((s - lo) / span if higher_is_better else (hi - s) / span)

    if len(trials) < 2:
        trend = "INSUFFICIENT_DATA"
    else:
        recent = trials[-trend_window:]
        rels = [rel(t["score"]) for t in recent]
        change = rels[-1] - rels[-2]
        if len(rels) >= 3 and statistics.pstdev(rels) < 0.02:
            trend = "STAGNANT"
        elif change > 0.02:
            trend = "IMPROVING"
        elif change < -0.02:
            trend = "DECLINING"
        else:
            trend = "FLAT"
        if stagnation >= max(5, int(0.15 * len(trials))) and trend != "IMPROVING":
            trend = "STAGNANT"

    phase_note = None
    if include_phase_note:
        n = len(trials)
        warm_end = max(1, int(math.ceil(0.1 * n)))
        last_prog = incumbent_update_trials[-1]
        plateau_len = stagnation
        phase_note = (
            f"warm 1–{warm_end}; progress with {len(incumbent_update_trials)} incumbent "
            f"updates until trial {last_prog}; plateau last {plateau_len} trials"
        )

    return {
        "trend": trend,
        "stagnation": int(stagnation),
        "incumbent_updates": len(incumbent_update_trials),
        "phase_note": phase_note,
        "_incumbent_update_trials": incumbent_update_trials,
    }


# ---------------------------------------------------------------------------
# Knowledge: factors, evidence, takeaways, avoid
# ---------------------------------------------------------------------------


def _cohorts(
    trials: Sequence[Dict[str, Any]],
    *,
    higher_is_better: bool,
    fraction: float = 0.2,
    min_k: int = 3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ordered = sorted(
        trials,
        key=lambda t: t["score"],
        reverse=higher_is_better,
    )
    k = max(min_k, int(math.ceil(fraction * len(ordered))))
    k = min(k, max(1, len(ordered) // 2))
    top = ordered[:k]
    worst = ordered[-k:]
    # Ensure disjoint when possible
    top_ids = {t["trial"] for t in top}
    worst = [t for t in worst if t["trial"] not in top_ids] or ordered[-k:]
    return top, worst


def _effect_threshold(score_lo: float, score_hi: float) -> float:
    span = abs(score_hi - score_lo)
    return max(0.05 * span, 1e-12)


def _factor_kind(values: Sequence[Any]) -> str:
    if all(_is_number(v) for v in values):
        return "numeric_knob"
    return "categorical_knob"


def analyze_factors(
    trials: Sequence[Dict[str, Any]],
    *,
    higher_is_better: bool = True,
    top_fraction: float = 0.2,
) -> Tuple[List[Dict[str, Any]], float]:
    if len(trials) < 4:
        return [], 0.0
    scores = [t["score"] for t in trials]
    score_lo, score_hi = min(scores), max(scores)
    delta = _effect_threshold(score_lo, score_hi)
    top, worst = _cohorts(trials, higher_is_better=higher_is_better, fraction=top_fraction)

    keys: List[str] = sorted({k for t in trials for k in t["config"]})
    raw_rows: List[Dict[str, Any]] = []

    for name in keys:
        vals_all = [t["config"][name] for t in trials if name in t["config"]]
        unique = {json.dumps(v, sort_keys=True, default=str) for v in vals_all}
        if len(unique) == 0:
            status = "NEVER_TESTED"
        elif len(unique) == 1:
            status = "CONSTANT"
        elif len(unique) < 2:
            status = "UNDER_TESTED"
        else:
            status = "TESTED"

        kind = _factor_kind(vals_all) if vals_all else "other"
        role = _infer_role(name)

        top_vals = [t["config"][name] for t in top if name in t["config"]]
        worst_vals = [t["config"][name] for t in worst if name in t["config"]]

        importance_raw = 0.0
        direction = "Insufficient contrast between top and worst cohorts."
        good_region: Any = None
        bad_region: Any = None

        if status == "TESTED" and top_vals and worst_vals:
            if kind == "numeric_knob":
                top_mean = statistics.mean(float(v) for v in top_vals)
                worst_mean = statistics.mean(float(v) for v in worst_vals)
                # Score gap between cohorts (not HP gap)
                top_score_mean = statistics.mean(t["score"] for t in top)
                worst_score_mean = statistics.mean(t["score"] for t in worst)
                score_gap = abs(top_score_mean - worst_score_mean)
                hp_gap = abs(top_mean - worst_mean)
                # Importance mixes score separation with HP separation (normalized later)
                importance_raw = score_gap * (1.0 + hp_gap / (abs(top_mean) + abs(worst_mean) + 1e-12))

                if score_gap >= delta and hp_gap > 0:
                    if top_mean > worst_mean:
                        direction = (
                            f"In this run, higher {name} was over-represented in top trials "
                            f"(top mean={top_mean:.4g}, worst mean={worst_mean:.4g})."
                        )
                        good_region = {"low": float(min(top_vals)), "high": float(max(top_vals))}
                        bad_region = {"low": float(min(worst_vals)), "high": float(max(worst_vals))}
                    else:
                        direction = (
                            f"In this run, lower {name} was over-represented in top trials "
                            f"(top mean={top_mean:.4g}, worst mean={worst_mean:.4g})."
                        )
                        good_region = {"low": float(min(top_vals)), "high": float(max(top_vals))}
                        bad_region = {"low": float(min(worst_vals)), "high": float(max(worst_vals))}
                else:
                    direction = f"No clear top-vs-worst separation for {name} (score gap < δ)."
                    importance_raw = 0.0
                    good_region = bad_region = None
            else:
                top_c = Counter(json.dumps(v, sort_keys=True, default=str) for v in top_vals)
                worst_c = Counter(json.dumps(v, sort_keys=True, default=str) for v in worst_vals)
                # Score means per category over all trials
                by_cat: Dict[str, List[float]] = defaultdict(list)
                for t in trials:
                    if name not in t["config"]:
                        continue
                    key = json.dumps(t["config"][name], sort_keys=True, default=str)
                    by_cat[key].append(float(t["score"]))
                cat_means = {k: statistics.mean(vs) for k, vs in by_cat.items() if vs}
                if len(cat_means) >= 2:
                    best_cat = max(cat_means, key=cat_means.get) if higher_is_better else min(cat_means, key=cat_means.get)
                    worst_cat = min(cat_means, key=cat_means.get) if higher_is_better else max(cat_means, key=cat_means.get)
                    score_gap = abs(cat_means[best_cat] - cat_means[worst_cat])
                    importance_raw = score_gap
                    if score_gap >= delta:
                        best_val = json.loads(best_cat)
                        worst_val = json.loads(worst_cat)
                        direction = (
                            f"In this run, {name}={best_val!r} associated with better scores "
                            f"than {worst_val!r}."
                        )
                        good_region = [best_val]
                        bad_region = [worst_val]
                    else:
                        direction = f"No clear categorical separation for {name}."
                        importance_raw = 0.0

        raw_rows.append(
            {
                "name": name,
                "kind": kind,
                "role": role,
                "status": status,
                "importance_raw": importance_raw,
                "method": "top_worst_gap",
                "direction": direction,
                "good_region": good_region,
                "bad_region": bad_region,
                "evidence_ids": [],
                "_top_trials": [int(t["trial"]) for t in top],
                "_worst_trials": [int(t["trial"]) for t in worst],
                "_score_gap": abs(
                    statistics.mean(t["score"] for t in top) - statistics.mean(t["score"] for t in worst)
                )
                if top and worst
                else 0.0,
            }
        )

    total = sum(r["importance_raw"] for r in raw_rows) or 1.0
    factors: List[Dict[str, Any]] = []
    for r in sorted(raw_rows, key=lambda x: x["importance_raw"], reverse=True):
        factors.append(
            {
                "name": r["name"],
                "kind": r["kind"],
                "role": r["role"],
                "status": r["status"],
                "importance": round(r["importance_raw"] / total, 4),
                "method": r["method"],
                "direction": r["direction"],
                "good_region": r["good_region"],
                "bad_region": r["bad_region"],
                "evidence_ids": [],
                "_top_trials": r["_top_trials"],
                "_worst_trials": r["_worst_trials"],
                "_score_gap": r["_score_gap"],
            }
        )
    return factors, delta


def build_evidence(
    trials: Sequence[Dict[str, Any]],
    factors: Sequence[Dict[str, Any]],
    dynamics: Dict[str, Any],
    *,
    higher_is_better: bool,
    delta: float,
) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    eid = 0

    def next_id() -> str:
        nonlocal eid
        eid += 1
        return f"ev_{eid:03d}"

    # Incumbent jumps
    update_trials = list(dynamics.get("_incumbent_update_trials") or [])
    by_id = {int(t["trial"]): t for t in trials}
    for a, b in zip(update_trials, update_trials[1:]):
        if a not in by_id or b not in by_id:
            continue
        prev, curr = by_id[a], by_id[b]
        changed = [
            k
            for k in sorted(set(prev["config"]) | set(curr["config"]))
            if prev["config"].get(k) != curr["config"].get(k)
        ]
        if not changed:
            continue
        dscore = float(curr["score"] - prev["score"])
        if not higher_is_better:
            dscore = -dscore
        ev_id = next_id()
        evidence.append(
            {
                "id": ev_id,
                "claim": (
                    f"Incumbent jump trial {a}→{b}: changed {', '.join(changed)}; "
                    f"score {prev['score']:.4g}→{curr['score']:.4g}."
                ),
                "kind": "incumbent_jump",
                "factors": changed,
                "trials": [a, b],
                "delta": round(dscore, 6),
                "confidence": _clip01(0.55 + min(0.35, abs(dscore) / (abs(delta) * 5 + 1e-12))),
            }
        )

    # Top vs worst per strong factor
    for fac in factors:
        if fac.get("status") != "TESTED" or fac.get("importance", 0) <= 0:
            continue
        if fac.get("good_region") is None and fac.get("bad_region") is None:
            continue
        if fac.get("_score_gap", 0) < delta:
            continue
        ev_id = next_id()
        evidence.append(
            {
                "id": ev_id,
                "claim": fac["direction"],
                "kind": "top_vs_worst",
                "factors": [fac["name"]],
                "trials": sorted(set(fac.get("_top_trials", [])[:5] + fac.get("_worst_trials", [])[:3])),
                "delta": round(float(fac.get("_score_gap", 0.0)), 6),
                "confidence": _clip01(0.5 + min(0.4, fac["importance"])),
            }
        )
        fac["evidence_ids"].append(ev_id)

    return evidence


def build_takeaways(
    factors: Sequence[Dict[str, Any]],
    evidence: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    by_id = {e["id"]: e for e in evidence}
    worked: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    transfer: List[Dict[str, Any]] = []

    for fac in factors:
        if fac.get("status") != "TESTED" or not fac.get("evidence_ids"):
            continue
        confs = [by_id[i]["confidence"] for i in fac["evidence_ids"] if i in by_id]
        conf = statistics.mean(confs) if confs else 0.0
        if conf < 0.6 or fac.get("importance", 0) <= 0:
            continue
        eids = list(fac["evidence_ids"])
        if fac.get("good_region") is not None:
            worked.append(
                {
                    "text": f"Prefer {fac['name']} in {fac['good_region']!s} ({fac['direction']})",
                    "confidence": round(conf, 3),
                    "evidence_ids": eids,
                    "scope": "same_algorithm",
                }
            )
        if fac.get("bad_region") is not None:
            failed.append(
                {
                    "text": f"Avoid {fac['name']} in {fac['bad_region']!s} on this run.",
                    "confidence": round(conf, 3),
                    "evidence_ids": eids,
                    "scope": "same_algorithm",
                }
            )
        # Transfer only for strong, role-labeled, non-tiny factors
        if fac.get("role") and fac.get("importance", 0) >= 0.15 and conf >= 0.65:
            transfer.append(
                {
                    "text": (
                        f"Role `{fac['role']}` ({fac['name']}) showed top-vs-worst association "
                        f"in this run; treat as a candidate priority knob on similar tasks."
                    ),
                    "confidence": round(conf * 0.9, 3),
                    "evidence_ids": eids,
                    "scope": "same_domain",
                }
            )

    return {
        "worked": worked[:5],
        "failed": failed[:5],
        "transfer": transfer[:3],
    }


def build_avoid(
    trials: Sequence[Dict[str, Any]],
    factors: Sequence[Dict[str, Any]],
    *,
    baseline_score: float,
    score_lo: float,
    score_hi: float,
    delta: float,
    higher_is_better: bool,
    max_items: int = 3,
) -> List[Dict[str, Any]]:
    span = abs(score_hi - score_lo)
    poor_cut = (
        baseline_score
        if higher_is_better
        else baseline_score
    )
    # Also allow "near floor"
    floor = score_lo + 0.2 * span if higher_is_better else score_hi - 0.2 * span

    def is_poor(mean_score: float) -> bool:
        if higher_is_better:
            return mean_score <= poor_cut or mean_score <= floor
        return mean_score >= poor_cut or mean_score >= floor

    avoid: List[Dict[str, Any]] = []
    for fac in factors:
        if fac.get("bad_region") is None or fac.get("status") != "TESTED":
            continue
        if fac.get("_score_gap", 0) < delta:
            continue
        worst_ids = set(fac.get("_worst_trials") or [])
        if len(worst_ids) < 2:
            continue
        worst_trials = [t for t in trials if t["trial"] in worst_ids]
        mean_score = statistics.mean(t["score"] for t in worst_trials)
        if not is_poor(mean_score):
            continue
        avoid.append(
            {
                "text": f"Harmful association for {fac['name']}: {fac['bad_region']!s}",
                "region": {fac["name"]: fac["bad_region"]},
                "trials": sorted(worst_ids)[:8],
            }
        )
        if len(avoid) >= max_items:
            break
    return avoid


# ---------------------------------------------------------------------------
# Quality + render
# ---------------------------------------------------------------------------


def compute_quality(
    trials: Sequence[Dict[str, Any]],
    factors: Sequence[Dict[str, Any]],
    evidence: Sequence[Dict[str, Any]],
    dynamics: Dict[str, Any],
    n_total_including_failed: Optional[int] = None,
) -> Dict[str, Any]:
    n = len(trials)
    n_tot = n_total_including_failed or n
    success_rate = n / max(n_tot, 1)
    tested = [f for f in factors if f.get("status") == "TESTED"]
    coverage = len(tested) / max(len(factors), 1) if factors else 0.0
    log_n = _clip01(math.log1p(n) / math.log1p(100))
    stag = dynamics.get("stagnation", 0)
    stag_pen = _clip01(stag / max(n, 1))
    fidelity = _clip01(0.35 * success_rate + 0.25 * coverage + 0.20 * log_n + 0.20 * (1 - stag_pen))

    supported = sum(1 for e in evidence if e.get("trials"))
    total_e = max(len(evidence), 1)
    effect = statistics.mean([abs(e.get("delta") or 0) for e in evidence]) if evidence else 0.0
    # Normalize effect loosely
    effect_score = _clip01(effect * 10) if evidence else 0.0
    conf = _clip01(0.40 * (supported / total_e) + 0.30 * effect_score + 0.20 * 0.7 + 0.10 * coverage)
    overall = round(0.5 * fidelity + 0.5 * conf, 4)
    return {
        "fidelity": round(fidelity, 4),
        "confidence": round(conf, 4),
        "overall": overall,
        "note": f"{n}/{n_tot} success; {len(tested)}/{len(factors)} factors TESTED; stagnation={stag}",
    }


def render_embed_text(problem: Dict[str, Any], knowledge: Dict[str, Any], result: Dict[str, Any]) -> str:
    top = sorted(knowledge.get("factors") or [], key=lambda f: f.get("importance", 0), reverse=True)[:3]
    fac_bits = "; ".join(f"{f['name']}({f['importance']:.2f})" for f in top) or "none"
    avoid = knowledge.get("avoid") or []
    avoid_bit = avoid[0]["text"] if avoid else "none"
    dyn = result.get("dynamics") or {}
    return (
        f"{problem.get('context', '')} "
        f"gain={result.get('normalized_gain')}; "
        f"incumbent_updates={dyn.get('incumbent_updates')}; "
        f"trend={dyn.get('trend')}; stagnant={dyn.get('stagnation')}. "
        f"Top factors: {fac_bits}. Avoid: {avoid_bit}."
    ).strip()


def render_prompt_block(
    problem: Dict[str, Any],
    knowledge: Dict[str, Any],
    result: Dict[str, Any],
    quality: Dict[str, Any],
) -> str:
    top = sorted(knowledge.get("factors") or [], key=lambda f: f.get("importance", 0), reverse=True)[:3]
    lines = [
        f"EXPERIENCE ({problem.get('dataset_id')}/{problem.get('algorithm_id')}, q={quality.get('overall')}):",
        f"- gain={result.get('normalized_gain')} over baseline ({result.get('baseline_kind')}); "
        f"{result.get('dynamics', {}).get('incumbent_updates')} incumbent updates; "
        f"ended {result.get('dynamics', {}).get('trend')}, stagnant {result.get('dynamics', {}).get('stagnation')}.",
    ]
    if result.get("dynamics", {}).get("phase_note"):
        lines.append(f"- curve: {result['dynamics']['phase_note']}")
    for f in top:
        if f.get("importance", 0) <= 0:
            continue
        lines.append(f"- factor {f['name']}: {f['direction']}")
    for a in (knowledge.get("avoid") or [])[:2]:
        lines.append(f"- avoid: {a['text']}")
    return "\n".join(lines)


def _strip_private(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_private(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [_strip_private(x) for x in obj]
    return obj


def _solution_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "config",
        "config": _jsonable(cfg),
        "vector": None,
        "sequence": None,
        "artifacts": None,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_experience_from_trials(
    trials: Sequence[Dict[str, Any]],
    *,
    run_uri: str,
    optimizer: str = "unknown",
    domain: str = "tabular_ml",
    task_type: str = "classification",
    dataset_id: Optional[str] = None,
    algorithm_id: Optional[str] = None,
    metric: str = "bal_acc",
    higher_is_better: bool = True,
    context: Optional[str] = None,
    profile: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    baseline_score: Optional[float] = None,
    baseline_kind: Optional[str] = None,
    default_config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    include_phase_note: bool = True,
    n_failed: int = 0,
    experience_id: Optional[str] = None,
) -> Dict[str, Any]:
    if not trials:
        raise ValueError("Need at least one successful trial")

    scores = [t["score"] for t in trials]
    score_lo, score_hi = min(scores), max(scores)
    best_trial = max(trials, key=lambda t: t["score"]) if higher_is_better else min(trials, key=lambda t: t["score"])

    base_score, base_kind = compute_baseline(
        trials,
        baseline_score=baseline_score,
        baseline_kind=baseline_kind,
        default_config=default_config,
    )
    gain = normalized_gain(
        best_trial["score"], base_score, score_lo, score_hi, higher_is_better=higher_is_better
    )
    dynamics = compute_dynamics(trials, higher_is_better=higher_is_better, include_phase_note=include_phase_note)

    factors, delta = analyze_factors(trials, higher_is_better=higher_is_better)
    evidence = build_evidence(trials, factors, dynamics, higher_is_better=higher_is_better, delta=delta)
    takeaways = build_takeaways(factors, evidence)
    avoid = build_avoid(
        trials,
        factors,
        baseline_score=base_score,
        score_lo=score_lo,
        score_hi=score_hi,
        delta=delta,
        higher_is_better=higher_is_better,
    )
    quality = compute_quality(trials, factors, evidence, dynamics, n_total_including_failed=len(trials) + n_failed)

    # Space fingerprint from observed keys
    keys = sorted({k for t in trials for k in t["config"]})
    space_factors = []
    for name in keys:
        vals = [t["config"][name] for t in trials if name in t["config"]]
        kind = _factor_kind(vals) if vals else "other"
        space_factors.append(
            {
                "name": name,
                "kind": kind,
                "role": _infer_role(name),
                "type": "float" if kind == "numeric_knob" else "categorical",
                "log_scale": None,
                "bounds": None,
            }
        )
    space_hash = _stable_hash({"keys": keys, "algorithm": algorithm_id})

    algo = algorithm_id or (trials[0]["full_config"].get("algorithm") if trials[0].get("full_config") else None)
    ds = dataset_id or "unknown"
    ctx = context or (
        f"{task_type} on {ds}"
        + (f" with {algo}" if algo else "")
        + f"; optimize {metric} ({'higher' if higher_is_better else 'lower'} is better)."
    )

    problem = {
        "domain": domain,
        "task_type": task_type,
        "dataset_id": ds,
        "algorithm_id": algo,
        "context": ctx,
        "metric": metric,
        "higher_is_better": higher_is_better,
        "profile": profile or {},
        "tags": tags or [],
        "representation": "structured_config",
        "space": {
            "space_hash": space_hash,
            "dim": len(keys),
            "decoder": None,
            "bounds": None,
            "factors": space_factors,
            "space_text": ", ".join(keys),
        },
    }

    result = {
        "n_trials": len(trials) + n_failed,
        "n_success": len(trials),
        "n_failed": n_failed,
        "score_range": {"lo": score_lo, "hi": score_hi},
        "best": {
            "trial": int(best_trial["trial"]),
            "score": float(best_trial["score"]),
            "solution": _solution_from_config(best_trial["config"]),
        },
        "baseline_score": float(base_score),
        "baseline_kind": base_kind,
        "normalized_gain": round(gain, 4),
        "dynamics": {
            "trend": dynamics["trend"],
            "stagnation": dynamics["stagnation"],
            "incumbent_updates": dynamics["incumbent_updates"],
            "phase_note": dynamics.get("phase_note"),
        },
    }

    knowledge = {
        "factors": [_strip_private(f) for f in factors if f.get("importance", 0) > 0 or f.get("status") == "TESTED"],
        "evidence": evidence,
        "takeaways": takeaways,
        "avoid": avoid,
    }
    # Keep top factors only in delivery render; store all TESTED with signal
    knowledge["factors"] = [
        f for f in knowledge["factors"] if f.get("importance", 0) > 0 or f.get("good_region") is not None
    ][:12]

    embed = render_embed_text(problem, knowledge, result)
    prompt = render_prompt_block(problem, knowledge, result, quality)

    # Train good/bad from top/worst configs
    top, worst = _cohorts(trials, higher_is_better=higher_is_better)
    train = {
        "best": _solution_from_config(best_trial["config"]),
        "good": [_solution_from_config(t["config"]) for t in top[:5]],
        "bad": [_solution_from_config(t["config"]) for t in worst[:5]],
    }

    exp_id = experience_id or f"exp_{ds}_{algo or 'algo'}_{space_hash}"
    return {
        "schema_version": "experience.v2",
        "id": exp_id,
        "created_at": _utc_now(),
        "meta": {
            "run_uri": run_uri,
            "optimizer": optimizer,
            "seed": seed,
            "budget_trials": len(trials) + n_failed,
            "summarizer": {
                "facts": "top_worst_gap+incumbent_jumps+curve_dynamics",
                "llm": None,
                "template": "experience_builder.v2",
            },
        },
        "problem": problem,
        "result": result,
        "knowledge": knowledge,
        "delivery": {
            "quality": quality,
            "embed_text": embed,
            "filters": {
                "domain": domain,
                "task_type": task_type,
                "representation": "structured_config",
                "algorithm_id": algo,
                "space_hash": space_hash,
            },
            "prompt_block": prompt,
            "train": train,
        },
    }


def build_experience_from_bo_json(path: Path, **kwargs: Any) -> Dict[str, Any]:
    path = Path(path)
    # Infer ids from .../cls/bal_acc/{dataset}/{algo}.json
    parts = path.parts
    dataset_id = kwargs.pop("dataset_id", None)
    algorithm_id = kwargs.pop("algorithm_id", None)
    task_type = kwargs.get("task_type", "classification")
    metric = kwargs.get("metric", "bal_acc")
    domain = kwargs.get("domain", "tabular_ml")
    if dataset_id is None or algorithm_id is None:
        try:
            algorithm_id = algorithm_id or path.stem
            dataset_id = dataset_id or path.parent.name
            metric = path.parent.parent.name
            tt = path.parent.parent.parent.name
            task_type = "classification" if tt == "cls" else ("regression" if tt == "rgs" else task_type)
            kwargs["metric"] = metric
            kwargs["task_type"] = task_type
        except Exception:
            pass

    higher_is_better = kwargs.pop("higher_is_better", metric not in {"mse", "mae", "rmse", "latency_ms"})
    openbox_minimize = kwargs.pop("openbox_minimize", True)
    algorithm = kwargs.get("algorithm_id") or algorithm_id

    # Count failures
    raw = json.loads(path.read_text(encoding="utf-8"))
    n_failed = sum(1 for row in (raw.get("observations") or []) if int(row.get("trial_state", 0)) != 0)

    trials = load_trials_from_bo_json(
        path,
        algorithm=algorithm,
        higher_is_better=True if higher_is_better else False,
        openbox_minimize=openbox_minimize,
    )
    # If lower-is-better metric stored as minimize already, load_trials with higher_is_better False
    # uses raw objective; for mse openbox may store positive mse to minimize — handle via flag.
    if not higher_is_better:
        trials = load_trials_from_bo_json(
            path, algorithm=algorithm, higher_is_better=False, openbox_minimize=True
        )

    return build_experience_from_trials(
        trials,
        run_uri=str(path),
        dataset_id=dataset_id,
        algorithm_id=algorithm_id,
        higher_is_better=higher_is_better,
        n_failed=n_failed,
        domain=domain,
        **kwargs,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build Experience v2 JSON from a BO history file")
    p.add_argument("history_json", type=Path, help="OpenBox-style observations JSON")
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--optimizer", default="bo_history")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--baseline-score", type=float, default=None)
    p.add_argument("--baseline-kind", default=None)
    p.add_argument("--no-phase-note", action="store_true")
    p.add_argument("--validate", action="store_true", help="Validate against experience.v2 schema")
    args = p.parse_args(argv)

    exp = build_experience_from_bo_json(
        args.history_json,
        optimizer=args.optimizer,
        seed=args.seed,
        baseline_score=args.baseline_score,
        baseline_kind=args.baseline_kind,
        include_phase_note=not args.no_phase_note,
    )

    if args.validate:
        try:
            import jsonschema
        except ImportError as e:
            raise SystemExit("jsonschema not installed") from e
        schema_path = Path(__file__).resolve().parents[1] / "schema" / "experience.v2.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.validate(exp, schema)

    text = json.dumps(exp, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
