"""Offline experience-bank v3 gates: fANOVA keep-set, usable flag, minimal prompt.

All thresholds live here. Nothing in this module is shown to the LLM except
`render_minimal_prompt()` output.

fANOVA engine matrix (offline bank build; live Self-run gates are out of scope):

| engine            | space            | keep driver        | when used                          |
| fanova_full_cs    | real ConfigSpace | importance_1d      | **locked bank-build engine**       |
| fanova_full       | observed samples | importance_1d      | CS missing (sklearn forests, etc.) |
| fanova_full_ix    | real ConfigSpace | importance_total   | ablation only                      |
| fanova_rf         | observed         | RF impurity / PD   | ablation / live fallback           |
| proxy             | observed 1D bins | bin-mean variance  | n < 24 or worker failure           |

`importance_1d` = main effect. `importance_total` = main + interactions (Sobol total).
Keep-set uses 1d; total is an exclusion filter for interaction-dominated HPs.
"""

from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mindware.components.experience.builder import _infer_role, _strip_private
from mindware.components.optimizers.experience_gate import SKIP_HP_NAMES


DEFAULT_GATE_CONFIG: Dict[str, Any] = {
    "keep_floor": 0.03,
    "keep_cum": 0.60,
    "keep_topk": 4,
    "keep_min": 2,
    "main_effect_ratio": 0.35,
    "require_region": True,
    "max_1d_record": 0.10,
    "min_keep_hps": 2,
    "interaction_only_reject": True,
    "ix_max_1d": 0.08,
    "ix_min_total": 0.25,
}

BANK_ENGINE = "fanova_full_cs"

_MODELS_ROOT = Path(__file__).resolve().parents[1] / "models"
_CS_CACHE: Dict[Tuple[str, str], Any] = {}


def hp_stem(name: str) -> str:
    raw = str(name or "")
    return raw.split(":")[-1] if ":" in raw else raw


def normalize_config(cfg: Optional[Dict[str, Any]], algorithm: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    prefix = f"{algorithm}:" if algorithm else None
    for key, val in (cfg or {}).items():
        if key in SKIP_HP_NAMES or key == "algorithm":
            continue
        stem = hp_stem(key)
        if prefix and key.startswith(prefix):
            stem = key[len(prefix) :]
        out[stem] = val
    return out


def merge_gate_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = dict(DEFAULT_GATE_CONFIG)
    if overrides:
        cfg.update(overrides)
    return cfg


def load_estimator_cs(algorithm: str, task_type: str):
    """Load HPO ConfigSpace without importing sklearn-heavy package __init__."""
    algo = str(algorithm or "").strip().lower()
    task = str(task_type or "").strip().lower()
    kind = "classification"
    if task in ("regression", "rgs") or "reg" in task:
        kind = "regression"
    key = (algo, kind)
    if key in _CS_CACHE:
        return _CS_CACHE[key]
    path = _MODELS_ROOT / kind / f"{algo}.py"
    if not path.is_file():
        _CS_CACHE[key] = None
        return None
    spec = importlib.util.spec_from_file_location(f"_bank_cs_{kind}_{algo}", str(path))
    if spec is None or spec.loader is None:
        _CS_CACHE[key] = None
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        _CS_CACHE[key] = None
        return None
    skip = {"BaseClassificationModel", "BaseRegressionModel", "BaseModel"}
    for obj in vars(mod).values():
        if not isinstance(obj, type) or obj.__name__ in skip:
            continue
        fn = getattr(obj, "get_hyperparameter_search_space", None)
        if not callable(fn):
            continue
        try:
            cs = fn()
        except TypeError:
            try:
                cs = fn(None)
            except Exception:
                continue
        except Exception:
            continue
        if cs is None:
            continue
        _CS_CACHE[key] = cs
        return cs
    _CS_CACHE[key] = None
    return None


def calibrated_keep(
    imps: Dict[str, float],
    *,
    floor: float = 0.03,
    min_n: int = 2,
    cum: float = 0.60,
    topk: int = 4,
) -> List[str]:
    ranked = sorted(
        ((k, float(v)) for k, v in imps.items() if float(v) >= float(floor)),
        key=lambda kv: -kv[1],
    )
    if len(ranked) < int(min_n):
        return []
    keep: List[str] = []
    acc = 0.0
    for name, val in ranked:
        keep.append(name)
        acc += val
        if len(keep) >= int(min_n) and (acc >= float(cum) or len(keep) >= int(topk)):
            break
    return keep


def _stem_importance(imps: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, val in (imps or {}).items():
        stem = hp_stem(name)
        if stem in SKIP_HP_NAMES or stem == "algorithm":
            continue
        out[stem] = max(float(val), out.get(stem, 0.0))
    return out


def _factor_index(factors: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for fac in factors or []:
        name = str(fac.get("name") or "")
        if not name:
            continue
        out[name] = fac
        out[hp_stem(name)] = fac
    return out


def has_region(fac: Optional[Dict[str, Any]]) -> bool:
    if not fac:
        return False
    return fac.get("good_region") is not None or fac.get("bad_region") is not None


def score_span_ok(trials: Sequence[Dict[str, Any]]) -> bool:
    scores = [float(t["score"]) for t in trials if t.get("score") is not None]
    if len(scores) < 8:
        return False
    return (max(scores) - min(scores)) > 1e-12


def stage1_keep_hps(
    imps_1d: Dict[str, float],
    imps_total: Dict[str, float],
    factors: Sequence[Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
) -> List[str]:
    cfg = merge_gate_config(cfg)
    one = _stem_importance(imps_1d)
    tot = _stem_importance(imps_total) or dict(one)
    idx = _factor_index(factors)
    names = calibrated_keep(
        one,
        floor=float(cfg["keep_floor"]),
        min_n=int(cfg["keep_min"]),
        cum=float(cfg["keep_cum"]),
        topk=int(cfg["keep_topk"]),
    )
    kept: List[str] = []
    ratio_min = float(cfg["main_effect_ratio"])
    for name in names:
        i1 = float(one.get(name, 0.0))
        it = float(tot.get(name, i1) or i1)
        if it > 1e-12 and (i1 / it) < ratio_min:
            continue
        fac = idx.get(name)
        if cfg.get("require_region") and not has_region(fac):
            continue
        kept.append(name)
    return kept


def record_usable(
    *,
    imps_1d: Dict[str, float],
    imps_total: Dict[str, float],
    keep_hps: Sequence[str],
    trials: Sequence[Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
) -> bool:
    cfg = merge_gate_config(cfg)
    if not score_span_ok(trials):
        return False
    one = _stem_importance(imps_1d)
    tot = _stem_importance(imps_total) or dict(one)
    max_1d = max(one.values()) if one else 0.0
    max_tot = max(tot.values()) if tot else 0.0
    if max_1d < float(cfg["max_1d_record"]):
        return False
    if cfg.get("interaction_only_reject") and max_1d < float(cfg["ix_max_1d"]) and max_tot > float(cfg["ix_min_total"]):
        return False
    if len(keep_hps) < int(cfg["min_keep_hps"]):
        return False
    return True


def _fmt_num(x: Any) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if abs(v - round(v)) < 1e-9 and abs(v) >= 1.0:
        return str(int(round(v)))
    text = f"{v:.4g}"
    return text


def _prefer_phrase(fac: Dict[str, Any]) -> Optional[str]:
    direction = str(fac.get("direction") or "").lower()
    if "higher" in direction and "over-represented" in direction:
        pref = "higher"
    elif "lower" in direction and "over-represented" in direction:
        pref = "lower"
    else:
        pref = None
    good = fac.get("good_region")
    name = hp_stem(str(fac.get("name") or ""))
    if not name or name in SKIP_HP_NAMES:
        return None
    if isinstance(good, dict) and "low" in good and "high" in good:
        lo, hi = _fmt_num(good["low"]), _fmt_num(good["high"])
        if pref:
            return f"- {name}: prefer {pref}, roughly {lo}–{hi}"
        return f"- {name}: prefer roughly {lo}–{hi}"
    if isinstance(good, list) and good:
        return f"- {name}: prefer {good[0]!r}"
    if pref:
        return f"- {name}: prefer {pref}"
    return None


def render_minimal_prompt(
    dataset_id: str,
    algorithm_id: str,
    keep_hps: Sequence[str],
    factors: Sequence[Dict[str, Any]],
) -> str:
    idx = _factor_index(factors)
    lines = [f"Similar task: {dataset_id} ({algorithm_id})"]
    for name in keep_hps:
        fac = idx.get(name)
        if not fac:
            continue
        phrase = _prefer_phrase(fac)
        if phrase:
            lines.append(phrase)
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def importance_by_role_from_1d(imps_1d: Dict[str, float]) -> Dict[str, float]:
    acc: Dict[str, float] = {}
    for name, val in (_stem_importance(imps_1d) or {}).items():
        role = _infer_role(name)
        if not role:
            continue
        acc[role] = acc.get(role, 0.0) + float(val)
    total = sum(acc.values()) or 1.0
    return {k: v / total for k, v in acc.items()}


def apply_gates(
    experience: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    *,
    trials: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Replay keep-set / usable / prompt from stored fANOVA + factors_all."""
    cfg = merge_gate_config(cfg)
    knowledge = experience.setdefault("knowledge", {})
    fanova = dict(knowledge.get("fanova") or {})
    imps_1d = dict(fanova.get("importance_1d") or {})
    imps_total = dict(fanova.get("importance_total") or imps_1d)
    factors_all = list(knowledge.get("factors_all") or knowledge.get("factors") or [])
    keep = stage1_keep_hps(imps_1d, imps_total, factors_all, cfg)
    trial_rows = trials
    if trial_rows is None:
        trial_rows = (experience.get("retrieval") or {}).get("trials") or []
    usable = record_usable(
        imps_1d=imps_1d,
        imps_total=imps_total,
        keep_hps=keep,
        trials=trial_rows,
        cfg=cfg,
    )
    idx = _factor_index(factors_all)
    kept_factors = []
    for name in keep:
        fac = idx.get(name)
        if fac:
            row = _strip_private(dict(fac))
            row["name"] = hp_stem(str(fac.get("name") or name))
            if name in imps_1d or hp_stem(name) in _stem_importance(imps_1d):
                row["importance"] = round(float(_stem_importance(imps_1d).get(hp_stem(name), fac.get("importance") or 0.0)), 4)
                row["method"] = "fanova_1d"
            kept_factors.append(row)
    knowledge["factors"] = kept_factors
    fanova["keep_hps"] = list(keep)
    knowledge["fanova"] = fanova
    prob = experience.get("problem") or {}
    prompt = render_minimal_prompt(
        str(prob.get("dataset_id") or ""),
        str(prob.get("algorithm_id") or ""),
        keep,
        kept_factors or factors_all,
    )
    delivery = experience.setdefault("delivery", {})
    delivery["usable_for_prompt"] = bool(usable and prompt)
    delivery["prompt_block"] = prompt if delivery["usable_for_prompt"] else ""
    delivery["gate_config"] = {k: cfg[k] for k in DEFAULT_GATE_CONFIG}
    return experience


def clone_and_apply(experience: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return apply_gates(copy.deepcopy(experience), cfg)


def gate_view(experience: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Cheap keep/usable/prompt replay without mutating the stored record."""
    cfg = merge_gate_config(cfg)
    knowledge = experience.get("knowledge") or {}
    fanova = knowledge.get("fanova") or {}
    imps_1d = dict(fanova.get("importance_1d") or {})
    imps_total = dict(fanova.get("importance_total") or imps_1d)
    factors_all = list(knowledge.get("factors_all") or knowledge.get("factors") or [])
    keep = stage1_keep_hps(imps_1d, imps_total, factors_all, cfg)
    trials = (experience.get("retrieval") or {}).get("trials") or []
    usable = record_usable(
        imps_1d=imps_1d,
        imps_total=imps_total,
        keep_hps=keep,
        trials=trials,
        cfg=cfg,
    )
    idx = _factor_index(factors_all)
    kept_factors = [idx[n] for n in keep if n in idx]
    prob = experience.get("problem") or {}
    prompt = render_minimal_prompt(
        str(prob.get("dataset_id") or ""),
        str(prob.get("algorithm_id") or ""),
        keep,
        kept_factors or factors_all,
    )
    one = _stem_importance(imps_1d)
    tot = _stem_importance(imps_total) or dict(one)
    return {
        "keep_hps": list(keep),
        "usable_for_prompt": bool(usable and prompt),
        "prompt_block": prompt if (usable and prompt) else "",
        "n_keep": len(keep),
        "max_1d": max(one.values()) if one else 0.0,
        "max_total": max(tot.values()) if tot else 0.0,
    }


def fanova_roles_attach(experience: Dict[str, Any]) -> None:
    ret = experience.setdefault("retrieval", {})
    imps = ((experience.get("knowledge") or {}).get("fanova") or {}).get("importance_1d") or {}
    if imps:
        ret["importance_by_role"] = importance_by_role_from_1d(imps)
    roles = []
    for fac in ((experience.get("problem") or {}).get("space") or {}).get("factors") or []:
        role = fac.get("role") or _infer_role(str(fac.get("name") or ""))
        if role:
            roles.append(role)
    if roles:
        ret["roles"] = sorted(set(roles))


def attach_fanova(
    experience: Dict[str, Any],
    trials: Sequence[Dict[str, Any]],
    *,
    pair_topk: int = 0,
    gate_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run official fANOVA (real CS when available) and apply v3 gates."""
    from mindware.components.optimizers.experience_gate import hp_importances_both

    prob = experience.get("problem") or {}
    algo = str(prob.get("algorithm_id") or "")
    task = str(prob.get("task_type") or "classification")
    hib = bool(prob.get("higher_is_better", True))
    successes = []
    for i, t in enumerate(trials, 1):
        cfg = normalize_config(t.get("config") or {}, algo)
        if not cfg:
            continue
        successes.append({"trial": t.get("trial", i), "config": cfg, "score": float(t["score"])})
    cs = load_estimator_cs(algo, task)
    one, tot, pairs, engine = hp_importances_both(
        successes, config_space=cs, pair_topk=int(pair_topk or 0)
    )
    knowledge = experience.setdefault("knowledge", {})
    factors_all = [_strip_private(f) for f in (knowledge.get("factors") or [])]
    knowledge["factors_all"] = factors_all
    knowledge["fanova"] = {
        "engine": engine,
        "n_trials": len(successes),
        "n_trees": 32,
        "importance_1d": {k: round(float(v), 6) for k, v in sorted(one.items())},
        "importance_total": {k: round(float(v), 6) for k, v in sorted(tot.items())},
        "top_pairs": pairs,
        "keep_hps": [],
    }
    apply_gates(experience, gate_config, trials=successes)
    fanova_roles_attach(experience)
    experience["schema_version"] = "experience.v2.1"
    meta = experience.setdefault("meta", {})
    summ = meta.setdefault("summarizer", {})
    summ["facts"] = "fanova_1d+top_worst_regions"
    summ["template"] = "experience_builder.v2.1"
    _ = hib
    return experience
