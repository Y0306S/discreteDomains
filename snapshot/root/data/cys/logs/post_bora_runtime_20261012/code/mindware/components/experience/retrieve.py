"""Role-aligned importance retrieval + online Spearman reweight (Layers B/C)."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mindware.components.experience.profile import cosine, profile_vector

from mindware.components.experience.builder import _infer_role, analyze_factors
from mindware.components.optimizers.experience_gate import SKIP_HP_NAMES


DEFAULT_WEIGHTS = {
    "importance_cos": 0.45,
    "algo_match": 0.25,
    "profile_cos": 0.15,
    "role_jaccard": 0.15,
}

# Used before the current run has an importance vector (trials < rerank_after).
COLD_WEIGHTS = {
    "importance_cos": 0.0,
    "algo_match": 0.40,
    "profile_cos": 0.25,
    "role_jaccard": 0.25,
    "semantic_profile_cos": 0.10,
}

_GATE_CONFIG_PATH = Path(__file__).with_name("gate_config.json")


def load_retrieve_knobs() -> Dict[str, Any]:
    """Layer-C knobs written by the offline gate sweep. Missing file → v3 defaults."""
    knobs = {
        "k_max": 3,
        "require_usable": True,
        "spearman_floor": 0.5,
        "dedup": True,
        "ranker": "surrogate",
        "clash_gap": 0.15,
    }
    if _GATE_CONFIG_PATH.is_file():
        try:
            rec = json.loads(_GATE_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return knobs
        if "k_max" in rec:
            knobs["k_max"] = int(rec["k_max"])
        if "require_usable" in rec:
            knobs["require_usable"] = bool(rec["require_usable"])
        if "spearman_floor" in rec:
            knobs["spearman_floor"] = float(rec["spearman_floor"])
        if "dedup" in rec:
            knobs["dedup"] = bool(rec["dedup"])
        if "ranker" in rec:
            knobs["ranker"] = str(rec["ranker"] or "surrogate")
        if "clash_gap" in rec:
            knobs["clash_gap"] = float(rec["clash_gap"])
    return knobs


def infer_role(name: str) -> Optional[str]:
    return _infer_role(name)


def importance_by_role(experience: Dict[str, Any]) -> Dict[str, float]:
    fan = ((experience.get("knowledge") or {}).get("fanova") or {}).get("importance_1d")
    if fan:
        from mindware.components.experience.bank_gates import importance_by_role_from_1d

        return importance_by_role_from_1d(fan)
    acc: Dict[str, float] = {}
    for fac in (experience.get("knowledge") or {}).get("factors") or []:
        role = fac.get("role") or infer_role(str(fac.get("name") or ""))
        if not role:
            continue
        acc[role] = acc.get(role, 0.0) + float(fac.get("importance") or 0.0)
    total = sum(acc.values()) or 1.0
    return {k: v / total for k, v in acc.items()}


def roles_of(experience: Dict[str, Any]) -> List[str]:
    roles = []
    for fac in ((experience.get("problem") or {}).get("space") or {}).get("factors") or []:
        r = fac.get("role") or infer_role(str(fac.get("name") or ""))
        if r:
            roles.append(r)
    if roles:
        return sorted(set(roles))
    return sorted(importance_by_role(experience).keys())


def role_jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(len(sa | sb), 1)


def importance_cosine(query: Dict[str, float], source: Dict[str, float]) -> float:
    keys = sorted(set(query) | set(source))
    if not keys:
        return 0.0
    return cosine([query.get(k, 0.0) for k in keys], [source.get(k, 0.0) for k in keys])


def _rankdata(xs: Sequence[float]) -> List[float]:
    n = len(xs)
    order = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or len(a) < 3:
        return 0.0
    ra = _rankdata([float(x) for x in a])
    rb = _rankdata([float(x) for x in b])
    return cosine([x - sum(ra) / len(ra) for x in ra], [x - sum(rb) / len(rb) for x in rb])


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


LOG_SCALE_ROLES = frozenset({"learning_rate", "regularization"})
NEAR_NN_DIST = 0.25  # unit-cube L2; neighbour closer than this counts as coverage
_BANK_HP_BOUNDS: Dict[Tuple[str, str], Tuple[float, float]] = {}
_LABEL_B_STATS = {"n_hits": 0, "n_lines_seen": 0, "n_lines_dropped": 0, "n_hits_with_drop": 0}


def label_b_stats() -> Dict[str, int]:
    return dict(_LABEL_B_STATS)


def reset_label_b_stats() -> None:
    for k in _LABEL_B_STATS:
        _LABEL_B_STATS[k] = 0


def higher_is_better_of(exp: Dict[str, Any], default: bool = True) -> bool:
    v = (exp.get("problem") or {}).get("higher_is_better")
    if v is None:
        return default
    return bool(v)


def compute_bank_hp_bounds(experiences: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str], Tuple[float, float]]:
    acc: Dict[Tuple[str, str], List[float]] = {}
    for exp in experiences:
        algo = str(((exp.get("problem") or {}).get("algorithm_id") or "")).lower()
        for t in (exp.get("retrieval") or {}).get("trials") or []:
            cfg = t.get("config") or {}
            for k, v in cfg.items():
                if _is_number(v):
                    acc.setdefault((algo, k), []).append(float(v))
    return {key: (min(xs), max(xs)) for key, xs in acc.items() if xs}


def set_bank_hp_bounds(
    bounds: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None,
    experiences: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[Tuple[str, str], Tuple[float, float]]:
    global _BANK_HP_BOUNDS
    if bounds is None:
        bounds = compute_bank_hp_bounds(experiences or [])
    _BANK_HP_BOUNDS = dict(bounds)
    return _BANK_HP_BOUNDS


def _hp_stem(name: str) -> str:
    return name.split(":")[-1] if ":" in name else name


def _log_scale_hp(name: str) -> bool:
    role = infer_role(name)
    if role in LOG_SCALE_ROLES:
        return True
    stem = _hp_stem(name).lower()
    return stem in {"learning_rate", "lr", "reg_lambda", "reg_alpha", "regularization"}


def _transform_hp(name: str, v: float) -> float:
    x = float(v)
    if _log_scale_hp(name):
        return math.log10(abs(x) + 1e-12)
    return x


def _orient_score(score: float, higher_is_better: bool) -> float:
    return float(score) if higher_is_better else -float(score)


def _shared_role_names(
    src_cfg: Dict[str, Any],
    tgt_cfg: Dict[str, Any],
    *,
    same_algo: bool = False,
) -> List[Tuple[str, str]]:
    """Pairs of (source_key, target_key). Same-algo: all shared numeric names."""
    if same_algo:
        pairs: List[Tuple[str, str]] = []
        for k in src_cfg:
            if k in tgt_cfg and _is_number(src_cfg.get(k)) and _is_number(tgt_cfg.get(k)):
                pairs.append((k, k))
        if pairs:
            return pairs
    src_by_role: Dict[str, List[str]] = {}
    tgt_by_role: Dict[str, List[str]] = {}
    for k in src_cfg:
        r = infer_role(k)
        if r:
            src_by_role.setdefault(r, []).append(k)
    for k in tgt_cfg:
        r = infer_role(k)
        if r:
            tgt_by_role.setdefault(r, []).append(k)
    pairs = []
    used_s, used_t = set(), set()
    for r in sorted(set(src_by_role) & set(tgt_by_role)):
        sk, tk = src_by_role[r][0], tgt_by_role[r][0]
        pairs.append((sk, tk))
        used_s.add(sk)
        used_t.add(tk)
    for k in src_cfg:
        if k in tgt_cfg and k not in used_s and k not in used_t:
            pairs.append((k, k))
    return pairs


def _unit(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return (float(v) - lo) / (hi - lo)


def _bounds_for(algo: str, key: str, cfgs: Sequence[Dict[str, Any]]) -> Tuple[float, float]:
    bank = _BANK_HP_BOUNDS.get((algo.lower(), key))
    if bank is not None:
        lo, hi = bank
        if _log_scale_hp(key):
            return (_transform_hp(key, lo), _transform_hp(key, hi))
        return (float(lo), float(hi))
    xs = [_transform_hp(key, float(c[key])) for c in cfgs if key in c and _is_number(c[key])]
    return (min(xs), max(xs)) if xs else (0.0, 1.0)


def nearest_source_scores(
    source_trials: Sequence[Dict[str, Any]],
    target_trials: Sequence[Dict[str, Any]],
    *,
    src_algo: str = "",
    tgt_algo: str = "",
    src_higher_is_better: bool = True,
    tgt_higher_is_better: bool = True,
) -> Optional[Tuple[List[float], List[float]]]:
    packed = nearest_source_scores_ex(
        source_trials,
        target_trials,
        src_algo=src_algo,
        tgt_algo=tgt_algo,
        src_higher_is_better=src_higher_is_better,
        tgt_higher_is_better=tgt_higher_is_better,
    )
    if packed is None:
        return None
    return packed["pred"], packed["obs"]


def nearest_source_scores_ex(
    source_trials: Sequence[Dict[str, Any]],
    target_trials: Sequence[Dict[str, Any]],
    *,
    src_algo: str = "",
    tgt_algo: str = "",
    src_higher_is_better: bool = True,
    tgt_higher_is_better: bool = True,
) -> Optional[Dict[str, Any]]:
    """For each target trial, 1-NN source trial in role-aligned HP space → paired scores."""
    if len(source_trials) < 3 or len(target_trials) < 3:
        return None
    src_cfgs = [t.get("config") or {} for t in source_trials]
    tgt_cfgs = [t.get("config") or {} for t in target_trials]
    same_algo = bool(src_algo) and src_algo.lower() == (tgt_algo or src_algo).lower()
    pairs = _shared_role_names(src_cfgs[0], tgt_cfgs[0], same_algo=same_algo)
    num_pairs = [(s, t) for s, t in pairs if _is_number(src_cfgs[0].get(s)) and _is_number(tgt_cfgs[0].get(t))]
    if len(num_pairs) < 2:
        return None

    src_b = {s: _bounds_for(src_algo, s, src_cfgs) for s, _ in num_pairs}
    tgt_b = {t: _bounds_for(tgt_algo or src_algo, t, tgt_cfgs) for _, t in num_pairs}

    src_pts = []
    for c in src_cfgs:
        src_pts.append(
            [_unit(_transform_hp(s, float(c.get(s, src_b[s][0]))), *src_b[s]) for s, _ in num_pairs]
        )
    pred: List[float] = []
    obs: List[float] = []
    nn_dists: List[float] = []
    for tc, tt in zip(tgt_cfgs, target_trials):
        q = [_unit(_transform_hp(t, float(tc.get(t, tgt_b[t][0]))), *tgt_b[t]) for _, t in num_pairs]
        best_i, best_d = 0, 1e18
        for i, p in enumerate(src_pts):
            d = sum((a - b) ** 2 for a, b in zip(q, p))
            if d < best_d:
                best_d, best_i = d, i
        pred.append(_orient_score(source_trials[best_i]["score"], src_higher_is_better))
        obs.append(_orient_score(tt["score"], tgt_higher_is_better))
        nn_dists.append(math.sqrt(best_d) if best_d >= 0 else 0.0)
    n_overlap = len(pred)
    coverage = float(sum(1 for d in nn_dists if d <= NEAR_NN_DIST) / n_overlap) if n_overlap else 0.0
    return {"pred": pred, "obs": obs, "nn_dists": nn_dists, "n_overlap": n_overlap, "coverage": coverage}


def spearman_transfer(
    source: Dict[str, Any],
    target_trials: Sequence[Dict[str, Any]],
    *,
    target_algo: Optional[str] = None,
    target_higher_is_better: Optional[bool] = None,
) -> float:
    stats = spearman_transfer_stats(
        source,
        target_trials,
        target_algo=target_algo,
        target_higher_is_better=target_higher_is_better,
    )
    return float(stats.get("rho") or 0.0)


def spearman_transfer_stats(
    source: Dict[str, Any],
    target_trials: Sequence[Dict[str, Any]],
    *,
    target_algo: Optional[str] = None,
    target_higher_is_better: Optional[bool] = None,
    shrink_k: float = 0.0,
) -> Dict[str, float]:
    src_trials = (source.get("retrieval") or {}).get("trials") or []
    src_algo = str(((source.get("problem") or {}).get("algorithm_id") or "")).lower()
    tgt_algo = (target_algo or src_algo or "").lower()
    src_hib = higher_is_better_of(source, True)
    tgt_hib = src_hib if target_higher_is_better is None else bool(target_higher_is_better)
    packed = nearest_source_scores_ex(
        src_trials,
        target_trials,
        src_algo=src_algo,
        tgt_algo=tgt_algo,
        src_higher_is_better=src_hib,
        tgt_higher_is_better=tgt_hib,
    )
    if packed is None:
        return {"rho": 0.0, "rho_adj": 0.0, "n_overlap": 0.0, "coverage": 0.0}
    rho = spearman(packed["pred"], packed["obs"])
    n_ov = float(packed["n_overlap"])
    cov = float(packed["coverage"])
    k = max(0.0, float(shrink_k))
    rho_adj = rho * n_ov / (n_ov + k) if (n_ov + k) > 0 else 0.0
    return {"rho": rho, "rho_adj": rho_adj, "n_overlap": n_ov, "coverage": cov}


def query_importance_from_trials(
    trials: Sequence[Dict[str, Any]],
    higher_is_better: bool = True,
) -> Dict[str, float]:
    if len(trials) < 4:
        return {}
    packed = [{"trial": t.get("trial", i), "config": t["config"], "score": t["score"]} for i, t in enumerate(trials, 1)]
    factors, _ = analyze_factors(packed, higher_is_better=bool(higher_is_better))
    fake = {"knowledge": {"factors": factors}}
    return importance_by_role(fake)


def factor_direction_sign(text: str) -> Optional[int]:
    low = str(text or "").lower()
    if "higher" in low and "over-represented" in low:
        return 1
    if "lower" in low and "over-represented" in low:
        return -1
    return None


def role_signs_from_experience(exp: Dict[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for fac in (exp.get("knowledge") or {}).get("factors") or []:
        role = fac.get("role") or infer_role(str(fac.get("name") or ""))
        if not role:
            continue
        sign = factor_direction_sign(str(fac.get("direction") or ""))
        if sign is not None:
            out[role] = sign
    return out


def role_signs_from_trials(
    trials: Sequence[Dict[str, Any]],
    higher_is_better: bool = True,
) -> Dict[str, int]:
    if len(trials) < 4:
        return {}
    packed = [{"trial": t.get("trial", i), "config": t["config"], "score": t["score"]} for i, t in enumerate(trials, 1)]
    factors, _ = analyze_factors(packed, higher_is_better=bool(higher_is_better))
    out: Dict[str, int] = {}
    for fac in factors:
        role = fac.get("role") or infer_role(str(fac.get("name") or ""))
        if not role:
            continue
        sign = factor_direction_sign(str(fac.get("direction") or ""))
        if sign is not None:
            out[role] = sign
    return out


@dataclass
class RetrievalHit:
    experience: Dict[str, Any]
    score: float
    features: Dict[str, float]
    spearman: Optional[float] = None

    @property
    def exp_id(self) -> str:
        return str(self.experience.get("id") or "")

    @property
    def prompt_block(self) -> str:
        return str((self.experience.get("delivery") or {}).get("prompt_block") or "")


@dataclass
class ExperienceBank:
    experiences: List[Dict[str, Any]] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    source_gp: Any = None

    @classmethod
    def load(cls, path: Path) -> "ExperienceBank":
        path = Path(path)
        rows: List[Dict[str, Any]] = []
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                rows = payload
            else:
                rows = payload.get("experiences") or []
                extra_w = payload.get("weights") or {}
                bank = cls(experiences=rows, weights={**DEFAULT_WEIGHTS, **extra_w})
                set_bank_hp_bounds(experiences=bank.experiences)
                return bank
        wpath = path.with_suffix(".weights.json")
        weights = dict(DEFAULT_WEIGHTS)
        if wpath.is_file():
            weights.update(json.loads(wpath.read_text(encoding="utf-8")))
        bank = cls(experiences=rows, weights=weights)
        set_bank_hp_bounds(experiences=bank.experiences)
        return bank

    def __len__(self) -> int:
        return len(self.experiences)


def _filters_ok(exp: Dict[str, Any], query: Dict[str, Any]) -> bool:
    flt = (exp.get("delivery") or {}).get("filters") or {}
    qf = query.get("filters") or {}
    if qf.get("domain") and flt.get("domain") and qf["domain"] != flt["domain"]:
        return False
    if qf.get("representation") and flt.get("representation") and qf["representation"] != flt["representation"]:
        return False
    if qf.get("task_type") and flt.get("task_type") and qf["task_type"] != flt["task_type"]:
        return False
    return True


def _is_self(exp: Dict[str, Any], query: Dict[str, Any]) -> bool:
    ds = ((exp.get("problem") or {}).get("dataset_id") or "").lower()
    qds = (query.get("dataset_id") or "").lower()
    algo = ((exp.get("problem") or {}).get("algorithm_id") or "").lower()
    qalgo = (query.get("algorithm_id") or "").lower()
    if ds and qds and ds == qds and algo and qalgo and algo == qalgo:
        return True
    if ds and qds and ds == qds:
        return True
    return False


def cold_features(
    query: Dict[str, Any],
    exp: Dict[str, Any],
) -> Dict[str, float]:
    q_imp = query.get("importance_by_role") or {}
    s_imp = (exp.get("retrieval") or {}).get("importance_by_role") or importance_by_role(exp)
    q_roles = query.get("roles") or list(q_imp)
    s_roles = (exp.get("retrieval") or {}).get("roles") or roles_of(exp)
    q_prof = query.get("profile_vec") or {}
    s_prof = (exp.get("retrieval") or {}).get("profile_vec") or {}
    q_algo = (query.get("algorithm_id") or "").lower()
    s_algo = str(((exp.get("problem") or {}).get("algorithm_id") or "")).lower()
    q_sem = query.get("semantic_vec") or []
    s_sem = (exp.get("retrieval") or {}).get("semantic_vec") or []
    q_vae = query.get("vae_vec") or []
    s_vae = (exp.get("retrieval") or {}).get("vae_vec") or []
    semantic_cos = 0.0
    if q_sem and s_sem and len(q_sem) == len(s_sem):
        semantic_cos = cosine(q_sem, s_sem)
    elif q_vae and s_vae:
        semantic_cos = cosine(q_vae, s_vae)
    vae_cos = cosine(q_vae, s_vae) if q_vae and s_vae else 0.0
    return {
        "importance_cos": importance_cosine(q_imp, s_imp) if q_imp and s_imp else 0.0,
        "algo_match": 1.0 if q_algo and q_algo == s_algo else 0.0,
        "profile_cos": cosine(profile_vector(q_prof), profile_vector(s_prof)) if q_prof and s_prof else 0.0,
        "role_jaccard": role_jaccard(q_roles, s_roles),
        "semantic_profile_cos": semantic_cos,
        "vae_latent_cos": vae_cos,
    }


def combine_score(feats: Dict[str, float], weights: Dict[str, float]) -> float:
    return sum(float(weights.get(k, 0.0)) * float(feats.get(k, 0.0)) for k in weights)


def _usable_ok(exp: Dict[str, Any], require_usable: bool) -> bool:
    if not require_usable:
        return True
    flag = (exp.get("delivery") or {}).get("usable_for_prompt")
    # v2 records have no flag — keep them. v3 false → drop.
    return flag is not False


def _dedup_hits(hits: Sequence[RetrievalHit]) -> List[RetrievalHit]:
    """Drop later cards whose prompt body is identical to an earlier one."""
    seen: set = set()
    out: List[RetrievalHit] = []
    for h in hits:
        text = str((h.experience.get("delivery") or {}).get("prompt_block") or "").strip()
        body = "\n".join(ln for ln in text.splitlines() if ln.strip().startswith("- "))
        key = body or text
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(h)
    return out


def _same_algo_ok(exp: Dict[str, Any], query: Dict[str, Any]) -> bool:
    qalgo = str(query.get("algorithm_id") or "").lower()
    if not qalgo:
        return True
    salgo = str(((exp.get("problem") or {}).get("algorithm_id") or "")).lower()
    return (not salgo) or salgo == qalgo


def retrieve_experiences(
    bank: ExperienceBank,
    query: Dict[str, Any],
    *,
    k: int = 3,
    rerank_after: int = 24,
    target_trials: Optional[Sequence[Dict[str, Any]]] = None,
    candidate_pool: int = 12,
    spearman_mix: float = 0.4,
    drop_negative_spearman: bool = True,
    exclude_self: bool = True,
    ranker: Optional[str] = None,
    require_usable: bool = True,
    spearman_floor: Optional[float] = None,
    dedup: bool = True,
) -> List[RetrievalHit]:
    """fANOVA-usable pool → GP similar-task (Spearman backup) → floor → top k.

    Unusable bank cards never enter ranking. Weak-sim neighbors are not used to
    fill k. Empty is allowed.
    """
    knobs = load_retrieve_knobs()
    k_max = max(0, int(k))
    if spearman_floor is None:
        spearman_floor = float(knobs["spearman_floor"])
    ranker = str(ranker or knobs.get("ranker") or "surrogate").strip().lower()
    n_local = len(target_trials or [])
    weights = dict(COLD_WEIGHTS if n_local < rerank_after else bank.weights)
    hits: List[RetrievalHit] = []
    for exp in bank.experiences:
        if not _filters_ok(exp, query):
            continue
        if exclude_self and _is_self(exp, query):
            continue
        if not _same_algo_ok(exp, query):
            continue
        if not _usable_ok(exp, require_usable):
            continue
        feats = cold_features(query, exp)
        score = combine_score(feats, weights)
        hits.append(RetrievalHit(experience=exp, score=score, features=feats))
    hits.sort(key=lambda h: h.score, reverse=True)

    if ranker == "surrogate":
        hits = _rerank_surrogate(
            hits,
            query,
            target_trials or [],
            k=max(k_max, candidate_pool),
            candidate_pool=candidate_pool,
            min_prefix=3,
            spearman_floor=float(spearman_floor),
            spearman_mix=spearman_mix,
        )
        if dedup:
            hits = _dedup_hits(hits)
        return hits[:k_max]

    if target_trials is not None and n_local >= rerank_after and hits:
        pool = hits[: max(candidate_pool, k_max)]
        reranked: List[RetrievalHit] = []
        tgt_algo = str(query.get("algorithm_id") or "")
        tgt_hib = query.get("higher_is_better")
        floor = float(spearman_floor)
        for h in pool:
            rho = spearman_transfer(
                h.experience,
                target_trials,
                target_algo=tgt_algo or None,
                target_higher_is_better=None if tgt_hib is None else bool(tgt_hib),
            )
            h.spearman = rho
            if drop_negative_spearman and rho < 0.0:
                continue
            if rho < floor:
                continue
            h.score = (1.0 - spearman_mix) * h.score + spearman_mix * max(0.0, rho)
            reranked.append(h)
        reranked.sort(key=lambda h: h.score, reverse=True)
        hits = reranked
    else:
        hits = hits[: max(candidate_pool, k_max)]
    if dedup:
        hits = _dedup_hits(hits)
    return hits[:k_max]


def _load_landscape_cache(prefix: str):
    """Prefer the full landscape cache (eval targets included) over holdout."""
    from mindware.components.experience.surrogate_landscape import LandscapeCache

    meta = Path(__file__).resolve().parents[3] / "history_data" / "meta_resource"
    for name in (f"surrogate_landscape_{prefix}.pkl", f"surrogate_landscape_holdout_{prefix}.pkl"):
        path = meta / name
        if path.is_file():
            return LandscapeCache(path)
    return None


def _rerank_surrogate(
    hits: List[RetrievalHit],
    query: Dict[str, Any],
    target_trials: Sequence[Dict[str, Any]],
    *,
    k: int,
    candidate_pool: int,
    min_prefix: int = 3,
    spearman_floor: float = 0.5,
    spearman_mix: float = 0.4,
) -> List[RetrievalHit]:
    """Rank neighbors by GP landscape sim from trial 1.

    Primary: precomputed target vector in surrogate_landscape_{cls,rgs}.pkl.
    Fallback: fit a GP on the live prefix once it has ``min_prefix`` trials.
    """
    if not hits:
        return hits
    pool = hits[: max(candidate_pool, k)]
    n_local = len(target_trials or [])
    cache = None
    qvec = None
    algo = str(query.get("algorithm_id") or "").lower()
    target_ds = str(query.get("dataset_id") or "").lower()
    try:
        from mindware.components.experience.surrogate_landscape import (
            get_algo_cs,
            predict_landscape,
            sample_query_configs,
            spearman_sim,
        )

        task = str((query.get("filters") or {}).get("task_type") or query.get("task_type") or "").lower()
        prefix = "cls" if "class" in task or task == "cls" else "rgs"
        cache = _load_landscape_cache(prefix)
        if cache is not None and target_ds:
            qvec = cache.pred(algo, target_ds)
            if qvec is not None:
                for h in pool:
                    h.features["landscape_source"] = "cache"
        if qvec is None and n_local >= int(min_prefix) and cache is not None:
            n_query = int((cache.payload.get("hyperparams") or {}).get("n_query_configs") or 25)
            cs = get_algo_cs(algo, task)
            if cs is not None:
                qcfgs = sample_query_configs(cs, n_query)
                qvec = predict_landscape(list(target_trials), cs, algo, qcfgs, min_trials=min_prefix)
                if qvec is not None:
                    for h in pool:
                        h.features["landscape_source"] = "local_gp"
    except Exception:
        cache = None
        qvec = None
        spearman_sim = None  # type: ignore

    if qvec is None or cache is None:
        return pool[:k]

    tgt_hib = query.get("higher_is_better")
    floor = float(spearman_floor)
    mix = float(spearman_mix)
    reranked: List[RetrievalHit] = []
    for h in pool:
        ds = str(((h.experience.get("problem") or {}).get("dataset_id") or "")).lower()
        sim: Optional[float] = None
        if qvec is not None and cache is not None:
            svec = cache.pred(algo, ds)
            if svec is not None:
                sim = float(spearman_sim(qvec, svec))
                h.features["surrogate_landscape_sim"] = sim
        if sim is None:
            sim = float(
                spearman_transfer(
                    h.experience,
                    target_trials,
                    target_algo=algo or None,
                    target_higher_is_better=None if tgt_hib is None else bool(tgt_hib),
                )
            )
            h.features["spearman_backup"] = 1.0
        h.spearman = sim
        if sim < floor:
            continue
        h.score = (1.0 - mix) * float(h.score) + mix * max(0.0, sim)
        reranked.append(h)
    reranked.sort(key=lambda h: h.score, reverse=True)
    return reranked


def _prompt_config_for_target(
    cfg: Dict[str, Any],
    target_name_by_role: Dict[str, str],
) -> Dict[str, Any]:
    """Map a source trial config onto target HP names (strip algo prefix)."""
    out: Dict[str, Any] = {}
    for k, v in (cfg or {}).items():
        stem = _hp_stem(k)
        if stem == "algorithm" or k == "algorithm":
            continue
        role = infer_role(k)
        name = target_name_by_role.get(role) if role else None
        if not name:
            name = stem
        out[name] = v
    return out


def coerce_retrieve_k(val, default: int = 1) -> int:
    """Parse a top/bottom-k; 0 is allowed (omit that side)."""
    if val is None or val == "":
        return default
    return max(0, int(val))


def _is_prefer_advice_line(ln: str) -> bool:
    s = ln.strip().lower()
    return s.startswith("- ") and "prefer" in s


def _fmt_hp_num(x: Any) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if abs(v - round(v)) < 1e-9 and abs(v) >= 1.0:
        return str(int(round(v)))
    return f"{v:.4g}"


def _keep_hp_stems(exp: Dict[str, Any]) -> List[str]:
    keep = list(((exp.get("knowledge") or {}).get("fanova") or {}).get("keep_hps") or [])
    if not keep:
        keep = [
            str(f.get("name") or "")
            for f in ((exp.get("knowledge") or {}).get("factors") or [])
        ]
    out: List[str] = []
    seen = set()
    for name in keep:
        stem = _hp_stem(str(name))
        if not stem or stem in SKIP_HP_NAMES or stem == "algorithm" or stem in seen:
            continue
        seen.add(stem)
        out.append(stem)
    return out


def _trial_groups_by_score(
    trials: Sequence[Dict[str, Any]],
    *,
    hib: bool,
    frac: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ordered = sorted(trials, key=lambda t: float(t.get("score") or 0.0), reverse=hib)
    n = len(ordered)
    if n < 2:
        return [], []
    k = max(1, int(math.floor(float(frac) * n)))
    if 2 * k > n:
        k = max(1, n // 2)
    tops = list(ordered[:k])
    bottoms = list(reversed(ordered[-k:]))
    top_ids = {id(t) for t in tops}
    bottoms = [t for t in bottoms if id(t) not in top_ids]
    return tops, bottoms


def _hp_vals_in_group(group: Sequence[Dict[str, Any]], stem: str) -> List[Any]:
    vals: List[Any] = []
    for t in group:
        cfg = t.get("config") or {}
        found = None
        for k, v in cfg.items():
            if _hp_stem(str(k)) == stem:
                found = v
                break
        if found is None:
            continue
        if isinstance(found, bool) or found is None:
            vals.append(found)
            continue
        if isinstance(found, (int, float)) and not isinstance(found, bool):
            vals.append(float(found))
            continue
        try:
            vals.append(float(found))
        except (TypeError, ValueError):
            vals.append(found)
    return vals


def _disjoint_numeric_ranges(
    top_vals: Sequence[float],
    bot_vals: Sequence[float],
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    tlo, thi = min(top_vals), max(top_vals)
    blo, bhi = min(bot_vals), max(bot_vals)
    if thi < blo or bhi < tlo:
        return (tlo, thi), (blo, bhi)
    t_med = sorted(top_vals)[len(top_vals) // 2]
    b_med = sorted(bot_vals)[len(bot_vals) // 2]
    if abs(t_med - b_med) < 1e-12:
        return None
    split = 0.5 * (t_med + b_med)
    if t_med > b_med:
        top_f = [v for v in top_vals if v >= split]
        bot_f = [v for v in bot_vals if v < split]
    else:
        top_f = [v for v in top_vals if v < split]
        bot_f = [v for v in bot_vals if v >= split]
    if not top_f or not bot_f:
        return None
    tr = (min(top_f), max(top_f))
    br = (min(bot_f), max(bot_f))
    if tr[1] < br[0] or br[1] < tr[0] or tr[1] == br[0] or br[1] == tr[0]:
        return tr, br
    return None


def _fmt_closed_range(lo: float, hi: float) -> str:
    if abs(float(hi) - float(lo)) < 1e-12:
        return _fmt_hp_num(lo)
    return f"{_fmt_hp_num(lo)}–{_fmt_hp_num(hi)}"


def _exclusive_ranges(vals: Sequence[float], overlap_lo: float, overlap_hi: float) -> List[Tuple[float, float]]:
    left = [v for v in vals if v < overlap_lo - 1e-12]
    right = [v for v in vals if v > overlap_hi + 1e-12]
    parts: List[Tuple[float, float]] = []
    if left:
        parts.append((min(left), max(left)))
    if right:
        parts.append((min(right), max(right)))
    return parts


def three_segment_numeric_ranges(
    top_vals: Sequence[float],
    bot_vals: Sequence[float],
) -> Dict[str, Any]:
    """Split one HP into exclusive-top / mixed overlap / exclusive-bottom.

    Bounding-box overlap is mixed (ambiguous). Values strictly outside that
    interval stay in top or bottom. Example: top {1,2,3,10}, bottom {5,6,7,15}
    → top 1–3, mixed 5–10, bottom 15.
    """
    tlo, thi = min(top_vals), max(top_vals)
    blo, bhi = min(bot_vals), max(bot_vals)
    overlap_lo, overlap_hi = max(tlo, blo), min(thi, bhi)
    if overlap_lo > overlap_hi + 1e-12:
        return {
            "top": [(tlo, thi)],
            "mixed": None,
            "bottom": [(blo, bhi)],
        }
    return {
        "top": _exclusive_ranges(top_vals, overlap_lo, overlap_hi),
        "mixed": (overlap_lo, overlap_hi),
        "bottom": _exclusive_ranges(bot_vals, overlap_lo, overlap_hi),
    }


def quantile_region_lines(
    exp: Dict[str, Any],
    target_name_by_role: Dict[str, str],
    *,
    frac: float = 0.2,
) -> List[str]:
    """Top/bottom score-quantile HP ranges; skip an HP if the two ranges would overlap."""
    trials = (exp.get("retrieval") or {}).get("trials") or []
    hib = higher_is_better_of(exp, True)
    tops, bottoms = _trial_groups_by_score(trials, hib=hib, frac=frac)
    if not tops or not bottoms:
        return []
    prob = exp.get("problem") or {}
    ds = str(prob.get("dataset_id") or "?")
    algo = str(prob.get("algorithm_id") or "?")
    pct = int(round(float(frac) * 100))
    lines = [f"Similar task: {ds} ({algo})"]
    for stem in _keep_hp_stems(exp):
        role = infer_role(stem)
        shown = (target_name_by_role.get(role) if role else None) or stem
        tv = _hp_vals_in_group(tops, stem)
        bv = _hp_vals_in_group(bottoms, stem)
        if not tv or not bv:
            continue
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in tv + bv)
        if numeric:
            segs = three_segment_numeric_ranges([float(v) for v in tv], [float(v) for v in bv])
            bits: List[str] = []
            if segs["top"]:
                bits.append("top " + ", ".join(_fmt_closed_range(a, b) for a, b in segs["top"]))
            if segs["mixed"] is not None:
                bits.append("mixed " + _fmt_closed_range(*segs["mixed"]))
            if segs["bottom"]:
                bits.append("bottom " + ", ".join(_fmt_closed_range(a, b) for a, b in segs["bottom"]))
            if not bits:
                continue
            lines.append(f"- {shown}: " + "; ".join(bits))
            continue
        tset, bset = set(map(str, tv)), set(map(str, bv))
        t_only, b_only, both = tset - bset, bset - tset, tset & bset
        bits = []
        if t_only:
            bits.append("top {" + ", ".join(sorted(t_only)) + "}")
        if both:
            bits.append("mixed {" + ", ".join(sorted(both)) + "}")
        if b_only:
            bits.append("bottom {" + ", ".join(sorted(b_only)) + "}")
        if not bits:
            continue
        lines.append(f"- {shown}: " + "; ".join(bits))
    return lines if len(lines) > 1 else []


def _trial_config_line(
    t: Dict[str, Any],
    target_name_by_role: Dict[str, str],
    *,
    tag: Optional[str] = None,
) -> Optional[str]:
    cfg = _prompt_config_for_target(t.get("config") or {}, target_name_by_role)
    if not cfg:
        return None
    score = float(t.get("score") or 0.0)
    trial = t.get("trial", "?")
    prefix = f"- {tag} trial=" if tag else "- trial="
    return (
        f"{prefix}{trial}, score={score:.4g}, "
        f"config={json.dumps(cfg, ensure_ascii=True, sort_keys=True, default=str)}"
    )


def top_bottom_trial_lines(
    exp: Dict[str, Any],
    target_name_by_role: Dict[str, str],
    *,
    top_k: int = 1,
    bottom_k: int = 1,
    layout: str = "flat",
) -> List[str]:
    """Best/worst trial configs from a similar task's stored history."""
    trials = (exp.get("retrieval") or {}).get("trials") or []
    if len(trials) < 2 or (top_k <= 0 and bottom_k <= 0):
        return []
    hib = higher_is_better_of(exp, True)
    ordered = sorted(trials, key=lambda t: float(t.get("score") or 0.0), reverse=hib)
    tops = ordered[: max(0, min(int(top_k), len(ordered)))]
    top_ids = {id(t) for t in tops}
    n_bot = max(0, int(bottom_k))
    bottoms = [t for t in ordered[-n_bot:] if id(t) not in top_ids] if n_bot else []
    if n_bot > 0 and not bottoms:
        bottoms = ordered[-n_bot:]
    bottoms = list(reversed(bottoms))  # worst first
    lines: List[str] = []
    grouped = str(layout or "flat").lower() == "grouped"
    if grouped:
        top_rows: List[str] = []
        for t in tops:
            ln = _trial_config_line(t, target_name_by_role)
            if ln:
                top_rows.append(ln)
        bot_rows: List[str] = []
        for t in bottoms:
            ln = _trial_config_line(t, target_name_by_role)
            if ln:
                bot_rows.append(ln)
        if top_rows:
            lines.append(f"top{len(top_rows)}:")
            lines.extend(top_rows)
        if bot_rows:
            lines.append(f"bottom{len(bot_rows)}:")
            lines.extend(bot_rows)
        return lines
    for tag, group in (("top", tops), ("bottom", bottoms)):
        for t in group:
            ln = _trial_config_line(t, target_name_by_role, tag=tag)
            if ln:
                lines.append(ln)
    return lines


def rewrite_advice_by_role(exp: Dict[str, Any], target_name_by_role: Dict[str, str]) -> str:
    """Rewrite source HP names onto the target algorithm via shared roles."""
    block = str((exp.get("delivery") or {}).get("prompt_block") or "")
    src_names: Dict[str, str] = {}
    for fac in ((exp.get("problem") or {}).get("space") or {}).get("factors") or []:
        role = fac.get("role") or infer_role(str(fac.get("name") or ""))
        if role and fac.get("name"):
            src_names[str(fac["name"])] = role
    for fac in (exp.get("knowledge") or {}).get("factors") or []:
        role = fac.get("role") or infer_role(str(fac.get("name") or ""))
        if role and fac.get("name"):
            src_names.setdefault(str(fac["name"]), role)
    for src_name, role in sorted(src_names.items(), key=lambda kv: -len(kv[0])):
        tgt = target_name_by_role.get(role)
        if not tgt:
            continue
        seen = set()
        for src in (src_name, _hp_stem(src_name)):
            if not src or src == tgt or src in seen:
                continue
            seen.add(src)
            # Do not match a stem inside an already-prefixed name (catboost:learning_rate).
            block = re.sub(rf"(?<![:\\w]){re.escape(src)}(?!\\w)", tgt, block)
    return block


def drop_flipped_factor_lines(
    block: str,
    exp: Dict[str, Any],
    target_name_by_role: Dict[str, str],
    target_role_signs: Dict[str, int],
) -> str:
    """Drop factor/avoid lines whose role direction disagrees with the target run (Label B)."""
    src_signs = role_signs_from_experience(exp)
    flipped = {r for r, s in src_signs.items() if r in target_role_signs and target_role_signs[r] != s}
    if not flipped:
        _LABEL_B_STATS["n_lines_seen"] += len(block.splitlines())
        return block
    drop_names: set = set()
    for fac in ((exp.get("problem") or {}).get("space") or {}).get("factors") or []:
        role = fac.get("role") or infer_role(str(fac.get("name") or ""))
        if role not in flipped:
            continue
        name = str(fac.get("name") or "")
        if name:
            drop_names.add(name)
            drop_names.add(_hp_stem(name))
        tgt = target_name_by_role.get(role)
        if tgt:
            drop_names.add(tgt)
            drop_names.add(_hp_stem(tgt))
    for fac in (exp.get("knowledge") or {}).get("factors") or []:
        role = fac.get("role") or infer_role(str(fac.get("name") or ""))
        if role in flipped and fac.get("name"):
            drop_names.add(str(fac["name"]))
            drop_names.add(_hp_stem(str(fac["name"])))
    kept: List[str] = []
    dropped = 0
    for ln in block.splitlines():
        stripped = ln.strip().lower()
        factorish = (
            stripped.startswith("- factor ")
            or stripped.startswith("- avoid:")
            or (stripped.startswith("- ") and "prefer" in stripped)
        )
        if factorish and any(n and n in ln for n in sorted(drop_names, key=len, reverse=True)):
            dropped += 1
            continue
        kept.append(ln)
    _LABEL_B_STATS["n_lines_seen"] += len(block.splitlines())
    _LABEL_B_STATS["n_lines_dropped"] += dropped
    if dropped:
        _LABEL_B_STATS["n_hits_with_drop"] += 1
    return "\n".join(kept)


_PREFER_LINE = re.compile(
    r"^(?P<indent>\s*-\s+)(?P<hp>[^:]+):\s+prefer\s+(?P<dir>lower|higher)(?P<rest>.*)$",
    re.IGNORECASE,
)


def resolve_prefer_clashes(
    cards: Sequence[Tuple[float, List[str]]],
    *,
    gap: float = 0.15,
) -> List[List[str]]:
    """Keep tasks; drop a clashing HP (or keep the higher-sim card's line if the gap is large)."""
    if len(cards) < 2:
        return [list(lines) for _, lines in cards]
    out = [list(lines) for _, lines in cards]
    mentions: Dict[str, List[Tuple[int, str]]] = {}
    for i, lines in enumerate(out):
        for ln in lines:
            m = _PREFER_LINE.match(ln)
            if not m:
                continue
            key = _hp_stem(m.group("hp")).lower()
            mentions.setdefault(key, []).append((i, m.group("dir").lower()))
    drop_all: set = set()
    keep_winner: Dict[str, int] = {}
    for hp, rows in mentions.items():
        dirs = {d for _, d in rows}
        if len(dirs) < 2:
            continue
        best: Dict[str, Tuple[float, int]] = {}
        for i, direction in rows:
            sim = float(cards[i][0])
            prev = best.get(direction)
            if prev is None or sim > prev[0]:
                best[direction] = (sim, i)
        ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)
        win_sim, win_i = ranked[0][1]
        lose_sim = ranked[1][1][0]
        if (win_sim - lose_sim) >= float(gap):
            keep_winner[hp] = win_i
        else:
            drop_all.add(hp)

    def keep_line(card_i: int, ln: str) -> bool:
        m = _PREFER_LINE.match(ln)
        if not m:
            return True
        key = _hp_stem(m.group("hp")).lower()
        if key in drop_all:
            return False
        if key in keep_winner and keep_winner[key] != card_i:
            return False
        return True

    cleaned: List[List[str]] = []
    for i, lines in enumerate(out):
        had_prefer = any(_PREFER_LINE.match(ln) for ln in lines)
        kept = [ln for ln in lines if keep_line(i, ln)]
        has_prefer = any(_PREFER_LINE.match(ln) for ln in kept)
        if had_prefer and not has_prefer:
            continue
        cleaned.append(kept)
    return cleaned


def format_retrieved_block(
    hits: Sequence[RetrievalHit],
    target_name_by_role: Optional[Dict[str, str]] = None,
    target_role_signs: Optional[Dict[str, int]] = None,
    clash_gap: Optional[float] = None,
    *,
    include_top_bottom: bool = False,
    retrieve_top_k: int = 1,
    retrieve_bottom_k: int = 1,
    include_prefer: bool = True,
    region_style: str = "prefer",
    tb_layout: str = "flat",
    quantile_frac: float = 0.2,
) -> str:
    """LLM-facing text: dataset + HP region lines; optional top/bottom trial configs."""
    if not hits:
        return ""
    names = target_name_by_role or {}
    if clash_gap is None:
        clash_gap = float(load_retrieve_knobs().get("clash_gap") or 0.15)
    top_k = coerce_retrieve_k(retrieve_top_k, 1)
    bot_k = coerce_retrieve_k(retrieve_bottom_k, 1)
    style = str(region_style or "prefer").strip().lower()
    layout = str(tb_layout or "flat").strip().lower()
    staged: List[Tuple[float, List[str]]] = []
    for h in hits:
        _LABEL_B_STATS["n_hits"] += 1
        if style in ("quantile20", "quantile"):
            lines = quantile_region_lines(h.experience, names, frac=float(quantile_frac or 0.2))
            if not lines:
                prob = h.experience.get("problem") or {}
                lines = [
                    f"Similar task: {prob.get('dataset_id') or '?'} ({prob.get('algorithm_id') or '?'})"
                ]
        else:
            text = rewrite_advice_by_role(h.experience, names)
            if target_role_signs:
                text = drop_flipped_factor_lines(text, h.experience, names, target_role_signs)
            lines = [ln for ln in text.splitlines() if ln.strip()]
            if not include_prefer:
                lines = [ln for ln in lines if not _is_prefer_advice_line(ln)]
            has_v3 = any(_is_prefer_advice_line(ln) for ln in lines)
            if not lines:
                continue
            if not has_v3:
                lines = lines[:8]
        if include_top_bottom:
            lines.extend(
                top_bottom_trial_lines(
                    h.experience,
                    names,
                    top_k=top_k,
                    bottom_k=bot_k,
                    layout=layout,
                )
            )
        if not any(ln.strip() for ln in lines):
            continue
        sim = float(h.spearman) if h.spearman is not None else float(h.score)
        staged.append((sim, lines))
    cards = resolve_prefer_clashes(staged, gap=float(clash_gap))
    return "\n\n".join("\n".join(lines) for lines in cards if lines)


def target_roles_from_config_space(config_space, algorithm: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        names = [hp.name for hp in config_space.get_hyperparameters()]
    except Exception:
        return out
    for name in names:
        if name == "algorithm":
            continue
        role = infer_role(name)
        if role and role not in out:
            out[role] = name
    return out


def subsample_trials(
    trials: Sequence[Dict[str, Any]],
    max_trials: int = 40,
    higher_is_better: bool = True,
) -> List[Dict[str, Any]]:
    if max_trials <= 0 or len(trials) <= max_trials:
        compact = list(trials)
    else:
        n = len(trials)
        idxs = sorted(set(int(round(i * (n - 1) / (max_trials - 1))) for i in range(max_trials)))
        compact = [trials[i] for i in idxs]
        if higher_is_better:
            best = max(trials, key=lambda t: float(t.get("score") or 0.0))
        else:
            best = min(trials, key=lambda t: float(t.get("score") or 0.0))
        if best not in compact:
            compact[-1] = best
    out = []
    for t in compact:
        cfg = t.get("config") or {}
        out.append({"trial": t.get("trial"), "config": dict(cfg), "score": float(t["score"])})
    return out


def attach_retrieval_index(
    experience: Dict[str, Any],
    trials: Sequence[Dict[str, Any]],
    *,
    profile_vec: Optional[Dict[str, float]] = None,
    max_trials: int = 40,
    tabpfn_vec: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Mutate experience with compact fields used by the retrieval cascade.

    ``max_trials <= 0`` stores the full history (v3 bank). Default 40 keeps v2 compact.
    """
    hib = higher_is_better_of(experience, True)
    prev = experience.get("retrieval") or {}
    experience["retrieval"] = {
        "importance_by_role": importance_by_role(experience),
        "roles": roles_of(experience),
        "profile_vec": dict(
            profile_vec or prev.get("profile_vec") or (experience.get("problem") or {}).get("profile") or {}
        ),
        "trials": subsample_trials(trials, max_trials=max_trials, higher_is_better=hib),
        "tabpfn_vec": tabpfn_vec if tabpfn_vec is not None else prev.get("tabpfn_vec"),
    }
    return experience


def attach_semantic_vectors(
    query: Dict[str, Any],
    bank: Optional["ExperienceBank"] = None,
    *,
    cache=None,
    vae=None,
) -> Dict[str, Any]:
    """Fill query (and bank experiences) with semantic / VAE vectors when missing."""
    try:
        from mindware.components.experience.semantic_embed import (
            EmbeddingCache,
            TinyVAE,
            default_cache_path,
            default_vae_path,
            embed_profile,
            hash_embed,
            pad_views,
            profile_text,
        )
        import pickle as _pkl
    except Exception:
        return query
    if cache is None:
        prefix = "cls"
        task_hint = str((query.get("filters") or {}).get("task_type") or "")
        if "reg" in task_hint:
            prefix = "rgs"
        cpath = default_cache_path(prefix)
        if cpath.is_file():
            cache = EmbeddingCache(cpath)
    if vae is None:
        prefix = "cls"
        task_hint = str((query.get("filters") or {}).get("task_type") or "")
        if "reg" in task_hint:
            prefix = "rgs"
        vpath = default_vae_path(prefix)
        if vpath.is_file():
            try:
                vae = _pkl.loads(vpath.read_bytes())
            except Exception:
                vae = None
    task = str((query.get("filters") or {}).get("task_type") or query.get("task_type") or "")
    ds = str(query.get("dataset_id") or "")
    if not query.get("semantic_vec"):
        cached = cache.get(task, ds) if cache is not None and ds else None
        if cached:
            query["semantic_vec"] = list(cached)
        else:
            query["semantic_vec"] = embed_profile(
                ds,
                query.get("dataset_profile") or query.get("profile_vec"),
                task,
                algorithm_id=query.get("algorithm_id"),
            )
            if cache is not None and ds:
                cache.put(task, ds, query["semantic_vec"])
    if bank is not None:
        for exp in bank.experiences:
            ret = exp.setdefault("retrieval", {})
            if ret.get("semantic_vec"):
                continue
            prob = exp.get("problem") or {}
            eds = str(prob.get("dataset_id") or "")
            etask = str(prob.get("task_type") or task)
            cached = cache.get(etask, eds) if cache is not None and eds else None
            if cached:
                ret["semantic_vec"] = list(cached)
                continue
            text = profile_text(eds, ret.get("profile_vec") or (prob.get("profile") or {}), etask)
            ret["semantic_vec"] = hash_embed(text)
            if cache is not None and eds:
                cache.put(etask, eds, ret["semantic_vec"])
    if vae is None:
        return query
    q_view = pad_views(query.get("semantic_vec"), list(profile_vector(query.get("profile_vec") or {})))
    if q_view:
        query["vae_vec"] = vae.encode(q_view)
    if bank is not None:
        for exp in bank.experiences:
            ret = exp.setdefault("retrieval", {})
            if ret.get("vae_vec"):
                continue
            view = pad_views(ret.get("semantic_vec"), list(profile_vector(ret.get("profile_vec") or {})))
            if view:
                ret["vae_vec"] = vae.encode(view)
    return query


def build_run_query(
    *,
    dataset_id: Optional[str],
    algorithm_id: Optional[str],
    task_type: Optional[str],
    dataset_profile: Optional[Dict[str, Any]],
    successes: Sequence[Dict[str, Any]],
    config_space=None,
) -> Dict[str, Any]:
    from mindware.components.experience.profile import profile_from_dataset_profile

    roles = list(target_roles_from_config_space(config_space, algorithm_id or "").keys()) if config_space is not None else []
    if not roles:
        roles = sorted({infer_role(k) for t in successes for k in (t.get("config") or {}) if infer_role(k)})
        roles = [r for r in roles if r]
    packed = [{"trial": t.get("trial", i), "config": t["config"], "score": t["score"]} for i, t in enumerate(successes, 1)]
    # Live optimizer scores are always higher-is-better (rgs uses negated MSE).
    hib = True
    return {
        "dataset_id": dataset_id,
        "algorithm_id": algorithm_id,
        "filters": {
            "domain": "tabular_ml",
            "task_type": task_type,
            "representation": "structured_config",
        },
        "higher_is_better": hib,
        "importance_by_role": query_importance_from_trials(packed, higher_is_better=hib),
        "roles": roles,
        "dataset_profile": dataset_profile,
        "profile_vec": profile_from_dataset_profile(dataset_profile)
        if dataset_profile
        else {},
    }


def retrieve_for_run(
    bank: ExperienceBank,
    *,
    dataset_id: Optional[str],
    algorithm_id: Optional[str],
    task_type: Optional[str],
    dataset_profile: Optional[Dict[str, Any]],
    successes: Sequence[Dict[str, Any]],
    config_space=None,
    k: int = 3,
    start_trial: int = 1,
    rerank_after: int = 24,
    current_trial: int = 1,
    fade_after: Optional[int] = None,
    stop_trial: Optional[int] = None,
    ranker: Optional[str] = None,
    require_usable: Optional[bool] = None,
    spearman_floor: Optional[float] = None,
    dedup: Optional[bool] = None,
    include_top_bottom: bool = False,
    retrieve_top_k: int = 1,
    retrieve_bottom_k: int = 1,
    include_prefer: bool = True,
    region_style: str = "prefer",
    tb_layout: str = "flat",
    quantile_frac: float = 0.2,
    audit: Optional[Dict[str, Any]] = None,
    seed: int = 1,
) -> Tuple[str, List[RetrievalHit]]:
    if ranker == "source_gp":
        from mindware.components.experience.source_gp import SourceGPError, rs_position
        state = rs_position(current_trial, start_trial, stop_trial or 9)
        if audit is not None:
            audit.update(state)
        if bank.source_gp is None:
            raise SourceGPError("source_gp retrieval has no verified source model artifact")
        if not state["window_active"]:
            return "", []
        trial_ids = [int(row["trial"]) for row in successes]
        if (trial_ids != sorted(set(trial_ids)) or
                any(trial < 1 or trial >= current_trial for trial in trial_ids)):
            raise SourceGPError("RS observations must be unique completed evaluations before the current proposal")
        block, hits, details = bank.source_gp.rank(
            dataset_id, algorithm_id, task_type, list(successes), k=k,
            floor=0.5 if spearman_floor is None else float(spearman_floor),
            seed=seed,
        )
        if audit is not None:
            audit.update(details)
        return block, hits
    if current_trial < max(1, int(start_trial or 1)):
        return "", []
    if stop_trial is not None and current_trial >= int(stop_trial):
        return "", []
    knobs = load_retrieve_knobs()
    if require_usable is None:
        require_usable = bool(knobs["require_usable"])
    if spearman_floor is None:
        spearman_floor = float(knobs["spearman_floor"])
    if dedup is None:
        dedup = bool(knobs["dedup"])
    if ranker is None:
        ranker = str(knobs.get("ranker") or "surrogate")
    k_eff = min(int(k), int(knobs.get("k_max") or k))
    if fade_after is not None and current_trial >= int(fade_after):
        k_eff = max(1, k_eff - 1)
    query = build_run_query(
        dataset_id=dataset_id,
        algorithm_id=algorithm_id,
        task_type=task_type,
        dataset_profile=dataset_profile,
        successes=successes,
        config_space=config_space,
    )
    attach_semantic_vectors(query, bank)
    packed = [{"trial": t.get("trial", i), "config": t["config"], "score": t["score"]} for i, t in enumerate(successes, 1)]
    hits = retrieve_experiences(
        bank,
        query,
        k=k_eff,
        rerank_after=rerank_after,
        target_trials=packed if packed else None,
        ranker=ranker,
        require_usable=require_usable,
        spearman_floor=spearman_floor,
        dedup=dedup,
    )
    names = target_roles_from_config_space(config_space, algorithm_id or "") if config_space is not None else {}
    tgt_signs = role_signs_from_trials(packed, higher_is_better=True) if packed else {}
    clash_gap = float(knobs.get("clash_gap") or 0.15)
    return format_retrieved_block(
        hits,
        names,
        target_role_signs=tgt_signs,
        clash_gap=clash_gap,
        include_top_bottom=include_top_bottom,
        retrieve_top_k=retrieve_top_k,
        retrieve_bottom_k=retrieve_bottom_k,
        include_prefer=include_prefer,
        region_style=region_style,
        tb_layout=tb_layout,
        quantile_frac=quantile_frac,
    ), hits
