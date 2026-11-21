"""TPE / trust-region TPE / CMA-ES / HEBO / random / LLM-semantic hybrid loop over a DomainAdapter."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from mindware.components.optimizers.hybrid_selector import selector_from_params
from mindware.domains.base import DomainAdapter, config_to_dict, z_from_config, z_to_config
from mindware.domains.bo_checkpoint import (
    load_optuna_study,
    load_pending,
    load_smac_advisor,
    persist_search_checkpoint,
    save_optuna_study,
    write_pending,
)
from mindware.domains.evaluator import DomainEvaluator
from mindware.domains.history import observation, write_openbox_history, write_rotating_openbox_history
from mindware.domains.prompts import build_domain_messages, parse_semantic_action
from mindware.domains.protein.bolt_peptide_e2e import (
    load_peptide_vae_checkpoint,
    peptide_vae_checkpoint_path,
    reencode_observation_latents,
    save_peptide_vae_checkpoint,
    sequences_for_peptide_e2e,
    TPE_RUN_CHECKPOINT_INTERVAL,
)
from mindware.domains.protein.bolt_seed0_init import BOLT_SEED0_HPARAMS


def _search_objective_from_eval(last: Any, feasible: bool) -> float:
    """Minimize runtime for query execution oracle; censored rows keep penalty runtime."""
    if last is None:
        return 1e10
    if feasible:
        return float(last.objective)
    info = getattr(last, "info", None) or {}
    if info.get("oracle") == "bolt_execute":
        raw = info.get("raw_runtime", info.get("raw_cost"))
        if raw is not None:
            val = float(raw)
            if val < 1e9:
                return val
        return float(last.objective)
    return 1e10


def _cfg_from_dict(cs: ConfigurationSpace, values: Dict[str, Any]) -> Dict[str, Any]:
    filled = {}
    for hp in cs.get_hyperparameters():
        if hp.name in values and values[hp.name] is not None:
            filled[hp.name] = values[hp.name]
        else:
            filled[hp.name] = hp.default_value
    try:
        cfg = Configuration(cs, values=filled)
        return dict(cfg.get_dictionary())
    except Exception:
        return filled


def _suggest_hp_optuna(trial, hp) -> Any:
    name = hp.name
    if isinstance(hp, UniformFloatHyperparameter):
        return trial.suggest_float(name, float(hp.lower), float(hp.upper), log=bool(getattr(hp, "log", False)))
    if isinstance(hp, UniformIntegerHyperparameter):
        return trial.suggest_int(name, int(hp.lower), int(hp.upper))
    if isinstance(hp, CategoricalHyperparameter):
        return trial.suggest_categorical(name, list(hp.choices))
    return hp.default_value


def _cash_parent_name(cs: ConfigurationSpace) -> Optional[str]:
    names = {hp.name for hp in cs.get_hyperparameters()}
    for candidate in ("optimizer_family", "algorithm"):
        if candidate in names:
            return candidate
    return None


def _conditions_satisfied(cs: ConfigurationSpace, hp_name: str, values: Dict[str, Any]) -> bool:
    try:
        conditions = cs.get_parent_conditions_of(hp_name)
    except Exception:
        return True
    for condition in conditions:
        parent = getattr(condition, "parent", None)
        parent_name = getattr(parent, "name", None)
        if parent_name is None or parent_name not in values:
            return False
        actual = values[parent_name]
        class_name = condition.__class__.__name__
        if "InCondition" in class_name:
            if actual not in getattr(condition, "values", []):
                return False
        elif "NotEqualsCondition" in class_name:
            if actual == getattr(condition, "value", None):
                return False
        else:
            expected = getattr(condition, "value", None)
            if expected is not None and actual != expected:
                return False
    return True


def _suggest_optuna(trial, cs: ConfigurationSpace) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    parent = _cash_parent_name(cs)
    if parent:
        hp = cs.get_hyperparameter(parent)
        values[parent] = _suggest_hp_optuna(trial, hp)
        family = values[parent]
        hps = list(cs.get_hyperparameters())
        for _ in range(3):
            changed = False
            for hp in hps:
                name = hp.name
                if name in values:
                    continue
                if ":" in name and family is not None and not name.startswith(f"{family}:"):
                    continue
                if not _conditions_satisfied(cs, name, values):
                    continue
                values[name] = _suggest_hp_optuna(trial, hp)
                changed = True
            if not changed:
                break
        try:
            cfg = Configuration(cs, values=values)
            return dict(cfg.get_dictionary())
        except Exception:
            return values
    for hp in cs.get_hyperparameters():
        values[hp.name] = _suggest_hp_optuna(trial, hp)
    return values


def _optuna_distributions(cs: ConfigurationSpace) -> Dict[str, Any]:
    import optuna

    dists: Dict[str, Any] = {}
    for hp in cs.get_hyperparameters():
        name = hp.name
        if isinstance(hp, UniformFloatHyperparameter):
            dists[name] = optuna.distributions.FloatDistribution(
                float(hp.lower), float(hp.upper), log=bool(getattr(hp, "log", False))
            )
        elif isinstance(hp, UniformIntegerHyperparameter):
            dists[name] = optuna.distributions.IntDistribution(int(hp.lower), int(hp.upper))
        elif isinstance(hp, CategoricalHyperparameter):
            dists[name] = optuna.distributions.CategoricalDistribution(list(hp.choices))
    return dists


def _cfg_for_optuna(cs: ConfigurationSpace, raw: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for hp in cs.get_hyperparameters():
        name = hp.name
        if name in raw and raw[name] is not None:
            val = raw[name]
        else:
            val = hp.default_value
        if isinstance(hp, UniformFloatHyperparameter):
            val = float(min(float(hp.upper), max(float(hp.lower), float(val))))
        elif isinstance(hp, UniformIntegerHyperparameter):
            val = int(min(int(hp.upper), max(int(hp.lower), int(val))))
        params[name] = val
    return params


def _optuna_trial_params_and_dists(cs: ConfigurationSpace, raw: Dict[str, Any]) -> tuple:
    """Active params only (CASH / nested TPE). Filling inactive defaults poisons TPE."""
    dists_all = _optuna_distributions(cs)
    parent = _cash_parent_name(cs)
    raw = dict(raw or {})
    if not parent:
        params = _cfg_for_optuna(cs, raw)
        dists = {k: dists_all[k] for k in params if k in dists_all}
        return params, dists
    family = raw.get(parent)
    params: Dict[str, Any] = {}
    if family is not None:
        params[parent] = family
    prefix = f"{family}:" if family is not None else ""
    for key, val in raw.items():
        if key == parent:
            continue
        if prefix and not str(key).startswith(prefix):
            continue
        if key in dists_all:
            params[key] = val
    dists = {k: dists_all[k] for k in params if k in dists_all}
    return params, dists


def _replay_optuna_observations(study, cs: ConfigurationSpace, observations: List[Dict[str, Any]]) -> int:
    """Seed an Optuna study with prior OpenBox-style observations (no re-eval)."""
    if study is None or not observations:
        return 0
    from optuna.trial import TrialState, create_trial

    n_ok = 0
    for row in observations:
        params, dists = _optuna_trial_params_and_dists(cs, row.get("config") or {})
        if not params or not dists:
            continue
        objs = row.get("objectives") or [1e10]
        try:
            val = float(objs[0])
        except (TypeError, ValueError):
            val = 1e10
        if not np.isfinite(val):
            val = 1e10
        try:
            study.add_trial(
                create_trial(params=params, distributions=dists, value=val, state=TrialState.COMPLETE)
            )
            n_ok += 1
        except Exception:
            continue
    return n_ok


def _replay_smac_observations(advisor, cs: ConfigurationSpace, observations: List[Dict[str, Any]]) -> int:
    if advisor is None or not observations:
        return 0
    n_ok = 0
    for row in observations:
        params = dict(row.get("config") or {})
        objs = row.get("objectives") or [1e10]
        try:
            val = float(objs[0])
        except (TypeError, ValueError):
            val = 1e10
        if not np.isfinite(val):
            val = 1e10
        pending_cfg = None
        try:
            pending_cfg = Configuration(cs, values=params)
        except Exception:
            try:
                pending_cfg = Configuration(cs, values=_cfg_from_dict(cs, params))
            except Exception:
                pending_cfg = None
        _smac_tell(advisor, params, cs, val, pending=pending_cfg)
        n_ok += 1
    return n_ok


def _optuna_running_trials(study) -> list:
    if study is None:
        return []
    try:
        from optuna.trial import TrialState

        return [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.RUNNING]
    except Exception:
        return []


def _traj_from_observations(observations: List[Dict[str, Any]]) -> tuple:
    best = float("inf")
    rows = []
    for i, o in enumerate(observations):
        try:
            obj = float((o.get("objectives") or [1e10])[0])
        except (TypeError, ValueError):
            obj = 1e10
        feas = int(o.get("trial_state", 0) or 0) == 0 and obj < 1e9
        if feas and obj < best:
            best = obj
        extra = o.get("extra_info") or {}
        rows.append(
            {
                "trial": i + 1,
                "source": extra.get("source") or "resume",
                "objective": obj,
                "best_so_far": best if best < float("inf") else obj,
                "feasible": int(feas),
            }
        )
    return rows, best


def _latent_bound_from_cs(cs: ConfigurationSpace) -> float:
    bound = 4.0
    for hp in cs.get_hyperparameters():
        if str(hp.name).startswith("z_"):
            bound = max(bound, float(abs(hp.lower)), float(abs(hp.upper)))
    return float(bound)


def _init_z_params(cs: ConfigurationSpace, init_z: List[float]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    zs = [float(v) for v in init_z]
    for hp in cs.get_hyperparameters():
        name = str(hp.name)
        if name.startswith("z_"):
            try:
                i = int(name.split("_", 1)[1])
            except ValueError:
                i = 0
            v = zs[i] if i < len(zs) else 0.0
            lo, hi = float(getattr(hp, "lower", -3)), float(getattr(hp, "upper", 3))
            params[name] = float(min(hi, max(lo, v)))
        else:
            params[name] = hp.default_value
    return params


def _build_domain_hebo(cs: ConfigurationSpace, seed: int, n_startup: int, model_name: str = "gp"):
    import pandas as pd  # noqa: F401
    from hebo.design_space.design_space import DesignSpace
    from hebo.optimizers.hebo import HEBO

    from mindware.components.optimizers.hebo_space import hp_to_hebo_spec

    recs = []
    hp_by_name: Dict[str, Any] = {}
    for hp in cs.get_hyperparameters():
        spec = hp_to_hebo_spec(hp)
        if spec is None:
            continue
        recs.append(spec)
        hp_by_name[hp.name] = hp
    if not recs:
        raise ValueError("HEBO space is empty")
    space = DesignSpace().parse(recs)
    hebo = HEBO(
        space,
        model_name=str(model_name or "gp"),
        rand_sample=max(2, int(n_startup)),
        scramble_seed=int(seed),
    )
    return hebo, hp_by_name


def _build_domain_smac(cs: ConfigurationSpace, seed: int, n_startup: int):
    from openbox.core.generic_advisor import Advisor

    return Advisor(
        config_space=cs,
        num_objectives=1,
        num_constraints=0,
        initial_trials=max(1, int(n_startup)),
        surrogate_type="prf",
        acq_type="ei",
        random_state=int(seed),
        task_id="domain_smac",
    )


def _persist_domain_checkpoint(
    output_dir: Optional[Path],
    adapter: DomainAdapter,
    observations: List[Dict[str, Any]],
    *,
    opt: str,
    seed: int,
    n_done: int,
    study=None,
    smac_advisor=None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    if output_dir is None:
        return None
    meta = {
        "domain": adapter.domain_id,
        "optimizer": opt,
        "seed": seed,
        "resumed_from": n_done,
        "n_observations": len(observations),
        "has_optuna_pkl": study is not None,
        "has_smac_pkl": smac_advisor is not None,
    }
    if extra_meta:
        meta.update(extra_meta)
    return persist_search_checkpoint(
        Path(output_dir),
        task_id=adapter.task_id(),
        observations=observations,
        meta=meta,
        study=study,
        smac_advisor=smac_advisor,
    )


def _smac_tell(advisor, cfg: Dict[str, Any], cs: ConfigurationSpace, obj: float, pending=None) -> None:
    from openbox.utils.constants import SUCCESS
    from openbox.utils.history import Observation as OBObservation

    config_obj = pending
    if config_obj is None:
        try:
            config_obj = Configuration(cs, values=_cfg_from_dict(cs, cfg))
        except Exception:
            return
    try:
        advisor.update_observation(OBObservation(config=config_obj, objectives=[float(obj)], trial_state=SUCCESS))
    except Exception:
        pass


def _hebo_frame(hebo, cfg: Dict[str, Any]):
    import pandas as pd

    row = {name: cfg[name] for name in hebo.space.para_names if name in cfg}
    return pd.DataFrame([row], columns=hebo.space.para_names)


def _hebo_suggest(hebo, cs: ConfigurationSpace) -> tuple:
    import contextlib
    import io

    import torch

    buf = io.StringIO()
    with torch.device("cpu"), contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rec = hebo.suggest(n_suggestions=1)
    row = rec.iloc[0]
    cfg: Dict[str, Any] = {}
    for hp in cs.get_hyperparameters():
        if hp.name in rec.columns:
            cfg[hp.name] = float(row[hp.name])
        else:
            cfg[hp.name] = hp.default_value
    return cfg, rec


def _hebo_observe(hebo, rec, obj: float) -> None:
    if rec is None or not np.isfinite(float(obj)):
        return
    try:
        hebo.observe(rec, np.asarray([[float(obj)]], dtype=float))
    except Exception:
        pass


def _replay_hebo_observations(hebo, cs: ConfigurationSpace, observations: List[Dict[str, Any]]) -> int:
    n_ok = 0
    for row in observations:
        params = _cfg_for_optuna(cs, row.get("config") or {})
        rec = _hebo_frame(hebo, params)
        try:
            val = float((row.get("objectives") or [1e10])[0])
        except (TypeError, ValueError):
            val = 1e10
        if not np.isfinite(val):
            val = 1e10
        _hebo_observe(hebo, rec, val)
        n_ok += 1
    return n_ok


def _sample_semantic(cs: ConfigurationSpace, adapter: DomainAdapter, rng: np.random.RandomState) -> Dict[str, Any]:
    """LLM-arm fallback: sample named semantic keys + N(0, 0.8) latent."""
    cfg: Dict[str, Any] = {}
    semantic = set(adapter.semantic_keys())
    for hp in cs.get_hyperparameters():
        if hp.name in semantic and isinstance(hp, CategoricalHyperparameter):
            cfg[hp.name] = hp.choices[int(rng.randint(0, len(hp.choices)))]
        elif hp.name.startswith("z_"):
            lo, hi = float(getattr(hp, "lower", -3)), float(getattr(hp, "upper", 3))
            cfg[hp.name] = float(np.clip(rng.normal(0.0, 0.8), lo, hi))
        elif isinstance(hp, UniformFloatHyperparameter):
            cfg[hp.name] = float(rng.uniform(float(hp.lower), float(hp.upper)))
        elif isinstance(hp, UniformIntegerHyperparameter):
            cfg[hp.name] = int(rng.randint(int(hp.lower), int(hp.upper) + 1))
        elif isinstance(hp, CategoricalHyperparameter):
            cfg[hp.name] = hp.choices[int(rng.randint(0, len(hp.choices)))]
        else:
            cfg[hp.name] = hp.default_value
    for name, choices in (adapter.semantic_choices() or {}).items():
        if name not in cfg and choices:
            cfg[name] = choices[int(rng.randint(0, len(choices)))]
    filled = _cfg_from_dict(cs, cfg)
    for k, v in cfg.items():
        filled.setdefault(k, v)
    return filled


def _save_tpe_run_checkpoint(
    output_dir: Path,
    adapter: DomainAdapter,
    observations: List[Dict[str, Any]],
    *,
    opt: str,
    seed: int,
    e2e_updates: int,
    e2e_freq: Optional[int],
    n_done: int,
) -> bool:
    vae_ok = (
        save_peptide_vae_checkpoint(
            getattr(adapter, "vae", None),
            peptide_vae_checkpoint_path(output_dir),
            e2e_updates=e2e_updates,
            n_observations=len(observations),
        )
        is not None
    )
    write_rotating_openbox_history(
        output_dir / "history.json",
        task_id=adapter.task_id(),
        observations=observations,
        meta={
            "domain": adapter.domain_id,
            "optimizer": opt,
            "seed": seed,
            "e2e_updates": e2e_updates,
            "e2e_freq": e2e_freq,
            "has_vae_checkpoint": bool(vae_ok),
            "resumed_from": n_done,
            "n_observations": len(observations),
        },
    )
    return bool(vae_ok)


def run_domain_search(
    adapter: DomainAdapter,
    *,
    n_trials: int = 16,
    seed: int = 1,
    optimizer: str = "cmaes",
    llm_prob: float = 0.7,
    n_warmup: int = 4,
    hybrid_selector: str = "bernoulli",
    bo_arm: Optional[str] = None,
    output_dir: Optional[Path] = None,
    llm_client=None,
    prompt_variant: str = "P5",
    init_z: Optional[List[float]] = None,
    n_startup_trials: Optional[int] = None,
    use_separable_cma: Optional[bool] = None,
    warm_start: Optional[Dict[str, Any]] = None,
    resume_observations: Optional[List[Dict[str, Any]]] = None,
    e2e_freq: Optional[int] = None,
    e2e_update_epochs: Optional[int] = None,
) -> Dict[str, Any]:
    cs = adapter.search_space()
    evaluator = DomainEvaluator(adapter)
    rng = np.random.RandomState(int(seed))
    opt = str(optimizer or "cmaes").lower()
    if bo_arm is None:
        if opt in {"cmaes", "ptllm_cmaes", "lolbo"}:
            bo_arm = "cmaes"
        elif opt in {"hebo", "ptllm_hebo"}:
            bo_arm = "hebo"
        elif opt in {"tr_tpe", "trust_region_tpe", "tpe_tr", "ptllm_tr_tpe"}:
            bo_arm = "tr_tpe"
        elif opt in {"smac", "ptllm_smac"}:
            bo_arm = "smac"
        else:
            bo_arm = "tpe"
    hebo = None
    pending_hebo_rec = None
    smac_advisor = None
    pending_smac_cfg = None
    tr_tpe_sampler = None
    force_init_cfg: Optional[Dict[str, Any]] = None
    n_start = (
        int(n_startup_trials)
        if n_startup_trials is not None
        else max(1, min(8, max(1, int(n_trials) // 10)))
    )
    n_start = max(1, min(n_start, max(1, int(n_trials) - 1)))
    if opt in {"random", "rand"}:
        selector = None
        study = None
    elif opt in {"smac", "ptllm_smac"}:
        selector = None
        study = None
        smac_advisor = _build_domain_smac(cs, seed, n_start)
        if opt.startswith("ptllm"):
            selector = selector_from_params(
                {
                    "hybrid_selector": hybrid_selector,
                    "llm_prob": float(llm_prob),
                    "bo_arm": bo_arm,
                },
                rng=rng,
            )
    elif opt in {"hebo", "ptllm_hebo"}:
        selector = None
        study = None
        hebo, _hebo_hps = _build_domain_hebo(cs, seed, n_start)
        if init_z is not None and not resume_observations:
            force_init_cfg = _init_z_params(cs, init_z)
        if opt.startswith("ptllm"):
            selector = selector_from_params(
                {
                    "hybrid_selector": hybrid_selector,
                    "llm_prob": float(llm_prob),
                    "bo_arm": bo_arm,
                },
                rng=rng,
            )
    else:
        import optuna

        z_dim = sum(1 for hp in cs.get_hyperparameters() if str(hp.name).startswith("z_"))
        sep = bool(use_separable_cma) if use_separable_cma is not None else z_dim >= 32
        if opt in {"cmaes", "ptllm_cmaes", "lolbo"} or bo_arm == "cmaes":
            cma_kwargs = {"seed": int(seed), "n_startup_trials": n_start}
            if sep:
                cma_kwargs["use_separable_cma"] = True
            sampler = optuna.samplers.CmaEsSampler(**cma_kwargs)
        elif opt in {"tr_tpe", "trust_region_tpe", "tpe_tr", "ptllm_tr_tpe"} or bo_arm == "tr_tpe":
            from mindware.components.optimizers.trust_region_tpe import build_trust_region_tpe_sampler

            tr_tpe_sampler = build_trust_region_tpe_sampler(
                cs,
                seed=int(seed),
                n_startup_trials=n_start,
                latent_bound=_latent_bound_from_cs(cs),
            )
            sampler = tr_tpe_sampler
        else:
            sampler = optuna.samplers.TPESampler(seed=int(seed), n_startup_trials=n_start)
        try:
            study = optuna.create_study(direction="minimize", sampler=sampler)
        except Exception:
            study = optuna.create_study(
                direction="minimize", sampler=optuna.samplers.TPESampler(seed=int(seed))
            )
            if bo_arm == "cmaes":
                bo_arm = "tpe"
        if init_z is not None and study is not None:
            params = _init_z_params(cs, init_z)
            if not resume_observations:
                try:
                    study.enqueue_trial(params)
                except Exception:
                    pass
        selector = None
        if opt.startswith("ptllm"):
            selector = selector_from_params(
                {
                    "hybrid_selector": hybrid_selector,
                    "llm_prob": float(llm_prob),
                    "bo_arm": bo_arm,
                },
                rng=rng,
            )

    observations: List[Dict[str, Any]] = list(resume_observations or [])
    traj_rows: List[Dict[str, Any]] = []
    messages_log = []
    best = float("inf")
    t0 = time.time()
    top_plans: List[Any] = []
    progress_fails = 0
    e2e_updates = 0
    last_tpe_ckpt_n = 0
    bolt_native = str(getattr(adapter, "oracle_kind", "") or "").lower() == "bolt_native"
    if e2e_freq is None:
        if opt == "lolbo":
            e2e_freq = 10
        elif bolt_native and opt in {"tr_tpe", "trust_region_tpe", "tpe_tr", "ptllm_tr_tpe"}:
            e2e_freq = int(BOLT_SEED0_HPARAMS.get("e2e_freq") or 10)
        else:
            e2e_freq = 10**9
    else:
        e2e_freq = int(e2e_freq)
    e2e_epochs = int(e2e_update_epochs or BOLT_SEED0_HPARAMS.get("num_update_epochs") or 2)
    e2e_recent = int(BOLT_SEED0_HPARAMS.get("bsz") or 50)
    e2e_top_k = min(1000, max(e2e_recent, int(BOLT_SEED0_HPARAMS.get("k") or 1000)))
    use_peptide_e2e = bolt_native and e2e_freq < 10**9
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    reject_dup_orders = bool(getattr(adapter, "query_layers", False))
    reject_dup_sequences = bolt_native and (
        bool(getattr(adapter, "bolt_strict_decode", False))
        or bool(getattr(adapter, "bolt_reject_dup_sequences", False))
    )
    seen_orders: set = set()
    seen_sequences: set = set()
    n_done = 0
    vae_resume_meta: Dict[str, Any] = {}
    if use_peptide_e2e and output_dir is not None:
        hist_meta_path = Path(output_dir) / "history.json"
        if hist_meta_path.is_file():
            try:
                hist_payload = json.loads(hist_meta_path.read_text(encoding="utf-8"))
                e2e_updates = int((hist_payload.get("meta_info") or {}).get("e2e_updates") or 0)
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                pass
        vae_resume_meta = load_peptide_vae_checkpoint(
            getattr(adapter, "vae", None),
            peptide_vae_checkpoint_path(output_dir),
        )
        if vae_resume_meta.get("vae_restored"):
            e2e_updates = int(vae_resume_meta.get("e2e_updates") or e2e_updates)
            print(
                json.dumps(
                    {
                        "task_id": adapter.task_id(),
                        "vae_resume": True,
                        "e2e_updates": int(e2e_updates),
                        "n_observations": int(vae_resume_meta.get("n_observations") or 0),
                    }
                ),
                flush=True,
            )
        elif int(vae_resume_meta.get("e2e_updates") or e2e_updates) > 0:
            print(
                json.dumps(
                    {
                        "task_id": adapter.task_id(),
                        "vae_resume_warning": True,
                        "e2e_updates": int(e2e_updates),
                        "message": "history has E2E updates but no finetuned VAE checkpoint; using base VAE",
                    }
                ),
                flush=True,
            )
    if observations:
        traj_rows, best = _traj_from_observations(observations)
        n_done = len(observations)
        last_elapsed = float(observations[-1].get("elapsed_time") or 0.0)
        t0 = time.time() - last_elapsed
        for o in observations:
            extra = o.get("extra_info") or {}
            art = extra.get("artifact")
            if reject_dup_sequences and art is not None:
                seen_sequences.add(str(art))
            if reject_dup_orders and art is not None:
                seen_orders.add(tuple(art) if not isinstance(art, tuple) else art)
            try:
                obj = float((o.get("objectives") or [1e10])[0])
            except (TypeError, ValueError):
                continue
            if art and obj <= best:
                top_plans = [art] + [p for p in top_plans if p != art]
                top_plans = top_plans[:8]
        if study is not None:
            loaded_study = None
            if output_dir is not None:
                loaded_study = load_optuna_study(Path(output_dir), n_complete=n_done)
            if loaded_study is not None:
                study = loaded_study
            else:
                _replay_optuna_observations(study, cs, observations)
            if tr_tpe_sampler is not None:
                tr_tpe_sampler.reseed_from_observations(observations)
        if smac_advisor is not None:
            loaded_smac = None
            if output_dir is not None:
                loaded_smac = load_smac_advisor(Path(output_dir), n_complete=n_done)
            if loaded_smac is not None:
                smac_advisor = loaded_smac
            else:
                _replay_smac_observations(smac_advisor, cs, observations)
        if hebo is not None:
            _replay_hebo_observations(hebo, cs, observations)
        if n_done >= int(n_trials):
            out: Dict[str, Any] = {
                "task_id": adapter.task_id(),
                "domain": adapter.domain_id,
                "optimizer": opt,
                "n_trials": n_trials,
                "best_objective": best if best < float("inf") else None,
                "observations": observations,
                "traj": traj_rows,
                "resumed_from": n_done,
            }
            if output_dir is not None:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                hist = write_openbox_history(
                    output_dir / "history.json",
                    task_id=adapter.task_id(),
                    observations=observations,
                    meta={"domain": adapter.domain_id, "optimizer": opt, "seed": seed, "resumed": True},
                )
                out["history_path"] = str(hist)
            return out

    pending_cfg: Optional[Dict[str, Any]] = None
    if output_dir is not None:
        pending_bundle = load_pending(Path(output_dir))
        if pending_bundle:
            pending_cfg = dict(pending_bundle.get("config") or {})
            if not pending_cfg:
                pending_cfg = None

    if (not observations) and warm_start and warm_start.get("order") is not None:
        ws_order = list(warm_start["order"])
        ws_res = adapter.evaluate(ws_order)
        ws_obj = float(ws_res.objective)
        ws_feas = bool(ws_res.feasible)
        if ws_feas and ws_obj < best:
            best = ws_obj
            if ws_res.artifact:
                top_plans = [ws_res.artifact]
        if reject_dup_orders and ws_res.artifact:
            seen_orders.add(tuple(ws_res.artifact))
        ws_cfg = z_to_config(init_z) if init_z else {}
        observations.append(
            observation(
                ws_cfg,
                ws_obj if ws_feas else 1e10,
                trial_state=0 if ws_feas else 1,
                elapsed_time=time.time() - t0,
                extra={
                    "source": "warm_start",
                    "artifact": ws_res.artifact,
                    "info": {**(ws_res.info or {}), "warm_start": True},
                },
            )
        )
        traj_rows.append(
            {
                "trial": 0,
                "source": "warm_start",
                "objective": ws_obj,
                "best_so_far": best,
                "feasible": int(ws_feas),
            }
        )

    max_dup_resamples = 128 if reject_dup_sequences else 1
    while len(observations) < int(n_trials):
        if reject_dup_sequences:
            adapter._bolt_seen_sequences = seen_sequences  # type: ignore[attr-defined]
        i = len(observations)
        trial = None
        source = "random"
        if selector is not None and i >= n_warmup:
            m = i - n_warmup
            M = max(1, n_trials - n_warmup)
            source = selector.choose(m, M, higher_is_better=False)
        elif selector is not None:
            source = "llm_warm_start"
        elif hebo is not None:
            source = "hebo"
        elif smac_advisor is not None:
            source = "smac"
        elif study is not None:
            source = bo_arm
        elif opt in {"random", "rand"}:
            source = "random"

        cfg: Optional[Dict[str, Any]] = None
        seq_preview: Optional[str] = None
        duplicate_sequence = False
        for _dup_attempt in range(max_dup_resamples):
            if pending_cfg is not None:
                cfg = dict(pending_cfg)
                pending_cfg = None
                source = "resume_pending"
                running = _optuna_running_trials(study)
                if running:
                    trial = running[-1]
                if smac_advisor is not None:
                    try:
                        pending_smac_cfg = Configuration(cs, values=cfg)
                    except Exception:
                        pending_smac_cfg = None
            elif source in {"llm", "llm_warm_start"}:
                cfg = _sample_semantic(cs, adapter, rng)
                if llm_client is not None:
                    msgs = build_domain_messages(adapter, traj_rows, variant=prompt_variant)
                    try:
                        text = llm_client.chat(msgs) if hasattr(llm_client, "chat") else ""
                        parsed = parse_semantic_action(text, cs)
                        if parsed:
                            cfg = _cfg_from_dict(cs, {**cfg, **parsed})
                        messages_log.append({"trial": i + 1, "messages": msgs, "completion": text, "source": source})
                    except Exception:
                        messages_log.append({"trial": i + 1, "source": source, "error": "llm_failed"})
            elif hebo is not None:
                if force_init_cfg is not None:
                    cfg = dict(force_init_cfg)
                    pending_hebo_rec = _hebo_frame(hebo, cfg)
                    force_init_cfg = None
                    source = "hebo_init"
                else:
                    cfg, pending_hebo_rec = _hebo_suggest(hebo, cs)
                    source = "hebo"
            elif smac_advisor is not None:
                pending_smac_cfg = smac_advisor.get_suggestion()
                cfg = dict(pending_smac_cfg.get_dictionary()) if hasattr(pending_smac_cfg, "get_dictionary") else dict(pending_smac_cfg)
                source = "smac"
            elif study is not None:
                trial = study.ask()
                cfg = _suggest_optuna(trial, cs)
                source = bo_arm if source in {bo_arm, "tpe", "tr_tpe", "cmaes", "hebo"} else source
            else:
                cfg = dict(cs.sample_configuration().get_dictionary())
                source = "random"

            if reject_dup_sequences and cfg is not None:
                try:
                    seq_preview = str(adapter.decode(cfg))
                except Exception:
                    seq_preview = ""
                if seq_preview and seq_preview in seen_sequences:
                    duplicate_sequence = True
                    if study is not None and trial is not None:
                        try:
                            import optuna

                            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                        except Exception:
                            pass
                    trial = None
                    continue
            break
        if cfg is None:
            break
        if output_dir is not None and source != "resume_pending":
            write_pending(Path(output_dir), cfg, extra={"source": source, "trial": i + 1})
            if study is not None:
                save_optuna_study(study, Path(output_dir))
        if reject_dup_sequences and (not seq_preview or seq_preview in seen_sequences):
            continue

        if reject_dup_sequences and seq_preview:
            eval_res = adapter.evaluate(seq_preview, adapter.task)
            last = eval_res
            feasible = bool(eval_res.feasible)
            obj = _search_objective_from_eval(last, feasible)
        else:
            obj = float(evaluator(cfg))
            last = evaluator.last
            feasible = bool(last.feasible) if last else True
            if last is not None and not feasible:
                obj = _search_objective_from_eval(last, feasible)
        duplicate_order = False
        art = getattr(last, "artifact", None) if last else seq_preview
        if reject_dup_sequences and art is not None:
            seq_key = str(art)
            if seq_key in seen_sequences:
                duplicate_sequence = True
                if study is not None and trial is not None:
                    try:
                        import optuna

                        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                    except Exception:
                        pass
                continue
            seen_sequences.add(seq_key)
        if reject_dup_orders and feasible and art is not None:
            key = tuple(art)
            if key in seen_orders:
                duplicate_order = True
                obj = 1e10
                feasible = False
            else:
                seen_orders.add(key)
        tell_obj = obj if (feasible or obj < 1e9) else 1e10
        if study is not None and trial is not None:
            try:
                study.tell(trial, tell_obj)
            except Exception:
                try:
                    study.tell(getattr(trial, "number", trial), tell_obj)
                except Exception:
                    pass
        elif study is not None and source == "resume_pending":
            _replay_optuna_observations(study, cs, [observation(cfg, tell_obj)])
        if hebo is not None:
            rec = pending_hebo_rec
            pending_hebo_rec = None
            if rec is None:
                rec = _hebo_frame(hebo, cfg)
            _hebo_observe(hebo, rec, tell_obj)
        if smac_advisor is not None:
            _smac_tell(smac_advisor, cfg, cs, tell_obj, pending=pending_smac_cfg)
            pending_smac_cfg = None
        if selector is not None and source in selector.arms:
            selector.update(source, -obj, higher_is_better=True, beat_incumbent=obj < best)
        if obj < best and feasible:
            best = obj
            progress_fails = 0
            art = getattr(last, "artifact", None)
            if art:
                top_plans = [art] + [p for p in top_plans if p != art]
                top_plans = top_plans[:8]
        else:
            progress_fails += 1
        tr_meta: Dict[str, Any] = {}
        if tr_tpe_sampler is not None and tr_tpe_sampler.tr_state is not None:
            tr_meta = {
                "tr_length": float(tr_tpe_sampler.tr_state.length),
                "tr_restart": bool(tr_tpe_sampler.tr_state.restart_triggered),
            }
        observations.append(
            observation(
                cfg,
                obj if feasible else 1e10,
                trial_state=0 if feasible else 1,
                elapsed_time=time.time() - t0,
                extra={
                    "source": source,
                    "artifact": getattr(last, "artifact", None),
                    "info": {
                        **(getattr(last, "info", {}) or {}),
                        **({"duplicate_order": True} if duplicate_order else {}),
                        **({"duplicate_sequence": True} if duplicate_sequence else {}),
                        **tr_meta,
                    },
                },
            )
        )
        _persist_domain_checkpoint(
            output_dir,
            adapter,
            observations,
            opt=opt,
            seed=seed,
            n_done=n_done,
            study=study,
            smac_advisor=smac_advisor,
            extra_meta={
                "e2e_updates": e2e_updates,
                "e2e_freq": e2e_freq if use_peptide_e2e else None,
            },
        )
        last_tpe_ckpt_n = len(observations)
        n_obs_now = len(observations)
        if use_peptide_e2e and n_obs_now > 0 and (n_obs_now % e2e_freq) == 0:
            finetune = getattr(adapter, "e2e_finetune", None)
            e2e_seqs = sequences_for_peptide_e2e(
                observations,
                recent=e2e_recent,
                top_k=e2e_top_k,
            )
            if not e2e_seqs and top_plans:
                e2e_seqs = [str(s) for s in top_plans if s]
            did_e2e = False
            if callable(finetune) and e2e_seqs:
                try:
                    steps = int(finetune(e2e_seqs, n_steps=e2e_epochs) or 0)
                    if steps <= 0:
                        print(
                            json.dumps(
                                {
                                    "task_id": adapter.task_id(),
                                    "e2e_finetune_skipped": True,
                                    "reason": "zero_steps",
                                    "n_seqs": len(e2e_seqs),
                                }
                            ),
                            flush=True,
                        )
                    else:
                        vae = getattr(adapter, "vae", None)
                        if vae is not None:
                            reencode_observation_latents(
                                vae,
                                observations,
                                latent_dim=int(getattr(adapter, "latent_dim", 256)),
                            )
                        if study is not None:
                            sampler = study.sampler
                            import optuna

                            study = optuna.create_study(direction="minimize", sampler=sampler)
                            _replay_optuna_observations(study, cs, observations)
                        if tr_tpe_sampler is not None:
                            tr_tpe_sampler.reseed_from_observations(observations)
                        e2e_updates += 1
                        did_e2e = True
                        if output_dir is not None:
                            _save_tpe_run_checkpoint(
                                Path(output_dir),
                                adapter,
                                observations,
                                opt=opt,
                                seed=seed,
                                e2e_updates=e2e_updates,
                                e2e_freq=e2e_freq if use_peptide_e2e else None,
                                n_done=n_done,
                            )
                            last_tpe_ckpt_n = len(observations)
                        print(
                            json.dumps(
                                {
                                    "task_id": adapter.task_id(),
                                    "e2e_reencode": True,
                                    "n_obs": len(observations),
                                    "e2e_updates": int(e2e_updates),
                                }
                            ),
                            flush=True,
                        )
                except Exception as exc:
                    print(
                        json.dumps(
                            {
                                "task_id": adapter.task_id(),
                                "e2e_finetune_error": str(exc),
                            }
                        ),
                        flush=True,
                    )
            if did_e2e:
                progress_fails = 0
        if (
            use_peptide_e2e
            and output_dir is not None
            and len(observations) > 0
            and len(observations) % int(TPE_RUN_CHECKPOINT_INTERVAL) == 0
            and len(observations) != last_tpe_ckpt_n
        ):
            _save_tpe_run_checkpoint(
                Path(output_dir),
                adapter,
                observations,
                opt=opt,
                seed=seed,
                e2e_updates=e2e_updates,
                e2e_freq=e2e_freq,
                n_done=n_done,
            )
            last_tpe_ckpt_n = len(observations)
        traj_rows.append(
            {
                "trial": i + 1,
                "source": source,
                "objective": obj,
                "best_so_far": best,
                "feasible": int(feasible),
            }
        )
        if use_peptide_e2e and len(observations) % 100 == 0:
            print(
                json.dumps(
                    {
                        "task_id": adapter.task_id(),
                        "optimizer": opt,
                        "trial": len(observations),
                        "n_obs": len(observations),
                        "best_objective": best if best < float("inf") else None,
                        "e2e_updates": e2e_updates,
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                ),
                flush=True,
            )

    out: Dict[str, Any] = {
        "task_id": adapter.task_id(),
        "domain": adapter.domain_id,
        "optimizer": opt,
        "n_trials": n_trials,
        "best_objective": best,
        "resumed_from": n_done,
        "e2e_updates": e2e_updates,
        "observations": observations,
        "traj": traj_rows,
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        vae_ckpt_saved = False
        if use_peptide_e2e:
            vae_ckpt_saved = _save_tpe_run_checkpoint(
                output_dir,
                adapter,
                observations,
                opt=opt,
                seed=seed,
                e2e_updates=e2e_updates,
                e2e_freq=e2e_freq,
                n_done=n_done,
            )
        else:
            hist = _persist_domain_checkpoint(
                output_dir,
                adapter,
                observations,
                opt=opt,
                seed=seed,
                n_done=n_done,
                study=study,
                smac_advisor=smac_advisor,
                extra_meta={
                    "e2e_updates": e2e_updates,
                    "e2e_freq": e2e_freq if use_peptide_e2e else None,
                },
            )
        if use_peptide_e2e:
            hist = output_dir / "history.json"
        traj_path = output_dir / f"{adapter.task_id()}_opt_{opt}_traj.csv"
        with traj_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["trial", "source", "objective", "best_so_far", "feasible"])
            w.writeheader()
            w.writerows(traj_rows)
        if messages_log:
            conv = output_dir / "llm_conversations"
            conv.mkdir(exist_ok=True)
            for row in messages_log:
                (conv / f"trial_{int(row.get('trial', 0)):04d}.json").write_text(
                    json.dumps(row, indent=2, default=str) + "\n", encoding="utf-8"
                )
        out["history_path"] = str(hist)
        out["traj_path"] = str(traj_path)
    return out
