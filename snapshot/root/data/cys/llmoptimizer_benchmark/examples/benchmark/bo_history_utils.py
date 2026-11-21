"""Load nested BO history JSONs for RL warm-start / prefix prompts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_BO_HISTORY_ROOT = Path(
    "/root/data/cys/blockdata_archive/llmoptimizer_benchmark/history_data/history_data"
)


def bo_history_path(
    task_type: str,
    dataset: str,
    algorithm: str,
    *,
    metric: Optional[str] = None,
    root: Path = DEFAULT_BO_HISTORY_ROOT,
) -> Path:
    metric = metric or ("bal_acc" if task_type == "cls" else "mse")
    return Path(root) / task_type / metric / dataset / f"{algorithm}.json"


def load_bo_observations(
    path: Path,
    *,
    algorithm: Optional[str] = None,
    max_n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return successful observations (OpenBox trial_state==0), truncated to max_n."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    obs = list(obj.get("observations") or [])
    out: List[Dict[str, Any]] = []
    for row in obs:
        if int(row.get("trial_state", -1)) != 0:
            continue
        cfg = dict(row.get("config") or {})
        if algorithm and cfg.get("algorithm") not in (None, algorithm):
            continue
        objectives = row.get("objectives") or []
        if not objectives or not np.isfinite(float(objectives[0])):
            continue
        out.append(row)
        if max_n is not None and len(out) >= max_n:
            break
    return out


def observations_to_perf_pairs(
    observations: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """Convert BO obs -> (flat config dicts, higher-is-better perfs).

    OpenBox stores minimize-objectives; MindWare KD/P5 history uses higher-is-better,
    so perf = -objective.
    """
    configs: List[Dict[str, Any]] = []
    perfs: List[float] = []
    for row in observations:
        cfg = dict(row.get("config") or {})
        raw_obj = float(row["objectives"][0])
        configs.append(cfg)
        perfs.append(-raw_obj)
    return configs, perfs


def seed_optimizer_with_bo(
    optimizer,
    observations: Sequence[Dict[str, Any]],
    *,
    source: str = "bo_history",
) -> int:
    """Inject BO trials into PT4OptLLMOptimizer / KD history (no LLM dump).

    Returns number of seeded trials. Caller should set evaluation_num_limit to
    len(seed) + n_new so CASH continues for n_new LLM trials.
    """
    configs, perfs = observations_to_perf_pairs(observations)
    if not configs:
        return 0
    kd = optimizer.KD
    algo = optimizer.fixed_algorithm
    for cfg_dict, perf in zip(configs, perfs):
        flat = dict(cfg_dict)
        flat.setdefault("algorithm", algo)
        cfg = kd._dict_to_cfg(flat)
        kd.configs.append(cfg)
        kd.perfs.append(float(perf))
        if hasattr(kd, "configs_mab") and algo in kd.configs_mab:
            kd.configs_mab[algo].append(cfg)
            kd.perfs_mab[algo].append(float(perf))
        optimizer.trial_sources[len(kd.configs)] = source
        # seen_objectives stores raw minimize objective (= -perf)
        optimizer._mark_seen(cfg, objective=-float(perf))
    optimizer._sync_seen_signatures()
    optimizer._update_incumbent()
    return len(configs)


def load_light16_tasks(
    path: Path = Path(
        "/root/data/cys/blockdata_archive/llmoptimizer_benchmark/benchmark_data/rl_task_selection/light_16.json"
    ),
) -> List[Dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
