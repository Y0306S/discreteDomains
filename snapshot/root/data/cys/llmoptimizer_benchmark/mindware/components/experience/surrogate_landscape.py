"""Per-algorithm GP landscape similarity (Spearman on shared query configs).

One OpenBox GP is fit per (dataset, algorithm). Query configs are sampled once
per algorithm so predicted vectors are comparable. Cross-algorithm scores are 0.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mindware.components.experience.retrieve import spearman
from mindware.components.utils.constants import CLASSIFICATION, CLS_TASKS, REGRESSION

SEED = 42


def task_type_cs(label: str) -> int:
    lab = str(label or "").lower()
    if lab in ("classification", "cls", "binary", "multiclass"):
        return CLASSIFICATION
    if lab in ("regression", "rgs"):
        return REGRESSION
    return CLASSIFICATION


def default_cache_path(task_prefix: str, repo: Optional[Path] = None) -> Path:
    root = repo or Path(__file__).resolve().parents[3]
    return root / "history_data" / "meta_resource" / f"surrogate_landscape_{task_prefix}.pkl"


def get_algo_cs(algorithm: str, task_type: str):
    from mindware.components.experience.bank_gates import load_estimator_cs

    cs = load_estimator_cs(algorithm, task_type)
    if cs is not None:
        return cs
    from mindware.components.config_space.cs_builder import get_hpo_cs

    return get_hpo_cs(estimator_id=algorithm, task_type=task_type_cs(task_type))


def sample_query_configs(cs, n: int, seed: int = SEED):
    cs.seed(int(seed))
    n = max(3, int(n))
    try:
        cfgs = cs.sample_configuration(n)
    except Exception:
        cfgs = [cs.sample_configuration() for _ in range(n)]
    if not isinstance(cfgs, list):
        cfgs = [cfgs]
    return cfgs[:n]


def _cfg_get(cfg: Dict[str, Any], name: str, algorithm: str):
    if name in cfg and cfg[name] is not None:
        return cfg[name]
    pref = f"{algorithm}:{name}"
    if pref in cfg and cfg[pref] is not None:
        return cfg[pref]
    stem = name.split(":")[-1]
    for k, v in cfg.items():
        if str(k).split(":")[-1] == stem and v is not None:
            return v
    return None


def _configuration_from_trial(cs, cfg: Dict[str, Any], algorithm: str):
    from ConfigSpace import Configuration
    from ConfigSpace.hyperparameters import Constant, UnParametrizedHyperparameter

    values: Dict[str, Any] = {}
    for hp in cs.get_hyperparameters():
        name = hp.name
        raw = _cfg_get(cfg, name, algorithm)
        if raw is not None:
            values[name] = raw
        elif isinstance(hp, (Constant, UnParametrizedHyperparameter)):
            values[name] = getattr(hp, "value", getattr(hp, "default_value", None))
        else:
            values[name] = hp.default_value
    try:
        return Configuration(cs, values=values)
    except Exception:
        try:
            return Configuration(cs, values=values, allow_inactive_with_values=True)
        except Exception:
            return None


def trials_to_xy(trials: Sequence[Dict[str, Any]], cs, algorithm: str):
    from openbox.utils.config_space.util import convert_configurations_to_array

    configs = []
    ys = []
    for t in trials:
        cfg = t.get("config") or {}
        if not isinstance(cfg, dict):
            continue
        conf = _configuration_from_trial(cs, cfg, algorithm)
        if conf is None:
            continue
        try:
            ys.append(float(t.get("score")))
        except (TypeError, ValueError):
            continue
        configs.append(conf)
    if len(configs) < 3:
        return None, None
    try:
        X = convert_configurations_to_array(configs)
    except Exception:
        return None, None
    Y = np.asarray(ys, dtype=float).reshape(-1, 1)
    return X, Y


def configs_to_array(configs, cs):
    from openbox.utils.config_space.util import convert_configurations_to_array

    return convert_configurations_to_array(configs)


def fit_gp_predict(train_X, train_Y, test_X, cs, rng=None):
    """Fit OpenBox GP; fall back to sklearn GP if OpenBox fails."""
    rng = rng or np.random.RandomState(SEED)
    y = np.asarray(train_Y, dtype=float).reshape(-1, 1)
    try:
        from openbox.surrogate.base.build_gp import create_gp_model
        from openbox.utils.util_funcs import get_types

        types, bounds = get_types(cs)
        model = create_gp_model(
            model_type="gp", config_space=cs, types=types, bounds=bounds, rng=rng
        )
        model.train(np.asarray(train_X, dtype=float), y)
        pred, _ = model.predict(np.asarray(test_X, dtype=float))
        return np.asarray(pred, dtype=float).reshape(-1)
    except Exception:
        pass
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

        kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=0.05)
        gp = GaussianProcessRegressor(kernel=kernel, random_state=SEED, normalize_y=True)
        gp.fit(np.asarray(train_X, dtype=float), y.reshape(-1))
        pred = gp.predict(np.asarray(test_X, dtype=float))
        return np.asarray(pred, dtype=float).reshape(-1)
    except Exception:
        return None


def predict_landscape(
    trials: Sequence[Dict[str, Any]],
    cs,
    algorithm: str,
    query_configs,
    min_trials: int = 8,
) -> Optional[np.ndarray]:
    if len(trials) < int(min_trials):
        return None
    X, Y = trials_to_xy(trials, cs, algorithm)
    if X is None:
        return None
    try:
        test_X = configs_to_array(query_configs, cs)
    except Exception:
        return None
    return fit_gp_predict(X, Y, test_X, cs)


def spearman_sim(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    if a is None or b is None:
        return 0.0
    aa = [float(x) for x in np.asarray(a).reshape(-1)]
    bb = [float(x) for x in np.asarray(b).reshape(-1)]
    n = min(len(aa), len(bb))
    if n < 3:
        return 0.0
    return float(spearman(aa[:n], bb[:n]))


class LandscapeCache:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.payload: Dict[str, Any] = {
            "pred_cache": {},
            "similarity": {},
            "sampled_configurations": {},
            "hyperparams": {},
        }
        if self.path and self.path.is_file():
            try:
                loaded = pickle.loads(self.path.read_bytes())
                if isinstance(loaded, dict):
                    self.payload.update(loaded)
            except Exception:
                pass

    def pred(self, algorithm: str, dataset: str) -> Optional[np.ndarray]:
        block = (self.payload.get("pred_cache") or {}).get(algorithm) or {}
        vec = block.get(dataset)
        if vec is None:
            return None
        return np.asarray(vec, dtype=float).reshape(-1)

    def set_pred(self, algorithm: str, dataset: str, vec: Sequence[float]) -> None:
        pc = self.payload.setdefault("pred_cache", {})
        pc.setdefault(algorithm, {})[dataset] = np.asarray(vec, dtype=float).reshape(-1)

    def pair_sim(self, algorithm: str, d1: str, d2: str) -> Optional[float]:
        mat = (self.payload.get("similarity") or {}).get(algorithm) or {}
        if isinstance(mat, dict):
            row = mat.get(d1) or {}
            if d2 in row:
                return float(row[d2])
        return None

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(pickle.dumps(self.payload))
