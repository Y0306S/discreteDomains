import time
import os
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple

from ConfigSpace import Configuration
from mindware.components.optimizers.utils import MyConfiguration
from openbox.utils.limit import run_obj_func
from openbox.utils.history import Observation
from openbox.utils.util_funcs import parse_result
from openbox.utils.constants import SUCCESS, FAILED, TIMEOUT

def _evaluate_dict(self, cfg_dict: Dict[str, Any], config_space) -> float:
    """
    用 evaluator 评估一个配置字典（会调用 self._dict_to_cfg 做补全/类型矫正）。
    """
    t0_eval = time.time()
    if isinstance(cfg_dict, dict):
        cfg = _dict_to_cfg(cfg_dict, config_space)
    else:
        cfg = cfg_dict
    obs = self.objective_function(cfg)
    eval_time = time.time() - t0_eval

    # evaluator 返回的是要最小化的 loss？这里与 run 内保持一致：外部用 “-val” 当作精度。
    return obs, eval_time

def _dict_to_cfg(d: Dict[str, Any], config_space) -> MyConfiguration:
        """
        Converts a dictionary of hyperparameters into a ConfigSpace.Configuration object.

        This method handles filling in default values, casting types, and ensuring
        that the configuration is valid for the given search space, especially in
        complex conditional spaces (e.g., algorithm-specific parameters).
        """
        import math
        from ConfigSpace.hyperparameters import (
            UniformIntegerHyperparameter, UniformFloatHyperparameter,
            CategoricalHyperparameter, Constant, UnParametrizedHyperparameter
        )

        def _is_nan(x):
            return isinstance(x, float) and math.isnan(x)

        def _none_like(x):
            return (x is None) or (isinstance(x, str) and x.strip().lower() in ("none", "null"))

        hp_list = config_space.get_hyperparameters()
        hp_map = {hp.name: hp for hp in hp_list}

        # 1) Determine the active algorithm.
        algo_hp = hp_map.get('algorithm', None)
        algo = d.get('algorithm')
        if algo is None:
            # Infer algorithm from parameter prefixes (e.g., 'xgboost:n_estimators').
            prefixes = {k.split(':', 1)[0] for k in d.keys() if ':' in k}
            if prefixes:
                algo = list(prefixes)[0]
            elif algo_hp is not None:
                algo = algo_hp.default_value
            else:
                raise ValueError("No 'algorithm' provided and cannot infer from keys.")
        if algo_hp is not None and algo not in list(algo_hp.choices):
            raise ValueError(f"Invalid algorithm '{algo}'.")

        # 2) Pre-fill default values for the active algorithm's branch.
        values: Dict[str, Any] = {'algorithm': algo}
        for hp in hp_list:
            name = hp.name
            if name == 'algorithm':
                continue
            # Only consider parameters relevant to the selected algorithm.
            if name.startswith(algo + ':'):
                if isinstance(hp, (Constant, UnParametrizedHyperparameter)):
                    values[name] = getattr(hp, 'value', getattr(hp, 'default_value', None))
                else:
                    if hasattr(hp, 'default_value'):
                        values[name] = hp.default_value
                    elif hasattr(hp, 'value'):
                        values[name] = hp.value

        # 3) Override defaults with values from the input dictionary.
        # This includes type casting and skipping invalid or NaN values.
        for k_in, v_in in d.items():
            if k_in == 'algorithm':
                continue

            if _is_nan(v_in):
                continue

            if _none_like(v_in):
                v_in = None

            # Handle parameter names that might not have the algorithm prefix.
            if not k_in.startswith(algo + ':'):
                target = f"{algo}:{k_in}"
                if target in hp_map:
                    k = target
                else:
                    continue
            else:
                k = k_in

            hp = hp_map.get(k)
            # Skip if hyperparameter does not exist or is a constant.
            if hp is None or isinstance(hp, (Constant, UnParametrizedHyperparameter)):
                continue

            # Attempt to cast the value to the correct type.
            try:
                if isinstance(hp, UniformIntegerHyperparameter):
                    v = float(v_in)
                    q = getattr(hp, "q", None)
                    if q is not None:
                        q = float(q)
                        v = round((v - float(hp.lower)) / q) * q + float(hp.lower)
                    v = int(round(v))
                    if v < hp.lower:
                        v = int(hp.lower)
                    if v > hp.upper:
                        v = int(hp.upper)
                elif isinstance(hp, UniformFloatHyperparameter):
                    if v_in is None:
                        v = None
                    else:
                        v = float(v_in)
                        q = getattr(hp, "q", None)
                        if q is not None:
                            q = float(q)
                            v = round((v - float(hp.lower)) / q) * q + float(hp.lower)
                        if v < hp.lower:
                            v = float(hp.lower)
                        if v > hp.upper:
                            v = float(hp.upper)
                elif isinstance(hp, CategoricalHyperparameter):
                    choices = list(hp.choices)
                    if v_in not in choices:
                        vs = 'None' if v_in is None else str(v_in)
                        matched = False
                        for c in choices:
                            if str(c) == vs:
                                v = c
                                matched = True
                                break
                        if not matched:
                            continue
                    else:
                        v = v_in
                else:
                    v = v_in
            except Exception as e:
                print(f"Parse Error! {d}")
                continue

            values[k] = v

        return MyConfiguration(Configuration(config_space, values=values))