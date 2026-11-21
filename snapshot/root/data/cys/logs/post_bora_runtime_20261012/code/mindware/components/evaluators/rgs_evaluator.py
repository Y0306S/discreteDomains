from mindware.components.utils.class_loader import get_combined_candidtates
from mindware.components.models.regression import _regressors, _addons
from mindware.components.utils.constants import *
import os
import psutil


def _available_cpu_count():
    process = psutil.Process()
    if hasattr(process, 'cpu_affinity'):
        return len(process.cpu_affinity())
    return os.cpu_count() or 1


def get_estimator(config, estimator_id):
    regressor_type = estimator_id
    config_ = config.copy()
    config_['%s:random_state' % regressor_type] = 1
    hpo_config = dict()
    for key in config_:
        key_name = key.split(':')[0]
        if regressor_type == key_name:
            act_key = key.split(':')[1]
            hpo_config[act_key] = config_[key]

    _candidates = get_combined_candidtates(_regressors, _addons)
    estimator = _candidates[regressor_type](**hpo_config)
    if hasattr(estimator, 'n_jobs'):
        available_cpu = _available_cpu_count()
        cap = int(os.environ.get("EVAL_N_JOBS", "16") or 16)
        setattr(estimator, 'n_jobs', max(1, min(cap, available_cpu)))
    return regressor_type, estimator
