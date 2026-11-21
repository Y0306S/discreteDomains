from mindware.components.utils.class_loader import get_combined_candidtates
from mindware.components.models.classification import _classifiers, _addons
from mindware.components.utils.constants import *
import os
import psutil


def _available_cpu_count():
    process = psutil.Process()
    if hasattr(process, 'cpu_affinity'):
        return len(process.cpu_affinity())
    return os.cpu_count() or 1


def get_estimator(config, estimator_id):
    classifier_type = estimator_id
    config_ = config.copy()
    config_['%s:random_state' % classifier_type] = 1
    hpo_config = dict()
    for key in config_:
        key_name = key.split(':')[0]
        if classifier_type == key_name:
            act_key = key.split(':')[1]
            hpo_config[act_key] = config_[key]

    _candidates = get_combined_candidtates(_classifiers, _addons)
    estimator = _candidates[classifier_type](**hpo_config)
    if hasattr(estimator, 'n_jobs'):
        available_cpu = _available_cpu_count()
        setattr(estimator, 'n_jobs', min(16, available_cpu))
    return classifier_type, estimator
