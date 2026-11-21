from abc import ABCMeta
from mindware.components.metrics.metric import get_metric
from mindware.components.utils.constants import *

class _BaseEvaluator(metaclass=ABCMeta):
    def __init__(self, estimator, metric, task_type,
                 evaluation_strategy, **evaluation_params):
        self.estimator = estimator
        if task_type not in TASK_TYPES:
            raise ValueError('Unsupported task type: %s' % task_type)
        self.metric = get_metric(metric)
        self.metric_name = metric
        self.evaluation_strategy = evaluation_strategy
        self.evaluation_params = evaluation_params

        if self.evaluation_strategy == 'holdout':
            if 'train_size' not in self.evaluation_params:
                self.evaluation_params['train_size']

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()
