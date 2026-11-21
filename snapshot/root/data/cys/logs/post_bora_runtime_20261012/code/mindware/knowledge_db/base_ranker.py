import os
import numpy as np
import pickle as pkl
from collections import OrderedDict
from mindware.utils.logging_utils import get_logger
from mindware.components.utils.constants import CLS_TASKS, RGS_TASKS



class BaseAdvisor(object):
    def __init__(self, 
                 meta_algorithm='lightgbm',
                 metadata_manager=None):

        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)
        
        self.meta_algo = meta_algorithm

        # Init BaseAdvisor via MetaDataManager
        self.task_type = metadata_manager.task_type
        self.metric = metadata_manager.metric
        self.hash_id = metadata_manager.hash_id

        self.builtin_algorithms = metadata_manager.builtin_algorithms  # 所有内置算法
        self.n_algo_candidates = len(self.builtin_algorithms)

        self.algorithms = metadata_manager.algorithms  # 用户指定的算法列表

        self.datasets = metadata_manager.builtin_datasets
        self.n_dataset_candidates = len(self.datasets)

        self.metadata_manager = metadata_manager
        self.meta_dir = metadata_manager.meta_dir
        
        # MetaDataManager(metadata_dir=self.meta_dir, exclude_datasets=self.exclude_datasets, metric=metric, task_type=task_type)
        self.meta_learner = None


    def fetch_run_results(self, dataset):
        scores = self.metadata_manager.fetch_meta_runs(dataset)
        idxs = np.argsort(-scores)
        sorted_algos = [self.algorithms[idx] for idx in idxs]
        sorted_scores = [scores[idx] for idx in idxs]
        return OrderedDict(zip(sorted_algos, sorted_scores))

    def fit(self):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()
