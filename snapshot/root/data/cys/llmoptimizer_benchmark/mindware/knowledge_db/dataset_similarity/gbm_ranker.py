import numpy as np
import os
import lightgbm as lgb
import pickle as pkl
from mindware.utils.logging_utils import get_logger
from mindware.knowledge_db.base_ranker import BaseAdvisor
from mindware.knowledge_db.metadata_manager import MetaDataManager
from catboost import CatBoostRegressor
from collections import OrderedDict


class GBMAdvisor(BaseAdvisor):
    def __init__(self, 
                 metadata_manager:MetaDataManager=None):
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)
        super().__init__(meta_algorithm='lightgbm', metadata_manager=metadata_manager)

        self.model = None
        self.embeddings = None

    @staticmethod
    def create_pairwise_data(X, y):

        assert len(X) == len(y)

        n_dataset = y.shape[1]
        X1, labels = list(), list()
        _instance_num = 0

        for i in range(n_dataset):
            for j in range(n_dataset):
                
                if i == j: continue

                if np.isnan(y[i,j]):
                    continue

                X1.append(np.concatenate([X[i], X[j]]))
                labels.append(y[i,j])
                _instance_num += 1

        return np.asarray(X1), np.asarray(labels)

    def fit(self, save_flag=True):
        self.metadata_manager.fetch_all_algo_dataset_similarity()
        _X, _y = self.metadata_manager.load_dataset_similarity_meta_data()
        self.embeddings = _X
        self.model = {}

        for i, algo in enumerate(self.algorithms):
            meta_learner_dir = os.path.join(self.meta_dir, "meta_learner", "sim_model_%s_%s" % (self.meta_algo, self.metric))
            meta_learner_filename = os.path.join(meta_learner_dir, 'sim_model_%s_%s_%s_%s.pkl' % (self.meta_algo, self.metric, algo, self.hash_id))
            if save_flag and not os.path.exists(meta_learner_dir):
                os.makedirs(meta_learner_dir)
            if save_flag and os.path.exists(meta_learner_filename):
                # print("load model...")
                with open(meta_learner_filename, 'rb') as f:
                    self.model[algo] = pkl.load(f)
            else:
                # print(_X.shape, _y.shape)
                X, y = self.create_pairwise_data(_X, _y[algo])
                surrogate_cat = CatBoostRegressor()
                surrogate_cat.fit(X, y, silent=True)
                # surrogate_cat = lgb.LGBMRegressor(verbose=0)
                # surrogate_cat.fit(X, y)
                self.model[algo] = surrogate_cat
                if save_flag:
                    with open(meta_learner_filename, 'wb') as f:
                        pkl.dump(surrogate_cat, f)

    def predict(self, meta_feature):
        n_dataset = self.n_dataset_candidates
        _X = list()
        for i in range(n_dataset):

            meta_x = np.array(meta_feature).copy()
            meta_y = self.embeddings[i].copy()
            _X.append(np.concatenate((meta_x, meta_y)))

        sims_dict = {}
        for algo in self.algorithms:
            sims_dict[algo] = self.model[algo].predict(_X)

        return sims_dict


    def fetch_dataset_similarity(self, target_feature, algorithms):
        sims_dict = self.predict(target_feature)
        res_dict = {}
        for algo in algorithms:
            idxs = np.argsort(-sims_dict[algo])
            sorted_datasets = [self.datasets[idx] for idx in idxs]
            sorted_scores = [sims_dict[algo][idx] for idx in idxs]
            res_dict[algo] = OrderedDict(zip(sorted_datasets, sorted_scores))
        return res_dict
