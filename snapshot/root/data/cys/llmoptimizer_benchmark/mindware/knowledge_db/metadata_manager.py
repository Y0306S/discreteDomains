import os
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
from mindware.utils.logging_utils import get_logger

from mindware.knowledge_db.utils.meta_features import calculate_metafeatures
from mindware.components.utils.constants import CLS_TASKS, RGS_TASKS
from scipy.stats import kendalltau
from filelock import FileLock

def write_pkl_file(file_path, data):
    lock_path = file_path + '.lock'
    lock = FileLock(lock_path)

    with lock:
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
        print(f"Data written. {file_path}")

def read_pkl_file(file_path):
    lock_path = file_path + '.lock'
    lock = FileLock(lock_path)

    with lock:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        print(f"Data read. {file_path}")
        return data


# 65个
_cls_builtin_datasets = ['KDDCup09_appetency', 'ada', 'adult', 'airlines', 'albert', 'amazon-commerce-reviews', 'Amazon_employee_access', 'APSFailure', 'arcene', 'Australian', 'bank-marketing', 'Bioresponse', 
                         'blood-transfusion-service-center', 'car', 'christine', 'churn', 'Click_prediction_small', 'cmc', 'cnae-9', 'connect-4', 'covertype', 'credit-g', 'Diabetes130US', 
                         'dilbert', 'dionis', 'dna', 'eucalyptus', 'fabert', 'Fashion-MNIST', 'first-order-theorem-proving', 'GesturePhaseSegmentationProcessed', 'gina', 'helena', 
                         'Internet-Advertisements', 'jannis', 'jasmine', 'jungle_chess_2pcs_raw_endgame_complete', 'kc1', 'kick', 'kr-vs-kp', 'madeline', 'mfeat-factors', 
                         'micro-mass', 'MiniBooNE', 'nomao', 'numerai28.6', 'okcupid-stem', 'ozone-level-8hr', 'pc4', 'philippine', 'PhishingWebsites', 'phoneme', 'porto-seguro', 'qsar-biodeg', 
                         'Satellite', 'segment', 'sf-police-incidents', 'shuttle', 'steel-plates-fault', 'sylvine', 'vehicle', 'volkert', 'wilt', 'wine-quality-white', 'yeast']
# 30个
_rgs_builtin_datasets = ['abalone', 'Allstate_Claims_Severity', 'black_friday', 'boston', 'Brazilian_houses', 'colleges', 'diamonds', 'elevators', 'house_16H', 'house_prices_nominal', 'house_sales', 
                         'Mercedes_Benz_Greener_Manufacturing', 'MIP-2016-regression', 'Moneyball', 'nyc-taxi-green-dec-2016', 'OnlineNewsPopularity', 'pol', 'QSAR-TID-10980', 'QSAR-TID-11', 'quake', 
                         'Santander_transaction_value', 'SAT11-HAND-runtime-regression', 'sensory', 'socmob', 'space_ga', 'tecator', 'topo_2_1', 'us_crime', 'wine_quality', 'yprop_4_1']



_cls_builtin_algorithms = ['catboost', 'lightgbm', 'random_forest', 'extra_trees', 'adaboost', 'gradient_boosting', 'xgboost']
_rgs_builtin_algorithms = ['catboost', 'lightgbm', 'random_forest', 'extra_trees', 'adaboost', 'gradient_boosting', 'xgboost']



class MetaDataManager(object):

    def __init__(self, task_type, metric, meta_dir, original_data_dir, exclude_datasets=None, include_algorithms=None):

        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)

        builtin_loc = os.path.dirname(__file__)
        builtin_loc = os.path.join(builtin_loc, 'meta_resource')
        self.meta_dir = meta_dir if meta_dir is not None else builtin_loc

        self.original_data_dir = original_data_dir

        self.task_type = task_type
        if task_type in CLS_TASKS:
            self.task_prefix = 'cls'
            self.builtin_datasets = _cls_builtin_datasets
            self.builtin_algorithms = _cls_builtin_algorithms
            if metric not in ['bal_acc', 'acc', 'f1', 'auc']:
                self.logger.info('Meta information about metric-%s does not exist, use accuracy instead.' % str(metric))
                metric = 'bal_acc'
        elif task_type in RGS_TASKS:
            self.task_prefix = 'rgs'
            self.builtin_datasets = _rgs_builtin_datasets
            self.builtin_algorithms = _rgs_builtin_algorithms
            if metric not in ['mse', 'r2', 'mae']:
                self.logger.info('Meta information about metric-%s does not exist, use accuracy instead.' % str(metric))
                metric = 'mse'
        else:
            raise ValueError('Invalid metric: %s.' % metric)

        # 真正用上的
        if include_algorithms is not None:
            self.algorithms = [t for t in self.builtin_algorithms if t in include_algorithms]
        else:
            self.algorithms = self.builtin_algorithms

        self.metric = metric

        self._dataset_embedding = dict()  # dataset: embsedding
        self._dataset_perf4algo = dict()  # dataset: {algorithm: perf}
        self._dataset_similarity = dict() # algorithm: {dataset1: {dataset2: sim}} Dataframe


        if exclude_datasets is None:
            exclude_datasets = []
        else:
            exclude_datasets = [t for t in exclude_datasets if t in self.builtin_datasets]

        if len(exclude_datasets) == 0:
            self.hash_id = 'none'
        else:
            self.builtin_datasets = [t for t in self.builtin_datasets if t not in exclude_datasets]
            exclude_datasets = list(set(exclude_datasets))
            exclude_str = ','.join(sorted(exclude_datasets))
            md5 = hashlib.md5()
            md5.update(exclude_str.encode('utf-8'))
            self.hash_id = md5.hexdigest()

        self.already_load_all_meta_features = False
        self.already_load_all_algorithm_performance = False
        self.already_load_all_algorithm_dataset_similarity = False

        self.fetch_all_meta_features()

    # def fetch_meta_runs(self, dataset):
    #     meta_dataset_dir = os.path.join(self.meta_dir, 'meta_dataset_vec')
    #     save_path2 = os.path.join(meta_dataset_dir, '%s_meta_dataset_algo2perf.pkl' % self.task_prefix)
    #     assert os.path.exists(save_path2)

    #     data2 = read_pkl_file(save_path2)

    #     task_id = 'init_%s' % dataset
    #     idx = data2['task_ids'].index(task_id)
    #     sel_idx2 = [data2['algorithms_included'].index(t) for t in self.builtin_algorithms]
    #     return data2['perf4algo'][self.metric][idx, sel_idx2]


    def get_meta_features(self, dataset):

        if isinstance(dataset, str):

            save_path = os.path.join(self.meta_dir, 'meta_dataset_vec', '%s_meta_dataset_embedding.pkl' % self.task_prefix)

            if os.path.exists(save_path):
                data1 = read_pkl_file(save_path)
                if dataset in data1:
                    return data1[dataset]

        meta_feature = calculate_metafeatures(dataset, task_type=self.task_type)
        sorted_keys = sorted(meta_feature.keys())
        meta_feature = [meta_feature[key] for key in sorted_keys]

        return meta_feature


    # 加载所有的元特征
    def fetch_all_meta_features(self):

        if self.already_load_all_meta_features:
            return

        meta_dataset_dir = os.path.join(self.meta_dir, 'meta_dataset_vec')
        if not os.path.exists(meta_dataset_dir):
            os.makedirs(meta_dataset_dir)
        save_path = os.path.join(meta_dataset_dir, '%s_meta_dataset_embedding.pkl' % self.task_prefix)

        change_flag = False
        if os.path.exists(save_path):
            data1 = read_pkl_file(save_path)
        else:
            data1 = dict()


        for task_id in self.builtin_datasets:
            if task_id in data1:
                self._dataset_embedding[task_id] = data1[task_id]
            else:
                meta_feature = calculate_metafeatures(task_id, task_type=self.task_type)
                sorted_keys = sorted(meta_feature.keys())
                meta_feature = [meta_feature[key] for key in sorted_keys]
                self._dataset_embedding[task_id] = meta_feature
                data1[task_id] = meta_feature
                change_flag = True

        if change_flag:
            write_pkl_file(save_path, data1)
        self.already_load_all_meta_features = True

        
    def _algorithm_history_path(self, dataset, algorithm):
        from mindware.modules.configs import CONFIGS

        scope = getattr(CONFIGS, "HISTORY_RUN_SCOPE", None)
        if scope:
            return os.path.join(scope, f"{algorithm}.json")
        return os.path.join(
            self.original_data_dir, self.task_prefix, self.metric, dataset, f"{algorithm}.json"
        )

    def fetch_algorithm_runs(self, dataset, algorithm):

        algorithm_path = self._algorithm_history_path(dataset, algorithm)
        if not os.path.exists(algorithm_path):
            return None, None
        
        with open(algorithm_path, 'r') as f:
            run_data = json.load(f)

        run_data = run_data['observations']
        configurations = [run['config'] for run in run_data]
        perfs = [run['objectives'][0] for run in run_data]

        return configurations, perfs


    def fetch_algorithm_runs_history(self, dataset, algorithm, cs=None):

        if cs is None:
            from mindware.components.config_space.cs_builder import get_hpo_cs
            cs = get_hpo_cs(estimator_id=algorithm, task_type=self.task_type)

        from openbox import History

        algorithm_path = self._algorithm_history_path(dataset, algorithm)
        if not os.path.exists(algorithm_path):
            return None

        history = History.load_json(algorithm_path, config_space=cs)

        return history


    # 加载所有的综合表现
    def fetch_all_algo_perfs(self):

        if self.already_load_all_algorithm_performance:
            return

        meta_dataset_dir = os.path.join(self.meta_dir, 'meta_dataset_vec')
        if not os.path.exists(meta_dataset_dir):
            os.makedirs(meta_dataset_dir)
        save_path2 = os.path.join(meta_dataset_dir, '%s_meta_dataset_algo2perf.pkl' % self.task_prefix)

        change_flag = False
        if os.path.exists(save_path2):

            data2 = read_pkl_file(save_path2)

        else:
            data2 = dict()

        if self.metric not in data2:
            data2[self.metric] = dict()

        for task_id in self.builtin_datasets:
            self._dataset_perf4algo[task_id] = {}

            if task_id not in data2[self.metric]:
                data2[self.metric][task_id] = dict()

            for algo in self.builtin_algorithms:
                # 已经有了
                if algo in data2[self.metric][task_id]:
                    self._dataset_perf4algo[task_id][algo] = data2[self.metric][task_id][algo]
                else:
                    _, perfs = self.fetch_algorithm_runs(task_id, algo)

                    if perfs is None or len(perfs) == 0:
                        continue

                    perf = - min(perfs)  # 转化为越大越好
                    self._dataset_perf4algo[task_id][algo] = perf
                    # 更新进去
                    data2[self.metric][task_id][algo] = perf
                    change_flag = True

        if change_flag:
            write_pkl_file(save_path2, data2)

        self.already_load_all_algorithm_performance = True


    def load_dataset_algorithm_meta_data(self):

        X = []
        y = []

        for dataset in self.builtin_datasets:
            meta_feature = self._dataset_embedding[dataset]
            perf4algo = self._dataset_perf4algo[dataset]


            perf_list = []
            failed = False
            for algo in self.builtin_algorithms:
                
                if algo not in perf4algo:
                    failed = True
                    break
                perf_list.append(perf4algo[algo])

            if not failed:
                X.append(meta_feature)
                y.append(perf_list)

        X = np.array(X)
        y = np.array(y)

        return X, y

    
    def load_dataset_similarity_meta_data(self):
        
        X = np.array([self._dataset_embedding[dataset] for dataset in self.builtin_datasets])

        Y = {algorithm: self._dataset_similarity[algorithm].values for algorithm in self.algorithms}

        return X, Y



    # 加载所有的综合表现
    def fetch_all_algo_dataset_similarity(self):

        from mindware.components.config_space.cs_builder import get_hpo_cs
        from openbox.utils.util_funcs import get_types
        from openbox.utils.config_space.util import convert_configurations_to_array
        from openbox.utils.config_space.space_utils import get_config_from_dict
        from openbox.surrogate.base.build_gp import create_gp_model


        seed = 42
        rng = np.random.RandomState(seed)

        if self.already_load_all_algorithm_dataset_similarity:
            return

        # 写一段快速排序 算法的代码
        meta_dataset_dir = os.path.join(self.meta_dir, 'meta_dataset_vec')
        save_path3 = os.path.join(meta_dataset_dir, '%s_meta_dataset_similarity.pkl' % self.task_prefix)
        pred_cache_path = os.path.join(meta_dataset_dir, '%s_meta_dataset_pred_cache.pkl' % self.task_prefix)

        change_flag = False
        if os.path.exists(save_path3):
            data3 = read_pkl_file(save_path3)
        else:
            data3 = dict()

        if os.path.exists(pred_cache_path):
            pred_cache = read_pkl_file(pred_cache_path)
        else:
            pred_cache = dict()

        if self.metric not in data3:
            data3[self.metric] = dict()
        if self.metric not in pred_cache:
            pred_cache[self.metric] = dict()
        if 'sampled_configurations' not in pred_cache:
            pred_cache['sampled_configurations'] = dict()

        for algorithm in self.builtin_algorithms:

            if algorithm not in data3[self.metric]:
                data3[self.metric][algorithm] = pd.DataFrame()

            if algorithm not in pred_cache[self.metric]:
                pred_cache[self.metric][algorithm] = dict()

            if algorithm not in pred_cache['sampled_configurations']:
                cs = get_hpo_cs(estimator_id=algorithm, task_type=self.task_type)
                cs.seed(seed)
                # 采样25个随机样本用于预测计算相似度
                configurations = cs.sample_configuration(100)
                pred_cache['sampled_configurations'][algorithm] = configurations
            else:
                configurations = pred_cache['sampled_configurations'][algorithm]
            testX = convert_configurations_to_array(configurations)

            self._dataset_similarity[algorithm] = pd.DataFrame(index=self.builtin_datasets, columns=self.builtin_datasets)
            
            # 对角线填1
            for dataset in self.builtin_datasets:
                self._dataset_similarity[algorithm].loc[dataset, dataset] = 1.0



            # 找到data3[self.metric]中已经包含的
            """
                * * * * *       all_caches
                    * * * * *   self.builtin_datasets
                    * * *       already_exists
                          * *   remainings
                    *     * *   need_cal
            """
            all_caches = list(data3[self.metric][algorithm].index)
            already_exists = list(set(all_caches).intersection(set(self.builtin_datasets)))
            remainings = [dataset for dataset in self.builtin_datasets if dataset not in already_exists]


            # 已经有的，直接填入
            if len(already_exists) > 0:
                self._dataset_similarity[algorithm].loc[already_exists, already_exists] = data3[self.metric][algorithm].loc[already_exists, already_exists]

            # 找到已经有的，但是是nan的
            nan_mask = self._dataset_similarity[algorithm].isna()
            need_cal = [dataset for dataset in self.builtin_datasets if nan_mask.loc[dataset].any()]


            
            for dataset in need_cal:
                if dataset in pred_cache[self.metric][algorithm]:
                    continue

                history = self.fetch_algorithm_runs_history(dataset, algorithm, cs)
                if history is None:
                    continue

                self.logger.info('Fitting surrogate for dataset-%s, algorithm-%s.' % (dataset, algorithm))

                types, bounds = get_types(cs)
                surrogate_gp = create_gp_model(model_type='gp', config_space=cs, types=types, bounds=bounds, rng=rng)

                train_X = history.get_config_array()
                train_Y = history.get_objectives()
                surrogate_gp.train(train_X, train_Y)

                pred_Y, _ = surrogate_gp.predict(testX)
                pred_cache[self.metric][algorithm][dataset] = pred_Y

            # 计算相似度
            if len(remainings) > 0:
                # 将remain_datasets拼接到data3[self.metric][algorithm]后面和下面，增加行和列
                # 获取新的索引和列名（包括已有的和 remainings）
                current_df = data3[self.metric][algorithm]
                new_index = list(current_df.index) + remainings
                new_columns = list(current_df.columns) + remainings

                # 使用 reindex 扩展 DataFrame
                expanded_df = current_df.reindex(index=new_index, columns=new_columns)
                data3[self.metric][algorithm] = expanded_df


            for dataset1 in self.builtin_datasets:
                # 给
                for dataset2 in self.builtin_datasets:
                    if dataset1 == dataset2:
                        continue
                    if not nan_mask.loc[dataset1, dataset2]:
                        continue
                    if dataset1 not in pred_cache[self.metric][algorithm] or dataset2 not in pred_cache[self.metric][algorithm]:
                        continue

                    pred1 = pred_cache[self.metric][algorithm][dataset1]
                    pred2 = pred_cache[self.metric][algorithm][dataset2]

                    # 肯德尔相似度，掉包计算

                    sim, _ = kendalltau(pred1, pred2)
                    sim = (sim + 1) / 2  # 归一化到0-1之间

                    self._dataset_similarity[algorithm].loc[dataset1, dataset2] = sim
                    self._dataset_similarity[algorithm].loc[dataset2, dataset1] = sim
                    data3[self.metric][algorithm].loc[dataset1, dataset2] = sim
                    data3[self.metric][algorithm].loc[dataset2, dataset1] = sim

            # 保存下来
            write_pkl_file(save_path3, data3)
            write_pkl_file(pred_cache_path, pred_cache)


        self.already_load_all_algorithm_dataset_similarity = True