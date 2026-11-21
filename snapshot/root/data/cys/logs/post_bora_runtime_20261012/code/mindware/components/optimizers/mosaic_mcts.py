import numpy as np
import random
import os
import time
import json
from copy import deepcopy
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from ConfigSpace import Configuration, ConfigurationSpace, Constant, UniformFloatHyperparameter, CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
from openbox.utils.config_space.util import convert_configurations_to_array
from openbox.utils.samplers import LatinHypercubeSampler
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase




class ExpandStrategy:

    def __init__(self, selected_algo: str, parent_id: int):
        self.selected_algo = selected_algo

        self.parent_id = parent_id

    # 写一个转化为dict的方法
    def to_dict(self) -> Dict:
        return {
            "selected_algo": self.selected_algo,
            "parent_id": self.parent_id
        }


class Node:

    global_id = 0
    node_dict = {}

    def __init__(self, parent=None):

        self.visit_mode = 'sqrt'
        self.reward_mode = 'median'

        self.id = Node.global_id
        Node.global_id += 1

        # 注册节点
        Node.node_dict[self.id] = self

        self.parent = parent
        self.visits = 0
        self.best_perf = -np.inf  # 这颗树所有节点的最优值
        self.all_perfs = []  # 这颗树所有子孙节点的perf值

        print(f"Created node {self.id} with parent {self.parent.id if self.parent else None}")

    def _best_child(self, children: list, exploration_weight: float = 1.414, low_perf: float = -np.inf, high_perf: float = -np.inf, only_exploration=False, prior_pi=None) -> 'Node':
        """根据UCB公式选择最佳子节点"""

        if prior_pi is None:
            prior_pi = {child.id: 1 for child in children}

        # UCB公式：value/visits + exploration_weight * sqrt(log(parent_visits)/visits)
        def ucb_score(child):
            if child.visits == 0:
                return float('inf')
            perf_norm = 0
            if self.visit_mode == 'ln':
                explore = exploration_weight * np.sqrt(np.log(self.visits) / child.visits) * prior_pi[child.id]
            elif self.visit_mode == 'sqrt':
                explore = exploration_weight * np.sqrt(self.visits) / (1 + child.visits) * prior_pi[child.id]
            else:
                raise ValueError(f"Invalid visit_mode {self.visit_mode}.")


            if np.isfinite(high_perf) and np.isfinite(low_perf) and high_perf > low_perf:
                if self.reward_mode == 'best':
                    perf_norm = child.best_perf
                elif self.reward_mode == 'median':
                    perf_norm = np.median(child.all_perfs)
                else:
                    raise ValueError(f"Invalid reward_mode {self.reward_mode}.")
                # 增加perf归一化项
                perf_norm = (perf_norm - low_perf) / (high_perf - low_perf)
                perf_norm = max(0, perf_norm)

            return  explore + perf_norm, (perf_norm, explore, prior_pi[child.id])

        children.sort(key=lambda x: ucb_score(x)[0], reverse=True)
        
        # print
        print("Low and high perf:", low_perf, high_perf)
        for child in children:
            print(child, "UCB score:", ucb_score(child))

        return children[0]


    def fetch_all_children(self) -> List['Node']:

        raise NotImplementedError("This method should be implemented in subclasses.")


    def __str__(self):
        if isinstance(self, HPORootNode):
            return f"HPORootNode(id={self.id}, algo={self.algorithm}, visits={self.visits}, best_perf={self.best_perf}, median_perf={np.median(self.all_perfs)})"
        else:
            return f"Node(id={self.id}, visits={self.visits}, best_perf={self.best_perf}, median_perf={np.median(self.all_perfs)})"



class CASHNode(Node):

    name = "CASHNode"

    """蒙特卡洛树CASH算法选择搜索根节点"""

    def __init__(self, include_algorithms: List[str], parent=None):
        super().__init__(parent=parent)

        # include_algorithms 是按照顺序排列的算法列表，按照预测的优先级
        self.include_algorithms = include_algorithms
        self.children = {
            algo: None for algo in include_algorithms
        }  # 根节点的子节点列表

        # 是否启用prior知识
        self.enable_prior = True

        if self.enable_prior:
            num_random_config = 100
            # 每个算法在对应搜索空间里面LHS采样一百个锚点配置
            self.prior_points = {}
            self.prior_models = {}
            for algo in include_algorithms:
                self.prior_points[algo] = self.uniform_sample_configs(algo)
                self.prior_models[algo] = None


    def uniform_sample_configs(self, algo: str, num_configs: int = 100) -> np.ndarray:
        """在指定算法的搜索空间中均匀采样配置点"""
        kd = KnowledgeDatabase.instance()
        cs = kd.bo_history_mab[algo].config_space
        full_cs = kd.bo_history.config_space
        # 把数值型和类别型分开采样
        numerical_cs = ConfigurationSpace()
        categorical_cs = ConfigurationSpace()


        for hp in cs.get_hyperparameters():
            if isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
                numerical_cs.add_hyperparameter(deepcopy(hp))
            else:
                categorical_cs.add_hyperparameter(hp)

        # 一个个处理categorical的组合
        cat_configs = [{}]
        for hp in categorical_cs.get_hyperparameters():
            if isinstance(hp, CategoricalHyperparameter):
                new_cat_configs = []
                for config in cat_configs:
                    for choice in hp.choices:
                        new_config = deepcopy(config)
                        new_config[hp.name] = choice
                        new_cat_configs.append(new_config)
                cat_configs = new_cat_configs
            else:
                for config in cat_configs:
                    config[hp.name] = hp.default_value

        num_configs_per_cat = max(1, num_configs // len(cat_configs))

        lhs = LatinHypercubeSampler(numerical_cs, num_configs_per_cat, criterion='maximin')
        numerical_configs = lhs.generate(return_config=True)

        all_configs = []
        for cat_config in cat_configs:
            for num_config in numerical_configs:
                full_config_dict = deepcopy(cat_config)
                numerical_config_dict = num_config.get_dictionary()
                full_config_dict.update(numerical_config_dict)
                full_config = Configuration(full_cs, values=full_config_dict)
                all_configs.append(full_config)

        # 不够采样数量就随机采样补齐
        while len(all_configs) < num_configs:
            rand_config = cs.sample_configuration()
            full_config_dict = deepcopy(rand_config.get_dictionary())
            full_config = Configuration(full_cs, values=full_config_dict)
            all_configs.append(full_config)

        return convert_configurations_to_array(all_configs[:num_configs])


    def fetch_prior_pi(self, low_perf: float = -np.inf, high_perf: float = -np.inf):

        # 计算每个算法的prior pi值
        prior_pi = {}

        for algo in self.include_algorithms:
            model = self.prior_models[algo]
            algo_id = self.children[algo].id

            points = self.prior_points[algo]
            # 预测这些点的性能
            mean, std = model.predict(points)
            mean = - mean.flatten().mean() # 转化为越大越好
            # 计算这些点的pi值
            if np.isfinite(high_perf) and np.isfinite(low_perf) and high_perf > low_perf:
                # 归一化
                mean = (mean - low_perf) / (high_perf - low_perf)
            else:
                mean = 0.0

            prior_pi[algo_id] = mean

        # sigmoid
        exp_values = np.exp(list(prior_pi.values()))
        sum_exp = np.sum(exp_values)
        for algo_id in prior_pi:
            prior_pi[algo_id] = np.exp(prior_pi[algo_id]) / sum_exp

        return prior_pi


    def is_fully_expanded(self) -> bool:
        """检查节点是否完全扩展"""
        return all(child is not None for child in self.children.values())

    def is_leaf_node(self) -> bool:
        """检查节点是否为叶子节点"""
        return all(child is None for child in self.children.values())

    def fetch_all_children(self) -> List['HPORootNode']:

        return list(self.children.values())

    def best_child(self, exploration_weight: float = 1.414, low_perf: float = -np.inf, high_perf: float = -np.inf) -> 'HPORootNode':
        """根据UCB公式选择最佳子节点"""

        children = self.fetch_all_children()
        assert all(child is not None for child in children), "Not all children are expanded"

        if self.enable_prior:
            prior_pi = self.fetch_prior_pi(low_perf, high_perf)
            print("Prior pi values:", prior_pi)
        else:
            prior_pi = {self.children[algo].id: 1 for algo in self.include_algorithms}

        return super()._best_child(children, exploration_weight, low_perf, high_perf, prior_pi=prior_pi)

    def expand(self):

        for algo in self.include_algorithms:
            if self.children[algo] is None:
                self.children[algo] = HPORootNode(algorithm=algo, parent=self, visit_mode=self.visit_mode, reward_mode=self.reward_mode)

                # 不会停留在选择算法，一定得停留在hpo节点
                return self.children[algo].expand()

        raise ValueError("All algorithm children are already expanded.")



class HPORootNode(Node):

    name = "HPORootNode"

    """蒙特卡洛树每个算法的搜索根节点"""

    def __init__(self, algorithm, parent=None):
        super().__init__(parent=parent)

        self.algorithm = algorithm # 标记是哪个算法

    def is_fully_expanded(self) -> bool:
        """检查节点是否完全扩展"""
        return False # 只有两层

    def expand(self):

        """扩展节点，给出拓展方案"""

        expand_strategy = ExpandStrategy(selected_algo=self.algorithm, parent_id=self.id)

        return expand_strategy

    def get_algorithm(self) -> str:
        return self.algorithm





class MCTSTree:
    """基于蒙特卡洛树搜索的特征工程评估器"""

    def __init__(self, include_algorithms: List[str],
                 exploration_weight: float = 1.414,
                 enable_prior: bool = True,
                 output_dir = './results'):

        self.output_dir = output_dir
        self.exploration_weight = exploration_weight

        # 存储搜索历史
        self.global_scores = []
        self.include_algorithms = include_algorithms

        self.enable_prior = enable_prior
        self.tree = CASHNode(include_algorithms=include_algorithms)

        self.kd = KnowledgeDatabase.instance()

    def update_prior_models(self, model):
        if not self.enable_prior:
            raise ValueError("Prior models are not enabled.")

        # 所有算法共享一个模型
        for algo in self.include_algorithms:
            self.tree.prior_models[algo] = model


    def selection(self) -> 'Node':
        """选择阶段：选择最优子节点直到叶子节点(未被完全探索过的节点)"""

        median_perf = np.percentile(self.global_scores, 5) if len(self.global_scores) > 0 else -np.inf
        # if median_perf is 无穷或者nan, 从小到大排序，找到第一个不是-np.inf的值
        if np.isinf(median_perf) or np.isnan(median_perf):
            sorted_global_scores = sorted(self.global_scores)
            for score in sorted_global_scores:
                if np.isfinite(score):
                    median_perf = score
                    break
        # 找到最优和50分位数
        best_perf = self.tree.best_perf

        current_node = self.tree

        while current_node.is_fully_expanded():
            current_node = current_node.best_child(self.exploration_weight, low_perf=median_perf, high_perf=best_perf)

        assert isinstance(current_node, HPORootNode), "Current node is not a HPORootNode."

        return current_node

    def warmup_selection(self, algorithm) -> 'HPORootNode':
        """warmup选择阶段：选择指定算法的根节点"""

        if self.tree.children[algorithm] is None:
            print("Creating new HPORootNode for algorithm:", algorithm)
            self.tree.children[algorithm] = HPORootNode(algorithm=algorithm, parent=self.tree)

        return self.tree.children[algorithm]

    def expansion(self, node: Node) -> 'ExpandStrategy':
        """扩展阶段：扩展叶子节点"""


        return node.expand()



    def update_results(self, config: Configuration, perf: float, expand_strategy: ExpandStrategy):
        """更新搜索树, 添加新的HPO节点"""

        # perf: 新配置的表现， config: 新配置， action: 采取的动作（大模型的thought或者bo采样），expand_strategy: 扩展策略

        selected_algo = config.get("algorithm", "unknown_algorithm")
        assert selected_algo == expand_strategy.selected_algo, "Selected algorithm mismatch."

        parent_id = expand_strategy.parent_id
        parent = Node.node_dict[parent_id]

        parent.visits += 1
        parent.best_perf = max(parent.best_perf, perf)
        parent.all_perfs.append(perf)
        
        self.tree.visits += 1
        self.tree.best_perf = max(self.tree.best_perf, perf)
        self.tree.all_perfs.append(perf)

        self.global_scores.append(perf)

        return parent

