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

LLMEXPLORATION = 'EXPLORATION'
LLMEXPLOITATION = 'EXPLOITATION'
BAYESIANLOCAL = 'bo_local_search'
BAYESIANRANDOM = 'bo_random_search'
WARMUP = 'warmup'



class ExpandStrategy:

    def __init__(self, selected_algo: str, strategy: str, parent_id: int):
        self.selected_algo = selected_algo

        assert strategy in [LLMEXPLORATION, LLMEXPLOITATION, BAYESIANRANDOM, BAYESIANLOCAL, WARMUP], "Invalid strategy type."
        self.strategy = strategy
        self.parent_id = parent_id

    # 写一个转化为dict的方法
    def to_dict(self) -> Dict:
        return {
            "selected_algo": self.selected_algo,
            "strategy": self.strategy,
            "parent_id": self.parent_id
        }


class Node:

    global_id = 0
    node_dict = {}

    def __init__(self, parent=None, visit_mode: str = 'ln', reward_mode: str = 'best'):

        assert visit_mode in ['ln', 'sqrt'], "Invalid visit_mode."
        self.visit_mode = visit_mode
        assert reward_mode in ['best', 'median'], "Invalid reward_mode."
        self.reward_mode = reward_mode

        self.id = Node.global_id
        Node.global_id += 1

        # 注册节点
        Node.node_dict[self.id] = self

        self.parent = parent
        self.visits = 0
        self.value = 0.0  # 累计奖励值
        self.best_perf = -np.inf  # 这颗树所有节点的最优值
        self.all_perfs = []  # 这颗树所有子孙节点的perf值

        print(f"Created node {self.id} with parent {self.parent.id if self.parent else None}")

    def _best_child(self, children: list, exploration_weight: float = 1.414, low_perf: float = -np.inf, high_perf: float = -np.inf, only_exploration=False, prior_pi=None) -> 'Node':
        """根据UCB公式选择最佳子节点"""

        if prior_pi is None:
            prior_pi = {child.id: 1 / len(children) for child in children}

        # UCB公式：value/visits + exploration_weight * sqrt(log(parent_visits)/visits)
        def ucb_score(child):
            if child.visits == 0:
                return float('inf')
            perf_norm = 0
            exploit = 0
            if self.visit_mode == 'ln':
                explore = exploration_weight * np.sqrt(np.log(self.visits) / child.visits) * prior_pi[child.id]
            elif self.visit_mode == 'sqrt':
                explore = exploration_weight * np.sqrt(self.visits) / (1 + child.visits) * prior_pi[child.id]
            else:
                raise ValueError(f"Invalid visit_mode {self.visit_mode}.")
            if not only_exploration:
                exploit = child.value / child.visits

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

            return exploit + explore + perf_norm, (prior_pi[child.id], exploit, perf_norm, explore)

        children.sort(key=lambda x: ucb_score(x)[0], reverse=True)
        
        # print
        print("Low and high perf:", low_perf, high_perf)
        for child in children:
            print(child, "UCB score:", ucb_score(child))

        if len(children) >= 2 and ucb_score(children[0])[0] == ucb_score(children[1])[0]:
            print(f"UCB tie between nodes {children[0].id} and {children[1].id} with score {ucb_score(children[0])[0]:.4f}. Selecting better perf.")
            if isinstance(children[0], HPONode) and isinstance(children[1], HPONode):
                if children[0].perf != children[1].perf:
                    return children[0] if children[0].perf > children[1].perf else children[1]

        return children[0]

    def reset(self):
        """重置节点统计信息"""
        self.visits = 0
        self.value = 0.0
        self.best_perf = -np.inf

        print(f"reset node {self.id}")


    def fetch_all_children(self) -> List['Node']:

        raise NotImplementedError("This method should be implemented in subclasses.")


    def __str__(self):
        if isinstance(self, HPORootNode):
            return f"HPORootNode(id={self.id}, algo={self.algorithm}, visits={self.visits}, value={self.value}, best_perf={self.best_perf}, median_perf={np.median(self.all_perfs)})"
        else:
            return f"Node(id={self.id}, visits={self.visits}, value={self.value}, best_perf={self.best_perf}, median_perf={np.median(self.all_perfs)})"



class CASHNode(Node):

    name = "CASHNode"

    """蒙特卡洛树CASH算法选择搜索根节点"""

    def __init__(self, include_algorithms: List[str], parent=None, warmup_num: int = 10, enable_prior: bool = True, visit_mode: str = 'ln', reward_mode: str = 'best'):
        super().__init__(parent=parent, visit_mode=visit_mode, reward_mode=reward_mode)

        # include_algorithms 是按照顺序排列的算法列表，按照预测的优先级
        self.include_algorithms = include_algorithms
        self.children = {
            algo: None for algo in include_algorithms
        }  # 根节点的子节点列表

        self.warmup_num = warmup_num

        # 是否启用prior知识
        self.enable_prior = enable_prior

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
                full_config = Configuration(cs, values=full_config_dict)
                all_configs.append(full_config)

        # 不够采样数量就随机采样补齐
        while len(all_configs) < num_configs:
            rand_config = cs.sample_configuration()
            all_configs.append(rand_config)

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

        only_exploration = False
        # 确保一下每个算法至少都探索个10次
        if any(child.visits < self.warmup_num for child in children):
            only_exploration = True
            print(f"Selecting under-explored algorithm node first. only_exploration={only_exploration}")

        if not only_exploration and self.enable_prior:
            prior_pi = self.fetch_prior_pi(low_perf, high_perf)
            print("Prior pi values:", prior_pi)
        else:
            prior_pi = None

        return super()._best_child(children, exploration_weight, low_perf, high_perf, only_exploration, prior_pi=prior_pi)

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

    def __init__(self, algorithm, parent=None, warmup_num: int = 3, visit_mode: str = 'ln', reward_mode: str = 'best'):
        super().__init__(parent=parent, visit_mode=visit_mode, reward_mode=reward_mode)

        self.algorithm = algorithm # 标记是哪个算法

        self.warmup_num = warmup_num

        self.children = {
            "warmup": [],
            "bo_random_search": []
        }  # 根节点的子节点列表

    def is_fully_expanded(self) -> bool:
        """检查节点是否完全扩展"""
        return len(self.children["warmup"]) >= self.warmup_num # 假设每个算法至少有两个初始子节点

    def is_leaf_node(self) -> bool:
        """检查节点是否为叶子节点"""
        return len(self.children["warmup"]) == 0 and len(self.children["bo_random_search"]) == 0

    def fetch_all_children(self) -> List['HPONode']:

        return self.children["warmup"] + self.children["bo_random_search"]


    def best_child(self, exploration_weight: float = 1.414, low_perf: float = -np.inf, high_perf: float = -np.inf) -> 'HPONode':
        """根据UCB公式选择最佳子节点"""

        children = self.fetch_all_children()
        # 用perf排序，只保留前四
        children.sort(key=lambda x: x.perf, reverse=True)
        children = children[:self.warmup_num + 2]
        assert len(children) >= self.warmup_num, "No children to select from"

        return super()._best_child(children, exploration_weight, low_perf, high_perf)

    def expand(self):

        """扩展节点，给出拓展方案"""

        expand_strategy = ExpandStrategy(selected_algo=self.algorithm, strategy=WARMUP, parent_id=self.id)

        return expand_strategy

    def add_child(self, child_node, strategy):
        """添加子节点"""
        if strategy == WARMUP:
            self.children["warmup"].append(child_node)
        elif strategy == BAYESIANRANDOM:
            self.children["bo_random_search"].append(child_node)
            # 根据perf排序bo_random_search节点列表
            self.children["bo_random_search"].sort(key=lambda x: x.perf, reverse=True)
        else:
            raise ValueError(f"Invalid strategy type {strategy} for adding child to HPORootNode.")

    def get_algorithm(self) -> str:
        return self.algorithm



class HPONode(Node):

    name = "HPONode"

    """蒙特卡洛树搜索节点"""

    def __init__(self, config: Configuration, perf: float, action: str, reflection:str, parent=None, visit_mode: str = 'ln', reward_mode: str = 'best'):
        super().__init__(parent=parent, visit_mode=visit_mode, reward_mode=reward_mode)

        # 把node_id存入config中，便于在bo采样的时候，根据mcts_node_id标记好local search的配置点对应哪个node
        config.extra_info['mcts_node_id'] = self.id

        self.config = config  # 当前状态，包括config和perf
        self.perf = perf
        self.action = action  # 导致此节点的动作超参数的调整
        self.reflection = reflection
        self.children = {
            "exploration": None,
            "exploitation": None,
            "bo": []
        }

    def is_fully_expanded(self) -> bool:
        """检查节点是否完全扩展"""
        return self.children['exploration'] is not None and \
               self.children['exploitation'] is not None

    def is_leaf_node(self) -> bool:
        """检查节点是否为叶子节点"""
        return self.children['exploration'] is None and \
                self.children['exploitation'] is None and \
                len(self.children['bo']) == 0

    def fetch_all_children(self) -> List['HPONode']:

        return [self.children['exploration'], self.children['exploitation']] + self.children['bo'][:2]

    def best_child(self, exploration_weight: float = 1.414, low_perf: float = -np.inf, high_perf: float = -np.inf) -> 'HPONode':
        """根据UCB公式选择最佳子节点"""

        children = self.fetch_all_children()
        assert all(child is not None for child in children), "Not all children are expanded"

        return super()._best_child(children, exploration_weight, low_perf, high_perf)

    def expand(self):

        """扩展节点，给出拓展方案"""
        # 向上回溯，找到当前节点所属的 HPORootNode (即算法根节点)
        algo_node = self.parent
        while not isinstance(algo_node, HPORootNode):
            algo_node = algo_node.parent

        selected_algo = algo_node.algorithm # 选择算法

        # expolration 和 exploitation 二选一
        # if self.children['exploration'] is None and self.children['exploitation'] is None:
        #     strategy = random.choice([LLMEXPLORATION, LLMEXPLOITATION])
        if self.children['exploration'] is None: # 优先探索
            strategy = LLMEXPLORATION
        elif self.children['exploitation'] is None:
            strategy = LLMEXPLOITATION
        else:
            raise ValueError("Both exploration and exploitation children are already expanded.")

        expand_strategy = ExpandStrategy(selected_algo=selected_algo, strategy=strategy, parent_id=self.id)

        return expand_strategy

    def add_child(self, child_node, strategy: str):
        """添加子节点"""
        if strategy == LLMEXPLORATION:
            self.children['exploration'] = child_node
        elif strategy == LLMEXPLOITATION:
            self.children['exploitation'] = child_node
        elif strategy == BAYESIANLOCAL:
            self.children['bo'].append(child_node)
            # 根据perf排序bo节点列表
            self.children['bo'].sort(key=lambda x: x.perf, reverse=True)
        else:
            raise ValueError(f"Invalid strategy type {strategy} for adding child to HPONode.")

    def get_algorithm(self) -> str:
        # 向上回溯，找到当前节点所属的 HPORootNode (即算法根节点)
        algo_node = self.parent
        while not isinstance(algo_node, HPORootNode):
            algo_node = algo_node.parent

        return algo_node.algorithm



class MCTSTree:
    """基于蒙特卡洛树搜索的特征工程评估器"""

    def __init__(self, include_algorithms: List[str],
                 exploration_weight: float = 1.414,
                 warmup_num: int = 10,
                 enable_prior: bool = True,
                 visit_mode: str = 'sqrt',
                 reward_mode: str = 'best',
                 output_dir = './results'):

        self.output_dir = output_dir
        self.exploration_weight = exploration_weight

        # 存储搜索历史
        self.global_scores = []
        self.algo_scores = {algo: [] for algo in include_algorithms}

        self.warmup_num = warmup_num
        self.enable_prior = enable_prior
        self.visit_mode = visit_mode
        self.reward_mode = reward_mode
        self.tree = CASHNode(include_algorithms=include_algorithms, warmup_num=warmup_num, enable_prior=enable_prior, visit_mode=visit_mode, reward_mode=reward_mode)

        self.best_nodes = {
            algo: [] for algo in include_algorithms
        }  # 每个算法的最优点

        self.already_reset_warmup = False # 最多重置一次warmup节点

        self.kd = KnowledgeDatabase.instance()
        self.cpp_dict = {}

        for algo in include_algorithms:
            cs = self.kd.bo_history_mab[algo].config_space

            cpp = ChangePathPareto(cs=cs)
            self.cpp_dict[algo] = cpp


    def update_prior_models(self, algo: str, model):
        if not self.enable_prior:
            raise ValueError("Prior models are not enabled.")

        self.tree.prior_models[algo] = model


    def reset_warmup_nodes(self):
        # warmup 完成后，重新设置CASHNode、HPORootNode和warmup节点的value
        # 因为warmup阶段点太少，相对于全局或者局部提升没有意义

        assert not self.already_reset_warmup, "Warmup nodes have already been reset."

        print("Resetting warmup nodes")

        self.visualize("before_warmup_reset")
        self.tree.reset()

        warmup_nodes = []
        for algo in self.tree.children:
            child = self.tree.children[algo]
            # 重置best_nodes，只保留warmup里面最好的一个
            if len(self.best_nodes[algo]) > 1:
                self.best_nodes[algo] = [max(self.best_nodes[algo], key=lambda x: x.perf)]
            if child is None:
                continue

            child.reset()
            for warmup_node in child.children["warmup"] + child.children["bo_random_search"]:
                assert warmup_node.children["exploration"] is None and warmup_node.children["exploitation"] is None and \
                    len(warmup_node.children["bo"]) == 0, "Warmup node should not have children"
                warmup_node.reset()
                warmup_nodes.append(warmup_node)

        # 排序
        warmup_nodes.sort(key=lambda x: x.perf, reverse=True)
        # 取出前三个
        best_warmup_nodes = warmup_nodes[:3]
        # 兼容perf一样的
        for node in warmup_nodes[3:]:
            if node.perf == best_warmup_nodes[-1].perf:
                best_warmup_nodes.append(node)
        best_warmup_algos = [node.get_algorithm() for node in best_warmup_nodes]
        # 每种算法最多保留两个
        drop_idxs = []
        for i in range(len(best_warmup_algos)):
            if best_warmup_algos[:i].count(best_warmup_algos[i]) >= 2:
                drop_idxs.append(i)
        best_warmup_nodes = [node for i, node in enumerate(best_warmup_nodes) if i not in drop_idxs]
        best_warmup_ids = [node.id for node in best_warmup_nodes]

        # 给这些节点赋值
        for node in warmup_nodes:
            perf = node.perf
            reward = 1.0 if node.id in best_warmup_ids else 0.0
            self.backpropagation(node, perf=perf, reward=reward)

        print(f"Warmup nodes reset completed. Only best nodes {best_warmup_ids} retained reward values.")

        self.already_reset_warmup = True
        self.visualize("after_warmup_reset")


    def selection(self) -> 'Node':
        """选择阶段：选择最优子节点直到叶子节点(未被完全探索过的节点)"""

        median_perf = np.percentile(self.global_scores, 10) if len(self.global_scores) > 0 else -np.inf
        if np.isinf(median_perf) or np.isnan(median_perf):
            sorted_global_scores = sorted(self.global_scores)
            for score in sorted_global_scores:
                if np.isfinite(score):
                    median_perf = score
                    break
        # 找到最优和50分位数
        best_perf = self.tree.best_perf

        current_node = self.tree

        while current_node.children:
            if not current_node.is_fully_expanded():
                break
            current_node = current_node.best_child(self.exploration_weight, low_perf=median_perf, high_perf=best_perf)

        # 如果不是在HPORootNode停下，先看一下当前算法的最优节点有没有探索完
        if self.already_reset_warmup and isinstance(current_node, HPONode):
            algo = current_node.get_algorithm()
            best_nodes = self.best_nodes[algo]
            for node in best_nodes:
                if not node.is_fully_expanded():
                    print("Switching to best node", node.id, "of algorithm", algo, "for expansion.")
                    return node

        return current_node

    def warmup_selection(self, algorithm) -> 'HPORootNode':
        """warmup选择阶段：选择指定算法的根节点"""

        if self.tree.children[algorithm] is None:
            print("Creating new HPORootNode for algorithm:", algorithm)
            self.tree.children[algorithm] = HPORootNode(algorithm=algorithm, parent=self.tree, visit_mode=self.visit_mode, reward_mode=self.reward_mode)

        return self.tree.children[algorithm]

    def expansion(self, node: Node, max_depth=4) -> 'ExpandStrategy':
        """扩展阶段：扩展叶子节点"""

        # 往回最多回溯 max_depth 层，但是不能回溯到HPORootNode，返回一个链表
        node_path = {'parent': [], 'sibling': []}
        current_node = node
        for _ in range(max_depth):
            if current_node is None or isinstance(current_node, HPORootNode):
                break
            # 往前加
            node_path['parent'] = [current_node] + node_path['parent']
            current_node = current_node.parent

        # 把兄弟节点也加上
        for sibling_node in node.fetch_all_children():
            if sibling_node is not None:
                node_path['sibling'].append(sibling_node)

        return node.expand(), node_path

    def get_reward_scale(self, trail_count: int, is_global=True) -> float:
        """根据当前是第几次试验，返回奖励的缩放因子"""

        # 越后期，有提升的奖励越大，最小值为1，最大值为2
        def custom_function(x, k, x0, L):
            # 使用逻辑函数来实现缓升后急升的效果
            # 调整参数以满足给定条件
            # L = 2  # 最终值
            # k = 0.1  # 控制曲线陡峭程度
            # x0 = 75  # 曲线的中心点
            y = 1 + (L - 1) / (1 + np.exp(-k * (x - x0)))
            return y

        if is_global:
            func_score = custom_function(trail_count, k=0.1, x0=75, L=2)
        else:
            func_score = custom_function(trail_count, k=0.25, x0=25, L=1.5)

        # warmup阶段，禁用！
        func_score *= (self.already_reset_warmup is True)
        if func_score == 0:
            print("Warning: Reward scale is zero during warmup phase.")

        return func_score


    def check_reset_warmup(self, expand_strategy):
        # 检查是否需要重置warmup节点
        if self.already_reset_warmup:
            return

        # 第一次出现非warmup或者bo_random_search节点，说明warmup阶段结束
        if expand_strategy.strategy not in [WARMUP, BAYESIANRANDOM]:
            self.reset_warmup_nodes()

        # # 每个算法都被探索了10次以上，说明warmup阶段结束
        # need_reset = True

        # for algo in self.algo_scores:
        #     if len(self.algo_scores[algo]) < self.warmup_num:
        #         need_reset = False
        #         break

        # if need_reset:
        #     self.reset_warmup_nodes()


    def update_results(self, config: Configuration, perf: float, action: str, reflection:str, expand_strategy: ExpandStrategy) -> 'HPONode':
        """更新搜索树, 添加新的HPO节点"""

        # perf: 新配置的表现， config: 新配置， action: 采取的动作（大模型的thought或者bo采样），expand_strategy: 扩展策略

        selected_algo = config.get("algorithm", "unknown_algorithm")
        assert selected_algo == expand_strategy.selected_algo, "Selected algorithm mismatch."

        parent_id = expand_strategy.parent_id
        parent = Node.node_dict[parent_id]
        strategy = expand_strategy.strategy
        if strategy in [WARMUP, BAYESIANRANDOM]:
            assert isinstance(parent, HPORootNode), "Parent must be HPORootNode for warmup strategy or Bayesian random search."
        elif strategy in [BAYESIANLOCAL, LLMEXPLORATION, LLMEXPLOITATION]:
            assert isinstance(parent, HPONode), "Parent must be HPONode for Bayesian local search strategy or LLM strategies."
        else:
            raise ValueError("Invalid strategy type.")

        # 检查是否需要重置warmup节点
        self.check_reset_warmup(expand_strategy)

        new_hpo_node = HPONode(config=config, perf=perf, action=action, reflection=reflection, parent=parent, visit_mode=self.visit_mode, reward_mode=self.reward_mode)

        parent.add_child(new_hpo_node, strategy)

        # 计算奖励
        reward = 0
        tar_scores = self.algo_scores[selected_algo]
        if np.isfinite(self.tree.best_perf) and perf > self.tree.best_perf:
            reward = 1.0 * self.get_reward_scale(len(self.global_scores))

        if reward == 0 and len(tar_scores) > 0 and perf > tar_scores[-1]:
            reward = 0.1 * self.get_reward_scale(len(tar_scores), is_global=False)

        if reward == 0:
            # 局部提升奖励，是当前路径上最好的
            local_best = -np.inf
            current_node = parent
            while not (current_node is None or isinstance(current_node, HPORootNode)):
                local_best = max(local_best, current_node.perf)
                current_node = current_node.parent
            if np.isfinite(local_best) and perf > local_best:
                reward = 0.01 * self.get_reward_scale(len(tar_scores), is_global=False)


        if len(tar_scores) == 0 or perf > tar_scores[-1]:  # 当前算法发现最优点，优先拓展
            self.best_nodes[selected_algo].append(new_hpo_node)
        # 记录该算法的表现，按顺序，越大的放在越后面
        tar_scores.append(perf)
        self.global_scores.append(perf)
        tar_scores.sort()
        self.global_scores.sort()

        visit = 1
        # if strategy == BAYESIANRANDOM:
        #     visit = 0.5  # bo_random_search节点访问次数减半
        self.backpropagation(new_hpo_node, perf=perf, reward=reward, visit=visit)

        # 更新change path pareto
        cpp = self.cpp_dict[selected_algo]
        cp = ChangePath(parent_node=parent, current_node=new_hpo_node, cs=cpp.cs)
        cpp.add_path(cp)

        return new_hpo_node


    def backpropagation(self, node: Node, perf: float, reward: float, visit: int = 1):
        """回溯阶段：更新节点统计信息"""
        current_node = node

        while current_node is not None:
            current_node.visits += visit
            current_node.value += reward
            current_node.best_perf = max(current_node.best_perf, perf)
            current_node.all_perfs.append(perf)
            current_node = current_node.parent
            # reward *= 0.9  # 衰减因子


    # 用graphviz画出树结构
    def visualize(self, appendix: str = ""):
        mcts_dir = os.path.join(self.output_dir, "mcts_tree")
        if not os.path.exists(mcts_dir):
            os.makedirs(mcts_dir)

        appendix = f"_{appendix}" if appendix else ""
        filename = os.path.join(mcts_dir, f"{int(time.time())}_mcts_tree{appendix}")
        from graphviz import Digraph

        dot = Digraph(comment='MCTS Tree')

        def add_nodes_edges(node: Node, edge_str: str = ""):
            if isinstance(node, CASHNode):
                label = f"ID:{node.id}\nCASHNode\nBestPerf:{node.best_perf:.6e}, NodeNum:{len(Node.node_dict)}\nVisits:{node.visits}, Value:{node.value:.2f}"
            elif isinstance(node, HPORootNode):
                label = f"ID:{node.id}\nHPORootNode\nAlgo:{node.algorithm}, BestPerf:{node.best_perf:.6e}\nVisits:{node.visits}, Value:{node.value:.2f}"
            elif isinstance(node, HPONode):
                label = f"ID:{node.id}\nPerf:{node.perf:.6e}, BestPerf:{node.best_perf:.6e}\nVisits:{node.visits}, Value:{node.value:.2f}"
            dot.node(str(node.id), label)

            if node.parent is not None:
                dot.edge(str(node.parent.id), str(node.id), label=edge_str)

            if isinstance(node, CASHNode):
                for child in node.children.values():
                    if child is not None:
                        add_nodes_edges(child, edge_str=f"sel {child.algorithm}")
            elif isinstance(node, HPORootNode):
                for child in node.children["warmup"]:
                    add_nodes_edges(child, edge_str=WARMUP)
                for child in node.children["bo_random_search"]:
                    # value大于0或者有子节点
                    if child.value > 0 or not child.is_leaf_node():
                        add_nodes_edges(child, edge_str=BAYESIANRANDOM)
            elif isinstance(node, HPONode):
                if node.children['exploration'] is not None:
                    add_nodes_edges(node.children['exploration'], edge_str=LLMEXPLORATION)
                if node.children['exploitation'] is not None:
                    add_nodes_edges(node.children['exploitation'], edge_str=LLMEXPLOITATION)
                for child in node.children['bo']:
                    if child.value > 0 or not child.is_leaf_node():
                        add_nodes_edges(child, edge_str=BAYESIANLOCAL)

        add_nodes_edges(self.tree)
        dot.render(filename, format='png', cleanup=True)




# 一个改变的路径，从配置A到配置B

class ChangePath:

    def __init__(self, parent_node: Node, current_node: Node, cs: ConfigurationSpace):

        self.parent_id = None
        self.parent_config = None
        self.parent_perf = None
        if isinstance(parent_node, HPONode):
            tmp_config = Configuration(cs, values=deepcopy(parent_node.config.get_dictionary()))
            self.parent_id = parent_node.id
            self.parent_config = tmp_config
            self.parent_perf = parent_node.perf

        assert isinstance(current_node, HPONode), "Current node must be HPONode."
        tmp_config = Configuration(cs, values=deepcopy(current_node.config.get_dictionary()))
        self.current_id = current_node.id
        self.current_config = tmp_config
        self.current_perf = current_node.perf

        self._sign = -1 if KnowledgeDatabase.instance().lower_is_better else 1


        self.thought_process = current_node.action
        self.reflection = current_node.reflection

        # 自动生成性能变化描述并追加到reflection中
        perf_change_str = "Performance unchanged."
        if self.parent_perf is None:
            perf_change_str = f"Initial performance: {self.current_perf * self._sign}."
        else:
            if self.current_perf > self.parent_perf:
                perf_change_str = f"Performance improved from {self.parent_perf * self._sign} to {self.current_perf * self._sign}."
            elif self.current_perf < self.parent_perf:
                perf_change_str = f"Performance declined from {self.parent_perf * self._sign} to {self.current_perf * self._sign}."

        self.reflection += " " + perf_change_str

        self.config_sim = -1 # 和当前basic配置的相似度，暂时不计算

    def assign_similarity(self, sim: float):
        self.config_sim = sim

    def __str__(self):

        # title_str = f"  --- Node {self.parent_id} to Node {self.current_id} ---" if self.parent_id is not None else f"  --- Initial Node {self.current_id} ---"
        title_str = f"  --- Node {self.current_id} ---"
        parent_config_str = json.dumps(KnowledgeDatabase.instance()._cfg_to_dict(self.parent_config)) if self.parent_config else 'N/A'
        parent_perf_str = f"{self.parent_perf * self._sign}" if self.parent_perf is not None else 'N/A'
        result_config_str = json.dumps(KnowledgeDatabase.instance()._cfg_to_dict(self.current_config))
        entry = [
            f"  {title_str}",
            f"  Configuration: {result_config_str}",
            f"  Performance: {self.current_perf * self._sign}",
            f"  Reflection: {self.reflection}"
        ]
        # f"  Parent Config: {parent_config_str}",
        # f"  Parent Performance: {parent_perf_str}",
        # f"  Action Taken from Parent: {self.thought_process}",

        return "\n".join(entry)

    # 判断是否相等
    def __eq__(self, other):
        if not isinstance(other, ChangePath):
            return False
        return self.parent_id == other.parent_id and self.current_id == other.current_id


class ChangePathPareto:

    def __init__(self, cs):
        self.improve_paths = []  # List of ChangePath objects where performance improved
        self.decline_paths = []  # List of ChangePath objects where performance declined
        self.all_paths = []  # List of ChangePath objects where performance was unchanged

        self.improve_pareto = []
        self.decline_pareto = []

        self.gp_kernel = None  # Placeholder for GP kernel if needed in future
        self.cs = cs  # Configuration space for similarity calculations

    def add_path(self, change_path: ChangePath):
        if change_path.parent_perf is None:
            self.improve_paths.append(change_path)
            self.decline_paths.append(change_path)

        elif change_path.current_perf > change_path.parent_perf:
            self.improve_paths.append(change_path)
        elif change_path.current_perf < change_path.parent_perf:
            self.decline_paths.append(change_path)

        self.all_paths.append(change_path)

        # 排序
        self.improve_paths.sort(key=lambda x: x.current_perf, reverse=True)
        self.decline_paths.sort(key=lambda x: x.current_perf)
        self.all_paths.sort(key=lambda x: x.current_perf, reverse=True)

        print(f"Added ChangePath from Node {change_path.parent_id} to Node {change_path.current_id}. Current Perf: {change_path.current_perf}, Parent Perf: {change_path.parent_perf}")


    def is_dominated(self, candidate, others: List[ChangePath], direction = 1):
        """检查 candidate 是否被 others 中的任意一个支配"""
        for other in others:
            if self.is_dominated_by(candidate, other, direction):
                return True
        return False

    def is_dominated_by(self, candidate: ChangePath, other: ChangePath, direction = 1):
        """检查 candidate 是否被 other 支配"""
        if direction == 1:
            candidate = [candidate.current_perf, candidate.config_sim]
            other = [other.current_perf, other.config_sim]
        else:
            candidate = [ -candidate.current_perf, candidate.config_sim]
            other = [ -other.current_perf, other.config_sim]

        return all(o >= c for o, c in zip(other, candidate)) and any(o > c for o, c in zip(other, candidate))

    def update_pareto(self, exclude_ids: List[int] = []):

        # 排除current_id在exclude_ids中的路径
        improve_paths = [path for path in self.improve_paths if path.current_id not in exclude_ids]
        all_paths = [path for path in self.all_paths if path.current_id not in exclude_ids]
        """更新帕累托前沿"""
        imp_num = len(all_paths) // 10
        imp_num = max(3, min(imp_num, 6))

        self.improve_pareto = []
        if len(improve_paths) <= imp_num:
            self.improve_pareto = improve_paths.copy()
        else:
            for path in improve_paths:
                if not self.is_dominated(path, improve_paths, direction=1):
                    self.improve_pareto.append(path)
        for best_path in improve_paths[:3]:
            if best_path not in self.improve_pareto:
                self.improve_pareto.append(best_path)
        improve_paths = improve_paths[3:]

        # 最多保留5个，按照perf排序
        if len(self.improve_pareto) < imp_num:
            # 补齐
            count = 0
            for path in improve_paths:
                count += 1
                if path not in self.improve_pareto:
                    self.improve_pareto.append(path)
                if len(self.improve_pareto) >= imp_num:
                    break
            improve_paths = improve_paths[count:]

        self.improve_pareto.sort(key=lambda x: x.current_perf, reverse=True)
        self.improve_pareto = self.improve_pareto[:imp_num]



        self.decline_pareto = []


        print(f"Updated Pareto fronts: {len(self.improve_pareto)} improving paths, {len(self.decline_pareto)} declining paths.")

    def update_gp_kernel(self, kernel):
        """更新GP核函数"""
        self.gp_kernel = kernel
        print("Updated GP kernel for similarity calculations.")

    def update_similarities(self, basic_config: Configuration):
        """计算所有路径相对于 basic_config 的相似度"""

        basic_config = Configuration(self.cs, values=deepcopy(basic_config.get_dictionary()))
        basic_config = convert_configurations_to_array([basic_config])

        configurations = []
        valid_idxs = []

        for i, path in enumerate(self.improve_paths):
            if path.parent_config is not None:
                configurations.append(path.parent_config)
                valid_idxs.append(i)

        if len(configurations) > 0:
            
            config_array = convert_configurations_to_array(configurations)
            sim_matrix = self.gp_kernel(basic_config, config_array)[0]

            for idx, sim in zip(valid_idxs, sim_matrix):
                self.improve_paths[idx].assign_similarity(sim)


        configurations = []
        valid_idxs = []

        for i, path in enumerate(self.decline_paths):
            if path.parent_config is not None:
                configurations.append(path.parent_config)
                valid_idxs.append(i)

        if len(configurations) > 0:
            config_array = convert_configurations_to_array(configurations)
            sim_matrix = self.gp_kernel(basic_config, config_array)[0]

            for idx, sim in zip(valid_idxs, sim_matrix):
                self.decline_paths[idx].assign_similarity(sim)

        print("Updated similarities for ChangePaths.")

