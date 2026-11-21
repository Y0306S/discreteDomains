import numpy as np
import shap
from lightgbm import LGBMRegressor
from openbox import History
from openai import OpenAI
import re
import os
import json
import numpy as np
from copy import deepcopy
import pickle
from ConfigSpace.hyperparameters import Constant, UnParametrizedHyperparameter, CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
from mindware.utils.logging_utils import get_logger
from ConfigSpace import ConfigurationSpace, Configuration
from mindware.components.optimizers.utils import MyConfiguration
from typing import Any, Dict, List, Tuple
from collections import defaultdict
from mindware.components.metrics.metric import get_metric
from mindware.modules.configs import LLMCONFIGS, AVAILABLE_LLMs
from mindware.third_party.opro.utils import _constraints_text_for_cs, _build_dataset_summary_text
from mindware.rag.utils.prompt import get_user_requirement_summary
from mindware.knowledge_db.metadata_manager import write_pkl_file, read_pkl_file
from mindware.components.utils.constants import CLS_TASKS, RGS_TASKS

"""
    cls/rgs_meta_dataset_embedding.pkl: {dataset_id: [embedding_vector]}
    cls/rgs_meta_algo2perf.pkl: {metric: {dataset: {algorithm: perf}}}
    cls/rgs_similarity.pkl: {metric: {dataset1: {dataset2: sim}}}
    cls/rgs_meta_pred_cache: {metric: {algorithm: {dataset: pred_vector}}}

    cls/rgs_meta_dataset_shapknowledge.pkl: {metric: {algorithm: {dataset: {'llm_shap_experimence':..., 'llm_shap_overall':..., 'llm_shap_response':...}}}}

"""

class KnowledgeDatabase:

    _instance = None

    @classmethod
    def instance(cls, *args, **kwargs):
        # 只维护一个
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)
        else:
            # 已经有了，就看include_algorithms有没有变化，变化了就重新创建
            include_algorithms = kwargs.get('include_algorithms', None)
            if include_algorithms is not None and cls._instance.include_algorithms != include_algorithms:
                cls._instance = cls(*args, **kwargs)
            else:
                task_id = kwargs.get('task_id', None)
                if task_id is not None and cls._instance.task_id != task_id:
                    cls._instance = cls(*args, **kwargs)

        return cls._instance

    def __init__(
        self, task_type, metric, task_id, datanode, config_space=None, include_algorithms=None, 
        original_data_dir="./history_data", rag_data_dir='./rag_data',
        llm = LLMCONFIGS.kd_chat_engine, **kwargs
    ):

        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)
        self.task_id = task_id
        self.metric = metric
        tmp_metric = get_metric(metric)

        self.lower_is_better = tmp_metric._sign == -1 # Mindware会把所有metric都转化为越大越好，所以如果_sign==-1，说明原始metric是越小越好
        self.task_type = task_type


        # from mindware.knowledge_db.metadata_manager import MetaDataManager
        # self.metadata_manager = MetaDataManager(task_type=task_type, metric=metric, meta_dir=None, original_data_dir=original_data_dir, exclude_datasets=[], include_algorithms=include_algorithms)
        # self.meta_dir = self.metadata_manager.meta_dir
        # self.task_prefix = self.metadata_manager.task_prefix
        # self.include_algorithms = include_algorithms if include_algorithms is not None else self.metadata_manager.builtin_algorithms
        # 获得目标数据集的metafeature
        # self.target_metafeature = self.metadata_manager.get_meta_features(datanode)

        builtin_loc = os.path.dirname(__file__)
        builtin_loc = os.path.join(builtin_loc, 'meta_resource')
        self.meta_dir = builtin_loc
        if task_type in CLS_TASKS:
            self.task_prefix = 'cls'
        elif task_type in RGS_TASKS:
            self.task_prefix = 'rgs'
        else:
            raise ValueError(f"Invalid task type: {task_type}")
        self.include_algorithms = include_algorithms


        self.llm_model = AVAILABLE_LLMs[llm]["model"]
        self.client = OpenAI(
            base_url=AVAILABLE_LLMs[llm]["base_url"],
            api_key=AVAILABLE_LLMs[llm]["api_key"],
        )

        self.config_space = config_space


        # history for BO
        self.bo_history = History(task_id=self.task_id, config_space=self.config_space)

        self.bo_history_mab = {algo: None for algo in self.include_algorithms}

        for algo in self.include_algorithms:
            cs = ConfigurationSpace()
            cs.add_hyperparameter(Constant('algorithm', algo))
            # Add active hyperparameters
            hps = self.config_space.get_hyperparameters()
            for hp in hps:
                if hp.name.split(':')[0] == algo:
                    cs.add_hyperparameter(hp)
            # Add active conditions
            conds = self.config_space.get_conditions()
            for cond in conds:
                try:
                    cs.add_condition(cond)
                except:
                    pass
            # Add active forbidden clauses
            forbids = self.config_space.get_forbiddens()
            for forbid in forbids:
                try:
                    cs.add_forbidden_clause(forbid)
                except:
                    pass

            self.bo_history_mab[algo] = History(task_id=self.task_id, config_space=cs)

        # history for LLM
        def history_entry_factory():
            best_init = np.inf if self.lower_is_better else -np.inf
            worst_init = -np.inf if self.lower_is_better else np.inf
            return {
                'count': 0,
                'scores': [],
                'best_score': best_init,
                'worst_score': worst_init,
                'avg_score': 0.0,
                'thoughts': [] # 记录与该算法相关的思考
            }
        self.algo_selection_history: Dict[str, Dict] = defaultdict(history_entry_factory)

        self.param_tuning_history: Dict[str, List[Dict]] = defaultdict(list)


        self.perfs = [] # Higher is better
        self.configs = [] # Global configurations

        self.perfs_mab = {algo: [] for algo in self.include_algorithms} # Higher is better
        self.configs_mab = {algo: [] for algo in self.include_algorithms} # Configurations per algorithm

        self.historical_knowledge = None # 只维护一份
        self.similar_datasets = None # 只维护一份
        self.hyper_interactions = None # 只维护一份

        self.dataset_profile = kwargs.get('dataset_profile', None)
        self.rag_data_dir = rag_data_dir
        self.original_data_dir = original_data_dir
        self.optimizer_params = dict(kwargs or {})
        self.experience_bank_path = kwargs.get('experience_bank_path')
        self._experience_bank = None
        self._experience_bank_warned = False
        self._retrieve_hits = {}


    # 从历史任务中推荐模型顺序
    def recommand_models(self):

        from mindware.knowledge_db.algorithm_ranking.net_torch_ranker import RankNetAdvisor as RankNetAdvisor_W

        alad = RankNetAdvisor_W(metadata_manager=self.metadata_manager)  # exclude_datasets=[task_id]
        alad.fit()

        model_scores = alad.fetch_algorithm_score(target_feature=self.target_metafeature, algorithms=self.include_algorithms)

        # 排序: (model, score)
        idxs = np.argsort(model_scores)
        model_and_scores = [(self.include_algorithms[idx], model_scores[idx]) for idx in reversed(idxs)]
        print(model_and_scores)

        return model_and_scores

    def _task_type_label(self) -> str:
        if self.task_type in CLS_TASKS:
            return "classification"
        if self.task_type in RGS_TASKS:
            return "regression"
        return str(self.task_type)

    def _resolve_experience_bank_path(self):
        from pathlib import Path as _Path

        raw = self.experience_bank_path or (self.optimizer_params or {}).get("experience_bank_path")
        if raw:
            p = _Path(raw)
            if p.is_file():
                return p
        for cand in (
            _Path("history_data/experience_bank_v2.jsonl"),
            _Path(self.original_data_dir or ".") / "experience_bank_v2.jsonl",
            _Path(__file__).resolve().parents[2] / "history_data" / "experience_bank_v2.jsonl",
        ):
            if cand.is_file():
                return cand
        return _Path("history_data/experience_bank_v2.jsonl")

    def _get_experience_bank(self):
        if self._experience_bank is not None:
            return self._experience_bank
        from pathlib import Path as _Path
        from mindware.components.experience.retrieve import ExperienceBank

        path = self._resolve_experience_bank_path()
        if not _Path(path).is_file():
            if not self._experience_bank_warned:
                self.logger.warning("experience bank not found: %s", path)
                self._experience_bank_warned = True
            return None
        try:
            self._experience_bank = ExperienceBank.load(path)
            self.logger.info("Loaded experience bank %s (%s rows)", path, len(self._experience_bank))
        except Exception as exc:
            if not self._experience_bank_warned:
                self.logger.warning("failed to load experience bank %s: %s", path, exc)
                self._experience_bank_warned = True
            return None
        return self._experience_bank

    def _local_successes(self, algorithm: str):
        configs = list(self.configs_mab.get(algorithm) or [])
        perfs = list(self.perfs_mab.get(algorithm) or [])
        if not configs:
            configs = list(self.configs or [])
            perfs = list(self.perfs or [])
        out = []
        for i, (cfg, perf) in enumerate(zip(configs, perfs), 1):
            try:
                cfg_dict = self._cfg_to_dict(cfg) if not isinstance(cfg, dict) else dict(cfg)
            except Exception:
                cfg_dict = dict(cfg) if isinstance(cfg, dict) else {}
            try:
                score = float(perf)
            except (TypeError, ValueError):
                continue
            out.append({"trial": i, "config": cfg_dict, "score": score})
        return out

    def retrieve_similar_experiences(self, algorithm=None, k=3, rerank_after=24):
        """Cascade retrieve for one algorithm (or all include_algorithms)."""
        from mindware.components.experience.retrieve import (
            attach_semantic_vectors,
            build_run_query,
            format_retrieved_block,
            retrieve_experiences,
            role_signs_from_trials,
            target_roles_from_config_space,
        )

        bank = self._get_experience_bank()
        if bank is None:
            return {}
        algos = [algorithm] if algorithm else list(self.include_algorithms or [])
        out = {}
        task_label = self._task_type_label()
        for algo in algos:
            successes = self._local_successes(algo)
            query = build_run_query(
                dataset_id=self.task_id,
                algorithm_id=algo,
                task_type=task_label,
                dataset_profile=self.dataset_profile,
                successes=successes,
                config_space=self.config_space,
            )
            attach_semantic_vectors(query, bank)
            packed = [
                {"trial": t.get("trial", i), "config": t["config"], "score": t["score"]}
                for i, t in enumerate(successes, 1)
            ]
            hits = retrieve_experiences(
                bank,
                query,
                k=int(k),
                rerank_after=int(rerank_after),
                target_trials=packed if packed else None,
            )
            try:
                from mindware.components.experience.surrogate_landscape import (
                    LandscapeCache,
                    default_cache_path,
                    spearman_sim,
                )

                prefix = "cls" if task_label.startswith("class") else "rgs"
                cache = LandscapeCache(default_cache_path(prefix))
                qvec = cache.pred(algo, str(self.task_id).lower())
                if qvec is not None:
                    for h in hits:
                        ds = str(((h.experience.get("problem") or {}).get("dataset_id") or "")).lower()
                        svec = cache.pred(algo, ds)
                        sim = spearman_sim(qvec, svec)
                        h.features["surrogate_landscape_sim"] = sim
                        if h.spearman is None and sim:
                            h.score = 0.7 * h.score + 0.3 * max(0.0, sim)
            except Exception:
                pass
            names = (
                target_roles_from_config_space(self.config_space, algo)
                if self.config_space is not None
                else {}
            )
            tgt_signs = role_signs_from_trials(packed, higher_is_better=True) if packed else {}
            block = format_retrieved_block(hits, names, target_role_signs=tgt_signs)
            out[algo] = {"hits": hits, "prompt_block": block, "query": query}
        self._last_retrieved = out
        return out

    # 从历史任务中推荐相似数据集
    def recommand_datasets(self, thr=0.6, da_thr=3):

        if self.similar_datasets is not None:
            return self.similar_datasets

        retrieved = self.retrieve_similar_experiences(k=max(1, int(da_thr)), rerank_after=24)
        recommended_datasets = {}
        self._retrieve_hits = {}
        for algo, payload in retrieved.items():
            hits = payload.get("hits") or []
            self._retrieve_hits[algo] = hits
            selected = []
            for h in hits:
                ds = str(((h.experience.get("problem") or {}).get("dataset_id") or ""))
                score = float(h.score)
                if h.spearman is not None:
                    score = float(h.spearman) if score == 0 else score
                if score >= thr or len(selected) < 1:
                    selected.append([ds, score])
                if len(selected) >= da_thr:
                    break
            recommended_datasets[algo] = selected

        self.similar_datasets = recommended_datasets
        return recommended_datasets


    def _parse_experimence(self, response_text: str):

        pattern = r'```json\n\s*((?:{.*?})|(?:\[.*?\]))\s*,?\n```'
        json_str_match = re.findall(pattern, response_text, re.DOTALL)
        if not json_str_match:
            return None

        results = [json.loads(tmp) for tmp in json_str_match]

        if len(results) == 1:
            return results[0]

        return results

        # # Assume `response` contains the output from the language model

        # # Use regular expressions to extract each section
        # trends_match = re.search(r"\(1\) \*\*Trends:\*\* (.*?)(?=\(2\))", response_text, re.DOTALL)
        # optimal_ranges_match = re.search(r"\(2\) \*\*Optimal ranges for exploitation:\*\* (.*?)(?=\(3\))", response_text, re.DOTALL)
        # bad_ranges_match = re.search(r"\(3\) \*\*Bad ranges to avoid:\*\* (.*?)(?=\(4\))", response_text, re.DOTALL)
        # cautious_ranges_match = re.search(r"\(4\) \*\*Cautious ranges for exploration:\*\* (.*)", response_text, re.DOTALL)

        # # Extract the matched groups
        # trends = trends_match.group(1).strip() if trends_match else None
        # optimal_ranges = optimal_ranges_match.group(1).strip() if optimal_ranges_match else None
        # bad_ranges = bad_ranges_match.group(1).strip() if bad_ranges_match else None
        # cautious_ranges = cautious_ranges_match.group(1).strip() if cautious_ranges_match else None

        # # Print or store the results
        # # print("Trends:", trends)
        # # print("Optimal Ranges:", optimal_ranges)
        # # print("Bad Ranges:", bad_ranges)
        # # print("Cautious Ranges:", cautious_ranges)

        # results = {
        #     "trends": trends,
        #     "optimal_ranges": optimal_ranges,
        #     "bad_ranges": bad_ranges,
        #     "cautious_ranges": cautious_ranges
        # }

        # return results


    def _shap_analysis(self, algorithm=None, dataset=None, hyper_interaction=None):


        meta_dataset_dir = os.path.join(self.meta_dir, 'meta_dataset_vec')
        if not os.path.exists(meta_dataset_dir):
            os.makedirs(meta_dataset_dir)
        save_path = os.path.join(meta_dataset_dir, '%s_meta_dataset_shapknowledge.pkl' % self.task_prefix)

        if os.path.exists(save_path):
            data = read_pkl_file(save_path)
        else:
            data = {}

        if self.metric not in data:
            data[self.metric] = {}

        if algorithm in data[self.metric] and dataset in data[self.metric][algorithm]:
            llm_shap_experimence = data[self.metric][algorithm][dataset]['llm_shap_experimence']
            llm_shap_overall = data[self.metric][algorithm][dataset]['llm_shap_overall']
            return llm_shap_experimence, llm_shap_overall


        history = self.metadata_manager.fetch_algorithm_runs_history(dataset, algorithm)
        if history is None or len(history) == 0:
            raise FileNotFoundError(f"No history for dataset: {dataset}, algorithm: {algorithm}!")

        X = history.get_config_array()
        configurations = history.configurations
        Y = - history.get_objectives()
        cs = history.get_config_space()

        if algorithm is None:
            algorithm = cs.get_hyperparameter('algorithm').default_value
        else:
            assert algorithm == cs.get_hyperparameter('algorithm').default_value

        if dataset is None:
            dataset = history["task_id"]
        else:
            assert dataset == history.task_id

        paras = cs.get_hyperparameter_names()

        lgbr = LGBMRegressor(n_jobs=1, random_state=1, min_child_samples=3)
        lgbr.fit(X, Y)
        explainer = shap.TreeExplainer(lgbr)
        shap_values = explainer.shap_values(X)
        feature_importance = np.mean(np.abs(shap_values), axis=0)


        if algorithm not in data[self.metric]:
            data[self.metric][algorithm] = {}

        llm_shap_experimence = {}
        llm_shap_overall = {}
        llm_shap_response = {}

        for i in range(1, len(paras)):
            para = paras[i]
            parameter = cs.get_hyperparameter(para)

            # 如果是Constant，就跳过
            if isinstance(parameter, (Constant, UnParametrizedHyperparameter)):
                continue

            self.logger.info(f"Analyzing SHAP for algorithm: {algorithm}, parameter: {para} on dataset: {dataset}")

            tmp_cs = ConfigurationSpace()
            tmp_cs.add_hyperparameter(deepcopy(parameter))
            # properties = _build_hparam_constraints(tmp_cs)[para]
            # p_type = properties.get("type")
            # if p_type == "categorical":
            #     para_line = f"- {para}: Categorical, Choices: {properties.get('choices')}"
            # elif p_type in ("int", "float"):
            #     log_scale = " (log scale)" if properties.get("log") else ""
            #     para_line = f"- {para}: {p_type.capitalize()}, Range: [{properties.get('lower')}, {properties.get('upper')}]" + log_scale
            para_line = _constraints_text_for_cs(tmp_cs)
            print(para_line)

            values = [configurations[j][para] for j in range(X.shape[0])]
            shaps = shap_values[:, i]
            # 按照values重新排序，从小到大
            sorted_idx = np.argsort(values)
            values = np.array(values)[sorted_idx]
            shaps = shaps[sorted_idx]

            value_shape_pair = [(values[j], shaps[j]) for j in range(len(values))]
            if isinstance(value_shape_pair[0][0], (str, np.str_)):
                data_points_str = "\n\t".join([f"({v[0]}, {v[1]:.2e})" for v in value_shape_pair])
            else:
                data_points_str = "\n\t".join([f"({v[0]:.2e}, {v[1]:.2e})" for v in value_shape_pair])

            # 给出输出区间的模板
            if isinstance(parameter, CategoricalHyperparameter):
                range_str = "[choice_1, choice_2, ...]"
            else:
                range_str = "[min_value, max_value]"

            output_format_block = f"""```json
        {{
        "Trends": "(Describe trends observed.)",
        "Optimal ranges for exploitation": {range_str},
        "Bad ranges to avoid": {range_str}
        "Cautious ranges for exploration": {range_str}
        }}
        ```"""

            system_prompt = "You are a machine learning expert. Analyze the relationship between hyperparameter values and their SHAP values based on the provided data. Provide tuning experience on how changes in the hyperparameter values impact model performance, as indicated by the SHAP values. Highlight any trends, patterns, or significant observations that can guide hyperparameter tuning for better model performance."


            prompt = f"""
        ### **OVERALL MISSION: Automated Machine Learning (AutoML)**
        Your ultimate goal is to solve a **Hyperparameter optimization** problem: finding the optimal hyperparameters for algorithm '{algorithm}' in the current dataset.

        ### **CURRENT TASK: High-Level Conclusion of Hyperparameters Contribution**

        **1. Directive:**
        As the Chief Strategist, your goal is to analyse the value contribution of parameter '{para}' to identify different value ranges.

        **2. Dataset Summary:**
        {_build_dataset_summary_text({'dataset_profile': self.dataset_profile})}

        **3. Parameter Information:**
        {para_line}

        **4. Available Intelligence:**
        I'm tuning a {algorithm} model on a specific dataset aiming for higher metric values. Based on historical data, I have calculated the SHAP value for each value of {para}. Larger SHAP values indicate a greater positive impact on the target metric, while smaller values indicate a negative impact.

            # The hyperparameter information:
            {str(parameter)}

            # Here is the data showing the relationship between hyperparameter values and their corresponding SHAP values:
            Data (value, SHAP value):
        {data_points_str}

        **5. Guiding Principles for Value Ranges Identification:**
        Use these principles to strategically identify value ranges of three types: optimal ranges for exploitation, bad ranges to avoid, and cautious ranges for exploration.
        - **Optimal ranges for exploitation**: Identify **a small value range** to focus on. 
            - These ranges should consistently yield **high positive SHAP values**, indicating a strong positive impact on model performance.
        - **Bad ranges to avoid**: Identify **a small value range** that should be avoided.
            - These ranges should consistently yield **negative SHAP values**, indicating a detrimental effect on model performance.
        - **Cautious ranges for exploration**, the value range may have the following characteristics:
            - The SHAP values fluctuate between positive and negative, indicating uncertainty in their impact on model performance.
            - The SHAP values are close to zero, indicating a negligible impact on model performance.
            - The value range is rarely explored in the provided data, leading to insufficient information about its impact on model performance.
        - **Each value range can be an interval or a set of discrete values**, depending on the parameter type. It is acceptable if some value ranges overlap or are empty.

        **6. Your Analysis Task:**
        Please analyze the data and provide tuning experience on how changes in this hyperparameter values impact model performance, as indicated by the SHAP values.
        In the future, I will use the tuning experience to guide hyperparameter tuning for better model performance.
        You must ensure that your analysis and recommendations strictly pertain to the provided parameter range in '2. Parameter Information'.

        **7. Required Output Format:**
        You must provide your reasoning and your final tuning experience in this format:

        **Observations and Thought:**
        (According to the Guiding Principles and Available Intelligence, explain your rationale. Highlight any trends, patterns, or significant observations that can be used to summarize tuning experimence.)

        **Tuning Experience:**
        You must provide your final tuning experience in a single JSON block. Do not add any text before or after it.
        {output_format_block}
            """

            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            response = response.choices[0].message.content.strip()

            llm_shap_experimence[para] = self._parse_experimence(response)
            llm_shap_overall[para] = feature_importance[i]
            llm_shap_response[para] = response

        # interaction_shap_values = explainer.shap_interaction_values(X)

        data[self.metric][algorithm][dataset] = {
            'llm_shap_experimence': llm_shap_experimence,
            'llm_shap_overall': llm_shap_overall,
            'llm_shap_response': response
        }


        write_pkl_file(save_path, data)

        return llm_shap_experimence, llm_shap_overall

    def _fill_shap_analysis(self, shap_experimence):

        """
        shap_experimence example:
        {
            para: {'Trends': , 'Optimal ranges for exploitation': , 'Bad ranges to avoid': , 'Cautious ranges for exploration': } 
            para2: {...}, ...
        }

        """

        def is_list_of_lists(obj):
            if isinstance(obj, list):
                return all(isinstance(item, list) for item in obj)
            return False

        def is_list_of_values(obj):
            if isinstance(obj, list):
                return all(not isinstance(item, list) for item in obj)
            return False

        # 将搜索空间中没有被三个区域覆盖的区域，都加到'Cautious ranges for exploration'中
        for para in shap_experimence:
            parameter = self.config_space.get_hyperparameter(para)
            if isinstance(parameter, CategoricalHyperparameter):
                all_choices = set(parameter.choices)
                covered_choices = set()
                for key in ['Optimal ranges for exploitation', 'Bad ranges to avoid']:
                    range_area = shap_experimence[para][key]
                    # 提取choice_1, choice_2, ...
                    covered_choices.update(range_area)
                cautious_choices = all_choices - covered_choices
                original_cautious = shap_experimence[para]['Cautious ranges for exploration']
                # 取并集
                cautious_choices = set(original_cautious).union(cautious_choices)
                shap_experimence[para]['Cautious ranges for exploration'] = list(cautious_choices)
            else:
                lower = parameter.lower
                upper = parameter.upper

                covered_intervals = []
                for key in ['Optimal ranges for exploitation', 'Bad ranges to avoid']:
                    range_area = shap_experimence[para][key]
                    if len(range_area) == 0: continue

                    # 要么是[[min, max], [min, max]]
                    if is_list_of_lists(range_area):
                        for interval in range_area:
                            if len(interval) == 2:
                                covered_intervals.append(interval)

                    # 要么是[min, max]
                    elif is_list_of_values(range_area):
                        if len(range_area) == 2:
                            covered_intervals.append(range_area)

                # 计算未覆盖的区间
                covered_intervals.sort()
                cautious_intervals = []
                current_lower = lower
                for interval in covered_intervals:
                    if interval[0] > current_lower:
                        cautious_intervals.append([current_lower, interval[0]])
                    current_lower = max(current_lower, interval[1])
                if current_lower < upper:
                    cautious_intervals.append([current_lower, upper])

                # 和原来的cautious区间合并，如果有区间有重叠，要合并
                original_cautious = shap_experimence[para]['Cautious ranges for exploration']
                if len(original_cautious) != 0:
                    if is_list_of_lists(original_cautious):
                        for interval in original_cautious:
                            if len(interval) == 2:
                                cautious_intervals.append(interval)
                    elif is_list_of_values(original_cautious):
                        if len(original_cautious) == 2:
                            cautious_intervals.append(original_cautious)

                cautious_intervals.sort()
                merged_cautious = []
                for interval in cautious_intervals:
                    if not merged_cautious or interval[0] > merged_cautious[-1][1]:
                        merged_cautious.append(interval)
                    else:
                        merged_cautious[-1] = [merged_cautious[-1][0], max(merged_cautious[-1][1], interval[1])]

                cautious_intervals = merged_cautious


                shap_experimence[para]['Cautious ranges for exploration'] = cautious_intervals


    def _bestconf_knowledge(self, algo_datasets):

        # 从知识库中获取指定算法和数据集的bestconf信息
        # algo_datasets: {algorithm: [(dataset1, sim1), (dataset2, sim2), ...], ...}

        bestconf_knowledge = {
            algorithm:
            {
                'Best Configurations': []
            }
            for algorithm in algo_datasets
        }


        for algorithm in algo_datasets:
            for dataset, sim in algo_datasets[algorithm]:
                history = self.metadata_manager.fetch_algorithm_runs_history(dataset, algorithm)
                if history is None or len(history) == 0:
                    self.logger.info(f"No history for dataset: {dataset}, algorithm: {algorithm}!")
                    print(f"No history for dataset: {dataset}, algorithm: {algorithm}!")
                    continue

                # 获取bestconf
                best_config = history.get_incumbents()
                if len(best_config) > 0:
                    conf = best_config[0].config
                    conf = Configuration(self.config_space, values=conf)
                    conf = MyConfiguration(conf)
                    bestconf_knowledge[algorithm]['Best Configurations'].append(conf)

        return bestconf_knowledge


    def _shap_knowledge(self, algo_datasets, hyper_interactions=None):

        # 从知识库中获取指定算法和数据集的shap和bestconf信息
        # algo_datasets: {algorithm: [(dataset1, sim1), (dataset2, sim2), ...], ...}
        # hyper_interactions: {algorithm: [ [para1, para2], [para3, para4, para5], ... ], ...} 需要一起调的超参列表

        shap_bestconf_knowledge = {
            algorithm:
            {
                'SHAP Analysis': [],
                'SHAP Overall Importance': []
            }
            for algorithm in algo_datasets
        }


        for algorithm in algo_datasets:
            for dataset, sim in algo_datasets[algorithm]:

                # 获取shap分析
                hyper_interaction = None
                if hyper_interactions is not None and algorithm in hyper_interactions:
                    hyper_interaction = hyper_interactions[algorithm]['Hyperparameter Tuned Togother']
                shap_experimence, shap_overall = self._shap_analysis(algorithm, dataset, hyper_interaction)

                # 把搜索空间剩下没被覆盖的都放到cautious里
                self._fill_shap_analysis(shap_experimence)

                shap_bestconf_knowledge[algorithm]['SHAP Analysis'].append(shap_experimence)
                shap_bestconf_knowledge[algorithm]['SHAP Overall Importance'].append(shap_overall)

    
        return shap_bestconf_knowledge


    def fetch_historical_analysis_knowledge(self):

        if self.historical_knowledge is not None:
            return self.historical_knowledge

        dataset_similarity = self.recommand_datasets(da_thr=3)
        retrieved = getattr(self, "_last_retrieved", None) or self.retrieve_similar_experiences(
            k=3, rerank_after=24
        )

        historical_knowledge = {}
        for algorithm in dataset_similarity:
            payload = retrieved.get(algorithm) or {}
            historical_knowledge[algorithm] = {
                "dataset_similarity": dataset_similarity[algorithm],
                "prompt_block": payload.get("prompt_block") or "",
            }

        self.historical_knowledge = historical_knowledge
        return historical_knowledge


    def fetch_rag_knowledge(self):

        meta_dataset_dir = os.path.join(self.meta_dir, 'meta_dataset_vec')
        if not os.path.exists(meta_dataset_dir):
            os.makedirs(meta_dataset_dir)
        save_path = os.path.join(meta_dataset_dir, '%s_meta_dataset_ragknowledge.pkl' % self.task_prefix)

        if os.path.exists(save_path):
            data = read_pkl_file(save_path)
        else:
            data = {}

        if self.metric not in data:
            data[self.metric] = {}

        if self.task_id not in data[self.metric]:
            data[self.metric][self.task_id] = {}

        cache_response = {}
        remain_response = []
        for model in self.include_algorithms:
            if model in data[self.metric][self.task_id]:
                cache_response[model] = data[self.metric][self.task_id][model]
            else:
                remain_response.append(model)


        user_requirement_summary_dict = {
            model: get_user_requirement_summary(self.task_id, self.dataset_profile, self.task_type, model=model)
            for model in remain_response
        }

        from mindware.rag.retriver import retrieve_decompose_knowledge

        response = retrieve_decompose_knowledge(
            task_type=self.task_type, user_requirement_summary_dict=user_requirement_summary_dict, 
            llm_model=self.llm_model, client=self.client, top_k=50, include_algorithms=remain_response, data_dir=self.rag_data_dir)

        data[self.metric][self.task_id].update(response)
        response.update(cache_response)

        if len(remain_response) > 0:
            write_pkl_file(save_path, data)

        parse_rag_knowledge = {}

        for algorithm in response:
            # best_confs, hyper_interaction = self._parse_experimence(response[algorithm])   # {algo: best_conf, ...} , [ [para1, para2], [para3, para4, para5], ... ]
            best_confs = self._parse_experimence(response[algorithm])
            # best_conf和hyper_interaction里面的参数有的没有以algorithm:开头，这里补全及检查是否在搜索空间内
            for idx, best_conf in enumerate(best_confs):
                new_conf = {}
                for para in best_conf:
                    if not para.startswith(f'{algorithm}:'):
                        full_para = f"{algorithm}:{para}"
                    else:
                        full_para = para

                    if full_para in self.config_space.get_hyperparameter_names():
                        new_conf[full_para] = best_conf[para]
                best_confs[idx] = self._dict_to_cfg(new_conf)

            # for sublist in hyper_interaction if hyper_interaction is not None else []:
            #     for i in range(len(sublist)):
            #         para = sublist[i]
            #         if not para.startswith(f'{algorithm}:'):
            #             full_para = f"{algorithm}:{para}"
            #         else:
            #             full_para = para

            #         if full_para in self.config_space.get_hyperparameter_names():
            #             sublist[i] = full_para
            #         else:
            #             # 删除该参数
            #             sublist[i] = None
            #     # 删除None
            #     sublist[:] = [para for para in sublist if para is not None]

            # # 删除只有一个或者空的子列表
            # hyper_interaction = [sublist for sublist in hyper_interaction if len(sublist) > 1]

            # # 合并hyper_interaction中有重叠的子列表
            # if hyper_interaction is not None:
            #     merged_hyper_interaction = []
            #     for sublist in hyper_interaction:
            #         found = False
            #         for merged in merged_hyper_interaction:
            #             if not set(sublist).isdisjoint(set(merged)):
            #                 merged.extend(sublist)
            #                 found = True
            #                 break
            #         if not found:
            #             merged_hyper_interaction.append(sublist)
            #     # 去重
            #     merged_hyper_interaction = [list(set(sublist)) for sublist in merged_hyper_interaction]
            #     hyper_interaction = merged_hyper_interaction
            parse_rag_knowledge[algorithm] = {
                'Best Configurations': best_confs,
                # 'Hyperparameter Tuned Togother': hyper_interaction
            }

        # 先不打开超参交互
        # if self.hyper_interactions is None:
        #     hyper_interactions = {}
        #     for algorithm in parse_rag_knowledge:
        #         hyper_interactions[algorithm] = parse_rag_knowledge[algorithm]['Hyperparameter Tuned Togother']
        #     self.hyper_interactions = hyper_interactions

        return parse_rag_knowledge


    def fetch_algo_cate_knowledge(self):

        return [[algo, None] for algo in self.include_algorithms]

#         meta_dataset_dir = os.path.join(self.meta_dir, 'meta_dataset_vec')
#         if not os.path.exists(meta_dataset_dir):
#             os.makedirs(meta_dataset_dir)
#         save_path = os.path.join(meta_dataset_dir, '%s_meta_dataset_algocateknowledge.pkl' % self.task_prefix)

#         if os.path.exists(save_path):
#             with open(save_path, 'rb') as f:
#                 data = pickle.load(f)
#         else:
#             data = {}

#         if self.metric not in data:
#             data[self.metric] = {}


#         if self.task_id not in data[self.metric]:
#             data[self.metric][self.task_id] = {}

#         # 将self.include_algorithms转化为key
#         ia_key = '_'.join(sorted(self.include_algorithms))
#         if ia_key in data[self.metric][self.task_id]:
#             return data[self.metric][self.task_id][ia_key]


# #        model_scores = self.recommand_models() # [("random_forest", 0.95), ("lightgbm", 0.90), ...],
# #         historical_summary = f"""
# # **Algorithm Order and Priority Score:**
# # {model_scores}
# #         """
#         user_requirement_summary = get_user_requirement_summary(self.task_id, self.dataset_profile, self.task_type, model=self.include_algorithms)
#         response = retrieve_algorithm_category_knowledge(
#             task_type=self.task_type, user_requirement_summary=user_requirement_summary, 
#             llm_model=self.llm_model, client=self.client, top_k=50, data_dir=self.rag_data_dir)  # historical_knowledge=historical_summary, 

#         algo_cate = self._parse_experimence(response)  # [(algo1, priority1), (algo2, priority2), ...]

#         # 检查是否是否所有算法都被包含，检查是否有重叠
#         # all_algos = set()
#         # for cate in algo_cate:
#         #     for algo in algo_cate[cate]:
#         #         assert algo not in all_algos
#         #         assert algo in self.include_algorithms
#         #         all_algos.add(algo)

#         all_algos = [tmp[0] for tmp in algo_cate]
#         assert len(all_algos) == len(self.include_algorithms)
#         assert set(all_algos) == set(self.include_algorithms)


#         data[self.metric][self.task_id][ia_key] = algo_cate
#         with open(save_path, 'wb') as f:
#             pickle.dump(data, f)

#         return algo_cate


    def fetch_best_conf_knowledge(self):

        # 从RAG知识库中获取每个算法的调参经验：包括 最优配置 和 超参交互
        rag_knowledge = self.fetch_rag_knowledge()

        bestconf_knowledge = {}
        for algorithm in rag_knowledge:
            bestconf_knowledge[algorithm] = {
                'Best Configurations': rag_knowledge[algorithm]['Best Configurations']
            }



        # 从历史知识库中获取每个算法的调参经验：包括 dataset_similarity，Best Configurations，SHAP Analysis，SHAP Overall Importance
        # dataset_similarity = self.recommand_datasets(da_thr=1)  # {algorithm: [(dataset1, sim1), (dataset2, sim2), ...], ...}
        # bestconf_knowledge = self._bestconf_knowledge(dataset_similarity)  # {algorithm: {'Best Configurations': [...]}, ...}

        # for algorithm in dataset_similarity:
        #     bestconf_knowledge[algorithm].update({'dataset_similarity': dataset_similarity[algorithm]})
        #     bestconf_knowledge[algorithm]['Best Configurations'].extend(rag_knowledge[algorithm].get('Best Configurations', []))

        return bestconf_knowledge



    def update_history(self, obs, source, **extra_infos):

        BO_STRATEGY = "suggested by bayesian optimization"
        # 越大越好的 metric
        max_perf = - float(obs.objectives[0])
        # 原始metric，恢复所有正负处理
        perf = float(obs.objectives[0]) * (1 if self.lower_is_better else -1)


        selected_algo = obs.config['algorithm']
        ##### 全局配置和性能记录
        self.configs.append(obs.config)
        self.perfs.append(max_perf)

        self.configs_mab[selected_algo].append(obs.config)
        self.perfs_mab[selected_algo].append(max_perf)


        assert source in ['llm', 'bo', 'bo_mab', 'warmup'], f"Unknown source: {source}"
        # update bo history
        if not source.startswith('bo') or source == 'bo_mab': # 如果是bo，内部已经更新
            ori_config = obs.config
            new_obs = deepcopy(obs)
            new_config = Configuration(self.config_space, values=ori_config.get_dictionary())
            new_config = MyConfiguration(new_config, extra_info=ori_config.extra_info) # 直接传入引用，之后设置extra_info的时候，这里也会更新
            # 修改搜索空间
            new_obs.config = new_config
            self.bo_history.update_observation(new_obs)

        if not source.startswith('bo') or source == 'bo':
            ori_config = obs.config
            new_obs = deepcopy(obs)
            new_config = Configuration(self.bo_history_mab[selected_algo].config_space, values=ori_config.get_dictionary())
            new_config = MyConfiguration(new_config, extra_info=ori_config.extra_info)
            # 修改搜索空间
            new_obs.config = new_config
            self.bo_history_mab[selected_algo].update_observation(new_obs)


        # update llm history
        if source == 'llm':
            full_history_entry = { 
                "trail_id": extra_infos["trail_id"],
                "detailed_thought": extra_infos["detailed_thought"], 
                "thought_summary": extra_infos["thought_summary"], 
                "action_config": extra_infos["action_config"], 
                "observation_perf": extra_infos["observation_perf"], 
                "reflection_summary": extra_infos["reflection_summary"], 
            }
        elif source in ['bo', 'bo_mab']:
            full_history_entry = {
                "trail_id": extra_infos["trail_id"],
                "detailed_thought": BO_STRATEGY, 
                "thought_summary": BO_STRATEGY, 
                "action_config": self._cfg_to_dict(obs.config),
                "observation_perf": perf, # 原始的perf
                "reflection_summary": BO_STRATEGY,
            }
        elif source == 'warmup':
            full_history_entry = {
                "trail_id": extra_infos["trail_id"],
                "detailed_thought": "warmup configuration", 
                "thought_summary": "warmup configuration", 
                "action_config": self._cfg_to_dict(obs.config),
                "observation_perf": perf, # 原始的perf
                "reflection_summary": "warmup configuration",
            }
        else:
            raise ValueError(f"Unknown source: {source}")

        self.param_tuning_history[selected_algo].append(full_history_entry)


        algo_stats = self.algo_selection_history[selected_algo]
        algo_stats['count'] += 1
        algo_stats['scores'].append(perf)
        algo_stats['avg_score'] = np.mean(algo_stats['scores'])
        if self.lower_is_better:
            algo_stats['best_score'] = min(algo_stats['best_score'], perf)
            algo_stats['worst_score'] = max(algo_stats['worst_score'], perf)
        else:
            algo_stats['best_score'] = max(algo_stats['best_score'], perf)
            algo_stats['worst_score'] = min(algo_stats['worst_score'], perf)

        if source == 'llm':
            algo_stats['thoughts'].append({
                'trial_id': extra_infos["trail_id"],
                'detailed_thought': extra_infos["detailed_thought"], 'summary_thought': extra_infos["thought_summary"]
            })
        elif source in ['bo', 'bo_mab']:
            algo_stats['thoughts'].append({
                'trial_id': extra_infos["trail_id"],
                'detailed_thought': BO_STRATEGY,
                'summary_thought': BO_STRATEGY
            })
        else:
            algo_stats['thoughts'].append({
                'trial_id': extra_infos["trail_id"],
                'detailed_thought': "warmup configuration",
                'summary_thought': "warmup configuration"
            })


    def _cfg_to_dict(self, cfg: Configuration) -> Dict[str, Any]:
        # 去掉Constant类型的参数
        cfg_dict = {}

        for k in cfg:
            v = cfg[k]
            parameter = self.config_space.get_hyperparameter(k)
            if k != 'algorithm' and isinstance(parameter, (Constant, UnParametrizedHyperparameter)):
                continue
            cfg_dict[k] = v

        return cfg_dict


    def _dict_to_cfg(self, d: Dict[str, Any]) -> MyConfiguration:
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

        def _choices_for(hp):
            if hp is None:
                return []
            if isinstance(hp, (Constant, UnParametrizedHyperparameter)):
                return [getattr(hp, "value", getattr(hp, "default_value", None))]
            if hasattr(hp, "choices"):
                return list(hp.choices)
            if hasattr(hp, "sequence"):
                return list(hp.sequence)
            return [getattr(hp, "default_value", getattr(hp, "value", None))]

        hp_list = self.config_space.get_hyperparameters()
        hp_map = {hp.name: hp for hp in hp_list}

        # 1) Determine the active algorithm.
        algo_hp = hp_map.get('algorithm', None)
        algo_choices = _choices_for(algo_hp)
        algo = d.get('algorithm')
        if algo is None:
            # Infer algorithm from parameter prefixes (e.g., 'xgboost:n_estimators').
            prefixes = {k.split(':', 1)[0] for k in d.keys() if ':' in k}
            if prefixes:
                algo = list(prefixes)[0]
            elif algo_hp is not None:
                algo = getattr(algo_hp, 'default_value', getattr(algo_hp, 'value', None))
                if algo is None and algo_choices:
                    algo = algo_choices[0]
            else:
                raise ValueError("No 'algorithm' provided and cannot infer from keys.")
        if algo_hp is not None and algo not in algo_choices:
            raise ValueError(f"Invalid algorithm '{algo}'. Choices: {algo_choices}")

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

        return MyConfiguration(Configuration(self.config_space, values=values))
import numpy as np
import shap
from lightgbm import LGBMRegressor
from openbox import History
from openai import OpenAI
import re
import os
import json
import numpy as np
from copy import deepcopy
import pickle
from ConfigSpace.hyperparameters import Constant, UnParametrizedHyperparameter, CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
from mindware.utils.logging_utils import get_logger
from ConfigSpace import ConfigurationSpace, Configuration
from mindware.components.optimizers.utils import MyConfiguration
from typing import Any, Dict, List, Tuple
from collections import defaultdict
from mindware.components.metrics.metric import get_metric
from mindware.modules.configs import LLMCONFIGS, AVAILABLE_LLMs
from mindware.third_party.opro.utils import _constraints_text_for_cs, _build_dataset_summary_text
from mindware.rag.utils.prompt import get_user_requirement_summary
from mindware.knowledge_db.metadata_manager import write_pkl_file, read_pkl_file
from mindware.components.utils.constants import CLS_TASKS, RGS_TASKS

"""
    cls/rgs_meta_dataset_embedding.pkl: {dataset_id: [embedding_vector]}
    cls/rgs_meta_algo2perf.pkl: {metric: {dataset: {algorithm: perf}}}
    cls/rgs_similarity.pkl: {metric: {dataset1: {dataset2: sim}}}
    cls/rgs_meta_pred_cache: {metric: {algorithm: {dataset: pred_vector}}}

    cls/rgs_meta_dataset_shapknowledge.pkl: {metric: {algorithm: {dataset: {'llm_shap_experimence':..., 'llm_shap_overall':..., 'llm_shap_response':...}}}}

"""

class KnowledgeDatabase:

    _instance = None

    @classmethod
    def instance(cls, *args, **kwargs):
        # 只维护一个
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)
        else:
            # 已经有了，就看include_algorithms有没有变化，变化了就重新创建
            include_algorithms = kwargs.get('include_algorithms', None)
            if include_algorithms is not None and cls._instance.include_algorithms != include_algorithms:
                cls._instance = cls(*args, **kwargs)
            else:
                task_id = kwargs.get('task_id', None)
                if task_id is not None and cls._instance.task_id != task_id:
                    cls._instance = cls(*args, **kwargs)

        return cls._instance

    def __init__(
        self, task_type, metric, task_id, datanode, config_space=None, include_algorithms=None, 
        original_data_dir="./history_data", rag_data_dir='./rag_data',
        llm = LLMCONFIGS.kd_chat_engine, **kwargs
    ):

        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)
        self.task_id = task_id
        self.metric = metric
        tmp_metric = get_metric(metric)

        self.lower_is_better = tmp_metric._sign == -1 # Mindware会把所有metric都转化为越大越好，所以如果_sign==-1，说明原始metric是越小越好
        self.task_type = task_type


        # from mindware.knowledge_db.metadata_manager import MetaDataManager
        # self.metadata_manager = MetaDataManager(task_type=task_type, metric=metric, meta_dir=None, original_data_dir=original_data_dir, exclude_datasets=[], include_algorithms=include_algorithms)
        # self.meta_dir = self.metadata_manager.meta_dir
        # self.task_prefix = self.metadata_manager.task_prefix
        # self.include_algorithms = include_algorithms if include_algorithms is not None else self.metadata_manager.builtin_algorithms
        # 获得目标数据集的metafeature
        # self.target_metafeature = self.metadata_manager.get_meta_features(datanode)

        builtin_loc = os.path.dirname(__file__)
        builtin_loc = os.path.join(builtin_loc, 'meta_resource')
        self.meta_dir = builtin_loc
        if task_type in CLS_TASKS:
            self.task_prefix = 'cls'
        elif task_type in RGS_TASKS:
            self.task_prefix = 'rgs'
        else:
            raise ValueError(f"Invalid task type: {task_type}")
        self.include_algorithms = include_algorithms


        self.llm_model = AVAILABLE_LLMs[llm]["model"]
        self.client = OpenAI(
            base_url=AVAILABLE_LLMs[llm]["base_url"],
            api_key=AVAILABLE_LLMs[llm]["api_key"],
        )

        self.config_space = config_space


        # history for BO
        self.bo_history = History(task_id=self.task_id, config_space=self.config_space)

        self.bo_history_mab = {algo: None for algo in self.include_algorithms}

        for algo in self.include_algorithms:
            cs = ConfigurationSpace()
            cs.add_hyperparameter(Constant('algorithm', algo))
            # Add active hyperparameters
            hps = self.config_space.get_hyperparameters()
            for hp in hps:
                if hp.name.split(':')[0] == algo:
                    cs.add_hyperparameter(hp)
            # Add active conditions
            conds = self.config_space.get_conditions()
            for cond in conds:
                try:
                    cs.add_condition(cond)
                except:
                    pass
            # Add active forbidden clauses
            forbids = self.config_space.get_forbiddens()
            for forbid in forbids:
                try:
                    cs.add_forbidden_clause(forbid)
                except:
                    pass

            self.bo_history_mab[algo] = History(task_id=self.task_id, config_space=cs)

        # history for LLM
        def history_entry_factory():
            best_init = np.inf if self.lower_is_better else -np.inf
            worst_init = -np.inf if self.lower_is_better else np.inf
            return {
                'count': 0,
                'scores': [],
                'best_score': best_init,
                'worst_score': worst_init,
                'avg_score': 0.0,
                'thoughts': [] # 记录与该算法相关的思考
            }
        self.algo_selection_history: Dict[str, Dict] = defaultdict(history_entry_factory)

        self.param_tuning_history: Dict[str, List[Dict]] = defaultdict(list)


        self.perfs = [] # Higher is better
        self.configs = [] # Global configurations

        self.perfs_mab = {algo: [] for algo in self.include_algorithms} # Higher is better
        self.configs_mab = {algo: [] for algo in self.include_algorithms} # Configurations per algorithm

        self.historical_knowledge = None # 只维护一份
        self.similar_datasets = None # 只维护一份
        self.hyper_interactions = None # 只维护一份

        self.dataset_profile = kwargs.get('dataset_profile', None)
        self.rag_data_dir = rag_data_dir
        self.original_data_dir = original_data_dir
        self.optimizer_params = dict(kwargs or {})
        self.experience_bank_path = kwargs.get('experience_bank_path')
        self._experience_bank = None
        self._experience_bank_warned = False
        self._retrieve_hits = {}


    # 从历史任务中推荐模型顺序
    def recommand_models(self):

        from mindware.knowledge_db.algorithm_ranking.net_torch_ranker import RankNetAdvisor as RankNetAdvisor_W

        alad = RankNetAdvisor_W(metadata_manager=self.metadata_manager)  # exclude_datasets=[task_id]
        alad.fit()

        model_scores = alad.fetch_algorithm_score(target_feature=self.target_metafeature, algorithms=self.include_algorithms)

        # 排序: (model, score)
        idxs = np.argsort(model_scores)
        model_and_scores = [(self.include_algorithms[idx], model_scores[idx]) for idx in reversed(idxs)]
        print(model_and_scores)

        return model_and_scores

    def _task_type_label(self) -> str:
        if self.task_type in CLS_TASKS:
            return "classification"
        if self.task_type in RGS_TASKS:
            return "regression"
        return str(self.task_type)

    def _resolve_experience_bank_path(self):
        from pathlib import Path as _Path

        raw = self.experience_bank_path or (self.optimizer_params or {}).get("experience_bank_path")
        if raw:
            p = _Path(raw)
            if p.is_file():
                return p
        for cand in (
            _Path("history_data/experience_bank_v2.jsonl"),
            _Path(self.original_data_dir or ".") / "experience_bank_v2.jsonl",
            _Path(__file__).resolve().parents[2] / "history_data" / "experience_bank_v2.jsonl",
        ):
            if cand.is_file():
                return cand
        return _Path("history_data/experience_bank_v2.jsonl")

    def _get_experience_bank(self):
        if self._experience_bank is not None:
            return self._experience_bank
        from pathlib import Path as _Path
        from mindware.components.experience.retrieve import ExperienceBank

        path = self._resolve_experience_bank_path()
        if not _Path(path).is_file():
            if not self._experience_bank_warned:
                self.logger.warning("experience bank not found: %s", path)
                self._experience_bank_warned = True
            return None
        try:
            self._experience_bank = ExperienceBank.load(path)
            self.logger.info("Loaded experience bank %s (%s rows)", path, len(self._experience_bank))
        except Exception as exc:
            if not self._experience_bank_warned:
                self.logger.warning("failed to load experience bank %s: %s", path, exc)
                self._experience_bank_warned = True
            return None
        return self._experience_bank

    def _local_successes(self, algorithm: str):
        configs = list(self.configs_mab.get(algorithm) or [])
        perfs = list(self.perfs_mab.get(algorithm) or [])
        if not configs:
            configs = list(self.configs or [])
            perfs = list(self.perfs or [])
        out = []
        for i, (cfg, perf) in enumerate(zip(configs, perfs), 1):
            try:
                cfg_dict = self._cfg_to_dict(cfg) if not isinstance(cfg, dict) else dict(cfg)
            except Exception:
                cfg_dict = dict(cfg) if isinstance(cfg, dict) else {}
            try:
                score = float(perf)
            except (TypeError, ValueError):
                continue
            out.append({"trial": i, "config": cfg_dict, "score": score})
        return out

    def retrieve_similar_experiences(self, algorithm=None, k=3, rerank_after=24):
        """Cascade retrieve for one algorithm (or all include_algorithms)."""
        from mindware.components.experience.retrieve import (
            attach_semantic_vectors,
            build_run_query,
            format_retrieved_block,
            retrieve_experiences,
            role_signs_from_trials,
            target_roles_from_config_space,
        )

        bank = self._get_experience_bank()
        if bank is None:
            return {}
        algos = [algorithm] if algorithm else list(self.include_algorithms or [])
        out = {}
        task_label = self._task_type_label()
        for algo in algos:
            successes = self._local_successes(algo)
            query = build_run_query(
                dataset_id=self.task_id,
                algorithm_id=algo,
                task_type=task_label,
                dataset_profile=self.dataset_profile,
                successes=successes,
                config_space=self.config_space,
            )
            attach_semantic_vectors(query, bank)
            packed = [
                {"trial": t.get("trial", i), "config": t["config"], "score": t["score"]}
                for i, t in enumerate(successes, 1)
            ]
            hits = retrieve_experiences(
                bank,
                query,
                k=int(k),
                rerank_after=int(rerank_after),
                target_trials=packed if packed else None,
            )
            try:
                from mindware.components.experience.surrogate_landscape import (
                    LandscapeCache,
                    default_cache_path,
                    spearman_sim,
                )

                prefix = "cls" if task_label.startswith("class") else "rgs"
                cache = LandscapeCache(default_cache_path(prefix))
                qvec = cache.pred(algo, str(self.task_id).lower())
                if qvec is not None:
                    for h in hits:
                        ds = str(((h.experience.get("problem") or {}).get("dataset_id") or "")).lower()
                        svec = cache.pred(algo, ds)
                        sim = spearman_sim(qvec, svec)
                        h.features["surrogate_landscape_sim"] = sim
                        if h.spearman is None and sim:
                            h.score = 0.7 * h.score + 0.3 * max(0.0, sim)
            except Exception:
                pass
            names = (
                target_roles_from_config_space(self.config_space, algo)
                if self.config_space is not None
                else {}
            )
            tgt_signs = role_signs_from_trials(packed, higher_is_better=True) if packed else {}
            block = format_retrieved_block(hits, names, target_role_signs=tgt_signs)
            out[algo] = {"hits": hits, "prompt_block": block, "query": query}
        self._last_retrieved = out
        return out

    # 从历史任务中推荐相似数据集
    def recommand_datasets(self, thr=0.6, da_thr=3):

        if self.similar_datasets is not None:
            return self.similar_datasets

        retrieved = self.retrieve_similar_experiences(k=max(1, int(da_thr)), rerank_after=24)
        recommended_datasets = {}
        self._retrieve_hits = {}
        for algo, payload in retrieved.items():
            hits = payload.get("hits") or []
            self._retrieve_hits[algo] = hits
            selected = []
            for h in hits:
                ds = str(((h.experience.get("problem") or {}).get("dataset_id") or ""))
                score = float(h.score)
                if h.spearman is not None:
                    score = float(h.spearman) if score == 0 else score
                if score >= thr or len(selected) < 1:
                    selected.append([ds, score])
                if len(selected) >= da_thr:
                    break
            recommended_datasets[algo] = selected

        self.similar_datasets = recommended_datasets
        return recommended_datasets


    def _parse_experimence(self, response_text: str):

        pattern = r'```json\n\s*((?:{.*?})|(?:\[.*?\]))\s*,?\n```'
        json_str_match = re.findall(pattern, response_text, re.DOTALL)
        if not json_str_match:
            return None

        results = [json.loads(tmp) for tmp in json_str_match]

        if len(results) == 1:
            return results[0]

        return results

        # # Assume `response` contains the output from the language model

        # # Use regular expressions to extract each section
        # trends_match = re.search(r"\(1\) \*\*Trends:\*\* (.*?)(?=\(2\))", response_text, re.DOTALL)
        # optimal_ranges_match = re.search(r"\(2\) \*\*Optimal ranges for exploitation:\*\* (.*?)(?=\(3\))", response_text, re.DOTALL)
        # bad_ranges_match = re.search(r"\(3\) \*\*Bad ranges to avoid:\*\* (.*?)(?=\(4\))", response_text, re.DOTALL)
        # cautious_ranges_match = re.search(r"\(4\) \*\*Cautious ranges for exploration:\*\* (.*)", response_text, re.DOTALL)

        # # Extract the matched groups
        # trends = trends_match.group(1).strip() if trends_match else None
        # optimal_ranges = optimal_ranges_match.group(1).strip() if optimal_ranges_match else None
        # bad_ranges = bad_ranges_match.group(1).strip() if bad_ranges_match else None
        # cautious_ranges = cautious_ranges_match.group(1).strip() if cautious_ranges_match else None

        # # Print or store the results
        # # print("Trends:", trends)
        # # print("Optimal Ranges:", optimal_ranges)
        # # print("Bad Ranges:", bad_ranges)
        # # print("Cautious Ranges:", cautious_ranges)

        # results = {
        #     "trends": trends,
        #     "optimal_ranges": optimal_ranges,
        #     "bad_ranges": bad_ranges,
        #     "cautious_ranges": cautious_ranges
        # }

        # return results


    def _shap_analysis(self, algorithm=None, dataset=None, hyper_interaction=None):


        meta_dataset_dir = os.path.join(self.meta_dir, 'meta_dataset_vec')
        if not os.path.exists(meta_dataset_dir):
            os.makedirs(meta_dataset_dir)
        save_path = os.path.join(meta_dataset_dir, '%s_meta_dataset_shapknowledge.pkl' % self.task_prefix)

        if os.path.exists(save_path):
            data = read_pkl_file(save_path)
        else:
            data = {}

        if self.metric not in data:
            data[self.metric] = {}

        if algorithm in data[self.metric] and dataset in data[self.metric][algorithm]:
            llm_shap_experimence = data[self.metric][algorithm][dataset]['llm_shap_experimence']
            llm_shap_overall = data[self.metric][algorithm][dataset]['llm_shap_overall']
            return llm_shap_experimence, llm_shap_overall


        history = self.metadata_manager.fetch_algorithm_runs_history(dataset, algorithm)
        if history is None or len(history) == 0:
            raise FileNotFoundError(f"No history for dataset: {dataset}, algorithm: {algorithm}!")

        X = history.get_config_array()
        configurations = history.configurations
        Y = - history.get_objectives()
        cs = history.get_config_space()

        if algorithm is None:
            algorithm = cs.get_hyperparameter('algorithm').default_value
        else:
            assert algorithm == cs.get_hyperparameter('algorithm').default_value

        if dataset is None:
            dataset = history["task_id"]
        else:
            assert dataset == history.task_id

        paras = cs.get_hyperparameter_names()

        lgbr = LGBMRegressor(n_jobs=1, random_state=1, min_child_samples=3)
        lgbr.fit(X, Y)
        explainer = shap.TreeExplainer(lgbr)
        shap_values = explainer.shap_values(X)
        feature_importance = np.mean(np.abs(shap_values), axis=0)


        if algorithm not in data[self.metric]:
            data[self.metric][algorithm] = {}

        llm_shap_experimence = {}
        llm_shap_overall = {}
        llm_shap_response = {}

        for i in range(1, len(paras)):
            para = paras[i]
            parameter = cs.get_hyperparameter(para)

            # 如果是Constant，就跳过
            if isinstance(parameter, (Constant, UnParametrizedHyperparameter)):
                continue

            self.logger.info(f"Analyzing SHAP for algorithm: {algorithm}, parameter: {para} on dataset: {dataset}")

            tmp_cs = ConfigurationSpace()
            tmp_cs.add_hyperparameter(deepcopy(parameter))
            # properties = _build_hparam_constraints(tmp_cs)[para]
            # p_type = properties.get("type")
            # if p_type == "categorical":
            #     para_line = f"- {para}: Categorical, Choices: {properties.get('choices')}"
            # elif p_type in ("int", "float"):
            #     log_scale = " (log scale)" if properties.get("log") else ""
            #     para_line = f"- {para}: {p_type.capitalize()}, Range: [{properties.get('lower')}, {properties.get('upper')}]" + log_scale
            para_line = _constraints_text_for_cs(tmp_cs)
            print(para_line)

            values = [configurations[j][para] for j in range(X.shape[0])]
            shaps = shap_values[:, i]
            # 按照values重新排序，从小到大
            sorted_idx = np.argsort(values)
            values = np.array(values)[sorted_idx]
            shaps = shaps[sorted_idx]

            value_shape_pair = [(values[j], shaps[j]) for j in range(len(values))]
            if isinstance(value_shape_pair[0][0], (str, np.str_)):
                data_points_str = "\n\t".join([f"({v[0]}, {v[1]:.2e})" for v in value_shape_pair])
            else:
                data_points_str = "\n\t".join([f"({v[0]:.2e}, {v[1]:.2e})" for v in value_shape_pair])

            # 给出输出区间的模板
            if isinstance(parameter, CategoricalHyperparameter):
                range_str = "[choice_1, choice_2, ...]"
            else:
                range_str = "[min_value, max_value]"

            output_format_block = f"""```json
        {{
        "Trends": "(Describe trends observed.)",
        "Optimal ranges for exploitation": {range_str},
        "Bad ranges to avoid": {range_str}
        "Cautious ranges for exploration": {range_str}
        }}
        ```"""

            system_prompt = "You are a machine learning expert. Analyze the relationship between hyperparameter values and their SHAP values based on the provided data. Provide tuning experience on how changes in the hyperparameter values impact model performance, as indicated by the SHAP values. Highlight any trends, patterns, or significant observations that can guide hyperparameter tuning for better model performance."


            prompt = f"""
        ### **OVERALL MISSION: Automated Machine Learning (AutoML)**
        Your ultimate goal is to solve a **Hyperparameter optimization** problem: finding the optimal hyperparameters for algorithm '{algorithm}' in the current dataset.

        ### **CURRENT TASK: High-Level Conclusion of Hyperparameters Contribution**

        **1. Directive:**
        As the Chief Strategist, your goal is to analyse the value contribution of parameter '{para}' to identify different value ranges.

        **2. Dataset Summary:**
        {_build_dataset_summary_text({'dataset_profile': self.dataset_profile})}

        **3. Parameter Information:**
        {para_line}

        **4. Available Intelligence:**
        I'm tuning a {algorithm} model on a specific dataset aiming for higher metric values. Based on historical data, I have calculated the SHAP value for each value of {para}. Larger SHAP values indicate a greater positive impact on the target metric, while smaller values indicate a negative impact.

            # The hyperparameter information:
            {str(parameter)}

            # Here is the data showing the relationship between hyperparameter values and their corresponding SHAP values:
            Data (value, SHAP value):
        {data_points_str}

        **5. Guiding Principles for Value Ranges Identification:**
        Use these principles to strategically identify value ranges of three types: optimal ranges for exploitation, bad ranges to avoid, and cautious ranges for exploration.
        - **Optimal ranges for exploitation**: Identify **a small value range** to focus on. 
            - These ranges should consistently yield **high positive SHAP values**, indicating a strong positive impact on model performance.
        - **Bad ranges to avoid**: Identify **a small value range** that should be avoided.
            - These ranges should consistently yield **negative SHAP values**, indicating a detrimental effect on model performance.
        - **Cautious ranges for exploration**, the value range may have the following characteristics:
            - The SHAP values fluctuate between positive and negative, indicating uncertainty in their impact on model performance.
            - The SHAP values are close to zero, indicating a negligible impact on model performance.
            - The value range is rarely explored in the provided data, leading to insufficient information about its impact on model performance.
        - **Each value range can be an interval or a set of discrete values**, depending on the parameter type. It is acceptable if some value ranges overlap or are empty.

        **6. Your Analysis Task:**
        Please analyze the data and provide tuning experience on how changes in this hyperparameter values impact model performance, as indicated by the SHAP values.
        In the future, I will use the tuning experience to guide hyperparameter tuning for better model performance.
        You must ensure that your analysis and recommendations strictly pertain to the provided parameter range in '2. Parameter Information'.

        **7. Required Output Format:**
        You must provide your reasoning and your final tuning experience in this format:

        **Observations and Thought:**
        (According to the Guiding Principles and Available Intelligence, explain your rationale. Highlight any trends, patterns, or significant observations that can be used to summarize tuning experimence.)

        **Tuning Experience:**
        You must provide your final tuning experience in a single JSON block. Do not add any text before or after it.
        {output_format_block}
            """

            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            response = response.choices[0].message.content.strip()

            llm_shap_experimence[para] = self._parse_experimence(response)
            llm_shap_overall[para] = feature_importance[i]
            llm_shap_response[para] = response

        # interaction_shap_values = explainer.shap_interaction_values(X)

        data[self.metric][algorithm][dataset] = {
            'llm_shap_experimence': llm_shap_experimence,
            'llm_shap_overall': llm_shap_overall,
            'llm_shap_response': response
        }


        write_pkl_file(save_path, data)

        return llm_shap_experimence, llm_shap_overall

    def _fill_shap_analysis(self, shap_experimence):

        """
        shap_experimence example:
        {
            para: {'Trends': , 'Optimal ranges for exploitation': , 'Bad ranges to avoid': , 'Cautious ranges for exploration': } 
            para2: {...}, ...
        }

        """

        def is_list_of_lists(obj):
            if isinstance(obj, list):
                return all(isinstance(item, list) for item in obj)
            return False

        def is_list_of_values(obj):
            if isinstance(obj, list):
                return all(not isinstance(item, list) for item in obj)
            return False

        # 将搜索空间中没有被三个区域覆盖的区域，都加到'Cautious ranges for exploration'中
        for para in shap_experimence:
            parameter = self.config_space.get_hyperparameter(para)
            if isinstance(parameter, CategoricalHyperparameter):
                all_choices = set(parameter.choices)
                covered_choices = set()
                for key in ['Optimal ranges for exploitation', 'Bad ranges to avoid']:
                    range_area = shap_experimence[para][key]
                    # 提取choice_1, choice_2, ...
                    covered_choices.update(range_area)
                cautious_choices = all_choices - covered_choices
                original_cautious = shap_experimence[para]['Cautious ranges for exploration']
                # 取并集
                cautious_choices = set(original_cautious).union(cautious_choices)
                shap_experimence[para]['Cautious ranges for exploration'] = list(cautious_choices)
            else:
                lower = parameter.lower
                upper = parameter.upper

                covered_intervals = []
                for key in ['Optimal ranges for exploitation', 'Bad ranges to avoid']:
                    range_area = shap_experimence[para][key]
                    if len(range_area) == 0: continue

                    # 要么是[[min, max], [min, max]]
                    if is_list_of_lists(range_area):
                        for interval in range_area:
                            if len(interval) == 2:
                                covered_intervals.append(interval)

                    # 要么是[min, max]
                    elif is_list_of_values(range_area):
                        if len(range_area) == 2:
                            covered_intervals.append(range_area)

                # 计算未覆盖的区间
                covered_intervals.sort()
                cautious_intervals = []
                current_lower = lower
                for interval in covered_intervals:
                    if interval[0] > current_lower:
                        cautious_intervals.append([current_lower, interval[0]])
                    current_lower = max(current_lower, interval[1])
                if current_lower < upper:
                    cautious_intervals.append([current_lower, upper])

                # 和原来的cautious区间合并，如果有区间有重叠，要合并
                original_cautious = shap_experimence[para]['Cautious ranges for exploration']
                if len(original_cautious) != 0:
                    if is_list_of_lists(original_cautious):
                        for interval in original_cautious:
                            if len(interval) == 2:
                                cautious_intervals.append(interval)
                    elif is_list_of_values(original_cautious):
                        if len(original_cautious) == 2:
                            cautious_intervals.append(original_cautious)

                cautious_intervals.sort()
                merged_cautious = []
                for interval in cautious_intervals:
                    if not merged_cautious or interval[0] > merged_cautious[-1][1]:
                        merged_cautious.append(interval)
                    else:
                        merged_cautious[-1] = [merged_cautious[-1][0], max(merged_cautious[-1][1], interval[1])]

                cautious_intervals = merged_cautious


                shap_experimence[para]['Cautious ranges for exploration'] = cautious_intervals


    def _bestconf_knowledge(self, algo_datasets):

        # 从知识库中获取指定算法和数据集的bestconf信息
        # algo_datasets: {algorithm: [(dataset1, sim1), (dataset2, sim2), ...], ...}

        bestconf_knowledge = {
            algorithm:
            {
                'Best Configurations': []
            }
            for algorithm in algo_datasets
        }


        for algorithm in algo_datasets:
            for dataset, sim in algo_datasets[algorithm]:
                history = self.metadata_manager.fetch_algorithm_runs_history(dataset, algorithm)
                if history is None or len(history) == 0:
                    self.logger.info(f"No history for dataset: {dataset}, algorithm: {algorithm}!")
                    print(f"No history for dataset: {dataset}, algorithm: {algorithm}!")
                    continue

                # 获取bestconf
                best_config = history.get_incumbents()
                if len(best_config) > 0:
                    conf = best_config[0].config
                    conf = Configuration(self.config_space, values=conf)
                    conf = MyConfiguration(conf)
                    bestconf_knowledge[algorithm]['Best Configurations'].append(conf)

        return bestconf_knowledge


    def _shap_knowledge(self, algo_datasets, hyper_interactions=None):

        # 从知识库中获取指定算法和数据集的shap和bestconf信息
        # algo_datasets: {algorithm: [(dataset1, sim1), (dataset2, sim2), ...], ...}
        # hyper_interactions: {algorithm: [ [para1, para2], [para3, para4, para5], ... ], ...} 需要一起调的超参列表

        shap_bestconf_knowledge = {
            algorithm:
            {
                'SHAP Analysis': [],
                'SHAP Overall Importance': []
            }
            for algorithm in algo_datasets
        }


        for algorithm in algo_datasets:
            for dataset, sim in algo_datasets[algorithm]:

                # 获取shap分析
                hyper_interaction = None
                if hyper_interactions is not None and algorithm in hyper_interactions:
                    hyper_interaction = hyper_interactions[algorithm]['Hyperparameter Tuned Togother']
                shap_experimence, shap_overall = self._shap_analysis(algorithm, dataset, hyper_interaction)

                # 把搜索空间剩下没被覆盖的都放到cautious里
                self._fill_shap_analysis(shap_experimence)

                shap_bestconf_knowledge[algorithm]['SHAP Analysis'].append(shap_experimence)
                shap_bestconf_knowledge[algorithm]['SHAP Overall Importance'].append(shap_overall)

    
        return shap_bestconf_knowledge


    def fetch_historical_analysis_knowledge(self):

        if self.historical_knowledge is not None:
            return self.historical_knowledge

        dataset_similarity = self.recommand_datasets(da_thr=3)
        retrieved = getattr(self, "_last_retrieved", None) or self.retrieve_similar_experiences(
            k=3, rerank_after=24
        )

        historical_knowledge = {}
        for algorithm in dataset_similarity:
            payload = retrieved.get(algorithm) or {}
            historical_knowledge[algorithm] = {
                "dataset_similarity": dataset_similarity[algorithm],
                "prompt_block": payload.get("prompt_block") or "",
            }

        self.historical_knowledge = historical_knowledge
        return historical_knowledge


    def fetch_rag_knowledge(self):

        meta_dataset_dir = os.path.join(self.meta_dir, 'meta_dataset_vec')
        if not os.path.exists(meta_dataset_dir):
            os.makedirs(meta_dataset_dir)
        save_path = os.path.join(meta_dataset_dir, '%s_meta_dataset_ragknowledge.pkl' % self.task_prefix)

        if os.path.exists(save_path):
            data = read_pkl_file(save_path)
        else:
            data = {}

        if self.metric not in data:
            data[self.metric] = {}

        if self.task_id not in data[self.metric]:
            data[self.metric][self.task_id] = {}

        cache_response = {}
        remain_response = []
        for model in self.include_algorithms:
            if model in data[self.metric][self.task_id]:
                cache_response[model] = data[self.metric][self.task_id][model]
            else:
                remain_response.append(model)


        user_requirement_summary_dict = {
            model: get_user_requirement_summary(self.task_id, self.dataset_profile, self.task_type, model=model)
            for model in remain_response
        }

        from mindware.rag.retriver import retrieve_decompose_knowledge

        response = retrieve_decompose_knowledge(
            task_type=self.task_type, user_requirement_summary_dict=user_requirement_summary_dict, 
            llm_model=self.llm_model, client=self.client, top_k=50, include_algorithms=remain_response, data_dir=self.rag_data_dir)

        data[self.metric][self.task_id].update(response)
        response.update(cache_response)

        if len(remain_response) > 0:
            write_pkl_file(save_path, data)

        parse_rag_knowledge = {}

        for algorithm in response:
            # best_confs, hyper_interaction = self._parse_experimence(response[algorithm])   # {algo: best_conf, ...} , [ [para1, para2], [para3, para4, para5], ... ]
            best_confs = self._parse_experimence(response[algorithm])
            # best_conf和hyper_interaction里面的参数有的没有以algorithm:开头，这里补全及检查是否在搜索空间内
            for idx, best_conf in enumerate(best_confs):
                new_conf = {}
                for para in best_conf:
                    if not para.startswith(f'{algorithm}:'):
                        full_para = f"{algorithm}:{para}"
                    else:
                        full_para = para

                    if full_para in self.config_space.get_hyperparameter_names():
                        new_conf[full_para] = best_conf[para]
                best_confs[idx] = self._dict_to_cfg(new_conf)

            # for sublist in hyper_interaction if hyper_interaction is not None else []:
            #     for i in range(len(sublist)):
            #         para = sublist[i]
            #         if not para.startswith(f'{algorithm}:'):
            #             full_para = f"{algorithm}:{para}"
            #         else:
            #             full_para = para

            #         if full_para in self.config_space.get_hyperparameter_names():
            #             sublist[i] = full_para
            #         else:
            #             # 删除该参数
            #             sublist[i] = None
            #     # 删除None
            #     sublist[:] = [para for para in sublist if para is not None]

            # # 删除只有一个或者空的子列表
            # hyper_interaction = [sublist for sublist in hyper_interaction if len(sublist) > 1]

            # # 合并hyper_interaction中有重叠的子列表
            # if hyper_interaction is not None:
            #     merged_hyper_interaction = []
            #     for sublist in hyper_interaction:
            #         found = False
            #         for merged in merged_hyper_interaction:
            #             if not set(sublist).isdisjoint(set(merged)):
            #                 merged.extend(sublist)
            #                 found = True
            #                 break
            #         if not found:
            #             merged_hyper_interaction.append(sublist)
            #     # 去重
            #     merged_hyper_interaction = [list(set(sublist)) for sublist in merged_hyper_interaction]
            #     hyper_interaction = merged_hyper_interaction
            parse_rag_knowledge[algorithm] = {
                'Best Configurations': best_confs,
                # 'Hyperparameter Tuned Togother': hyper_interaction
            }

        # 先不打开超参交互
        # if self.hyper_interactions is None:
        #     hyper_interactions = {}
        #     for algorithm in parse_rag_knowledge:
        #         hyper_interactions[algorithm] = parse_rag_knowledge[algorithm]['Hyperparameter Tuned Togother']
        #     self.hyper_interactions = hyper_interactions

        return parse_rag_knowledge


    def fetch_algo_cate_knowledge(self):

        return [[algo, None] for algo in self.include_algorithms]

#         meta_dataset_dir = os.path.join(self.meta_dir, 'meta_dataset_vec')
#         if not os.path.exists(meta_dataset_dir):
#             os.makedirs(meta_dataset_dir)
#         save_path = os.path.join(meta_dataset_dir, '%s_meta_dataset_algocateknowledge.pkl' % self.task_prefix)

#         if os.path.exists(save_path):
#             with open(save_path, 'rb') as f:
#                 data = pickle.load(f)
#         else:
#             data = {}

#         if self.metric not in data:
#             data[self.metric] = {}


#         if self.task_id not in data[self.metric]:
#             data[self.metric][self.task_id] = {}

#         # 将self.include_algorithms转化为key
#         ia_key = '_'.join(sorted(self.include_algorithms))
#         if ia_key in data[self.metric][self.task_id]:
#             return data[self.metric][self.task_id][ia_key]


# #        model_scores = self.recommand_models() # [("random_forest", 0.95), ("lightgbm", 0.90), ...],
# #         historical_summary = f"""
# # **Algorithm Order and Priority Score:**
# # {model_scores}
# #         """
#         user_requirement_summary = get_user_requirement_summary(self.task_id, self.dataset_profile, self.task_type, model=self.include_algorithms)
#         response = retrieve_algorithm_category_knowledge(
#             task_type=self.task_type, user_requirement_summary=user_requirement_summary, 
#             llm_model=self.llm_model, client=self.client, top_k=50, data_dir=self.rag_data_dir)  # historical_knowledge=historical_summary, 

#         algo_cate = self._parse_experimence(response)  # [(algo1, priority1), (algo2, priority2), ...]

#         # 检查是否是否所有算法都被包含，检查是否有重叠
#         # all_algos = set()
#         # for cate in algo_cate:
#         #     for algo in algo_cate[cate]:
#         #         assert algo not in all_algos
#         #         assert algo in self.include_algorithms
#         #         all_algos.add(algo)

#         all_algos = [tmp[0] for tmp in algo_cate]
#         assert len(all_algos) == len(self.include_algorithms)
#         assert set(all_algos) == set(self.include_algorithms)


#         data[self.metric][self.task_id][ia_key] = algo_cate
#         with open(save_path, 'wb') as f:
#             pickle.dump(data, f)

#         return algo_cate


    def fetch_best_conf_knowledge(self):

        # 从RAG知识库中获取每个算法的调参经验：包括 最优配置 和 超参交互
        rag_knowledge = self.fetch_rag_knowledge()

        bestconf_knowledge = {}
        for algorithm in rag_knowledge:
            bestconf_knowledge[algorithm] = {
                'Best Configurations': rag_knowledge[algorithm]['Best Configurations']
            }



        # 从历史知识库中获取每个算法的调参经验：包括 dataset_similarity，Best Configurations，SHAP Analysis，SHAP Overall Importance
        # dataset_similarity = self.recommand_datasets(da_thr=1)  # {algorithm: [(dataset1, sim1), (dataset2, sim2), ...], ...}
        # bestconf_knowledge = self._bestconf_knowledge(dataset_similarity)  # {algorithm: {'Best Configurations': [...]}, ...}

        # for algorithm in dataset_similarity:
        #     bestconf_knowledge[algorithm].update({'dataset_similarity': dataset_similarity[algorithm]})
        #     bestconf_knowledge[algorithm]['Best Configurations'].extend(rag_knowledge[algorithm].get('Best Configurations', []))

        return bestconf_knowledge



    def update_history(self, obs, source, **extra_infos):

        BO_STRATEGY = "suggested by bayesian optimization"
        # 越大越好的 metric
        max_perf = - float(obs.objectives[0])
        # 原始metric，恢复所有正负处理
        perf = float(obs.objectives[0]) * (1 if self.lower_is_better else -1)


        selected_algo = obs.config['algorithm']
        ##### 全局配置和性能记录
        self.configs.append(obs.config)
        self.perfs.append(max_perf)

        self.configs_mab[selected_algo].append(obs.config)
        self.perfs_mab[selected_algo].append(max_perf)


        assert source in ['llm', 'bo', 'bo_mab', 'warmup'], f"Unknown source: {source}"
        # update bo history
        if not source.startswith('bo') or source == 'bo_mab': # 如果是bo，内部已经更新
            ori_config = obs.config
            new_obs = deepcopy(obs)
            new_config = Configuration(self.config_space, values=ori_config.get_dictionary())
            new_config = MyConfiguration(new_config, extra_info=ori_config.extra_info) # 直接传入引用，之后设置extra_info的时候，这里也会更新
            # 修改搜索空间
            new_obs.config = new_config
            self.bo_history.update_observation(new_obs)

        if not source.startswith('bo') or source == 'bo':
            ori_config = obs.config
            new_obs = deepcopy(obs)
            new_config = Configuration(self.bo_history_mab[selected_algo].config_space, values=ori_config.get_dictionary())
            new_config = MyConfiguration(new_config, extra_info=ori_config.extra_info)
            # 修改搜索空间
            new_obs.config = new_config
            self.bo_history_mab[selected_algo].update_observation(new_obs)


        # update llm history
        if source == 'llm':
            full_history_entry = { 
                "trail_id": extra_infos["trail_id"],
                "detailed_thought": extra_infos["detailed_thought"], 
                "thought_summary": extra_infos["thought_summary"], 
                "action_config": extra_infos["action_config"], 
                "observation_perf": extra_infos["observation_perf"], 
                "reflection_summary": extra_infos["reflection_summary"], 
            }
        elif source in ['bo', 'bo_mab']:
            full_history_entry = {
                "trail_id": extra_infos["trail_id"],
                "detailed_thought": BO_STRATEGY, 
                "thought_summary": BO_STRATEGY, 
                "action_config": self._cfg_to_dict(obs.config),
                "observation_perf": perf, # 原始的perf
                "reflection_summary": BO_STRATEGY,
            }
        elif source == 'warmup':
            full_history_entry = {
                "trail_id": extra_infos["trail_id"],
                "detailed_thought": "warmup configuration", 
                "thought_summary": "warmup configuration", 
                "action_config": self._cfg_to_dict(obs.config),
                "observation_perf": perf, # 原始的perf
                "reflection_summary": "warmup configuration",
            }
        else:
            raise ValueError(f"Unknown source: {source}")

        self.param_tuning_history[selected_algo].append(full_history_entry)


        algo_stats = self.algo_selection_history[selected_algo]
        algo_stats['count'] += 1
        algo_stats['scores'].append(perf)
        algo_stats['avg_score'] = np.mean(algo_stats['scores'])
        if self.lower_is_better:
            algo_stats['best_score'] = min(algo_stats['best_score'], perf)
            algo_stats['worst_score'] = max(algo_stats['worst_score'], perf)
        else:
            algo_stats['best_score'] = max(algo_stats['best_score'], perf)
            algo_stats['worst_score'] = min(algo_stats['worst_score'], perf)

        if source == 'llm':
            algo_stats['thoughts'].append({
                'trial_id': extra_infos["trail_id"],
                'detailed_thought': extra_infos["detailed_thought"], 'summary_thought': extra_infos["thought_summary"]
            })
        elif source in ['bo', 'bo_mab']:
            algo_stats['thoughts'].append({
                'trial_id': extra_infos["trail_id"],
                'detailed_thought': BO_STRATEGY,
                'summary_thought': BO_STRATEGY
            })
        else:
            algo_stats['thoughts'].append({
                'trial_id': extra_infos["trail_id"],
                'detailed_thought': "warmup configuration",
                'summary_thought': "warmup configuration"
            })


    def _cfg_to_dict(self, cfg: Configuration) -> Dict[str, Any]:
        # 去掉Constant类型的参数
        cfg_dict = {}

        for k in cfg:
            v = cfg[k]
            parameter = self.config_space.get_hyperparameter(k)
            if k != 'algorithm' and isinstance(parameter, (Constant, UnParametrizedHyperparameter)):
                continue
            cfg_dict[k] = v

        return cfg_dict


    def _dict_to_cfg(self, d: Dict[str, Any]) -> MyConfiguration:
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

        def _choices_for(hp):
            if hp is None:
                return []
            if isinstance(hp, (Constant, UnParametrizedHyperparameter)):
                return [getattr(hp, "value", getattr(hp, "default_value", None))]
            if hasattr(hp, "choices"):
                return list(hp.choices)
            if hasattr(hp, "sequence"):
                return list(hp.sequence)
            return [getattr(hp, "default_value", getattr(hp, "value", None))]

        hp_list = self.config_space.get_hyperparameters()
        hp_map = {hp.name: hp for hp in hp_list}

        # 1) Determine the active algorithm.
        algo_hp = hp_map.get('algorithm', None)
        algo_choices = _choices_for(algo_hp)
        algo = d.get('algorithm')
        if algo is None:
            # Infer algorithm from parameter prefixes (e.g., 'xgboost:n_estimators').
            prefixes = {k.split(':', 1)[0] for k in d.keys() if ':' in k}
            if prefixes:
                algo = list(prefixes)[0]
            elif algo_hp is not None:
                algo = getattr(algo_hp, 'default_value', getattr(algo_hp, 'value', None))
                if algo is None and algo_choices:
                    algo = algo_choices[0]
            else:
                raise ValueError("No 'algorithm' provided and cannot infer from keys.")
        if algo_hp is not None and algo not in algo_choices:
            raise ValueError(f"Invalid algorithm '{algo}'. Choices: {algo_choices}")

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

        return MyConfiguration(Configuration(self.config_space, values=values))
