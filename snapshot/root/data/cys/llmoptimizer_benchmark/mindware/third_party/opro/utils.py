# utils.py
import re
import math
import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    Constant,
    UnParametrizedHyperparameter,
    OrdinalHyperparameter
)


__all__ = [
    "_count_decimal_places",
    "_serialize_row",
    "_prepare_configurations_acquisition",
    "_constraints_text",
    "_dataset_summary_block",
    "_convert_to_dict",
    "_normalize_candidate",
    "_within_constraint",
    "_filter_and_validate",
    "_select_example_indices",
]

_FLOAT_RE = re.compile(r"^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$")



class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


# ----------------- formatting helpers -----------------
def _count_decimal_places(n: float) -> int:
    s = format(float(n), ".10f")
    return 0 if "." not in s else len(s.split(".")[1].rstrip("0"))


def _serialize_row(acq, row: pd.Series) -> str:
    """
    将一行 config（Series）序列化为 "## key: val, key: val, ... ##" 文本，
    根据 acq.constraints 的类型信息决定数值精度与呈现。
    """
    parts = []
    for k, v in row.items():
        if pd.isna(v):
            continue
        c = acq.constraints.get(k, {"type": "unknown"})
        t = c.get("type")
        if t == "int":
            parts.append(f"{k}: {int(v)}")
        elif t in ("float", "ordinal"):
            lb, ub = c.get("lower"), c.get("upper")
            dp = 6
            if lb is not None and ub is not None:
                dp = max(_count_decimal_places(lb), _count_decimal_places(ub))
            parts.append(f"{k}: {float(v):.{dp}f}")
        elif t == "categorical":
            parts.append(f"{k}: {str(v)}")
        elif t == "constant":
            parts.append(f"{k}: {str(c.get('value'))}")
        else:
            parts.append(f"{k}: {str(v)}")
    return "## " + ", ".join(parts) + " ##"


# ----------------- few-shot preparation -----------------
def _prepare_configurations_acquisition(
    acq, observed_configs: pd.DataFrame = None, observed_fvals: pd.DataFrame = None, seed: int = None
) -> List[Dict[str, Any]]:
    """
    构造 few-shot 示例。返回形如：
    [
      {"Q": "## ... ##", "A": "0.876000", "Label": "BEST"},
      ...
    ]
    其中 Label ∈ {"BEST", "RECENT", "WORST"}；多余字段对 LangChain FewShot 无影响。
    """
    examples: List[Dict[str, Any]] = []
    if observed_configs is None:
        if observed_fvals is not None:
            return [{"A": f"{float(observed_fvals):.6f}", "Label": "BEST"}]
        raise ValueError("observed_configs and observed_fvals are both None.")

    # 选子集（含标签）
    idx_with_tags = _select_example_indices(acq, observed_configs, observed_fvals)

    # 可选：增强多样性（按需开启）
    # rng = np.random.default_rng(seed or 0)
    # rng.shuffle(idx_with_tags)

    for i, tag in idx_with_tags:
        row = observed_configs.loc[i]
        ex = {"Q": _serialize_row(acq, row), "Label": tag}
        if observed_fvals is not None:
            perf = float(observed_fvals.loc[i].values[0])
            ex["A"] = f"{perf:.6f}"
        examples.append(ex)
    return examples


def _constraints_text(acq) -> str:
    lines = []
    if acq.algo_choices:
        lines.append(f"- algorithm: categorical in {acq.algo_choices}")
    for name, c in acq.constraints.items():
        if name == "algorithm":
            continue
        t = c.get("type")
        if t == "categorical":
            lines.append(f"- {name}: categorical in {c.get('choices')}")
        elif t == "int":
            lb, ub = c["lower"], c["upper"]
            if c.get("log", False) and acq.apply_warping:
                lb, ub = np.log10(lb), np.log10(ub)
            n_dp = max(_count_decimal_places(lb), _count_decimal_places(ub))
            lines.append(
                f"- {name}: [{lb:.{n_dp}f}, {ub:.{n_dp}f}] (int"
                f"{' log' if c.get('log', False) and acq.apply_warping else ''})"
            )
        elif t == "float":
            lb, ub = c["lower"], c["upper"]
            if c.get("log", False) and acq.apply_warping:
                lb, ub = np.log10(lb), np.log10(ub)
            n_dp = max(_count_decimal_places(lb), _count_decimal_places(ub))
            sfx = "log scale" if c.get("log", False) and acq.apply_warping else "float"
            lines.append(f"- {name}: [{lb:.{n_dp}f}, {ub:.{n_dp}f}] ({sfx}, precise to {n_dp} decimals)")
        elif t == "constant":
            lines.append(f"- {name}: constant = {c.get('value')}")
    return "\n".join(lines)



# 解析历史调优经验
def _build_algorithm_analysis_text(algorithm_analysis_knowledge: dict, strategy_name):


    """
    algorithm_analysis_knowledge = {
    'SHAP Analysis': [
        {
        'adaboost:learning_rate': {
            'Trends': 'Very low learning rates (<0.025) consistently harm performance. Optimal performance is achieved with learning rates between 0.6 and 1.97, with the strongest contributions around 0.6-1.0. The range 0.025-0.1 is uncertain (cautious), and 0.1-0.6 shows mixed results.',
            'Optimal ranges for exploitation': [0.6, 1.97],
            'Bad ranges to avoid': [0.01, 0.025],
            'Cautious ranges for exploration': [0.025, 0.1]
        },
        'adaboost:max_depth': {
            'Trends': 'SHAP values increase with max_depth, indicating deeper trees improve performance. The transition from negative to positive impact occurs around max_depth=4.',
            'Optimal ranges for exploitation': [5, 8],
            'Bad ranges to avoid': [2, 3],
            'Cautious ranges for exploration': [4, 4]
        },
        'adaboost:n_estimators': {
            'Trends': 'SHAP values transition from negative to positive as n_estimators increase, peaking in the mid-range (220-310). High values (310-500) show unpredictable performance.',
            'Optimal ranges for exploitation': [220, 310],
            'Bad ranges to avoid': [50, 200],
            'Cautious ranges for exploration': [310, 500]
        }
    ],

    """
    from mindware.components.optimizers.mcts import LLMEXPLORATION, LLMEXPLOITATION
    # strategy_name = EXPLORATION or EXPLOITATION

    lines = [f'Based on historical tuning experience, for {strategy_name}, we provide the following tuning advice:']
    shap_analysis = algorithm_analysis_knowledge['SHAP Analysis'][0]

    for param, analysis in shap_analysis.items():
        trends = analysis.get('Trends', None)
        optimal_ranges = analysis.get('Optimal ranges for exploitation', None)
        bad_ranges = analysis.get('Bad ranges to avoid', None)
        cautious_ranges = analysis.get('Cautious ranges for exploration', None)

        # if trends is not None:
        #     trends = "With higher SHAP values indicating better performance. " + trends
        #     lines.append(f"    Trends: {trends}")

        lines.append(f"- Parameter '{param}':")
        if strategy_name == LLMEXPLOITATION and optimal_ranges:
            lines.append(f"    Recommended range for exploitation: {str(optimal_ranges)}")
        if strategy_name == LLMEXPLORATION and cautious_ranges:
            lines.append(f"    Recommended range for exploration: {str(cautious_ranges)}")
        if bad_ranges:
            lines.append(f"    Bad ranges to avoid: {str(bad_ranges)}")


    return "\n    ".join(lines)




# 解析搜索空间与约束
def _build_hparam_constraints(cs: ConfigurationSpace) -> Dict[str, Dict[str, Any]]:
    space = {}
    for hp in cs.get_hyperparameters():
        if isinstance(hp, CategoricalHyperparameter):
            space[hp.name] = {"type": "categorical", "choices": list(hp.choices)}
        elif isinstance(hp, UniformFloatHyperparameter):
            desc = {"type": "float", "lower": float(hp.lower), "upper": float(hp.upper)}
            if getattr(hp, "log", False):
                desc["log"] = True
            if getattr(hp, "q", None) is not None:
                desc["q"] = float(hp.q)
            space[hp.name] = desc
        elif isinstance(hp, UniformIntegerHyperparameter):
            desc = {"type": "int", "lower": int(hp.lower), "upper": int(hp.upper)}
            if getattr(hp, "log", False):
                desc["log"] = True
            if getattr(hp, "q", None) is not None:
                desc["q"] = int(hp.q)
            space[hp.name] = desc
        elif isinstance(hp, Constant):
            space[hp.name] = {"type": "constant", "value": hp.value}
        elif isinstance(hp, UnParametrizedHyperparameter):
            space[hp.name] = {"type": "unparametrized", "value": hp.value}
        else:
            space[hp.name] = {"type": type(hp).__name__}
    return space


# 根据解析出的约束，生成文本描述
def _constraints_text_for_algo(selected_algo: str, algo2params: Dict, constraints: Dict, algo_choices: List) -> str:
    """为选定的算法生成仅包含其相关参数的约束文本块。"""
    algo_params = algo2params.get(selected_algo, [])
    # 加入 'algorithm' 超参数本身以提供上下文
    relevant_params = ['algorithm'] + algo_params
    algo_constraints = {k: v for k, v in constraints.items() if k in relevant_params}
    
    lines = []
    for name, properties in algo_constraints.items():
        if name == 'algorithm':
        #     lines.append(f"- {name}: Categorical, Current Choice: {selected_algo}, All Choices: {algo_choices}")
            continue

        p_type = properties.get("type")
        if p_type == "categorical":
            lines.append(f"- {name}: Categorical, Choices: {properties.get('choices')}")
        elif p_type in ("int", "float"):
            log_scale = " (log scale)" if properties.get("log") else ""
            lines.append(f"- {name}: {p_type.capitalize()}, Range: [{properties.get('lower')}, {properties.get('upper')}]" + log_scale)
    return "\n\t".join(lines)

def _format_number_scientific(value, is_log=False):
    """
    统一的数字格式化逻辑：
    - 如果 is_log=True：强制转科学计数法 (默认精度 .e)，保留更多细节。
    - 否则：使用 .6g 智能格式 (自动去零、自动处理极值)，保持紧凑。
    """
    if value is None:
        return "None"
        
    if isinstance(value, (float, np.floating, int, np.integer)):
        if is_log:
            s = f"{value:.6e}"
            
            base, exponent = s.split('e')
            
            if "." in base:
                base = base.rstrip('0').rstrip('.')
            
            return f"{base}e{exponent}"
        else:
            return f"{value:.6g}"
            
    return str(value)

def _constraints_text_for_selected_algo(cs, selected_algo=None):
    """
    基于 ConfigSpace 生成约束描述文本。
    
    Args:
        cs: ConfigurationSpace 对象
        selected_algo: 当前选中的算法名称 (str)。如果提供，只生成该算法相关的参数。
    """
    lines = []
    
    # 获取所有超参数
    all_hps = cs.get_hyperparameters()
    
    for hp in all_hps:
        param_name = hp.name

        if selected_algo:
            if param_name == 'algorithm':
                continue
            
            prefix = f"{selected_algo}:"
            if not param_name.startswith(prefix):
                continue
        
        if isinstance(hp, (Constant, UnParametrizedHyperparameter)):
            continue
        
        is_log = getattr(hp, 'log', False)

        range_str = ""
        param_type = "Unknown"
        
        if isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
            lower_str = _format_number_scientific(hp.lower, is_log)
            upper_str = _format_number_scientific(hp.upper, is_log)
            range_str = f"[{lower_str}, {upper_str}]"
            param_type = "Float" if isinstance(hp, UniformFloatHyperparameter) else "Integer"
            
        elif isinstance(hp, (CategoricalHyperparameter, OrdinalHyperparameter)):
            choices = [str(c) for c in hp.choices] # 保持原样
            range_str = f"{{{', '.join(choices)}}}"
            param_type = "Categorical"
            
        else:
            range_str = str(hp)

        # 如果已选中算法，去掉参数名中的算法前缀
        if selected_algo:
            prefix = f"{selected_algo}:"
            if param_name.startswith(prefix):
                display_name = param_name[len(prefix):]  # 去掉前缀
            else:
                display_name = param_name
        else:
            display_name = param_name
        
        # 组装 Log 提示 - 符合论文格式：(Log-Scale)
        log_hint = " (Log-Scale)" if is_log else ""
        
        # 符合论文格式：
        # - 对于数值类型：parameter_name: Type, Range=[...] (Log-Scale)
        # - 对于Categorical类型：parameter_name: Categorical, {choices}
        if param_type == "Categorical":
            lines.append(f"    - {display_name}: {param_type}, {range_str}")
        else:
            lines.append(f"    - {display_name}: {param_type}, Range={range_str}{log_hint}")
        
    # 确保所有行都有正确的缩进，使用换行符连接（第一行已经在上面加了缩进）
    return "\n".join(lines)

# 从搜索空间直接生成约束文本描述
def _constraints_text_for_cs(cs: ConfigurationSpace) -> str:
    properties = _build_hparam_constraints(cs)

    lines = []
    for name, prop in properties.items():

        p_type = prop.get("type")
        if p_type == "categorical":
            lines.append(f"- {name}: Categorical, Choices: {prop.get('choices')}")
        elif p_type in ("int", "float"):
            log_scale = " (log scale)" if prop.get("log") else ""
            lines.append(f"- {name}: {p_type.capitalize()}, Range: [{prop.get('lower')}, {prop.get('upper')}]" + log_scale)
    return "\n\t".join(lines)



def _dataset_summary_block(acq) -> str:
    prof = acq.task_context.get("dataset_profile")
    metric = acq.task_context.get("metric", "score")
    if not prof:
        return ""
    lines = []
    lines.append(f"Task: {prof.get('task_type','unknown')}  |  Metric: {metric}")
    lines.append(
        f"Dataset: {prof.get('n_samples',0)} samples, {prof.get('n_features',0)} features "
        f"({prof.get('n_numeric',0)} numeric, {prof.get('n_categorical',0)} categorical)"
    )
    mr = prof.get("missing_rate", 0.0)
    tops = prof.get("missing_top", [])
    if mr and mr > 0:
        tops_str = ", ".join([f"{d['col']}:{d['rate']:.3f}" for d in tops])
        lines.append(f"Missing rate: {mr:.4f} (top cols: {tops_str})")
    if prof.get("task_type") == "classification" and "class_distribution" in prof:
        cd = prof["class_distribution"]
        dist = cd.get("distribution", {})
        dist_str = ", ".join([f"{k}:{v:.3f}" for k, v in dist.items()])
        lines.append(f"Classes: {cd.get('n_classes','?')}  |  Distribution: {dist_str}")
    if prof.get("task_type") == "regression" and "target_stats" in prof:
        ts = prof["target_stats"]
        parts = []
        for k in ["mean", "std", "min", "median", "max"]:
            if k in ts:
                parts.append(f"{k}={ts[k]:.4f}")
        if parts:
            lines.append("Target stats: " + ", ".join(parts))
    return "\n".join(lines)

# def _build_dataset_summary_text(task_context: Dict) -> str:
#     """根据任务上下文中的数据集画像，创建自然语言描述。"""
#     prof = task_context.get("dataset_profile", {}) or {}
#     metric = task_context.get("metric", "score")
#     task_type = prof.get("task_type", "unknown task")
#     parts = [f"This is a {task_type} problem. The optimization metric is '{metric}'."]
    
#     n_samples = prof.get("n_samples")
#     n_features = prof.get("n_features")
#     if n_samples is not None and n_features is not None:
#         parts.append(f"The dataset contains {n_samples} samples and {n_features} features.")
    
#     return " ".join(parts)

def _build_dataset_summary_text(task_context: Dict) -> str:
    """根据任务上下文中的数据集画像，创建详细的自然语言描述（已移除缺失值信息）。"""
    
    # 1. 从 task_context 获取 dataset_profile
    # (这个 prof 字典是由你的 benchmark_utils.py 中的 build_dataset_profile 函数生成的)
    prof = task_context.get("dataset_profile", {}) or {}
    
    # 2. 基础信息
    task_type = prof.get("task_type", "unknown task")
    metric = prof.get("metric", task_context.get("metric", "score"))
    metric_direction = prof.get("metric_direction", "")
    metric = f"'{metric}'"
    if metric_direction:
        metric += f" ({metric_direction})"
    parts = [f"Task & Metric: This is a **{task_type}** problem. The optimization metric is **{metric}**."]
    
    # 3. 维度信息
    n_samples = prof.get("n_samples")
    n_features = prof.get("n_features")
    if n_samples is not None and n_features is not None:
        parts.append(f"Dimensions: The dataset contains **{n_samples} samples** and **{n_features} total features**.")
    
    # 4. 特征类型
    n_numeric = prof.get("n_numeric")
    n_categorical = prof.get("n_categorical")
    if n_numeric is not None and n_categorical is not None:
        parts.append(f"Feature Types: {n_numeric} numeric features and {n_categorical} categorical features.")
        
    # 5. 缺失值信息 (*** 已按要求移除 ***)
    # missing_rate = prof.get("missing_rate")
    # if missing_rate is not None:
    #    ... (相关代码已删除) ...

    # 6. 目标变量信息 (分类)
    # 'class_distribution': {'n_classes': n, 'distribution': {...}}
    class_dist_info = prof.get("class_distribution")
    if class_dist_info:
        n_classes = class_dist_info.get("n_classes")
        distribution = class_dist_info.get("distribution")
        
        if n_classes is not None:
            parts.append(f"Target (Classification): This is a **{n_classes}-class** problem.")
        
        if distribution:
            # 检查不平衡
            dist_values = [v for k, v in distribution.items() if k != 'others']
            is_imbalanced = False
            if dist_values and min(dist_values) > 0 and len(dist_values) > 1:
                 if (max(dist_values) / min(dist_values) > 10):
                     is_imbalanced = True
            
            if is_imbalanced:
                parts.append("  - **Warning**: The classes appear to be **imbalanced**.")
                
            dist_str = ", ".join([f"'{k}': {v:.1%}" for k, v in distribution.items()])
            parts.append(f"Class distribution: {dist_str}.")

    # 7. 目标变量信息 (回归)
    # 'target_stats': {'mean': ..., 'std': ..., ...}
    target_stats = prof.get("target_stats")
    if target_stats:
        parts.append(f"**Target (Regression)**: Target variable statistics:")
        stats_parts = []
        if 'mean' in target_stats and 'std' in target_stats:
            stats_parts.append(f"Mean={target_stats['mean']:.2f} (StdDev={target_stats['std']:.2f})")
        if 'min' in target_stats and 'max' in target_stats:
            stats_parts.append(f"Range=[{target_stats['min']:.2f}, {target_stats['max']:.2f}]")
        if 'median' in target_stats:
            stats_parts.append(f"Median={target_stats['median']:.2f}")
        
        if stats_parts:
            parts.append(f"{'; '.join(stats_parts)}.")

    return "\n    ".join(parts)


def _build_dataset_summary_text_onehot(task_context: Dict, selected_algo: str=None) -> str:
    """根据任务上下文中的数据集画像，创建详细的自然语言描述（已移除缺失值信息）。"""
    
    # 1. 从 task_context 获取 dataset_profile
    # (这个 prof 字典是由你的 benchmark_utils.py 中的 build_dataset_profile 函数生成的)
    prof = task_context.get("dataset_profile", {}) or {}
    
    # 2. 基础信息
    task_type = prof.get("task_type", "unknown task")
    metric = prof.get("metric", task_context.get("metric", "score"))
    metric_direction = prof.get("metric_direction", "")
    metric = f"'{metric}'"
    if metric_direction:
        metric += f" ({metric_direction})"
    parts = [f"Task & Metric: This is a **{task_type}** problem. The optimization metric is **{metric}**."]
    
    # 3. 维度信息
    n_samples = prof.get("n_samples")
    n_features = prof.get("n_features")
    
    if n_samples is not None and n_features is not None:
        parts.append(f"Dimensions: The dataset contains **{n_samples} samples** and **{n_features} total features**.")
    
    # 4. 特征类型
    n_numeric = prof.get("n_numeric")
    n_categorical = prof.get("n_categorical")
    n_features_post = prof.get("n_features_post_processing")
    if n_numeric is not None and n_categorical is not None:
        feat_str = f"Feature Types: {n_numeric} numeric features and {n_categorical} categorical features."
        if n_categorical > 0 and n_features_post is not None:
            if selected_algo != "catboost":
                feat_str += f" (After one-hot encoding categorical features, the total feature dimension becomes **{n_features_post}**)."
        
        parts.append(feat_str)
        
    # 5. 缺失值信息 (*** 已按要求移除 ***)
    # missing_rate = prof.get("missing_rate")
    # if missing_rate is not None:
    #    ... (相关代码已删除) ...

    # 6. 目标变量信息 (分类)
    # 'class_distribution': {'n_classes': n, 'distribution': {...}}
    class_dist_info = prof.get("class_distribution")
    if class_dist_info:
        n_classes = class_dist_info.get("n_classes")
        distribution = class_dist_info.get("distribution")
        
        if n_classes is not None:
            parts.append(f"Target (Classification): This is a **{n_classes}-class** problem.")
        
        if distribution:
            # 检查不平衡
            dist_values = [v for k, v in distribution.items() if k != 'others']
            is_imbalanced = False
            if dist_values and min(dist_values) > 0 and len(dist_values) > 1:
                 if (max(dist_values) / min(dist_values) > 10):
                     is_imbalanced = True
            
            if is_imbalanced:
                parts.append("  - **Warning**: The classes appear to be **imbalanced**.")
                
            dist_str = ", ".join([f"'{k}': {v:.1%}" for k, v in distribution.items()])
            parts.append(f"Class distribution: {dist_str}.")

    # 7. 目标变量信息 (回归)
    # 'target_stats': {'mean': ..., 'std': ..., ...}
    target_stats = prof.get("target_stats")
    if target_stats:
        parts.append(f"**Target (Regression)**: Target variable statistics:")
        stats_parts = []
        if 'mean' in target_stats and 'std' in target_stats:
            stats_parts.append(f"Mean={target_stats['mean']:.2f} (StdDev={target_stats['std']:.2f})")
        if 'min' in target_stats and 'max' in target_stats:
            stats_parts.append(f"Range=[{target_stats['min']:.2f}, {target_stats['max']:.2f}]")
        if 'median' in target_stats:
            stats_parts.append(f"Median={target_stats['median']:.2f}")
        
        if stats_parts:
            parts.append(f"{'; '.join(stats_parts)}.")

    return "\n    ".join(parts)


# ----------------- parsing & validation -----------------
def _convert_to_dict(s: str) -> Dict[str, Any]:
    """Parses 'key: value, ...' string into a dictionary."""
    s = s.strip()
    if s.startswith("##"): s = s[2:]
    if s.endswith("##"): s = s[:-2]
    pairs = [p.strip() for p in s.split(",") if p.strip()]
    out: Dict[str, Any] = {}
    for pair in pairs:
        if ":" not in pair: continue
        k_raw, v_raw = [x.strip() for x in pair.rsplit(":", 1)]
        k = ":".join([t.strip() for t in k_raw.split(":")])
        v = v_raw.strip().strip('\'"`')
        vl = v.lower()
        if vl in ("true", "false"):
            out[k] = vl == "true"
        elif vl in ("none", "null"):
            out[k] = None
        elif _FLOAT_RE.match(v):
            try: out[k] = float(v)
            except: out[k] = v
        else:
            out[k] = v
    return out


def _normalize_candidate(acq, cand: Dict[str, Any]) -> Dict[str, Any]:

    """
    归一化：
    - 推断/确保 algorithm 存在；
    - 裸参数名（无前缀）在当前 algo 下做唯一映射 '<algo>:param'；
    - 丢弃与当前 algo 无关的键（如 'xgboost' 这种根键）。
    """
    cand = dict(cand)  # copy
    dbg = []

    algo = cand.get("algorithm")
    if algo is None:
        prefixes = {k.split(":", 1)[0] for k in cand.keys() if ":" in k}
        if len(prefixes) == 1:
            algo = list(prefixes)[0]
            cand["algorithm"] = algo
            dbg.append(f"infer algorithm -> {algo}")
        else:
            dbg.append(f"cannot infer algorithm from prefixes {prefixes}")

    if algo not in acq.algo_choices:
        print(f"[ACQ-Debug][Normalize] BAD_ALGO: {algo}, allowed={acq.algo_choices}. cand={cand}")
        return {}

    # 映射裸参数名 -> '<algo>:param'
    algo_params = acq.algo2params.get(algo, [])
    base_names = {p.split(":", 1)[1]: p for p in algo_params}  # 'subsample' -> 'lightgbm:subsample'

    to_add, to_del = {}, []
    for k, v in cand.items():
        if k == "algorithm":
            continue
        if ":" not in k:
            mapped = base_names.get(k)
            if mapped is not None:
                to_add[mapped] = v
                to_del.append(k)
                dbg.append(f"map bare '{k}' -> '{mapped}'")
        else:
            if not k.startswith(algo + ":"):
                to_del.append(k)
                dbg.append(f"drop cross-algo key '{k}' (algo={algo})")

        if k in acq.algo_choices and k != "algorithm":
            to_del.append(k)
            dbg.append(f"drop stray root algo key '{k}'")

    for k in to_del:
        cand.pop(k, None)
    cand.update(to_add)

    print(f"[ACQ-Debug][Normalize] algo={algo} | actions={dbg} | result={cand}")
    return cand


def _within_constraint(acq, k: str, v: Any) -> bool:
    c = acq.constraints.get(k)
    if c is None:
        return True

    t = c.get("type")

    if t == "categorical":
        ok = str(v) in set(map(str, c.get("choices", [])))
        if not ok:
            print(f"[ACQ-Debug][Check] {k}='{v}' NOT IN choices={c.get('choices')}")
        return ok

    if t == "constant":
        target = c.get("value")
        ok = str(v) == str(target)
        if not ok:
            try:
                ok = float(v) == float(target)
            except (ValueError, TypeError):
                pass
        if not ok:
            print(f"[ACQ-Debug][Check] {k}='{v}' != constant_value={target}")
        return ok

    if t == "int":
        ok = v is not None and isinstance(v, (int, float)) and float(v).is_integer()
        if not ok:
            print(f"[ACQ-Debug][Check] {k}='{v}' is not integer-like")
            return False
        val = int(v)
        lb, ub = int(c["lower"]), int(c["upper"])
        
        # MODIFIED: Use <= to create an inclusive (closed) interval [lb, ub]
        ok = (lb <= val <= ub)
        
        if not ok:
            # MODIFIED: Update print statement to reflect the inclusive interval
            print(f"[ACQ-Debug][Check] {k}={val} NOT IN [{lb}, {ub}]")
        return ok

    if t == "float":
        try:
            val = float(v)
        except Exception:
            return False

        lb, ub = float(c["lower"]), float(c["upper"])
        if c.get("log", False) and acq.apply_warping:
            lb, ub = np.log10(lb), np.log10(ub)

        # 浮点容差，避免边界抖动
        dp = max(_count_decimal_places(lb), _count_decimal_places(ub))
        eps = max(1e-12, 10 ** (-(dp + 2)))

        # 默认闭区间；对 *:max_features 用 (0, 1]
        lower_inclusive = True
        upper_inclusive = True
        if k.endswith(":max_features"):
            lower_inclusive = False
            upper_inclusive = True

        lower_ok = (lb - eps) <= val if lower_inclusive else (val > lb + eps)
        upper_ok = val <= (ub + eps) if upper_inclusive else (val < ub - eps)

        ok = lower_ok and upper_ok
        if not ok:
            print(f"[ACQ-Debug][Check] {k}={val} NOT IN interval with bounds ({lb}, {ub}) and inclusivity ({lower_inclusive}, {upper_inclusive})")
        return ok

    if t == "ordinal":
        rng = c.get("range") or c.get("choices") or []
        try:
            fval = float(v)
            return any(math.isclose(fval, float(x), abs_tol=1e-8) for x in rng)
        except Exception:
            return str(v) in set(map(str, rng))

    # 其他类型放过
    return True


def _filter_and_validate(
    acq, observed_points: List[dict], candidate_points: List[dict]
) -> pd.DataFrame:
    """
    - 归一化 & 约束校验
    - 去除与已观测重复的配置（基于四舍五入后的 dict）
    - 保留去重后的有效候选
    """
    def round_dict(d: Dict[str, Any], precision: int = 8) -> Dict[str, Any]:
        out = {}
        for k, v in d.items():
            out[k] = round(v, precision) if isinstance(v, float) else v
        return out

    observed_norm = [round_dict(d) for d in observed_points]
    valid: List[Dict[str, Any]] = []

    for idx, raw in enumerate(candidate_points):
        cand = _normalize_candidate(acq, raw)
        if not cand or "algorithm" not in cand:
            print(f"[ACQ-Debug][Cand] REJECT: EMPTY_AFTER_NORMALIZE or MISSING_ALGO")
            continue

        bad_keys = []
        for k, v in cand.items():
            if not _within_constraint(acq, k, v):
                bad_keys.append((k, v, acq.constraints.get(k)))
        if bad_keys:
            print(f"[ACQ-Debug][Cand {idx}] REJECT: CONSTRAINT_FAIL -> {bad_keys}")
            continue

        norm = round_dict(cand)
        if norm in observed_norm:
            continue
        valid.append(cand)

    if not valid:
        return pd.DataFrame()
    return pd.DataFrame(valid).drop_duplicates().reset_index(drop=True)


# ----------------- history selection -----------------
def _select_example_indices(
    acq, observed_configs: pd.DataFrame, observed_fvals: pd.DataFrame
) -> List[Tuple[int, str]]:
    """
    选择一个较小的索引集合（带标签）：
      - Top-K 最优 -> "BEST"
      - Worst-K 最差 -> "WORST"
      - Recent-K 最近 -> "RECENT"
    并施加 per-algo 上限与总体 history_budget 限制。
    返回: List[(index, label)]
    """
    if observed_configs is None or observed_fvals is None or len(observed_configs) == 0:
        return []

    df = observed_configs.copy()
    df["_score"] = observed_fvals.values.astype(float).ravel()

    ascending = True if acq.lower_is_better else False
    df_sorted = df.sort_values("_score", ascending=ascending)

    best_idx = list(df_sorted.index[: acq.top_k])
    worst_idx = list(df_sorted.index[-acq.worst_k :]) if acq.worst_k > 0 else []
    recent_idx = list(df.index[-acq.recent_k :]) if acq.recent_k > 0 else []

    buckets = {
        "best": [(i, "BEST") for i in best_idx],
        "recent": [(i, "RECENT") for i in recent_idx],
        "worst": [(i, "WORST") for i in worst_idx],
    }

    merged: List[Tuple[int, str]] = []
    for key in acq.interleave_order:
        for item in buckets.get(key, []):
            if item[0] not in [m[0] for m in merged]:
                merged.append(item)

    # per-algo cap
    algo_col = "algorithm"
    algo_counts: Dict[str, int] = {}
    selected: List[Tuple[int, str]] = []
    for i, tag in merged:
        if algo_col in df.columns:
            algo = df.loc[i, algo_col]
        else:
            # 若无显式 algo 列，尽量从参数列前缀推断
            algo = None
            for c in df.columns:
                if ":" in c and not pd.isna(df.loc[i, c]):
                    algo = c.split(":", 1)[0]
                    break

        if algo is None:
            selected.append((i, tag))
            continue

        cnt = algo_counts.get(algo, 0)
        if cnt < acq.per_algo_cap:
            selected.append((i, tag))
            algo_counts[algo] = cnt + 1

    # 总体预算
    if len(selected) > acq.history_budget:
        selected = selected[: acq.history_budget]

    return selected


def random_sample_algo_config(algo: str, config_space):
    # 从搜索空间中随机采样一个配置，但是algorithm一定要是指定的algo

    config = config_space.sample_configuration()
    while config.get("algorithm") != algo:
        config = config_space.sample_configuration()

    return config.get_dictionary()