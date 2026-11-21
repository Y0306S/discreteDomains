import time
import os
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple

from ConfigSpace import Configuration
from openbox.utils.limit import run_obj_func
from openbox.utils.history import Observation
from openbox.utils.util_funcs import parse_result
from openbox.utils.constants import SUCCESS, FAILED, TIMEOUT


def wrapper_evaluater(config, evaluator, per_run_time_limit=None):

    # evaluate configuration on objective_function
    obj_args, obj_kwargs = (config,), dict()
    if per_run_time_limit is not None:
        from mindware.components.optimizers.isolated_evaluator import run_isolated

        result = run_isolated(evaluator, obj_args, obj_kwargs, per_run_time_limit)
    else:
        result = run_obj_func(evaluator, obj_args, obj_kwargs, None)

    # parse result
    ret, timeout_status, traceback_msg, elapsed_time = (
        result['result'], result['timeout'], result['traceback'], result['elapsed_time'])
    if timeout_status:
        trial_state = TIMEOUT
    elif traceback_msg is not None:
        trial_state = FAILED
        print(f'Exception in objective function:\n{traceback_msg}\nconfig: {config}')
    else:
        trial_state = SUCCESS
    if trial_state == SUCCESS:
        objectives, constraints, extra_info = parse_result(ret)
    else:
        objectives, constraints, extra_info = [np.inf], None, None

    # update observation to advisor
    observation = Observation(
        config=config, objectives=objectives, constraints=constraints,
        trial_state=trial_state, elapsed_time=elapsed_time, extra_info=extra_info,
    )

    return observation

# 写一个Exploration与Exploitation平衡的工具函数
# class名字简洁明了
class ExplorationExploitationBalancer:
    def __init__(self, alpha: float = 0.6, algo_choices=[], back_window: int = 5):
        self.alpha = alpha  # 控制探索与利用的平衡

        self.best_perf = {a: -np.inf for a in algo_choices + ['GLOBAL']}  # 记录每个算法的最优分数

        self.cumulative_rewards = {
            a: {
                'exploration': [],
                'exploitation': []
            } for a in algo_choices + ['GLOBAL']
        }  # 记录exploration-exploitation的累计奖励

        self.algo_prob_record = []
        self.param_prob_record = []

        self.back_window = back_window  # 用于计算reward均值的窗口大小

    # 只更新最优分数
    def update_best_perf(self, perf: float, algo_name):

        for algo in [algo_name, 'GLOBAL']:
            if perf > self.best_perf[algo]:
                self.best_perf[algo] = perf

    # 计算奖励后，更新最优分数
    def update_cumulative_rewards(self, perf: float, algo_strategy_name, param_strategy_name, algo_name):

        # 计算奖励

        for algo in [algo_name, 'GLOBAL']:

            if not np.isfinite(self.best_perf[algo]):
                self.best_perf[algo] = perf
                continue

            reward = max(0, perf - self.best_perf[algo])  # 只在有提升时给予奖励


            strategy_name = algo_strategy_name if algo == 'GLOBAL' else param_strategy_name

            # 根据策略名称更新累计奖励
            if 'exploration' in strategy_name.lower():
                self.cumulative_rewards[algo]['exploration'].append(reward)
            else:
                self.cumulative_rewards[algo]['exploitation'].append(reward)

            # 更新最佳性能
            if perf > self.best_perf[algo]:
                self.best_perf[algo] = perf

    def get_algo_exploration_prob(self) -> float:
        if len(self.cumulative_rewards['GLOBAL']['exploration']) < 2 or len(self.cumulative_rewards['GLOBAL']['exploitation']) < 2:
            smoothed_ratio = 0.5  # 初始时探索与利用各占一半
        else:
            avg_exploration = np.mean(self.cumulative_rewards['GLOBAL']['exploration'][-self.back_window:])
            avg_exploitation = np.mean(self.cumulative_rewards['GLOBAL']['exploitation'][-self.back_window:])

            if avg_exploration == avg_exploitation:
                ratio = 0.5
            else:
                ratio = avg_exploration / (avg_exploration + avg_exploitation + 1e-10)

            # 通过将当前计算的比例与0.5（均衡点）进行加权平均，alpha控制了新信息的影响力。
            smoothed_ratio = (1 - self.alpha) * 0.5 + self.alpha * ratio

        self.algo_prob_record.append(smoothed_ratio)

        return smoothed_ratio

    def get_param_exploration_prob(self, algo_name) -> float:
        if len(self.cumulative_rewards[algo_name]['exploration']) < 2 or len(self.cumulative_rewards[algo_name]['exploitation']) < 2:
            smoothed_ratio = 0.5  # 初始时探索与利用各占一半
        else:
            avg_exploration = np.mean(self.cumulative_rewards[algo_name]['exploration'][-self.back_window:])
            avg_exploitation = np.mean(self.cumulative_rewards[algo_name]['exploitation'][-self.back_window:])

            if avg_exploration == avg_exploitation:
                ratio = 0.5
            else:
                ratio = avg_exploration / (avg_exploration + avg_exploitation + 1e-10)

            # 通过将当前计算的比例与0.5（均衡点）进行加权平均，alpha控制了新信息的影响力。
            smoothed_ratio = (1 - self.alpha) * 0.5 + self.alpha * ratio

        self.param_prob_record.append(smoothed_ratio)

        return smoothed_ratio





def _record_event(
    self,
    stage: str,
    cfg: Configuration,
    perf: float,
    llm_cost: float = 0.0,
    llm_time: float = 0.0,
    eval_time: float = 0.0,
    iter_time: float = 0.0,
    note: str = "",
    iter_idx: int = None,
    algo_exploration_prob: float = None,
    param_exploration_prob: float = None,
    param_strategy_name: str = None,
    parent_node_id: int = None
):
    """
    Records a single evaluation event and its metrics.
    [MODIFIED] Correctly calculates and logs cumulative LLM cost.
    """
    from datetime import datetime
    now = time.time()
    if self.start_wall_time is None:
        self.start_wall_time = now
    wall_elapsed = float(now - self.start_wall_time)

    d = cfg.get_dictionary() if hasattr(cfg, "get_dictionary") else dict(cfg)
    algo = d.get("algorithm", "unknown")
    brief = {k: v for k, v in d.items() if (k == "algorithm") or k.startswith(f"{algo}:")}
    
    # --- 修正点 1: 修正累加逻辑 ---
    # 我们直接使用 react_acq 中的累加器，这是最准确的数据源。
    # 对于后备方案，我们手动将当前迭代的成本加到历史总和中。
    cumulative_llm_cost = 0.0
    if self.prompt_strategy == 'react' and hasattr(self.acq_func, 'get_total_llm_cost'):
        cumulative_llm_cost = self.acq_func.get_total_llm_cost()
    else:
        # 正确的后备逻辑: 将历史总成本与本次迭代的成本相加
        previous_total = sum(self.llm_query_cost)
        cumulative_llm_cost = previous_total + llm_cost

    row = {
        "ts": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
        "stage": stage,
        "iter_idx": iter_idx,  
        "perf": float(perf),
        "best_so_far": float(max(self.perfs) if self.perfs else perf),
        "algorithm": algo,
        "config_brief": brief, # 注意：这一列也需要被添加到header中才能显示
        "llm_cost": float(llm_cost),
        "cumulative_llm_cost": cumulative_llm_cost, # 包含正确的累计成本
        "llm_time": float(llm_time),
        "eval_time": float(eval_time),
        "iter_time": float(iter_time),
        "wall_elapsed_sec": wall_elapsed,
        "note": note,
        "algo_exploration_prob": algo_exploration_prob,
        "param_exploration_prob": param_exploration_prob,
        "param_strategy_name": param_strategy_name,
        "parent_node_id": parent_node_id
    }
    self.traj_rows.append(row)

    try:
        import csv
        # --- 修正点 2: 将新列添加到CSV表头 ---
        # 确保所有你希望记录在CSV中的列都在这个列表中
        header = [
            "ts", "stage", "iter_idx", "perf", "best_so_far", "algorithm",
            "llm_cost", "cumulative_llm_cost", "llm_time", "eval_time", 
            "iter_time", "wall_elapsed_sec", "note", "config_brief", "algo_exploration_prob", "param_exploration_prob", 
            "param_strategy_name", "parent_node_id"
        ]
        file_exists = os.path.exists(self.traj_csv_path)
        with open(self.traj_csv_path, "a", newline="", encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=header)
            if not file_exists:
                w.writeheader()
            # 使用 writerow 前，筛选字典，确保只写入 header 中存在的键
            w.writerow({k: v for k, v in row.items() if k in header})
    except Exception as e:
        # 建议打印错误，以便调试
        print(f"[ERROR] Failed to write to CSV in _record_event: {e}")
        pass

    # 日志打印部分保持不变...
    if hasattr(self, "logger") and self.logger is not None:
        self.logger.info(
            f"[{stage}] iter={row['iter_idx']} "
            f"algo={row['algorithm']} "
            f"perf={row['perf']:.4f} "
            f"best_so_far={row['best_so_far']:.4f} "
            f"llm_cost={row['llm_cost']:.2f} "
            f"cum_cost={row['cumulative_llm_cost']:.2f} " # 可选：在日志中也打印出来
            f"note={note}"
        )


def _evaluate_dict(self, cfg_dict: Dict[str, Any]) -> float:
    """
    用 evaluator 评估一个配置字典（会调用 self._dict_to_cfg 做补全/类型矫正）。
    """
    t0_eval = time.time()
    if isinstance(cfg_dict, dict):
        cfg = self.KD._dict_to_cfg(cfg_dict)
    else:
        cfg = cfg_dict
    obs = self.objective_function(cfg)
    eval_time = time.time() - t0_eval

    # evaluator 返回的是要最小化的 loss？这里与 run 内保持一致：外部用 “-val” 当作精度。
    return obs, eval_time



def _cfgs_df(self) -> pd.DataFrame:
    rows = []
    for cfg in self.configs:
        d = cfg.get_dictionary()
        rows.append(d)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _init_f(self, n: int) -> List[Dict[str, Any]]:
    lst = []
    for _ in range(n):
        cfg = self.config_space.sample_configuration()
        d = cfg.get_dictionary()
        lst.append(d)
    return lst



def _update_incumbent_from_current(self):
    if not self.perfs:
        return
    best = int(np.argmax(self.perfs))  # acc 最大
    self.incumbent_config = self.configs[best].get_dictionary().copy()
    self.incumbent_perf = float(self.perfs[best])


def _time_is_up(self, start_ts: float) -> bool:
    return self.time_limit is not None and (time.time() - start_ts) > self.time_limit


def _quota_is_up(self) -> bool:
    return self.evaluation_num_limit is not None and len(self.perfs) >= self.evaluation_num_limit


def _build_dataset_profile_from_df(self, df, label_column, task_type_str: str, metric_name: str):
    import numpy as np, pandas as pd

    # 解析 y
    def _resolve_y(df, label_column):
        if isinstance(label_column, int):
            if label_column == -1:
                return df.iloc[:, -1], df.columns[-1]
            return df.iloc[:, label_column], df.columns[label_column]
        else:
            return df[label_column], label_column

    y, label_name = _resolve_y(df, label_column)
    X = df.drop(columns=[label_name], errors='ignore')

    n_features = int(X.shape[1])
    obj_bool_cols = list(X.select_dtypes(include=['object','bool']).columns)
    num_df = X.select_dtypes(include=['number'])
    cat_like_numeric = []
    for c in num_df.columns:
        ser = num_df[c]
        if pd.api.types.is_integer_dtype(ser) or np.allclose(ser.dropna() % 1, 0):
            uniq = ser.dropna().unique()
            if len(uniq) <= 10 or len(uniq) <= max(1, int(0.02 * len(ser))):
                cat_like_numeric.append(c)
    cat_cols = set(obj_bool_cols) | set(cat_like_numeric)
    n_categorical = int(len(cat_cols))
    n_numeric = int(n_features - n_categorical)

    cells = int(X.shape[0] * max(1, X.shape[1]))
    miss_total = int(X.isna().sum().sum())
    miss_rate = float(miss_total / cells) if cells > 0 else 0.0
    per_col = (X.isna().sum() / max(1, X.shape[0])).sort_values(ascending=False)
    top_cols = per_col[per_col > 0].head(5)

    prof = {
        'n_samples': int(df.shape[0]),
        'n_features': n_features,
        'n_numeric': n_numeric,
        'n_categorical': n_categorical,
        'missing_rate': round(miss_rate, 6),
        'missing_top': [{'col': c, 'rate': round(float(r), 6)} for c, r in top_cols.items()],
        'task_type': 'classification' if task_type_str == 'CLS' else 'regression',
        'metric': metric_name
    }
    if task_type_str == 'CLS':
        vc = y.value_counts(dropna=False)
        total = int(vc.sum())
        items = [(str(k), int(v)) for k, v in vc.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        head, others = items[:8], items[8:]
        dist = {k: round(v/total, 6) for k, v in head}
        if others:
            dist['others'] = round(sum(v for _, v in others)/total, 6)
        prof['class_distribution'] = {'n_classes': int(len(vc)), 'distribution': dist}
    else:
        arr = pd.to_numeric(y, errors='coerce').values
        arr = arr[~np.isnan(arr)]
        if arr.size > 0:
            prof['target_stats'] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'median': float(np.median(arr)),
                'max': float(np.max(arr)),
            }
    return prof
