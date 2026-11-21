#!/usr/bin/env python3
"""
最简单的脚本：在 porto-seguro 数据集上验证一个配置，会报 "Input y contains NaN" 错误。

这个错误是因为数据集本身的目标变量 (y) 包含 NaN 值，
不是模型生成的配置有问题。
"""

import sys
import types
from pathlib import Path

# 路径设置
BENCHMARK_ROOT = Path("/jizhicfs/pkuhetu/bcxu/llmoptimizer_benchmark")

# 补丁函数
def patch():
    import ConfigSpace.hyperparameters as hp_mod
    orig_i = hp_mod.UniformIntegerHyperparameter
    orig_f = hp_mod.UniformFloatHyperparameter
    def ui(name, lower, upper, default_value=None, q=None, log=False, meta=None, **kwargs):
        meta = dict(meta or {})
        if q is not None: meta["q"] = q
        return orig_i(name, lower, upper, default_value=default_value, log=log, meta=meta)
    def uf(name, lower, upper, default_value=None, q=None, log=False, meta=None, **kwargs):
        meta = dict(meta or {})
        if q is not None: meta["q"] = q
        return orig_f(name, lower, upper, default_value=default_value, log=log, meta=meta)
    ui._p = True; uf._p = True
    hp_mod.UniformIntegerHyperparameter = ui
    hp_mod.UniformFloatHyperparameter = uf
    
    import sklearn
    if getattr(sklearn, "__version__", "") > "1.3.2":
        sklearn.__version__ = "1.3.2"
    import sklearn.metrics._scorer as sm
    if not hasattr(sm, "_ThresholdScorer") and hasattr(sm, "_Scorer"):
        sm._ThresholdScorer = sm._Scorer
    
    mindware = types.ModuleType("mindware")
    mindware.__path__ = [str(BENCHMARK_ROOT / "mindware")]
    sys.modules["mindware"] = mindware
    
    for p in [str(BENCHMARK_ROOT), str(BENCHMARK_ROOT / "examples" / "benchmark")]:
        if p not in sys.path: sys.path.insert(0, p)

patch()

# 导入 benchmark 模块
from benchmark_utils import get_dataset_info
from mindware.components.config_space.cs_builder import get_cs_args, get_hpo_cs
from mindware.components.metrics.metric import get_metric
from mindware.components.utils.constants import CLASSIFICATION
from mindware.modules.cash.cash_evaluator import CASHCLSEvaluator
from mindware.utils.data_manager import DataManager
from ConfigSpace import Configuration

# 数据集和配置
DATASET = "porto-seguro"
ALGORITHM = "adaboost"
TASK_TYPE = "cls"  # classification

# 一个有效的配置（模型生成的）
CONFIG = {
    "adaboost:algorithm": "SAMME.R",
    "adaboost:learning_rate": 0.010138019064548476,
    "adaboost:max_depth": 2,
    "adaboost:n_estimators": 494,
}

print("=" * 60)
print("最简单的报错复现脚本")
print("=" * 60)
print(f"数据集: {DATASET}")
print(f"算法: {ALGORITHM}")
print(f"任务类型: {TASK_TYPE}")
print()
print("配置:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")
print()
print("开始验证...")
print()

# 加载数据
dataset_path, label_column, header, sep = get_dataset_info(TASK_TYPE, DATASET)
dataset_path = Path(dataset_path)
if not dataset_path.is_absolute():
    dataset_path = BENCHMARK_ROOT / dataset_path

dm = DataManager()
df = dm.load_csv(str(dataset_path), header=header, sep=sep)
train_df, _ = dm.split_data(df, label_col=label_column, test_size=0.2, random_state=1, task_type=CLASSIFICATION)
train_node = dm.from_train_df(train_df, label_col=label_column)
train_node = dm.preprocess_fit(train_node, CLASSIFICATION)

# 构建 ConfigSpace
cs = get_hpo_cs(
    estimator_id=ALGORITHM,
    task_type=CLASSIFICATION,
    **get_cs_args(
        resampling_params={"test_size": 0.25, "shuffle": False},
        data_node=train_node,
    ),
)

# 创建配置
cfg_dict = cs.get_default_configuration().get_dictionary()
cfg_dict.update(CONFIG)
cfg = Configuration(cs, values=cfg_dict)

# 运行评估
scorer = get_metric("bal_acc")
evaluator = CASHCLSEvaluator(
    scorer=scorer,
    data_node=train_node,
    resampling_strategy="holdout",
    resampling_params={"test_size": 0.25, "shuffle": False},
    timestamp=__import__("time").time(),
    seed=1,
)

try:
    result = evaluator(cfg.get_dictionary(), save=False)
    print(f"成功! 目标值: {result['objectives'][0]}")
except Exception as e:
    print(f"报错: {type(e).__name__}: {e}")
    print()
    print("说明: 这个错误是因为数据集 porto-seguro 的目标变量(y)包含 NaN 值,")
    print("      sklearn 的 AdaBoost 无法处理 NaN 目标值。")
    print("      这不是配置的问题，是数据集本身的问题。")
