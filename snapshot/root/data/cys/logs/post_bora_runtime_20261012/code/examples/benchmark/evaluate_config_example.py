import json
import os
import sys
import time

from ConfigSpace import Configuration

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmark_utils import get_dataset_info
from mindware.components.config_space.cs_builder import get_cs_args, get_hpo_cs
from mindware.components.metrics.metric import get_metric
from mindware.components.utils.constants import CLASSIFICATION, REGRESSION
from mindware.modules.cash.cash_evaluator import CASHCLSEvaluator, CASHRGSEvaluator
from mindware.utils.data_manager import DataManager


# ===== 1. Set the dataset, algorithm, and hyperparameters. =====
TASK_TYPE = "cls"  # "cls" or "rgs"
task_type_id = CLASSIFICATION if TASK_TYPE == "cls" else REGRESSION
SEED = 1  # 我们约定就用1作为随机种子，保持和benchmark一致，方便结果对齐。

TEST_SIZE = 0.2
VALID_SIZE = 0.25
METRIC = "bal_acc" if TASK_TYPE == "cls" else "mse"

DATASET = "ada"
ALGORITHM = "lightgbm"
HYPERPARAMETERS = {
    "learning_rate": 0.05,
    "num_leaves": 64,
    "n_estimators": 200,
}


# ===== 2. Load and preprocess the dataset. =====
dataset_path, label_column, header, sep = get_dataset_info(TASK_TYPE, DATASET)

dm = DataManager()
df = dm.load_csv(dataset_path, header=header, sep=sep)
train_df, _ = dm.split_data(
    df,
    label_col=label_column,
    test_size=TEST_SIZE,
    random_state=SEED,
    task_type=task_type_id,
)
train_node = dm.from_train_df(train_df, label_col=label_column)
train_node = dm.preprocess_fit(train_node, task_type_id)


# ===== 3. Get and print the search space. =====
cs = get_hpo_cs(
    estimator_id=ALGORITHM,
    task_type=task_type_id,
    **get_cs_args(
        resampling_params={"test_size": VALID_SIZE, "shuffle": False},
        data_node=train_node,
    ),
)
print("\n[Search Space]")
print(cs)

# ===== 4. Build a valid Mindware configuration. =====
config_dict = cs.get_default_configuration().get_dictionary()
config_dict["algorithm"] = ALGORITHM

for name, value in HYPERPARAMETERS.items():
    full_name = name if ":" in name else f"{ALGORITHM}:{name}"
    config_dict[full_name] = value

config = Configuration(cs, values=config_dict)


# ===== 5. Evaluate the configuration on a validation split. =====
scorer = get_metric(METRIC)
evaluator_cls = CASHCLSEvaluator if TASK_TYPE == "cls" else CASHRGSEvaluator
evaluator = evaluator_cls(
    scorer=scorer,
    data_node=train_node,
    resampling_strategy="holdout",
    resampling_params={"test_size": VALID_SIZE, "shuffle": False},
    timestamp=time.time(),
    seed=SEED,
)

result = evaluator(config, save=False)


# ===== 6. Print the OpenBox objective. =====
# The objective is minimized: smaller is better.
# For accuracy-like metrics, Mindware returns the negative score.
print(json.dumps({
    "task_type": TASK_TYPE,
    "dataset": DATASET,
    "algorithm": ALGORITHM,
    "metric": METRIC,
    "objective": float(result["objectives"][0]),
    "objective_direction": "minimize",
    "config": config.get_dictionary(),
}, indent=2, ensure_ascii=False))
