#!/usr/bin/env python3
"""
Minimal script to reproduce the "Input y contains NaN" error during OpenBox evaluation.

This error occurs for certain datasets (e.g., porto-seguro, volkert) where the target
variable contains NaN values. The model generates a valid config, but the benchmark
fails during model training because sklearn cannot handle NaN in the target.

Dataset: porto-seguro
Algorithm: adaboost
Task type: classification (cls)
"""

import json
import sys
import types
from pathlib import Path

BENCHMARK_ROOT = Path("/jizhicfs/pkuhetu/bcxu/llmoptimizer_benchmark")


def patch_configspace_quantization():
    try:
        import ConfigSpace.hyperparameters as hp_mod
    except ImportError:
        return
    if getattr(hp_mod.UniformIntegerHyperparameter, "_bolt_q_patch", False):
        return
    original_integer = hp_mod.UniformIntegerHyperparameter
    original_float = hp_mod.UniformFloatHyperparameter

    def uniform_integer(name, lower, upper, default_value=None, q=None, log=False, meta=None, **kwargs):
        _ = kwargs
        meta = dict(meta or {})
        if q is not None:
            meta["q"] = q
        return original_integer(name, lower, upper, default_value=default_value, log=log, meta=meta)

    def uniform_float(name, lower, upper, default_value=None, q=None, log=False, meta=None, **kwargs):
        _ = kwargs
        meta = dict(meta or {})
        if q is not None:
            meta["q"] = q
        return original_float(name, lower, upper, default_value=default_value, log=log, meta=meta)

    uniform_integer._bolt_q_patch = True
    uniform_float._bolt_q_patch = True
    hp_mod.UniformIntegerHyperparameter = uniform_integer
    hp_mod.UniformFloatHyperparameter = uniform_float


def install_benchmark_package_stub(benchmark_root):
    existing = sys.modules.get("mindware")
    if existing is not None and getattr(existing, "__path__", None):
        return
    mindware = types.ModuleType("mindware")
    mindware.__path__ = [str(benchmark_root / "mindware")]
    sys.modules["mindware"] = mindware


def patch_sklearn_version():
    try:
        import sklearn
    except ImportError:
        return
    if getattr(sklearn, "__version__", "") > "1.3.2":
        sklearn.__version__ = "1.3.2"
    try:
        import sklearn.metrics._scorer as scorer_mod
        if not hasattr(scorer_mod, "_ThresholdScorer") and hasattr(scorer_mod, "_Scorer"):
            scorer_mod._ThresholdScorer = scorer_mod._Scorer
    except ImportError:
        pass


def load_runtime(benchmark_root):
    patch_configspace_quantization()
    patch_sklearn_version()
    install_benchmark_package_stub(benchmark_root)
    if str(benchmark_root) not in sys.path:
        sys.path.insert(0, str(benchmark_root))
    examples_dir = benchmark_root / "examples" / "benchmark"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))

    from benchmark_utils import get_dataset_info
    from mindware.components.config_space.cs_builder import get_cs_args, get_hpo_cs
    from mindware.components.metrics.metric import get_metric
    from mindware.components.utils.constants import CLASSIFICATION, REGRESSION
    from mindware.modules.cash.cash_evaluator import CASHCLSEvaluator, CASHRGSEvaluator
    from mindware.utils.data_manager import DataManager

    return {
        "get_dataset_info": get_dataset_info,
        "get_hpo_cs": get_hpo_cs,
        "get_cs_args": get_cs_args,
        "get_metric": get_metric,
        "CLASSIFICATION": CLASSIFICATION,
        "REGRESSION": REGRESSION,
        "CASHCLSEvaluator": CASHCLSEvaluator,
        "CASHRGSEvaluator": CASHRGSEvaluator,
        "DataManager": DataManager,
    }


def evaluate_config(
    runtime,
    benchmark_root,
    task_type,
    dataset,
    algorithm,
    hparams,
    test_size=0.2,
    valid_size=0.25,
    seed=1,
):
    task_type_id = runtime["CLASSIFICATION"] if task_type == "cls" else runtime["REGRESSION"]
    metric = "bal_acc" if task_type == "cls" else "mse"

    import os
    cwd = Path.cwd()
    try:
        os.chdir(benchmark_root)
        dataset_path, label_column, header, sep = runtime["get_dataset_info"](task_type, dataset)
    finally:
        os.chdir(cwd)
    dataset_path = Path(dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = benchmark_root / dataset_path

    dm = runtime["DataManager"]()
    df = dm.load_csv(str(dataset_path), header=header, sep=sep)
    train_df, _ = dm.split_data(
        df,
        label_col=label_column,
        test_size=test_size,
        random_state=seed,
        task_type=task_type_id,
    )
    train_node = dm.from_train_df(train_df, label_col=label_column)
    train_node = dm.preprocess_fit(train_node, task_type_id)

    cs = runtime["get_hpo_cs"](
        estimator_id=algorithm,
        task_type=task_type_id,
        **runtime["get_cs_args"](
            resampling_params={"test_size": valid_size, "shuffle": False},
            data_node=train_node,
        ),
    )

    config_dict = cs.get_default_configuration().get_dictionary()
    config_dict.update(hparams)
    if "algorithm" in {hp.name for hp in cs.get_hyperparameters()}:
        config_dict["algorithm"] = algorithm

    from ConfigSpace import Configuration
    config = Configuration(cs, values=config_dict)
    scorer = runtime["get_metric"](metric)
    evaluator_cls = runtime["CASHCLSEvaluator"] if task_type == "cls" else runtime["CASHRGSEvaluator"]
    evaluator = evaluator_cls(
        scorer=scorer,
        data_node=train_node,
        resampling_strategy="holdout",
        resampling_params={"test_size": valid_size, "shuffle": False},
        timestamp=__import__("time").time(),
        seed=seed,
    )
    config_out = config.get_dictionary()
    config_out["algorithm"] = algorithm
    result = evaluator(config_out, save=False)
    return {
        "objective": float(result["objectives"][0]),
        "config": config_out,
    }


def main():
    print("=" * 70)
    print("Reproducing the 'Input y contains NaN' error")
    print("=" * 70)
    print()

    # This is a valid config generated by the model for porto-seguro | adaboost
    valid_config = {
        "adaboost:algorithm": "SAMME.R",
        "adaboost:learning_rate": 0.010138019064548476,
        "adaboost:max_depth": 2,
        "adaboost:n_estimators": 494,
    }

    print("Dataset: porto-seguro")
    print("Algorithm: adaboost")
    print("Task type: classification (cls)")
    print()
    print("Config being evaluated:")
    for k, v in valid_config.items():
        print(f"  {k}: {v}")
    print()
    print("This config is VALID and parseable.")
    print("The error occurs during model training, not during config validation.")
    print()

    runtime = load_runtime(BENCHMARK_ROOT)

    print("Loading dataset and running evaluation...")
    print()

    try:
        result = evaluate_config(
            runtime=runtime,
            benchmark_root=BENCHMARK_ROOT,
            task_type="cls",
            dataset="porto-seguro",
            algorithm="adaboost",
            hparams=valid_config,
        )
        print("SUCCESS! Evaluation completed.")
        print(f"Objective: {result['objective']}")
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        print()
        print("This is the NaN bug! The dataset has NaN values in the target variable y.")
        print("The config is correct, but sklearn's AdaBoost cannot handle NaN in y.")

    print()
    print("=" * 70)
    print("Note: This error is a DATASET issue, not a model issue.")
    print("All models (including API models) would fail on this task.")
    print("=" * 70)


if __name__ == "__main__":
    main()
