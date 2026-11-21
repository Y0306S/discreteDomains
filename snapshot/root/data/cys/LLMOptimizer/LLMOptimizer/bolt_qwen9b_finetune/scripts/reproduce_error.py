#!/usr/bin/env python3
"""
Minimal script to reproduce the "illegal value 'poisson'" error.

This demonstrates the bug where the prompt includes 'poisson' as a valid
criterion choice, but the benchmark's ConfigSpace rejects it for certain
datasets (e.g., QSAR-TID-11).
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
    from mindware.components.utils.constants import REGRESSION
    from mindware.utils.data_manager import DataManager

    return {
        "get_dataset_info": get_dataset_info,
        "get_hpo_cs": get_hpo_cs,
        "get_cs_args": get_cs_args,
        "REGRESSION": REGRESSION,
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
    task_type_id = runtime["REGRESSION"]

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

    # Print ConfigSpace criterion choices
    for hp in cs.get_hyperparameters():
        if "criterion" in getattr(hp, "name", ""):
            print(f"  ConfigSpace criterion choices: {getattr(hp, 'choices', 'N/A')}")

    config_dict = cs.get_default_configuration().get_dictionary()
    config_dict.update(hparams)
    if "algorithm" in {hp.name for hp in cs.get_hyperparameters()}:
        config_dict["algorithm"] = algorithm

    from ConfigSpace import Configuration
    config = Configuration(cs, values=config_dict)
    return config.get_dictionary()


def main():
    print("=" * 70)
    print("Reproducing the 'illegal value poisson' error")
    print("=" * 70)
    print()

    # This is the config generated by the baseline model for QSAR-TID-11 | random_forest
    # The prompt told the model that 'poisson' is a valid choice,
    # but the ConfigSpace for this dataset does NOT include it.
    bad_config = {
        "random_forest:bootstrap": "False",
        "random_forest:criterion": "poisson",  # <-- This causes the error!
        "random_forest:max_depth": "None",
        "random_forest:max_features": 0.8308089382760484,
        "random_forest:max_leaf_nodes": "None",
        "random_forest:min_impurity_decrease": 0.0,
        "random_forest:min_samples_leaf": 1,
        "random_forest:min_samples_split": 2,
        "random_forest:min_weight_fraction_leaf": 0.0,
    }

    print("Dataset: QSAR-TID-11")
    print("Algorithm: random_forest")
    print("Task type: regression (rgs)")
    print()
    print("Config being evaluated:")
    for k, v in bad_config.items():
        print(f"  {k}: {v}")
    print()
    print("The prompt told the model:")
    print("  random_forest:criterion: categorical, choices=['squared_error', 'absolute_error', 'friedman_mse', 'poisson']")
    print()

    runtime = load_runtime(BENCHMARK_ROOT)

    print("Loading dataset and building ConfigSpace...")
    print()

    try:
        result = evaluate_config(
            runtime=runtime,
            benchmark_root=BENCHMARK_ROOT,
            task_type="rgs",
            dataset="QSAR-TID-11",
            algorithm="random_forest",
            hparams=bad_config,
        )
        print("SUCCESS! Config was accepted.")
        print(f"Result: {json.dumps(result, indent=2)}")
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        print()
        print("This is the bug! The prompt says 'poisson' is valid,")
        print("but the ConfigSpace rejects it for this dataset.")

    print()
    print("=" * 70)
    print("Now testing with a VALID config (squared_error)")
    print("=" * 70)
    print()

    good_config = dict(bad_config)
    good_config["random_forest:criterion"] = "squared_error"

    try:
        result = evaluate_config(
            runtime=runtime,
            benchmark_root=BENCHMARK_ROOT,
            task_type="rgs",
            dataset="QSAR-TID-11",
            algorithm="random_forest",
            hparams=good_config,
        )
        print("SUCCESS! Config was accepted.")
        print(f"Result: {json.dumps(result, indent=2)}")
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
