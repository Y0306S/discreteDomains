import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmark_utils import get_dataset_info
from mindware import CASH
from mindware.components.metrics.metric import get_metric
from mindware.components.utils.constants import CLASSIFICATION, REGRESSION
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase
from mindware.modules.configs import CONFIGS
from mindware.utils.data_manager import DataManager


CLS_ALGORITHMS = [
    "adaboost",
    "catboost",
    "extra_trees",
    "gradient_boosting",
    "k_nearest_neighbors",
    "lightgbm",
    "random_forest",
    "xgboost",
]

RGS_ALGORITHMS = [
    "adaboost",
    "catboost",
    "extra_trees",
    "gradient_boosting",
    "k_nearest_neighbors",
    "lightgbm",
    "random_forest",
    "xgboost",
]


def load_preprocessed_data(args, task_type_const):
    dataset_path, label_column, header, sep = get_dataset_info(args.task_type, args.dataset)
    print(f"[Load] {dataset_path}")

    dm = DataManager()
    df = dm.load_csv(
        dataset_path,
        header=header,
        sep=sep,
    )
    train_df, test_df = dm.split_data(
        df,
        label_col=label_column,
        test_size=args.test_size,
        random_state=args.seed,
        task_type=task_type_const,
    )

    train_node = dm.from_train_df(train_df, label_col=label_column)
    train_node = dm.preprocess_fit(train_node, task_type_const)

    test_node = dm.from_test_df(test_df, has_label=True)
    test_node = dm.preprocess_transform(test_node)
    return dm, train_node, test_node


def save_history_json(history_dir: Path, task_type: str, metric: str, dataset: str, algorithm: str):
    kd = KnowledgeDatabase.instance()
    history = kd.bo_history_mab.get(algorithm)
    if history is None:
        raise RuntimeError(f"No BO history found for algorithm '{algorithm}'.")

    target_dir = history_dir / task_type / metric / dataset
    target_dir.mkdir(parents=True, exist_ok=True)
    history_path = target_dir / f"{algorithm}.json"
    history.save_json(str(history_path))
    print(f"[History] saved {history_path}")
    return history_path


def run_one_algorithm(args, algorithm: str, dm, train_node, test_node, task_type_const, metric: str):
    print(f"\n[HPO] dataset={args.dataset}, algorithm={algorithm}, metric={metric}")

    # Single-algorithm CASH is equivalent to HPO here, and it initializes
    # KnowledgeDatabase for the modified SMAC optimizer in this repository.
    opt = CASH(
        include_algorithms=[algorithm],
        task_type=task_type_const,
        metric=metric,
        data_node=train_node,
        evaluation=args.evaluation,
        resampling_params={"test_size": args.valid_size, "shuffle": False},
        optimizer="smac",
        time_limit=args.time_limit,
        amount_of_resource=args.epoch,
        per_run_time_limit=args.per_run_time_limit,
        inner_iter_num_per_iter=1,
        output_dir=args.output_dir,
        seed=args.seed,
        n_jobs=1,
        task_id=args.dataset,
        topk=args.epoch,
    )
    print(opt.get_conf(save=True))

    best_perf, stats_path = opt.run(refit=args.refit)
    print(f"[HPO] best validation score: {best_perf:.6f}")
    print(f"[HPO] top-k stats: {stats_path}")

    scorer = get_metric(metric)
    pred = opt.predict(test_node, refit=args.refit, ens=False)
    test_perf = scorer._score_func(test_node.data[1], pred) * scorer._sign
    print(f"[HPO] test score: {test_perf:.6f}")

    history_path = save_history_json(
        Path(args.history_dir),
        args.task_type,
        metric,
        args.dataset,
        algorithm,
    )
    opt.get_model_info(
        save=True,
        extra_info={
            "test_perf": float(test_perf),
            "stats_path": stats_path,
            "history_path": str(history_path),
        },
    )
    return best_perf, test_perf, history_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run single-algorithm HPO and save an OpenBox History JSON compatible with history_data/."
    )
    parser.add_argument("--task_type", choices=["cls", "rgs"], default="cls")
    parser.add_argument("--dataset", default="ada")
    parser.add_argument(
        "--algorithm",
        nargs="+",
        default=["lightgbm"],
        help="One or more algorithms, or 'all'.",
    )
    parser.add_argument("--history_dir", default="./history_data")
    parser.add_argument("--output_dir", default="./benchmark_data/hpo_example")
    parser.add_argument("--metric", default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--valid_size", type=float, default=0.25)
    parser.add_argument("--evaluation", choices=["holdout", "cv"], default="holdout")
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--time_limit", type=int, default=3600)
    parser.add_argument("--per_run_time_limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--refit", choices=["partial", "full", "cv"], default="partial")
    return parser.parse_args()


def main():
    args = parse_args()
    task_type_const = CLASSIFICATION if args.task_type == "cls" else REGRESSION
    metric = args.metric or ("bal_acc" if args.task_type == "cls" else "mse")

    history_dir = Path(args.history_dir).resolve()
    CONFIGS.ORIGINAL_DATA_DIR = str(history_dir)

    algorithms = args.algorithm
    if len(algorithms) == 1 and algorithms[0] == "all":
        algorithms = CLS_ALGORITHMS if args.task_type == "cls" else RGS_ALGORITHMS

    dm, train_node, test_node = load_preprocessed_data(args, task_type_const)

    results = []
    for algorithm in algorithms:
        result = run_one_algorithm(
            args=args,
            algorithm=algorithm,
            dm=dm,
            train_node=train_node,
            test_node=test_node,
            task_type_const=task_type_const,
            metric=metric,
        )
        results.append((algorithm, *result))

    print("\n[Summary]")
    for algorithm, best_perf, test_perf, history_path in results:
        print(
            f"{algorithm}: valid_best={best_perf:.6f}, "
            f"test={test_perf:.6f}, history={history_path}"
        )


if __name__ == "__main__":
    main()
