"""Exercise a trained transfer worker through four small sklearn target evaluations."""

import argparse
import tempfile
from unittest.mock import patch

import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from mindware.components.optimizers import pt4opt_transfer_optimizer as transfer
from mindware.components.optimizers.transfer_hpo.schema import build_hpo_cs
from test_transfer_hpo import MemoryKD


class RegressionEvaluator:
    fixed_config = None

    def __init__(self):
        x, y = make_regression(n_samples=160, n_features=6, noise=1.0, random_state=123)
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(x, y, random_state=123)

    def __call__(self, config):
        params = {key.split(":", 1)[1]: value for key, value in dict(config).items() if ":" in key}
        model = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=int(params.pop("max_depth"))),
                                  random_state=123, **params)
        model.fit(self.train_x, self.train_y)
        return mean_squared_error(self.test_y, model.predict(self.test_x))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("rgpe", "fsbo"), default="rgpe")
    args = parser.parse_args()
    optimizer_type = transfer.RGPEOptimizer if args.method == "rgpe" else transfer.FSBOOptimizer
    expected_note = "RGPE(TAF)" if args.method == "rgpe" else "FSBO paper deep-kernel EI"
    with tempfile.TemporaryDirectory(prefix=args.method + "_ml_test_") as directory:
        kd = MemoryKD()
        with patch.object(transfer.KnowledgeDatabase, "instance", return_value=kd):
            optimizer = optimizer_type(
                RegressionEvaluator(), build_hpo_cs("rgs", "adaboost"), "hpo", "test",
                evaluation_limit=4, inner_iter_num_per_iter=3, per_run_time_limit=60,
                output_dir=directory, seed=1,
                optimizer_params={"task_type": "rgs", "n_startup_trials": 3, "n_pending": 16})
        optimizer.run()
        assert optimizer.trial_cnt == 4
        assert np.isfinite(optimizer.perfs).all()
        assert optimizer.traj_rows[-1]["note"] == expected_note
        print("PASSED four real sklearn target fits, last selected by", args.method,
              "; best MSE =", -optimizer.incumbent_perf)


if __name__ == "__main__":
    main()
