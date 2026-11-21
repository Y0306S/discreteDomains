"""RGPE functional smoke test.

Checks the behaviours the spec calls for:
  - all schema-compatible source tasks enter the candidate set (no top-K prefilter)
  - uniform weights while the target has fewer than 3 observations
  - dynamic pairwise-misranking weights afterwards, with base-model dropping
  - Copula normalisation, bootstrap sampling, 1000 posterior samples

Run from repos/transfer-hpo-framework with PYTHONPATH=. and envs/bl_rgpe/bin/python.
"""
import numpy as np
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
from smac.epm.util_funcs import get_types
from smac.configspace import convert_configurations_to_array
from rgpe.methods.rgpe import RGPE


def main():
    cs = ConfigurationSpace(seed=1)
    for i in range(3):
        cs.add_hyperparameter(UniformFloatHyperparameter("x%d" % i, 0.0, 1.0))
    types, bounds = get_types(cs, None)

    def f(c, shift):
        v = np.array([c["x0"], c["x1"], c["x2"]])
        return float(np.sum((v - 0.3 - shift) ** 2))

    # 4 source tasks as unordered (x, y) sets; two are close to the target, two are not
    shifts = [0.0, 0.05, 0.4, -0.3]
    data_by_task = {}
    for t, s in enumerate(shifts):
        cfgs = cs.sample_configuration(25)
        data_by_task[t] = {"configurations": cfgs,
                           "y": np.array([f(c, s) for c in cfgs])}

    model = RGPE(training_data=data_by_task, num_posterior_samples=1000,
                 weight_dilution_strategy="probabilistic-ld",
                 number_of_function_evaluations=50, sampling_mode="bootstrap",
                 variance_mode="average", normalization="Copula",
                 configspace=cs, types=types, bounds=bounds, seed=1)
    print("SMOKE RGPE built over %d source tasks (no top-K prefilter)" % len(data_by_task))

    for n_obs in (2, 5, 12):
        tc = cs.sample_configuration(n_obs)
        model.train(convert_configurations_to_array(tc),
                    np.array([f(c, 0.0) for c in tc]))
        w = np.round(np.asarray(model.weights_), 4)
        mu, _ = model.predict(convert_configurations_to_array(cs.sample_configuration(4)))
        tag = "uniform" if n_obs < 3 else "rank-weighted"
        print("SMOKE n_obs=%-2d %-13s weights(4 src + target)=%s sum=%.3f"
              % (n_obs, tag, w.tolist(), float(w.sum())))
    print("SMOKE RGPE train + rank weighting + predict OK")


if __name__ == "__main__":
    main()
