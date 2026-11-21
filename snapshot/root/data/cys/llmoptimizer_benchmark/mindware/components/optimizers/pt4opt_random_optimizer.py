import optuna

from mindware.components.optimizers.pt4opt_optuna_optimizer import PT4OptOptunaOptimizer


class RandomSearchBaselineOptimizer(PT4OptOptunaOptimizer):
    """Random Search baseline: uniform sampling in the task's own search space.

    Implemented on the same Optuna ask/tell path as TPEOptimizer / CMAESOptimizer
    rather than via the legacy openbox-based ``RandomSearchOptimizer`` (which is
    unreachable -- it never forwards ``task_id`` to BaseOptimizer, so it raises
    TypeError on construction). Sharing this path means Random Search and TPE
    differ only in the sampler, which is what makes them comparable.
    """

    optimizer_label = "random_search"

    def _build_sampler(self):
        params = dict(self.optimizer_params)
        params.setdefault("seed", self.seed)
        return optuna.samplers.RandomSampler(**params)
