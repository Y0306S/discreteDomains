import optuna

from mindware.components.optimizers.pt4opt_optuna_optimizer import PT4OptOptunaOptimizer
from mindware.components.optimizers.ml_evaluation_feedback import tpe_sampler_params


class TPEOptimizer(PT4OptOptunaOptimizer):
    optimizer_label = "tpe"

    def _build_sampler(self):
        params = dict(self.optimizer_params)
        params.setdefault("seed", self.seed)
        return optuna.samplers.TPESampler(**tpe_sampler_params(params))
