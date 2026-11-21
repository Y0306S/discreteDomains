import optuna

from mindware.components.optimizers.pt4opt_optuna_optimizer import PT4OptOptunaOptimizer


class CMAESOptimizer(PT4OptOptunaOptimizer):
    optimizer_label = "cmaes"

    # CmaEsSampler reconstructs its internal optimizer state from the study's
    # completed trials, so a point it never proposed perturbs that reconstruction.
    # Unlike TPE, which merely gains one KDE observation, so fallback evaluations
    # are recorded in the trajectory but not injected back into the study.
    supports_external_trial_injection = False

    def _build_sampler(self):
        params = dict(self.optimizer_params)
        params.setdefault("seed", self.seed)
        return optuna.samplers.CmaEsSampler(**params)
