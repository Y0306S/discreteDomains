from mindware.components.optimizers.lolbo_sampler import LOLBOTurboSampler
from mindware.components.optimizers.pt4opt_optuna_optimizer import PT4OptOptunaOptimizer


class LOLBOOptimizer(PT4OptOptunaOptimizer):
    """LOL-BO baseline for ML / DL HPO (no VAE), per pt4opt-baselines.md.

    Ported from nataliemaus/lolbo -- see lolbo_turbo_port.py for provenance.
    """

    optimizer_label = "lolbo"

    # The trust-region state and GP are rebuilt from the study's completed trials,
    # so a point the sampler never proposed would perturb that reconstruction --
    # same reasoning as CMAESOptimizer.
    supports_external_trial_injection = False

    def _build_sampler(self):
        params = dict(self.optimizer_params)
        params.setdefault("seed", self.seed)
        return LOLBOTurboSampler(**params)
