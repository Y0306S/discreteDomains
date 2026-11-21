"""TaskSet CASH encoding for the same no-VAE LOLBO backend used by ML HPO."""

import torch
from ConfigSpace import Configuration
from ConfigSpace.conditions import EqualsCondition
from optuna.distributions import CategoricalDistribution

from mindware.components.optimizers.lolbo_sampler import LOLBOTurboSampler, _Codec
from mindware.domains.dlhpo.taskset_space import FAMILY_HP


class TaskSetCodec(_Codec):
    """Fixed union of family parameters; the family one-hot encodes activity.

    Inactive numeric coordinates are 0.5 and inactive categorical blocks are
    zero. They are canonicalized before posterior scoring, not just at decode.
    """

    def __init__(self, cs, search_space):
        super().__init__(search_space)
        self.cs = cs
        self.families = list(search_space[FAMILY_HP].choices)
        self.defaults = {hp.name: hp.default_value for hp in cs.get_hyperparameters()}
        self.parent_indices = {}
        for name in self.names:
            if name == FAMILY_HP:
                continue
            conditions = cs.get_parent_conditions_of(name)
            if (len(conditions) != 1 or not isinstance(conditions[0], EqualsCondition)
                    or conditions[0].parent.name != FAMILY_HP):
                raise ValueError(f"Unsupported TaskSet CASH condition for {name}")
            self.parent_indices[name] = self.families.index(conditions[0].value)

    def encode(self, params):
        family = self.families.index(params[FAMILY_HP])
        v = super().encode({**self.defaults, **params})
        for name, parent in self.parent_indices.items():
            if parent != family:
                v[self.slices[name]] = 0.0 if isinstance(self.dists[name], CategoricalDistribution) else 0.5
        return v

    def decode(self, v):
        values = super().decode(v)
        family = self.families.index(values[FAMILY_HP])
        active = {name: value for name, value in values.items()
                  if name == FAMILY_HP or self.parent_indices[name] == family}
        cfg = Configuration(self.cs, values=active)
        self.cs.check_configuration(cfg)
        return dict(cfg.get_dictionary())

    def project_candidates(self, candidates, center):
        candidates = candidates.clone()
        for start, stop in self.cat_blocks:
            choice = candidates[:, start:stop].argmax(dim=1)
            candidates[:, start:stop] = 0
            candidates[torch.arange(len(candidates), device=candidates.device), start + choice] = 1
        parent_slice = self.slices[FAMILY_HP]
        families = candidates[:, parent_slice].argmax(dim=1)
        center_family = int(center[parent_slice].argmax())
        for name, parent in self.parent_indices.items():
            sl = self.slices[name]
            active = families == parent
            categorical = isinstance(self.dists[name], CategoricalDistribution)
            # A newly activated family has no incumbent coordinates. Explore its
            # full domain instead of a trust region around inactive placeholders.
            if parent != center_family:
                count = int(active.sum())
                if categorical:
                    width = sl.stop - sl.start
                    block = candidates.new_zeros((count, width))
                    choices = torch.randint(width, (count,), device=candidates.device)
                    block[torch.arange(count, device=candidates.device), choices] = 1.0
                    candidates[active, sl] = block
                else:
                    candidates[active, sl] = torch.rand(
                        count, 1, dtype=candidates.dtype, device=candidates.device,
                    )
            candidates[~active, sl] = 0.0 if categorical else 0.5
        return candidates


class TaskSetLOLBOSampler(LOLBOTurboSampler):
    """CPU CASH adapter with checkpointed RNG, GP, trust region and pending batch."""

    def __init__(self, cs, search_space, **kwargs):
        device = kwargs.pop("device", "cpu")
        if str(device) != "cpu":
            raise ValueError("TaskSet LOLBO currently supports CPU only")
        super().__init__(device="cpu", **kwargs)
        if self.n_init < 2 or self.bsz < 1:
            raise ValueError("TaskSet LOLBO needs n_init >= 2 and bsz >= 1")
        if self.n_candidates is not None and self.n_candidates < self.bsz:
            raise ValueError("LOLBO n_candidates must be >= bsz")
        self.cs = cs
        self.fixed_space = dict(search_space)
        self._codec = TaskSetCodec(cs, self.fixed_space)
        self._torch_rng_state = torch.Generator().manual_seed(self.seed).get_state()
        self.n_model_updates = 0
        self.last_proposal_phase = "random_init"

    def infer_relative_search_space(self, study, trial):
        return self.fixed_space

    def _make_codec(self, search_space):
        return TaskSetCodec(self.cs, search_space)

    def _trial_has_params(self, trial, search_space):
        family = trial.params.get(FAMILY_HP)
        if family not in self._codec.families:
            return False
        parent = self._codec.families.index(family)
        return all(name in trial.params for name, index in self._codec.parent_indices.items()
                   if index == parent)

    def sample_relative(self, study, trial, search_space):
        n_before = self._n_seen
        # Torch's global RNG is not serialized by Optuna. Isolate and checkpoint
        # it so resumed GP fits and Thompson samples continue the same sequence.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self._torch_rng_state)
            result = super().sample_relative(study, trial, search_space)
            self._torch_rng_state = torch.get_rng_state()
        self.n_model_updates += int(self._n_seen > n_before)
        self.last_proposal_phase = "gp_trust_region" if result else "random_init"
        return result

    def diagnostics(self):
        return {
            "optimizer_backend": "lolbo_no_vae_cash",
            "proposal_phase": self.last_proposal_phase,
            "surrogate_fit_count": self.n_model_updates,
            "modeled_parameters": len(self.fixed_space),
            "encoded_dimension": self._codec.dim,
            "tr_length": float(self._tr_state.length) if self._tr_state is not None else None,
        }
