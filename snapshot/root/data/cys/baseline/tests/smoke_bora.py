"""BORA functional smoke test: the three scheduled actions and the pure-BO backend.

The LLM-driven actions (ASSISTANT_SUGGESTS / ASSISTANT_SELECTS) need an
OpenAI-compatible endpoint, so this test exercises the BO arm and confirms the
policy's action set matches the spec ("pure BO / LLM proposes / LLM re-ranks").

Run from repos/bora-the-explorer with PYTHONPATH=. and envs/bl_bora/bin/python.
"""
import numpy as np
from bora.policy import Action
from bora.target_space import TargetSpace
from bora.acquisition import UpperConfidenceBound, ExpectedImprovement
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


def main():
    names = [a.name for a in Action]
    assert set(names) == {"BO", "ASSISTANT_SUGGESTS", "ASSISTANT_SELECTS"}, names
    print("SMOKE BORA actions (BO / LLM-proposes / LLM-reranks):", names)

    f = lambda x, y: -((x - 0.3) ** 2 + (y + 0.2) ** 2)
    space = TargetSpace(target_func=None, pbounds={'x': (-1., 1.), 'y': (-1., 1.)},
                        key_order=['x', 'y'], random_state=0)
    for p in space.random_points(14):
        space.register(params=p, target=f(*p))
    print("SMOKE TargetSpace: %d obs | dim %d | best %.4f"
          % (len(space), space.dim, space.target.max()))

    gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), normalize_y=True, random_state=0)
    gp.fit(space.params, space.target)
    for name, acq in [("UCB", UpperConfidenceBound(kappa=2.576, random_state=0)),
                      ("EI", ExpectedImprovement(xi=0.01, random_state=0))]:
        s = np.asarray(acq.suggest(gp=gp, target_space=space, fit_gp=False), dtype=float)
        print("SMOKE %-3s suggest -> %s" % (name, np.round(s.ravel(), 4).tolist()))
    print("SMOKE BORA BO backend OK")


if __name__ == "__main__":
    main()
