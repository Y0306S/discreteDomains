"""LOL-BO functional smoke test: trust-region DKL-GP loop on a toy latent objective.

Run from repos/lolbo with PYTHONPATH=. and envs/bl_lolbo_exact/bin/python.
"""
import torch, torch.nn as nn, numpy as np, gpytorch, botorch
from lolbo.lolbo import LOLBOState
from lolbo.latent_space_objective import LatentSpaceObjective


class IdVAE(nn.Module):
    """Identity 8-d 'VAE' with one trainable layer so the joint update has gradients."""
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, z):
        return z


class ToyObj(LatentSpaceObjective):
    def __init__(self):
        super().__init__(num_calls=0, task_id='toy')

    def initialize_vae(self):
        self.vae = IdVAE()

    def initialize_objective_func(self):
        self.dim = 8

    def vae_decode(self, z):
        return [tuple(round(v, 6) for v in row.tolist()) for row in z]

    def vae_forward(self, xs_batch):
        z = torch.tensor(np.array(xs_batch), dtype=torch.float32)
        return z, (self.vae.lin(z) ** 2).mean()

    def query_oracle(self, x):
        return -float(np.sum(np.square(np.array(x))))


def main():
    torch.manual_seed(0)
    obj = ToyObj()
    init_z = torch.randn(30, 8)
    init_x = obj.vae_decode(init_z)
    init_y = torch.tensor([obj.query_oracle(x) for x in init_x],
                          dtype=torch.float32).unsqueeze(-1)
    state = LOLBOState(objective=obj, train_x=init_x, train_y=init_y, train_z=init_z,
                       k=10, minimize=False, num_update_epochs=1, init_n_epochs=2,
                       bsz=5, verbose=False)
    best0 = float(state.best_score_seen)
    for _ in range(5):
        state.update_surrogate_model()
        state.acquisition()
        if state.tr_state.restart_triggered:
            state.initialize_tr_state()
    print("SMOKE LOL-BO imports: torch %s gpytorch %s botorch %s"
          % (torch.__version__, gpytorch.__version__, botorch.__version__))
    print("SMOKE LOL-BO 5 acquisition rounds OK")
    print("SMOKE best %.4f -> %.4f | oracle calls %d | train_z %s"
          % (best0, float(state.best_score_seen), obj.num_calls, tuple(state.train_z.shape)))


if __name__ == "__main__":
    main()
