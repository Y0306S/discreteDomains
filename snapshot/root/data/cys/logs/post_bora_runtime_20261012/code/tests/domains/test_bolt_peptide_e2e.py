"""Tests for peptide E2E VAE helpers."""

from __future__ import annotations

import torch
from torch import nn

from mindware.domains.history import observation
from mindware.domains.protein.bolt_peptide_e2e import (
    capture_vae_state_dict,
    load_peptide_vae_checkpoint,
    reencode_lolbo_state_latents,
    refit_lolbo_surrogate_on_all,
    save_peptide_vae_checkpoint,
    sequences_for_peptide_e2e,
)


class _Wrap:
    def __init__(self) -> None:
        self.vae = nn.Linear(4, 2)


def test_sequences_for_peptide_e2e_recent_and_top():
    obs = []
    for i in range(5):
        obs.append(
            observation(
                {"z_0": float(i)},
                float(10 - i),
                extra={"artifact": f"SEQ{i}", "info": {"sequence": f"SEQ{i}"}},
            )
        )
    seqs = sequences_for_peptide_e2e(obs, recent=2, top_k=2)
    assert seqs == ["SEQ3", "SEQ4"]


def test_peptide_vae_checkpoint_roundtrip(tmp_path):
    wrap = _Wrap()
    path = tmp_path / "vae_checkpoint.pt"
    sd = capture_vae_state_dict(wrap)
    assert sd is not None
    save_peptide_vae_checkpoint(wrap, path, e2e_updates=2, n_observations=42)
    save_peptide_vae_checkpoint(wrap, path, e2e_updates=3, n_observations=43)
    assert (tmp_path / "vae_checkpoint.prev.pt").is_file()
    wrap2 = _Wrap()
    with torch.no_grad():
        wrap2.vae.weight.zero_()
    meta = load_peptide_vae_checkpoint(wrap2, path)
    assert meta["vae_restored"] is True
    assert meta["e2e_updates"] == 3
    assert meta["n_observations"] == 43
    assert torch.allclose(wrap.vae.weight, wrap2.vae.weight)


class _FakeObj:
    def __init__(self) -> None:
        self.num_calls = 0

    def vae_forward(self, xs):
        z = torch.stack([torch.full((4,), float(len(str(x)))) for x in xs], dim=0)
        return z, None


class _FakeState:
    def __init__(self) -> None:
        self.train_x = ["AAA", "CCCC"]
        self.train_y = torch.tensor([[1.0], [2.0]])
        self.train_z = torch.zeros(2, 4)
        self.top_k_xs = ["CCCC"]
        self.top_k_zs = [self.train_z[1].unsqueeze(-2)]
        self.objective = _FakeObj()
        self.num_update_epochs = 2
        self.init_n_epochs = 20
        self.initial_model_training_complete = True
        self.refit_called = False

    def update_surrogate_model(self):
        self.refit_called = True
        self.initial_model_training_complete = True


def test_reencode_lolbo_state_latents_keeps_x_y_and_length():
    state = _FakeState()
    old_y = state.train_y.clone()
    n = reencode_lolbo_state_latents(state)
    assert n == 2
    assert state.train_x == ["AAA", "CCCC"]
    assert torch.equal(state.train_y, old_y)
    assert state.train_z.shape == (2, 4)
    assert float(state.train_z[0, 0]) == 3.0
    assert float(state.train_z[1, 0]) == 4.0
    assert state.objective.num_calls == 0
    assert len(state.top_k_zs) == 1
    assert float(state.top_k_zs[0].reshape(-1)[0]) == 4.0


def test_refit_lolbo_surrogate_on_all_uses_fast_epochs():
    state = _FakeState()
    refit_lolbo_surrogate_on_all(state)
    assert state.refit_called is True
    assert state.init_n_epochs == 20
