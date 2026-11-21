"""VAE state round-trip in LOLBO checkpoints."""

from __future__ import annotations

import torch
from torch import nn

from mindware.domains.protein.bolt_lolbo_checkpoint import (
    restore_vae_to_objective,
    save_checkpoint,
    vae_state_dict_from_objective,
)


class _Obj:
    def __init__(self) -> None:
        self.vae = nn.Linear(4, 2)


def test_vae_state_dict_roundtrip(tmp_path):
    obj = _Obj()
    sd = vae_state_dict_from_objective(obj)
    assert sd is not None
    ckpt = save_checkpoint(
        tmp_path / "checkpoint.pt",
        train_x=["AAA"],
        train_y=torch.tensor([[1.0]]),
        train_z=torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
        train_c=None,
        num_calls=1,
        xs_to_scores_dict={"AAA": 1.0},
        n_init_loaded=1,
        tot_num_e2e_updates=0,
        best_x_seen="AAA",
        best_score_seen=1.0,
        minimize=False,
        vae_state_dict=sd,
    )
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "vae_state_dict" in payload
    obj2 = _Obj()
    with torch.no_grad():
        obj2.vae.weight.fill_(0.0)
    assert restore_vae_to_objective(obj2, payload["vae_state_dict"])
    assert torch.allclose(obj.vae.weight, obj2.vae.weight)
