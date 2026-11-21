"""Tests for peptide E2E VAE helpers."""

from __future__ import annotations

import torch
from torch import nn

from mindware.domains.history import observation
from mindware.domains.protein.bolt_peptide_e2e import (
    capture_vae_state_dict,
    load_peptide_vae_checkpoint,
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
