"""Two-deep checkpoint rotation."""

from __future__ import annotations

from pathlib import Path

from mindware.domains.protein.checkpoint_rotate import (
    checkpoint_candidates,
    commit_two_deep_checkpoint,
    prev_checkpoint_path,
)


def test_commit_two_deep_keeps_only_two(tmp_path: Path) -> None:
    latest = tmp_path / "vae_checkpoint.pt"
    for i in range(3):
        tmp = latest.with_name(latest.name + ".tmp")
        tmp.write_text(f"v{i}")
        commit_two_deep_checkpoint(tmp, latest)
    assert latest.read_text() == "v2"
    assert prev_checkpoint_path(latest).read_text() == "v1"
    assert sorted(p.name for p in tmp_path.glob("vae_checkpoint*")) == [
        "vae_checkpoint.prev.pt",
        "vae_checkpoint.pt",
    ]


def test_checkpoint_candidates_order(tmp_path: Path) -> None:
    latest = tmp_path / "ck.pt"
    latest.write_text("new")
    prev_checkpoint_path(latest).write_text("old")
    names = [p.name for p in checkpoint_candidates(latest)]
    assert names == ["ck.pt", "ck.prev.pt"]
