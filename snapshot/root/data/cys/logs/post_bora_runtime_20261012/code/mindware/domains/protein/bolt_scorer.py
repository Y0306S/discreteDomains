"""BOLT seed_0 peptide oracle: bacteria_0 MIC + template similarity constraint."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

from mindware.domains.paths import BOLT_PEPTIDES
from mindware.domains.protein.bolt_constraints import (
    bolt_reference_sequence,
    ensure_one_cysteine,
    is_feasible as bolt_is_feasible,
    template_similarity,
)

BOLT_APEX = BOLT_PEPTIDES / "apex_oracle"


class BoltBacteria0Scorer:
    """Minimize A. baumannii predicted MIC (BOLT bacteria_0), with BOLT constraints."""

    def __init__(
        self,
        template: Optional[str] = None,
        *,
        constraint_index: int = 0,
        min_similarity: float = 0.75,
        apex_dir: Optional[Path] = None,
    ):
        self.constraint_index = int(constraint_index)
        self.template = str(template or bolt_reference_sequence(constraint_index=self.constraint_index))
        self.min_similarity = float(min_similarity)
        self.apex_dir = Path(apex_dir) if apex_dir is not None else BOLT_APEX
        self._fn = None
        self._cache: Dict[str, Tuple[float, bool]] = {}

    def _load(self) -> None:
        if self._fn is not None:
            return
        apex = str(self.apex_dir)
        if apex not in sys.path:
            sys.path.insert(0, apex)
        import torch

        orig = torch.load

        def _load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            kwargs.setdefault("map_location", "cpu")
            return orig(*args, **kwargs)

        torch.load = _load  # type: ignore[method-assign]
        try:
            from APEX_predict import apex_wrapper  # type: ignore
        finally:
            torch.load = orig  # type: ignore[method-assign]
        self._fn = apex_wrapper

    def feasible(self, seq: str) -> bool:
        seq = ensure_one_cysteine(str(seq or "").strip()[:150])
        return bolt_is_feasible(seq, self.template, min_similarity=self.min_similarity)

    def __call__(self, seq: str) -> Tuple[float, bool]:
        key = ensure_one_cysteine(str(seq or "").strip()[:150])
        if not key:
            return 1e6, False
        if key in self._cache:
            return self._cache[key]
        if not self.feasible(key):
            out = (1e6, False)
            self._cache[key] = out
            return out
        self._load()
        mic_row = self._fn([key])[0]
        mic0 = float(mic_row[0])
        if mic0 != mic0:
            out = (1e6, False)
        else:
            out = (mic0, True)
        self._cache[key] = out
        return out
