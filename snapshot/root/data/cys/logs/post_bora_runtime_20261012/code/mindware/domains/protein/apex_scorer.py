"""Lazy APEX ensemble MIC scorer (BOLT apex_oracle/APEX_predict.py)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

from mindware.domains.paths import BOLT_PEPTIDES


class ApexMeanMIC:
    """Minimize mean predicted MIC (µM) across the APEX pathogen ensemble."""

    def __init__(self, apex_dir: Optional[Path] = None):
        self.apex_dir = Path(apex_dir) if apex_dir is not None else (BOLT_PEPTIDES / "apex_oracle")
        self._fn = None
        self._cache: Dict[str, float] = {}

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
            from APEX_predict import apex_mean_wrapper  # type: ignore
        finally:
            torch.load = orig  # type: ignore[method-assign]
        self._fn = apex_mean_wrapper

    def __call__(self, seq: str) -> Tuple[float, bool]:
        key = str(seq or "").strip()
        if not key:
            return 1e6, False
        if key in self._cache:
            return self._cache[key], True
        self._load()
        mic = float(self._fn([key])[0])  # type: ignore[misc]
        if mic != mic:  # NaN
            return 1e6, False
        self._cache[key] = mic
        return mic, True
