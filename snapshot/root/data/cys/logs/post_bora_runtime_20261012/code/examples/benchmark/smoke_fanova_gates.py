#!/usr/bin/env python3
"""Smoke: 1D proxy vs sklearn RF vs full fANOVA worker. No HPO launch."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("FANOVA_PYTHON", "/root/venvs/fanova/bin/python")


def _load_gate():
    import importlib.util

    path = ROOT / "mindware" / "components" / "optimizers" / "experience_gate.py"
    spec = importlib.util.spec_from_file_location("experience_gate", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["experience_gate"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def synthetic_run(n: int = 40) -> list[dict]:
    import numpy as np

    rng = np.random.RandomState(0)
    rows = []
    for i in range(n):
        lr = float(rng.uniform(0.01, 0.5))
        depth = int(rng.randint(2, 13))
        l2 = float(rng.uniform(1.0, 10.0))
        bag = float(rng.uniform(0.5, 1.0))
        score = 0.90 - 1.2 * lr + 0.002 * depth + 0.01 * rng.randn()
        rows.append(
            {
                "trial": i + 1,
                "score": float(score),
                "config": {
                    "algorithm": "catboost",
                    "learning_rate": lr,
                    "depth": depth,
                    "l2_leaf_reg": l2,
                    "bagging_temperature": bag,
                },
            }
        )
    return rows


def test_proxy_and_rf() -> None:
    gate = _load_gate()
    filter_fanova = gate.filter_fanova
    hp_importances_proxy = gate.hp_importances_proxy
    hp_importances_rf = gate.hp_importances_rf

    rows = synthetic_run()
    proxy = hp_importances_proxy(rows)
    rf = hp_importances_rf(rows)
    assert proxy["learning_rate"] > proxy["l2_leaf_reg"]
    assert rf["learning_rate"] == max(rf.values())
    block = "\n".join(
        [
            "- learning_rate: prefer ~ [0.01, 0.08] over [0.3, 0.5]",
            "- l2_leaf_reg: prefer ~ [2, 4] over [8, 10]",
            "- bagging_temperature: avoid ~ [0.5, 0.6]",
        ]
    )
    filtered = filter_fanova(block, rows, 0.10, engine="rf")
    assert "learning_rate" in filtered
    assert "bagging_temperature" not in filtered or rf.get("bagging_temperature", 0) >= 0.10
    print("ok proxy/rf", {k: round(v, 3) for k, v in proxy.items()}, {k: round(v, 3) for k, v in rf.items()})


def test_full_worker() -> None:
    gate = _load_gate()
    default_fanova_python = gate.default_fanova_python
    hp_importances_full = gate.hp_importances_full

    py = default_fanova_python()
    if not py:
        print("skip full fANOVA: no FANOVA_PYTHON")
        return
    rows = synthetic_run()
    imps, used = hp_importances_full(rows)
    print("ok full", used, {k: round(v, 3) for k, v in imps.items()})
    assert used in ("fanova", "rf_fallback", "proxy_fallback", "proxy_too_few")
    if used == "fanova":
        assert imps["learning_rate"] == max(imps.values())


def test_tpe_xor() -> None:
    gate = _load_gate()
    block = "\n".join(
        [
            "- learning_rate: prefer ~{\"low\": 0.05, \"high\": 0.12}",
            "- depth: prefer 6 over 12",
        ]
    )
    cons = gate.parse_prefer_constraints(block)
    assert len(cons) == 2, cons
    inside = {"learning_rate": 0.08, "depth": 6, "l2": 3.0}
    outside = {"learning_rate": 0.40, "depth": 12, "l2": 3.0}
    assert gate.config_prefer_hit_ratio(inside, cons) == 1.0
    assert gate.config_prefer_hit_ratio(outside, cons) == 0.0

    pol = gate.ExplorePolicy(mode="tpe", seed=1, p_default=0.5)
    pol.tpe_peek = lambda: inside
    rows = [{"trial": i, "score": 0.5, "config": inside} for i in range(30)]
    assert pol.choose(trial=50, successes=rows, exp_block=block) == "exploit"

    pol = gate.ExplorePolicy(mode="tpe", seed=1, p_default=0.5)
    pol.tpe_peek = lambda: outside
    assert pol.choose(trial=50, successes=rows, exp_block=block) == "explore"

    pol = gate.ExplorePolicy(mode="tpe_soft", seed=1, p_default=0.5)
    pol.tpe_peek = lambda: inside
    pol.choose(trial=51, successes=rows, exp_block=block)
    assert pol.last_p <= 0.25, pol.last_p
    pol = gate.ExplorePolicy(mode="tpe_llm", seed=1, p_default=0.5)
    pol.tpe_peek = lambda llm_only=False: inside if llm_only else outside
    assert pol.choose(trial=52, successes=rows, exp_block=block) == "exploit"
    assert pol.last_meta.get("experience_explore_tpe_scope") == "llm"
    print("ok tpe xor", pol.last_meta)


def main() -> int:
    test_proxy_and_rf()
    test_full_worker()
    test_tpe_xor()
    print("all fanova smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
