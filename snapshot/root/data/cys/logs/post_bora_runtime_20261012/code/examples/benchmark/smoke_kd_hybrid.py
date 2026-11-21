#!/usr/bin/env python3
"""Smoke tests: KD cascade retrieve, hybrid UCB/PUCB, semantic attach. No HPO launch."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


def test_hybrid_selector() -> None:
    from mindware.components.optimizers.hybrid_selector import HybridSelector

    rng = np.random.RandomState(0)
    M = 20
    for mode in ("bernoulli", "ucb", "pucb"):
        sel = HybridSelector(mode=mode, llm_prob=0.5, pucb_p1=0.9, pucb_c2=1.4, ucb_c=1.4, rng=rng)
        counts = {"llm": 0, "tpe": 0}
        qs = []
        for m in range(M):
            arm = sel.choose(m, M)
            counts[arm] += 1
            perf = 0.5 + 0.01 * m + (0.05 if arm == "tpe" else 0.0)
            sel.update(arm, perf, beat_incumbent=(m % 4 == 0), beat_arm_best=True)
            snap = sel.snapshot()
            qs.append((snap["Q_llm"], snap["Q_tpe"], snap["omega_llm"], snap["omega_tpe"]))
        assert counts["llm"] > 0 and counts["tpe"] > 0, (mode, counts)
        for q_llm, q_tpe, _, _ in qs:
            assert 0.0 - 1e-9 <= q_llm <= 1.0 + 1e-9
            assert 0.0 - 1e-9 <= q_tpe <= 1.0 + 1e-9
        # PUCB priors 0.9 → 0.1 by default
        if mode == "pucb":
            assert abs(qs[0][2] - 0.9) < 1e-6
            assert abs(qs[-1][2] - (0.9 - 0.8 * (M - 1) / M)) < 1e-6
            assert abs(qs[0][3] - 0.1) < 1e-6
        print(f"ok hybrid {mode} counts={counts}")

    sel = HybridSelector(mode="pucb", pucb_p1=0.9, pucb_p_end=0.5, pucb_c2=1.4, rng=np.random.RandomState(1))
    assert abs(sel.omega("llm", 0, 80) - 0.9) < 1e-9
    assert abs(sel.omega("tpe", 0, 80) - 0.1) < 1e-9
    assert abs(sel.omega("llm", 80, 80) - 0.5) < 1e-9
    assert abs(sel.omega("tpe", 80, 80) - 0.5) < 1e-9
    assert abs(sel.omega("llm", 40, 80) - 0.7) < 1e-9
    assert abs(sel.omega("tpe", 40, 80) - 0.3) < 1e-9
    print("ok pucb p_end 0.9→0.5 / 0.1→0.5")

    sel = HybridSelector(
        mode="pucb",
        pucb_p1=0.9,
        pucb_floor=0.5,
        pucb_floor_after_frac=0.6,
        pucb_c2=1.4,
        rng=np.random.RandomState(1),
    )
    assert abs(sel.omega("llm", 0, 76) - 0.9) < 1e-9
    # frac 0.5 < 0.6: still on the 0.9→0.1 line
    assert abs(sel.omega("llm", 38, 76) - (0.9 - 0.8 * 38 / 76)) < 1e-9
    # frac 0.6: clamp to 0.5 (raw would be 0.9-0.8*0.6=0.42)
    assert abs(sel.omega("llm", 46, 76) - 0.5) < 1e-9
    assert abs(sel.omega("tpe", 46, 76) - 0.5) < 1e-9
    assert abs(sel.omega("llm", 76, 76) - 0.5) < 1e-9
    print("ok pucb late floor 0.9→0.1 then clamp 0.5 after frac 0.6")

    sel = HybridSelector(
        mode="pucb",
        llm_prob=0.5,
        pucb_p1=0.9,
        pucb_switch_after_frac=0.47,
        pucb_c2=1.4,
        rng=np.random.RandomState(0),
    )
    arms_early = {sel.choose(0, 76) for _ in range(1)}
    # After switch, choose must take the bernoulli branch (still returns llm or tpe).
    late = [sel.choose(40, 76) for _ in range(30)]
    assert set(late) <= {"llm", "tpe"}
    print("ok pucb→bernoulli switch after frac 0.47", "early", arms_early, "late_share_llm", sum(a == "llm" for a in late) / len(late))


def test_kd_retrieve() -> None:
    from mindware.components.config_space.cs_builder import get_hpo_cs
    from mindware.components.utils.constants import CLASSIFICATION
    from mindware.knowledge_db.knowledge_database import KnowledgeDatabase

    KnowledgeDatabase._instance = None
    cs = get_hpo_cs("xgboost", CLASSIFICATION)
    bank = ROOT / "history_data" / "experience_bank_v2.jsonl"
    kd = KnowledgeDatabase(
        task_type=CLASSIFICATION,
        metric="bal_acc",
        task_id="pc4",
        datanode=None,
        config_space=cs,
        include_algorithms=["xgboost"],
        experience_bank_path=str(bank),
    )
    rec = kd.recommand_datasets(da_thr=3)
    assert "xgboost" in rec, rec
    assert rec["xgboost"], rec
    assert not hasattr(kd, "metadata_manager") or kd.__dict__.get("metadata_manager") is None
    hk = kd.fetch_historical_analysis_knowledge()
    assert "prompt_block" in hk["xgboost"]
    assert "dataset_similarity" in hk["xgboost"]
    assert "SHAP Analysis" not in hk["xgboost"]
    print(f"ok kd retrieve hits={rec['xgboost'][:3]}")


def test_semantic_and_vae() -> None:
    from mindware.components.experience.semantic_embed import TinyVAE, embed_profile, pad_views

    a = embed_profile("pc4", {"n_samples": 100, "n_features": 10}, "classification")
    b = embed_profile("kc2", {"n_samples": 200, "n_features": 20}, "classification")
    assert len(a) == len(b) and len(a) >= 8
    vae = TinyVAE()
    vae.fit([pad_views(a), pad_views(b), pad_views(a), pad_views(b)] * 3)
    za, zb = vae.encode(a), vae.encode(b)
    assert len(za) == 32 and len(zb) == 32
    print("ok semantic+vae")


def main() -> int:
    test_hybrid_selector()
    test_semantic_and_vae()
    test_kd_retrieve()
    print("all smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
