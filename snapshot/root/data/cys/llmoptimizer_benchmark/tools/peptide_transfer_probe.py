#!/usr/bin/env python3
"""Is there a shared transfer coordinate across peptide BO tasks?

Each task optimizes within an edit-distance ball around its own seed, and those balls
are DISJOINT across tasks (measured: 0 shared feasible sequences in 300 random task
pairs). So no absolute sequence or region transfers. The candidate shared space is
seed-RELATIVE physicochemical descriptor deltas.

This probe asks three things, offline, from finished runs:
  Q1  Are the winning deltas systematic, or noise? (mean/CI per descriptor)
  Q2  Do descriptor deltas explain achieved MIC?  -> R^2 of MIC ~ f(delta)
      If low, descriptor space is NOT a sufficient transfer coordinate.
  Q3  Does seed similarity predict delta similarity? -> the retrieval premise.

Usage:  PYTHONPATH=. python tools/peptide_transfer_probe.py [--run <index.jsonl dir>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

BOLT = "/root/data/cys/BOLT/optimization/peptides"
DEFAULT_RUN = "/root/data/cys/blockdata_archive/domain_protein/stbo_init100_r100_bsz50_e2e10"

# Kyte-Doolittle hydropathy
KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2,
          I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9,
          Y=-1.3, V=4.2)
# Eisenberg consensus scale, used for the hydrophobic moment
EIS = dict(A=0.62, R=-2.53, N=-0.78, D=-0.90, C=0.29, Q=-0.85, E=-0.74, G=0.48,
           H=-0.40, I=1.38, L=1.06, K=-1.50, M=0.64, F=1.19, P=0.12, S=-0.18,
           T=-0.05, W=0.81, Y=0.26, V=1.08)
AROMATIC = set("FWY")


def net_charge(s: str) -> float:
    return sum(1.0 for c in s if c in "KR") + 0.1 * s.count("H") - sum(1.0 for c in s if c in "DE")


def gravy(s: str) -> float:
    v = [KD[c] for c in s if c in KD]
    return float(np.mean(v)) if v else 0.0


def hydrophobic_moment(s: str, angle_deg: float = 100.0) -> float:
    """muH: amphipathicity. Vector sum of Eisenberg hydrophobicity around an alpha helix."""
    ang = np.deg2rad(angle_deg)
    h = np.array([EIS.get(c, 0.0) for c in s])
    if h.size == 0:
        return 0.0
    k = np.arange(h.size)
    return float(np.hypot((h * np.cos(k * ang)).sum(), (h * np.sin(k * ang)).sum()) / h.size)


def aromaticity(s: str) -> float:
    return sum(1 for c in s if c in AROMATIC) / max(1, len(s))


DESCRIPTORS = {
    "length": lambda s: float(len(s)),
    "net_charge": net_charge,
    "gravy": gravy,
    "hyd_moment": hydrophobic_moment,
    "aromaticity": aromaticity,
    "n_cys": lambda s: float(s.count("C")),
}
NAMES = list(DESCRIPTORS)


def vec(s: str) -> np.ndarray:
    return np.array([DESCRIPTORS[n](s) for n in NAMES], dtype=float)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=DEFAULT_RUN)
    a = p.parse_args()

    sys.path.insert(0, BOLT)
    from apex_oracle.refseqs import REFERENCE_SEQUENCE as R  # noqa: PLC0415

    by: Dict[str, dict] = {}
    for line in open(Path(a.run) / "index.jsonl", encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            by[r["task_id"]] = r
    rows = [r for r in by.values() if r.get("best_feasible_seq")]
    print(f"tasks: {len(rows)}\n")

    idx = np.array([int(r["task_id"].split("_")[1]) for r in rows])
    seeds = [R[i] for i in idx]
    best = [r["best_feasible_seq"] for r in rows]
    mic = np.array([-r["best_feasible_objective"] for r in rows])
    trunc = np.array([bool(r.get("truncated")) for r in rows])

    S = np.stack([vec(s) for s in seeds])          # seed descriptors
    B = np.stack([vec(s) for s in best])           # best-found descriptors
    D = B - S                                      # the candidate shared coordinate
    logmic = np.log10(mic)

    # ---------------- Q1: are the winning deltas systematic? ----------------
    print("=" * 78)
    print("Q1  Winning descriptor deltas (best_found - seed). Systematic or noise?")
    print("=" * 78)
    print(f"{'descriptor':<14}{'mean':>9}{'sd':>9}{'95% CI of mean':>22}{'sign consistency':>19}")
    for j, n in enumerate(NAMES):
        d = D[:, j]
        se = d.std(ddof=1) / np.sqrt(d.size)
        lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se
        frac = max((d > 0).mean(), (d < 0).mean())
        star = " *" if lo * hi > 0 else ""
        print(f"{n:<14}{d.mean():9.3f}{d.std(ddof=1):9.3f}   [{lo:7.3f},{hi:7.3f}]{frac:17.0%}{star}")
    print("  * = CI excludes 0, i.e. a systematic direction shared across tasks")

    # ---------------- Q2: do deltas explain MIC? ----------------
    print("\n" + "=" * 78)
    print("Q2  Does descriptor-delta space explain achieved MIC?  (R^2, 5-fold CV)")
    print("=" * 78)

    def cv_r2(X: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0) -> float:
        rng = np.random.default_rng(seed)
        order = rng.permutation(y.size)
        X, y = X[order], y[order]
        Xa = np.column_stack([np.ones(len(X)), X])
        pred = np.empty_like(y)
        for f in range(folds):
            te = np.arange(y.size) % folds == f
            tr = ~te
            beta, *_ = np.linalg.lstsq(Xa[tr], y[tr], rcond=None)
            pred[te] = Xa[te] @ beta
        ss_res = ((y - pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return 1 - ss_res / ss_tot

    for label, X in (("delta only", D), ("seed only", S), ("seed + delta", np.column_stack([S, D]))):
        print(f"  log10(MIC) ~ {label:<14} CV R^2 = {cv_r2(X, logmic):6.3f}")
    print("  (A low delta-only R^2 means descriptor deltas are NOT a sufficient")
    print("   transfer coordinate: MIC depends on the exact sequence, not just its")
    print("   aggregate physicochemistry.)")

    # ---------------- Q3: does seed similarity predict delta similarity? ----------------
    print("\n" + "=" * 78)
    print("Q3  Retrieval premise: do similar SEEDS want similar DELTAS?")
    print("=" * 78)
    Sz = (S - S.mean(0)) / (S.std(0) + 1e-12)
    Dz = (D - D.mean(0)) / (D.std(0) + 1e-12)
    rng = np.random.default_rng(0)
    n = len(rows)
    pairs = rng.integers(0, n, size=(40000, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    ds = np.linalg.norm(Sz[pairs[:, 0]] - Sz[pairs[:, 1]], axis=1)   # seed distance
    dd = np.linalg.norm(Dz[pairs[:, 0]] - Dz[pairs[:, 1]], axis=1)   # delta distance
    from scipy.stats import spearmanr  # noqa: PLC0415

    rho, pv = spearmanr(ds, dd)
    print(f"  Spearman(seed distance, delta distance) over {len(pairs)} pairs:")
    print(f"    rho = {rho:+.4f}   p = {pv:.2e}")
    print("    (positive rho => nearby seeds want similar edits => retrieval can work)")

    q = np.quantile(ds, [0.1, 0.9])
    near, far = dd[ds <= q[0]], dd[ds >= q[1]]
    print(f"\n  delta distance | nearest-10% seed pairs : mean {near.mean():.3f}")
    print(f"  delta distance | farthest-10% seed pairs: mean {far.mean():.3f}")
    print(f"  separation: {100*(far.mean()/near.mean()-1):+.1f}%  "
          f"(want strongly positive for retrieval to add value)")

    # ---------------- context: seed MIC vs achieved, and truncation ----------------
    print("\n" + "=" * 78)
    print("Context")
    print("=" * 78)
    for nm, m in (("all", np.ones(n, bool)), ("truncated", trunc), ("not truncated", ~trunc)):
        print(f"  {nm:<15} n={m.sum():4d}  median MIC={np.median(mic[m]):8.2f}  "
              f"median |delta_charge|={np.median(np.abs(D[m,1])):5.2f}  "
              f"median |delta_gravy|={np.median(np.abs(D[m,2])):5.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
