#!/usr/bin/env python3
"""Family-STRATIFIED 800/200 train/test split over the 1000 BOLT seed peptides.

Each family contributes ~20% of its members to test, so both splits carry the same
mix of family types. Contrast with a family-DISJOINT split (whole families on one
side), which is what you want to measure generalisation to *unseen families*.

  stratified  -> "new member of a known family"  (family members straddle the split)
  disjoint    -> "new family"                    (zero cross-split similarity)

Stratified therefore has hard leakage BY DESIGN: a test reference will usually have a
family member in train, sometimes at similarity 0.750, at which point that training
seed is itself a feasible candidate for the test task. That inflates any method which
can transfer across tasks (BOLT) relative to one that cannot (STBO). Report results
split by `has_family_member_in_train` to control for it.

Families = connected components of the graph {sim(i,j) >= --family-threshold}, where
sim is Levenshtein similarity normalised by the LONGER sequence (so truncation series
of a common parent land in one family).

Exact 200 is hit with SIZE-CLASS-STRATIFIED apportionment, not plain largest-remainder.
Largest-remainder is exact overall but starves the class with the smallest fractional
remainder: at 20% every size-2 family has remainder 0.40 and every singleton 0.20, so
all 49 size-2 families win a slot (50% test each) while the 300 singletons -- 30% of the
library -- get 4.3% instead of 20%. See apportion() for the fix.

Usage:
  PYTHONPATH=. python tools/build_peptide_family_split.py \
      --family-threshold 0.65 --n-test 200 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

BOLT = "/root/data/cys/BOLT/optimization/peptides"
OUT = Path("/root/data/cys/blockdata_archive/llmoptimizer_benchmark/docs")


def similarity_matrix(refs: List[str]) -> np.ndarray:
    from Levenshtein import distance as lev  # noqa: PLC0415

    n = len(refs)
    S = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        a = refs[i]
        for j in range(i + 1, n):
            b = refs[j]
            m = max(len(a), len(b))
            s = (m - lev(a, b)) / m
            S[i, j] = S[j, i] = s
    return S


def components(S: np.ndarray, thr: float) -> List[List[int]]:
    n = S.shape[0]
    seen = np.zeros(n, bool)
    out: List[List[int]] = []
    for s0 in range(n):
        if seen[s0]:
            continue
        stack, comp = [s0], [s0]
        seen[s0] = True
        while stack:
            u = stack.pop()
            for v in np.where((S[u] >= thr) & (~seen))[0]:
                seen[v] = True
                stack.append(int(v))
                comp.append(int(v))
        out.append(sorted(comp))
    return out


def apportion(fams: List[List[int]], n_test: int, rng: np.random.Generator) -> List[int]:
    """Size-class-stratified apportionment of n_test slots.

    Plain largest-remainder is exact but systematically starves whichever size class has
    the smallest fractional remainder. At 20% every size-2 family has remainder 0.40 and
    every singleton 0.20, so all 49 size-2 families win a slot (50% test each) while the
    300 singletons get 4.3% instead of 20%.

    Fix: allocate each SIZE CLASS its own exact quota (frac * n_members_in_class), giving
    every family floor(frac*size) and distributing that class's fractional remainder at
    random within the class. Each class then lands on `frac` on average, so the split is
    proportional per family rather than proportional per remainder.
    """
    total = sum(len(f) for f in fams)
    frac = n_test / total
    base = [int(np.floor(frac * len(f))) for f in fams]

    by_size: Dict[int, List[int]] = {}
    for k, f in enumerate(fams):
        by_size.setdefault(len(f), []).append(k)

    for size, ks in by_size.items():
        # slots this class still owes, from the shared fractional part
        owed = int(round(len(ks) * (frac * size - np.floor(frac * size))))
        # a family with >1 member never goes wholly into test (keeps train coverage)
        cap = (size - 1) if size > 1 else 1
        eligible = [k for k in ks if base[k] < cap]
        owed = min(owed, len(eligible))
        for k in rng.choice(eligible, size=owed, replace=False).tolist() if owed else []:
            base[int(k)] += 1

    # exact total: nudge by single slots, largest families first (smallest relative effect)
    order = sorted(range(len(fams)), key=lambda k: -len(fams[k]))
    while sum(base) != n_test:
        if sum(base) < n_test:
            for k in order:
                cap = (len(fams[k]) - 1) if len(fams[k]) > 1 else 1
                if base[k] < cap:
                    base[k] += 1
                    break
            else:
                break
        else:
            for k in order:
                if base[k] > 0:
                    base[k] -= 1
                    break
            else:
                break
    return base


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--family-threshold", type=float, default=0.65)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=OUT / "peptide_split_800_200_stratified.json")
    a = p.parse_args()

    sys.path.insert(0, BOLT)
    from apex_oracle.refseqs import REFERENCE_SEQUENCE as R  # noqa: PLC0415

    refs = list(R)
    n = len(refs)
    cache = Path("/tmp/ref_sim.npy")
    S = np.load(cache) if cache.is_file() and np.load(cache).shape == (n, n) else similarity_matrix(refs)
    np.save(cache, S)

    fams = components(S, a.family_threshold)
    fams.sort(key=lambda f: (-len(f), f[0]))
    rng = np.random.default_rng(a.seed)
    slots = apportion(fams, a.n_test, rng)
    test: List[int] = []
    per_family = []
    for f, k in zip(fams, slots):
        chosen = sorted(rng.choice(f, size=k, replace=False).tolist()) if k else []
        test += chosen
        per_family.append(
            {"size": len(f), "n_test": k, "test_frac": round(k / len(f), 3),
             "members": f, "test_members": chosen}
        )
    test_set = set(test)
    train = [i for i in range(n) if i not in test_set]

    def leaks(thr: float) -> List[tuple]:
        return [(i, int(j), float(S[i, j]))
                for i in sorted(test_set)
                for j in np.where(S[i] >= thr)[0] if int(j) not in test_set]

    print(f"family threshold {a.family_threshold}: {len(fams)} families, "
          f"largest {len(fams[0])}, singletons {sum(1 for f in fams if len(f)==1)}")
    print(f"split: {len(train)} train / {len(test_set)} test\n")

    print("per-family test fraction (target 0.200):")
    fr = np.array([d["test_frac"] for d in per_family])
    multi = np.array([d["test_frac"] for d in per_family if d["size"] > 1])
    print(f"  all families : mean {fr.mean():.3f}  median {np.median(fr):.3f}")
    print(f"  size>1 only  : mean {multi.mean():.3f}  median {np.median(multi):.3f} "
          f"min {multi.min():.3f} max {multi.max():.3f}")
    print(f"\n  by family size: {'size':>5}{'n fams':>8}{'mean frac':>11}{'test':>7}")
    import collections as _c
    bysz = _c.defaultdict(list)
    for d in per_family:
        bysz[d["size"]].append(d)
    for sz in sorted(bysz):
        ds = bysz[sz]
        print(f"{'':18}{sz:5d}{len(ds):8d}{np.mean([x['n_test']/sz for x in ds]):11.3f}"
              f"{sum(x['n_test'] for x in ds):7d}")

    big = [d for d in per_family if d["size"] >= 10]
    print(f"\n  largest families: {'size':>5}{'n_test':>8}{'frac':>7}")
    for d in big[:8]:
        print(f"{'':18}{d['size']:5d}{d['n_test']:8d}{d['test_frac']:7.3f}")

    print("\nleakage (a TEST ref within thr of a TRAIN ref) — high BY DESIGN here:")
    for thr in (0.75, 0.65, 0.55):
        lk = leaks(thr)
        print(f"  thr {thr}: {len(lk):5d} pairs, {len(set(x[0] for x in lk)):3d}/{len(test_set)} test refs")

    fam_of = {i: k for k, f in enumerate(fams) for i in f}
    has_kin = {i: any(fam_of[j] == fam_of[i] for j in train) for i in sorted(test_set)}
    n_kin = sum(has_kin.values())
    print(f"\ncontrol variable: test refs WITH a family member in train: {n_kin}/{len(test_set)}"
          f"  (without: {len(test_set)-n_kin})")

    Lt = [len(refs[i]) for i in sorted(test_set)]
    Lr = [len(refs[i]) for i in train]
    print(f"length balance: test med {np.median(Lt):.0f} [{min(Lt)},{max(Lt)}] | "
          f"train med {np.median(Lr):.0f} [{min(Lr)},{max(Lr)}]")

    payload = {
        "kind": "family-stratified",
        "family_threshold": a.family_threshold,
        "similarity": "levenshtein, normalised by the longer sequence",
        "seed": a.seed,
        "n_train": len(train),
        "n_test": len(test_set),
        "n_families": len(fams),
        "leakage_pairs": {str(t): len(leaks(t)) for t in (0.75, 0.65, 0.55)},
        "test_refs_with_family_member_in_train": n_kin,
        "warning": (
            "Family members straddle the split by design. Hard leakage is expected. "
            "Report metrics split by has_family_member_in_train to control for it."
        ),
        "train_ids": [f"pep_{i:03d}" for i in train],
        "test_ids": [f"pep_{i:03d}" for i in sorted(test_set)],
        "has_family_member_in_train": {f"pep_{i:03d}": bool(v) for i, v in has_kin.items()},
        "families": per_family,
    }
    a.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
