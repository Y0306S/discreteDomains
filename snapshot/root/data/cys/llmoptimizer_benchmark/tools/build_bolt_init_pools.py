#!/usr/bin/env python3
"""Build per-reference BOLT init pools (sequences + APEX scores) for the peptide BO benchmark.

Why this exists
---------------
BOLT ships exactly one init block, ``apex_oracle/init_data/seed_0_init.txt``, which is
1000 random mutants of ``REFERENCE_SEQUENCE[0]``. Its BO loop selects a block with

    bacteria_num = constraint_types[0] - init_offset_helper          # optimization script
    train_x = all_rows[bacteria_num * N : (bacteria_num + 1) * N]

where ``constraint_types[0]`` is the reference index (see ``APEXSimilarityConstraint``)
and ``init_offset_helper`` is the index of the *first reference present in the file*.
So running reference ``i`` requires a file that actually contains a block for ``i``.

Running 1000 tasks against the single ref-0 block leaves 999/1000 tasks with **zero**
init points satisfying the >=0.75 similarity constraint, which silently degenerates the
whole run (empty top-k, ``best_x_seen is None``, E2E finetuning on 10 sequences).

This tool generates the missing blocks, faithfully reusing BOLT's own mutation sampler
(``apex_oracle/init_data/create_mutations.py``) and score convention
(``get_scores_helper.py``: ``score = -apex_wrapper(seqs)[:, 0]``, i.e. the maximize
objective for bacteria_0). Verified to reproduce ``seed_0_scores.csv`` to float32.

Usage
-----
    python tools/build_bolt_init_pools.py --ref-start 0 --ref-end 1000

Outputs (into --out-dir):
    refs_{start}_{end}_init.txt       num_mutations rows per reference, blocks in order
    refs_{start}_{end}_scores.csv     aligned 1:1 with the init file
    refs_{start}_{end}_manifest.json  records init_offset_helper == ref_start, + checksums

Point the BO harness at the init/scores pair and set ``init_offset_helper = ref_start``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

DEFAULT_BOLT_ROOT = Path("/root/data/cys/BOLT/optimization/peptides")
# BOLT's constraint is similarity >= 0.75, i.e. edit_distance / len(ref) <= 0.25.
DEFAULT_MAX_MUTATION_DISTANCE = 0.25
DEFAULT_MIN_SIMILARITY = 0.75
ALL_AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def _load_bolt(bolt_root: Path):
    """Import BOLT's refseqs, mutation sampler and APEX oracle from the release tree."""
    if str(bolt_root) not in sys.path:
        sys.path.insert(0, str(bolt_root))

    from Levenshtein import distance as lev  # noqa: PLC0415

    from apex_oracle import apex_wrapper  # noqa: PLC0415
    from apex_oracle.refseqs import REFERENCE_SEQUENCE  # noqa: PLC0415

    # create_mutations.py is a script, so load it by path rather than as a package.
    cm_path = bolt_root / "apex_oracle/init_data/create_mutations.py"
    spec = importlib.util.spec_from_file_location("bolt_create_mutations", cm_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {cm_path}")
    cm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cm)
    return REFERENCE_SEQUENCE, cm.generate_unique_mutations, apex_wrapper, lev


def similarity(seq: str, ref: str, lev) -> float:
    return (len(ref) - lev(seq, ref)) / len(ref)


def _topup(
    ref: str,
    have: List[str],
    target: int,
    *,
    lev,
    rng: np.random.Generator,
    max_mutation_distance: float,
    min_similarity: float,
    max_attempts_factor: int = 400,
) -> List[str]:
    """Deterministic-ish fallback if BOLT's rejection sampler underfills a short reference.

    Same mutation family as BOLT (substitutions only, <=25% edit distance), just driven
    harder. Only triggers for references where 1000 unique feasible mutants are scarce.
    """
    seen = set(have)
    out = list(have)
    n_pos_max = max(2, int(len(ref) * max_mutation_distance))
    attempts = 0
    limit = target * max_attempts_factor
    while len(out) < target and attempts < limit:
        attempts += 1
        n_pos = int(rng.integers(1, n_pos_max + 1))
        mutant = list(ref)
        positions = rng.choice(len(ref), size=min(n_pos, len(ref)), replace=False)
        for p in positions:
            choices = [aa for aa in ALL_AMINO_ACIDS if aa != mutant[p]]
            mutant[p] = choices[int(rng.integers(0, len(choices)))]
        cand = "".join(mutant)
        if cand == ref or cand in seen:
            continue
        if similarity(cand, ref, lev) < min_similarity:
            continue
        seen.add(cand)
        out.append(cand)
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(
    *,
    bolt_root: Path,
    ref_start: int,
    ref_end: int,
    num_mutations: int,
    out_dir: Path,
    seed: int,
    batch_size: int,
    include_reference: bool,
    max_mutation_distance: float,
    min_similarity: float,
) -> Dict[str, object]:
    refs_all, generate_unique_mutations, apex_wrapper, lev = _load_bolt(bolt_root)
    if not 0 <= ref_start < ref_end <= len(refs_all):
        raise SystemExit(
            f"--ref-start/--ref-end must satisfy 0 <= start < end <= {len(refs_all)}"
        )
    refs = list(refs_all[ref_start:ref_end])
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)  # BOLT's sampler uses the global numpy RNG
    rng = np.random.default_rng(seed)

    print(
        f"[init-pools] refs [{ref_start},{ref_end}) n={len(refs)} "
        f"num_mutations={num_mutations} similarity>={min_similarity}",
        flush=True,
    )
    blocks = generate_unique_mutations(
        refs,
        num_mutations=num_mutations,
        max_mutation_distance=max_mutation_distance,
    )

    # BOLT's sampler does not guarantee feasibility or a full block; enforce both, and
    # keep every block exactly num_mutations long so block slicing stays valid.
    all_seqs: List[str] = []
    per_ref: List[Dict[str, object]] = []
    n_topped = 0
    for offset, (ref, muts) in enumerate(zip(refs, blocks)):
        ref_idx = ref_start + offset
        muts = [m for m in muts if similarity(m, ref, lev) >= min_similarity]
        muts = list(dict.fromkeys(muts))  # dedupe, preserve order
        if include_reference and ref not in muts:
            muts.insert(0, ref)
        if len(muts) < num_mutations:
            before = len(muts)
            muts = _topup(
                ref,
                muts,
                num_mutations,
                lev=lev,
                rng=rng,
                max_mutation_distance=max_mutation_distance,
                min_similarity=min_similarity,
            )
            n_topped += 1
            print(
                f"[init-pools] ref[{ref_idx}] len={len(ref)} topped up {before} -> {len(muts)}",
                flush=True,
            )
        if len(muts) < num_mutations:
            raise SystemExit(
                f"ref[{ref_idx}] ({ref!r}, len={len(ref)}): only {len(muts)} unique "
                f"feasible mutants, need {num_mutations}. Lower --num-mutations."
            )
        muts = muts[:num_mutations]
        sims = [similarity(m, ref, lev) for m in muts]
        assert min(sims) >= min_similarity
        all_seqs.extend(muts)
        per_ref.append(
            {
                "ref_index": ref_idx,
                "reference": ref,
                "len": len(ref),
                "n": len(muts),
                "min_similarity": round(float(min(sims)), 4),
            }
        )

    expected = len(refs) * num_mutations
    assert len(all_seqs) == expected, (len(all_seqs), expected)

    # Score with the APEX oracle using BOLT's exact convention (get_scores_helper.py):
    #   score = -apex_wrapper(seqs)[:, 0]   -> maximize objective for bacteria_0
    print(f"[init-pools] scoring {len(all_seqs)} sequences with APEX ...", flush=True)
    scores = np.empty(len(all_seqs), dtype=np.float64)
    for start in range(0, len(all_seqs), batch_size):
        stop = min(start + batch_size, len(all_seqs))
        scores[start:stop] = -np.asarray(apex_wrapper(all_seqs[start:stop]))[:, 0]
        print(f"[init-pools]   scored {stop}/{len(all_seqs)}", flush=True)
    if not np.all(np.isfinite(scores)):
        raise SystemExit(f"APEX produced {int((~np.isfinite(scores)).sum())} non-finite scores")

    tag = f"refs_{ref_start}_{ref_end}"
    init_path = out_dir / f"{tag}_init.txt"
    scores_path = out_dir / f"{tag}_scores.csv"
    manifest_path = out_dir / f"{tag}_manifest.json"

    # No trailing newline on the init file, matching seed_0_init.txt's format.
    init_path.write_text("\n".join(all_seqs), encoding="utf-8")
    scores_path.write_text("\n".join(f"{s:.8f}" for s in scores) + "\n", encoding="utf-8")

    manifest = {
        "ref_start": ref_start,
        "ref_end": ref_end,
        "n_refs": len(refs),
        "num_mutations": num_mutations,
        "rows": len(all_seqs),
        # This is the value to pass to the BO runner. bacteria_num = ref_index - ref_start.
        "init_offset_helper": ref_start,
        "min_similarity": min_similarity,
        "max_mutation_distance": max_mutation_distance,
        "include_reference": include_reference,
        "seed": seed,
        "score_convention": "-apex_wrapper(seqs)[:, 0]  (maximize, bacteria_0)",
        "init_path": str(init_path),
        "scores_path": str(scores_path),
        "init_sha256": _sha256(init_path),
        "scores_sha256": _sha256(scores_path),
        "n_refs_topped_up": n_topped,
        "per_ref": per_ref,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(
        f"\n[init-pools] wrote {len(all_seqs)} rows for {len(refs)} refs\n"
        f"  init     : {init_path}\n"
        f"  scores   : {scores_path}\n"
        f"  manifest : {manifest_path}\n"
        f"  --> pass init_offset_helper={ref_start} to the BO runner",
        flush=True,
    )
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bolt-root", type=Path, default=DEFAULT_BOLT_ROOT)
    p.add_argument("--ref-start", type=int, default=0, help="first reference index (inclusive)")
    p.add_argument("--ref-end", type=int, default=1000, help="last reference index (exclusive)")
    p.add_argument("--num-mutations", type=int, default=1000, help="init points per reference")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_BOLT_ROOT / "apex_oracle/init_data/generated",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=2000, help="APEX scoring batch size")
    p.add_argument(
        "--include-reference",
        action="store_true",
        help="prepend the reference itself to each block (BOLT excludes it; off by default)",
    )
    p.add_argument("--max-mutation-distance", type=float, default=DEFAULT_MAX_MUTATION_DISTANCE)
    p.add_argument("--min-similarity", type=float, default=DEFAULT_MIN_SIMILARITY)
    a = p.parse_args()
    build(
        bolt_root=a.bolt_root,
        ref_start=a.ref_start,
        ref_end=a.ref_end,
        num_mutations=a.num_mutations,
        out_dir=a.out_dir,
        seed=a.seed,
        batch_size=a.batch_size,
        include_reference=a.include_reference,
        max_mutation_distance=a.max_mutation_distance,
        min_similarity=a.min_similarity,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
