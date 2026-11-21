#!/usr/bin/env python3
"""Write scored CSVs + quantile splits for GFP/AAV/GB1 from existing dms_lookup.

Also records GB1 domain note (full SPG1 L=448; papers often use 55-aa domain).
Does not require MSA. Idempotent.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DEST = ROOT / "history_data/domain_protein/proteingym"

LANDSCAPES = ("gfp", "aav", "gb1")


def _quantile_split(scores: np.ndarray, seed: int = 0) -> np.ndarray:
    """Stratified by fitness quintile → train 70% / val 15% / test 15%."""
    rng = np.random.RandomState(seed)
    n = len(scores)
    labels = np.empty(n, dtype=object)
    # bin by rank quintiles
    order = np.argsort(scores)
    bins = np.array_split(order, 5)
    for bi, idxs in enumerate(bins):
        idxs = np.asarray(idxs)
        rng.shuffle(idxs)
        n_i = len(idxs)
        n_tr = int(0.70 * n_i)
        n_va = int(0.15 * n_i)
        labels[idxs[:n_tr]] = "train"
        labels[idxs[n_tr : n_tr + n_va]] = "val"
        labels[idxs[n_tr + n_va :]] = "test"
    return labels


def process_landscape(dest: Path, name: str, seed: int) -> dict:
    lookup_path = dest / name / "dms_lookup.json"
    if not lookup_path.is_file():
        raise FileNotFoundError(lookup_path)
    lookup = json.loads(lookup_path.read_text(encoding="utf-8"))
    if isinstance(lookup, dict) and "scores" in lookup:
        lookup = lookup["scores"]
    pairs = [(str(s), float(v)) for s, v in lookup.items() if s]
    pairs.sort(key=lambda x: x[0])
    seqs = [p[0] for p in pairs]
    scores = np.asarray([p[1] for p in pairs], dtype=np.float64)
    splits = _quantile_split(scores, seed=seed + hash(name) % 1000)

    out_csv = dest / name / "sequences_scored.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sequence", "score", "split", "length"])
        w.writeheader()
        for seq, score, split in zip(seqs, scores, splits):
            w.writerow(
                {
                    "sequence": seq,
                    "score": f"{float(score):.8g}",
                    "split": str(split),
                    "length": len(seq),
                }
            )

    lengths = [len(s) for s in seqs]
    meta = {
        "landscape": name,
        "n_scored": len(seqs),
        "max_len": max(lengths) if lengths else 0,
        "min_len": min(lengths) if lengths else 0,
        "mean_len": float(sum(lengths) / len(lengths)) if lengths else 0.0,
        "unique_lengths": sorted(set(lengths)),
        "split_counts": {
            "train": int((splits == "train").sum()),
            "val": int((splits == "val").sum()),
            "test": int((splits == "test").sum()),
        },
        "score_mean": float(scores.mean()) if len(scores) else 0.0,
        "score_std": float(scores.std()) if len(scores) else 0.0,
        "sequences_scored_csv": str(out_csv),
    }
    if name == "gb1":
        meta["domain_note"] = (
            "All sequences are length 448 (full SPG1_STRSG). "
            "Many latent-BO papers use the 55-aa GB1 domain instead. "
            "Deferred from primary VAE+BO until a domain-length slice is curated."
        )
        meta["primary_for_vae_bo"] = False
    else:
        meta["primary_for_vae_bo"] = True

    (dest / name / "scored_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(out_csv), **{k: meta[k] for k in ("n_scored", "max_len", "split_counts", "primary_for_vae_bo")}}))
    return meta


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    summary = {}
    for name in LANDSCAPES:
        summary[name] = process_landscape(args.dest, name, args.seed)
    (args.dest / "scored_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    # MSA status note
    msa = Path("/root/data/proteingym/DMS_msa_files.zip")
    msa_note = {
        "path": str(msa),
        "exists": msa.is_file(),
        "size_bytes": msa.stat().st_size if msa.is_file() else 0,
        "aria2_stub": Path("/root/data/proteingym/DMS_msa_files.zip.aria2").is_file(),
        "usable": False,
        "action": "VAE trains on scored sequences only until MSA zip is complete (~5.2GB expected).",
    }
    if msa.is_file() and msa.stat().st_size > 4_000_000_000:
        try:
            import zipfile

            with zipfile.ZipFile(msa) as zf:
                zf.testzip()
            msa_note["usable"] = True
            msa_note["action"] = "MSA zip OK"
        except Exception as exc:
            msa_note["error"] = str(exc)
    (args.dest / "msa_status.json").write_text(json.dumps(msa_note, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"msa_status": msa_note}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
