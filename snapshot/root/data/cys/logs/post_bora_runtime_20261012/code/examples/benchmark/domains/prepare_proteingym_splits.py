#!/usr/bin/env python3
"""Write scored CSVs + quantile train/val/test splits for GFP/AAV(/GB1).

Reads ProteinGym substitution zip + existing dms_lookup.json.
GB1 note: extracted SPG1 assays are full-length (448 aa), not the 55-aa
GB1 domain used in many latent-opt papers — mark deferred for BO.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
AA = set("ACDEFGHIKLMNPQRSTVWY")

LANDSCAPES = {
    "gfp": {
        "files": ("GFP_AEQVI_Sarkisyan_2016.csv",),
        "max_len": 238,
        "bo_ready": True,
        "note": "Sarkisyan 2016 GFP; standard VAE/BO landscape",
    },
    "aav": {
        "files": ("CAPSD_AAV2S_Sinai_2021.csv",),
        "max_len": 735,
        "bo_ready": True,
        "note": "Sinai 2021 AAV2 capsid; standard VAE/BO landscape",
    },
    "gb1": {
        "files": ("SPG1_STRSG_Olson_2014.csv", "SPG1_STRSG_Wu_2016.csv"),
        "max_len": 448,
        "bo_ready": False,
        "note": (
            "Full SPG1 (448 aa), not 55-aa GB1 domain used in LatProtRL/GGS-style "
            "papers. Defer BO until domain-sliced assay is prepared."
        ),
    },
}


def _clean(seq: str) -> str:
    return "".join(c for c in str(seq or "").upper() if c in AA)


def extract_scored(zip_path: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {k: [] for k in LANDSCAPES}
    with zipfile.ZipFile(zip_path) as zf:
        for landscape, spec in LANDSCAPES.items():
            for fname in spec["files"]:
                # find matching member
                members = [n for n in zf.namelist() if Path(n).name == fname]
                if not members:
                    continue
                with zf.open(members[0]) as fh:
                    reader = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8"))
                    for rec in reader:
                        seq = _clean(rec.get("mutated_sequence") or "")
                        try:
                            score = float(rec.get("DMS_score"))
                        except (TypeError, ValueError):
                            continue
                        if not seq or not math.isfinite(score):
                            continue
                        out[landscape].append(
                            {
                                "sequence": seq,
                                "score": score,
                                "mutant": rec.get("mutant") or "",
                                "dms_id": fname,
                            }
                        )
    return out


def quantile_split(rows: list[dict], seed: int = 0) -> dict[str, list[dict]]:
    """Stratify by fitness quintile into 70/15/15 train/val/test."""
    rng = np.random.RandomState(seed)
    scores = np.array([r["score"] for r in rows], dtype=np.float64)
    # quintile bins
    qs = np.quantile(scores, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    qs[-1] = qs[-1] + 1e-9
    bins = np.digitize(scores, qs[1:-1], right=True)
    train, val, test = [], [], []
    for b in range(int(bins.max()) + 1):
        idx = np.where(bins == b)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(0.15 * n)) if n >= 10 else max(0, n // 10)
        n_val = max(1, int(0.15 * n)) if n >= 10 else max(0, n // 10)
        test_idx = idx[:n_test]
        val_idx = idx[n_test : n_test + n_val]
        train_idx = idx[n_test + n_val :]
        if len(train_idx) == 0 and len(idx):
            train_idx = idx[-1:]
        for i in train_idx:
            train.append(rows[int(i)])
        for i in val_idx:
            val.append(rows[int(i)])
        for i in test_idx:
            test.append(rows[int(i)])
    return {"train": train, "val": val, "test": test}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sequence", "score", "mutant", "dms_id", "split"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--zip",
        type=Path,
        default=Path("/root/data/proteingym/DMS_ProteinGym_substitutions.zip"),
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=ROOT / "history_data/domain_protein/proteingym",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    scored = extract_scored(args.zip)
    summary = {}
    for landscape, rows in scored.items():
        # dedupe by sequence keeping max score
        best: dict[str, dict] = {}
        for r in rows:
            prev = best.get(r["sequence"])
            if prev is None or r["score"] > prev["score"]:
                best[r["sequence"]] = r
        uniq = list(best.values())
        splits = quantile_split(uniq, seed=args.seed)
        all_rows = []
        for split_name, part in splits.items():
            for r in part:
                all_rows.append({**r, "split": split_name})
        out_dir = args.dest / landscape
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(out_dir / "sequences_scored.csv", all_rows)
        # refresh lookup from scored
        lookup = {r["sequence"]: float(r["score"]) for r in uniq}
        (out_dir / "dms_lookup.json").write_text(json.dumps(lookup) + "\n", encoding="utf-8")
        lengths = [len(r["sequence"]) for r in uniq]
        meta = {
            "landscape": landscape,
            "n_seqs": len(uniq),
            "n_scored": len(uniq),
            "n_train": len(splits["train"]),
            "n_val": len(splits["val"]),
            "n_test": len(splits["test"]),
            "max_len": max(lengths) if lengths else 0,
            "mean_len": float(sum(lengths) / len(lengths)) if lengths else 0.0,
            "score_min": float(min(r["score"] for r in uniq)) if uniq else None,
            "score_max": float(max(r["score"] for r in uniq)) if uniq else None,
            "bo_ready": bool(LANDSCAPES[landscape]["bo_ready"]),
            "note": LANDSCAPES[landscape]["note"],
            "dms_files": list(LANDSCAPES[landscape]["files"]),
            "msa_used": False,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        summary[landscape] = meta
        print(json.dumps({"wrote": str(out_dir), **{k: meta[k] for k in ("n_scored", "n_train", "bo_ready", "max_len")}}))

    (args.dest / "splits_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
