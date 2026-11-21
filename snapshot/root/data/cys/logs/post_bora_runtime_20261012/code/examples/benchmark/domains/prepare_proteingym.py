#!/usr/bin/env python3
"""Extract ProteinGym GFP / AAV / GB1 DMS (+ optional MSA) into landscape CSVs.

Writes under history_data/domain_protein/proteingym/{gfp,aav,gb1}/.
Prefers local Harvard zips on /root/data/proteingym; otherwise streams HuggingFace.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AA = set("ACDEFGHIKLMNPQRSTVWY")

LANDSCAPES = {
    "gfp": {"prefixes": ("GFP_AEQVI",), "max_len": 250},
    "aav": {"prefixes": ("CAPSD_AAV2S",), "max_len": 800},
    "gb1": {"prefixes": ("SPG1_STRSG",), "max_len": 480},
}


def _clean_seq(seq: str) -> str:
    s = re.sub(r"[^A-Za-z]", "", str(seq or "").upper())
    return "".join(c for c in s if c in AA)


def _match_landscape(dms_id: str) -> str | None:
    did = str(dms_id or "")
    for name, spec in LANDSCAPES.items():
        if any(did.startswith(p) for p in spec["prefixes"]):
            return name
    return None


def _out_root(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def extract_from_zip(zip_path: Path) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = defaultdict(list)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        csv_names = [n for n in names if n.lower().endswith(".csv")]
        for name in csv_names:
            base = Path(name).name
            landscape = None
            for ls, spec in LANDSCAPES.items():
                if any(base.startswith(p) for p in spec["prefixes"]):
                    landscape = ls
                    break
            if landscape is None and "reference" in base.lower():
                continue
            if landscape is None:
                continue
            with zf.open(name) as fh:
                text = fh.read().decode("utf-8", errors="replace").splitlines()
            reader = csv.DictReader(text)
            for rec in reader:
                seq = _clean_seq(rec.get("mutated_sequence") or rec.get("sequence") or "")
                wt = _clean_seq(rec.get("target_seq") or rec.get("wildtype") or "")
                try:
                    score = float(rec.get("DMS_score") or rec.get("score") or "nan")
                except ValueError:
                    score = float("nan")
                if seq:
                    rows[landscape].append(
                        {
                            "sequence": seq,
                            "wt": wt,
                            "score": score,
                            "mutant": rec.get("mutant") or "",
                            "dms_id": rec.get("DMS_id") or base,
                        }
                    )
    return rows


def extract_from_huggingface() -> dict[str, list[dict]]:
    from datasets import load_dataset

    rows: dict[str, list[dict]] = defaultdict(list)
    ds = load_dataset("OATML-Markslab/ProteinGym_v1", name="DMS_substitutions", split="train")
    for rec in ds:
        did = rec.get("DMS_id") or ""
        landscape = _match_landscape(did)
        if landscape is None:
            continue
        seq = _clean_seq(rec.get("mutated_sequence") or "")
        wt = _clean_seq(rec.get("target_seq") or "")
        try:
            score = float(rec.get("DMS_score"))
        except (TypeError, ValueError):
            score = float("nan")
        if seq:
            rows[landscape].append(
                {
                    "sequence": seq,
                    "wt": wt,
                    "score": score,
                    "mutant": rec.get("mutant") or "",
                    "dms_id": did,
                }
            )
    return rows


def add_msa_sequences(msa_zip: Path, rows: dict[str, list[dict]]) -> None:
    if not msa_zip.is_file() or msa_zip.stat().st_size < 10_000:
        return
    with zipfile.ZipFile(msa_zip) as zf:
        for name in zf.namelist():
            base = Path(name).name
            landscape = None
            for ls, spec in LANDSCAPES.items():
                if any(p.lower() in base.lower() for p in spec["prefixes"]):
                    landscape = ls
                    break
            if landscape is None:
                continue
            if not any(name.endswith(ext) for ext in (".a2m", ".fasta", ".fa", ".a3m")):
                continue
            with zf.open(name) as fh:
                text = fh.read().decode("utf-8", errors="replace")
            seq = []
            for line in text.splitlines():
                if line.startswith(">"):
                    cleaned = _clean_seq("".join(seq))
                    if cleaned:
                        rows[landscape].append(
                            {"sequence": cleaned, "wt": "", "score": float("nan"), "mutant": "", "dms_id": f"msa:{base}"}
                        )
                    seq = []
                else:
                    seq.append(line.strip())
            cleaned = _clean_seq("".join(seq))
            if cleaned:
                rows[landscape].append(
                    {"sequence": cleaned, "wt": "", "score": float("nan"), "mutant": "", "dms_id": f"msa:{base}"}
                )


def write_landscapes(rows: dict[str, list[dict]], dest: Path) -> None:
    dest = _out_root(dest)
    for landscape, recs in rows.items():
        out_dir = dest / landscape
        out_dir.mkdir(parents=True, exist_ok=True)
        seen = set()
        seqs = []
        lookup = {}
        dms_ids = set()
        lengths = []
        for rec in recs:
            seq = rec["sequence"]
            if not seq or seq in seen:
                continue
            seen.add(seq)
            seqs.append(seq)
            lengths.append(len(seq))
            dms_ids.add(str(rec.get("dms_id") or ""))
            score = rec.get("score")
            if score == score:  # not NaN
                lookup[seq] = float(score)
            wt = rec.get("wt") or ""
            if wt and wt not in seen:
                seen.add(wt)
                seqs.append(wt)
                lengths.append(len(wt))
        csv_path = out_dir / "sequences.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["sequence"])
            for s in seqs:
                w.writerow([s])
        lookup_path = out_dir / "dms_lookup.json"
        lookup_path.write_text(json.dumps(lookup, indent=0), encoding="utf-8")
        meta = {
            "landscape": landscape,
            "n_seqs": len(seqs),
            "n_scored": len(lookup),
            "dms_ids": sorted(x for x in dms_ids if x),
            "max_len": max(lengths) if lengths else 0,
            "mean_len": (sum(lengths) / len(lengths)) if lengths else 0,
            "prefixes": list(LANDSCAPES[landscape]["prefixes"]),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"wrote": str(out_dir), **{k: meta[k] for k in ("n_seqs", "n_scored", "max_len")}}))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip", type=Path, default=Path("/root/data/proteingym/DMS_ProteinGym_substitutions.zip"))
    p.add_argument("--msa-zip", type=Path, default=Path("/root/data/proteingym/DMS_msa_files.zip"))
    p.add_argument(
        "--dest",
        type=Path,
        default=ROOT / "history_data/domain_protein/proteingym",
    )
    p.add_argument("--huggingface", action="store_true")
    args = p.parse_args(argv)

    rows: dict[str, list[dict]] = defaultdict(list)
    zip_ok = args.zip.is_file() and args.zip.stat().st_size > 1_000_000
    if zip_ok and not args.huggingface:
        print(f"[proteingym] extracting {args.zip}")
        rows = extract_from_zip(args.zip)
    if not rows:
        print("[proteingym] streaming HuggingFace OATML-Markslab/ProteinGym_v1")
        rows = extract_from_huggingface()
    if args.msa_zip.is_file() and args.msa_zip.stat().st_size > 1_000_000:
        print(f"[proteingym] adding MSA sequences from {args.msa_zip}")
        add_msa_sequences(args.msa_zip, rows)
    if not rows:
        raise SystemExit("no GFP/AAV/GB1 rows found")
    write_landscapes(rows, args.dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Extract ProteinGym GFP / AAV / GB1 DMS (+ optional MSA) into landscape CSVs.

Writes under history_data/domain_protein/proteingym/{gfp,aav,gb1}/.
Prefers local Harvard zips on /root/data/proteingym; otherwise streams HuggingFace.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AA = set("ACDEFGHIKLMNPQRSTVWY")

LANDSCAPES = {
    "gfp": {"prefixes": ("GFP_AEQVI",), "max_len": 250},
    "aav": {"prefixes": ("CAPSD_AAV2S",), "max_len": 800},
    "gb1": {"prefixes": ("SPG1_STRSG",), "max_len": 480},
}


def _clean_seq(seq: str) -> str:
    s = re.sub(r"[^A-Za-z]", "", str(seq or "").upper())
    return "".join(c for c in s if c in AA)


def _match_landscape(dms_id: str) -> str | None:
    did = str(dms_id or "")
    for name, spec in LANDSCAPES.items():
        if any(did.startswith(p) for p in spec["prefixes"]):
            return name
    return None


def _out_root(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def extract_from_zip(zip_path: Path) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = defaultdict(list)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        csv_names = [n for n in names if n.lower().endswith(".csv")]
        for name in csv_names:
            base = Path(name).name
            landscape = None
            for ls, spec in LANDSCAPES.items():
                if any(base.startswith(p) for p in spec["prefixes"]):
                    landscape = ls
                    break
            if landscape is None and "reference" in base.lower():
                continue
            if landscape is None:
                continue
            with zf.open(name) as fh:
                text = fh.read().decode("utf-8", errors="replace").splitlines()
            reader = csv.DictReader(text)
            for rec in reader:
                seq = _clean_seq(rec.get("mutated_sequence") or rec.get("sequence") or "")
                wt = _clean_seq(rec.get("target_seq") or rec.get("wildtype") or "")
                try:
                    score = float(rec.get("DMS_score") or rec.get("score") or "nan")
                except ValueError:
                    score = float("nan")
                if seq:
                    rows[landscape].append(
                        {
                            "sequence": seq,
                            "wt": wt,
                            "score": score,
                            "mutant": rec.get("mutant") or "",
                            "dms_id": rec.get("DMS_id") or base,
                        }
                    )
    return rows


def extract_from_huggingface() -> dict[str, list[dict]]:
    from datasets import load_dataset

    rows: dict[str, list[dict]] = defaultdict(list)
    ds = load_dataset("OATML-Markslab/ProteinGym_v1", name="DMS_substitutions", split="train")
    for rec in ds:
        did = rec.get("DMS_id") or ""
        landscape = _match_landscape(did)
        if landscape is None:
            continue
        seq = _clean_seq(rec.get("mutated_sequence") or "")
        wt = _clean_seq(rec.get("target_seq") or "")
        try:
            score = float(rec.get("DMS_score"))
        except (TypeError, ValueError):
            score = float("nan")
        if seq:
            rows[landscape].append(
                {
                    "sequence": seq,
                    "wt": wt,
                    "score": score,
                    "mutant": rec.get("mutant") or "",
                    "dms_id": did,
                }
            )
    return rows


def add_msa_sequences(msa_zip: Path, rows: dict[str, list[dict]]) -> None:
    if not msa_zip.is_file() or msa_zip.stat().st_size < 10_000:
        return
    with zipfile.ZipFile(msa_zip) as zf:
        for name in zf.namelist():
            base = Path(name).name
            landscape = None
            for ls, spec in LANDSCAPES.items():
                if any(p.lower() in base.lower() for p in spec["prefixes"]):
                    landscape = ls
                    break
            if landscape is None:
                continue
            if not any(name.endswith(ext) for ext in (".a2m", ".fasta", ".fa", ".a3m")):
                continue
            with zf.open(name) as fh:
                text = fh.read().decode("utf-8", errors="replace")
            seq = []
            for line in text.splitlines():
                if line.startswith(">"):
                    cleaned = _clean_seq("".join(seq))
                    if cleaned:
                        rows[landscape].append(
                            {"sequence": cleaned, "wt": "", "score": float("nan"), "mutant": "", "dms_id": f"msa:{base}"}
                        )
                    seq = []
                else:
                    seq.append(line.strip())
            cleaned = _clean_seq("".join(seq))
            if cleaned:
                rows[landscape].append(
                    {"sequence": cleaned, "wt": "", "score": float("nan"), "mutant": "", "dms_id": f"msa:{base}"}
                )


def write_landscapes(rows: dict[str, list[dict]], dest: Path) -> None:
    dest = _out_root(dest)
    for landscape, recs in rows.items():
        out_dir = dest / landscape
        out_dir.mkdir(parents=True, exist_ok=True)
        seen = set()
        seqs = []
        lookup = {}
        dms_ids = set()
        lengths = []
        for rec in recs:
            seq = rec["sequence"]
            if not seq or seq in seen:
                continue
            seen.add(seq)
            seqs.append(seq)
            lengths.append(len(seq))
            dms_ids.add(str(rec.get("dms_id") or ""))
            score = rec.get("score")
            if score == score:  # not NaN
                lookup[seq] = float(score)
            wt = rec.get("wt") or ""
            if wt and wt not in seen:
                seen.add(wt)
                seqs.append(wt)
                lengths.append(len(wt))
        csv_path = out_dir / "sequences.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["sequence"])
            for s in seqs:
                w.writerow([s])
        lookup_path = out_dir / "dms_lookup.json"
        lookup_path.write_text(json.dumps(lookup, indent=0), encoding="utf-8")
        meta = {
            "landscape": landscape,
            "n_seqs": len(seqs),
            "n_scored": len(lookup),
            "dms_ids": sorted(x for x in dms_ids if x),
            "max_len": max(lengths) if lengths else 0,
            "mean_len": (sum(lengths) / len(lengths)) if lengths else 0,
            "prefixes": list(LANDSCAPES[landscape]["prefixes"]),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"wrote": str(out_dir), **{k: meta[k] for k in ("n_seqs", "n_scored", "max_len")}}))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip", type=Path, default=Path("/root/data/proteingym/DMS_ProteinGym_substitutions.zip"))
    p.add_argument("--msa-zip", type=Path, default=Path("/root/data/proteingym/DMS_msa_files.zip"))
    p.add_argument(
        "--dest",
        type=Path,
        default=ROOT / "history_data/domain_protein/proteingym",
    )
    p.add_argument("--huggingface", action="store_true")
    args = p.parse_args(argv)

    rows: dict[str, list[dict]] = defaultdict(list)
    zip_ok = args.zip.is_file() and args.zip.stat().st_size > 1_000_000
    if zip_ok and not args.huggingface:
        print(f"[proteingym] extracting {args.zip}")
        rows = extract_from_zip(args.zip)
    if not rows:
        print("[proteingym] streaming HuggingFace OATML-Markslab/ProteinGym_v1")
        rows = extract_from_huggingface()
    if args.msa_zip.is_file() and args.msa_zip.stat().st_size > 1_000_000:
        print(f"[proteingym] adding MSA sequences from {args.msa_zip}")
        add_msa_sequences(args.msa_zip, rows)
    if not rows:
        raise SystemExit("no GFP/AAV/GB1 rows found")
    write_landscapes(rows, args.dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
