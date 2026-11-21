"""Stable TUNE (70) / LOCK (25) dataset split. Do not retune on LOCK."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

SPLIT_PATH = Path(__file__).resolve().parent / "tune_lock_split.json"
SPLIT_SEED = "lodo-tune-lock-v1"
N_LOCK = 25


def dataset_hash(name: str) -> str:
    return hashlib.sha256(f"{SPLIT_SEED}:{name.lower()}".encode("utf-8")).hexdigest()


def make_split(datasets: Sequence[str], n_lock: int = N_LOCK) -> dict:
    unique = sorted({str(d) for d in datasets if d})
    ranked = sorted(unique, key=dataset_hash)
    lock = ranked[:n_lock]
    tune = ranked[n_lock:]
    return {
        "seed": SPLIT_SEED,
        "n_lock": n_lock,
        "n_tune": len(tune),
        "n_total": len(unique),
        "lock": lock,
        "tune": tune,
        "policy": "sha256(seed:dataset) sort; first n_lock = LOCK; remainder = TUNE. Frozen weights. No fit on LOCK.",
    }


def load_split(path: Path = SPLIT_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def is_tune(dataset: str, split: dict | None = None) -> bool:
    split = split or load_split()
    return str(dataset).lower() in {x.lower() for x in split["tune"]}


def is_lock(dataset: str, split: dict | None = None) -> bool:
    split = split or load_split()
    return str(dataset).lower() in {x.lower() for x in split["lock"]}


def datasets_from_bank(bank_path: Path) -> list[str]:
    names = []
    with Path(bank_path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ds = ((rec.get("problem") or {}).get("dataset_id") or "").strip()
            if ds:
                names.append(ds)
    return sorted(set(names))


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bank",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "history_data" / "experience_bank.jsonl",
    )
    p.add_argument("--output", type=Path, default=SPLIT_PATH)
    args = p.parse_args()
    payload = make_split(datasets_from_bank(args.bank))
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {args.output}: n_total={payload['n_total']} "
        f"tune={payload['n_tune']} lock={payload['n_lock']}"
    )
    print("LOCK:", ", ".join(payload["lock"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
