#!/usr/bin/env python3
"""Build an experience JSONL bank from OpenBox history dumps."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.components.experience.builder import build_experience_from_bo_json, load_trials_from_bo_json
from mindware.components.experience.contrastive import fit_bank_weights, save_weights
from mindware.components.experience.profile import load_dataset_info_table
from mindware.components.experience.retrieve import ExperienceBank, attach_retrieval_index
from mindware.components.experience.tabpfn_embed import tabpfn_available


def _iter_history_files(src: Path) -> list[Path]:
    """Prefer nested history_data/history_data/{cls,rgs}/{metric}/{ds}/{algo}.json."""
    nested = src / "history_data"
    root = nested if nested.is_dir() and any(nested.iterdir()) else src
    files = sorted(root.glob("cls/*/*/*.json")) + sorted(root.glob("rgs/*/*/*.json"))
    # Drop LLM eval dumps that live one directory deeper: .../{ds}/{variant}/{algo}.json
    files = [p for p in files if p.parent.parent.parent.name in {"cls", "rgs"}]
    return files


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--history-root",
        type=Path,
        default=ROOT / "history_data",
        help="Root that contains cls/ and rgs/, or a nested history_data/ folder.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "history_data" / "experience_bank.jsonl",
    )
    p.add_argument("--min-trials", type=int, default=8)
    p.add_argument("--max-index-trials", type=int, default=40)
    p.add_argument("--max-files", type=int, default=0, help="0 = all")
    p.add_argument("--fit-weights", action="store_true", default=True)
    p.add_argument("--no-fit-weights", action="store_false", dest="fit_weights")
    p.add_argument("--max-pairs", type=int, default=400)
    args = p.parse_args(argv)

    files = _iter_history_files(args.history_root)
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        raise SystemExit(f"No history JSON under {args.history_root}")

    info = load_dataset_info_table(ROOT)
    print(f"TabPFN available: {tabpfn_available()}")
    print(f"Building bank from {len(files)} histories → {args.output}")

    rows = []
    skipped = 0
    errors = 0
    for i, path in enumerate(files, 1):
        try:
            parts = path.parts
            # .../{cls|rgs}/{metric}/{dataset}/{algo}.json
            algo = path.stem
            dataset = path.parent.name
            metric = path.parent.parent.name
            tt_short = path.parent.parent.parent.name
            task_type = "classification" if tt_short == "cls" else "regression"
            higher = metric not in {"mse", "mae", "rmse"}
            trials = load_trials_from_bo_json(
                path,
                algorithm=algo,
                higher_is_better=True if higher else False,
                openbox_minimize=True,
            )
            if len(trials) < args.min_trials:
                skipped += 1
                continue
            profile = info.get((task_type, dataset)) or info.get((tt_short, dataset)) or {}
            exp = build_experience_from_bo_json(
                path,
                optimizer="bo_history",
                dataset_id=dataset,
                algorithm_id=algo,
                task_type=task_type,
                metric=metric,
                higher_is_better=higher,
                profile=profile,
            )
            attach_retrieval_index(
                exp,
                trials,
                profile_vec=profile,
                max_trials=args.max_index_trials,
            )
            # Drop bulky unused fields from the bank line
            if "embed_text" in (exp.get("delivery") or {}):
                # keep prompt_block; embed_text is unused by the cascade
                exp["delivery"].pop("embed_text", None)
            rows.append(exp)
        except Exception as exc:
            errors += 1
            print(f"[skip] {path}: {exc}")
            if errors <= 3:
                traceback.print_exc()
        if i % 50 == 0:
            print(f"  {i}/{len(files)} scanned, {len(rows)} kept, {skipped} short, {errors} errors")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for exp in rows:
            f.write(json.dumps(exp, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} experiences to {args.output}")

    bank = ExperienceBank(experiences=rows)
    if args.fit_weights and len(rows) >= 16:
        weights = fit_bank_weights(bank, max_pairs=args.max_pairs)
        wpath = args.output.with_suffix(".weights.json")
        save_weights(weights, wpath)
        print(f"Contrastive weights → {wpath}: {weights}")
    else:
        print("Skipped weight fit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
