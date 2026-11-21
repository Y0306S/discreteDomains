#!/usr/bin/env python3
"""Build domain experience JSONL from history_data/domain_{protein,query,dl}."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.components.experience.builder import build_experience_from_bo_json
from mindware.domains.dlhpo.zaxie import DL_ROLE_HINTS
from mindware.domains.paths import domain_history
from mindware.domains.protein.features import PROTEIN_ROLE_HINTS
from mindware.domains.queryopt.sql_parse import QUERY_ROLE_HINTS

ROLE_MAP = {
    "protein": PROTEIN_ROLE_HINTS,
    "query": QUERY_ROLE_HINTS,
    "dl": DL_ROLE_HINTS,
}

METRIC_HIGHER = {
    "fitness": True,
    "bal_acc": True,
    "val_loss": False,
    "cost": False,
    "latency_ms": False,
}


def _iter_histories(root: Path) -> list[Path]:
    skip = {
        "tune_lock_split.json",
        "experience_bank.jsonl",
        "sequence_vae.meta.json",
        "library_pairs.json",
        "bolt_vae.meta.json",
    }
    skip_stems = {"synthetic_tpe", "library_pairs"}
    files = []
    for p in Path(root).rglob("*.json"):
        if p.name in skip or p.stem in skip_stems or "vae" in p.parts:
            continue
        if p.parent.name in {"apexgo_gramneg", "apexgo_mean", "bolt_peptides"}:
            continue
        if p.parent.parent.name in {"fitness", "cost", "val_loss"}:
            files.append(p)
    return sorted(files)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=["protein", "query", "dl"], required=True)
    p.add_argument("--history-root", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--min-trials", type=int, default=8)
    args = p.parse_args(argv)

    hist_root = args.history_root or domain_history(args.domain)
    out = args.output or (hist_root / "experience_bank.jsonl")
    files = _iter_histories(hist_root)
    if not files:
        raise SystemExit(f"No histories under {hist_root}")

    rows = []
    skipped = 0
    for path in files:
        try:
            metric = path.parent.parent.name
            dataset = path.parent.name
            algo = path.stem
            higher = METRIC_HIGHER.get(metric, False)
            exp = build_experience_from_bo_json(
                path,
                dataset_id=dataset,
                algorithm_id=algo,
                domain=args.domain,
                task_type=args.domain,
                metric=metric,
                higher_is_better=higher,
                extra_role_hints=ROLE_MAP[args.domain],
            )
        except Exception as exc:
            skipped += 1
            print(f"skip {path}: {exc}")
            continue
        n = int((exp.get("result") or {}).get("n_success") or 0)
        if n < args.min_trials:
            skipped += 1
            continue
        exp["delivery"]["filters"]["domain"] = args.domain
        rows.append(exp)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {out} n={len(rows)} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Build domain experience JSONL from history_data/domain_{protein,query,dl}."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.components.experience.builder import build_experience_from_bo_json
from mindware.domains.dlhpo.zaxie import DL_ROLE_HINTS
from mindware.domains.paths import domain_history
from mindware.domains.protein.features import PROTEIN_ROLE_HINTS
from mindware.domains.queryopt.sql_parse import QUERY_ROLE_HINTS

ROLE_MAP = {
    "protein": PROTEIN_ROLE_HINTS,
    "query": QUERY_ROLE_HINTS,
    "dl": DL_ROLE_HINTS,
}

METRIC_HIGHER = {
    "fitness": True,
    "bal_acc": True,
    "val_loss": False,
    "cost": False,
    "latency_ms": False,
}


def _iter_histories(root: Path) -> list[Path]:
    skip = {
        "tune_lock_split.json",
        "experience_bank.jsonl",
        "sequence_vae.meta.json",
        "library_pairs.json",
        "bolt_vae.meta.json",
    }
    skip_stems = {"synthetic_tpe", "library_pairs"}
    files = []
    for p in Path(root).rglob("*.json"):
        if p.name in skip or p.stem in skip_stems or "vae" in p.parts:
            continue
        if p.parent.name in {"apexgo_gramneg", "apexgo_mean", "bolt_peptides"}:
            continue
        if p.parent.parent.name in {"fitness", "cost", "val_loss"}:
            files.append(p)
    return sorted(files)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=["protein", "query", "dl"], required=True)
    p.add_argument("--history-root", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--min-trials", type=int, default=8)
    args = p.parse_args(argv)

    hist_root = args.history_root or domain_history(args.domain)
    out = args.output or (hist_root / "experience_bank.jsonl")
    files = _iter_histories(hist_root)
    if not files:
        raise SystemExit(f"No histories under {hist_root}")

    rows = []
    skipped = 0
    for path in files:
        try:
            metric = path.parent.parent.name
            dataset = path.parent.name
            algo = path.stem
            higher = METRIC_HIGHER.get(metric, False)
            exp = build_experience_from_bo_json(
                path,
                dataset_id=dataset,
                algorithm_id=algo,
                domain=args.domain,
                task_type=args.domain,
                metric=metric,
                higher_is_better=higher,
                extra_role_hints=ROLE_MAP[args.domain],
            )
        except Exception as exc:
            skipped += 1
            print(f"skip {path}: {exc}")
            continue
        n = int((exp.get("result") or {}).get("n_success") or 0)
        if n < args.min_trials:
            skipped += 1
            continue
        exp["delivery"]["filters"]["domain"] = args.domain
        rows.append(exp)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {out} n={len(rows)} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Build domain experience JSONL from history_data/domain_{protein,query,dl}."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.components.experience.builder import build_experience_from_bo_json
from mindware.domains.dlhpo.zaxie import DL_ROLE_HINTS
from mindware.domains.paths import domain_history
from mindware.domains.protein.features import PROTEIN_ROLE_HINTS
from mindware.domains.queryopt.sql_parse import QUERY_ROLE_HINTS

ROLE_MAP = {
    "protein": PROTEIN_ROLE_HINTS,
    "query": QUERY_ROLE_HINTS,
    "dl": DL_ROLE_HINTS,
}

METRIC_HIGHER = {
    "fitness": True,
    "bal_acc": True,
    "val_loss": False,
    "cost": False,
    "latency_ms": False,
}


def _iter_histories(root: Path) -> list[Path]:
    skip = {
        "tune_lock_split.json",
        "experience_bank.jsonl",
        "sequence_vae.meta.json",
        "library_pairs.json",
        "bolt_vae.meta.json",
    }
    skip_stems = {"synthetic_tpe", "library_pairs"}
    files = []
    for p in Path(root).rglob("*.json"):
        if p.name in skip or p.stem in skip_stems or "vae" in p.parts:
            continue
        if p.parent.name in {"apexgo_gramneg", "apexgo_mean", "bolt_peptides"}:
            continue
        if p.parent.parent.name in {"fitness", "cost", "val_loss"}:
            files.append(p)
    return sorted(files)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=["protein", "query", "dl"], required=True)
    p.add_argument("--history-root", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--min-trials", type=int, default=8)
    args = p.parse_args(argv)

    hist_root = args.history_root or domain_history(args.domain)
    out = args.output or (hist_root / "experience_bank.jsonl")
    files = _iter_histories(hist_root)
    if not files:
        raise SystemExit(f"No histories under {hist_root}")

    rows = []
    skipped = 0
    for path in files:
        try:
            metric = path.parent.parent.name
            dataset = path.parent.name
            algo = path.stem
            higher = METRIC_HIGHER.get(metric, False)
            exp = build_experience_from_bo_json(
                path,
                dataset_id=dataset,
                algorithm_id=algo,
                domain=args.domain,
                task_type=args.domain,
                metric=metric,
                higher_is_better=higher,
                extra_role_hints=ROLE_MAP[args.domain],
            )
        except Exception as exc:
            skipped += 1
            print(f"skip {path}: {exc}")
            continue
        n = int((exp.get("result") or {}).get("n_success") or 0)
        if n < args.min_trials:
            skipped += 1
            continue
        exp["delivery"]["filters"]["domain"] = args.domain
        rows.append(exp)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {out} n={len(rows)} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Build domain experience JSONL from history_data/domain_{protein,query,dl}."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.components.experience.builder import build_experience_from_bo_json
from mindware.domains.dlhpo.zaxie import DL_ROLE_HINTS
from mindware.domains.paths import domain_history
from mindware.domains.protein.features import PROTEIN_ROLE_HINTS
from mindware.domains.queryopt.sql_parse import QUERY_ROLE_HINTS

ROLE_MAP = {
    "protein": PROTEIN_ROLE_HINTS,
    "query": QUERY_ROLE_HINTS,
    "dl": DL_ROLE_HINTS,
}

METRIC_HIGHER = {
    "fitness": True,
    "bal_acc": True,
    "val_loss": False,
    "cost": False,
    "latency_ms": False,
}


def _iter_histories(root: Path) -> list[Path]:
    skip = {
        "tune_lock_split.json",
        "experience_bank.jsonl",
        "sequence_vae.meta.json",
        "library_pairs.json",
        "bolt_vae.meta.json",
    }
    skip_stems = {"synthetic_tpe", "library_pairs"}
    files = []
    for p in Path(root).rglob("*.json"):
        if p.name in skip or p.stem in skip_stems or "vae" in p.parts:
            continue
        if p.parent.name in {"apexgo_gramneg", "apexgo_mean", "bolt_peptides"}:
            continue
        if p.parent.parent.name in {"fitness", "cost", "val_loss"}:
            files.append(p)
    return sorted(files)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=["protein", "query", "dl"], required=True)
    p.add_argument("--history-root", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--min-trials", type=int, default=8)
    args = p.parse_args(argv)

    hist_root = args.history_root or domain_history(args.domain)
    out = args.output or (hist_root / "experience_bank.jsonl")
    files = _iter_histories(hist_root)
    if not files:
        raise SystemExit(f"No histories under {hist_root}")

    rows = []
    skipped = 0
    for path in files:
        try:
            metric = path.parent.parent.name
            dataset = path.parent.name
            algo = path.stem
            higher = METRIC_HIGHER.get(metric, False)
            exp = build_experience_from_bo_json(
                path,
                dataset_id=dataset,
                algorithm_id=algo,
                domain=args.domain,
                task_type=args.domain,
                metric=metric,
                higher_is_better=higher,
                extra_role_hints=ROLE_MAP[args.domain],
            )
        except Exception as exc:
            skipped += 1
            print(f"skip {path}: {exc}")
            continue
        n = int((exp.get("result") or {}).get("n_success") or 0)
        if n < args.min_trials:
            skipped += 1
            continue
        exp["delivery"]["filters"]["domain"] = args.domain
        rows.append(exp)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {out} n={len(rows)} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
