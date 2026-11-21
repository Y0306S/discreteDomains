#!/usr/bin/env python3
"""Bootstrap fake protein BO from BOLT/APEXGo scored sequence libraries.

Trains (or loads) a sequence VAE, encodes each scored seq into z, and writes
OpenBox History JSON with decoded sequence features attached to the config.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains.history import observation, sort_improving, write_openbox_history
from mindware.domains.paths import APEXGO_INIT, BOLT_PEPTIDES, domain_history
from mindware.domains.protein.features import extra_from_seq
from mindware.domains.protein.oracle import load_seq_score_pairs
from mindware.domains.sequence_vae import AA_ALPHABET, SequenceVAE, train_sequence_vae


def _landscape_specs() -> list[dict]:
    return [
        {
            "task_id": "bolt_peptides",
            "seq": BOLT_PEPTIDES / "apex_oracle/init_data/seed_0_init.txt",
            "score": BOLT_PEPTIDES / "apex_oracle/init_data/seed_0_scores.csv",
        },
        {
            "task_id": "apexgo_mean",
            "seq": APEXGO_INIT / "init_seqs.csv",
            "score": APEXGO_INIT / "mean_scores.csv",
        },
        {
            "task_id": "apexgo_gramneg",
            "seq": APEXGO_INIT / "init_seqs.csv",
            "score": APEXGO_INIT / "gramnegonly_mean_scores.csv",
        },
    ]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=domain_history("protein"))
    p.add_argument("--vae-out", type=Path, default=None)
    p.add_argument("--limit", type=int, default=400)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--improving-order", action="store_true", default=True)
    args = p.parse_args(argv)

    specs = [s for s in _landscape_specs() if Path(s["seq"]).is_file() and Path(s["score"]).is_file()]
    if not specs:
        raise SystemExit("No protein libraries found under BOLT/APEXGo")

    all_seqs = []
    tables = {}
    for spec in specs:
        table = load_seq_score_pairs(spec["seq"], spec["score"])
        tables[spec["task_id"]] = table
        all_seqs.extend(list(table)[: args.limit])

    vae_path = args.vae_out or (args.out / "vae/sequence_vae.pkl")
    vae = train_sequence_vae(
        all_seqs[: max(args.limit, 200)],
        list(AA_ALPHABET),
        out_path=vae_path,
        steps=args.steps,
        max_len=48,
    )

    written = []
    for spec in specs:
        table = tables[spec["task_id"]]
        items = list(table.items())[: args.limit]
        rows = []
        for seq, score in items:
            z = vae.encode(seq)
            # Library scores are typically higher-is-better (negative MIC). Store minimize = -score.
            obj = -float(score) if math.isfinite(score) else 1e10
            cfg = {f"z_{i}": float(v) for i, v in enumerate(z)}
            cfg.update(extra_from_seq(seq))
            cfg["sequence"] = seq
            rows.append(observation(cfg, obj, extra={"source": "library_pairs", "sequence": seq}))
        if args.improving_order:
            rows = sort_improving(rows, minimize=True)
        dest = args.out / "fitness" / spec["task_id"] / "library_pairs.json"
        write_openbox_history(
            dest,
            task_id=spec["task_id"],
            observations=rows,
            meta={
                "source": "protein_scored_library",
                "n": len(rows),
                "family_hash": hashlib.sha256(spec["task_id"].encode()).hexdigest()[:8],
                "vae": str(vae_path),
            },
        )
        written.append(dest)
        print(f"Wrote {dest} n={len(rows)}")
    print(f"VAE {vae_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Bootstrap fake protein BO from BOLT/APEXGo scored sequence libraries.

Trains (or loads) a sequence VAE, encodes each scored seq into z, and writes
OpenBox History JSON with decoded sequence features attached to the config.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains.history import observation, sort_improving, write_openbox_history
from mindware.domains.paths import APEXGO_INIT, BOLT_PEPTIDES, domain_history
from mindware.domains.protein.features import extra_from_seq
from mindware.domains.protein.oracle import load_seq_score_pairs
from mindware.domains.sequence_vae import AA_ALPHABET, SequenceVAE, train_sequence_vae


def _landscape_specs() -> list[dict]:
    return [
        {
            "task_id": "bolt_peptides",
            "seq": BOLT_PEPTIDES / "apex_oracle/init_data/seed_0_init.txt",
            "score": BOLT_PEPTIDES / "apex_oracle/init_data/seed_0_scores.csv",
        },
        {
            "task_id": "apexgo_mean",
            "seq": APEXGO_INIT / "init_seqs.csv",
            "score": APEXGO_INIT / "mean_scores.csv",
        },
        {
            "task_id": "apexgo_gramneg",
            "seq": APEXGO_INIT / "init_seqs.csv",
            "score": APEXGO_INIT / "gramnegonly_mean_scores.csv",
        },
    ]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=domain_history("protein"))
    p.add_argument("--vae-out", type=Path, default=None)
    p.add_argument("--limit", type=int, default=400)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--improving-order", action="store_true", default=True)
    args = p.parse_args(argv)

    specs = [s for s in _landscape_specs() if Path(s["seq"]).is_file() and Path(s["score"]).is_file()]
    if not specs:
        raise SystemExit("No protein libraries found under BOLT/APEXGo")

    all_seqs = []
    tables = {}
    for spec in specs:
        table = load_seq_score_pairs(spec["seq"], spec["score"])
        tables[spec["task_id"]] = table
        all_seqs.extend(list(table)[: args.limit])

    vae_path = args.vae_out or (args.out / "vae/sequence_vae.pkl")
    vae = train_sequence_vae(
        all_seqs[: max(args.limit, 200)],
        list(AA_ALPHABET),
        out_path=vae_path,
        steps=args.steps,
        max_len=48,
    )

    written = []
    for spec in specs:
        table = tables[spec["task_id"]]
        items = list(table.items())[: args.limit]
        rows = []
        for seq, score in items:
            z = vae.encode(seq)
            # Library scores are typically higher-is-better (negative MIC). Store minimize = -score.
            obj = -float(score) if math.isfinite(score) else 1e10
            cfg = {f"z_{i}": float(v) for i, v in enumerate(z)}
            cfg.update(extra_from_seq(seq))
            cfg["sequence"] = seq
            rows.append(observation(cfg, obj, extra={"source": "library_pairs", "sequence": seq}))
        if args.improving_order:
            rows = sort_improving(rows, minimize=True)
        dest = args.out / "fitness" / spec["task_id"] / "library_pairs.json"
        write_openbox_history(
            dest,
            task_id=spec["task_id"],
            observations=rows,
            meta={
                "source": "protein_scored_library",
                "n": len(rows),
                "family_hash": hashlib.sha256(spec["task_id"].encode()).hexdigest()[:8],
                "vae": str(vae_path),
            },
        )
        written.append(dest)
        print(f"Wrote {dest} n={len(rows)}")
    print(f"VAE {vae_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
