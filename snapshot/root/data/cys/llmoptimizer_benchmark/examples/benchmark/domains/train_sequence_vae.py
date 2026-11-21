#!/usr/bin/env python3
"""Train the shared sequence VAE (protein alphabet or join-order table ids)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains.paths import HISTORY_ROOT
from mindware.domains.sequence_vae import AA_ALPHABET, train_sequence_vae


def _load_protein_seqs(paths: list[Path], limit: int) -> list[str]:
    seqs: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith(">") or "," in line:
                continue
            if all(c.isalpha() for c in line):
                seqs.append(line.upper())
            if limit and len(seqs) >= limit:
                return seqs
    return seqs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alphabet", choices=["protein", "query"], default="protein")
    p.add_argument("--seq-file", type=Path, action="append", default=[])
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--max-len", type=int, default=48)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.alphabet == "protein":
        alphabet = list(AA_ALPHABET)
        default_files = [
            Path("/root/data/cys/BOLT/optimization/peptides/apex_oracle/init_data/seed_0_init.txt"),
            Path("/root/data/cys/APEXGo/optimization/apex_oracle/init_data/init_seqs.csv"),
        ]
        files = args.seq_file or default_files
        seqs = _load_protein_seqs(files, args.limit)
        out = args.out or (HISTORY_ROOT / "domain_protein/vae/sequence_vae.pkl")
    else:
        # Table-id alphabet: IMDB-ish tokens; sequences supplied as space-separated lines.
        alphabet = []
        seqs = []
        for path in args.seq_file:
            for line in path.read_text(encoding="utf-8").splitlines():
                toks = [t for t in line.strip().split() if t]
                if toks:
                    seqs.append(toks)
                    alphabet.extend(toks)
            if args.limit and len(seqs) >= args.limit:
                seqs = seqs[: args.limit]
                break
        alphabet = sorted(set(alphabet)) or ["title", "cast_info", "name"]
        out = args.out or (HISTORY_ROOT / "domain_query/vae/sequence_vae.pkl")

    if not seqs:
        raise SystemExit("No sequences found to train the VAE")
    vae = train_sequence_vae(
        seqs,
        alphabet,
        out_path=out,
        latent_dim=args.latent_dim,
        max_len=args.max_len,
        steps=args.steps,
        seed=args.seed,
    )
    print(f"Wrote {out}  n_seq={len(seqs)} latent={vae.latent_dim} trained={vae.trained}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Train the shared sequence VAE (protein alphabet or join-order table ids)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains.paths import HISTORY_ROOT
from mindware.domains.sequence_vae import AA_ALPHABET, train_sequence_vae


def _load_protein_seqs(paths: list[Path], limit: int) -> list[str]:
    seqs: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith(">") or "," in line:
                continue
            if all(c.isalpha() for c in line):
                seqs.append(line.upper())
            if limit and len(seqs) >= limit:
                return seqs
    return seqs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alphabet", choices=["protein", "query"], default="protein")
    p.add_argument("--seq-file", type=Path, action="append", default=[])
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--max-len", type=int, default=48)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.alphabet == "protein":
        alphabet = list(AA_ALPHABET)
        default_files = [
            Path("/root/data/cys/BOLT/optimization/peptides/apex_oracle/init_data/seed_0_init.txt"),
            Path("/root/data/cys/APEXGo/optimization/apex_oracle/init_data/init_seqs.csv"),
        ]
        files = args.seq_file or default_files
        seqs = _load_protein_seqs(files, args.limit)
        out = args.out or (HISTORY_ROOT / "domain_protein/vae/sequence_vae.pkl")
    else:
        # Table-id alphabet: IMDB-ish tokens; sequences supplied as space-separated lines.
        alphabet = []
        seqs = []
        for path in args.seq_file:
            for line in path.read_text(encoding="utf-8").splitlines():
                toks = [t for t in line.strip().split() if t]
                if toks:
                    seqs.append(toks)
                    alphabet.extend(toks)
            if args.limit and len(seqs) >= args.limit:
                seqs = seqs[: args.limit]
                break
        alphabet = sorted(set(alphabet)) or ["title", "cast_info", "name"]
        out = args.out or (HISTORY_ROOT / "domain_query/vae/sequence_vae.pkl")

    if not seqs:
        raise SystemExit("No sequences found to train the VAE")
    vae = train_sequence_vae(
        seqs,
        alphabet,
        out_path=out,
        latent_dim=args.latent_dim,
        max_len=args.max_len,
        steps=args.steps,
        seed=args.seed,
    )
    print(f"Wrote {out}  n_seq={len(seqs)} latent={vae.latent_dim} trained={vae.trained}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
