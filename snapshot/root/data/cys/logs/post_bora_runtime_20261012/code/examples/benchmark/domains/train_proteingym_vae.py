#!/usr/bin/env python3
"""Train landscape-specific ProteinGym VAEs (GFP / AAV)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains.protein.torch_vae import ProteinGymVAE, train_proteingym_vae

DEFAULT_DEST = ROOT / "history_data/domain_protein/proteingym"
VAE_OUT = ROOT / "history_data/domain_protein/proteingym/vae"

LANDSCAPE_DEFAULTS = {
    "gfp": {"max_len": 238, "latent_dim": 32, "epochs": 25, "batch_size": 64},
    "aav": {"max_len": 735, "latent_dim": 64, "epochs": 20, "batch_size": 24},
}


def load_train_seqs(dest: Path, landscape: str, max_seqs: int | None = None) -> list[str]:
    scored = dest / landscape / "sequences_scored.csv"
    if not scored.is_file():
        raise SystemExit(f"missing {scored}; run prepare_proteingym_scored.py first")
    seqs = []
    with scored.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("split") == "train":
                seqs.append(row["sequence"])
    if max_seqs is not None and len(seqs) > int(max_seqs):
        rng = __import__("numpy").random.RandomState(0)
        idx = rng.choice(len(seqs), size=int(max_seqs), replace=False)
        seqs = [seqs[int(i)] for i in sorted(idx.tolist())]
    return seqs


def diagnose(vae: ProteinGymVAE, seqs: list[str], n: int = 20) -> dict:
    import numpy as np

    rng = np.random.RandomState(0)
    hit = 0
    unique = set()
    lens = []
    for i in range(min(n, len(seqs))):
        s = seqs[i]
        z = vae.encode(s)
        d = vae.decode_string(z, temperature=0.01)
        if d == s:
            hit += 1
        lens.append(len(d))
    for _ in range(n):
        z = [float(rng.randn()) for _ in range(vae.latent_dim)]
        unique.add(vae.decode_string(z, temperature=0.8))
    return {
        "exact_roundtrip": hit,
        "exact_roundtrip_n": min(n, len(seqs)),
        "rand_unique": len(unique),
        "rand_n": n,
        "mean_decode_len": float(sum(lens) / max(len(lens), 1)),
        "target_len": vae.max_len,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--landscape", choices=["gfp", "aav"], required=True)
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--latent-dim", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-train-seqs", type=int, default=None)
    args = p.parse_args()

    cfg = dict(LANDSCAPE_DEFAULTS[args.landscape])
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.latent_dim is not None:
        cfg["latent_dim"] = args.latent_dim

    seqs = load_train_seqs(args.dest, args.landscape, max_seqs=args.max_train_seqs)
    print(json.dumps({"landscape": args.landscape, "n_train": len(seqs), **cfg, "device": args.device}))
    VAE_OUT.mkdir(parents=True, exist_ok=True)
    out = VAE_OUT / f"{args.landscape}_vae.pt"
    vae = train_proteingym_vae(
        seqs,
        max_len=cfg["max_len"],
        latent_dim=cfg["latent_dim"],
        device=args.device,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=args.lr,
        beta=args.beta,
        out_path=out,
        seed=args.seed,
    )
    diag = diagnose(vae, seqs[:50], n=20)
    diag_path = VAE_OUT / f"{args.landscape}_vae_diag.json"
    diag_path.write_text(json.dumps(diag, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ckpt": str(out), "diag": diag}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
