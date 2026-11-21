#!/usr/bin/env python3
"""Train a per-landscape InfoTransformer VAE (APEXGo recipe) on GPU 0.

Same hparams as generation/train.sh. Shorter/longer max_string_length per assay.
Uses UniRef 1-mer vocab so checkpoints load through BOLT load_vae.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("WANDB_MODE", "offline")

APEX_GEN = Path("/root/data/cys/APEXGo/generation")
BOLT_PEP = Path("/root/data/cys/BOLT/optimization/peptides")
ROOT = Path(__file__).resolve().parents[3]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-epochs", type=int, default=118)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-string-length", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    args = p.parse_args(argv)

    os.chdir(APEX_GEN)
    if str(APEX_GEN) not in sys.path:
        sys.path.insert(0, str(APEX_GEN))

    import torch
    from data import DataModuleKmers, DatasetKmers
    from vae import InfoTransformerVAE

    vocab_path = BOLT_PEP / "uniref_vae/1mer_vocab.csv"
    import pandas as pd

    vocab = pd.read_csv(vocab_path, header=None).values.squeeze().tolist()
    vocab2idx = {v: i for i, v in enumerate(vocab)}

    datamodule = DataModuleKmers(
        args.batch_size, k=1, version=1, load_data=True, data_path=str(args.csv)
    )
    # Re-bind UniRef k=1 vocab so the state dict matches BOLT load_vae.
    for split in (datamodule.train, datamodule.val, datamodule.test):
        split.vocab = vocab
        split.vocab2idx = vocab2idx

    model = InfoTransformerVAE(
        dataset=datamodule.train,
        d_model=128,
        kl_factor=0.0001,
        encoder_dropout=0.05,
        decoder_dropout=0.05,
        encoder_dim_feedforward=256,
        decoder_dim_feedforward=256,
        encoder_num_layers=6,
        decoder_num_layers=6,
    )
    model.max_string_length = int(args.max_string_length)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"device": str(device), "visible": os.environ.get("CUDA_VISIBLE_DEVICES")}), flush=True)
    model = model.to(device)
    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    t0 = time.time()
    for epoch in range(int(args.max_epochs)):
        model.train()
        running = 0.0
        n = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                out = model(batch)
                loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())
            n += 1
        train_loss = running / max(1, n)
        val_loss = None
        if (epoch + 1) % 3 == 0 or epoch + 1 == int(args.max_epochs):
            model.eval()
            vsum, vn = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch)
                    vsum += float(out["loss"].detach())
                    vn += 1
            val_loss = vsum / max(1, vn)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), args.out)
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "train_loss": round(train_loss, 4),
                    "val_loss": None if val_loss is None else round(val_loss, 4),
                    "secs": round(time.time() - t0, 1),
                }
            ),
            flush=True,
        )

    final_path = args.out.with_name(args.out.stem + f"_epoch_{args.max_epochs}.pkl")
    torch.save(model.state_dict(), final_path)
    if not args.out.is_file():
        torch.save(model.state_dict(), args.out)
    meta = {
        "ckpt": str(args.out),
        "final": str(final_path),
        "max_epochs": int(args.max_epochs),
        "max_string_length": int(args.max_string_length),
        "batch_size": int(args.batch_size),
        "latent_dim": 256,
        "d_model": 128,
        "k": 1,
        "best_val": best_val,
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[landscape_vae] wrote {args.out} and {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
