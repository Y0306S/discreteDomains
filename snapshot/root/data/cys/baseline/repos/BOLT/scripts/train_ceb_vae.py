#!/usr/bin/env python3
"""Train query-plan InfoTransformer VAE (CEB_64.ckpt) from encoded plan CSVs.

Expects CSVs from training_data/codec.py `encode_alias_plans` (comma-separated flat encodings).
Run from repo with query_plan env on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

# Single-GPU paper pipeline: never enumerate the other A800s.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

BOLT_QP = Path(__file__).resolve().parents[1] / "optimization" / "query_plans" / "query_plan_optimization"
sys.path.insert(0, str(BOLT_QP))

from vae.model import VAEModule, build_vocab  # noqa: E402
from workload.schema import build_table_order  # noqa: E402
from workload.workloads import IMDB_SCHEMA_PATH, load_ceb3k  # noqa: E402


class FlatPlanDataset(Dataset):
    def __init__(self, rows: list[list[int]]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def collate_flat(batch):
    return batch


def load_encoded_csvs(encoded_dir: Path) -> list[list[int]]:
    rows: list[list[int]] = []
    for path in sorted(encoded_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            for row in reader:
                if len(row) < 3:
                    continue
                # alias export: query_id, plan_type, flat_encoding
                flat = row[-1]
                try:
                    tokens = [int(x) for x in flat.split(",") if x.strip()]
                except ValueError:
                    continue
                if len(tokens) >= 5 and len(tokens) % 5 == 0:
                    rows.append(tokens)
    return rows


def synthetic_rows(n: int, num_tables: int, num_joins: int, seed: int) -> list[list[int]]:
    """Random flat encodings (smoke test only — not valid join trees)."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        tokens = []
        for _ in range(num_joins):
            t1, a1 = rng.randrange(num_tables), rng.randrange(1, 4)
            t2, a2 = rng.randrange(num_tables), rng.randrange(1, 4)
            op = rng.choice([1, 2, 3])
            tokens.extend([t1, a1, t2, a2, op])
        rows.append(tokens)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encoded-dir",
        type=Path,
        default=BOLT_QP / "training_data" / "encoded_ceb",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=BOLT_QP / "vae" / "CEB_64.ckpt",
    )
    parser.add_argument("--d-neck", type=int, default=32)
    parser.add_argument("--num-aliases", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--gpu-mem-fraction", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--synthetic-n", type=int, default=5000)
    parser.add_argument("--init-ckpt", type=Path, default=None)
    parser.add_argument("--extra-csv", type=Path, action="append", default=[])
    parser.add_argument("--oversample-min-joins", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    load_ceb3k()
    table_order = build_table_order(IMDB_SCHEMA_PATH)
    num_tables = len(table_order)
    vocab, rev_vocab = build_vocab(num_tables, args.num_aliases)

    if args.synthetic_smoke:
        print("[train_ceb_vae] WARNING: using synthetic random encodings (smoke test only)")
        all_rows = synthetic_rows(args.synthetic_n, num_tables, num_joins=4, seed=42)
    else:
        if not args.encoded_dir.is_dir():
            print(f"[train_ceb_vae] missing encoded dir: {args.encoded_dir}")
            print("  Run scripts/collect_ceb_plans_for_vae.sh first (needs IMDB Postgres + workload.db).")
            return 1
        all_rows = load_encoded_csvs(args.encoded_dir)
        if not all_rows:
            print(f"[train_ceb_vae] no rows in {args.encoded_dir}")
            return 1
        extra_files = [Path(p) for p in (args.extra_csv or []) if Path(p).is_file()]
        encoded_files = {p.resolve() for p in args.encoded_dir.glob("*.csv")}
        for extra in extra_files:
            if extra.resolve() in encoded_files:
                print(f"[train_ceb_vae] extra already in encoded-dir, skip {extra}")
                continue
            before = len(all_rows)
            with extra.open(newline="", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=";")
                for row in reader:
                    if len(row) < 3:
                        continue
                    flat = row[-1]
                    try:
                        tokens = [int(x) for x in flat.split(",") if x.strip()]
                    except ValueError:
                        continue
                    if len(tokens) >= 5 and len(tokens) % 5 == 0:
                        all_rows.append(tokens)
            print(f"[train_ceb_vae] extra {extra} +{len(all_rows) - before}")

    if int(args.oversample_min_joins) > 0:
        wide = [r for r in all_rows if (len(r) // 5) >= int(args.oversample_min_joins)]
        if wide:
            n_add = len(wide)
            all_rows.extend(random.choices(wide, k=n_add))
            print(f"[train_ceb_vae] oversampled n_joins>={args.oversample_min_joins}: +{n_add}")

    random.shuffle(all_rows)
    n_val = max(1, int(len(all_rows) * args.val_fraction))
    val_rows = all_rows[:n_val]
    train_rows = all_rows[n_val:]
    print(f"[train_ceb_vae] train={len(train_rows)} val={len(val_rows)} tables={num_tables}")

    if args.gpus != 1:
        print("[train_ceb_vae] forcing 1 GPU (CUDA_VISIBLE_DEVICES=0)")
    if torch.cuda.is_available() and args.gpu_mem_fraction < 1.0:
        try:
            torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction, device=0)
        except Exception as exc:
            print(f"[train_ceb_vae] mem fraction on gpu0: {exc}")

    if args.init_ckpt and Path(args.init_ckpt).is_file():
        print(f"[train_ceb_vae] continue from {args.init_ckpt}")
        try:
            module = VAEModule.load_from_checkpoint(str(args.init_ckpt), map_location="cpu")
        except Exception as exc:
            print(f"[train_ceb_vae] load_from_checkpoint failed ({exc}); building fresh then loading state")
            module = VAEModule(
                vocab=vocab,
                rev_vocab=rev_vocab,
                bn_size=2,
                d_enc=128,
                d_dec=128,
                d_neck=args.d_neck,
                kl_factor=0.1,
            )
            import torch as _torch

            blob = _torch.load(str(args.init_ckpt), map_location="cpu")
            state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
            module.load_state_dict(state, strict=False)
    else:
        module = VAEModule(
            vocab=vocab,
            rev_vocab=rev_vocab,
            bn_size=2,
            d_enc=128,
            d_dec=128,
            d_neck=args.d_neck,
            kl_factor=0.1,
        )

    train_loader = DataLoader(
        FlatPlanDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_flat,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        FlatPlanDataset(val_rows),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_flat,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=args.out.parent,
        filename=args.out.stem,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[ckpt_cb],
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    trainer.fit(module, train_loader, val_loader)

    best = ckpt_cb.best_model_path
    if best:
        import shutil

        shutil.copy2(best, args.out)
        if args.out.name == "CEB_64.ckpt":
            link_64 = args.out.parent / "64.ckpt"
            if link_64.exists() or link_64.is_symlink():
                link_64.unlink()
            link_64.symlink_to(args.out.name)
        print(f"[train_ceb_vae] wrote {args.out} (from {best})")
    else:
        trainer.save_checkpoint(args.out)
        print(f"[train_ceb_vae] wrote {args.out}")

    meta = {
        "table_order": [str(t).lower() for t in table_order],
        "num_aliases": int(args.num_aliases),
        "d_neck": int(args.d_neck),
        "bn_size": 2,
        "latent_dim": 2 * int(args.d_neck),
        "max_string_length": 90,
        "encoded_dir": str(args.encoded_dir),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "synthetic": bool(args.synthetic_smoke),
    }
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(__import__("json").dumps(meta, indent=2) + "\n", encoding="utf-8")
    hist_meta = Path("/root/data/cys/blockdata_archive/llmoptimizer_benchmark/history_data/domain_query/vae/bolt_vae.meta.json")
    hist_meta.parent.mkdir(parents=True, exist_ok=True)
    hist_meta.write_text(meta_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[train_ceb_vae] wrote meta {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
