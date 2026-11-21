#!/usr/bin/env python3
"""Train BOLT query-plan VAEModule and write CEB_64.ckpt (64-d latent).

Uses left-deep encodings of real CEB join permutations (not Postgres). Matches
BOLT LOLBO dim=64 via bn_size=2 × d_neck=32. IMDB job/schema.sql is absent here,
so table ids follow the frozen alphabetical IMDB list in queryopt.ceb.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains.bolt_vae import QUERY_VAE_CKPT, QUERY_VAE_META, left_deep_flat, save_query_vae_meta
from mindware.domains.paths import QUERY_ROOT, domain_history
from mindware.domains.queryopt.ceb import IMDB_TABLES, load_sql, load_task_ids
from mindware.domains.queryopt.sql_parse import parse_tables


def _encodings(n_queries: int, n_perm: int, seed: int) -> tuple[list[list[int]], list[str]]:
    table_order = list(IMDB_TABLES)
    table_to_id = {t: i for i, t in enumerate(table_order)}
    rng = random.Random(seed)
    rows: list[list[int]] = []
    for tid in load_task_ids(limit=n_queries):
        sql = load_sql(tid)
        pairs = parse_tables(sql)
        tables = [t for t, _ in pairs]
        ids = [table_to_id.get(t, 0) for t in tables]
        if len(ids) < 2:
            continue
        rows.append(left_deep_flat(ids))
        for _ in range(max(0, n_perm)):
            perm = ids[:]
            rng.shuffle(perm)
            rows.append(left_deep_flat(perm, alias=rng.choice([1, 2, 3]), op=rng.choice([1, 2, 3])))
    rng.shuffle(rows)
    return rows, table_order


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=QUERY_VAE_CKPT)
    p.add_argument("--d-neck", type=int, default=32, help="latent dim = 2 * d_neck (BOLT LOLBO uses 64)")
    p.add_argument("--num-aliases", type=int, default=3)
    p.add_argument("--max-epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-queries", type=int, default=200)
    p.add_argument("--n-perm", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--devices", type=int, default=1)
    args = p.parse_args(argv)

    rows, table_order = _encodings(args.n_queries, args.n_perm, args.seed)
    if len(rows) < 32:
        raise SystemExit(f"too few encodings ({len(rows)}); check CEB SQL under {QUERY_ROOT}")
    print(f"[train_bolt_query_vae] encodings={len(rows)} tables={len(table_order)} d_neck={args.d_neck}")

    qp = str(QUERY_ROOT / "query_plan_optimization")
    if qp not in sys.path:
        sys.path.insert(0, qp)

    import torch
    from lightning import Trainer
    from torch.utils.data import DataLoader, Dataset

    from vae.model import VAEModule, build_vocab  # type: ignore

    class FlatPlanDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    def _scalarize(out):
        logged = {}
        for k, v in out.items():
            if k in ("z", "mu", "sigma"):
                continue
            if hasattr(v, "numel") and int(v.numel()) > 1:
                v = v.mean()
            logged[k] = v
        return logged

    def training_step(self, batch, batch_idx):
        out = self(batch)
        logged = {f"train/{k}": v for k, v in _scalarize(out).items()}
        self.log_dict(logged, prog_bar=True, logger=True, batch_size=len(batch))
        return logged["train/loss"]

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        logged = {f"val/{k}": v for k, v in _scalarize(out).items()}
        self.log_dict(logged, prog_bar=True, logger=True, batch_size=len(batch))
        return logged["val/loss"]

    VAEModule.training_step = training_step  # type: ignore[method-assign]
    VAEModule.validation_step = validation_step  # type: ignore[method-assign]

    vocab, rev_vocab = build_vocab(len(table_order), args.num_aliases)
    n_val = max(32, int(0.05 * len(rows)))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    module = VAEModule(
        vocab=vocab,
        rev_vocab=rev_vocab,
        bn_size=2,
        d_enc=128,
        d_dec=128,
        d_neck=int(args.d_neck),
        kl_factor=0.1,
    )
    train_loader = DataLoader(
        FlatPlanDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: b,
        num_workers=0,
    )
    val_loader = DataLoader(
        FlatPlanDataset(val_rows),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: b,
        num_workers=0,
    )
    log_dir = domain_history("query") / "vae/lightning_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        max_epochs=int(args.max_epochs),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=int(args.devices) if torch.cuda.is_available() else 1,
        default_root_dir=str(log_dir),
        enable_progress_bar=True,
        log_every_n_steps=20,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, train_loader, val_loader)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(out))
    hist_ckpt = domain_history("query") / "vae/CEB_64.ckpt"
    hist_ckpt.parent.mkdir(parents=True, exist_ok=True)
    if hist_ckpt.resolve() != out.resolve():
        shutil.copy2(out, hist_ckpt)
    save_query_vae_meta(
        QUERY_VAE_META,
        table_order=table_order,
        num_aliases=args.num_aliases,
        d_neck=int(args.d_neck),
        bn_size=2,
        max_string_length=48,
    )
    print(f"[train_bolt_query_vae] wrote {out} ({out.stat().st_size} bytes)")
    print(f"[train_bolt_query_vae] meta {QUERY_VAE_META}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Train BOLT query-plan VAEModule and write CEB_64.ckpt (64-d latent).

Uses left-deep encodings of real CEB join permutations (not Postgres). Matches
BOLT LOLBO dim=64 via bn_size=2 × d_neck=32. IMDB job/schema.sql is absent here,
so table ids follow the frozen alphabetical IMDB list in queryopt.ceb.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains.bolt_vae import QUERY_VAE_CKPT, QUERY_VAE_META, left_deep_flat, save_query_vae_meta
from mindware.domains.paths import QUERY_ROOT, domain_history
from mindware.domains.queryopt.ceb import IMDB_TABLES, load_sql, load_task_ids
from mindware.domains.queryopt.sql_parse import parse_tables


def _encodings(n_queries: int, n_perm: int, seed: int) -> tuple[list[list[int]], list[str]]:
    table_order = list(IMDB_TABLES)
    table_to_id = {t: i for i, t in enumerate(table_order)}
    rng = random.Random(seed)
    rows: list[list[int]] = []
    for tid in load_task_ids(limit=n_queries):
        sql = load_sql(tid)
        pairs = parse_tables(sql)
        tables = [t for t, _ in pairs]
        ids = [table_to_id.get(t, 0) for t in tables]
        if len(ids) < 2:
            continue
        rows.append(left_deep_flat(ids))
        for _ in range(max(0, n_perm)):
            perm = ids[:]
            rng.shuffle(perm)
            rows.append(left_deep_flat(perm, alias=rng.choice([1, 2, 3]), op=rng.choice([1, 2, 3])))
    rng.shuffle(rows)
    return rows, table_order


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=QUERY_VAE_CKPT)
    p.add_argument("--d-neck", type=int, default=32, help="latent dim = 2 * d_neck (BOLT LOLBO uses 64)")
    p.add_argument("--num-aliases", type=int, default=3)
    p.add_argument("--max-epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-queries", type=int, default=200)
    p.add_argument("--n-perm", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--devices", type=int, default=1)
    args = p.parse_args(argv)

    rows, table_order = _encodings(args.n_queries, args.n_perm, args.seed)
    if len(rows) < 32:
        raise SystemExit(f"too few encodings ({len(rows)}); check CEB SQL under {QUERY_ROOT}")
    print(f"[train_bolt_query_vae] encodings={len(rows)} tables={len(table_order)} d_neck={args.d_neck}")

    qp = str(QUERY_ROOT / "query_plan_optimization")
    if qp not in sys.path:
        sys.path.insert(0, qp)

    import torch
    from lightning import Trainer
    from torch.utils.data import DataLoader, Dataset

    from vae.model import VAEModule, build_vocab  # type: ignore

    class FlatPlanDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    def _scalarize(out):
        logged = {}
        for k, v in out.items():
            if k in ("z", "mu", "sigma"):
                continue
            if hasattr(v, "numel") and int(v.numel()) > 1:
                v = v.mean()
            logged[k] = v
        return logged

    def training_step(self, batch, batch_idx):
        out = self(batch)
        logged = {f"train/{k}": v for k, v in _scalarize(out).items()}
        self.log_dict(logged, prog_bar=True, logger=True, batch_size=len(batch))
        return logged["train/loss"]

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        logged = {f"val/{k}": v for k, v in _scalarize(out).items()}
        self.log_dict(logged, prog_bar=True, logger=True, batch_size=len(batch))
        return logged["val/loss"]

    VAEModule.training_step = training_step  # type: ignore[method-assign]
    VAEModule.validation_step = validation_step  # type: ignore[method-assign]

    vocab, rev_vocab = build_vocab(len(table_order), args.num_aliases)
    n_val = max(32, int(0.05 * len(rows)))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    module = VAEModule(
        vocab=vocab,
        rev_vocab=rev_vocab,
        bn_size=2,
        d_enc=128,
        d_dec=128,
        d_neck=int(args.d_neck),
        kl_factor=0.1,
    )
    train_loader = DataLoader(
        FlatPlanDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: b,
        num_workers=0,
    )
    val_loader = DataLoader(
        FlatPlanDataset(val_rows),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: b,
        num_workers=0,
    )
    log_dir = domain_history("query") / "vae/lightning_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        max_epochs=int(args.max_epochs),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=int(args.devices) if torch.cuda.is_available() else 1,
        default_root_dir=str(log_dir),
        enable_progress_bar=True,
        log_every_n_steps=20,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, train_loader, val_loader)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(out))
    hist_ckpt = domain_history("query") / "vae/CEB_64.ckpt"
    hist_ckpt.parent.mkdir(parents=True, exist_ok=True)
    if hist_ckpt.resolve() != out.resolve():
        shutil.copy2(out, hist_ckpt)
    save_query_vae_meta(
        QUERY_VAE_META,
        table_order=table_order,
        num_aliases=args.num_aliases,
        d_neck=int(args.d_neck),
        bn_size=2,
        max_string_length=48,
    )
    print(f"[train_bolt_query_vae] wrote {out} ({out.stat().st_size} bytes)")
    print(f"[train_bolt_query_vae] meta {QUERY_VAE_META}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
