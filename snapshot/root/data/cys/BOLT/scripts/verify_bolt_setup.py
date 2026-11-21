#!/usr/bin/env python3
"""Verify BOLT peptide + query-plan evaluation prerequisites."""

from __future__ import annotations

import argparse
import os
import socket
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ok(msg: str) -> str:
    return f"OK   {msg}"


def fail(msg: str) -> str:
    return f"FAIL {msg}"


def warn(msg: str) -> str:
    return f"WARN {msg}"


def check_file(path: Path, min_bytes: int = 1) -> tuple[bool, str]:
    if not path.is_file():
        return False, fail(f"missing: {path}")
    size = path.stat().st_size
    if size < min_bytes:
        return False, fail(f"too small ({size} B): {path}")
    return True, ok(f"{path} ({size // 1024} KiB)")


def check_dir_nonempty(path: Path) -> tuple[bool, str]:
    if not path.is_dir():
        return False, fail(f"missing dir: {path}")
    n = sum(1 for _ in path.rglob("*") if _.is_file())
    if n == 0:
        return False, fail(f"empty dir: {path}")
    return True, ok(f"{path} ({n} files)")


def pg_reachable(host: str, port: int) -> tuple[bool, str]:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True, ok(f"PostgreSQL reachable at {host}:{port}")
    except OSError as exc:
        return False, warn(f"PostgreSQL not reachable at {host}:{port} ({exc})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-host", default=os.getenv("DB_HOST", "127.0.0.1"))
    parser.add_argument("--db-port", type=int, default=int(os.getenv("DB_PORT", "5432")))
    args = parser.parse_args()

    lines: list[str] = []
    all_required_ok = True

    lines.append("=== Peptides (protein / APEX) ===")
    peptide_checks = [
        check_file(
            ROOT
            / "optimization/peptides/uniref_vae/saved_models/dim128_k1_kl0001_eff256_dff256_pious-sea-2_model_state_epoch_118.pkl",
            min_bytes=1_000_000,
        ),
        check_dir_nonempty(ROOT / "optimization/peptides/apex_oracle/APEX_pathogen_models"),
        check_file(ROOT / "optimization/peptides/apex_oracle/init_data/seed_0_init.txt", 1000),
        check_file(ROOT / "optimization/peptides/apex_oracle/init_data/seed_0_scores.csv", 1000),
    ]
    for passed, msg in peptide_checks:
        lines.append(msg)
        all_required_ok &= passed

    lines.append("")
    lines.append("=== Query plans (CEB / PostgreSQL) ===")
    ceb_zip = ROOT / "optimization/query_plans/query_plan_optimization/workload/ceb-3k/ceb_3k.zip"
    ceb_sql_root = ROOT / "optimization/query_plans/query_plan_optimization/workload/ceb-3k/1a"
    qp_checks = [
        check_file(ceb_zip, 100_000),
    ]
    if ceb_sql_root.is_dir():
        qp_checks.append(check_dir_nonempty(ceb_sql_root))
    else:
        qp_checks.append(
            (
                False,
                warn(
                    "CEB SQL tree not extracted — run scripts/prepare_query_workloads.sh "
                    f"(expected e.g. {ceb_sql_root})",
                ),
            )
        )

    vae_ckpt = ROOT / "optimization/query_plans/query_plan_optimization/vae/CEB_64.ckpt"
    if vae_ckpt.is_file() and vae_ckpt.stat().st_size > 1_000_000:
        qp_checks.append(check_file(vae_ckpt, 1_000_000))
    else:
        qp_checks.append(
            (
                False,
                warn(
                    "Query-plan VAE checkpoint missing — place CEB_64.ckpt (or symlink 64.ckpt) "
                    f"at {vae_ckpt}; required for latent BO + init generation",
                ),
            )
        )

    # 64.ckpt alias used by create_initialization_data.py
    vae_64 = ROOT / "optimization/query_plans/query_plan_optimization/vae/64.ckpt"
    if not vae_64.exists() and vae_ckpt.is_file():
        qp_checks.append((False, warn(f"symlink missing: {vae_64} -> CEB_64.ckpt")))

    qp_checks.append(check_file(ROOT / "optimization/query_plans/tasks/ceb_tasks.txt", 100))

    pg_ok, pg_msg = pg_reachable(args.db_host, args.db_port)
    qp_checks.append((pg_ok, pg_msg))

    for passed, msg in qp_checks:
        lines.append(msg)
        if msg.startswith("FAIL"):
            all_required_ok = False

    lines.append("")
    lines.append("=== Fine-tuning data paths ===")
    ft_qp_workload = ROOT / "fine-tuning/query_plans/workload"
    if ft_qp_workload.exists():
        lines.append(ok(f"fine-tuning workload link: {ft_qp_workload}"))
    else:
        lines.append(
            warn(
                f"fine-tuning/query_plans/workload not linked — run scripts/prepare_finetuning_layout.sh"
            )
        )

    lines.append("")
    if all_required_ok:
        lines.append("Overall: READY for peptide eval; query-plan eval still needs VAE + IMDB Postgres if not set up.")
    else:
        lines.append("Overall: INCOMPLETE — see FAIL/WARN above.")
    print("\n".join(lines))
    return 0 if all_required_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
