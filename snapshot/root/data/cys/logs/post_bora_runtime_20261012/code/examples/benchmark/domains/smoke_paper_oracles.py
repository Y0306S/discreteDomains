#!/usr/bin/env python3
"""8-trial TPE smoke on GPU 0: APEX AMP, Postgres EXPLAIN, DMS lookup, ESMFold."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_ENDPOINT", os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains.bolt_vae import PEPTIDE_VAE_CKPT, PeptideBoltVAE, QueryBoltVAE, ckpt_is_real
from mindware.domains.paths import domain_bench, domain_history
from mindware.domains.queryopt.ceb import load_sql, load_task_ids
from mindware.domains.search import run_domain_search


def _run(adapter, *, n_trials, seed, out_dir, hist, init_z, extra):
    out_dir.mkdir(parents=True, exist_ok=True)
    hist.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result = run_domain_search(
        adapter,
        n_trials=n_trials,
        seed=seed,
        optimizer="tpe",
        output_dir=out_dir,
        init_z=init_z,
        n_startup_trials=min(4, n_trials),
    )
    src = Path(result["history_path"])
    if src.resolve() != hist.resolve():
        hist.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    row = {
        **extra,
        "n_trials": n_trials,
        "best_objective": result.get("best_objective"),
        "seconds": round(time.time() - t0, 2),
        "history": str(hist),
    }
    print(json.dumps(row), flush=True)
    return row


def smoke_amp(n_trials: int, device: str) -> dict:
    from mindware.domains import get_adapter
    from mindware.domains.protein.apex_scorer import ApexMeanMIC

    vae = PeptideBoltVAE(PEPTIDE_VAE_CKPT, device=device)
    apex = ApexMeanMIC()
    seq = "GIGKFLHSAKKFGKAFVGEIMNS"
    init_z = vae.encode(seq)
    adapter = get_adapter(
        "protein",
        task={"task_id": "pep_smoke", "start_seq": seq},
        vae=vae,
        apex=apex,
        dummy=False,
        max_len=50,
        oracle_kind="apex",
        latent_bound=4.0,
    )
    return _run(
        adapter,
        n_trials=n_trials,
        seed=7,
        out_dir=domain_bench("protein") / "smoke8" / "pep_smoke" / "tpe" / "seed7",
        hist=domain_history("protein") / "fitness" / "pep_smoke" / "tpe_bolt.json",
        init_z=init_z,
        extra={"domain": "protein", "oracle": "apex_mean_mic", "task_id": "pep_smoke"},
    )


def smoke_query(n_trials: int, device: str, dsn: str) -> dict:
    from mindware.domains import get_adapter

    vae = QueryBoltVAE(device=device)
    tid = load_task_ids(limit=1)[0]
    sql = load_sql(tid)
    adapter = get_adapter(
        "query",
        task={"task_id": tid, "sql": sql},
        vae=vae,
        dummy=False,
        dsn=dsn,
        latent_bound=4.0,
    )
    try:
        init_z = vae.encode([t for t, _ in adapter.pairs])
    except Exception:
        init_z = None
    return _run(
        adapter,
        n_trials=n_trials,
        seed=7,
        out_dir=domain_bench("query") / "smoke8" / tid / "tpe" / "seed7",
        hist=domain_history("query") / "cost" / tid / "tpe_bolt.json",
        init_z=init_z,
        extra={"domain": "query", "oracle": "postgres_explain", "task_id": tid},
    )


def smoke_structure(n_trials: int, device: str, oracle_kind: str, landscape: str) -> dict:
    from mindware.domains import get_adapter
    from mindware.domains.protein.oracle import ESMFoldOracle, LookupOracle, load_dms_json

    pg = domain_history("protein") / "proteingym" / landscape
    csv_path = pg / "sequences.csv"
    lookup_path = pg / "dms_lookup.json"
    ckpt = pg / "vae" / "epoch_118.pkl"
    meta = json.loads((pg / "meta.json").read_text(encoding="utf-8")) if (pg / "meta.json").is_file() else {}
    max_len = int(meta.get("max_len") or 250)
    if not ckpt_is_real(ckpt):
        raise SystemExit(f"landscape VAE missing: {ckpt}")
    vae = PeptideBoltVAE(ckpt, device=device, max_string_length=max_len)
    seqs = [
        ln.split(",")[-1].strip()
        for ln in csv_path.read_text(encoding="utf-8").splitlines()[1:]
        if ln.strip()
    ]
    seq = seqs[0] if seqs else "G"
    if oracle_kind == "esmfold":
        oracle = ESMFoldOracle(pg / "esmfold_cache.jsonl", device=device)
        kind = "esmfold"
    else:
        table = load_dms_json(lookup_path) if lookup_path.is_file() else {}
        oracle = LookupOracle(table, minimize=False)
        kind = "dms"
    adapter = get_adapter(
        "protein",
        task={"task_id": f"{landscape}_smoke", "landscape": landscape, "device": device},
        vae=vae,
        oracle=oracle,
        dummy=False,
        max_len=max_len,
        oracle_kind=kind,
        latent_bound=4.0,
    )
    init_z = vae.encode(seq)
    hist_root = domain_history("protein") / "structure" / landscape / "fitness" / f"{landscape}_smoke"
    return _run(
        adapter,
        n_trials=n_trials,
        seed=7,
        out_dir=domain_bench("protein") / "structure" / landscape / "smoke8" / "tpe" / "seed7",
        hist=hist_root / "tpe_bolt.json",
        init_z=init_z,
        extra={"domain": "protein_structure", "oracle": kind, "task_id": f"{landscape}_smoke", "landscape": landscape},
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-trials", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--query-dsn", default=os.environ.get("QUERYOPT_PG_DSN", "host=127.0.0.1 port=5432 dbname=imdb user=imdb password=imdb"))
    p.add_argument("--skip-amp", action="store_true")
    p.add_argument("--skip-query", action="store_true")
    p.add_argument("--skip-dms", action="store_true")
    p.add_argument("--skip-esmfold", action="store_true")
    args = p.parse_args(argv)

    rows = []
    if not args.skip_amp:
        rows.append(smoke_amp(args.n_trials, args.device))
    if not args.skip_query:
        os.environ["QUERYOPT_PG_DSN"] = args.query_dsn
        rows.append(smoke_query(args.n_trials, args.device, args.query_dsn))
    if not args.skip_dms:
        for ls in ("gfp", "gb1", "aav"):
            pg = domain_history("protein") / "proteingym" / ls / "vae" / "epoch_118.pkl"
            if ckpt_is_real(pg):
                rows.append(smoke_structure(args.n_trials, args.device, "dms", ls))
    if not args.skip_esmfold:
        gfp = domain_history("protein") / "proteingym" / "gfp" / "vae" / "epoch_118.pkl"
        if ckpt_is_real(gfp):
            rows.append(smoke_structure(min(args.n_trials, 2), args.device, "esmfold", "gfp"))
    summary = ROOT / "history_data/domain_protein/proteingym/smoke8_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {summary} n={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""8-trial TPE smoke on GPU 0: APEX AMP, Postgres EXPLAIN, DMS lookup, ESMFold."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_ENDPOINT", os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains.bolt_vae import PEPTIDE_VAE_CKPT, PeptideBoltVAE, QueryBoltVAE, ckpt_is_real
from mindware.domains.paths import domain_bench, domain_history
from mindware.domains.queryopt.ceb import load_sql, load_task_ids
from mindware.domains.search import run_domain_search


def _run(adapter, *, n_trials, seed, out_dir, hist, init_z, extra):
    out_dir.mkdir(parents=True, exist_ok=True)
    hist.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result = run_domain_search(
        adapter,
        n_trials=n_trials,
        seed=seed,
        optimizer="tpe",
        output_dir=out_dir,
        init_z=init_z,
        n_startup_trials=min(4, n_trials),
    )
    src = Path(result["history_path"])
    if src.resolve() != hist.resolve():
        hist.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    row = {
        **extra,
        "n_trials": n_trials,
        "best_objective": result.get("best_objective"),
        "seconds": round(time.time() - t0, 2),
        "history": str(hist),
    }
    print(json.dumps(row), flush=True)
    return row


def smoke_amp(n_trials: int, device: str) -> dict:
    from mindware.domains import get_adapter
    from mindware.domains.protein.apex_scorer import ApexMeanMIC

    vae = PeptideBoltVAE(PEPTIDE_VAE_CKPT, device=device)
    apex = ApexMeanMIC()
    seq = "GIGKFLHSAKKFGKAFVGEIMNS"
    init_z = vae.encode(seq)
    adapter = get_adapter(
        "protein",
        task={"task_id": "pep_smoke", "start_seq": seq},
        vae=vae,
        apex=apex,
        dummy=False,
        max_len=50,
        oracle_kind="apex",
        latent_bound=4.0,
    )
    return _run(
        adapter,
        n_trials=n_trials,
        seed=7,
        out_dir=domain_bench("protein") / "smoke8" / "pep_smoke" / "tpe" / "seed7",
        hist=domain_history("protein") / "fitness" / "pep_smoke" / "tpe_bolt.json",
        init_z=init_z,
        extra={"domain": "protein", "oracle": "apex_mean_mic", "task_id": "pep_smoke"},
    )


def smoke_query(n_trials: int, device: str, dsn: str) -> dict:
    from mindware.domains import get_adapter

    vae = QueryBoltVAE(device=device)
    tid = load_task_ids(limit=1)[0]
    sql = load_sql(tid)
    adapter = get_adapter(
        "query",
        task={"task_id": tid, "sql": sql},
        vae=vae,
        dummy=False,
        dsn=dsn,
        latent_bound=4.0,
    )
    try:
        init_z = vae.encode([t for t, _ in adapter.pairs])
    except Exception:
        init_z = None
    return _run(
        adapter,
        n_trials=n_trials,
        seed=7,
        out_dir=domain_bench("query") / "smoke8" / tid / "tpe" / "seed7",
        hist=domain_history("query") / "cost" / tid / "tpe_bolt.json",
        init_z=init_z,
        extra={"domain": "query", "oracle": "postgres_explain", "task_id": tid},
    )


def smoke_structure(n_trials: int, device: str, oracle_kind: str, landscape: str) -> dict:
    from mindware.domains import get_adapter
    from mindware.domains.protein.oracle import ESMFoldOracle, LookupOracle, load_dms_json

    pg = domain_history("protein") / "proteingym" / landscape
    csv_path = pg / "sequences.csv"
    lookup_path = pg / "dms_lookup.json"
    ckpt = pg / "vae" / "epoch_118.pkl"
    meta = json.loads((pg / "meta.json").read_text(encoding="utf-8")) if (pg / "meta.json").is_file() else {}
    max_len = int(meta.get("max_len") or 250)
    if not ckpt_is_real(ckpt):
        raise SystemExit(f"landscape VAE missing: {ckpt}")
    vae = PeptideBoltVAE(ckpt, device=device, max_string_length=max_len)
    seqs = [
        ln.split(",")[-1].strip()
        for ln in csv_path.read_text(encoding="utf-8").splitlines()[1:]
        if ln.strip()
    ]
    seq = seqs[0] if seqs else "G"
    if oracle_kind == "esmfold":
        oracle = ESMFoldOracle(pg / "esmfold_cache.jsonl", device=device)
        kind = "esmfold"
    else:
        table = load_dms_json(lookup_path) if lookup_path.is_file() else {}
        oracle = LookupOracle(table, minimize=False)
        kind = "dms"
    adapter = get_adapter(
        "protein",
        task={"task_id": f"{landscape}_smoke", "landscape": landscape, "device": device},
        vae=vae,
        oracle=oracle,
        dummy=False,
        max_len=max_len,
        oracle_kind=kind,
        latent_bound=4.0,
    )
    init_z = vae.encode(seq)
    hist_root = domain_history("protein") / "structure" / landscape / "fitness" / f"{landscape}_smoke"
    return _run(
        adapter,
        n_trials=n_trials,
        seed=7,
        out_dir=domain_bench("protein") / "structure" / landscape / "smoke8" / "tpe" / "seed7",
        hist=hist_root / "tpe_bolt.json",
        init_z=init_z,
        extra={"domain": "protein_structure", "oracle": kind, "task_id": f"{landscape}_smoke", "landscape": landscape},
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-trials", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--query-dsn", default=os.environ.get("QUERYOPT_PG_DSN", "host=127.0.0.1 port=5432 dbname=imdb user=imdb password=imdb"))
    p.add_argument("--skip-amp", action="store_true")
    p.add_argument("--skip-query", action="store_true")
    p.add_argument("--skip-dms", action="store_true")
    p.add_argument("--skip-esmfold", action="store_true")
    args = p.parse_args(argv)

    rows = []
    if not args.skip_amp:
        rows.append(smoke_amp(args.n_trials, args.device))
    if not args.skip_query:
        os.environ["QUERYOPT_PG_DSN"] = args.query_dsn
        rows.append(smoke_query(args.n_trials, args.device, args.query_dsn))
    if not args.skip_dms:
        for ls in ("gfp", "gb1", "aav"):
            pg = domain_history("protein") / "proteingym" / ls / "vae" / "epoch_118.pkl"
            if ckpt_is_real(pg):
                rows.append(smoke_structure(args.n_trials, args.device, "dms", ls))
    if not args.skip_esmfold:
        gfp = domain_history("protein") / "proteingym" / "gfp" / "vae" / "epoch_118.pkl"
        if ckpt_is_real(gfp):
            rows.append(smoke_structure(min(args.n_trials, 2), args.device, "esmfold", "gfp"))
    summary = ROOT / "history_data/domain_protein/proteingym/smoke8_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {summary} n={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
