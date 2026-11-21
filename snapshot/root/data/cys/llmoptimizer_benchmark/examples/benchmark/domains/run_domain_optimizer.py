#!/usr/bin/env python3
"""Run TPE / CMA-ES / random / LLM-semantic hybrid on a domain adapter.

Outputs under benchmark_data/domain_{protein,query,dl}/ — never the tabular algo40 tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains import get_adapter
from mindware.domains.dlhpo.adapter import lookup_from_history
from mindware.domains.paths import domain_bench, domain_history
from mindware.domains.protein.oracle import LookupOracle, load_seq_score_pairs
from mindware.domains.search import run_domain_search


def _protein_adapter(args):
    from mindware.domains.paths import APEXGO_INIT, BOLT_PEPTIDES
    from mindware.domains.sequence_vae import SequenceVAE

    vae_path = args.vae or (domain_history("protein") / "vae/sequence_vae.pkl")
    vae = SequenceVAE.load(vae_path) if vae_path.is_file() else None
    oracle = LookupOracle({})
    bolt_seq = BOLT_PEPTIDES / "apex_oracle/init_data/seed_0_init.txt"
    bolt_sc = BOLT_PEPTIDES / "apex_oracle/init_data/seed_0_scores.csv"
    if bolt_seq.is_file() and bolt_sc.is_file():
        oracle = LookupOracle(load_seq_score_pairs(bolt_seq, bolt_sc), minimize=False)
    task = {"task_id": args.task or "bolt_peptides"}
    return get_adapter("protein", task=task, vae=vae, oracle=oracle, dummy=args.dummy)


def _query_adapter(args):
    sql = ""
    if args.sql_file and Path(args.sql_file).is_file():
        sql = Path(args.sql_file).read_text(encoding="utf-8", errors="replace")
    task = {"task_id": args.task or "query_dummy", "sql": sql}
    vae_path = args.vae or (domain_history("query") / "vae/sequence_vae.pkl")
    return get_adapter("query", task=task, dummy=args.dummy or not sql, vae_path=vae_path)


def _dl_adapter(args):
    lookup = {}
    hist = args.history
    if hist and Path(hist).is_file():
        lookup = lookup_from_history(Path(hist))
    else:
        root = domain_history("dl") / "val_loss"
        if root.is_dir():
            files = sorted(root.glob("*/*.json"))
            if files:
                lookup = lookup_from_history(files[0])
                args.task = args.task or files[0].parent.name
    task = {"task_id": args.task or "dl_task", "n_trials": len(lookup)}
    return get_adapter("dl", task=task, lookup=lookup, dummy=args.dummy and not lookup)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=["protein", "query", "dl"], required=True)
    p.add_argument("--task", default=None)
    p.add_argument("--optimizer", default="cmaes", help="random|tpe|cmaes|ptllm_tpe|ptllm_cmaes")
    p.add_argument("--n-trials", type=int, default=16)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dummy", action="store_true")
    p.add_argument("--llm-prob", type=float, default=0.7)
    p.add_argument("--hybrid-selector", default="bernoulli")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--vae", type=Path, default=None)
    p.add_argument("--sql-file", type=Path, default=None)
    p.add_argument("--history", type=Path, default=None)
    args = p.parse_args(argv)

    if args.domain == "protein":
        adapter = _protein_adapter(args)
    elif args.domain == "query":
        adapter = _query_adapter(args)
    else:
        adapter = _dl_adapter(args)

    out = args.output_dir or (
        domain_bench(args.domain) / str(args.task or adapter.task_id()) / args.optimizer / f"seed{args.seed}"
    )
    result = run_domain_search(
        adapter,
        n_trials=args.n_trials,
        seed=args.seed,
        optimizer=args.optimizer,
        llm_prob=args.llm_prob,
        hybrid_selector=args.hybrid_selector,
        output_dir=out,
    )
    summary = {k: result[k] for k in ("task_id", "domain", "optimizer", "n_trials", "best_objective") if k in result}
    summary["output_dir"] = str(out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Run TPE / CMA-ES / random / LLM-semantic hybrid on a domain adapter.

Outputs under benchmark_data/domain_{protein,query,dl}/ — never the tabular algo40 tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains import get_adapter
from mindware.domains.dlhpo.adapter import lookup_from_history
from mindware.domains.paths import domain_bench, domain_history
from mindware.domains.protein.oracle import LookupOracle, load_seq_score_pairs
from mindware.domains.search import run_domain_search


def _protein_adapter(args):
    from mindware.domains.paths import APEXGO_INIT, BOLT_PEPTIDES
    from mindware.domains.sequence_vae import SequenceVAE

    vae_path = args.vae or (domain_history("protein") / "vae/sequence_vae.pkl")
    vae = SequenceVAE.load(vae_path) if vae_path.is_file() else None
    oracle = LookupOracle({})
    bolt_seq = BOLT_PEPTIDES / "apex_oracle/init_data/seed_0_init.txt"
    bolt_sc = BOLT_PEPTIDES / "apex_oracle/init_data/seed_0_scores.csv"
    if bolt_seq.is_file() and bolt_sc.is_file():
        oracle = LookupOracle(load_seq_score_pairs(bolt_seq, bolt_sc), minimize=False)
    task = {"task_id": args.task or "bolt_peptides"}
    return get_adapter("protein", task=task, vae=vae, oracle=oracle, dummy=args.dummy)


def _query_adapter(args):
    sql = ""
    if args.sql_file and Path(args.sql_file).is_file():
        sql = Path(args.sql_file).read_text(encoding="utf-8", errors="replace")
    task = {"task_id": args.task or "query_dummy", "sql": sql}
    vae_path = args.vae or (domain_history("query") / "vae/sequence_vae.pkl")
    return get_adapter("query", task=task, dummy=args.dummy or not sql, vae_path=vae_path)


def _dl_adapter(args):
    lookup = {}
    hist = args.history
    if hist and Path(hist).is_file():
        lookup = lookup_from_history(Path(hist))
    else:
        root = domain_history("dl") / "val_loss"
        if root.is_dir():
            files = sorted(root.glob("*/*.json"))
            if files:
                lookup = lookup_from_history(files[0])
                args.task = args.task or files[0].parent.name
    task = {"task_id": args.task or "dl_task", "n_trials": len(lookup)}
    return get_adapter("dl", task=task, lookup=lookup, dummy=args.dummy and not lookup)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=["protein", "query", "dl"], required=True)
    p.add_argument("--task", default=None)
    p.add_argument("--optimizer", default="cmaes", help="random|tpe|cmaes|ptllm_tpe|ptllm_cmaes")
    p.add_argument("--n-trials", type=int, default=16)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dummy", action="store_true")
    p.add_argument("--llm-prob", type=float, default=0.7)
    p.add_argument("--hybrid-selector", default="bernoulli")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--vae", type=Path, default=None)
    p.add_argument("--sql-file", type=Path, default=None)
    p.add_argument("--history", type=Path, default=None)
    args = p.parse_args(argv)

    if args.domain == "protein":
        adapter = _protein_adapter(args)
    elif args.domain == "query":
        adapter = _query_adapter(args)
    else:
        adapter = _dl_adapter(args)

    out = args.output_dir or (
        domain_bench(args.domain) / str(args.task or adapter.task_id()) / args.optimizer / f"seed{args.seed}"
    )
    result = run_domain_search(
        adapter,
        n_trials=args.n_trials,
        seed=args.seed,
        optimizer=args.optimizer,
        llm_prob=args.llm_prob,
        hybrid_selector=args.hybrid_selector,
        output_dir=out,
    )
    summary = {k: result[k] for k in ("task_id", "domain", "optimizer", "n_trials", "best_objective") if k in result}
    summary["output_dir"] = str(out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Run TPE / CMA-ES / random / LLM-semantic hybrid on a domain adapter.

Outputs under benchmark_data/domain_{protein,query,dl}/ — never the tabular algo40 tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains import get_adapter
from mindware.domains.dlhpo.adapter import lookup_from_history
from mindware.domains.paths import domain_bench, domain_history
from mindware.domains.protein.oracle import LookupOracle, load_seq_score_pairs
from mindware.domains.search import run_domain_search


def _protein_adapter(args):
    from mindware.domains.paths import APEXGO_INIT, BOLT_PEPTIDES
    from mindware.domains.sequence_vae import SequenceVAE

    vae_path = args.vae or (domain_history("protein") / "vae/sequence_vae.pkl")
    vae = SequenceVAE.load(vae_path) if vae_path.is_file() else None
    oracle = LookupOracle({})
    bolt_seq = BOLT_PEPTIDES / "apex_oracle/init_data/seed_0_init.txt"
    bolt_sc = BOLT_PEPTIDES / "apex_oracle/init_data/seed_0_scores.csv"
    if bolt_seq.is_file() and bolt_sc.is_file():
        oracle = LookupOracle(load_seq_score_pairs(bolt_seq, bolt_sc), minimize=False)
    task = {"task_id": args.task or "bolt_peptides"}
    return get_adapter("protein", task=task, vae=vae, oracle=oracle, dummy=args.dummy)


def _query_adapter(args):
    sql = ""
    if args.sql_file and Path(args.sql_file).is_file():
        sql = Path(args.sql_file).read_text(encoding="utf-8", errors="replace")
    task = {"task_id": args.task or "query_dummy", "sql": sql}
    vae_path = args.vae or (domain_history("query") / "vae/sequence_vae.pkl")
    return get_adapter("query", task=task, dummy=args.dummy or not sql, vae_path=vae_path)


def _dl_adapter(args):
    lookup = {}
    hist = args.history
    if hist and Path(hist).is_file():
        lookup = lookup_from_history(Path(hist))
    else:
        root = domain_history("dl") / "val_loss"
        if root.is_dir():
            files = sorted(root.glob("*/*.json"))
            if files:
                lookup = lookup_from_history(files[0])
                args.task = args.task or files[0].parent.name
    task = {"task_id": args.task or "dl_task", "n_trials": len(lookup)}
    return get_adapter("dl", task=task, lookup=lookup, dummy=args.dummy and not lookup)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=["protein", "query", "dl"], required=True)
    p.add_argument("--task", default=None)
    p.add_argument("--optimizer", default="cmaes", help="random|tpe|cmaes|ptllm_tpe|ptllm_cmaes")
    p.add_argument("--n-trials", type=int, default=16)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dummy", action="store_true")
    p.add_argument("--llm-prob", type=float, default=0.7)
    p.add_argument("--hybrid-selector", default="bernoulli")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--vae", type=Path, default=None)
    p.add_argument("--sql-file", type=Path, default=None)
    p.add_argument("--history", type=Path, default=None)
    args = p.parse_args(argv)

    if args.domain == "protein":
        adapter = _protein_adapter(args)
    elif args.domain == "query":
        adapter = _query_adapter(args)
    else:
        adapter = _dl_adapter(args)

    out = args.output_dir or (
        domain_bench(args.domain) / str(args.task or adapter.task_id()) / args.optimizer / f"seed{args.seed}"
    )
    result = run_domain_search(
        adapter,
        n_trials=args.n_trials,
        seed=args.seed,
        optimizer=args.optimizer,
        llm_prob=args.llm_prob,
        hybrid_selector=args.hybrid_selector,
        output_dir=out,
    )
    summary = {k: result[k] for k in ("task_id", "domain", "optimizer", "n_trials", "best_objective") if k in result}
    summary["output_dir"] = str(out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Run TPE / CMA-ES / random / LLM-semantic hybrid on a domain adapter.

Outputs under benchmark_data/domain_{protein,query,dl}/ — never the tabular algo40 tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains import get_adapter
from mindware.domains.dlhpo.adapter import lookup_from_history
from mindware.domains.paths import domain_bench, domain_history
from mindware.domains.protein.oracle import LookupOracle, load_seq_score_pairs
from mindware.domains.search import run_domain_search


def _protein_adapter(args):
    from mindware.domains.paths import APEXGO_INIT, BOLT_PEPTIDES
    from mindware.domains.sequence_vae import SequenceVAE

    vae_path = args.vae or (domain_history("protein") / "vae/sequence_vae.pkl")
    vae = SequenceVAE.load(vae_path) if vae_path.is_file() else None
    oracle = LookupOracle({})
    bolt_seq = BOLT_PEPTIDES / "apex_oracle/init_data/seed_0_init.txt"
    bolt_sc = BOLT_PEPTIDES / "apex_oracle/init_data/seed_0_scores.csv"
    if bolt_seq.is_file() and bolt_sc.is_file():
        oracle = LookupOracle(load_seq_score_pairs(bolt_seq, bolt_sc), minimize=False)
    task = {"task_id": args.task or "bolt_peptides"}
    return get_adapter("protein", task=task, vae=vae, oracle=oracle, dummy=args.dummy)


def _query_adapter(args):
    sql = ""
    if args.sql_file and Path(args.sql_file).is_file():
        sql = Path(args.sql_file).read_text(encoding="utf-8", errors="replace")
    task = {"task_id": args.task or "query_dummy", "sql": sql}
    vae_path = args.vae or (domain_history("query") / "vae/sequence_vae.pkl")
    return get_adapter("query", task=task, dummy=args.dummy or not sql, vae_path=vae_path)


def _dl_adapter(args):
    lookup = {}
    hist = args.history
    if hist and Path(hist).is_file():
        lookup = lookup_from_history(Path(hist))
    else:
        root = domain_history("dl") / "val_loss"
        if root.is_dir():
            files = sorted(root.glob("*/*.json"))
            if files:
                lookup = lookup_from_history(files[0])
                args.task = args.task or files[0].parent.name
    task = {"task_id": args.task or "dl_task", "n_trials": len(lookup)}
    return get_adapter("dl", task=task, lookup=lookup, dummy=args.dummy and not lookup)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=["protein", "query", "dl"], required=True)
    p.add_argument("--task", default=None)
    p.add_argument("--optimizer", default="cmaes", help="random|tpe|cmaes|ptllm_tpe|ptllm_cmaes")
    p.add_argument("--n-trials", type=int, default=16)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dummy", action="store_true")
    p.add_argument("--llm-prob", type=float, default=0.7)
    p.add_argument("--hybrid-selector", default="bernoulli")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--vae", type=Path, default=None)
    p.add_argument("--sql-file", type=Path, default=None)
    p.add_argument("--history", type=Path, default=None)
    args = p.parse_args(argv)

    if args.domain == "protein":
        adapter = _protein_adapter(args)
    elif args.domain == "query":
        adapter = _query_adapter(args)
    else:
        adapter = _dl_adapter(args)

    out = args.output_dir or (
        domain_bench(args.domain) / str(args.task or adapter.task_id()) / args.optimizer / f"seed{args.seed}"
    )
    result = run_domain_search(
        adapter,
        n_trials=args.n_trials,
        seed=args.seed,
        optimizer=args.optimizer,
        llm_prob=args.llm_prob,
        hybrid_selector=args.hybrid_selector,
        output_dir=out,
    )
    summary = {k: result[k] for k in ("task_id", "domain", "optimizer", "n_trials", "best_objective") if k in result}
    summary["output_dir"] = str(out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
