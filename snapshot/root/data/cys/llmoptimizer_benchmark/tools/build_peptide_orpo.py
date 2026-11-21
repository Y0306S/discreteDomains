#!/usr/bin/env python3
"""Build peptide ORPO/DPO preference pairs from the instrumented STBO run.

Ports the ML pipeline's criterion (examples/benchmark/export_p5_s2_dpo_pareto.py) from
single configs to batches of sequences:

    chosen   = the SFT-style batch (top-BATCH by MIC, contains the best config)
    rejected = same batch with its best K members swapped for sequences that are
               (A) maximally similar to what they replaced  and  (B) clearly worse
               -> Pareto front on (A, B), pick argmax 0.5*A + 0.5*B

Why swap-K and not consecutive incumbent snapshots: snapshots differ by ~1 member in 50,
so chosen and rejected are ~98% token-identical and the preference gradient is nearly
degenerate. Measured on the v2 run, K=10 gives 40/50 overlap with a +49..108% best-MIC
gap and swapped-in sequences still 0.93 similar to what they replaced -- a genuine hard
negative with a trainable difference.

The prompt is the SFT prompt verbatim, so chosen/rejected are both valid responses to it.
Improvement events anchor WHERE chosen comes from: only tasks whose incumbent actually
improved contribute, and the margin is recorded per row so it can be ablated.

Usage:
  PYTHONPATH=. python tools/build_peptide_orpo.py \
      --sft-train benchmark_data/rl_task_selection/peptide_sft_v3/peptide_sft_elite10_train.jsonl \
      --out-dir  benchmark_data/rl_task_selection/peptide_orpo_v3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains.protein.bolt_constraints import (  # noqa: E402
    bolt_reference_sequence,
    template_similarity,
)

BOLT = "/root/data/cys/BOLT/optimization/peptides"
DEFAULT_BO = Path(
    "/root/data/cys/blockdata_archive/domain_protein/stbo_init100_r100_bsz50_e2e10_v2_instrumented"
)
MIN_SIM = 0.75
MAX_LEN = 30
AA = set("ACDEFGHIKLMNPQRSTVWY")
RATIONALE = (
    "These are cationic, amphipathic variants of the seed that stay within the "
    "template-similarity budget while shifting charge and hydrophobicity toward motifs "
    "associated with low MIC."
)


def pep_idx(t: str) -> int:
    return int(str(t).strip().split("_")[-1])


def feasible(s: str, ref: str) -> bool:
    return (
        bool(s)
        and len(s) <= MAX_LEN
        and s.count("C") <= 1
        and all(c in AA for c in s)
        and template_similarity(s, ref) >= MIN_SIM
    )


def sim(a: str, b: str) -> float:
    from Levenshtein import distance as lev  # noqa: PLC0415

    m = max(len(a), len(b))
    return (m - lev(a, b)) / m if m else 0.0


def load_pool(hist: Path, ref: str) -> Tuple[List[Tuple[str, float]], int]:
    """Distinct feasible (seq, mic) for a task, best MIC per sequence, plus n improvements."""
    payload = json.loads(hist.read_text(encoding="utf-8"))
    best: Dict[str, float] = {}
    run = float("inf")
    n_imp = 0
    for o in payload.get("observations") or []:
        x = o.get("extra_info") or {}
        s = x.get("artifact")
        info = x.get("info") or {}
        if not s or not feasible(s, ref):
            continue
        m = float(info.get("mic_bacteria_0", -o["objectives"][0]))
        if s not in best or m < best[s]:
            best[s] = m
        if m < run - 1e-9:
            run = m
            n_imp += 1
    return sorted(best.items(), key=lambda kv: (kv[1], kv[0])), n_imp


def format_assistant(batch: List[Tuple[str, float]], rationale: bool) -> str:
    recs = [{"rank": i, "sequence": s} for i, (s, _) in enumerate(batch, start=1)]
    body = json.dumps({"recommendations": recs}, ensure_ascii=True)
    return f"Rationale: {RATIONALE}\n\nJSON:\n{body}" if rationale else f"JSON:\n{body}"


def build_rejected(
    chosen: List[Tuple[str, float]], ranked: List[Tuple[str, float]], k: int
) -> Optional[Tuple[List[Tuple[str, float]], Dict[str, Any]]]:
    """Swap the best k of `chosen` for Pareto-selected worse-but-similar sequences."""
    if len(chosen) <= k:
        return None
    keep = chosen[k:]
    out_seqs = [s for s, _ in chosen[:k]]
    chosen_set = {s for s, _ in chosen}
    worst_kept = chosen[k - 1][1]
    pool = [(s, m) for s, m in ranked if s not in chosen_set and m > worst_kept + 1e-9]
    if len(pool) < k:
        return None
    # A: similarity to the members being replaced (higher = harder negative)
    # B: how much worse, min-max normalised in-pool (higher = clearer preference)
    mics = [m for _, m in pool]
    lo, hi = min(mics), max(mics)
    scored = []
    for s, m in pool:
        a = max(sim(s, o) for o in out_seqs)
        b = (m - lo) / (hi - lo) if hi > lo else 0.0
        scored.append((s, m, a, b))
    # Pareto front on (A, B), then argmax of the 0.5/0.5 blend
    front = [
        c for c in scored
        if not any(o[2] >= c[2] and o[3] >= c[3] and (o[2] > c[2] or o[3] > c[3]) for o in scored)
    ]
    front.sort(key=lambda c: -(0.5 * c[2] + 0.5 * c[3]))
    picks = front[:k]
    if len(picks) < k:  # top up from the blend ranking
        rest = sorted(
            (c for c in scored if c not in picks), key=lambda c: -(0.5 * c[2] + 0.5 * c[3])
        )
        picks += rest[: k - len(picks)]
    if len(picks) < k:
        return None
    rej = keep + [(s, m) for s, m, _, _ in picks]
    rej.sort(key=lambda kv: (kv[1], kv[0]))
    stats = {
        "overlap": len(chosen_set & {s for s, _ in rej}),
        "chosen_best_mic": round(chosen[0][1], 4),
        "rejected_best_mic": round(min(m for _, m in rej), 4),
        "best_mic_gap_pct": round(100 * (min(m for _, m in rej) / chosen[0][1] - 1), 2),
        "mean_swap_similarity": round(sum(c[2] for c in picks) / len(picks), 4),
    }
    return rej, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-train", type=Path, required=True)
    ap.add_argument("--sft-eval", type=Path, default=None)
    ap.add_argument("--bo-root", type=Path, default=DEFAULT_BO)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--swap-k", type=int, default=10)
    ap.add_argument("--min-margin-pct", type=float, default=1.0,
                    help="drop pairs whose best-MIC gap is below this (near-noise pairs)")
    ap.add_argument("--require-improvement", action="store_true", default=True,
                    help="only tasks whose incumbent actually improved contribute")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    def convert(src: Path, tag: str) -> Dict[str, Any]:
        out = a.out_dir / f"peptide_orpo_{tag}.jsonl"
        n = kept = 0
        skips: Dict[str, int] = {}
        stats: List[Dict[str, Any]] = []
        with out.open("w", encoding="utf-8") as fh:
            for line in src.open(encoding="utf-8"):
                if not line.strip():
                    continue
                n += 1
                row = json.loads(line)
                tid = row["metadata"]["task_id"]
                ref = bolt_reference_sequence(constraint_index=pep_idx(tid))
                hist = a.bo_root / tid / "bolt_native" / "seed1" / "history.json"
                if not hist.is_file():
                    skips["missing_history"] = skips.get("missing_history", 0) + 1
                    continue
                ranked, n_imp = load_pool(hist, ref)
                if a.require_improvement and n_imp < 2:
                    skips["no_improvement_event"] = skips.get("no_improvement_event", 0) + 1
                    continue
                asst = row["messages"][-1]["content"]
                seqs = re.findall(r'"sequence":\s*"([A-Z]+)"', asst)
                mic = {s: m for s, m in ranked}
                if not seqs or any(s not in mic for s in seqs):
                    skips["chosen_not_in_pool"] = skips.get("chosen_not_in_pool", 0) + 1
                    continue
                chosen = sorted(((s, mic[s]) for s in seqs), key=lambda kv: (kv[1], kv[0]))
                built = build_rejected(chosen, ranked, a.swap_k)
                if built is None:
                    skips["no_rejected"] = skips.get("no_rejected", 0) + 1
                    continue
                rej, st = built
                if st["best_mic_gap_pct"] < a.min_margin_pct:
                    skips["margin_too_small"] = skips.get("margin_too_small", 0) + 1
                    continue
                has_r = asst.startswith("Rationale:")
                fh.write(json.dumps({
                    "messages": row["messages"][:-1],           # system + user, prompt intact
                    "chosen": {"role": "assistant", "content": format_assistant(chosen, has_r)},
                    "rejected": {"role": "assistant", "content": format_assistant(rej, has_r)},
                    "metadata": {**row["metadata"], "n_improvement_events": n_imp, **st},
                }, ensure_ascii=True) + "\n")
                kept += 1
                stats.append(st)
        agg = {}
        if stats:
            for key in ("overlap", "best_mic_gap_pct", "mean_swap_similarity"):
                v = sorted(s[key] for s in stats)
                agg[key] = {"min": v[0], "median": v[len(v) // 2], "max": v[-1]}
        print(f"  {tag}: {kept}/{n} pairs -> {out.name}   skips={skips}")
        for k2, v2 in agg.items():
            print(f"      {k2}: {v2}")
        return {"file": str(out), "rows": kept, "seen": n, "skips": skips, "agg": agg}

    print(f"swap_k={a.swap_k}  min_margin={a.min_margin_pct}%  bo_root={a.bo_root.name}")
    man = {"swap_k": a.swap_k, "min_margin_pct": a.min_margin_pct,
           "bo_root": str(a.bo_root), "sft_train": str(a.sft_train),
           "criterion": "Pareto 0.5*similarity_to_replaced + 0.5*normalised_worseness",
           "train": convert(a.sft_train, "train")}
    if a.sft_eval and a.sft_eval.is_file():
        man["eval"] = convert(a.sft_eval, "eval")
    (a.out_dir / "peptide_orpo_manifest.json").write_text(
        json.dumps(man, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote {a.out_dir}/peptide_orpo_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
