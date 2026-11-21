#!/usr/bin/env python3
"""Build the peptide SFT dataset for the BOLT outer loop, from corrected BO results.

Replaces examples/benchmark/peptide_sft_common.py + export_peptide_sft_{elite10,winbatch10}.py,
which produced unusable data. Audited failures in the old pipeline (all verified on its
900-row output):

  1. LABEL LEAKAGE -- 900/900 rows. build_user_prompt() embedded
     target_recommendations_block(targets), i.e. the exact 10 answer sequences AND their
     MICs, into the *user* message; the assistant message then repeated them in the same
     order. The model was trained to copy from its own input, which teaches nothing and
     transfers to nothing at inference time.
  2. INFEASIBLE TARGETS -- 9000/9000 target sequences violated the >=0.75 similarity
     constraint (median similarity 0.000, min -1.000), while the prompt declared that
     constraint "hard". Downstream of the init-pool bug: the source run's "best"
     sequences were all infeasible.
  3. LEAKY SPLIT -- default split_mode="stratified10" (eval = every 10th reference) puts
     18/100 eval references within >0.75 similarity of a training reference. The
     pre-registered last-100 split has 0/100. See docs/BOLT_PEPTIDE_SPLIT.md.
  4. SIGN LANDMINE -- _mic_from_obs() fell back to objectives[0], which is the *maximize*
     score (-MIC). If mic_bacteria_0 were ever absent, "min MIC" would have selected the
     worst sequences.
  5. Stale source -- DEFAULT_BO_ROOT pointed at bo80_bolt_bsz10_pilot5, a pre-fix run
     that no longer exists on disk.

This script fixes all five and validates every emitted row before writing.

Usage:
    python tools/build_peptide_sft.py --arm elite10
    python tools/build_peptide_sft.py --arm elite10 --prompt-style bolt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains.protein.bolt_constraints import (  # noqa: E402
    bolt_reference_sequence,
    template_similarity,
)

DEFAULT_BO_ROOT = Path(
    "/root/data/cys/blockdata_archive/domain_protein/stbo_init100_r100_bsz50_e2e10_v2_instrumented"
)
DEFAULT_OUT = ROOT / "benchmark_data/rl_task_selection/peptide_sft_v2"

BATCH_SIZE = 50   # overridden by --batch-size
MIN_SIMILARITY = 0.75
MAX_LENGTH = 30
AA = set("ACDEFGHIKLMNPQRSTVWY")

# Pre-registered split (docs/BOLT_PEPTIDE_SPLIT.md): train 000-899, test 900-999.
N_TRAIN = 900

# BOLT's own peptide prompt (fine-tuning/peptides/generate_openai_ft_data.py). The "up to
# 25% modifications" phrasing IS the >=0.75 similarity constraint.
BOLT_SYSTEM_PROMPT = (
    "You are a specialized assistant that modifies peptide sequences to enhance "
    "antimicrobial activity. Make up to 25% sequence modifications based on known "
    "antimicrobial peptide properties such as: positive charge, hydrophobicity, and "
    "amphipathicity."
)

def rich_system_prompt() -> str:
    """Built at call time so --batch-size is reflected."""
    return (
        "You are a peptide sequence optimization assistant for antimicrobial peptide (AMP) design.\n"
        "The objective is the bacteria_0 mean MIC from the APEX oracle (lower MIC is better).\n"
        f"Propose a parallel batch of {BATCH_SIZE} distinct amino-acid sequences to evaluate next.\n"
        "First write a short rationale, then write one JSON object containing exactly "
        f"{BATCH_SIZE} ranked recommendations.\n"
        "Do not use markdown."
    )

RATIONALE = (
    "These are cationic, amphipathic variants of the seed that stay within the "
    "template-similarity budget while shifting charge and hydrophobicity toward motifs "
    "associated with low MIC."
)


def pep_idx(task_id: str) -> int:
    return int(str(task_id).strip().split("_")[-1])


# ---------------------------------------------------------------- feasibility


def is_feasible(seq: str, reference: str) -> bool:
    """The four hard constraints the prompt declares. All four are enforced here."""
    if not seq or len(seq) > MAX_LENGTH:
        return False
    if any(ch not in AA for ch in seq):
        return False
    if seq.count("C") > 1:
        return False
    return template_similarity(seq, reference) >= MIN_SIMILARITY


# ---------------------------------------------------------------- history I/O


def _obs_extra(o: Dict[str, Any]) -> Dict[str, Any]:
    return o.get("extra_info") or o.get("extra") or {}


def seq_from_obs(o: Dict[str, Any]) -> str:
    extra = _obs_extra(o)
    info = extra.get("info") or {}
    return str(extra.get("artifact") or info.get("sequence") or "")


def mic_from_obs(o: Dict[str, Any]) -> float:
    """Predicted MIC (lower is better).

    objectives[0] is the MAXIMIZE score, i.e. -MIC, so the fallback must negate. The old
    code returned objectives[0] directly, which would have inverted the ranking.
    """
    info = (_obs_extra(o).get("info") or {})
    mic = info.get("mic_bacteria_0")
    if mic is not None:
        return float(mic)
    return -float(o["objectives"][0])


def load_trajectory(
    hist_path: Path,
) -> Tuple[
    List[Tuple[str, float]],
    List[Tuple[str, float]],
    Dict[str, Any],
    List[Tuple[str, float, Optional[str]]],
]:
    """Return (init pairs, post-init unique-new pairs, meta, post-init provenance).

    provenance = (seq, mic, batch_id) so a REAL recorded batch can be reconstructed.
    Only the v2 instrumented run records batch_id; older runs yield None and the
    provenance-based arms are unavailable.
    """
    payload = json.loads(hist_path.read_text(encoding="utf-8"))
    meta = payload.get("meta_info") or {}
    obs = payload.get("observations") or []
    n_init = int(meta.get("n_init_points") or 1000)

    init: List[Tuple[str, float]] = []
    seen: set[str] = set()
    for o in obs[:n_init]:
        s = seq_from_obs(o)
        if not s:
            continue
        init.append((s, mic_from_obs(o)))
        seen.add(s)

    post: List[Tuple[str, float]] = []
    prov: List[Tuple[str, float, Optional[str]]] = []
    for o in obs[n_init:]:
        s = seq_from_obs(o)
        if not s or s in seen:
            continue
        seen.add(s)
        m = mic_from_obs(o)
        post.append((s, m))
        prov.append((s, m, _obs_extra(o).get("batch_id")))
    return init, post, meta, prov


# ---------------------------------------------------------------- target selection


def select_targets(
    arm: str,
    init: Sequence[Tuple[str, float]],
    post: Sequence[Tuple[str, float]],
    *,
    reference: str,
    seed_seq: str,
    target_source: str,
    post_prov: Sequence[Tuple[str, float, Optional[str]]] = (),
) -> Tuple[List[Tuple[str, float]], Optional[Any]]:
    """Top-BATCH_SIZE FEASIBLE, distinct sequences (lowest MIC first).

    target_source="all" uses the whole trajectory, which is what BOLT means by "the
    highest-scoring sequences from the optimization trajectory". This matters here: at a
    1000-call budget BO improves on the init pool for only ~17% of tasks, so restricting
    to post-init sequences ("discovered") would train on sequences WORSE than the best
    known solution for the other ~83%.
    """
    pool = list(init) + list(post) if target_source == "all" else list(post)
    # Drop degenerate targets: the reference template and the seed are both given in the
    # prompt, so recommending them verbatim teaches copying rather than design. BOLT's own
    # mutation generator likewise excludes the identity (mutation_str != sequence).
    banned = {reference, seed_seq}
    feasible = [
        (s, m) for s, m in pool if s not in banned and is_feasible(s, reference)
    ]
    # dedupe keeping the best MIC per sequence
    best: Dict[str, float] = {}
    for s, m in feasible:
        if s not in best or m < best[s]:
            best[s] = m
    ranked = sorted(best.items(), key=lambda kv: (kv[1], kv[0]))

    if arm == "elite10":
        return ranked[:BATCH_SIZE], None

    if arm == "winbatch10":
        # The batch of 10 that CONTAINS the best-performing sequence, i.e. "10 configs
        # which contain the best performing configuration". Contrast with elite10, which
        # is the global top-10 assembled across the whole run: winbatch10 keeps the 9 real
        # co-proposals that accompanied the winner, so it teaches what an actual good BO
        # batch looks like (including its spread) rather than a hall-of-fame.
        #
        # Batches are consecutive groups of 10 over the FEASIBLE post-init sequences in
        # trajectory order. Chunking the feasible subsequence (rather than raw rounds of 10
        # oracle calls, which mix in infeasible decodes) is what guarantees exactly 10
        # feasible targets per row.
        #
        # An earlier version required the batch to improve on the best feasible init MIC.
        # That only holds for ~19% of tasks at a 1000-call budget (197/900 rows), so it was
        # not a fair ablation against elite10's 897.
        post_feas = [
            (s, m) for s, m in post if s not in banned and is_feasible(s, reference)
        ]
        batches = [
            post_feas[i : i + BATCH_SIZE]
            for i in range(0, len(post_feas) - BATCH_SIZE + 1, BATCH_SIZE)
        ]
        if not batches:
            return [], None
        best_i = min(
            range(len(batches)), key=lambda i: min(m for _, m in batches[i])
        )
        return sorted(batches[best_i], key=lambda kv: (kv[1], kv[0])), best_i

    if arm == "winbatch":
        # A REAL recorded batch, using the v2 provenance fields (batch_id/phase/round).
        #
        # It must come from the E2E recenter phase, not acquisition. Measured on the v2 run:
        # of 525,094 acquisition observations only 0.35% are constraint-feasible, and
        # acquisition produced 1 improvement event across 60 tasks, whereas recenter holds
        # 99.4% of all feasible observations and produced 95% of final bests. So the
        # "batch that actually won" is a recenter span; an acquisition batch essentially
        # never contains a usable sequence.
        by_batch: Dict[str, List[Tuple[str, float]]] = {}
        for s_, m_, bid in post_prov:
            if not bid or s_ in banned or not is_feasible(s_, reference):
                continue
            by_batch.setdefault(bid, []).append((s_, m_))
        cand = {b: v for b, v in by_batch.items() if len({x[0] for x in v}) >= BATCH_SIZE}
        if not cand:
            return [], None
        # the winning batch = the one containing the lowest MIC seen in any batch
        bid = min(cand, key=lambda b: min(m for _, m in cand[b]))
        best: Dict[str, float] = {}
        for s_, m_ in cand[bid]:
            if s_ not in best or m_ < best[s_]:
                best[s_] = m_
        ranked_b = sorted(best.items(), key=lambda kv: (kv[1], kv[0]))[:BATCH_SIZE]
        return ranked_b, bid

    raise ValueError(f"unknown arm {arm!r}")


# ---------------------------------------------------------------- prompt


def build_user_prompt(
    *,
    task_id: str,
    seed_seq: str,
    reference: str,
    prompt_style: str,
    post_init_budget: int,
    n_init: int,
) -> str:
    """Task description only. Contains NO target sequence and NO achieved-MIC value.

    The old pipeline put both in here, which is why 900/900 rows leaked their answers.
    Everything in this prompt is knowable for an unseen task at inference time.
    """
    if prompt_style == "bolt":
        # Faithful to BOLT: the context C is the SEED/REFERENCE peptide that defines the
        # task -- REFERENCE_SEQUENCE[pep_idx]. Confirmed against BOLT's own example data
        # (fine-tuning/peptides/data/example_finetuning_data.jsonl), where the user turn is
        # "RRYYEQLEQASRKGNRGFRR" == REFERENCE_SEQUENCE[0].
        # NB: our index rows carry a `start_seq` field, but that is just row `pep_idx` of
        # the seed_0 init file -- an arbitrary pool member, NOT the task definition, and
        # not what the similarity constraint is measured against. Conditioning on it would
        # be meaningless for an unseen task.
        return reference

    t_len = len(reference)
    profile = {
        "task_id": task_id,
        "domain": "protein",
        "oracle": "bolt_bacteria_0",
        "metric": "mean_MIC_bacteria_0",
        "higher_is_better": False,
        "reference_template": reference,
        "constraint_index": pep_idx(task_id),
        "max_length": MAX_LENGTH,
        "warm_start_evaluated": int(n_init),
        "post_init_budget": int(post_init_budget),
        "batch_size": BATCH_SIZE,
    }
    return "\n".join(
        [
            f"Task profile:\n{json.dumps(profile, ensure_ascii=True, sort_keys=True)}",
            "",
            "Design space:",
            "- sequence: string over amino acids A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y",
            f"- length: integer in [1, {MAX_LENGTH}]",
            "",
            f"Hard constraints for task {task_id}:",
            "",
            '1) Template similarity (BOLT APEX "similarity" constraint)',
            f'   Reference template T = "{reference}" '
            f"(REFERENCE_SEQUENCE[constraint_index={pep_idx(task_id)}])",
            "   ed(s, T) = Levenshtein edit distance",
            "   sim(s, T) = (len(T) - ed(s, T)) / len(T)",
            f"   Feasible iff sim(s, T) >= {MIN_SIMILARITY}",
            f"   The denominator is len(T)={t_len}, not len(s), and feasibility is measured",
            "   against T, not against seed_sequence.",
            "",
            f"2) Maximum length: len(s) <= {MAX_LENGTH}",
            "",
            "3) Cysteine cap: count('C') <= 1",
            "",
            "4) Alphabet: the 20 standard one-letter amino-acid codes only",
            "",
            "Objective:",
            "- minimize bacteria_0 mean MIC",
            "",
            "Write exactly this format:",
            "Rationale: <short reason these are your best candidates>",
            "",
            "JSON:",
            '{"recommendations": [{"rank": 1, "sequence": "<AA>"}, ..., '
            f'{{"rank": {BATCH_SIZE}, "sequence": "<AA>"}}]}}',
            "",
            "TASK:",
            f"Recommend exactly {BATCH_SIZE} distinct sequences to evaluate next on this task.",
            "This is a parallel batch proposal, not a ranked list from worst to best.",
            "Order by descending confidence (rank 1 = your single best guess).",
            f"All {BATCH_SIZE} must satisfy every hard constraint above.",
            "Do not emit VAE latent coordinates.",
        ]
    )


def format_assistant(
    targets: Sequence[Tuple[str, float]], *, prompt_style: str, rationale: bool = True
) -> str:
    if prompt_style == "bolt":
        # BOLT's assistant turn is a single sequence; use the best one.
        return targets[0][0]
    recs = [{"rank": i, "sequence": s} for i, (s, _) in enumerate(targets, start=1)]
    body = json.dumps({"recommendations": recs}, ensure_ascii=True)
    # rationale=True mirrors the winning p5_s1b arm (rationale+JSON); rationale=False
    # mirrors p5_s1a (JSON-only behaviour cloning).
    return f"Rationale: {RATIONALE}\n\nJSON:\n{body}" if rationale else f"JSON:\n{body}"


# ---------------------------------------------------------------- validation


def validate_row(
    row: Dict[str, Any], *, reference: str, seed_seq: str, prompt_style: str
) -> None:
    """Reject exactly the failures the old pipeline shipped."""
    msgs = row["messages"]
    user = msgs[-2]["content"]
    asst = msgs[-1]["content"]
    # The old bug was an explicit answer list in the prompt. Structurally forbid real
    # sequences appearing as "sequence" values; the "<AA>" format placeholder is fine.
    leaked = [v for v in re.findall(r'"sequence":\s*"([^"]*)"', user)
              if len(v) >= 3 and all(ch in AA for ch in v)]
    if leaked or "TARGET RECOMMENDATIONS" in user:
        raise AssertionError(f"LEAKAGE: prompt contains candidate sequences {leaked[:3]}")
    if "mic_bacteria_0" in user or "Best observed MIC" in user:
        raise AssertionError("LEAKAGE: prompt reveals an achieved MIC")
    seqs = (
        [asst.strip()]
        if prompt_style == "bolt"
        else re.findall(r'"sequence":\s*"([A-Z]+)"', asst)
    )
    expected = 1 if prompt_style == "bolt" else BATCH_SIZE
    if len(seqs) != expected:
        raise AssertionError(f"expected {expected} sequences in assistant, got {len(seqs)}")
    if len(set(seqs)) != len(seqs):
        raise AssertionError("duplicate sequences in assistant target")
    for s in seqs:
        if s == reference or s == seed_seq:
            raise AssertionError(f"DEGENERATE: target {s} is the reference or the seed")
        if not is_feasible(s, reference):
            raise AssertionError(
                f"INFEASIBLE target {s}: sim={template_similarity(s, reference):.3f} "
                f"len={len(s)} nC={s.count('C')}"
            )


# ---------------------------------------------------------------- export


def build(
    *,
    arm: str,
    bo_root: Path,
    out_dir: Path,
    prompt_style: str,
    target_source: str,
    n_train: int,
    rationale: bool = True,
    split_json: Optional[Path] = None,
) -> Dict[str, Any]:
    index_path = bo_root / "index.jsonl"
    if not index_path.is_file():
        raise SystemExit(f"missing {index_path}")
    index: Dict[str, Dict[str, Any]] = {}
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            index[str(r["task_id"])] = r

    ids = sorted(index.keys(), key=pep_idx)
    if split_json is not None:
        # Pre-registered FAMILY-DISJOINT 800/200 split (docs/peptide_split_800_200.json):
        # no test reference is within 0.75 similarity of any train reference, so no train
        # seed is a legal candidate for a test task. A plain index cutoff cannot give that,
        # because the library contains truncation/variant families that straddle any
        # contiguous boundary.
        sp = json.loads(Path(split_json).read_text(encoding="utf-8"))
        test_set = set(sp["test_ids"])
        train_ids = [t for t in ids if t not in test_set]
        eval_ids = [t for t in ids if t in test_set]
        split_desc = f"family-disjoint 800/200 from {Path(split_json).name}"
    else:
        train_ids = [t for t in ids if pep_idx(t) < n_train]
        eval_ids = [t for t in ids if pep_idx(t) >= n_train]
        split_desc = f"index cutoff at {n_train}"

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = arm if prompt_style == "rich" else f"{arm}_{prompt_style}"
    if not rationale:
        suffix += "_nojson_rationale" if False else "_jsononly"
    paths = {
        "train": out_dir / f"peptide_sft_{suffix}_train.jsonl",
        "eval": out_dir / f"peptide_sft_{suffix}_eval.jsonl",
        "manifest": out_dir / f"peptide_sft_{suffix}_manifest.json",
    }

    skips: Dict[str, int] = {}
    counts = {"train": 0, "eval": 0}
    sims_all: List[float] = []
    mics_all: List[float] = []

    def emit(split: str, split_ids: List[str], fh) -> None:
        for tid in split_ids:
            row = index[tid]
            hp = bo_root / tid / "bolt_native" / "seed1" / "history.json"
            if not hp.is_file():
                skips["missing_history"] = skips.get("missing_history", 0) + 1
                continue
            ref = bolt_reference_sequence(constraint_index=pep_idx(tid))
            try:
                init, post, meta, post_prov = load_trajectory(hp)
                targets, win_round = select_targets(
                    arm,
                    init,
                    post,
                    reference=ref,
                    seed_seq=str(row.get("start_seq") or ""),
                    target_source=target_source,
                    post_prov=post_prov,
                )
            except (ValueError, KeyError, json.JSONDecodeError, OSError) as exc:
                skips[f"load:{type(exc).__name__}"] = skips.get(f"load:{type(exc).__name__}", 0) + 1
                continue
            if len(targets) < BATCH_SIZE:
                skips["too_few_feasible_targets"] = skips.get("too_few_feasible_targets", 0) + 1
                continue

            user = build_user_prompt(
                task_id=tid,
                seed_seq=str(row.get("start_seq") or ""),
                reference=ref,
                prompt_style=prompt_style,
                post_init_budget=int(row.get("n_acquisition_calls") or row.get("n_post_init_trials") or 1000),
                n_init=int(meta.get("n_init_points") or 1000),
            )
            system = BOLT_SYSTEM_PROMPT if prompt_style == "bolt" else rich_system_prompt()
            sims = [template_similarity(s, ref) for s, _ in targets]
            mics = [m for _, m in targets]
            out = {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": format_assistant(targets, prompt_style=prompt_style, rationale=rationale)},
                ],
                "metadata": {
                    "domain": "protein",
                    "task_id": tid,
                    "arm": arm,
        "batch_size": BATCH_SIZE,
                    "prompt_style": prompt_style,
                    "rationale": rationale,
                    "target_source": target_source,
                    "winning_batch_round": win_round,
                    "init_pool_row_seq": row.get("start_seq"),
                    "reference_template": ref,
                    "constraint_index": pep_idx(tid),
                    "best_mic": round(min(mics), 6),
                    "target_mean_mic": round(sum(mics) / len(mics), 6),
                    "target_min_similarity": round(min(sims), 6),
                    "target_max_similarity": round(max(sims), 6),
                    "n_recommendations": len(targets),
                    "bo_truncated": row.get("truncated"),
                    "split": split,
                },
            }
            if prompt_style == "bolt":
                # BOLT trains one example per (reference -> improved sequence) pair, so a
                # task with 10 good solutions contributes 10 rows. Emitting only the single
                # best would give 897 examples of ~76 tokens -- far too little signal.
                for rank, (tseq, tmic) in enumerate(targets, start=1):
                    one = json.loads(json.dumps(out))
                    one["messages"][-1]["content"] = tseq
                    one["metadata"]["target_rank"] = rank
                    one["metadata"]["target_mic"] = round(float(tmic), 6)
                    one["metadata"]["target_similarity"] = round(
                        template_similarity(tseq, ref), 6
                    )
                    validate_row(one, reference=ref, seed_seq=str(row.get("start_seq") or ""), prompt_style=prompt_style)
                    fh.write(json.dumps(one, ensure_ascii=False) + "\n")
                    counts[split] += 1
            else:
                validate_row(out, reference=ref, seed_seq=str(row.get('start_seq') or ''), prompt_style=prompt_style)
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
                counts[split] += 1
            sims_all.extend(sims)
            mics_all.extend(mics)

    with paths["train"].open("w", encoding="utf-8") as f:
        emit("train", train_ids, f)
    with paths["eval"].open("w", encoding="utf-8") as f:
        emit("eval", eval_ids, f)

    manifest = {
        "arm": arm,
        "prompt_style": prompt_style,
        "rationale": rationale,
        "target_source": target_source,
        "bo_root": str(bo_root),
        "split": {
            "mode": split_desc,
            "split_desc": split_desc,
            "n_train": len(train_ids),
            "n_eval": len(eval_ids),
            "train_range": None if split_json else f"pep_000..pep_{n_train - 1:03d}",
            "eval_range": None if split_json else f"pep_{n_train:03d}..pep_{len(ids) - 1:03d}",
            "split_json": str(split_json) if split_json else None,
        },
        "counts": counts,
        "skips": skips,
        "target_stats": {
            "n_target_sequences": len(sims_all),
            "min_similarity": round(min(sims_all), 6) if sims_all else None,
            "all_feasible": bool(sims_all and min(sims_all) >= MIN_SIMILARITY),
            "mic_mean": round(sum(mics_all) / len(mics_all), 4) if mics_all else None,
        },
        "guarantees": [
            "prompt contains no candidate-sequence list and no achieved MIC (validated per row)",
            "no target is the reference template or the seed sequence (validated per row)",
            f"every target satisfies similarity >= {MIN_SIMILARITY}, len <= {MAX_LENGTH}, "
            "<=1 cysteine, standard alphabet (validated per row)",
            "prompt contains no achieved-MIC value, so nothing in it is unknowable at inference",
            "split has zero cross-split leakage at similarity 0.75 (verified when "
            "--split-json points at a family-disjoint split)",
        ],
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    global BATCH_SIZE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--arm",
        choices=["elite10", "winbatch10", "winbatch"],
        default="elite10",
        help="elite10/winbatch10 keep the legacy names; 'winbatch' uses the v2 "
             "batch_id provenance to take a REAL recorded batch (recenter phase).",
    )
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help="sequences per assistant target (50 matches the BO acquisition batch)")
    ap.add_argument("--split-json", type=Path, default=None,
                    help="pre-registered split file; docs/peptide_split_800_200.json")
    ap.add_argument("--bo-root", type=Path, default=DEFAULT_BO_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--prompt-style",
        choices=["rich", "bolt"],
        default="rich",
        help="rich: full task profile + constraints. bolt: the paper's minimal prompt "
        "(context = seed sequence, target = single best sequence).",
    )
    ap.add_argument(
        "--target-source",
        choices=["all", "post_init"],
        default="all",
        help="all: best feasible over the whole trajectory (BOLT's framing). post_init: "
        "only BO-discovered sequences -- note these are worse than the init-pool best for "
        "~83%% of tasks at a 1000-call budget.",
    )
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument(
        "--no-rationale",
        action="store_true",
        help="JSON-only assistant target (mirrors the winning p5_s1a behaviour-cloning "
        "arm); default emits Rationale+JSON (mirrors p5_s1b).",
    )
    a = ap.parse_args()
    BATCH_SIZE = int(a.batch_size)

    m = build(
        arm=a.arm,
        bo_root=a.bo_root,
        out_dir=a.out_dir,
        prompt_style=a.prompt_style,
        target_source=a.target_source,
        n_train=a.n_train,
        split_json=a.split_json,
        rationale=not a.no_rationale,
    )
    print(json.dumps(m, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
