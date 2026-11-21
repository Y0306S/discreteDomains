"""Shared helpers for peptide SFT ablation exports (elite10 vs winbatch10)."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from mindware.domains.protein.bolt_constraints import bolt_reference_sequence, template_similarity
from mindware.domains.protein.bolt_lolbo_runner import CONSISTENT10

ArmName = Literal["elite10", "winbatch10"]

DEFAULT_BO_ROOT = Path(
    "/root/data/cys/blockdata_archive/domain_protein/bo80_bolt_bsz10_pilot5"
)
N_INIT_DEFAULT = 1000
BATCH_SIZE = 10
MIN_SIMILARITY = 0.75
MAX_LENGTH = 30

SYSTEM_PROMPT = (
    "You are a peptide sequence optimization assistant for antimicrobial peptide (AMP) design.\n"
    "The search uses a BOLT UniRef VAE latent space with bacteria_0 mean-MIC oracle (lower MIC is better).\n"
    "Propose a parallel batch of 10 distinct amino-acid sequences for the next oracle evaluation on this task.\n"
    "First write a short rationale, then write one JSON object containing exactly 10 ranked recommendations.\n"
    "Do not use markdown."
)

RATIONALE_TEMPLATE = (
    "Propose a parallel batch of ten feasible L/R-rich cationic variants in the low-MIC motif "
    "basin found by search; rank 1 is the incumbent best, ranks 2-10 are the next-best distinct "
    "candidates still worth evaluating under template-similarity, length, and cysteine constraints."
)


def pep_idx(task_id: str) -> int:
    return int(str(task_id).strip().split("_")[-1])


def all_task_ids(n: int = 1000) -> List[str]:
    return [f"pep_{i:03d}" for i in range(int(n))]


def load_index(index_path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with index_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[str(row["task_id"])] = row
    return rows


def history_path(bo_root: Path, task_id: str) -> Path:
    return bo_root / task_id / "bolt_native" / "seed1" / "history.json"


def _seq_from_obs(o: Dict[str, Any]) -> str:
    extra = o.get("extra_info") or {}
    info = extra.get("info") or {}
    return str(extra.get("artifact") or info.get("sequence") or "")


def _mic_from_obs(o: Dict[str, Any]) -> float:
    extra = o.get("extra_info") or {}
    info = extra.get("info") or {}
    mic = info.get("mic_bacteria_0")
    if mic is not None:
        return float(mic)
    return float(o["objectives"][0])


def load_unique_new(
    hist_path: Path,
    *,
    n_init: int = N_INIT_DEFAULT,
) -> Tuple[List[Tuple[str, float]], float, Dict[str, Any]]:
    """Return (unique-new seq,mic pairs in order), best_init_mic, meta."""
    payload = json.loads(hist_path.read_text(encoding="utf-8"))
    meta = payload.get("meta_info") or {}
    obs = payload.get("observations") or []
    n_init = int(meta.get("n_init_points") or n_init)

    init_mics: List[float] = []
    seen: set[str] = set()
    for o in obs[:n_init]:
        seq = _seq_from_obs(o)
        seen.add(seq)
        init_mics.append(_mic_from_obs(o))
    best_init = min(init_mics) if init_mics else float("inf")

    unique_new: List[Tuple[str, float]] = []
    for o in obs[n_init:]:
        seq = _seq_from_obs(o)
        if not seq or seq in seen:
            continue
        seen.add(seq)
        unique_new.append((seq, _mic_from_obs(o)))
    return unique_new, best_init, meta


def batches_of_ten(unique_new: Sequence[Tuple[str, float]]) -> List[List[Tuple[str, float]]]:
    out: List[List[Tuple[str, float]]] = []
    for i in range(0, len(unique_new), BATCH_SIZE):
        chunk = list(unique_new[i : i + BATCH_SIZE])
        if len(chunk) == BATCH_SIZE:
            out.append(chunk)
    return out


def elite10_targets(unique_new: Sequence[Tuple[str, float]]) -> List[Tuple[str, float]]:
    ranked = sorted(unique_new, key=lambda x: (x[1], x[0]))
    if len(ranked) < BATCH_SIZE:
        raise ValueError(f"need at least {BATCH_SIZE} unique-new sequences, got {len(ranked)}")
    return ranked[:BATCH_SIZE]


def winbatch10_targets(
    unique_new: Sequence[Tuple[str, float]],
    *,
    best_init_mic: float,
) -> Tuple[List[Tuple[str, float]], int]:
    """Batch of 10 from the first round that sets a new global best MIC."""
    if len(unique_new) < BATCH_SIZE:
        raise ValueError(f"need at least {BATCH_SIZE} unique-new sequences, got {len(unique_new)}")
    batches = batches_of_ten(unique_new)
    global_best = min(m for _, m in unique_new)
    global_best_seq = min(unique_new, key=lambda x: (x[1], x[0]))[0]
    running = float(best_init_mic)
    for bi, batch in enumerate(batches):
        batch_best = min(m for _, m in batch)
        if batch_best >= running - 1e-9:
            continue
        if abs(batch_best - global_best) <= 1e-6 or any(s == global_best_seq for s, _ in batch):
            ranked = sorted(batch, key=lambda x: (x[1], x[0]))
            return ranked, bi
        running = batch_best
    # Fallback: batch containing global best by membership
    for bi, batch in enumerate(batches):
        if any(s == global_best_seq for s, _ in batch):
            return sorted(batch, key=lambda x: (x[1], x[0])), bi
    raise ValueError("could not locate winning batch for global best")


def extract_targets(
    arm: ArmName,
    unique_new: Sequence[Tuple[str, float]],
    *,
    best_init_mic: float,
) -> Tuple[List[Tuple[str, float]], Optional[int]]:
    if arm == "elite10":
        return elite10_targets(unique_new), None
    batch, round_idx = winbatch10_targets(unique_new, best_init_mic=best_init_mic)
    return batch, round_idx


def split_tasks(
    task_ids: Sequence[str],
    *,
    mode: str = "stratified10",
    seed: int = 42,
    n_eval: int = 100,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Return (train_ids, eval_ids, split_meta)."""
    ids = sorted(task_ids, key=pep_idx)
    if mode == "stratified10":
        eval_ids = [tid for tid in ids if pep_idx(tid) % 10 == 0]
        train_ids = [tid for tid in ids if tid not in set(eval_ids)]
    elif mode == "consistent10_plus90":
        eval_set = set(CONSISTENT10)
        remaining = [tid for tid in ids if tid not in eval_set]
        rng = random.Random(int(seed))
        extra = n_eval - len(eval_set)
        if extra < 0:
            raise ValueError(f"n_eval={n_eval} smaller than CONSISTENT10={len(CONSISTENT10)}")
        if extra > len(remaining):
            raise ValueError("not enough tasks to sample extra eval ids")
        eval_ids = list(CONSISTENT10) + rng.sample(remaining, extra)
        eval_ids = sorted(set(eval_ids), key=pep_idx)
        train_ids = [tid for tid in ids if tid not in set(eval_ids)]
    elif mode == "last100":
        eval_ids = ids[-n_eval:]
        train_ids = ids[:-n_eval]
    else:
        raise ValueError(f"unknown split mode: {mode}")

    meta = {
        "mode": mode,
        "seed": seed,
        "n_total": len(ids),
        "n_train": len(train_ids),
        "n_eval": len(eval_ids),
        "eval_ids": eval_ids,
        "train_ids_preview": train_ids[:20],
    }
    return train_ids, eval_ids, meta


def task_profile_json(row: Dict[str, Any], *, reference_template: str, constraint_index: int) -> str:
    profile = {
        "task_id": row["task_id"],
        "domain": "protein",
        "oracle": row.get("oracle", "bolt_bacteria_0"),
        "metric": "mean_MIC_bacteria_0",
        "higher_is_better": False,
        "start_sequence": row.get("start_seq", ""),
        "reference_template": reference_template,
        "constraint_index": int(constraint_index),
        "max_length": int(row.get("max_string_length") or MAX_LENGTH),
        "warm_start_evaluated": int(row.get("n_init_points") or N_INIT_DEFAULT),
        "post_init_budget": int(row.get("n_post_init_trials") or 1500),
        "batch_size": BATCH_SIZE,
    }
    return json.dumps(profile, ensure_ascii=True, sort_keys=True)


def constraints_block(*, task_id: str, reference_template: str) -> str:
    t_len = len(reference_template)
    return (
        f"Hard constraints for task {task_id}:\n\n"
        "1) Template similarity (BOLT APEX \"similarity\" constraint)\n"
        f"   Reference template T = \"{reference_template}\" "
        f"(REFERENCE_SEQUENCE[constraint_index={pep_idx(task_id)}])\n"
        "   Levenshtein edit distance ed(s, T) = minimum insert/delete/substitute operations to transform s into T\n"
        f"   Template similarity sim(s, T) = (len(T) - ed(s, T)) / len(T)\n"
        f"   Feasible iff sim(s, T) >= {MIN_SIMILARITY}\n"
        f"   Denominator is len(T)={t_len}, not len(s). Feasibility is measured against T, not against start_sequence.\n\n"
        f"2) Maximum length: len(s) <= {MAX_LENGTH}\n\n"
        "3) Cysteine cap: count('C' in s) <= 1\n\n"
        "4) Alphabet: standard 20 one-letter amino-acid codes only; no gaps or non-letter symbols"
    )


def target_recommendations_block(targets: Sequence[Tuple[str, float]]) -> str:
    items = []
    for rank, (seq, mic) in enumerate(targets, start=1):
        items.append(
            {
                "rank": rank,
                "sequence": seq,
                "mic_bacteria_0": round(float(mic), 6),
            }
        )
    return "TARGET RECOMMENDATIONS:\n" + json.dumps(items, ensure_ascii=True, indent=2)


def task_instructions_block() -> str:
    return (
        "Write exactly this format:\n"
        "Rationale: <short reason these 10 are your best next-batch candidates for this task>\n\n"
        'JSON:\n{"recommendations": [{"rank": 1, "sequence": "<AA>"}, ..., {"rank": 10, "sequence": "<AA>"}]}\n\n'
        "TASK:\n"
        "Recommend exactly 10 distinct sequences you believe are the best candidates to evaluate next on this task, "
        "given the task definition above.\n"
        "This is a parallel batch proposal (like LOLBO batch_size=10), not a ranked list from worst to best.\n"
        "Order by descending confidence (rank 1 = your single best guess; ranks 2-10 = other strong candidates "
        "for the same batch).\n"
        "All 10 must be feasible under the hard constraints above.\n"
        "Use only standard one-letter amino-acid codes. Do not emit VAE latent coordinates (no z_0, z_1, ...)."
    )


def build_user_prompt(
    row: Dict[str, Any],
    targets: Sequence[Tuple[str, float]],
    *,
    reference_template: str,
    constraint_index: int,
    best_mic: float,
) -> str:
    parts = [
        f"Task profile:\n{task_profile_json(row, reference_template=reference_template, constraint_index=constraint_index)}",
        "",
        "Fixed optimizer: bolt_lolbo (TuRBO trust region + Thompson sampling, batch_size=10)",
        "",
        "Design space:",
        "- sequence: string over amino acids A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y",
        f"- length: integer, range=[1, {MAX_LENGTH}] inclusive",
        "",
        constraints_block(task_id=str(row["task_id"]), reference_template=reference_template),
        "",
        "Objective:",
        "- bacteria_0 mean MIC (lower is better)",
        f"- Best observed MIC on this task after search: {round(float(best_mic), 6)}",
        "",
        "Selected historical trials: omitted for this final-recommendation query.",
        "",
        target_recommendations_block(targets),
        "",
        task_instructions_block(),
    ]
    return "\n".join(parts)


def format_assistant(targets: Sequence[Tuple[str, float]], *, rationale: Optional[str] = None) -> str:
    recs = [{"rank": i, "sequence": seq} for i, (seq, _) in enumerate(targets, start=1)]
    rat = (rationale or RATIONALE_TEMPLATE).strip()
    body = json.dumps({"recommendations": recs}, ensure_ascii=True)
    return f"Rationale: {rat}\n\nJSON:\n{body}"


def build_sft_row(
    row: Dict[str, Any],
    targets: Sequence[Tuple[str, float]],
    *,
    arm: ArmName,
    split: str,
    reference_template: str,
    constraint_index: int,
    win_round: Optional[int] = None,
    rationale: Optional[str] = None,
) -> Dict[str, Any]:
    best_mic = min(m for _, m in targets)
    user = build_user_prompt(
        row,
        targets,
        reference_template=reference_template,
        constraint_index=constraint_index,
        best_mic=best_mic,
    )
    sims = [template_similarity(seq, reference_template) for seq, _ in targets]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": format_assistant(targets, rationale=rationale)},
        ],
        "metadata": {
            "domain": "protein",
            "task_id": row["task_id"],
            "ablation": arm,
            "target_source": "final_top10_unique_post_init" if arm == "elite10" else "winning_batch_round",
            "winning_batch_round": win_round,
            "start_sequence": row.get("start_seq"),
            "reference_template": reference_template,
            "constraint_index": constraint_index,
            "best_mic": round(float(best_mic), 6),
            "target_mean_mic": round(sum(m for _, m in targets) / len(targets), 6),
            "target_min_template_sim": round(min(sims), 6),
            "target_max_template_sim": round(max(sims), 6),
            "n_recommendations": len(targets),
            "history_included": False,
            "split": split,
        },
    }


def export_arm(
    *,
    arm: ArmName,
    bo_root: Path,
    out_dir: Path,
    split_mode: str = "stratified10",
    seed: int = 42,
    task_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    index_path = bo_root / "index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    index = load_index(index_path)
    ids = sorted(task_ids or index.keys(), key=pep_idx)
    train_ids, eval_ids, split_meta = split_tasks(ids, mode=split_mode, seed=seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": out_dir / f"peptide_sft_{arm}_train.jsonl",
        "eval": out_dir / f"peptide_sft_{arm}_eval.jsonl",
        "split": out_dir / f"peptide_sft_{arm}_split.json",
        "manifest": out_dir / f"peptide_sft_{arm}_manifest.json",
    }

    skips: Dict[str, int] = {}
    counts = {"train": 0, "eval": 0}

    def _write_split(split_name: str, split_ids: List[str], fh) -> None:
        for tid in split_ids:
            row = index.get(tid)
            if row is None:
                skips["missing_index"] = skips.get("missing_index", 0) + 1
                continue
            hp = history_path(bo_root, tid)
            if not hp.is_file():
                skips["missing_history"] = skips.get("missing_history", 0) + 1
                continue
            try:
                unique_new, best_init, _meta = load_unique_new(hp)
                targets, win_round = extract_targets(arm, unique_new, best_init_mic=best_init)
            except (ValueError, json.JSONDecodeError, OSError) as exc:
                key = f"extract:{type(exc).__name__}"
                skips[key] = skips.get(key, 0) + 1
                continue
            cidx = pep_idx(tid)
            ref = bolt_reference_sequence(constraint_index=cidx)
            sft_row = build_sft_row(
                row,
                targets,
                arm=arm,
                split=split_name,
                reference_template=ref,
                constraint_index=cidx,
                win_round=win_round,
            )
            fh.write(json.dumps(sft_row, ensure_ascii=False) + "\n")
            counts[split_name] += 1

    with paths["train"].open("w", encoding="utf-8") as train_f:
        _write_split("train", train_ids, train_f)
    with paths["eval"].open("w", encoding="utf-8") as eval_f:
        _write_split("eval", eval_ids, eval_f)

    paths["split"].write_text(json.dumps(split_meta, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "arm": arm,
        "bo_root": str(bo_root),
        "split": split_meta,
        "counts": counts,
        "skips": skips,
        "outputs": {k: str(v) for k, v in paths.items()},
        "notes": [
            "SFT rows omit BO history (final-recommendation warm start).",
            "constraint_index=pep_idx; reference_template=REFERENCE_SEQUENCE[pep_idx].",
            "Default split stratified10: eval pep_XXX where XXX % 10 == 0 (100 tasks).",
        ],
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
