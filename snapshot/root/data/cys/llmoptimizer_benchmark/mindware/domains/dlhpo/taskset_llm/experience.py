"""TaskSet experience bank and CASH similar-task rerank.

Bank: one card per train838 task, built from its real TPE BO history (best config,
per-optimizer-family statistics, good/bad regions of the best family, top configs)
plus a flattened task-spec feature dict.

Rerank (TaskSet analogue of the ML ``source_gp`` ranker):
  * candidates = every train task of the target's task family (all train/test tasks
    share one CASH space, so no "same algorithm" filter is needed);
  * active on LLM turns at trials [RETRIEVE_START, RETRIEVE_STOP) with >= 3
    non-constant successful observations;
  * score = Spearman between each candidate's source GP (the RGPE base GPs) predicted
    loss and the target's observed loss at the target's evaluated configs; mixed-family
    configs make this a CASH-level agreement (which family / region is good);
  * floor 0.5, top-3; ties (frequent with 3-7 points) broken by within-family
    standardized task-spec cosine, then task id.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from mindware.domains.dlhpo.taskset_llm import spec as S
from mindware.domains.dlhpo.taskset_llm.p5_prompt import all_family_regions, annotate

# v2 (2026-10-06): regions for every optimizer family plus the family-choice line, built by
# the same function as the in-loop exploit block (v1 kept only the best family's regions).
BANK_SCHEMA = "taskset_experience.v2"
RETRIEVE_START, RETRIEVE_STOP = 4, 9
RETRIEVE_K = 3
SPEARMAN_FLOOR = 0.5
MIN_OBSERVATIONS = 3

ADVICE_HEADER = (
    "Source-task advice: Similarity is tentative and based on how well each source task's "
    "surrogate ranks the current run's observed configurations. These summaries describe "
    "historical source results and may not transfer to this task. Use them as suggestions, "
    "weigh the current task's results, and choose the next configuration from the full search "
    "space. Conflicting sources are alternatives."
)


# ------------------------------------------------------------ spec features
def spec_features(spec: Any) -> Dict[str, float]:
    """Flatten a TaskSet task spec into numeric features (log-scaled numbers, one-hot strings)."""
    out: Dict[str, float] = defaultdict(float)

    def num(x: float) -> float:
        return math.copysign(math.log1p(abs(float(x))), float(x))

    def walk(obj: Any, path: str) -> None:
        if isinstance(obj, bool):
            out[path] += float(obj)
        elif isinstance(obj, (int, float)):
            out[path] += num(obj)
        elif isinstance(obj, str):
            out[f"{path}={obj}"] += 1.0
        elif obj is None:
            out[f"{path}=null"] += 1.0
        elif isinstance(obj, dict):
            for key in sorted(obj):
                walk(obj[key], f"{path}.{key}" if path else str(key))
        elif isinstance(obj, (list, tuple)):
            out[f"{path}#len"] += float(len(obj))
            numbers = [x for x in obj if isinstance(x, (int, float)) and not isinstance(x, bool)]
            if numbers and len(numbers) == len(obj):
                out[f"{path}#mean"] += num(np.mean(numbers))
                out[f"{path}#max"] += num(max(numbers))
            else:
                for item in obj:
                    walk(item, f"{path}[]")

    walk(spec, "")
    return dict(out)


def profile_stats(features_by_task: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
    """Per task family: feature names with mean/std over the family's train tasks."""
    by_family: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for task, feats in features_by_task.items():
        by_family[S.task_family_of(task)].append(feats)
    stats = {}
    for family, rows in by_family.items():
        names = sorted({k for r in rows for k in r})
        matrix = np.array([[r.get(k, 0.0) for k in names] for r in rows], dtype=float)
        std = matrix.std(axis=0)
        keep = std > 1e-12
        stats[family] = {"names": [n for n, k in zip(names, keep) if k],
                         "mean": matrix.mean(axis=0)[keep].tolist(), "std": std[keep].tolist()}
    return stats


def profile_vector(features: Dict[str, float], stats: Dict[str, Any]) -> np.ndarray:
    x = np.array([features.get(n, 0.0) for n in stats["names"]], dtype=float)
    return (x - np.array(stats["mean"])) / np.array(stats["std"])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return 0.0 if na == 0 or nb == 0 else float(a @ b / (na * nb))


# --------------------------------------------------------------------- bank
def build_card(task_id: str, record: Dict[str, Any], spec_record: Dict[str, Any], cs) -> Dict[str, Any]:
    configs = record["configs"]
    losses = [float(y) for y in record["y_minimize"]]
    rows = [{"trial": i + 1, "config": c, "loss": y, "successful": True} for i, (c, y) in enumerate(zip(configs, losses))]
    successes = [{"trial": r["trial"], "config": r["config"], "score": -r["loss"],
                  "family": r["config"][S.FAMILY_HP]} for r in rows]
    info = annotate(successes)
    rel = [dict(s, score=info[s["trial"]]["rel_score"]) for s in successes]
    fams: Dict[str, Dict[str, Any]] = {}
    for family in S.families(cs):
        own = [y for c, y in zip(configs, losses) if c[S.FAMILY_HP] == family]
        if own:
            fams[family] = {"n": len(own), "best_loss": min(own), "median_loss": float(np.median(own))}
    ranking = sorted(fams, key=lambda f: fams[f]["best_loss"])
    order = sorted(range(len(losses)), key=lambda i: (losses[i], i))
    best = order[0]
    best_family = configs[best][S.FAMILY_HP]
    return {
        "schema_version": BANK_SCHEMA, "id": f"taskset/{task_id}",
        "problem": {"domain": "taskset_cash", "task_id": task_id, "task_family": S.task_family_of(task_id),
                    "spec": spec_record["spec"], "features": spec_features(spec_record["spec"])},
        "result": {"n_observations": len(losses), "best_loss": losses[best], "best_trial": best + 1,
                   "best_config": configs[best]},
        "families": fams, "family_ranking": ranking,
        "best_family": best_family,
        "regions": all_family_regions(rel, cs, configs[best]),
        "top": [{"trial": i + 1, "loss": losses[i], "config": configs[i]} for i in order[:3]],
        "source_sha256": record["source_sha256"],
    }


def load_bank(path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    path = Path(path)
    cards = {}
    with path.open() as stream:
        for line in stream:
            card = json.loads(line)
            if card.get("schema_version") != BANK_SCHEMA:
                raise ValueError(f"Unexpected bank schema in {path}")
            cards[card["problem"]["task_id"]] = card
    stats = json.loads(path.with_name(path.stem + ".profile_stats.json").read_text())
    return cards, stats


# ------------------------------------------------------------------- rerank
def rank_values(x: Sequence[float]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    ra, rb = rank_values(a), rank_values(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def retrieval_window(trial: int) -> bool:
    return RETRIEVE_START <= int(trial) < RETRIEVE_STOP


def candidates_for(task_id: str, cards: Dict[str, Dict[str, Any]], exclude_self: bool = True) -> List[str]:
    family = S.task_family_of(task_id)
    return sorted(t for t, c in cards.items()
                  if c["problem"]["task_family"] == family and not (exclude_self and t == task_id))


def rerank(
    task_id: str,
    target_rows: Sequence[Dict[str, Any]],
    cards: Dict[str, Dict[str, Any]],
    stats: Dict[str, Any],
    specs: Dict[str, Dict[str, Any]],
    predict: Any,
    *,
    k: int = RETRIEVE_K,
    floor: float = SPEARMAN_FLOOR,
) -> Dict[str, Any]:
    """Rank same-family sources; ``predict(tasks, configs) -> {task: [pred loss]}``."""
    ok = [r for r in target_rows if r.get("successful")]
    audit: Dict[str, Any] = {"n_observations": len(ok), "floor": floor, "k": k, "hits": [], "scored": []}
    losses = [float(r["loss"]) for r in ok]
    if len(ok) < MIN_OBSERVATIONS or len(set(losses)) < 2:
        audit["skipped"] = "too_few_or_constant_observations"
        return audit
    pool = candidates_for(task_id, cards)
    family_stats = stats.get(S.task_family_of(task_id))
    target_vec = profile_vector(spec_features(specs[task_id]["spec"]), family_stats) if family_stats else None
    predictions = predict(pool, [r["config"] for r in ok])
    scored = []
    for source in pool:
        rho = spearman(predictions[source], losses)
        sim = (cosine(target_vec, profile_vector(cards[source]["problem"]["features"], family_stats))
               if target_vec is not None else 0.0)
        scored.append({"task": source, "spearman": rho, "profile_cos": sim})
    valid = [s for s in scored if math.isfinite(s["spearman"])]
    valid.sort(key=lambda s: (-s["spearman"], -s["profile_cos"], s["task"]))
    audit["scored"] = valid
    audit["hits"] = [s for s in valid if s["spearman"] >= floor][:k]
    return audit


def card_lines(card: Dict[str, Any], hit: Dict[str, Any]) -> List[str]:
    fams = card["families"]
    ranking = ", ".join(f"{f} {fams[f]['best_loss']:.4g}" for f in card["family_ranking"])
    lines = [f"Similar task: {card['problem']['task_id']} (Spearman agreement={hit['spearman']:.2f})",
             f"- optimizer families by best loss (lower is better): {ranking}",
             f"- best config (loss {card['result']['best_loss']:.4g}): {S.display_config(card['result']['best_config'])}"]
    regions = card["regions"]
    if regions["family_level"] or regions["by_family"]:
        lines.append("- good/bad regions on that task:")
        if regions["family_level"]:
            lines.append("  " + regions["family_level"])
        for family in card["family_ranking"]:
            if family in regions["by_family"]:
                lines.append(f"  - {family}:")
                lines += ["    " + line.strip() for line in regions["by_family"][family]]
    return lines


def retrieval_block(audit: Dict[str, Any], cards: Dict[str, Dict[str, Any]]) -> List[str]:
    if not audit.get("hits"):
        return []
    lines = [ADVICE_HEADER]
    for hit in audit["hits"]:
        lines += [""] + card_lines(cards[hit["task"]], hit)
    return lines
