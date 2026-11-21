"""TaskSet ckpt32 training data: P5 SFT events, ORPO attribution pairs, GRPO prefixes.

Mirrors the ML ckpt32 recipe on the train838 TPE BO histories:

* SFT ("s1b"): one record per improvement event. At trial t whose loss beats every
  earlier trial, the prompt is the TaskSet P5 prompt over trials 1..t-1 (exactly
  what the in-loop model sees at trial t, including similar-task cards in trials
  4-8 and the explore/exploit block from trial 50) and the target is the trial-t
  configuration. Answers are json_only (``{"config": {...}}``), the inference format.
* ORPO ("attr_t2"): for each event, a rejected configuration from the same history
  (same optimizer family, worse loss) chosen on the Pareto front of
  (similarity, worseness) by argmax 0.5*sim + 0.5*worse; the chosen side is the
  rejected configuration with its *blamed* parameters repaired to the event's
  values (numeric: normalized difference >= 0.15 on the log/linear search-space
  scale; categorical: differing value). Pairs blaming nothing or more than half of
  the family's parameters are dropped.
* GRPO prefixes: frozen-history prefixes with a scenario per prefix length.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from mindware.domains.dlhpo.taskset_llm import experience as E
from mindware.domains.dlhpo.taskset_llm import p5_prompt as P
from mindware.domains.dlhpo.taskset_llm import spec as S

MIN_PREFIX = 3            # the in-loop LLM never sees fewer than the 3 warm-start trials
BLAME_THRESHOLD = 0.15    # ML attr_t2 SPACE_REL_THRESH
MAX_BLAME_FRACTION = 0.5
PROMPT_SEED = 1

# GRPO prefix schedule (TaskSet version of the ML v12 mix): k = number of history trials shown.
PREFIX_SCENARIOS = {3: "retrieval", 5: "retrieval", 7: "retrieval",
                    20: "normal", 30: "normal", 40: "normal",
                    50: "explore", 70: "explore", 90: "explore",
                    60: "exploit", 80: "exploit", 99: "exploit"}


def history_rows(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"trial": i + 1, "config": dict(c), "loss": float(y), "successful": True}
            for i, (c, y) in enumerate(zip(record["configs"], record["y_minimize"]))]


def improvement_events(rows: Sequence[Dict[str, Any]], min_prefix: int = MIN_PREFIX) -> List[Dict[str, Any]]:
    events, best = [], math.inf
    for i, row in enumerate(rows):
        if row["loss"] < best:
            if i >= min_prefix:
                prefix = [r["loss"] for r in rows[:i]]
                lo, hi = min(prefix), max(prefix)
                rel = 1.0 if hi == lo else (hi - row["loss"]) / (hi - lo)  # rel_score on the prefix scale
                events.append({"trial": i + 1, "prefix_len": i, "config": row["config"], "loss": row["loss"],
                               "delta_rel_score": rel - 1.0 if hi != lo else 0.0, "rel_on_prefix": rel})
            best = row["loss"]
    return events


def prefix_prompt(task_id: str, specs, cs: ConfigurationSpace, rows: Sequence[Dict[str, Any]], trial: int,
                  retrieval: Optional[Callable[[str, List[Dict[str, Any]]], List[str]]] = None,
                  force_branch: Optional[str] = None) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    prefix = list(rows[:trial - 1])
    lines = retrieval(task_id, prefix) if (retrieval and E.retrieval_window(trial)) else None
    system, user, meta = P.build_p5_prompt(task_id, specs, cs, prefix, trial=trial, seed=PROMPT_SEED,
                                           retrieval_lines=lines, force_branch=force_branch)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], meta


def answer(config: Dict[str, Any]) -> str:
    return S.config_json({"config": config})


# ------------------------------------------------------------- attribution
def _normalized(hp: Any, value: float) -> float:
    lo, hi, v = float(hp.lower), float(hp.upper), float(value)
    if getattr(hp, "log", False):
        return (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return (v - lo) / (hi - lo)


def blamed_params(cs: ConfigurationSpace, good: Dict[str, Any], bad: Dict[str, Any]) -> List[str]:
    family = good[S.FAMILY_HP]
    out = []
    for name in S.family_params(cs, family):
        hp = cs.get_hyperparameter(name)
        if isinstance(hp, CategoricalHyperparameter):
            if good[name] != bad[name]:
                out.append(name)
        elif abs(_normalized(hp, good[name]) - _normalized(hp, bad[name])) >= BLAME_THRESHOLD:
            out.append(name)
    return out


def similarity(cs: ConfigurationSpace, a: Dict[str, Any], b: Dict[str, Any]) -> float:
    names = S.family_params(cs, a[S.FAMILY_HP])
    total = 0.0
    for name in names:
        hp = cs.get_hyperparameter(name)
        if isinstance(hp, CategoricalHyperparameter):
            total += float(a[name] == b[name])
        else:
            total += 1.0 - abs(_normalized(hp, a[name]) - _normalized(hp, b[name]))
    return total / len(names)


def pick_rejected(cs: ConfigurationSpace, rows: Sequence[Dict[str, Any]], event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    family = event["config"][S.FAMILY_HP]
    worst = max(r["loss"] for r in rows)
    span = worst - event["loss"]
    cands = [r for r in rows if r["config"][S.FAMILY_HP] == family and r["loss"] > event["loss"]
             and S.signature(r["config"]) != S.signature(event["config"])]
    if not cands or span <= 0:
        return None
    scored = [(similarity(cs, event["config"], r["config"]), (r["loss"] - event["loss"]) / span, r) for r in cands]
    front = [s for s in scored if not any(o[0] >= s[0] and o[1] >= s[1] and (o[0] > s[0] or o[1] > s[1]) for o in scored)]
    return max(front, key=lambda s: (0.5 * s[0] + 0.5 * s[1], -s[2]["trial"]))[2]


def attribution_pair(cs: ConfigurationSpace, rows: Sequence[Dict[str, Any]], event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    rejected = pick_rejected(cs, rows, event)
    if rejected is None:
        return None
    blamed = blamed_params(cs, event["config"], rejected["config"])
    n = len(S.family_params(cs, event["config"][S.FAMILY_HP]))
    if not blamed or len(blamed) > MAX_BLAME_FRACTION * n:
        return None
    repaired = dict(rejected["config"])
    repaired.update({name: event["config"][name] for name in blamed})
    if S.signature(repaired) == S.signature(rejected["config"]):
        return None
    return {"chosen": repaired, "rejected": rejected["config"], "blamed": blamed,
            "rejected_trial": rejected["trial"], "rejected_loss": rejected["loss"]}
