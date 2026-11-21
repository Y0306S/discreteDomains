#!/usr/bin/env python3
"""TaskSet ckpt32 GRPO stage: task selection and frozen-prefix rollout collection.

TaskSet port of the ML "b_prefix rsmix pareto32" collector (run_p5_rl_bo_prefix_b.py,
v14 semantics). Nothing here runs unless invoked.

select  : Pareto front over train838 TPE histories on (new bests in trials 40-100,
          relative improvement from best@40 to best@100); 32 tasks spread over the
          12 families (round-robin by family, Pareto rank then improvement).
collect : for each selected task and prefix k in training_data.PREFIX_SCENARIOS, the
          P5 prompt over the first k frozen BO trials (scenario retrieval -> similar
          task cards; explore / exploit -> forced branch), 4 samples at T=1.0 (up to
          10 while none beats the prefix incumbent), each scored by a real TaskSet
          evaluation (1000 steps, min_normalized_valid_outer) with the ML reward
          (rl_reward_utils.compute_trial_reward on score = -loss).

Rollouts: <out>/rollouts_<task>.jsonl, schema taskset_rl_rollout.v1 (trainer fields:
messages, completion, reward, group_id, sample_i, train_member, evaluation_status).
"""

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples/benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mindware.domains.dlhpo.taskset_adapter import TaskSetAdapter
from mindware.domains.dlhpo.taskset_llm import experience as E, spec as S, training_data as T
from mindware.domains.dlhpo.taskset_space import space_from_taskset
from rl_reward_utils import RewardWeights, compute_trial_reward
from taskset_transfer_common import METRIC, PROTOCOL, atomic_json, load_corpus, read_split

CORPUS = Path("/root/data/logs/taskset_transfer_train838_20261002/corpus")
SCHEMA = "taskset_rl_rollout.v1"
N_BASE, N_MAX, TEMPERATURE = 4, 10, 1.0


# ------------------------------------------------------------------ select
def late_improvement(losses):
    best40 = min(losses[:40])
    best_curve, new_bests = best40, 0
    for y in losses[40:]:
        if y < best_curve:
            best_curve, new_bests = y, new_bests + 1
    return new_bests, (best40 - best_curve) / max(abs(best40), 1e-12)


def select(args):
    _, tasks = load_corpus(args.corpus, check_sources=False)
    stats = {t: late_improvement([float(y) for y in r["y_minimize"]]) for t, r in tasks.items()}
    remaining, rank = dict(stats), {}
    level = 0
    while remaining:
        front = [t for t, (n, g) in remaining.items()
                 if not any(n2 >= n and g2 >= g and (n2 > n or g2 > g) for n2, g2 in remaining.values())]
        for t in front:
            rank[t] = level
            remaining.pop(t)
        level += 1
    by_family = {}
    for t in sorted(stats, key=lambda t: (rank[t], -stats[t][1], t)):
        by_family.setdefault(S.task_family_of(t), []).append(t)
    picked, families = [], sorted(by_family)
    while len(picked) < args.n and any(by_family.values()):
        for family in families:
            if by_family[family] and len(picked) < args.n:
                picked.append(by_family[family].pop(0))
    atomic_json(args.out, {"tasks": picked, "criterion": "pareto(new_bests_40_100, rel_improvement_40_100)",
                           "stats": {t: {"new_bests": stats[t][0], "rel_improvement": stats[t][1],
                                         "pareto_rank": rank[t]} for t in picked}})
    print(json.dumps(picked))


# ----------------------------------------------------------------- collect
def train_members(records, rng):
    """ML choose_train_members_v14."""
    usable = [i for i, r in enumerate(records) if not r["excluded_from_training"]]
    train = usable[:N_BASE]
    if not any(records[i]["improver"] for i in train):
        extra = [i for i in usable[N_BASE:] if records[i]["improver"]]
        if extra:
            injected = max(extra, key=lambda i: (records[i]["reward"]["delta_best"], records[i]["reward"]["reward"], i))
            dropped = rng.choice(train)
            train = [i for i in train if i != dropped] + [injected]
    if len(train) < 2 or len({round(records[i]["reward"]["reward"], 12) for i in train}) < 2:
        return [], "no_reward_signal"
    return train, "ok"


def score_sample(adapter, cs, prefix, text, weights):
    prior = [-r["loss"] for r in prefix]
    try:
        config, notes, _ = S.parse_reply(text, cs)
    except Exception as exc:
        return dict(evaluation_status="PARSE_FAILURE", excluded_from_training=False, config=None, error=str(exc),
                    reward={"reward": -weights.parse_fail, "delta_best": 0.0, "parse_ok": False}, improver=False)
    if S.signature(config) in {S.signature(r["config"]) for r in prefix}:
        return dict(evaluation_status="DUPLICATE", excluded_from_training=False, config=config, repairs=notes,
                    reward={"reward": -weights.duplicate, "delta_best": 0.0, "duplicate": True}, improver=False)
    started = time.time()
    result = adapter.evaluate_config(config)
    loss = float(result.objective)
    if not (result.feasible and math.isfinite(loss)):
        return dict(evaluation_status="EVAL_FAILED", excluded_from_training=True, config=config, repairs=notes,
                    reward={"reward": 0.0, "delta_best": 0.0, "eval_ok": False}, improver=False,
                    eval_seconds=time.time() - started, info=result.info)
    reward = compute_trial_reward(perf=-loss, prior_perfs=prior, parse_ok=True, duplicate=False,
                                  eval_ok=True, weights=weights).to_dict()
    return dict(evaluation_status="SUCCESS", excluded_from_training=False, config=config, repairs=notes,
                loss=loss, reward=reward, improver=loss < min(r["loss"] for r in prefix),
                eval_seconds=time.time() - started, info=result.info)


def collect(args):
    from mindware.domains.dlhpo.taskset_llm.llm_client import QwenLoRAClient
    from run_taskset_llm_rgpe import SourcePredictor

    split = read_split()
    _, tasks = load_corpus(args.corpus, check_sources=False)
    selection = json.loads(args.selection.read_text())["tasks"]
    if set(selection) - set(split["train"]):
        raise ValueError("GRPO rollouts must use train838 tasks only")
    specs, cs, weights = S.load_task_specs(), space_from_taskset(), RewardWeights()
    cards, stats = E.load_bank(args.bank)
    predictor = SourcePredictor(args.rgpe_assets, args.out / "_source_rank")
    client = QwenLoRAClient(str(args.adapter), temperature=TEMPERATURE, max_new_tokens=args.max_new_tokens)
    args.out.mkdir(parents=True, exist_ok=True)
    for task_id in selection:
        path = args.out / f"rollouts_{task_id}.jsonl"
        done = {json.loads(l)["group_id"] for l in path.read_text().splitlines()} if path.exists() else set()
        rows = T.history_rows(tasks[task_id])
        adapter = TaskSetAdapter(task={"task_id": task_id}, dummy=False, oracle_work_dir=args.out / "_oracle" / task_id,
                                 baseline_cache=args.baseline_cache, objective_metric=METRIC, **PROTOCOL)
        for k, scenario in sorted(T.PREFIX_SCENARIOS.items()):
            group_id = f"{task_id}:k{k}"
            if group_id in done:
                continue
            retrieval = ((lambda t, p: E.retrieval_block(E.rerank(t, p, cards, stats, specs, predictor), cards))
                         if scenario == "retrieval" else None)
            branch = scenario if scenario in ("explore", "exploit") else None
            messages, meta = T.prefix_prompt(task_id, specs, cs, rows, k + 1, retrieval, force_branch=branch)
            prefix, records = rows[:k], []
            while len(records) < N_MAX:
                n = N_BASE if not records else 1
                for text in client.generate(messages, n=n):
                    records.append(dict(score_sample(adapter, cs, prefix, text, weights), completion=text))
                if len(records) >= N_BASE and any(r["improver"] for r in records):
                    break
            members, group_status = train_members(records, random.Random(f"{group_id}:{args.seed}"))
            with path.open("a") as stream:
                for i, record in enumerate(records):
                    stream.write(json.dumps(dict(
                        record, schema=SCHEMA, group_id=group_id, sample_i=i, train_member=i in members,
                        group_status=group_status, task_id=task_id, dataset=task_id, task_type="taskset_cash",
                        algorithm="cash", task_family=S.task_family_of(task_id), prefix_k=k, prompt_trial=k + 1,
                        scenario=scenario, prompt_meta=meta, messages=messages, response_format="json_only",
                        llm_temperature=TEMPERATURE, llm_max_tokens=args.max_new_tokens,
                        adapter_path=str(args.adapter), protocol=PROTOCOL, metric=METRIC),
                        sort_keys=True, default=str) + "\n")
            print(json.dumps({"group": group_id, "samples": len(records), "status": group_status,
                              "members": members}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("select")
    p.add_argument("--corpus", type=Path, default=CORPUS)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("collect")
    p.add_argument("--corpus", type=Path, default=CORPUS)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True, help="Init policy: the TaskSet SFT+ORPO adapter")
    p.add_argument("--bank", type=Path, default=ROOT / "history_data/domain_taskset/experience_bank_train838.jsonl")
    p.add_argument("--rgpe-assets", type=Path, default=Path("/root/data/logs/taskset_rgpe_test210_20261003/assets"))
    p.add_argument("--baseline-cache", type=Path,
                   default=Path("/root/data/logs/taskset_rgpe_test210_20261003/task_baselines.json"))
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (select if args.action == "select" else collect)(args)
