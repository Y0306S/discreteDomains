"""Load TaskSet GRPO rollouts (taskset_rl_rollout.v1) for train_p5_reinforce.py.

Returns records in the shape ``load_records_v14`` returns, so the ML v14 trainer can
train TaskSet rollouts unchanged apart from selecting this loader:

    elif args.rollout_schema == "taskset":
        from taskset_reinforce_records import load_records_taskset
        records, load_stats = load_records_taskset(args.buffer_root)

Checks: schema, train838-only tasks (never test210), unique (group, sample),
json_only completions that parse into the TaskSet CASH space, finite rewards; only
marked train members with a trainable status are kept, and groups need >= 2 members
with different rewards.
"""

import json
import math
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mindware.domains.dlhpo.taskset_llm import spec as S
from mindware.domains.dlhpo.taskset_space import space_from_taskset
from taskset_transfer_common import read_split

SCHEMA = "taskset_rl_rollout.v1"
TRAINABLE = ("SUCCESS", "DUPLICATE", "PARSE_FAILURE")


def load_records_taskset(buffer_root):
    split, cs = read_split(), space_from_taskset()
    train, test = set(split["train"]), set(split["test"])
    files = sorted(Path(buffer_root).rglob("rollouts_*.jsonl"))
    if not files:
        raise SystemExit(f"No rollouts_*.jsonl under {buffer_root}")
    rows, seen, stats = [], set(), defaultdict(int)
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            stats["rows_read"] += 1
            ident = (row.get("group_id"), row.get("sample_i"))
            if row.get("schema") != SCHEMA:
                raise SystemExit(f"{path}: non-{SCHEMA} row {ident}")
            if row["task_id"] not in train or row["task_id"] in test:
                raise SystemExit(f"{path}: task {row['task_id']} violates the train838/test210 split")
            if ident in seen:
                raise SystemExit(f"{path}: duplicate rollout {ident}")
            seen.add(ident)
            if row.get("response_format") != "json_only":
                raise SystemExit(f"{path}: unexpected response format {row.get('response_format')} {ident}")
            stats[f"status_{row['evaluation_status']}"] += 1
            if not row.get("train_member") or row.get("excluded_from_training"):
                continue
            if row["evaluation_status"] not in TRAINABLE:
                raise SystemExit(f"{path}: train member with untrainable status {row['evaluation_status']} {ident}")
            completion = row.get("completion")
            if not isinstance(completion, str) or not completion.strip():
                raise SystemExit(f"{path}: train member without model text {ident}")
            if row["evaluation_status"] != "PARSE_FAILURE":
                S.parse_reply(completion, cs)
            reward = float(row["reward"]["reward"])
            if not math.isfinite(reward):
                raise SystemExit(f"{path}: nonfinite reward {ident}")
            rows.append({"messages": row["messages"], "completion": completion, "reward": reward,
                         "dataset": row["task_id"], "task_type": row["task_type"], "trial": row.get("prompt_trial"),
                         "group_id": row["group_id"], "status": row["evaluation_status"],
                         "generation_limit": row.get("llm_max_tokens")})
    groups = defaultdict(list)
    for i, rec in enumerate(rows):
        groups[rec["group_id"]].append(i)
    keep = []
    for idxs in groups.values():
        if len(idxs) < 2 or len({round(rows[i]["reward"], 12) for i in idxs}) < 2:
            stats["groups_dropped_no_signal"] += 1
            continue
        keep.extend(idxs)
    out = [rows[i] for i in sorted(keep)]
    stats["groups_kept"] = len(groups) - stats["groups_dropped_no_signal"]
    stats["records_kept"] = len(out)
    return out, dict(stats)
