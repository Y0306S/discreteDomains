#!/usr/bin/env python3
"""Merge all ML data (train + val + test) into one training file for SFT.
Stratified by task type (classification vs regression)."""

import json
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

BASE = Path("/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/data")

def load_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]

def save_jsonl(path, records):
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def get_task_type(record):
    return record["metadata"]["task_type"]  # 'cls' or 'rgs'

# Load all data
train = load_jsonl(BASE / "train.jsonl")
val = load_jsonl(BASE / "validation.jsonl")
test = load_jsonl(BASE / "test.jsonl")

all_records = train + val + test
print(f"Total ML records: {len(all_records)}")

# Group by task type
by_type = defaultdict(list)
for r in all_records:
    by_type[get_task_type(r)].append(r)

print(f"Classification (cls): {len(by_type['cls'])}")
print(f"Regression (rgs): {len(by_type['rgs'])}")

# Shuffle each group
for k in by_type:
    random.shuffle(by_type[k])

# Interleave: cls, rgs, cls, rgs, ... to ensure balanced distribution
merged = []
cls_list = by_type['cls']
rgs_list = by_type['rgs']

# Interleave with periodic balancing
i_cls = i_rgs = 0
while i_cls < len(cls_list) or i_rgs < len(rgs_list):
    # Add 2 classification then 1 regression (since cls:rgs ratio is ~2:1)
    for _ in range(2):
        if i_cls < len(cls_list):
            merged.append(cls_list[i_cls])
            i_cls += 1
    if i_rgs < len(rgs_list):
        merged.append(rgs_list[i_rgs])
        i_rgs += 1

print(f"Merged total: {len(merged)}")

# Verify distribution
cls_count = sum(1 for r in merged if get_task_type(r) == 'cls')
rgs_count = sum(1 for r in merged if get_task_type(r) == 'rgs')
print(f"  Classification: {cls_count} ({cls_count/len(merged)*100:.1f}%)")
print(f"  Regression: {rgs_count} ({rgs_count/len(merged)*100:.1f}%)")

# Save
save_jsonl(BASE / "ml_train_all.jsonl", merged)
print(f"Saved to: {BASE / 'ml_train_all.jsonl'}")
