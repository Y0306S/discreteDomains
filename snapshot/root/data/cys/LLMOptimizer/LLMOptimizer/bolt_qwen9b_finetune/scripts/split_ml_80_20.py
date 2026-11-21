#!/usr/bin/env python3
"""Split ML data into 80/20 train/test with stratification by task type."""

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

# Load all original ML data
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

# For each task type, split 80/20
train_records = []
test_records = []

for task_type, records in sorted(by_type.items()):
    n = len(records)
    n_test = max(1, int(round(n * 0.20)))
    n_train = n - n_test
    
    random.shuffle(records)
    
    test_records.extend(records[:n_test])
    train_records.extend(records[n_test:])
    
    print(f"  {task_type}: total={n}, train={n_train}, test={n_test}")

# Shuffle final splits
random.shuffle(train_records)
random.shuffle(test_records)

print(f"\nFinal split:")
print(f"  Train: {len(train_records)} ({len(train_records)/len(all_records)*100:.1f}%)")
print(f"  Test:  {len(test_records)} ({len(test_records)/len(all_records)*100:.1f}%)")

# Verify
for name, records in [("train", train_records), ("test", test_records)]:
    cls_count = sum(1 for r in records if get_task_type(r) == 'cls')
    rgs_count = sum(1 for r in records if get_task_type(r) == 'rgs')
    print(f"\n{name}:")
    print(f"  Classification: {cls_count} ({cls_count/len(records)*100:.1f}%)")
    print(f"  Regression: {rgs_count} ({rgs_count/len(records)*100:.1f}%)")

# Save
save_jsonl(BASE / "ml_train_80.jsonl", train_records)
save_jsonl(BASE / "ml_test_20.jsonl", test_records)
print(f"\nSaved to:")
print(f"  {BASE / 'ml_train_80.jsonl'}")
print(f"  {BASE / 'ml_test_20.jsonl'}")
