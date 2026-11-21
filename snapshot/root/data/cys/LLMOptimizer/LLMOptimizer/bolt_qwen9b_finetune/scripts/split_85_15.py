#!/usr/bin/env python3
"""Re-split DL data into 85/15 train/test with stratification."""

import json
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

BASE = Path("/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/data")
OUTPUT = Path("/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/data")

def load_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]

def save_jsonl(path, records):
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def get_strata(record):
    """Extract stratification keys from a record."""
    user = json.loads(record["messages"][1]["content"])
    task = user["task"]
    assistant = json.loads(record["messages"][2]["content"])
    
    task_family = user["task_family"]
    optimizer_family = assistant.get("optimizer_family", "unknown")
    core_type = task.get("core", ["unknown"])[0] if isinstance(task.get("core"), list) else "unknown"
    
    return (task_family, optimizer_family, core_type)

# Load all data
train = load_jsonl(BASE / "dl_train.jsonl")
val = load_jsonl(BASE / "dl_validation.jsonl")
test = load_jsonl(BASE / "dl_test.jsonl")

all_records = train + val + test
print(f"Total records: {len(all_records)}")

# Group by strata
strata_groups = defaultdict(list)
for r in all_records:
    strata = get_strata(r)
    strata_groups[strata].append(r)

print(f"Unique strata: {len(strata_groups)}")

# For each stratum, split 85/15
train_records = []
test_records = []

for strata, records in sorted(strata_groups.items()):
    n = len(records)
    n_test = max(1, int(round(n * 0.15)))
    n_train = n - n_test
    
    # Shuffle within stratum
    random.shuffle(records)
    
    test_records.extend(records[:n_test])
    train_records.extend(records[n_test:])
    
    print(f"  {strata}: total={n}, train={n_train}, test={n_test}")

# Shuffle final splits
random.shuffle(train_records)
random.shuffle(test_records)

print(f"\nFinal split:")
print(f"  Train: {len(train_records)} ({len(train_records)/len(all_records)*100:.1f}%)")
print(f"  Test:  {len(test_records)} ({len(test_records)/len(all_records)*100:.1f}%)")

# Verify stratification
for name, records in [("train", train_records), ("test", test_records)]:
    families = defaultdict(int)
    opt_families = defaultdict(int)
    for r in records:
        s = get_strata(r)
        families[s[0]] += 1
        opt_families[s[1]] += 1
    print(f"\n{name} task families: {dict(families)}")
    print(f"{name} optimizer families: {dict(opt_families)}")

# Save
save_jsonl(OUTPUT / "dl_train_85.jsonl", train_records)
save_jsonl(OUTPUT / "dl_test_15.jsonl", test_records)
print(f"\nSaved to:")
print(f"  {OUTPUT / 'dl_train_85.jsonl'}")
print(f"  {OUTPUT / 'dl_test_15.jsonl'}")
