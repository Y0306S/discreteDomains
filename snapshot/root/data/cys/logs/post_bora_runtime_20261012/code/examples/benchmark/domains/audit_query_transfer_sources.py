#!/usr/bin/env python3
"""Audit the permitted CEB train800 corpus without fitting or selecting a policy."""

import argparse
import ast
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import re

from run_query_ceb_baselines import (
    QP_ROOT, TRAIN_FILE, TEST_FILE, atomic_json, fingerprint, load_split, sql_path,
)
from query_transfer_common import HISTORICAL_VAE_FILE, VAE_FILE, validate_assets


def audit_task(task, history_root):
    path = history_root / f'optimimze-db_nowandb_{task}_all-data-collected.csv'
    if not path.is_file() or not (history_root / (task + '.done')).is_file():
        raise ValueError(f'Missing completed source history: {task}')
    if not path.resolve().is_relative_to(history_root.resolve()):
        raise ValueError(f'Source history escaped the permitted directory: {task}')
    before = fingerprint(path)
    counts = Counter()
    unique_successes = {}
    repeated_outcome_changes = 0
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['train_x', 'train_y', 'censoring']:
            raise ValueError(f'Unexpected history schema: {path}')
        for index, row in enumerate(reader):
            counts['rows'] += 1
            score, censored = float(row['train_y']), float(row['censoring'])
            if not math.isfinite(score) or score >= 0 or censored not in (0, 1):
                raise ValueError(f'Invalid runtime/censoring in {path}, row {index}')
            if censored:
                counts['censored_rows'] += 1
                continue
            plan = ast.literal_eval(row['train_x'])
            if (not isinstance(plan, list) or not plan or len(plan) % 5
                    or any(type(value) is not int for value in plan)):
                raise ValueError(f'Invalid uncensored encoded plan in {path}, row {index}')
            counts['uncensored_rows'] += 1
            counts['uncensored_initial_rows' if index < 49 else 'uncensored_new_rows'] += 1
            key = tuple(plan)
            if key in unique_successes:
                counts['duplicate_uncensored_rows'] += 1
                if unique_successes[key]['runtime_seconds'] != -score:
                    repeated_outcome_changes += 1
            else:
                unique_successes[key] = {'source_row': index, 'runtime_seconds': -score}
    after = fingerprint(path)
    if before != after:
        raise ValueError(f'Source history changed during audit: {path}')
    successes = list(unique_successes.values())
    unique_count = len(successes)
    varying = len({point['runtime_seconds'] for point in successes}) > 1
    return {
        'task_id': task, 'family': re.match(r'CEB_(\d+[A-Z])', task)[1],
        'source_path': str(path), 'source_sha256': after,
        'counts': dict(counts), 'unique_uncensored_plans': unique_count,
        'distinct_uncensored_outcomes': len({point['runtime_seconds'] for point in successes}),
        'duplicate_plan_outcome_changes': repeated_outcome_changes,
        'candidate_uncensored_policy_eligible': unique_count >= 2 and varying,
        'candidate_exclusion_reason': None if unique_count >= 2 and varying else (
            'fewer_than_two_unique_uncensored_plans' if unique_count < 2 else 'constant_outcome'),
        'first_uncensored_row_per_distinct_plan': successes,
        'original_latent_vectors_available': False,
        'per_proposal_rounds_and_cache_events_available': False,
    }


def audit():
    validate_assets()
    train, test = load_split()
    history_root = QP_ROOT / 'lolbo_scripts/optimization_all_collected_data'
    records = []
    for task in train:
        sql_path(QP_ROOT, task)
        records.append(audit_task(task, history_root))
    totals = Counter()
    families = defaultdict(Counter)
    for record in records:
        totals.update(record['counts'])
        totals['unique_uncensored_plans'] += record['unique_uncensored_plans']
        eligible = record['candidate_uncensored_policy_eligible']
        key = 'eligible' if eligible else 'insufficient'
        families[record['family']][key] += 1
        totals[key + '_tasks'] += 1
        if eligible:
            totals['eligible_unique_uncensored_plans'] += record['unique_uncensored_plans']
    return {
        'format_version': 1, 'domain': 'ceb_query_runtime', 'status': 'source_audit_complete',
        'training_started': False, 'training_data_policy_selected': False,
        'train_split': str(TRAIN_FILE), 'train_split_sha256': fingerprint(TRAIN_FILE),
        'test_split': str(TEST_FILE), 'test_split_sha256': fingerprint(TEST_FILE),
        'training_tasks': train, 'test_tasks_excluded_from_sources': test,
        'permitted_history_root': str(history_root),
        'checkpoint': str(VAE_FILE),
        'checkpoint_sha256': fingerprint(VAE_FILE),
        'historical_checkpoint': str(HISTORICAL_VAE_FILE),
        'historical_checkpoint_sha256': fingerprint(HISTORICAL_VAE_FILE),
        'candidate_policy': 'uncensored, first successful row per distinct plan, at least two plans',
        'summary': dict(totals), 'families': {k: dict(v) for k, v in sorted(families.items())},
        'max_unique_uncensored_plans_per_task': max(r['unique_uncensored_plans'] for r in records),
        'warnings': [
            'Censored rows combine timeouts and execution errors; exact runtimes are unknown.',
            'Only encoded plans were saved; original proposal latents cannot be recovered.',
            'The feature encoder is the author VAE; histories used the older local VAE.',
            'The historical initializer could replace failed encodings with random latent vectors.',
            'CSV order preserves retained rows, not a complete proposal or acquisition-batch trace.',
            'Repeated successful plans can have different runtimes; first retained row is an audit convention.',
        ],
        'tasks': records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path to preserve the previous audit')
    payload = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)
    print(json.dumps({'audit': str(args.output.resolve()), **payload['summary'],
                      'max_unique_uncensored_plans_per_task': payload['max_unique_uncensored_plans_per_task'],
                      'training_started': False}, indent=2))


if __name__ == '__main__':
    main()
