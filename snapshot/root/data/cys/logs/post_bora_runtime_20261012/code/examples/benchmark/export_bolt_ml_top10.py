#!/usr/bin/env python3
"""Export BOLT-ML targets from the fixed train80 archive, without oracle calls."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ConfigSpace import Configuration
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

from examples.benchmark.benchmark_utils import class_distribution, regression_target_stats
from mindware.components.optimizers.transfer_hpo.protocol import (
    HISTORY_ROOT, allowed_task_ids, checked_source_path, sha256_file,
)
from mindware.components.optimizers.transfer_hpo.schema import (
    build_hpo_cs, configspace_spec, schema_fingerprint, schema_spec,
)
from mindware.utils.data_manager import default_missing_values

TRAIN80 = Path('/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/ml_train_80_fixed.jsonl')
DEFAULT_OUTPUT = REPO / 'benchmark_data/bolt_ml_top10_train80_v1'
K = 10
SYSTEM = 'You recommend hyperparameter configurations for tabular machine learning. Return only a JSON object.'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def select_top_k(observations, cs, algorithm, task_type, k=K):
    """Rank distinct valid successes by minimum objective; ties use archive order."""
    candidates, exclusions = [], []
    for index, obs in enumerate(observations):
        reason = None
        if type(obs.get('trial_state')) is not int or obs['trial_state'] != 0:
            reason = 'unsuccessful_trial'
        else:
            objectives = obs.get('objectives')
            if not isinstance(objectives, list) or len(objectives) != 1:
                reason = 'invalid_objective'
            else:
                try:
                    score = float(objectives[0])
                    if not math.isfinite(score) or abs(score) >= 1e18:
                        raise ValueError('Non-finite/sentinel objective')
                    if (task_type == 'cls' and not -1 <= score <= 0) or (task_type == 'rgs' and score < 0):
                        raise ValueError('Objective outside metric domain')
                except (TypeError, ValueError):
                    reason = 'invalid_objective'
        if reason is None:
            try:
                raw = obs['config']
                if not isinstance(raw, dict):
                    raise ValueError('Configuration must be an object')
                values = dict(raw)
                values.setdefault('algorithm', algorithm)
                if values['algorithm'] != algorithm:
                    raise ValueError('Wrong fixed algorithm')
                cfg = Configuration(cs, values=values)
                cs.check_configuration(cfg)
                # Do not change a scored configuration or default missing fields.
                if canonical(dict(cfg)) != canonical(values):
                    raise ValueError('ConfigSpace changed the recorded values')
                candidates.append({'observation_index': index, 'objective_minimize': score,
                                   'config': values, 'config_sha256': digest(values)})
            except (KeyError, TypeError, ValueError) as exc:
                reason = 'invalid_configuration'
                exclusions.append({'observation_index': index, 'reason': reason, 'detail': str(exc)})
                continue
        if reason:
            exclusions.append({'observation_index': index, 'reason': reason})

    candidates.sort(key=lambda row: (row['objective_minimize'], row['observation_index']))
    distinct, seen = [], set()
    for row in candidates:
        key = row['config_sha256']
        if key in seen:
            exclusions.append({'observation_index': row['observation_index'], 'reason': 'duplicate_configuration'})
        else:
            seen.add(key)
            distinct.append(row)
    for row in distinct[k:]:
        exclusions.append({'observation_index': row['observation_index'], 'reason': 'outside_top_k'})
    return distinct[:k], exclusions, len(distinct)


def dataset_context(task_type, dataset, split_seed=1, test_size=0.25):
    """Read profile/labels only; match the benchmark's CSV and outer split rules."""
    base = REPO / 'datasets_bundle'
    path = base / f'{task_type}_datasets_downsampled/{dataset}_downsampled.csv'
    if not path.is_file():
        path = base / f'{task_type}_datasets/{dataset}.csv'
    path = path.resolve(strict=True)
    path.relative_to(base.resolve())
    columns = pd.read_csv(path, nrows=0).columns
    frame = pd.read_csv(path, usecols=[len(columns) - 1], na_values=default_missing_values)
    y = frame.iloc[:, 0]
    profile = {'dataset': dataset, 'task_type': 'classification' if task_type == 'cls' else 'regression',
               'n_samples': len(y), 'n_features': len(columns) - 1}
    args = {}
    if task_type == 'cls':
        profile['class_distribution'] = class_distribution(y)
    else:
        profile['target_stats'] = regression_target_stats(y)
        train_indices, _ = next(ShuffleSplit(n_splits=1, test_size=test_size,
                                            random_state=split_seed).split(frame))
        train_y = pd.to_numeric(y.iloc[train_indices].dropna(), errors='raise')
        if train_y.empty or not np.isfinite(train_y).all():
            raise ValueError(f'Invalid regression labels: {path}')
        args['y_neg_mask'] = bool((train_y >= 0).all())
    provenance = {'path': str(path), 'sha256': sha256_file(path), 'label_column': str(columns[-1]),
                  'outer_split_seed': split_seed, 'outer_test_size': test_size,
                  'schema_kwargs': args}
    return profile, args, provenance


def build_messages(profile, algorithm, task_type, cs, config):
    spec = configspace_spec(cs)
    prompt_space = {'hyperparameters': schema_spec(cs), 'conditions': spec['conditions'],
                    'forbiddens': spec['forbiddens']}
    metric = ('Maximize validation balanced accuracy (the stored objective is its negative).'
              if task_type == 'cls' else 'Minimize validation mean squared error.')
    prompt = (
        'Recommend one hyperparameter configuration for the following ML task.\n'
        f'Dataset profile: {canonical(profile)}\nFixed algorithm: {algorithm}\nObjective: {metric}\n'
        'ConfigSpace search space:\n' + json.dumps(prompt_space, indent=2, allow_nan=False) + '\n'
        'Return one JSON object using the exact hyperparameter names and JSON value types above. '
        'Include all active hyperparameters and constants, omit inactive hyperparameters, and obey '
        'the bounds, choices, quantization, conditions and forbidden combinations. '
        'Do not include explanations, scores or markdown.'
    )
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': canonical(config)}]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_jsonl(path, rows):
    with path.open('w') as stream:
        for row in rows:
            stream.write(canonical(row) + '\n')


def export(out_dir):
    if os.environ.get('ML_TRANSFER_HISTORY_MIRROR'):
        raise ValueError('BOLT export requires the original authorized archive, not a mirror')
    if out_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing export: {out_dir}')
    tasks = sorted(allowed_task_ids(TRAIN80))
    examples, provenance, exclusions, task_records = [], [], [], []
    contexts, schemas = {}, {}
    raw_count = 0
    for task_id in tasks:
        task_type, dataset, algorithm = task_id.split('/')
        metric = 'bal_acc' if task_type == 'cls' else 'mse'
        source = checked_source_path(HISTORY_ROOT / task_type / metric / dataset / f'{algorithm}.json')
        payload = json.loads(source.read_text())
        observations = payload['observations']
        if not isinstance(observations, list):
            raise ValueError(f'Malformed history: {source}')
        raw_count += len(observations)
        key = (task_type, dataset)
        if key not in contexts:
            contexts[key] = dataset_context(*key)
        profile, cs_args, context_source = contexts[key]
        cs = build_hpo_cs(task_type, algorithm, **cs_args)
        fingerprint = schema_fingerprint(cs)
        schemas[fingerprint] = configspace_spec(cs)
        selected, rejected, n_distinct = select_top_k(observations, cs, algorithm, task_type)
        source_hash = sha256_file(source)
        task_records.append({'task_id': task_id, 'source_path': str(source), 'source_sha256': source_hash,
                             'observations': len(observations), 'valid_distinct': n_distinct,
                             'selected': len(selected), 'schema_fingerprint': fingerprint,
                             'context_source': context_source})
        exclusions.extend({'task_id': task_id, **row} for row in rejected)
        for rank, row in enumerate(selected, 1):
            example = {'messages': build_messages(profile, algorithm, task_type, cs, row['config'])}
            provenance.append({'example_index': len(examples), 'example_sha256': digest(example),
                               'task_id': task_id, 'rank': rank, 'source_path': str(source),
                               'source_sha256': source_hash, 'schema_fingerprint': fingerprint, **row})
            examples.append(example)
        if len(task_records) % 50 == 0:
            print(f'Checked {len(task_records)}/{len(tasks)} tasks; selected {len(examples)} examples', flush=True)

    if not examples:
        raise ValueError('No eligible training examples')
    out_dir.mkdir(parents=True)
    write_jsonl(out_dir / 'train.jsonl', examples)
    write_jsonl(out_dir / 'provenance.jsonl', provenance)
    write_jsonl(out_dir / 'exclusions.jsonl', exclusions)
    write_json(out_dir / 'schemas.json', schemas)
    write_json(out_dir / 'tasks.json', task_records)
    write_json(out_dir / 'example.json', {'provenance': provenance[0], **examples[0]})
    write_json(out_dir / 'dataset_info.json', {'bolt_ml_top10_train80': {
        'file_name': 'train.jsonl', 'formatting': 'sharegpt', 'columns': {'messages': 'messages'},
        'tags': {'role_tag': 'role', 'content_tag': 'content', 'user_tag': 'user',
                 'assistant_tag': 'assistant', 'system_tag': 'system'},
    }})
    manifest = {
        'format_version': 1, 'method': 'BOLT-ML', 'k': K, 'train_manifest': str(TRAIN80),
        'train_manifest_sha256': sha256_file(TRAIN80), 'history_root': str(HISTORY_ROOT),
        'allowed_tasks': len(tasks), 'represented_tasks': sum(t['selected'] > 0 for t in task_records),
        'dataset_contexts': len(contexts), 'source_observations': raw_count, 'examples': len(examples),
        'tasks_with_fewer_than_k': [t['task_id'] for t in task_records if t['selected'] < K],
        'exclusions': dict(Counter(row['reason'] for row in exclusions)),
        'group_examples': dict(sorted(Counter('/'.join(row['task_id'].split('/')[::2]) for row in provenance).items())),
        'selection': 'Lowest objective per distinct valid successful configuration; top 10 per train80 task',
        'schema_policy': 'Current ConfigSpace; no repairs or missing-field defaults to historical labels',
        'objective_direction': {'cls': 'minimize negative balanced accuracy', 'rgs': 'minimize MSE'},
        'files': {p.name: sha256_file(p) for p in sorted(out_dir.iterdir()) if p.is_file()},
        'exporter_sha256': sha256_file(Path(__file__)),
    }
    write_json(out_dir / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(export(args.out_dir.resolve()), indent=2))


if __name__ == '__main__':
    main()
