#!/usr/bin/env python3
"""Export real train838 histories for FSBO/RGPE; never run an evaluator."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from ConfigSpace import Configuration
from ConfigSpace.read_and_write import json as cs_json
from sklearn.model_selection import train_test_split
from mindware.domains.dlhpo.taskset_lolbo import TaskSetCodec
from mindware.domains.dlhpo.taskset_space import load_family_spaces, space_from_taskset
from mindware.domains.search import _optuna_distributions
from taskset_transfer_common import (
    CAMPAIGN, METRIC, PROTOCOL, SPLIT_SHA256, atomic_json, digest,
    history_bytes, load_corpus, read_split, run_lock,
)


def prepare(output, component_tasks=None):
    split = read_split()
    ids = sorted(component_tasks or split['train'])
    if len(ids) != len(set(ids)) or not set(ids) <= set(split['train']):
        raise ValueError('Requested sources must be unique train838 tasks')
    purpose = 'component_check' if component_tasks else 'production_training'
    with run_lock(output):
        if (output / 'manifest.json').exists():
            manifest, _ = load_corpus(output)
            if manifest['source_tasks'] != ids or manifest['purpose'] != purpose:
                raise ValueError('Existing corpus differs from requested export')
            return manifest
        specs = CAMPAIGN / 'taskset_data/scripts/optimizer_search_spaces.json'
        cs = space_from_taskset(load_family_spaces(specs))
        codec = TaskSetCodec(cs, _optuna_distributions(cs))
        records, features, offsets = [], [], [0]
        failed_count = 0
        for task in ids:
            raw = history_bytes(task)
            history = json.loads(raw)
            meta = history['meta_info']
            observations = history['observations']
            if (history['task_id'] != task or meta['domain'] != 'taskset'
                    or meta['optimizer'] != 'tpe' or meta['seed'] != 1 or len(observations) != 100):
                raise ValueError('Source is not a complete 100-evaluation TaskSet TPE run: ' + task)
            record = dict(task_id=task, source_sha256=hashlib.sha256(raw).hexdigest(),
                          configs=[], y_minimize=[], source_rows=[], excluded_rows=[])
            for index, observation in enumerate(observations, 1):
                config = Configuration(cs, values=observation['config'])
                cs.check_configuration(config)
                info = observation.get('extra_info', {}).get('info', {})
                objective = float(observation['objectives'][0])
                if observation['trial_state'] != 0 or not np.isfinite(objective):
                    record['excluded_rows'].append(index)
                    failed_count += 1
                    continue
                if info.get('oracle') != 'train_inner' or info.get('protocol') != PROTOCOL or info.get('objective_metric') != METRIC:
                    raise ValueError('Successful source has a different evaluation protocol: ' + task)
                values = dict(config.get_dictionary())
                vector = codec.encode(values)
                if not np.allclose(codec.encode(codec.decode(vector)), vector, atol=1e-7):
                    raise ValueError('TaskSet feature round-trip failed')
                record['configs'].append(values)
                record['y_minimize'].append(objective)
                record['source_rows'].append(index)
                features.append(vector)
            if len(record['configs']) < 2:
                raise ValueError('At least two successful observations are required: ' + task)
            records.append(record)
            offsets.append(len(features))
        if component_tasks:
            if len(ids) < 2:
                raise ValueError('Component corpus needs train and validation tasks')
            train, valid = ids[:-1], ids[-1:]
        else:
            train, valid = train_test_split(ids, test_size=0.2, random_state=123,
                                           stratify=[task.rsplit('_seed', 1)[0] for task in ids])
        atomic_json(output / 'tasks.json', records)
        schema = json.loads(cs_json.write(cs))
        atomic_json(output / 'configspace.json', schema)
        np.savez_compressed(str(output / 'features.npz'), X_fsbo=np.asarray(features, dtype=np.float64),
                            offsets=np.asarray(offsets, dtype=np.int64))
        manifest = dict(domain='taskset_cash', format_version=1, status='complete', purpose=purpose,
                        source_tasks=ids, split_sha256=SPLIT_SHA256, protocol=PROTOCOL, objective_metric=METRIC,
                        source_policy='successful_real_tpe_observations', observed_trials=len(ids) * 100,
                        observations=len(features), excluded_failed_observations=failed_count,
                        fsbo_dim=codec.dim, fsbo_encoding='TaskSetCodec_onehot_conditional_v1',
                        fsbo_train_tasks=sorted(train), fsbo_validation_tasks=sorted(valid),
                        validation_policy='20_percent_task_family_stratified_seed123' if not component_tasks else 'component_only',
                        search_space_sha256=digest(specs), exporter_sha256=digest(__file__),
                        files={name: digest(output / name) for name in ('tasks.json', 'features.npz', 'configspace.json')})
        atomic_json(output / 'manifest.json', manifest)
        load_corpus(output)
        return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--component-tasks', nargs='+')
    args = parser.parse_args()
    result = prepare(args.output, args.component_tasks)
    print(json.dumps({key: result[key] for key in ('purpose', 'observations', 'excluded_failed_observations', 'fsbo_dim')}))
