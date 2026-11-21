#!/usr/bin/env python3
"""Prepare BOLT TaskSet top-10 SFT examples from train838, without oracle calls."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ConfigSpace import Configuration
from mindware.components.optimizers.transfer_hpo.schema import configspace_spec, cs_from_spec, schema_fingerprint, schema_spec
from mindware.domains.dlhpo.taskset_llm.spec import load_task_specs
from mindware.domains.dlhpo.taskset_space import load_family_spaces, space_from_taskset
from taskset_transfer_common import CAMPAIGN, HISTORIES, METRIC, PROTOCOL, SPLIT_SHA256, digest, history_bytes, load_corpus, read_split

CORPUS = Path('/root/data/logs/taskset_transfer_train838_20261002/corpus')
OUTPUT = ROOT / 'benchmark_data/bolt_taskset_top10_train838_v4'
ML_EXPORTER = Path('/root/data/cys/blockdata_archive/llmoptimizer_benchmark/examples/benchmark/export_bolt_ml_top10.py')
NESTEROV_HP = 'nadamw_grid:use_nesterov'
LEGACY_NADAMW = CAMPAIGN / 'runtime_template/task_set/optimizers/nadamw.py'
# This audited factory omits use_nesterov; its optimizer defaults to false.
LEGACY_NADAMW_SHA256 = '3be1e62ad74a73f678b14eb3df70c33669c4d4c3b6e81402d8edddf7225f343e'
K = 10
SYSTEM = (
    'You are an automated hyperparameter tuning agent for TaskSet training and synthetic optimization tasks. '
    'Recommend an optimizer family and its hyperparameters to minimize normalized validation loss. '
    'Return only a JSON object.'
)
SCHEDULE_HORIZON = 10000
PARAMETER_NOTES = [
    '- A condition with type EQ means equality: child is active only when parent equals value. '
    'For example, nadamw_grid:* fields are active only when optimizer_family equals nadamw_grid.',
    '- Use numeric JSON values for UniformFloatHyperparameter, literal JSON choices for '
    'CategoricalHyperparameter, and the fixed value for Constant. Booleans are true/false, not strings.',
    '- lower and upper are inclusive bounds. log=true describes the sampling scale: output the '
    'actual positive value, not its logarithm. q=null means no quantization; default_value is a '
    'valid fallback, not a required recommendation.',
    '- <name>_inv stores 1 - <name> (for example beta1 = 1 - beta1_inv). '
    'Return the encoded _inv key, not an additional decoded key.',
    '- <name>_is_zero=true overrides the decoded value to 0; <name>_is_one=true overrides it to 1. '
    'The paired numeric field is still active and must be included with a value inside its own bounds. '
    'When the flag is false, the numeric field supplies the value.',
    '- The nadamw_grid fractions refer to a 10000-step learning-rate schedule. '
    'Warmup lasts min(warmup_fraction, constant_fraction) * 10000 steps; '
    'cosine decay starts at constant_fraction * 10000. Only the first 1000 steps are evaluated.',
]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def value_digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def effective_space(source_cs):
    """Keep the recorded space, but allow only the effective Nesterov value."""
    spec = configspace_spec(source_cs)
    for key, default_key in (('hyperparameters', 'default'), ('transfer_hyperparameters', 'default_value')):
        hp = next(row for row in spec[key] if row['name'] == NESTEROV_HP)
        hp['choices'] = [False]
        hp[default_key] = False
    return cs_from_spec(spec)


def historical_effective_config(values):
    result = dict(values)
    if result.get('optimizer_family') == 'nadamw_grid':
        result[NESTEROV_HP] = False
    return result


def select_top_k(observations, cs, k=K, *, effective_cs=None):
    """Match BOLT-ML: best outcome per distinct config, stable ties, then top k."""
    candidates, excluded = [], []
    for index, observation in enumerate(observations):
        reason = None
        if type(observation.get('trial_state')) is not int or observation['trial_state'] != 0:
            reason = 'unsuccessful_trial'
        else:
            try:
                objectives = observation['objectives']
                if not isinstance(objectives, list) or len(objectives) != 1 or isinstance(objectives[0], bool):
                    raise ValueError('Expected one numeric objective')
                score = float(objectives[0])
                if not math.isfinite(score) or abs(score) >= 1e18:
                    raise ValueError('Non-finite/sentinel objective')
            except (KeyError, TypeError, ValueError):
                reason = 'invalid_objective'
        if reason is None:
            try:
                values = observation['config']
                if not isinstance(values, dict):
                    raise ValueError('Configuration must be an object')
                configuration = Configuration(cs, values=values)
                cs.check_configuration(configuration)
                if canonical(dict(configuration)) != canonical(values):
                    raise ValueError('ConfigSpace changed the recorded values')
                provenance = {}
                if effective_cs is not None:
                    original = dict(values)
                    values = historical_effective_config(original)
                    effective = Configuration(effective_cs, values=values)
                    effective_cs.check_configuration(effective)
                    if canonical(dict(effective)) != canonical(values):
                        raise ValueError('ConfigSpace changed the effective values')
                    adjustments = []
                    if original != values:
                        adjustments.append(dict(parameter=NESTEROV_HP, recorded_value=original[NESTEROV_HP],
                                                effective_value=False, reason='legacy_factory_ignored_use_nesterov'))
                    provenance = dict(original_config=original, original_config_sha256=value_digest(original),
                                      configuration_adjustments=adjustments)
                candidates.append(dict(observation_index=index, objective_minimize=score,
                                       config=values, config_sha256=value_digest(values), **provenance))
            except (KeyError, TypeError, ValueError) as exc:
                excluded.append(dict(observation_index=index, reason='invalid_configuration', detail=str(exc)))
                continue
        if reason:
            excluded.append(dict(observation_index=index, reason=reason))
    candidates.sort(key=lambda row: (row['objective_minimize'], row['observation_index']))
    distinct, seen = [], set()
    for row in candidates:
        if row['config_sha256'] in seen:
            excluded.append(dict(observation_index=row['observation_index'], reason='duplicate_configuration'))
        else:
            seen.add(row['config_sha256'])
            distinct.append(row)
    excluded.extend(dict(observation_index=row['observation_index'], reason='outside_top_k') for row in distinct[k:])
    return distinct[:k], excluded, len(distinct)


def build_messages(task_id, task, cs, config):
    """The same context produces ten independent single-configuration targets."""
    spec = configspace_spec(cs)
    space = dict(hyperparameters=schema_spec(cs), conditions=spec['conditions'], forbiddens=spec['forbiddens'])
    profile = dict(task_id=task_id, task_family=task['task_family'], task_spec=task['spec'])
    prompt = (
        'Recommend one optimizer configuration for the following TaskSet optimization task.\n'
        f'Task profile: {canonical(profile)}\n'
        'The task profile is fixed context, not part of the optimizer configuration. '
        'It may describe a neural-network task or a synthetic optimization problem.\n'
        'Optimization: Choose optimizer_family and the hyperparameters of that family (CASH).\n'
        'Nesterov momentum is disabled: when optimizer_family is nadamw_grid, '
        'include nadamw_grid:use_nesterov with the JSON boolean value false.\n'
        'Evaluation budget: 1000 training steps; 1 replica. '
        'VALID_OUTER is measured at initialization (step 0) and step 1000.\n'
        f'Learning-rate schedule horizon: {SCHEDULE_HORIZON} steps for nadamw_grid; '
        'the evaluation stops after the first 1000 steps without rescaling that schedule.\n'
        f'Objective: Minimize normalized validation loss ({METRIC}); lower is better. '
        'The score is the minimum normalized VALID_OUTER loss across the recorded measurements. '
        'Normalization uses the fixed task reference loss and initialization loss, clipped to [0, 1]; '
        '1 means no improvement over initialization, and 0 means reaching or beating the reference.\n'
        'ConfigSpace search space:\n' + json.dumps(space, indent=2, allow_nan=False) + '\n'
        'Parameter encoding notes:\n' + '\n'.join(PARAMETER_NOTES) + '\n'
        'Return one JSON object using the exact hyperparameter names and JSON value types above. '
        'Include optimizer_family and all active hyperparameters and constants, omit inactive hyperparameters, '
        'and obey the bounds, choices, quantization, conditions and forbidden combinations. '
        'Do not include explanations, scores or markdown.'
    )
    return [dict(role='system', content=SYSTEM), dict(role='user', content=prompt),
            dict(role='assistant', content=canonical(config))]


def checked_history(task_id, raw, source):
    if hashlib.sha256(raw).hexdigest() != source['source_sha256']:
        raise ValueError('Changed train838 source: ' + task_id)
    history = json.loads(raw)
    meta, observations = history['meta_info'], history['observations']
    if (history['task_id'] != task_id or meta['domain'] != 'taskset' or meta['optimizer'] != 'tpe'
            or meta['seed'] != 1 or len(observations) != 100):
        raise ValueError('Expected a complete 100-evaluation TaskSet TPE history: ' + task_id)
    for observation in observations:
        if observation['trial_state'] == 0:
            info = observation.get('extra_info', {}).get('info', {})
            if (info.get('oracle') != 'train_inner' or info.get('protocol') != PROTOCOL
                    or info.get('objective_metric') != METRIC):
                raise ValueError('Historical evaluation protocol mismatch: ' + task_id)
    return observations


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_jsonl(path, rows):
    with path.open('w') as stream:
        for row in rows:
            stream.write(canonical(row) + '\n')


def export(output, corpus=CORPUS):
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing export: ' + str(output))
    split = read_split()
    manifest, sources = load_corpus(corpus, check_sources=False)
    if manifest['purpose'] != 'production_training' or set(sources) != set(split['train']):
        raise ValueError('BOLT training requires the complete train838 corpus')
    pinned = json.loads((CAMPAIGN / 'source_hashes.json').read_text())
    context_paths = sorted((CAMPAIGN / 'task_definitions').glob('*.json'))
    space_path = CAMPAIGN / 'taskset_data/scripts/optimizer_search_spaces.json'
    inputs = [*context_paths, space_path, LEGACY_NADAMW]
    for path in inputs:
        if digest(path) != pinned[str(path.relative_to(CAMPAIGN))]:
            raise ValueError('Changed frozen TaskSet input: ' + str(path))
    if digest(space_path) != manifest['search_space_sha256']:
        raise ValueError('Search space does not match the source corpus')
    if digest(LEGACY_NADAMW) != LEGACY_NADAMW_SHA256:
        raise ValueError('Historical Nesterov behavior requires a new runtime audit')
    specs = load_task_specs(CAMPAIGN / 'task_definitions')
    source_cs = space_from_taskset(load_family_spaces(space_path))
    source_fingerprint = schema_fingerprint(source_cs)
    cs = effective_space(source_cs)
    fingerprint = schema_fingerprint(cs)
    examples, provenance, exclusions, tasks = [], [], [], []
    for task_id in sorted(split['train']):
        source = sources[task_id]
        observations = checked_history(task_id, history_bytes(task_id), source)
        selected, excluded, distinct = select_top_k(observations, source_cs, effective_cs=cs)
        source_path = HISTORIES / task_id / 'tpe/seed1/history.json'
        tasks.append(dict(task_id=task_id, task_family=specs[task_id]['task_family'],
                          source_path=str(source_path), source_sha256=source['source_sha256'],
                          observations=len(observations), valid_distinct=distinct, selected=len(selected)))
        exclusions.extend(dict(task_id=task_id, **row) for row in excluded)
        for rank, row in enumerate(selected, 1):
            example = {'messages': build_messages(task_id, specs[task_id], cs, row['config'])}
            provenance.append(dict(example_index=len(examples), example_sha256=value_digest(example),
                                   task_id=task_id, task_family=specs[task_id]['task_family'], rank=rank,
                                   source_path=str(source_path), source_sha256=source['source_sha256'],
                                   source_schema_fingerprint=source_fingerprint,
                                   schema_fingerprint=fingerprint, **row))
            examples.append(example)
        if len(tasks) % 100 == 0:
            print(f'Checked {len(tasks)}/838 train histories; selected {len(examples)} examples', flush=True)
    if not examples:
        raise ValueError('No eligible training examples')
    assert len(examples) + len(exclusions) == 83800
    assert not {row['task_id'] for row in provenance} & set(split['test'])
    output.mkdir(parents=True)
    write_jsonl(output / 'train.jsonl', examples)
    write_jsonl(output / 'provenance.jsonl', provenance)
    write_jsonl(output / 'exclusions.jsonl', exclusions)
    write_json(output / 'tasks.json', tasks)
    write_json(output / 'schemas.json', {fingerprint: configspace_spec(cs),
                                        source_fingerprint: configspace_spec(source_cs)})
    write_json(output / 'dataset_info.json', {'bolt_taskset_top10_train838': {
        'file_name': 'train.jsonl', 'formatting': 'sharegpt', 'columns': {'messages': 'messages'},
        'tags': {'role_tag': 'role', 'content_tag': 'content', 'user_tag': 'user',
                 'assistant_tag': 'assistant', 'system_tag': 'system'},
    }})
    review_index = next((i for i, row in enumerate(provenance) if row['task_id'].startswith('mlp_family_seed')
                         and row['rank'] == 1), 0)
    review = examples[review_index]
    write_json(output / 'example.json', review)
    write_json(output / 'example_provenance.json', provenance[review_index])
    (output / 'example_prompt.txt').write_text(review['messages'][1]['content'] + '\n')
    same_task = [row for row in provenance if row['task_id'] == provenance[review_index]['task_id']]
    write_json(output / 'example_task_top10.json', same_task)
    normalized_index = next((i for i, row in enumerate(provenance) if row['configuration_adjustments']), None)
    if normalized_index is not None:
        write_json(output / 'example_nesterov_disabled.json', examples[normalized_index])
        write_json(output / 'example_nesterov_provenance.json', provenance[normalized_index])
    input_paths = [Path(__file__), Path(__file__).with_name('taskset_transfer_common.py'),
                   corpus / 'manifest.json', CAMPAIGN / 'source_hashes.json', *inputs]
    report = dict(format_version=3, method='BOLT-TaskSet', k=K,
                  train_manifest=str(CAMPAIGN / 'split.json'), train_manifest_sha256=SPLIT_SHA256,
                  corpus=str(corpus), source_policy='successful_real_tpe_observations',
                  allowed_tasks=838, represented_tasks=sum(t['selected'] > 0 for t in tasks),
                  source_observations=83800, examples=len(examples), test_tasks_used=0,
                  tasks_with_fewer_than_k=[t['task_id'] for t in tasks if t['selected'] < K],
                  objective_direction='minimize normalized validation loss', objective_metric=METRIC,
                  protocol=PROTOCOL, schedule_horizon_steps=SCHEDULE_HORIZON,
                  schedule_horizon_family='nadamw_grid', validation_measurement_steps=[0, 1000],
                  selection='Lowest objective per distinct effective configuration; top 10 per train838 task; ties use source order',
                  schema_policy='Validate original configs unchanged; map the ignored historical Nesterov flag to false; no other repair, rounding or default filling',
                  source_schema_fingerprint=source_fingerprint, effective_schema_fingerprint=fingerprint,
                  nesterov_policy=dict(parameter=NESTEROV_HP, allowed_values=[False],
                                       reason='legacy_factory_ignored_use_nesterov', runtime_path=str(LEGACY_NADAMW),
                                       runtime_sha256=LEGACY_NADAMW_SHA256,
                                       selected_targets_changed=sum(bool(r['configuration_adjustments']) for r in provenance)),
                  prompt_policy='Frozen task description, evaluation protocol and full CASH schema; no observations, scores or ranks',
                  exclusions=dict(Counter(row['reason'] for row in exclusions)),
                  task_family_examples=dict(sorted(Counter(row['task_family'] for row in provenance).items())),
                  optimizer_family_examples=dict(sorted(Counter(row['config']['optimizer_family'] for row in provenance).items())),
                  ml_reference_exporter=str(ML_EXPORTER), ml_reference_exporter_sha256=digest(ML_EXPORTER),
                  input_sha256={str(p): digest(p) for p in input_paths},
                  files={p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
                  training_started=False, review_pending=True)
    write_json(output / 'manifest.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=OUTPUT)
    parser.add_argument('--corpus', type=Path, default=CORPUS)
    args = parser.parse_args()
    report = export(args.out_dir.resolve(), args.corpus.resolve())
    print(json.dumps({key: report[key] for key in ('examples', 'represented_tasks', 'tasks_with_fewer_than_k',
                                                 'exclusions', 'task_family_examples')}, indent=2))
