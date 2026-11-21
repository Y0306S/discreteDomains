#!/usr/bin/env python3
"""Freeze train838 S1B events, measured S2-G pairs, and the derived test200 split."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ConfigSpace import Configuration
from mindware.domains.dlhpo.taskset_llm import p5_prompt as P, spec as S
from mindware.domains.dlhpo.taskset_space import load_family_spaces, space_from_taskset
from export_bolt_taskset_top10 import (checked_history, effective_space, historical_effective_config,
                                      LEGACY_NADAMW, LEGACY_NADAMW_SHA256, PARAMETER_NOTES)
from taskset_s1b_s2g_common import ML_SELECTOR, improvement_events, preference_pair, structured_space
from taskset_transfer_common import CAMPAIGN, METRIC, PROTOCOL, SPLIT_SHA256, atomic_json, digest, history_bytes, load_corpus, read_split

CORPUS = Path('/root/data/logs/taskset_transfer_train838_20261002/corpus')
PROTOCOL_TEXT = ('1000 evaluated training steps; VALID_OUTER measured at steps 0 and 1000; 1 replica. '
                 'nadamw_grid schedule horizon: 10000 steps, without rescaling to the evaluation budget. '
                 'Nesterov momentum is disabled.')


def write_jsonl(path, rows):
    with path.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')


def build(out):
    if out.exists():
        raise FileExistsError('Refusing to overwrite an existing export: ' + str(out))
    split = read_split()
    manifest, sources = load_corpus(CORPUS, check_sources=False)
    if manifest['purpose'] != 'production_training' or set(sources) != set(split['train']):
        raise ValueError('All train838 histories are required')
    if digest(LEGACY_NADAMW) != LEGACY_NADAMW_SHA256:
        raise ValueError('Historical Nesterov factory changed')
    specs = S.load_task_specs(CAMPAIGN / 'task_definitions')
    space_path = CAMPAIGN / 'taskset_data/scripts/optimizer_search_spaces.json'
    if digest(space_path) != manifest['search_space_sha256']:
        raise ValueError('Frozen search space changed')
    pinned = json.loads((CAMPAIGN / 'source_hashes.json').read_text())
    context_paths = [*sorted((CAMPAIGN / 'task_definitions').glob('*.json')), space_path, LEGACY_NADAMW]
    for path in context_paths:
        if digest(path) != pinned[str(path.relative_to(CAMPAIGN))]:
            raise ValueError('Frozen input changed: ' + str(path))
    original_cs = space_from_taskset(load_family_spaces(space_path))
    cs = effective_space(original_cs)
    space = structured_space(cs)
    events, preferences, provenance, counts, task_counts = [], [], [], Counter(), {}
    for task, source in sorted(sources.items()):
        observations = checked_history(task, history_bytes(task), source)
        rows = []
        successful_rows = {}
        for trial, obs in enumerate(observations, 1):
            original = obs['config']
            original_cs.check_configuration(Configuration(original_cs, values=original))
            config = historical_effective_config(original)
            cs.check_configuration(Configuration(cs, values=config))
            ok = obs['trial_state'] == 0
            row = dict(trial=trial, config=config, successful=ok,
                       loss=float(obs['objectives'][0]) if ok else None)
            rows.append(row)
            counts['successful_observations' if ok else 'failed_observations'] += 1
            if ok:
                successful_rows[trial] = (original, row['loss'])
        if list(successful_rows) != source['source_rows']:
            raise ValueError('Historical trial IDs do not match the frozen corpus')
        for trial, config, loss in zip(source['source_rows'], source['configs'], source['y_minimize']):
            if successful_rows[trial] != (config, float(loss)):
                raise ValueError('Historical configurations/outcomes changed')
        own = Counter()
        for event in improvement_events(rows):
            prefix = rows[:event['position']]
            kwargs = dict(task_id=task, specs=specs, cs=cs, rows=prefix, trial=event['trial'], seed=1,
                          prompt_mode='p5_h0', protocol_text=PROTOCOL_TEXT, parameter_notes=PARAMETER_NOTES)
            system, user, _ = P.build_p5_prompt(**kwargs, response_format='json_only')
            sft_system, sft_user, _ = P.build_p5_prompt(**kwargs, response_format='rationale_json')
            metadata = dict(task_id=task, task_family=S.task_family_of(task), trial=event['trial'],
                            prefix_successes=event['prefix_successes'], loss=event['loss'],
                            delta_rel_score=event['delta_rel_score'], source_sha256=source['source_sha256'])
            config_text = json.dumps(event['config'], sort_keys=True, allow_nan=False)
            _, context, _ = P.build_p5_prompt(**kwargs, include_output_instructions=False)
            rationale_user = (context + '\n\nTARGET CONFIGURATION (hindsight: this config improved best-so-far):\n\n'
                              + config_text + "\n\nWrite exactly one line starting with 'Rationale:' explaining why to propose this configuration next.")
            events.append(dict(metadata=metadata, config=event['config'],
                               messages=[dict(role='system', content=sft_system), dict(role='user', content=sft_user)],
                               rationale_messages=[dict(role='system', content='You write short, faithful rationales for hyperparameter optimization. Given the trial history and a target configuration that improved the incumbent, explain briefly why that configuration is a good next proposal. Do not invent scores. Do not output JSON.'),
                                                   dict(role='user', content=rationale_user)]))
            own['sft'] += 1
            pair = preference_pair(rows, event, space)
            if pair is None:
                counts['empty_preference_pool'] += 1
                continue
            for config in (pair['chosen'], pair['rejected']):
                cs.check_configuration(Configuration(cs, values=config))
            index = len(preferences)
            preferences.append(dict(conversations=[{'from': 'system', 'value': system}, {'from': 'human', 'value': user}],
                                    chosen={'from': 'gpt', 'value': S.config_json({'config': pair['chosen']})},
                                    rejected={'from': 'gpt', 'value': S.config_json({'config': pair['rejected']})}))
            provenance.append(dict(metadata, pair_index=index, original_chosen_config=observations[event['position']]['config'],
                                   original_rejected_config=observations[pair['rejected_trial'] - 1]['config'], **pair))
            own['orpo'] += 1
        task_counts[task] = dict(own)
        if len(task_counts) % 100 == 0:
            print(f'Validated {len(task_counts)}/838 histories; {len(events)} SFT events, {len(preferences)} ORPO pairs', flush=True)
    timing_path = Path('/root/reports/task_runtime_analysis_20261026/taskset_task_summary.csv')
    with timing_path.open() as stream:
        timings = {r['task_id']: r for r in csv.DictReader(stream)}
    if not set(split['test']) <= set(timings):
        raise ValueError('Missing timing evidence for the original test210')
    removed = sorted(split['test'], key=lambda t: (-float(timings[t]['median_method_eval_mean_s']), t))[:10]
    test200 = dict(train=split['train'], test=[t for t in split['test'] if t not in removed], excluded_test=removed,
                   parent_split_sha256=SPLIT_SHA256, policy='remove_ten_slowest_by_median_method_mean_evaluator_seconds',
                   timing_sha256=digest(timing_path), removed_timings={t: float(timings[t]['median_method_eval_mean_s']) for t in removed})
    assert len(test200['test']) == 200 and not set(test200['train']) & set(test200['test'] + removed)
    out.mkdir(parents=True)
    write_jsonl(out / 'sft_events.jsonl', events)
    write_jsonl(out / 'orpo_train.jsonl', preferences)
    write_jsonl(out / 'orpo_provenance.jsonl', provenance)
    atomic_json(out / 'test200.json', test200)
    atomic_json(out / 'dataset_info.json', {'taskset_s1b': {'file_name': 'sft_train.jsonl', 'formatting': 'sharegpt',
                 'columns': {'messages': 'messages'}, 'tags': {'role_tag': 'role', 'content_tag': 'content',
                  'user_tag': 'user', 'assistant_tag': 'assistant', 'system_tag': 'system'}},
                'taskset_s2g': {'file_name': 'orpo_train.jsonl', 'formatting': 'sharegpt', 'ranking': True,
                               'columns': {'messages': 'conversations', 'chosen': 'chosen', 'rejected': 'rejected'}}})
    report = dict(status='complete', source_tasks=838, sft_examples=len(events), orpo_pairs=len(preferences),
                  counts=dict(counts), task_counts=task_counts, split_sha256=SPLIT_SHA256,
                  source_policy=manifest['source_policy'], protocol=PROTOCOL, metric=METRIC,
                  schedule_horizon_steps=10000, nesterov_enabled=False, prompt_mode='p5_h0',
                  sft_response_format='rationale_json', orpo_response_format='json_only',
                  min_successful_prefix=2, source_corpus_manifest_sha256=digest(CORPUS / 'manifest.json'),
                  input_sha256={str(p): digest(p) for p in context_paths},
                  code_sha256={str(p): digest(p) for p in (Path(__file__), ML_SELECTOR, Path(P.__file__), Path(S.__file__),
                                                         Path(__file__).with_name('taskset_s1b_s2g_common.py'))},
                  files={p.name: digest(p) for p in out.iterdir() if p.is_file()})
    atomic_json(out / 'manifest.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in {'task_counts', 'input_sha256', 'code_sha256', 'files'}}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    build(parser.parse_args().out)
