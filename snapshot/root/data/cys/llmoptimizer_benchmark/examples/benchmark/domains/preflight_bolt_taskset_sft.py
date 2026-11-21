#!/usr/bin/env python3
"""Validate TaskSet BOLT data and prepare the complete ML full-SFT recipe."""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path

from export_bolt_taskset_top10 import (
    CORPUS, OUTPUT, NESTEROV_HP, LEGACY_NADAMW, LEGACY_NADAMW_SHA256,
    build_messages, canonical, value_digest, write_json,
)
from taskset_transfer_common import CAMPAIGN, digest, load_corpus, read_split

ML_CONFIG = Path('/root/data/cys/blockdata_archive/llmoptimizer_benchmark/configs/bolt_ml/full_sft.yaml')
MODEL_OUTPUT = Path('/root/data/cys/baseline/artifacts/bolt_taskset_top10_full_sft/model')


def validate_export(directory):
    from ConfigSpace import Configuration
    from mindware.components.optimizers.transfer_hpo.schema import cs_from_spec
    from mindware.domains.dlhpo.taskset_llm.spec import load_task_specs

    manifest = json.loads((directory / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if digest(directory / name) != expected:
            raise ValueError('Changed export file: ' + name)
    for name, expected in manifest['input_sha256'].items():
        if digest(name) != expected:
            raise ValueError('Changed export input: ' + name)
    split = read_split()
    _, sources = load_corpus(Path(manifest['corpus']), check_sources=False)
    with (directory / 'train.jsonl').open() as stream:
        examples = [json.loads(line) for line in stream]
    with (directory / 'provenance.jsonl').open() as stream:
        records = [json.loads(line) for line in stream]
    schemas = {key: cs_from_spec(value) for key, value in json.loads((directory / 'schemas.json').read_text()).items()}
    assert manifest['format_version'] == 3
    assert manifest['schedule_horizon_steps'] == 10000
    assert manifest['validation_measurement_steps'] == [0, 1000]
    assert digest(LEGACY_NADAMW) == manifest['nesterov_policy']['runtime_sha256'] == LEGACY_NADAMW_SHA256
    assert schemas[manifest['effective_schema_fingerprint']].get_hyperparameter(NESTEROV_HP).choices == (False,)
    specs = load_task_specs(CAMPAIGN / 'task_definitions')
    assert len(examples) == len(records) == manifest['examples'] == 8380
    assert Counter(r['task_id'] for r in records) == Counter({t: 10 for t in split['train']})
    assert not {r['task_id'] for r in records} & set(split['test'])
    expected = {}
    for task_id, source in sources.items():
        # Independently rank the verified successful corpus, retaining the best duplicate.
        best = {}
        for config, score, row in zip(source['configs'], source['y_minimize'], source['source_rows']):
            candidate = (float(score), row - 1)
            effective = dict(config)
            if effective['optimizer_family'] == 'nadamw_grid':
                effective[NESTEROV_HP] = False
            key = canonical(effective)
            if key not in best or candidate < best[key]:
                best[key] = candidate
        expected[task_id] = sorted(best, key=lambda key: best[key])[:10]
    for index, (example, record) in enumerate(zip(examples, records)):
        task_id, rank = record['task_id'], record['rank']
        assert record['example_index'] == index and record['example_sha256'] == value_digest(example)
        assert set(example) == {'messages'}
        assert [m['role'] for m in example['messages']] == ['system', 'user', 'assistant']
        source = sources[task_id]
        assert record['source_sha256'] == source['source_sha256']
        source_index = source['source_rows'].index(record['observation_index'] + 1)
        answer = json.loads(example['messages'][-1]['content'])
        original = source['configs'][source_index]
        assert canonical(original) == canonical(record['original_config'])
        assert value_digest(original) == record['original_config_sha256']
        source_cs = schemas[record['source_schema_fingerprint']]
        source_config = Configuration(source_cs, values=original)
        source_cs.check_configuration(source_config)
        assert canonical(dict(source_config)) == canonical(original)
        effective = dict(original)
        adjustments = []
        if effective['optimizer_family'] == 'nadamw_grid':
            assert type(effective[NESTEROV_HP]) is bool
            if effective[NESTEROV_HP]:
                adjustments.append(dict(parameter=NESTEROV_HP, recorded_value=True, effective_value=False,
                                        reason='legacy_factory_ignored_use_nesterov'))
            effective[NESTEROV_HP] = False
        assert record['configuration_adjustments'] == adjustments
        assert canonical(answer) == canonical(effective) == canonical(record['config'])
        assert value_digest(answer) == record['config_sha256']
        assert record['objective_minimize'] == float(source['y_minimize'][source_index])
        assert canonical(answer) == expected[task_id][rank - 1]
        cs = schemas[record['schema_fingerprint']]
        config = Configuration(cs, values=answer)
        cs.check_configuration(config)
        assert canonical(dict(config)) == canonical(answer)
        assert example['messages'] == build_messages(task_id, specs[task_id], cs, answer)
    assert sum(bool(r['configuration_adjustments']) for r in records) == manifest['nesterov_policy']['selected_targets_changed']
    return manifest, examples


def prepare(directory):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Set CUDA_VISIBLE_DEVICES empty for CPU-only preparation')
    import torch
    import yaml
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer

    manifest, examples = validate_export(directory)
    print('Verified all 8380 targets against the successful train838 corpus.', flush=True)
    original = yaml.safe_load(ML_CONFIG.read_text())
    config = dict(original, dataset_dir=str(directory), dataset='bolt_taskset_top10_train838',
                  output_dir=str(MODEL_OUTPUT), resume_from_checkpoint=None)
    config['deepspeed'] = str(Path(__file__).resolve().parents[3] / 'configs/bolt_taskset/zero3_cpu_adam.json')
    cpu_config = dict(config, deepspeed=None, bf16=False, use_cpu=True,
                      preprocessing_num_workers=1, dataloader_pin_memory=False)
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(cpu_config)
    assert finetuning_args.finetuning_type == 'full'
    assert not data_args.train_on_prompt and not data_args.packing and not data_args.val_size
    assert data_args.max_samples is None and not data_args.eval_dataset
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module['tokenizer']
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    expected, lengths, target_lengths = Counter(), [], []
    for index, example in enumerate(examples):
        system, user, assistant = example['messages']
        pairs = template.encode_multiturn(tokenizer, [user, assistant], system['content'], None)
        assert len(pairs) == 1 and not template.efficient_eos
        source, target = pairs[0]
        assert tokenizer.decode(target, skip_special_tokens=False) == assistant['content'] + '<|im_end|>\n'
        expected[value_digest([source + target, [-100] * len(source) + target])] += 1
        lengths.append(len(source) + len(target))
        target_lengths.append(len(target))
        if (index + 1) % 2000 == 0:
            print(f'Tokenized {index + 1}/8380 complete examples', flush=True)
    cutoff = max(original['cutoff_len'], 2 ** math.ceil(math.log2(max(lengths))))
    config['cutoff_len'] = data_args.cutoff_len = cutoff
    dataset_module = get_dataset(template, model_args, data_args, training_args, 'sft', **tokenizer_module)
    dataset = dataset_module['train_dataset']
    assert len(dataset) == len(examples)
    actual = Counter()
    for row in dataset:
        assert row['attention_mask'] == [1] * len(row['input_ids'])
        actual[value_digest([row['input_ids'], row['labels']])] += 1
    assert actual == expected, 'LLaMA-Factory changed or truncated the supervised tokens'
    assert not torch.cuda.is_initialized()
    config_path = directory / 'full_sft.yaml'
    config_path.write_text('# Full TaskSet BOLT SFT; see preflight.json for data verification.\n'
                           + yaml.safe_dump(config, sort_keys=False))
    report = dict(passed=True, examples=len(examples), tasks=838, test_tasks_used=0,
                  source_and_top10_ranking_verified=True, assistant_only_loss_verified=True,
                  nesterov_disabled_verified=True, original_config_provenance_verified=True,
                  schedule_horizon_steps=10000, validation_measurement_steps=[0, 1000],
                  normalized_targets=manifest['nesterov_policy']['selected_targets_changed'],
                  llama_factory_preprocessing_verified=True, truncated_examples=0,
                  min_tokens=min(lengths), max_tokens=max(lengths), cutoff_len=cutoff,
                  examples_over_ml_cutoff=sum(n > original['cutoff_len'] for n in lengths),
                  min_answer_tokens=min(target_lengths), max_answer_tokens=max(target_lengths),
                  ml_config=str(ML_CONFIG), ml_config_sha256=digest(ML_CONFIG),
                  review_config=str(config_path), review_config_sha256=digest(config_path),
                  recipe_changes={k: {'ml': original.get(k), 'taskset': v} for k, v in config.items() if original.get(k) != v},
                  effective_batch_size_on_two_gpus=16,
                  epochs=config['num_train_epochs'], planned_optimizer_steps=math.ceil(len(examples) / 16) * config['num_train_epochs'],
                  dataset_manifest_sha256=digest(directory / 'manifest.json'),
                  preflight_script_sha256=digest(Path(__file__)),
                  gpu_training_tested=False, training_started=False, review_pending=True)
    write_json(directory / 'preflight.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    prepare(args.data_dir.resolve())
