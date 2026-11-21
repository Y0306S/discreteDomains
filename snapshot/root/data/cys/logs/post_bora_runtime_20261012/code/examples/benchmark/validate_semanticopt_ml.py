#!/usr/bin/env python3
"""Run a saved SemanticOpt adapter on real, task-held-out ML evaluations."""

import argparse
from collections import Counter
import csv
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from examples.benchmark.semanticopt_ml_sft import CONFIG, SYSTEM, TRAIN80
from mindware.components.optimizers.semanticopt_protocol import (
    build_messages, normalize_response, objective_feedback,
)
from mindware.components.optimizers.transfer_hpo.protocol import (
    allowed_task_ids, atomic_json, sha256_file,
)
from mindware.components.optimizers.transfer_hpo.schema import ALGORITHMS

TEST = REPO / 'benchmark_data/ml_test_20_fixed_ckpt32_rgpe_20261002/tasks_132.json'
TIMINGS = Path('/root/reports/ml_runtime_20261001/task_runtime_across_methods.csv')


def read(path):
    return json.loads(Path(path).read_text())


def task_key(row):
    return '/'.join(row[k] for k in ('task_type', 'dataset', 'algorithm'))


def select_tasks():
    tasks = {task_key(row): row for row in read(TEST)}
    train = allowed_task_ids(TRAIN80)
    if {t.lower() for t in tasks} & {t.lower() for t in train}:
        raise ValueError('The fixed task splits overlap')
    with TIMINGS.open() as stream:
        runtimes = {r['task_id']: float(r['median_of_method_means_seconds']) for r in csv.DictReader(stream)}
    selected = []
    for task_type in ('cls', 'rgs'):
        for algorithm in ALGORITHMS:
            options = [t for t, row in tasks.items() if row['task_type'] == task_type
                       and row['algorithm'] == algorithm and t in runtimes and math.isfinite(runtimes[t])]
            if not options:
                raise ValueError(f'No held-out case for {task_type}/{algorithm}')
            task = min(options, key=lambda t: (runtimes[t], t))
            selected.append(dict(tasks[task], task_id=task, historical_mean_eval_seconds=runtimes[task]))
    return selected


def recipe(args):
    config = read(CONFIG)
    adapter = Path(config['output_dir']) / 'final_adapter'
    previous = read(Path(config['output_dir']) / 'inference_validation.json')
    adapter_hash = sha256_file(adapter / 'adapter_model.safetensors')
    if adapter_hash != previous['adapter']['sha256']:
        raise ValueError('SemanticOpt adapter changed since the verified reload')
    return {
        'method': 'semanticopt', 'adapter_path': str(adapter), 'adapter_sha256': adapter_hash,
        'base_model': config['model_name_or_path'], 'seed': args.seed,
        'initial_evaluations': 5, 'adaptive_evaluations': args.proposals,
        'evaluation_timeout_seconds': 2400, 'overall_timeout_seconds': None,
        'initial_policy': 'default configuration plus four distinct seeded random configurations',
        'repeat_policy': 'evaluate and record repeats without retries or fallback',
        'invalid_proposal_policy': 'save exact response and stop that task; no replacement',
        'selection': 'smallest historical mean evaluation time per task-type/algorithm; no score-based selection',
        'scope': '14 task-held-out cases; functional validation and descriptive prediction errors, not full benchmark quality',
        'tasks': select_tasks(), 'gpu': args.gpu, 'cpu': args.cpu,
        'outer_test_size': .25, 'validation_size': .35, 'objective_direction': 'minimize',
        'train_manifest_sha256': sha256_file(TRAIN80), 'test_manifest_sha256': sha256_file(TEST),
        'timings_sha256': sha256_file(TIMINGS), 'sft_config_sha256': sha256_file(CONFIG),
        'training_complete_sha256': sha256_file(Path(config['output_dir']) / 'training_complete.json'),
        'code_sha256': {str(p.relative_to(REPO)): sha256_file(p) for p in (
            Path(__file__), REPO / 'mindware/components/optimizers/semanticopt_protocol.py',
            REPO / 'examples/benchmark/semanticopt_ml_sft.py',
            REPO / 'mindware/components/optimizers/isolated_evaluator.py',
            REPO / 'mindware/modules/base_evaluator.py', REPO / 'mindware/utils/data_manager.py')},
        'versions': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'peft', 'scikit-learn', 'ConfigSpace')},
    }


def resource_snapshot(cpu, gpu):
    import subprocess
    occupied = []
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            pid = int(entry.name)
            affinity = os.sched_getaffinity(pid)
            if pid not in (os.getpid(), os.getppid()) and len(affinity) < 40 and cpu in affinity:
                occupied.append({'pid': pid, 'name': (entry / 'comm').read_text().strip(), 'cpus': sorted(affinity)})
        except (OSError, ProcessLookupError):
            continue
    if occupied:
        raise RuntimeError(f'CPU {cpu} already has assigned processes: {occupied}')
    usage = subprocess.check_output(['nvidia-smi', '--id=' + str(gpu),
                                     '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
    if int(usage.strip()) < 28000:
        raise RuntimeError('Less than 28 GB available on the selected GPU')
    memory = int(Path('/sys/fs/cgroup/memory.current').read_text())
    maximum = int(Path('/sys/fs/cgroup/memory.max').read_text())
    stats = dict(line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
    working = memory - int(stats['inactive_file'])
    if maximum - working < 30 * 2**30:
        raise RuntimeError('Less than 30 GiB of cgroup working-set headroom')
    return {'gpu_free_mib': int(usage), 'cpu': cpu, 'cgroup_working_bytes': working,
            'cgroup_limit_bytes': maximum, 'checked_at': time.time()}


class Generator:
    def __init__(self, identity):
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        from examples.benchmark.hf_qwen_client import load_verified_adapter

        self.torch = torch
        torch.set_num_threads(1)
        torch.manual_seed(identity['seed'])
        self.tokenizer = AutoTokenizer.from_pretrained(identity['adapter_path'], local_files_only=True)
        base = AutoModelForImageTextToText.from_pretrained(
            identity['base_model'], dtype=torch.bfloat16, device_map={'': 0},
            attn_implementation='sdpa', local_files_only=True)
        self.model, self.verification = load_verified_adapter(base, identity['adapter_path'])
        if self.verification['sha256'] != identity['adapter_sha256'] or self.verification['verified_tensors'] != 256:
            raise RuntimeError('SemanticOpt adapter verification failed')
        self.model.eval()
        self.eos = [self.tokenizer.convert_tokens_to_ids(token)
                    for token in ('<|im_end|>', '<|im_start|>', '<|endoftext|>')]

    def generate(self, messages):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False)
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to('cuda')
        if inputs['input_ids'].shape[1] + 512 > 32768:
            raise ValueError('Prompt exceeds the validated context budget; no silent truncation')
        started = time.monotonic()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=512,
                                         eos_token_id=self.eos,
                                         pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        suffix = output[0, inputs['input_ids'].shape[1]:].tolist()
        return {'messages': messages, 'prompt': prompt,
                'raw_response': self.tokenizer.decode(suffix, skip_special_tokens=True),
                'input_tokens': inputs['input_ids'].shape[1], 'generated_tokens': len(suffix),
                'stopped_on_eos': bool(suffix and suffix[-1] in self.eos),
                'generation_seconds': time.monotonic() - started}


def setup_task(task, directory, seed):
    from examples.benchmark.export_bolt_ml_top10 import dataset_context
    from mindware.components.config_space.cs_builder import get_cs_args
    from mindware.components.metrics.metric import get_metric
    from mindware.components.optimizers.transfer_hpo.schema import build_hpo_cs, schema_fingerprint
    from mindware.components.utils.constants import CLASSIFICATION, REGRESSION
    from mindware.modules.cash.cash_evaluator import CASHCLSEvaluator, CASHRGSEvaluator
    from mindware.utils.data_manager import DataManager

    profile, _, provenance = dataset_context(task['task_type'], task['dataset'], split_seed=seed)
    task_type = CLASSIFICATION if task['task_type'] == 'cls' else REGRESSION
    manager = DataManager()
    frame = manager.load_csv(provenance['path'], header='infer', sep=',')
    train, _ = manager.split_data(frame, label_col=-1, test_size=.25, random_state=seed, task_type=task_type)
    node = manager.preprocess_fit(manager.from_train_df(train, label_col=-1), task_type)
    params = {'test_size': .35, 'shuffle': False}
    cs = build_hpo_cs(task['task_type'], task['algorithm'], **get_cs_args(data_node=node, resampling_params=params))
    evaluator_type = CASHCLSEvaluator if task['task_type'] == 'cls' else CASHRGSEvaluator
    evaluator = evaluator_type(data_node=node, scorer=get_metric(task['metric']),
                               resampling_strategy='holdout', resampling_params=params,
                               timestamp=time.time(), output_dir=str(directory), seed=seed)
    return cs, evaluator, profile, dict(provenance, schema_fingerprint=schema_fingerprint(cs))


def initial_configurations(cs, seed):
    cs.seed(seed)
    configs = [dict(cs.get_default_configuration())]
    for _ in range(1000):
        if len(configs) == 5:
            return configs
        value = dict(cs.sample_configuration())
        if value not in configs:
            configs.append(value)
    raise RuntimeError('Could not obtain five distinct initial configurations')


def evaluate_pending(record, path, directory, evaluator, cs, task_type):
    from ConfigSpace import Configuration
    from mindware.components.optimizers.isolated_evaluator import run_isolated

    pending = record['pending']
    os.environ['ML_SESSION_DIR'] = str(directory)
    os.environ['ML_EVALUATION_ID'] = str(pending['evaluation'])
    result = run_isolated(evaluator, (Configuration(cs, values=pending['configuration']),), {}, 2400)
    feedback, status = objective_feedback(result, task_type)
    pending.update(feedback, evaluation_status=status, evaluation_seconds=result['elapsed_time'],
                   evaluation_traceback=result['traceback'])
    pending['repeat_of_evaluations'] = [row['evaluation'] for row in record['evaluations']
                                      if row['configuration'] == pending['configuration']]
    if pending['stage'] == 'semanticopt' and status == 'SUCCESS':
        mean, std = pending['proposal']['mean'], pending['proposal']['std']
        error = pending['objective'] - mean
        pending['prediction_error'] = {'signed_error': error, 'absolute_error': abs(error),
                                       'within_1std': abs(error) <= std,
                                       'within_1_96std': abs(error) <= 1.96 * std,
                                       'standardized_error': error / std if std > 0 else None,
                                       'zero_std_nonzero_error': std == 0 and error != 0}
    record['evaluations'].append(pending)
    record['pending'] = None
    atomic_json(path, record)
    print(record['task_id'], 'evaluation', pending['evaluation'], status,
          pending.get('objective'), flush=True)


def run_task(task, identity, generator, output):
    directory = output / 'tasks' / task['task_id']
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'trajectory.json'
    record = read(path) if path.exists() else {'task_id': task['task_id'], 'status': 'running',
                                              'evaluations': [], 'pending': None}
    if record['status'] in ('complete', 'proposal_failed'):
        return record
    cs, evaluator, profile, provenance = setup_task(task, directory, identity['seed'])
    if record.get('data_provenance', provenance) != provenance:
        raise RuntimeError('Data or schema changed since the saved task')
    record['data_provenance'] = provenance
    record.setdefault('initial_configurations', initial_configurations(cs, identity['seed']))
    total = 5 + identity['adaptive_evaluations']
    while len(record['evaluations']) < total:
        if not record['pending']:
            index = len(record['evaluations'])
            pending = {'evaluation': index + 1, 'stage': 'initial' if index < 5 else 'semanticopt'}
            if index < 5:
                pending['configuration'] = record['initial_configurations'][index]
            else:
                messages = build_messages(SYSTEM, profile, cs, task['algorithm'], task['task_type'], total,
                                          record['evaluations'][:5], record['evaluations'][5:])
                pending.update(generator.generate(messages))
                record['pending'] = pending
                atomic_json(path, record)
                try:
                    if not pending['stopped_on_eos']:
                        raise ValueError('Generation reached the token limit without EOS')
                    proposal, changes = normalize_response(pending['raw_response'], cs, task['algorithm'],
                        any(row['status'] == 'failure' for row in record['evaluations']))
                    pending.update(proposal=proposal, normalization=changes, configuration=proposal['next_configuration'])
                except (ValueError, KeyError, TypeError) as exc:
                    record.update(status='proposal_failed', proposal_error=str(exc))
                    atomic_json(path, record)
                    print(task['task_id'], 'PROPOSAL FAILED:', exc, flush=True)
                    return record
            record['pending'] = pending
            atomic_json(path, record)
        elif 'configuration' not in record['pending']:
            # A crash after generation must reuse the saved response, never another model call.
            pending = record['pending']
            if not pending['stopped_on_eos']:
                raise ValueError('Saved generation reached its token limit')
            proposal, changes = normalize_response(pending['raw_response'], cs, task['algorithm'],
                any(row['status'] == 'failure' for row in record['evaluations']))
            pending.update(proposal=proposal, normalization=changes, configuration=proposal['next_configuration'])
            atomic_json(path, record)
        evaluate_pending(record, path, directory, evaluator, cs, task['task_type'])
    record['status'] = 'complete'
    atomic_json(path, record)
    return record


def summarize(output, identity, errors):
    tasks, predictions = [], []
    for task in identity['tasks']:
        path = output / 'tasks' / task['task_id'] / 'trajectory.json'
        if not path.exists():
            continue
        record = read(path)
        rows = record['evaluations']
        initial = [r['objective'] for r in rows[:5] if r['status'] == 'success']
        adaptive = [r['objective'] for r in rows[5:] if r['status'] == 'success']
        best_initial, best_adaptive = min(initial, default=None), min(adaptive, default=None)
        tasks.append({'task_id': task['task_id'], 'status': record['status'],
                      'evaluations': len(rows), 'successes': sum(r['status'] == 'success' for r in rows),
                      'best_initial_objective': best_initial, 'best_adaptive_objective': best_adaptive,
                      'improved_over_initial': best_adaptive < best_initial if initial and adaptive else None,
                      'normalizations': dict(Counter(c for r in rows for c in r.get('normalization', []))),
                      'repeated_proposals': sum(bool(r['repeat_of_evaluations']) for r in rows[5:]),
                      'proposal_error': record.get('proposal_error')})
        for row in rows[5:]:
            if 'prediction_error' in row:
                predictions.append(dict(task_id=task['task_id'], evaluation=row['evaluation'],
                    objective=row['objective'], mean=row['proposal']['mean'], std=row['proposal']['std'],
                    **row['prediction_error']))
    summary = {'checked_at': time.time(), 'tasks': tasks, 'errors': errors,
               'actual_ml_evaluations': sum(t['evaluations'] for t in tasks),
               'successful_ml_evaluations': sum(t['successes'] for t in tasks),
               'complete_tasks': sum(t['status'] == 'complete' for t in tasks),
               'selected_tasks': len(identity['tasks']), 'successful_predictions': len(predictions),
               'predictions_within_1std': sum(p['within_1std'] for p in predictions),
               'predictions_within_1_96std': sum(p['within_1_96std'] for p in predictions),
               'zero_std_nonzero_error': sum(p['zero_std_nonzero_error'] for p in predictions),
               'tasks_improved_over_initial': sum(t['improved_over_initial'] is True for t in tasks),
               'scope': identity['scope'], 'full_benchmark_quality_verified': False}
    atomic_json(output / 'summary.json', summary)
    if predictions:
        with (output / 'prediction_errors.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--cpu', type=int, default=23)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--proposals', type=int, default=3)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.proposals < 1:
        raise ValueError('At least one adaptive proposal is required')
    os.sched_setaffinity(0, {args.cpu})
    os.chdir(REPO)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'EVAL_N_JOBS'):
        os.environ[key] = '1'
    identity = recipe(args)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'validation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.output / 'recipe.json'
        executed = (args.output / 'tasks').exists() or (args.output / 'adapter_verification.json').exists()
        if path.exists() and read(path) != identity and executed:
            raise ValueError('Existing output has different inputs/code/settings; use another directory')
        atomic_json(path, identity)
        if not args.execute:
            print(json.dumps(identity, indent=2))
            return
        atomic_json(args.output / 'resource_preflight.json', resource_snapshot(args.cpu, args.gpu))
        generator = Generator(identity)
        atomic_json(args.output / 'adapter_verification.json', generator.verification)
        errors = {}
        for task in identity['tasks']:
            try:
                run_task(task, identity, generator, args.output)
            except Exception:
                errors[task['task_id']] = traceback.format_exc()
                print(errors[task['task_id']], flush=True)
            summarize(args.output, identity, errors)
        print(json.dumps(summarize(args.output, identity, errors), indent=2), flush=True)


if __name__ == '__main__':
    main()
