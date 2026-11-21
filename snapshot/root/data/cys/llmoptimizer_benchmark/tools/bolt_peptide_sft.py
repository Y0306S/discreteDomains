#!/usr/bin/env python3
"""Measured top-ten peptide targets and isolated two-GPU BOLT full SFT."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
from functools import partial
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time

SOURCE = Path('/root/data/logs/peptide_bo_train800_20261027_v5b')
AUDIT = Path('/root/reports/peptide_train800_test200_audit_20261027/train800_audit.json')
DATA = Path('/root/data/cys/llmoptimizer_benchmark/benchmark_data/bolt_peptide_top10_train800_v1')
ARTIFACTS = Path('/root/data/cys/baseline/artifacts/bolt_peptide_top10_full_sft')
BASE = '/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B'
ML_RECIPE = Path('/root/data/cys/baseline/artifacts/bolt_ml_top10_full_sft/run_20261001T171400Z/training_config.json')
ML_ZERO = ML_RECIPE.with_name('deepspeed_config.json')
GPUS = (2, 3)
CPUS = (18, 42)
CONTROL_CPU = 43
ALPHABET = frozenset('ACDEFGHIKLMNPQRSTVWY')
SYSTEM = ('You recommend antimicrobial peptide sequences. Return only one JSON object '
          'with exactly the key sequence, whose value is one amino-acid string.')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def value_hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.%d.tmp' % os.getpid())
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def select_top10(observations, reference, minimum_similarity=0.75):
    import numpy as np
    from Levenshtein import distance

    seen, candidates, excluded = {}, [], Counter()
    for index, row in enumerate(observations):
        sequence, score, mic = row['sequence'], row['score'], row['mic']
        if (not isinstance(sequence, str) or not sequence or len(sequence) > 30
                or not set(sequence) <= ALPHABET or sequence.count('C') > 1):
            raise ValueError('Invalid measured sequence at observation %d' % index)
        if (type(score) not in (int, float) or type(mic) not in (int, float)
                or not math.isfinite(score) or not math.isfinite(mic) or mic <= 0
                or not math.isclose(score, -mic, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError('Invalid negative-MIC score at observation %d' % index)
        if sequence in seen:
            first = seen[sequence]
            if np.float32(first['score']) != np.float32(score):
                raise ValueError('Repeated sequence has conflicting measured scores')
            excluded['duplicate_sequence'] += 1
            continue
        seen[sequence] = row
        similarity = (len(reference) - distance(sequence, reference)) / len(reference)
        feasible = similarity >= minimum_similarity
        if type(row['feasible']) is not bool or row['feasible'] != feasible:
            raise ValueError('Stored feasibility disagrees with the reference')
        if not feasible:
            excluded['infeasible'] += 1
            continue
        candidates.append(dict(observation_index=index, sequence=sequence,
            mic=float(mic), score=float(score), similarity=similarity,
            phase=row['phase'], round=row['round'], cached=row['cached']))
    candidates.sort(key=lambda x: (x['mic'], x['observation_index']))
    if len(candidates) < 10:
        raise ValueError('Fewer than ten distinct feasible measured sequences')
    excluded['outside_top10'] = len(candidates) - 10
    return candidates[:10], dict(excluded), len(candidates)


def messages(task, reference, sequence):
    context = dict(task_id=task, domain='antimicrobial_peptide_optimization',
        reference_sequence=reference,
        objective=dict(name='APEX bacteria_0 predicted mean MIC', direction='minimize'),
        constraints=dict(maximum_length=30, alphabet=''.join(sorted(ALPHABET)),
            maximum_cysteine_count=1, minimum_reference_similarity=0.75,
            similarity_definition='(len(reference) - Levenshtein_distance(sequence, reference)) / len(reference)'))
    prompt = ('Recommend one promising feasible peptide sequence for this task.\n'
              + canonical(context) + '\nReturn {"sequence":"AMINO_ACID_STRING"}. '
              'Do not return latent vectors, scores, explanations, or markdown.')
    return [dict(role='system', content=SYSTEM), dict(role='user', content=prompt),
            dict(role='assistant', content=canonical(dict(sequence=sequence)))]


def export():
    if DATA.exists():
        raise FileExistsError('Refusing to overwrite an existing dataset: ' + str(DATA))
    recipe = json.loads((SOURCE / 'recipe.json').read_text())
    split = json.loads((SOURCE / 'split.json').read_text())
    audit = json.loads(AUDIT.read_text())
    train, test = recipe['train'], recipe['test']
    if (len(set(train)) != 800 or len(set(test)) != 200 or set(train) & set(test)
            or train != split['train_ids'] or test != split['test_ids']
            or not audit['summary']['passed'] or audit['summary']['tasks'] != 800):
        raise ValueError('Require the passed, disjoint, fixed train800/test200 source')
    records = {row['task']: row for row in audit['tasks']}
    if set(records) != set(train):
        raise ValueError('Training audit coverage differs from the fixed split')
    input_hashes = {str(p): digest(p) for p in (SOURCE / 'recipe.json', SOURCE / 'split.json', AUDIT)}
    DATA.mkdir(parents=True)
    examples, provenance, tasks = [], [], []
    for task in train:
        archive = SOURCE / 'runs' / (task + '.tar.gz')
        with tarfile.open(archive, 'r:gz') as packed:
            raw = packed.extractfile('history.json').read()
        checksum = hashlib.sha256(raw).hexdigest()
        if checksum != records[task]['member_checksums']['history.json']:
            raise ValueError('Source history checksum changed: ' + task)
        history = json.loads(raw)
        reference = recipe['references'][task]
        if history['status'] != 'complete' or history['reference'] != reference:
            raise ValueError('Wrong reference or incomplete history: ' + task)
        selected, excluded, valid = select_top10(history['observations'], reference, recipe['similarity'])
        tasks.append(dict(task_id=task, reference=reference, archive=str(archive),
            history_sha256=checksum, observations=len(history['observations']),
            valid_distinct=valid, selected=10, exclusions=excluded))
        for rank, row in enumerate(selected, 1):
            example = dict(messages=messages(task, reference, row['sequence']))
            provenance.append(dict(example_index=len(examples), example_sha256=value_hash(example),
                task_id=task, rank=rank, reference=reference, archive=str(archive),
                history_member='history.json', history_sha256=checksum, **row))
            examples.append(example)
        if len(tasks) % 50 == 0:
            atomic_json(DATA / 'export_progress.json', dict(completed_tasks=len(tasks), total_tasks=800,
                examples=len(examples), training_started=False))
            print(json.dumps(dict(exported_tasks=len(tasks), total=800)), flush=True)
    if len(examples) != 8000:
        raise ValueError('Expected exactly 8000 examples')
    for name, rows in (('train.jsonl', examples), ('provenance.jsonl', provenance)):
        with (DATA / name).open('w') as stream:
            for row in rows:
                stream.write(canonical(row) + '\n')
    atomic_json(DATA / 'tasks.json', tasks)
    atomic_json(DATA / 'dataset_info.json', {'bolt_peptide_top10_train800': dict(file_name='train.jsonl',
        formatting='sharegpt', columns=dict(messages='messages'), tags=dict(role_tag='role',
            content_tag='content', user_tag='user', assistant_tag='assistant', system_tag='system'))})
    if any(digest(p) != h for p, h in input_hashes.items()):
        raise ValueError('Pinned source inputs changed during export')
    manifest = dict(status='complete', domain='peptide', train_tasks=train, excluded_test_tasks=test,
        represented_tasks=800, examples=8000, top_k=10, input_sha256=input_hashes,
        exporter_sha256=digest(__file__), oracle_calls=0,
        selection='ten distinct feasible measured sequences per task, minimum MIC, archive-order ties',
        repeated_score_policy='first measured value; reject conflicts beyond float32 equivalence',
        initial_observations_included=True, output_format='JSON object with one sequence string',
        prompt_contains_target_scores=False, direct_sequence_output=True, latent_targets=False,
        files={name:digest(DATA / name) for name in ('train.jsonl','provenance.jsonl','tasks.json','dataset_info.json')})
    atomic_json(DATA / 'manifest.json', manifest)
    print(json.dumps(dict(data=str(DATA), tasks=800, examples=8000, oracle_calls=0)), flush=True)


def validate_data():
    manifest = json.loads((DATA / 'manifest.json').read_text())
    if (manifest['status'] != 'complete' or manifest['examples'] != 8000
            or len(set(manifest['train_tasks'])) != 800
            or set(manifest['train_tasks']) & set(manifest['excluded_test_tasks'])):
        raise ValueError('Incomplete or overlapping data')
    for path, checksum in manifest['input_sha256'].items():
        if digest(path) != checksum:
            raise ValueError('Frozen source changed: ' + path)
    for name, checksum in manifest['files'].items():
        if digest(DATA / name) != checksum:
            raise ValueError('Export file changed: ' + name)
    examples = [json.loads(x) for x in (DATA / 'train.jsonl').read_text().splitlines()]
    records = [json.loads(x) for x in (DATA / 'provenance.jsonl').read_text().splitlines()]
    if len(examples) != len(records) or len(examples) != 8000:
        raise ValueError('Data and provenance lengths differ')
    seen, counts = set(), Counter()
    for index, (example, row) in enumerate(zip(examples, records)):
        if (row['example_index'] != index or row['example_sha256'] != value_hash(example)
                or row['task_id'] not in manifest['train_tasks']):
            raise ValueError('Example provenance mismatch')
        if example['messages'] != messages(row['task_id'], row['reference'], row['sequence']):
            raise ValueError('Example output or context changed')
        key = (row['task_id'], row['sequence'])
        if key in seen:
            raise ValueError('Duplicate target within a task')
        seen.add(key)
        counts[row['task_id']] += 1
    if len(counts) != 800 or set(counts.values()) != {10}:
        raise ValueError('Every task must contribute ten targets')
    return manifest, examples


def recipe():
    config = json.loads(ML_RECIPE.read_text())
    config.update(dataset_dir=str(DATA), dataset='bolt_peptide_top10_train800',
        output_dir=str(ARTIFACTS / 'model'), resume_from_checkpoint=None,
        cutoff_len=1024, preprocessing_num_workers=1, dataloader_num_workers=0,
        gradient_accumulation_steps=8, save_steps=100, save_total_limit=2,
        deepspeed=str(DATA / 'deepspeed_config.json'))
    zero = json.loads(ML_ZERO.read_text())
    zero['zero_optimization'].pop('offload_param', None)
    zero['zero_optimization']['offload_optimizer'] = dict(device='cpu', pin_memory=False)
    return config, zero


def prepare():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU preflight requires CUDA_VISIBLE_DEVICES=')
    manifest, examples = validate_data()
    config, zero = recipe()
    atomic_json(DATA / 'deepspeed_config.json', zero)
    atomic_json(DATA / 'training_config.json', config)
    import torch
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer
    from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig

    cpu_config = dict(config, deepspeed=None, bf16=False, use_cpu=True, dataloader_pin_memory=False)
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(cpu_config)
    if (finetuning_args.finetuning_type != 'full' or data_args.train_on_prompt or data_args.packing
            or data_args.val_size or data_args.eval_dataset or data_args.max_samples is not None):
        raise ValueError('Expected full, assistant-only SFT on all examples')
    module = load_tokenizer(model_args)
    tokenizer = module['tokenizer']
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    expected, lengths, targets = Counter(), [], []
    for example in examples:
        system, user, answer = example['messages']
        pairs = template.encode_multiturn(tokenizer, [user, answer], system['content'], None)
        if len(pairs) != 1 or template.efficient_eos:
            raise ValueError('Unexpected template')
        source, target = pairs[0]
        if len(source) + len(target) > config['cutoff_len']:
            raise ValueError('Training cutoff would truncate an example')
        if tokenizer.decode(target) != answer['content'] + '<|im_end|>\n':
            raise ValueError('Assistant tokens differ from the direct sequence JSON')
        expected[value_hash([source + target, [-100] * len(source) + target])] += 1
        lengths.append(len(source) + len(target))
        targets.append(len(target))
    dataset = get_dataset(template, model_args, data_args, training_args, 'sft', **module)['train_dataset']
    actual = Counter(value_hash([r['input_ids'], r['labels']]) for r in dataset)
    if actual != expected or len(dataset) != 8000:
        raise ValueError('LLaMA-Factory dropped, changed, or truncated supervised examples')
    ds = HfTrainerDeepSpeedConfig(config['deepspeed'])
    ds_args = type('DSArgs', (), dict(vars(training_args), world_size=2, bf16=True))()
    ds.trainer_config_process(ds_args)
    if ds.mismatches or ds.config['train_batch_size'] != 16:
        raise ValueError('DeepSpeed recipe mismatch')
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    reference = torch.nn.Parameter(parameter.detach().clone())
    kwargs = dict(lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    optimizer = DeepSpeedCPUAdam([parameter], adamw_mode=True, **kwargs)
    reference_optimizer = torch.optim.AdamW([reference], **kwargs)
    for _ in range(3):
        parameter.grad = torch.tensor([0.2, -0.3])
        reference.grad = parameter.grad.clone()
        optimizer.step()
        reference_optimizer.step()
    torch.testing.assert_close(parameter, reference, rtol=1e-6, atol=1e-7)
    del optimizer
    if torch.cuda.is_initialized():
        raise ValueError('CPU preflight initialized CUDA')
    report = dict(passed=True, tasks=800, examples=8000, test_tasks_used=0,
        min_tokens=min(lengths), max_tokens=max(lengths), max_answer_tokens=max(targets),
        cutoff_len=config['cutoff_len'], truncated_examples=0, assistant_only_loss_verified=True,
        full_language_model_sft=True, cpu_adam_verified=True, effective_batch_size=16,
        epochs=5, planned_optimizer_steps=2500, training_started=False,
        manifest_sha256=digest(DATA / 'manifest.json'), config_sha256=digest(DATA / 'training_config.json'),
        deepspeed_sha256=digest(DATA / 'deepspeed_config.json'), script_sha256=digest(__file__),
        original_ml_recipe_sha256=digest(ML_RECIPE), original_ml_zero_sha256=digest(ML_ZERO),
        recipe_changes={k:dict(ml=json.loads(ML_RECIPE.read_text()).get(k), peptide=v)
            for k,v in config.items() if json.loads(ML_RECIPE.read_text()).get(k) != v},
        versions={k:importlib.metadata.version(k) for k in ('torch','transformers','llamafactory','deepspeed')})
    atomic_json(DATA / 'preflight.json', report)
    print(json.dumps(report, indent=2), flush=True)


def resources():
    import psutil
    cg = Path('/sys/fs/cgroup')
    limit, current = int((cg / 'memory.max').read_text()), int((cg / 'memory.current').read_text())
    stats = dict(line.split() for line in (cg / 'memory.stat').read_text().splitlines())
    reclaimable = min(int(stats.get('inactive_file', 0)),
        max(0, int(stats.get('file', 0)) - int(stats.get('shmem', 0))))
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,memory.total',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    gpu = {str(int(v[0])):float(v[1])/1024 for v in (line.split(',') for line in output.splitlines())}
    return dict(time=time.time(), gpu_free_gib=gpu, container_limit_gib=limit/2**30,
        container_used_gib=current/2**30,
        available_ram_gib=min(psutil.virtual_memory().available, max(0, limit-current+reclaimable))/2**30,
        disk_free_gib=shutil.disk_usage(DATA).free/2**30,
        memory_events=(cg/'memory.events').read_text())


def check_readiness():
    report = json.loads((DATA / 'preflight.json').read_text())
    for key, path in (('manifest_sha256', DATA/'manifest.json'), ('config_sha256', DATA/'training_config.json'),
                     ('deepspeed_sha256', DATA/'deepspeed_config.json'), ('script_sha256', Path(__file__))):
        if report[key] != digest(path):
            raise ValueError('Preflight became stale: ' + key)
    if not report['passed'] or report['truncated_examples']:
        raise ValueError('Require a passed, untruncated preflight')
    validate_data()
    sample = resources()
    if socket.gethostname() != 'cs-94bac-0693e-server':
        raise ValueError('This campaign is restricted to LLMOptimizer2')
    if (any(sample['gpu_free_gib'][str(g)] < 60 for g in GPUS)
            or sample['available_ram_gib'] < 180 or sample['disk_free_gib'] < 400):
        raise RuntimeError('Insufficient GPU, container RAM, or disk headroom: ' + canonical(sample))
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = [x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
            if not any(x.endswith('.py') for x in argv):
                continue
            affinity = os.sched_getaffinity(int(proc.name))
            if len(affinity) <= 8 and affinity.intersection(set(CPUS)|{CONTROL_CPU}):
                raise ValueError('Requested CPU overlaps an existing restricted process: ' + proc.name)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return sample


def verify_checkpoint(path, step):
    state = json.loads((path/'trainer_state.json').read_text())
    tag = (path/'latest').read_text().strip()
    if state['global_step'] != step or tag != 'global_step%d' % step:
        raise ValueError('Checkpoint step mismatch')
    index = path/'model.safetensors.index.json'
    names = set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
    if not names or any(Path(n).name != n for n in names):
        raise ValueError('Invalid checkpoint shard names')
    required = [path/n for n in names]
    for rank in range(2):
        required += [path/('rng_state_%d.pth' % rank),
            path/tag/('bf16_zero_pp_rank_%d_mp_rank_00_optim_states.pt' % rank),
            path/tag/('zero_pp_rank_%d_mp_rank_00_model_states.pt' % rank)]
    if any(not p.is_file() or not p.stat().st_size for p in required):
        raise ValueError('Incomplete two-rank model/optimizer checkpoint')


def worker(run_dir):
    rank = int(os.environ['LOCAL_RANK'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2,3' or rank not in (0,1):
        raise ValueError('Expected ranks on GPUs 2 and 3')
    os.sched_setaffinity(0, {CPUS[rank]})
    import torch
    from transformers import TrainerCallback
    from llamafactory.train.tuner import run_exp
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(rank)
    torch.cuda.set_per_process_memory_fraction(0.85, rank)
    config = json.loads((run_dir/'training_config.json').read_text())
    step_started = time.monotonic()

    class HealthCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            nonlocal step_started
            if state.global_step != 0 or state.max_steps != 2500:
                raise ValueError('Unexpected clean-base training extent')
            model = kwargs['model']
            optimizer = next(p for p in model.parameters() if p.requires_grad)._z3_optimizer.optimizer
            if type(optimizer).__name__ != 'DeepSpeedCPUAdam':
                raise ValueError('Unexpected optimizer')
            model.save_pretrained = partial(model.save_pretrained, max_shard_size='1GB')
            atomic_json(run_dir/('rank%d_started.json'%rank), dict(pid=os.getpid(), time=time.time(),
                rank=rank, physical_gpu=GPUS[rank], max_steps=state.max_steps,
                trainable_parameters=sum(p.ds_numel if hasattr(p,'ds_numel') else p.numel()
                    for p in model.parameters() if p.requires_grad),
                optimizer=type(optimizer).__name__, cpu_affinity=sorted(os.sched_getaffinity(0))))
            step_started = time.monotonic()

        def on_step_end(self, args, state, control, **kwargs):
            nonlocal step_started
            elapsed = time.monotonic()-step_started
            step_started = time.monotonic()
            atomic_json(run_dir/('rank%d_health.json'%rank), dict(pid=os.getpid(), time=time.time(),
                step=state.global_step, max_steps=state.max_steps, epoch=state.epoch,
                step_seconds=elapsed, gpu_allocated_gib=torch.cuda.memory_allocated(rank)/2**30,
                gpu_reserved_gib=torch.cuda.memory_reserved(rank)/2**30,
                peak_allocated_gib=torch.cuda.max_memory_allocated(rank)/2**30,
                cpu_affinity=sorted(os.sched_getaffinity(0))))

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and 'loss' in logs:
                if not math.isfinite(float(logs['loss'])):
                    raise FloatingPointError('Nonfinite SFT loss')
                atomic_json(run_dir/('rank%d_loss.json'%rank), dict(step=state.global_step,
                    time=time.time(), **logs))

        def on_save(self, args, state, control, **kwargs):
            torch.distributed.barrier(device_ids=[rank])
            path = Path(args.output_dir)/('checkpoint-%d'%state.global_step)
            verify_checkpoint(path, state.global_step)
            atomic_json(run_dir/('rank%d_checkpoint.json'%rank), dict(step=state.global_step,
                path=str(path), complete=True, time=time.time()))

        def on_train_end(self, args, state, control, **kwargs):
            if state.global_step != 2500:
                raise ValueError('Training ended before the full five epochs')
            atomic_json(run_dir/('rank%d_finished.json'%rank), dict(step=state.global_step,
                epoch=state.epoch, time=time.time()))

    run_exp(args=config, callbacks=[HealthCallback()])


def supervise(run_dir):
    os.sched_setaffinity(0, {CONTROL_CPU})
    with (ARTIFACTS/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        config = json.loads((DATA/'training_config.json').read_text())
        config['deepspeed'] = str(run_dir/'deepspeed_config.json')
        atomic_json(run_dir/'training_config.json', config)
        shutil.copy2(DATA/'deepspeed_config.json', run_dir/'deepspeed_config.json')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='2,3', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
            PYTHONDONTWRITEBYTECODE='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', MAX_JOBS='1')
        env['PATH'] = str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
        command = [sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',
            '--nproc_per_node=2',str(Path(__file__).resolve()),'worker','--run-dir',str(run_dir)]
        with (run_dir/'training.log').open('w') as stream:
            child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        status = dict(status='starting', supervisor_pid=os.getpid(), torchrun_pid=child.pid,
            host='LLMOptimizer2', gpus=list(GPUS), rank_cpus=list(CPUS), control_cpu=CONTROL_CPU,
            run_dir=str(run_dir), started_at=time.time(), examples=8000, tasks=800,
            epochs=5, max_steps=2500, effective_batch_size=16, direct_sequence_output=True,
            source_manifest_sha256=digest(DATA/'manifest.json'), script_sha256=digest(__file__))

        def interrupted(signum, frame):
            raise InterruptedError('Supervisor received signal %d'%signum)

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            while child.poll() is None:
                sample = resources()
                if sample['available_ram_gib'] < 40 or sample['disk_free_gib'] < 80:
                    raise RuntimeError('Container RAM or disk reserve reached')
                health = [run_dir/('rank%d_health.json'%r) for r in range(2)]
                status.update(status='running' if all(p.exists() for p in health) else 'starting',
                    checked_at=time.time(), resources=sample)
                atomic_json(run_dir/'status.json', status)
                atomic_json(ARTIFACTS/'current_run.json', status)
                time.sleep(10)
            status.update(status='completed' if child.returncode == 0 else 'failed', exit_code=child.returncode)
        except BaseException as exc:
            status.update(status='stopped', reason=repr(exc))
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=30)
            status['exit_code'] = child.returncode
            raise
        finally:
            status['checked_at'] = time.time()
            atomic_json(run_dir/'status.json', status)
            atomic_json(ARTIFACTS/'current_run.json', status)


def launch():
    sample = check_readiness()
    output = ARTIFACTS/'model'
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refusing to overwrite existing model output')
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    run_dir = ARTIFACTS/datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%SZ')
    run_dir.mkdir()
    script = run_dir/'bolt_peptide_sft.py'
    shutil.copy2(__file__, script)
    atomic_json(run_dir/'admission.json', dict(passed=True, resources=sample, gpus=list(GPUS),
        rank_cpus=list(CPUS), control_cpu=CONTROL_CPU, launch_authorized=True))
    with (run_dir/'supervisor.log').open('w') as stream:
        process = subprocess.Popen([sys.executable,str(script),'supervise','--run-dir',str(run_dir)],
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(supervisor_pid=process.pid, run_dir=str(run_dir))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('export','prepare','check','launch','supervise','worker'))
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    if args.action in ('supervise','worker') and args.run_dir is None:
        parser.error('--run-dir is required')
    if args.action == 'export':
        export()
    elif args.action == 'prepare':
        prepare()
    elif args.action == 'check':
        print(json.dumps(check_readiness(), indent=2))
    elif args.action == 'launch':
        launch()
    elif args.action == 'worker':
        worker(args.run_dir)
    else:
        supervise(args.run_dir)


if __name__ == '__main__':
    main()
