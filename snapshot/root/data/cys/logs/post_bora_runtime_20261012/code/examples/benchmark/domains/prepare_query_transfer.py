#!/usr/bin/env python3
"""Export approved train800 measurements and deterministic frozen-VAE features."""

import argparse
import ast
import csv
import json
import os
from pathlib import Path
import sys
import re

import numpy as np

from query_transfer_common import (
    POLICY, QP_ROOT, TRAIN_FILE, TEST_FILE, VAE_FILE, atomic_json, digest,
    load_corpus, read_split, run_lock, scale_latents, source_path, window_indices,
    validate_assets,
)


def export(audit_path, directory, batch_size, audit_only=False):
    validate_assets()
    audit = json.loads(audit_path.read_text())
    train, _ = read_split()
    if (audit['training_tasks'] != train
            or audit['train_split_sha256'] != digest(TRAIN_FILE)
            or audit['test_split_sha256'] != digest(TEST_FILE)
            or audit['checkpoint_sha256'] != digest(VAE_FILE)):
        raise ValueError('Source audit no longer matches the authoritative inputs')
    if [r['task_id'] for r in audit['tasks']] != train:
        raise ValueError('Audit must cover every permitted training task exactly once')
    import torch
    sys.path.insert(0, str(QP_ROOT))
    from vae.model import VAEModule, encode_batch
    from workload.workloads import get_workload_set, WorkloadSpec, OracleCodec
    from codec.codec import AliasesCodec
    from oracle.oracle import _decode_query
    workloads = get_workload_set('CEB_3K')
    torch.set_num_threads(2)
    model = VAEModule.load_from_checkpoint(str(VAE_FILE), map_location='cpu')
    model.eval().requires_grad_(False)
    records, means, offsets, excluded = [], [], [0], []
    total_changed, total_duplicates, raw_observations = 0, 0, 0
    for item in audit['tasks']:
        task = item['task_id']
        path = source_path(task)
        if digest(path) != item['source_sha256']:
            raise ValueError('Changed source history: ' + task)
        if item['unique_uncensored_plans'] < 2:
            excluded.append({'task_id': task, 'reason': 'fewer_than_two_distinct_completed_plans'})
            continue
        selected = {r['source_row']: r['runtime_seconds']
                    for r in item['first_uncensored_row_per_distinct_plan']}
        match = re.fullmatch(r'CEB_(\d+[A-Z])(\d+)', task)
        sql = QP_ROOT / 'workload/ceb-3k' / match[1].lower() / ((match[1] + match[2]).lower() + '.sql')
        spec = WorkloadSpec.from_definition(workloads.extract_query(sql.read_text(), task), OracleCodec.Aliases)
        codec = AliasesCodec(spec.all_tables)
        plans, raw_plans, rows, runtimes, duplicates = [], [], [], [], []
        seen = {}
        visited, changed = 0, 0
        with path.open(newline='') as stream:
            for index, row in enumerate(csv.DictReader(stream)):
                if index not in selected:
                    continue
                runtime = -float(row['train_y'])
                if float(row['censoring']) != 0 or runtime != selected[index]:
                    raise ValueError('Audit and source measurement disagree')
                visited += 1
                raw = ast.literal_eval(row['train_x'])
                canonical = codec.encode(codec.decode(spec.query_tables, raw.copy()))
                if _decode_query(spec, raw.copy()) != _decode_query(spec, canonical.copy()):
                    raise ValueError('Plan canonicalization changed the executed SQL: %s row %d' % (task, index))
                changed += int(raw != canonical)
                key = tuple(canonical)
                if key in seen:
                    duplicates.append(dict(source_row=index, kept_source_row=seen[key], runtime_seconds=runtime))
                    continue
                seen[key] = index
                plans.append(canonical)
                raw_plans.append(raw)
                rows.append(index)
                runtimes.append(runtime)
        if visited != len(selected) or digest(path) != item['source_sha256']:
            raise ValueError('Incomplete or concurrently changed source: ' + task)
        raw_observations += visited
        total_changed += changed
        total_duplicates += len(duplicates)
        if len(plans) < 2:
            excluded.append(dict(task_id=task, reason='fewer_than_two_distinct_completed_canonical_plans'))
            continue
        records.append(dict(task_id=task, family=item['family'], source_path=str(path),
                            source_sha256=item['source_sha256'], plans=plans, raw_plans=raw_plans,
                            sql_path=str(sql), sql_sha256=digest(sql),
                            canonicalized_rows=changed, canonical_duplicates=duplicates,
                            exact_sql_roundtrip_verified=True,
                            source_rows=rows, runtime_seconds=runtimes))
        if len(records) % 50 == 0:
            print(json.dumps({'canonicalized_tasks': len(records)}), flush=True)
    atomic_json(directory / 'canonical_tasks.json', records)
    unsupported = []
    for record in records:
        missing = set()
        for plan in record['plans']:
            for i in range(0, len(plan), 5):
                t1, a1, t2, a2, op = plan[i:i + 5]
                missing.update(token for token in ((t1, a1), (t2, a2), op) if token not in model.vocab)
        if missing:
            unsupported.append(dict(task_id=record['task_id'], family=record['family'],
                                    observations=len(record['plans']),
                                    missing_tokens=[repr(token) for token in sorted(missing, key=repr)]))
    compatibility = dict(
        status='blocked' if unsupported else 'passed', training_started=False,
        reason='canonical_aliases_outside_frozen_vae_vocabulary' if unsupported else None,
        source_tasks=len(records), observations=sum(len(r['plans']) for r in records),
        raw_distinct_encoded_observations=raw_observations, canonicalized_rows=total_changed,
        duplicate_canonical_rows=total_duplicates, excluded_tasks=excluded,
        unsupported_tasks=unsupported, supported_tasks=len(records) - len(unsupported),
        unsupported_observations=sum(r['observations'] for r in unsupported),
        canonical_tasks_sha256=digest(directory / 'canonical_tasks.json'),
        vae_sha256=digest(VAE_FILE), audit_sha256=digest(audit_path))
    atomic_json(directory / 'representation_report.json', compatibility)
    if audit_only:
        print(json.dumps({k: v for k, v in compatibility.items()
                          if k not in ('unsupported_tasks', 'excluded_tasks')}), flush=True)
        return
    if unsupported:
        raise ValueError('%d queries need unsupported VAE tokens; see representation_report.json. '
                         'No source exclusion or VAE replacement has been selected.' % len(unsupported))
    for record in records:
        plans, task = record['plans'], record['task_id']
        task_means = []
        with torch.inference_mode():
            for start in range(0, len(plans), batch_size):
                tokens = encode_batch(plans[start:start + batch_size], model.vocab)
                mu, _ = model.model.encode(tokens)
                task_means.append(mu.cpu().numpy())
            check_tokens = encode_batch(plans[:2], model.vocab)
            first, _ = model.model.encode(check_tokens)
            second, _ = model.model.encode(check_tokens)
            if not torch.equal(first, second):
                raise RuntimeError('Encoder means are not deterministic')
        features = np.concatenate(task_means)
        if features.shape != (len(plans), 64) or not np.isfinite(features).all():
            raise ValueError('Invalid encoder output: ' + task)
        means.append(features)
        offsets.append(offsets[-1] + len(plans))
        if len(means) % 50 == 0:
            print(json.dumps({'encoded_tasks': len(means), 'observations': offsets[-1]}), flush=True)
    raw = np.concatenate(means).astype(np.float32)
    lower = np.minimum(-4.0, raw.min(axis=0)).astype(float)
    upper = np.maximum(4.0, raw.max(axis=0)).astype(float)
    normalized = scale_latents(raw, lower, upper).astype(np.float64)
    atomic_json(directory / 'tasks.json', records)
    with (directory / 'features.npz.tmp').open('wb') as stream:
        np.savez_compressed(stream, encoder_means=raw, X=normalized, offsets=np.asarray(offsets))
    os.replace(directory / 'features.npz.tmp', directory / 'features.npz')
    manifest = dict(format_version=1, domain='ceb_query_runtime', status='complete',
                    policy=POLICY, representation='frozen_vae_encoder_mean_64',
                    train_split_sha256=digest(TRAIN_FILE), test_split_sha256=digest(TEST_FILE),
                    vae_path=str(VAE_FILE), vae_sha256=digest(VAE_FILE),
                    vae_variant='bayesqo_author_frozen',
                    vae_pretraining_provenance='external BayesQO release; not established as train800-only',
                    historical_vae_path=audit.get('historical_checkpoint'),
                    historical_vae_sha256=audit.get('historical_checkpoint_sha256'),
                    audit_path=str(audit_path.resolve()), audit_sha256=digest(audit_path),
                    source_tasks=[r['task_id'] for r in records], excluded_tasks=excluded,
                    observations=len(raw), lower=lower.tolist(), upper=upper.tolist(),
                    normalization='linear to [0,1]; bounds include [-4,4] and all eligible source means',
                    original_proposal_latents=False, complete_proposal_trajectories=False,
                    plan_representation='oracle_canonical_alias_encoding',
                    codec_sha256=digest(QP_ROOT / 'codec/codec.py'),
                    oracle_sha256=digest(QP_ROOT / 'oracle/oracle.py'),
                    raw_distinct_encoded_observations=raw_observations,
                    canonicalized_rows=total_changed, duplicate_canonical_rows=total_duplicates,
                    files={name: digest(directory / name) for name in ('tasks.json', 'features.npz')})
    atomic_json(directory / 'manifest.json', manifest)
    load_corpus(directory)
    print(json.dumps({'status': 'complete', 'sources': len(records), 'observations': len(raw),
                      'excluded': len(excluded), 'directory': str(directory)}), flush=True)


def export_windows(directory, destination, size):
    manifest, tasks = load_corpus(directory)
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / 'windows.jsonl'
    if output.exists():
        raise ValueError('Window export already exists; select a new directory')
    count = 0
    with output.with_suffix('.tmp').open('w') as stream:
        for task, record in tasks.items():
            for ordinal, indices in enumerate(window_indices(len(record['X']), size)):
                window = dict(task_id=task, window_id='%s/%04d' % (task, ordinal),
                              family=record['family'], ordered=True, shuffled=False,
                              behavior_policy='frozen_vae_latent_turbo',
                              reconstructed_censored_filtered=True,
                              source_rows=[record['source_rows'][i] for i in indices],
                              X=record['X'][indices].tolist(),
                              y_maximize=record['y_maximize'][indices].tolist())
                stream.write(json.dumps(window, allow_nan=False) + '\n')
                count += 1
    os.replace(output.with_suffix('.tmp'), output)
    atomic_json(destination / 'manifest.json', dict(
        domain='ceb_query_runtime', status='windows_prepared', training_started=False,
        corpus=str(directory.resolve()), corpus_sha256=digest(directory / 'manifest.json'),
        windows_sha256=digest(output), source_tasks=manifest['source_tasks'], windows=count,
        max_observations=size, inference_history_limit=size,
        tail_policy='singleton tail overlaps one preceding observation',
        validation_partition='not_selected', representation=manifest['representation']))
    print(json.dumps({'windows': count, 'max_observations': size, 'training_started': False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--windows', type=Path)
    parser.add_argument('--plan-audit-only', action='store_true')
    args = parser.parse_args()
    if args.plan_audit_only and args.windows:
        parser.error('Plan audit does not produce the latent features required by OptFormer windows')
    with run_lock(args.corpus):
        if not (args.corpus / 'manifest.json').exists():
            if args.audit is None or args.batch_size < 1:
                parser.error('An audit and positive encoder batch size are required')
            export(args.audit, args.corpus, args.batch_size, args.plan_audit_only)
        else:
            load_corpus(args.corpus)
        if args.windows:
            export_windows(args.corpus, args.windows, 32)


if __name__ == '__main__':
    main()
