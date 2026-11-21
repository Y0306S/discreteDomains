#!/usr/bin/env python3
"""Validate the released CEB VAE and recover the original CEB_11A21 Bao input.

Writes separate review artifacts only. Does not change active assets, histories,
or checkpoints, and never executes SQL or starts training.
"""

import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
import sys

from query_transfer_common import QP_ROOT, HISTORICAL_VAE_FILE as VAE_FILE, atomic_json, digest, read_split


RELEASE = 'fa8a58f6d0e3f424db945d393a89431fcdc3464c'
AUTHOR_SHA = '83a4942701bffaaf8311887e530e3157b6843134e52e9f53d59f957c24aebab1'
BAO_SHA = 'caaec38f3f68684ba2494c165f70c41107a396d4ceea1565c716efbeb37ee375'
BAYESQO = Path('/root/data/cys/BayesQO')
TASK = 'CEB_11A21'


def selected_rows(path, tasks):
    result = {task: [] for task in tasks}
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        for row in reader:
            if row['query_name'] in result:
                result[row['query_name']].append(row)
    return fields, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--local-bao', type=Path,
                        default=QP_ROOT / 'initialization_data/ceb_3k_bao_initialization.csv',
                        help='Use the saved pre-restoration backup to reproduce the corruption audit')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory to preserve previous audits')

    author = BAYESQO / 'bayes_lqo/vae/CEB_64.ckpt'
    original = BAYESQO / '.git/lfs/objects' / BAO_SHA[:2] / BAO_SHA[2:4] / BAO_SHA
    local = args.local_bao
    if digest(author) != AUTHOR_SHA or digest(original) != BAO_SHA:
        raise ValueError('Released asset digest mismatch')
    pointer = subprocess.check_output([
        'git', '-C', str(BAYESQO), 'show', RELEASE +
        ':bayes_lqo/initialization_data/ceb_3k_bao_initialization.csv',
    ], text=True)
    if 'oid sha256:' + BAO_SHA not in pointer.splitlines():
        raise ValueError('Bao data does not match the pinned release')
    code = subprocess.check_output([
        'git', '-C', str(BAYESQO), 'show', RELEASE + ':bayes_lqo/vae/CEB_64.ckpt',
    ])
    import hashlib
    if hashlib.sha256(code).hexdigest() != AUTHOR_SHA:
        raise ValueError('Author VAE does not match the pinned release')
    del code

    train, _ = read_split()
    fields, original_rows = selected_rows(original, [TASK])
    _, local_rows = selected_rows(local, [TASK])
    expected = original_rows[TASK]
    current = local_rows[TASK]
    if TASK not in train or len(expected) != 49 or len(current) != 49:
        raise ValueError('Expected a training query with 49 original and local rows')
    by_id = {row['plan_id']: row for row in expected}
    if len(by_id) != 49 or set(by_id) != {row['plan_id'] for row in current}:
        raise ValueError('Original and local Bao plan IDs differ')
    changes = [dict(row_index=i, local=row, original=by_id[row['plan_id']])
               for i, row in enumerate(current) if row != by_id[row['plan_id']]]
    if len(changes) != 1 or changes[0]['local']['plan_id'] != '95826':
        raise ValueError('Unexpected changes to CEB_11A21; review before recovery')
    repaired = [by_id[row['plan_id']] for row in current]

    import torch
    sys.path.insert(0, str(QP_ROOT))
    from vae.model import VAEModule, encode_batch
    from workload.workloads import get_workload_set, WorkloadSpec, OracleCodec
    from codec.codec import AliasesCodec
    from oracle.oracle import _decode_query, plan_has_crossjoin
    torch.set_num_threads(2)
    torch.manual_seed(0)
    active_model = VAEModule.load_from_checkpoint(str(VAE_FILE), map_location='cpu')
    author_model = VAEModule.load_from_checkpoint(str(author), map_location='cpu')
    for model in (active_model, author_model):
        model.eval().requires_grad_(False)
    with (args.corpus / 'canonical_tasks.json').open() as stream:
        records = json.load(stream)
    if not {record['task_id'] for record in records}.issubset(train):
        raise ValueError('Canonical source corpus contains a non-training task')
    compatibility = {}
    for name, model in [('active', active_model), ('author', author_model)]:
        missing = []
        for record in records:
            for plan in record['plans']:
                try:
                    encode_batch([plan], model.vocab)
                except (KeyError, ValueError) as exc:
                    missing.append(dict(task_id=record['task_id'], error=str(exc)))
                    break
        compatibility[name] = dict(
            vocabulary_size=len(model.vocab),
            aliases=sorted({key[1] for key in model.vocab if isinstance(key, tuple)}),
            incompatible_tasks=missing, compatible_tasks=len(records) - len(missing))
    if compatibility['author']['incompatible_tasks']:
        raise ValueError('Author VAE cannot encode all canonical training plans')

    workloads = get_workload_set('CEB_3K')
    _, examples = selected_rows(local, ['CEB_6A1', 'CEB_7A12'])
    examples[TASK] = repaired
    checks = []
    for task, rows in examples.items():
        match = re.fullmatch(r'CEB_(\d+[A-Z])(\d+)', task)
        sql = QP_ROOT / 'workload/ceb-3k' / match[1].lower() / (task[4:].lower() + '.sql')
        spec = WorkloadSpec.from_definition(workloads.extract_query(sql.read_text(), task), OracleCodec.Aliases)
        codec = AliasesCodec(spec.all_tables)
        raw = [[int(value) for value in row['encoded_plan'].split(',')] for row in rows]
        canonical = [codec.encode(codec.decode(spec.query_tables, plan.copy())) for plan in raw]
        if any(_decode_query(spec, a.copy()) != _decode_query(spec, b.copy())
               for a, b in zip(raw, canonical)):
            raise ValueError('Canonicalization changed executable SQL')
        models = [('author', author_model)]
        if task == TASK:
            models.append(('active', active_model))
        for name, model in models:
            # Restored Bao rows must encode directly, without codec padding/repair.
            tokens = encode_batch(raw, model.vocab)
            with torch.inference_mode():
                mu, _ = model.model.encode(tokens)
                generated = model.sample(z=mu[:1])[0]
            if mu.shape != (49, 64) or not torch.isfinite(mu).all():
                raise ValueError('Invalid warm-start latent vectors')
            tree = codec.decode(spec.query_tables, generated.copy())
            expected_tables = {(table, alias) for table, count in spec.query_tables
                               for alias in range(1, count + 1)}
            if set(tree.tables_aliases()) != expected_tables:
                raise ValueError('Decoded sample changed query table coverage')
            checks.append(dict(
                task_id=task, checkpoint=name, encoded_bao_rows=len(raw),
                latent_shape=list(mu.shape), finite=True,
                exact_sql_canonical_roundtrip=True,
                decoded_sample_complete_table_coverage=True,
                sql_sha256=digest(sql)))
        if task == TASK:
            restored = raw[changes[0]['row_index']]
            if len(restored) != 45 or plan_has_crossjoin(spec, restored.copy()):
                raise ValueError('Recovered record is malformed or has a cross join')

    args.output.mkdir(parents=True)
    repaired_path = args.output / (TASK + '_bao_initialization.repaired.csv')
    with repaired_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(repaired)
    report = dict(
        status='passed', production_assets_changed=False, historical_runs_repaired=False,
        database_calls=0, training_started=False, checkpoint_changed=False,
        source_release=RELEASE, author_checkpoint=str(author), author_sha256=AUTHOR_SHA,
        active_checkpoint=str(VAE_FILE), active_sha256=digest(VAE_FILE),
        upstream_bao_path=str(original), upstream_bao_sha256=BAO_SHA,
        local_bao_path=str(local), local_bao_sha256=digest(local),
        source_tasks=len(records), source_observations=sum(len(r['plans']) for r in records),
        canonical_tasks_sha256=digest(args.corpus / 'canonical_tasks.json'),
        vocabulary_audit=compatibility, component_checks=checks,
        recovered_task=TASK, changed_records=changes, repaired_rows=len(repaired),
        repaired_file=str(repaired_path), repaired_sha256=digest(repaired_path),
        recovered_record_is_censored=True,
        author_pretraining_train800_only='not_established',
        historical_run_requires_rerun=True)
    atomic_json(args.output / 'report.json', report)
    print(json.dumps({key: report[key] for key in (
        'status', 'source_tasks', 'source_observations', 'repaired_rows',
        'production_assets_changed', 'database_calls', 'training_started')}), flush=True)


if __name__ == '__main__':
    main()
