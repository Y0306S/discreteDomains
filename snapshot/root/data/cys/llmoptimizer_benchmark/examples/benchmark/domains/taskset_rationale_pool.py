"""Disjoint rationale assignments and verification before a single-writer merge."""

import json
from pathlib import Path

from taskset_rationale_validation import record_error


def validate_pool(plan, event_count):
    if 'local' not in plan['workers']:
        raise ValueError('The rationale pool requires a local worker')
    indices = [i for worker in plan['workers'].values() for i in worker['indices']]
    if len(indices) != len(set(indices)) or any(not isinstance(i, int) or i < 0 or i >= event_count for i in indices):
        raise ValueError('Rationale pool assignments overlap or are outside the corpus')
    if sorted(indices) != sorted(plan['pending_indices']):
        raise ValueError('Rationale pool assignments do not cover the pending examples')


def shard_path(campaign, worker_id):
    return Path(campaign) / 'rationale_shards' / (worker_id + '.jsonl')


def worker_status_path(campaign, worker_id):
    return Path(campaign) / ('rationale_worker_' + worker_id + '.json')


def pool_progress(campaign, plan, pool_sha256):
    statuses = {}
    for worker_id, worker in plan['workers'].items():
        path = worker_status_path(campaign, worker_id)
        status = json.loads(path.read_text()) if path.exists() else {}
        if status.get('pool_sha256') != pool_sha256:
            status = dict(status='pending', completed=0, total=len(worker['indices']))
        statuses[worker_id] = status
    return dict(completed=plan['retained_examples'] + sum(s.get('completed', 0) for s in statuses.values()),
                total=plan['total_examples'], workers=statuses, model_instances=len(statuses))


def checked_shard(rows, assigned_indices, events, tokenizer, event_digest):
    latest = {row['event_index']: row for row in rows}
    if set(latest) != set(assigned_indices):
        raise ValueError('Rationale shard has missing or unexpected examples')
    for i, row in latest.items():
        if row['event_sha256'] != event_digest(events[i]):
            raise ValueError('Rationale shard uses a different source event')
        error = record_error(row, tokenizer)
        if error:
            raise ValueError(f'Invalid rationale shard event {i}: {error}')
    return latest
