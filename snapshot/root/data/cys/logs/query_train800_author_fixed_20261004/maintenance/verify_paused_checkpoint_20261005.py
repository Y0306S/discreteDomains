#!/usr/bin/env python3
"""Validate a paused QueryOpt checkpoint on CPU without evaluations or updates."""

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path('/root/data/cys/logs/query_train800_author_fixed_20261004')
OUT = ROOT / 'maintenance/LLMOptimizer3_gpu_change_20261005'
sys.path.insert(0, str(ROOT / 'code'))
import run_query_ceb_baselines as runner

runner.native_path(ROOT / 'native')
# The production runner stores this class under __main__ in its pickle.
Engine = runner.Engine


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    import torch

    record = json.loads((OUT / 'worker_paused.json').read_text())
    directory = Path(record['checkpoint']).parent
    for relative, expected in record['artifacts_sha256'].items():
        assert digest(ROOT / relative) == expected, relative
    saved = torch.load(directory / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert saved['meta']['task_id'] == record['task']
    assert saved['meta']['method'] == 'lolbo'
    assert saved['total'] == 1049 and len(saved['initial']) == 49
    assert saved['engine'].method == 'lolbo'
    rows = [json.loads(path.read_text()) for path in
            sorted(directory.glob('proposal_[0-9][0-9][0-9][0-9].json'))]
    assert len(rows) == record['saved_proposals']
    assert [row['proposal_index'] for row in rows] == list(range(len(rows)))
    completed, pending = saved['completed'], saved['pending'] or []
    assert rows[:len(completed)] == completed
    assert len(completed) <= len(rows) <= len(completed) + len(pending)
    for offset, row in enumerate(rows[len(completed):]):
        assert row['proposal'] == pending[offset]
    assert all(not row['censored'] or row['runtime'] > 0 for row in rows)
    assert not any(not path.with_name(path.name.replace('.pending.json', '.json')).exists()
                   for path in directory.glob('proposal_*.pending.json'))
    model = saved['engine'].state.model
    parameters = dict(model.named_parameters())
    assert parameters and all(torch.isfinite(value).all().item() for value in parameters.values())
    finite_tensors, unbounded_constraints = 0, []
    for name, value in model.state_dict().items():
        if torch.isfinite(value).all().item():
            finite_tensors += 1
            continue
        # GPyTorch's positive constraints legitimately have an infinite upper bound.
        assert name.endswith('_constraint.upper_bound') and torch.isposinf(value).all().item(), name
        unbounded_constraints.append(name)
    assert saved['rng']['python'] and saved['rng']['numpy'] and saved['rng']['torch'].numel()
    assert not torch.cuda.is_initialized()
    summary = runner.write_history(directory, saved['meta'], saved['initial'], rows,
                                   saved['total'], 'paused')
    state_path = ROOT / 'workers/LLMOptimizer3.w00.json'
    state = json.loads(state_path.read_text())
    assert state['status'] == 'paused' and state['task'] == record['task']
    state['summary'] = summary
    runner.atomic_json(state_path, state)
    result = {'task': record['task'], 'checkpoint_load_on_cpu': 'passed',
              'checkpoint_completed_proposals': len(completed),
              'checkpoint_pending_proposals': len(pending), 'durable_saved_proposals': len(rows),
              'pending_results_already_saved': len(rows) - len(completed),
              'unresolved_database_calls': 0, 'finite_model_tensors': finite_tensors,
              'finite_parameter_tensors': len(parameters),
              'valid_unbounded_constraint_buffers': unbounded_constraints,
              'rng_state_present': True, 'gpu_initialized': False,
              'evaluations_run': 0, 'model_updates_run': 0,
              'checkpoint_sha256': digest(directory / 'checkpoint.pt'),
              'history_sha256': digest(directory / 'history.json')}
    runner.atomic_json(OUT / 'checkpoint_verified.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
