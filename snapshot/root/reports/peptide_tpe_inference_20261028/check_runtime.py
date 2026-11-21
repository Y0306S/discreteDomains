#!/usr/bin/env python3
"""Real inference equivalence and scored-pending recovery for infrastructure-only TPE."""

import hashlib
import http.client
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace
import urllib.request

os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
os.sched_setaffinity(0, {12})
TOOLS = Path('/root/data/cys/llmoptimizer_benchmark/tools')
sys.path.insert(0, str(TOOLS))
import peptide_test200 as adapter
from mindware.domains.protein.peptide_rpc import RemoteObjective, dumps, loads
from peptide_tpe_rpc import PersistentRemoteObjective
from peptide_test200 import bo, torch, protocol

ROOT = Path(__file__).resolve().parent
CAMPAIGN = Path('/root/data/logs/peptide_test200_baselines_20261027_v2/tpe')
SERVER = 'http://127.0.0.1:18876'
WRAPPER = TOOLS / 'peptide_tpe_runtime.py'


def health(url=SERVER):
    with urllib.request.urlopen(url + '/', timeout=5) as response:
        return json.load(response)


def raw_request(url, **value):
    data = dumps(value)
    request = urllib.request.Request(url + '/rpc', data=data,
        headers={'X-Request-ID': hashlib.sha256(data).hexdigest()})
    with urllib.request.urlopen(request, timeout=120) as response:
        result = loads(response.read())
    if result.get('error') or result.get('missing_weights'):
        raise RuntimeError(str(result))
    return result


def check_inference(recipe):
    assert health()['skip_identical_weights'] and health()['http_keepalive']
    weights = torch.load(recipe['vae'], map_location='cpu', weights_only=False)
    bank = 'tpe_inference_equivalence_' + str(time.time_ns())
    version = hashlib.sha256(dumps(weights)).hexdigest()
    sequences = ['RRYYEQLEQASRKGNRGFRR', 'KLKLKLKLKL']
    clients = []
    before = health()
    try:
        for url in ['http://127.0.0.1:18871', SERVER]:
            raw_request(url, identity=bank, op='weights', version=version, weights=weights)
        z = raw_request(SERVER, identity=bank, op='encode_mean', version=version,
                        seed=123, sequences=sequences)['z']
        expected = raw_request('http://127.0.0.1:18871', identity=bank, op='decode',
                               version=version, seed=456, z=z)['sequences']
        objective = SimpleNamespace(vae=SimpleNamespace(state_dict=lambda: weights), vae_forward=None)
        client = PersistentRemoteObjective(objective, SERVER, bank)
        clients.append(client)
        actual = client.request(dict(op='decode', version=version, seed=456, z=z, test_nonce=1))
        assert actual['sequences'] == expected
        socket_id = id(client.connection.sock)
        for i in range(5):
            answer = client.request(dict(op='decode', version=version, seed=456, z=z, test_nonce=i + 2))
            assert answer['sequences'] == expected and id(client.connection.sock) == socket_id
        second_bank = bank + '_identical_weights'
        raw_request(SERVER, identity=second_bank, op='weights', version=version, weights=weights)
        raw_request(SERVER, identity=second_bank, op='encode_mean', version=version,
                    seed=123, sequences=sequences)
        same_weight_stats = health()
        assert same_weight_stats['weight_loads'] - before['weight_loads'] == 1
        assert same_weight_stats['weight_loads_skipped'] - before['weight_loads_skipped'] >= 6

        changed = {k: v.clone() for k, v in weights.items()}
        key = next(k for k, v in changed.items() if v.is_floating_point() and v.numel())
        changed[key].view(-1)[0] += 0.001
        raw_request(SERVER, identity=bank + '_changed', op='weights', version='changed-test', weights=changed)
        raw_request(SERVER, identity=bank + '_changed', op='encode_mean', version='changed-test',
                    seed=123, sequences=sequences)
        restored = client.request(dict(op='encode_mean', version=version, seed=123,
                                       sequences=sequences, test_nonce=99))['z']
        assert torch.equal(restored, z)
        assert health()['weight_loads'] - same_weight_stats['weight_loads'] == 2

        # Discard a processed reply once: retry must reuse the server's idempotent result.
        oracle_before = health()['oracle_calls']
        original_getresponse = client.connection.getresponse
        discarded = [False]

        def lose_reply():
            response = original_getresponse()
            if not discarded[0]:
                response.read()
                discarded[0] = True
                raise http.client.RemoteDisconnected('Injected lost reply')
            return response

        client.connection.getresponse = lose_reply
        value = client.request(dict(op='oracle', sequences=sequences[:1], nonce=123456))
        assert len(value['scores']) == 1 and health()['oracle_calls'] - oracle_before == 1
        assert health()['reply_cache_hits'] > before['reply_cache_hits']
        return dict(seeded_decode_matches_original=True, persistent_socket_reused=True,
                    identical_weights_loaded_once=True, changed_weights_reloaded=True,
                    restored_frozen_encoding_identical=True, lost_reply_without_double_scoring=True,
                    oracle_calls=1, server=health())
    finally:
        for client in clients:
            client.close()
        for url in ['http://127.0.0.1:18871', SERVER]:
            for identity in [bank, bank + '_identical_weights', bank + '_changed']:
                raw_request(url, identity=identity, op='release')


def clone_case(name, recipe, task):
    destination = ROOT / name
    directory = destination / 'runs' / task
    directory.mkdir(parents=True, exist_ok=False)
    source = CAMPAIGN / 'runs' / task / 'runtime.sqlite'
    with sqlite3.connect('file:' + str(source) + '?mode=ro', uri=True) as db:
        with sqlite3.connect(str(directory / 'runtime.sqlite')) as copy:
            db.backup(copy)
    db = adapter.connect(directory / 'runtime.sqlite')
    saved = adapter.get(db, 'state')
    diagnostic = dict(recipe, purpose='component_check', proposals=saved['progress']['proposals'] + 3)
    bo.atomic_json(destination / 'recipe.json', diagnostic)
    saved['identity'] = dict(format=adapter.TPE_FORMAT,
                            recipe_sha256=bo.digest(destination / 'recipe.json'), task=task)
    with db:
        adapter.put(db, 'state', saved)
    db.close()
    return destination, diagnostic


def run_case(case, task, wrapper, extra=()):
    command = [sys.executable, str(wrapper), 'run-task', '--campaign', str(case), '--task', task,
               '--cpu', '12', '--server', SERVER] + list(extra)
    with (ROOT / (case.name + '.log')).open('ab') as output:
        subprocess.run(command, env=os.environ.copy(), check=True, stdout=output, stderr=subprocess.STDOUT)


def inject_scored_pending(case, recipe, task):
    import optuna
    from optuna.distributions import FloatDistribution
    from optuna.trial import create_trial
    db = adapter.connect(case / 'runs' / task / 'runtime.sqlite')
    saved, records = adapter.get(db, 'state'), adapter.trials(db)
    study = optuna.create_study(direction='maximize', sampler=saved['sampler'])
    study.sampler._constraints_func = adapter.tpe_constraints
    distributions = {'z%03d' % i: FloatDistribution(*recipe['tpe_bounds']) for i in range(256)}
    for record in records:
        study.add_trial(create_trial(params=record['params'], distributions=distributions,
            value=record['score'], user_attrs={'constraints': record['constraints']},
            system_attrs={'constraints': record['constraints']}))
    state, backend = protocol.create_state(recipe, task, remote_url=SERVER)
    state.objective.xs_to_scores_dict = {r['row']['sequence']: r['score'] for r in records}
    bo.restore_rng(saved['rng'])
    trial = study.ask(fixed_distributions=distributions)
    with torch.no_grad():
        sequence = state.objective.vae_decode(torch.tensor(
            [[trial.params[k] for k in distributions]], dtype=torch.float32))[0]
    before = adapter.state_meta(study, saved['progress'], saved['identity'])
    scored = bo.score_batch(state.objective, [sequence])
    pending = dict(number=trial.number, params=trial.params, sequence=sequence,
                   proposal_seconds=0.0, oracle_seconds=0.0, scored=scored)
    with db:
        adapter.put(db, 'pending', pending)
        adapter.put(db, 'state', before)
    db.close()
    backend.close()
    return saved['progress']['events'] + 1


def main():
    torch.set_num_threads(1)
    recipe = adapter.verify(CAMPAIGN)
    source_sha = bo.digest(CAMPAIGN / 'recipe.json')
    inference = check_inference(recipe)
    task = 'pep_123'
    original_db_sha = bo.digest(CAMPAIGN / 'runs' / task / 'runtime.sqlite')
    baseline, diagnostic = clone_case('baseline', recipe, task)
    optimized, _ = clone_case('optimized', recipe, task)
    run_case(baseline, task, Path(adapter.__file__))
    run_case(optimized, task, WRAPPER, ('--stop-after-events', '1'))
    event = inject_scored_pending(optimized, diagnostic, task)
    replay = """import sys; from pathlib import Path; from types import SimpleNamespace
sys.path.insert(0, sys.argv[1]); import peptide_tpe_runtime as r; r.install(); a=r.adapter
a.torch.set_num_threads(1)
def bad(*args, **kwargs): raise RuntimeError('Recorded pending must not be rescored')
a.bo.score_batch=bad
a.run(SimpleNamespace(campaign=Path(sys.argv[2]), task=sys.argv[3], server=sys.argv[4], launch=False, stop_after_events=int(sys.argv[5])))
"""
    with (ROOT / 'pending_recovery.log').open('ab') as output:
        subprocess.run([sys.executable, '-c', replay, str(TOOLS), str(optimized), task, SERVER, str(event)],
                       check=True, stdout=output, stderr=subprocess.STDOUT, env=os.environ.copy())
    run_case(optimized, task, WRAPPER)
    identity = dict(format=adapter.TPE_FORMAT, recipe_sha256=bo.digest(baseline / 'recipe.json'), task=task)
    first = adapter.verify_tpe_archive(baseline / 'runs' / (task + '.tar.gz'), identity)
    second = adapter.verify_tpe_archive(optimized / 'runs' / (task + '.tar.gz'), identity)
    assert first['proposals'] == second['proposals'] == diagnostic['proposals']
    assert first['fresh_evaluations'] == second['fresh_evaluations']
    for a, b in zip(first['observations'], second['observations']):
        assert a['sequence'] == b['sequence'] and a['score'] == b['score']
        assert a['feasible'] == b['feasible'] and a['cached'] == b['cached']
    assert bo.digest(CAMPAIGN / 'recipe.json') == source_sha
    assert bo.digest(CAMPAIGN / 'runs' / task / 'runtime.sqlite') == original_db_sha
    report = dict(passed=True, inference=inference, checks=dict(
        resumed_proposals_identical=True, outcomes_identical=True, budget_accounting=True,
        scored_pending_without_rescore=True, archive=True, original_campaign_unchanged=True),
        new_rounds_per_case=3, maximum_fresh_oracle_calls=7,
        runtime_scope='Infrastructure only: frozen VAE and existing batch-one latent TPE',
        wrapper_sha256=bo.digest(WRAPPER), rpc_sha256=bo.digest(TOOLS / 'peptide_tpe_rpc.py'))
    bo.atomic_json(ROOT / 'validation.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
