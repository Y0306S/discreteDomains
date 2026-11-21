"""Numerical/oracle checks before queuing, real Qwen branch checks before launch."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from common import ROOT, atomic, config, read, sha, status, valid, verify
from backend import Backend, bo_candidates, edit_pool
from worker import Worker


def manifest_check(cfg):
    initial = read(ROOT/'initialization.json')
    assert len(cfg['tasks']) == len(set(cfg['tasks'])) == len(initial) == 200
    assert set(cfg['tasks']).isdisjoint(cfg['train_tasks']) and len(cfg['train_tasks']) == 800
    for task in cfg['tasks']:
        p = initial[task]
        assert len(p['sequences']) == len(set(p['sequences'])) == len(p['scores']) == 100
        assert all(valid(s,p['reference']) for s in p['sequences'])
        assert np.isfinite(p['scores']).all()
    return dict(test_tasks=200,train_tasks=800,initial_observations=20000,overlap=0)


class FakeBackend:
    def __init__(self):
        self.cache = {}
        self.calls = []

    def encode(self, sequences):
        import hashlib
        for s in sequences:
            seed = int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], 'little')
            self.cache[s] = np.random.RandomState(seed).normal(size=256)
        return np.stack([self.cache[s] for s in sequences])

    def score(self, sequences):
        self.calls.extend(sequences)
        return np.array([float(sum(map(ord,s)) % 100+1) for s in sequences])


def lifecycle_checks(cfg):
    """Test durable scoring/resume, duplicate refusal, and unresolved-call protection."""
    test = copy.deepcopy(cfg)
    test['fresh_budget'] = 3
    test['gp']['restarts'] = 0
    task = cfg['tasks'][0]
    initial = read(ROOT/'initialization.json')[task]
    new = edit_pool(initial['reference'], initial['sequences'][:3], 4, initial['sequences'], np.random.RandomState(91))
    with tempfile.TemporaryDirectory(prefix='bora-lifecycle-',dir=ROOT/'validation') as directory:
        backend = FakeBackend()
        w = Worker(task, directory, test, backend=backend)
        w.state['queue'] = [dict(sequence=s,source='test_fixture') for s in new[:3]]
        w.run(stop_after=1)
        assert len(backend.calls) == 1 and len(w.state['rows']) == 101
        expected_rng = w.rng.get_state()
        w.guard.close()
        restored = Worker(task,directory,test,backend=backend)
        assert np.array_equal(expected_rng[1],restored.rng.get_state()[1])
        restored.run()
        assert len(backend.calls) == 3 and len(set(backend.calls)) == 3
        assert read(Path(directory)/'result.json')['evaluations'] == 3
        restored.guard.close()
    with tempfile.TemporaryDirectory(prefix='bora-duplicate-',dir=ROOT/'validation') as directory:
        backend = FakeBackend()
        w = Worker(task,directory,test,backend=backend)
        w.state['queue'] = [dict(sequence=initial['sequences'][0],source='duplicate_fixture')]
        try:
            w.run(stop_after=1)
        except RuntimeError as exc:
            assert 'Duplicate' in str(exc)
        else:
            raise AssertionError('Duplicate oracle request admitted')
        assert not backend.calls
        w.state['pending']['status'] = 'in_flight'
        w.save()
        w.guard.close()
        restored = Worker(task,directory,test,backend=backend)
        try:
            restored.run(stop_after=1)
        except RuntimeError as exc:
            assert 'Unresolved' in str(exc)
        else:
            raise AssertionError('Unresolved oracle request replayed')
        assert not backend.calls
        restored.guard.close()
    return dict(resume_without_duplicate_calls=True,exact_budget=True,duplicate_refused=True,
        unresolved_oracle_call_not_replayed=True)


def queue_gate_checks(cfg):
    import importlib.util
    import tempfile
    from unittest.mock import patch
    spec=importlib.util.spec_from_file_location('bora_campaign_queue',ROOT/'code/scheduler.py')
    queue=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
    with tempfile.TemporaryDirectory(prefix='bora-gate-',dir=ROOT/'validation') as directory:
        p=Path(directory)
        modified=dict(cfg,predecessor=str(p))
        atomic(p/'status.json',dict(status='running',completed=9,evaluations=39950))
        assert not queue.predecessor_gate(modified,process_check=False)[0]
        atomic(p/'status.json',dict(status='failed'))
        assert not queue.predecessor_gate(modified,process_check=False)[0]
        atomic(p/'status.json',dict(status='complete'))
        atomic(p/'summary.json',dict(status='complete',audits_passed=False))
        assert not queue.predecessor_gate(modified,process_check=False)[0]
        atomic(p/'summary.json',dict(status='complete',audits_passed=True))
        atomic(p/'stop_policy.json',dict(tasks=['fixture']))
        atomic(p/'peptide/runs/fixture/status.json',dict(status='complete',evaluations=4000))
        import sqlite3
        db=sqlite3.connect(p/'peptide/runs/fixture/observations.sqlite')
        db.executescript('create table records(n integer primary key,value text);create table state(key text primary key,value text);')
        db.executemany('insert into records values (?,?)',[(i,'{}') for i in range(4000)])
        db.commit();db.close()
        assert queue.predecessor_gate(modified,process_check=False)[0]
        with patch.object(queue,'proc_identity',return_value=dict(pid=1,argv=[str(p/'control/controller.py')],start_ticks=1)):
            assert not queue.predecessor_gate(modified,process_check=True)[0]
    return dict(running_predecessor_blocks=True,failed_predecessor_blocks=True,
        unaudited_predecessor_blocks=True,live_predecessor_process_blocks=True,audited_complete_passes=True)


def offline(cfg):
    begun=time.monotonic()
    manifest=manifest_check(cfg)
    gates=queue_gate_checks(cfg)
    lifecycle=lifecycle_checks(cfg)
    task=cfg['tasks'][0]
    pool=read(ROOT/'initialization.json')[task]
    backend=Backend(cfg)
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    x=backend.encode(pool['sequences'])
    assert x.shape==(100,256) and np.isfinite(x).all()
    assert np.array_equal(x,backend.encode(pool['sequences']))
    measured=backend.score(pool['sequences'][:3])
    expected=-np.array(pool['scores'][:3])
    reference=read(ROOT/'evidence/existing_cpu_oracle_reference.json')
    assert reference['task']==task and reference['sequences']==pool['sequences'][:3]
    assert np.allclose(measured,reference['measured'],rtol=1e-5,atol=1e-4),(measured,reference)
    initial_score_relative_drift=(np.abs(measured-expected)/np.maximum(np.abs(expected),1e-12)).tolist()
    low=x.min(axis=0);scale=np.maximum(x.max(axis=0)-low,1e-6)
    normalized=2*(x-low)/scale-1
    rng=np.random.RandomState(cfg['seed'])
    gp=GaussianProcessRegressor(kernel=Matern(nu=2.5),alpha=1e-6,normalize_y=True,
        n_restarts_optimizer=5,random_state=rng).fit(normalized,pool['scores'])
    import torch
    torch.manual_seed(cfg['seed'])
    candidates,info=bo_candidates(backend,gp,normalized,np.array(pool['scores']),low,scale,
        pool['reference'],pool['sequences'],rng,cfg,5)
    assert len(candidates)==len(set(candidates))==5
    assert not set(candidates)&set(pool['sequences'])
    assert all(valid(s,pool['reference']) for s in candidates)
    result=dict(passed=True,manifest=manifest,queue_gates=gates,lifecycle=lifecycle,
        author_vae_frozen=all(not p.requires_grad for p in backend.vae.parameters()),
        oracle_agrees_with_existing_SemanticOPT_CPU=True,measured=measured.tolist(),
        shared_saved_initial_scores=expected.tolist(),CPU_vs_saved_initial_relative_drift=initial_score_relative_drift,
        full_config_acquisition=info,candidates=candidates,seconds=time.monotonic()-begun)
    status(ROOT/'evidence/offline_validation.json',**result)
    print(json.dumps(result,indent=2))


def live(cfg):
    from assistant import SequenceComment
    begun=time.monotonic()
    task=cfg['tasks'][0]
    directory=ROOT/'validation/live'/task
    backend=Backend(cfg)
    w=Worker(task,directory,cfg,backend=backend)
    data=pd.DataFrame([dict(sequence=r['sequence'],score=-r['mic'],iteration=i+1) for i,r in enumerate(w.state['rows'])])
    assistant=w.get_assistant()
    suggestions=assistant.suggest(data,[r['sequence'] for r in w.state['rows']])
    assert suggestions and all(valid(s,w.reference) for s in suggestions),'No real-model feasible novel suggestions'
    assert not set(suggestions)&{r['sequence'] for r in w.state['rows']}
    candidates=edit_pool(w.reference,[r['sequence'] for r in w.state['rows'][:10]],5,
        [r['sequence'] for r in w.state['rows']],np.random.RandomState(12))
    selected=assistant.select(data,candidates)
    assert selected and len(set(selected))==len(selected) and set(selected)<=set(candidates),'Real-model selection did not preserve offered sequences'
    invalid=SequenceComment(json.dumps(dict(comment='invalid',hypotheses=[dict(name='bad',rationale='fixture',confidence=.5,
        points=[dict(sequence='INVALID!!!')])])),1,assistant._experiment)
    assert not invalid.is_valid
    # Exercise actual worker/GP/APEX/checkpoint and a clean restart without consuming production budget.
    w.state['queue']=[dict(sequence=s,source='validation_real_assistant') for s in suggestions[:2]]
    w.run(stop_after=2)
    calls=w.state['physical_oracle_calls']
    w.guard.close()
    w=Worker(task,directory,cfg,backend=backend)
    w.run(stop_after=3)
    assert w.state['physical_oracle_calls']==calls+1
    w.guard.close()
    # A long-history prompt must summarize deliberately and remain in the service envelope.
    large=pd.concat([data.assign(iteration=lambda x:x.iteration+i*100) for i in range(40)],ignore_index=True)
    text=assistant._generate_comment_prompt(large)
    assert w.prompt_tokens(assistant._chat_history+[dict(role='user',content=text)]) <= cfg['max_input_tokens']
    result=dict(passed=True,real_Qwen_suggestions=len(suggestions),real_Qwen_selected=len(selected),
        exact_selection_membership=True,invalid_sequence_rejected=True,real_worker_evaluations=3,
        resume_without_rescoring=True,long_history_context_validated=True,seconds=time.monotonic()-begun)
    status(ROOT/'evidence/live_validation.json',**result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--live',action='store_true')
    args=parser.parse_args()
    cfg=verify()
    (live if args.live else offline)(cfg)
