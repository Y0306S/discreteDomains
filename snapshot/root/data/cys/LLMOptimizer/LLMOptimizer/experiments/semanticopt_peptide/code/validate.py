"""Offline split/constraint audit plus meaningful interruption/replay checks."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
from common import *
import evaluate
from protocol import peptide_measure,select_candidate

def recovery_checks():
    calls=[]
    original=load(ROOT/'recipe.json')
    recipe=copy.deepcopy(original)
    recipe['peptide'].update(test=['test_task'],proposals=3,smoke_task='train_task')
    reference='AAAAAAAA'
    initial=[dict(configuration=dict(sequence=reference),objective=10.,status='success',censored=False,
        feasible=True,similarity=1.)]
    options=['AAAAAAAK','AAAAAAAR','AAAAAAAL']
    class Client:
        def __init__(self,*args):self.state={'counter':0}
        def import_state(self,state):
            if state:self.state=state
        def ensure_loaded(self):pass
        def complete_messages(self,*args,**kwargs):
            index=self.state['counter']//2
            self.state['counter']+=1
            return canonical(dict(next_configuration=dict(sequence=options[index]),mean=5.,std=1.))
    def oracle(config):
        calls.append(config)
        return dict(configuration=config,objective=5.-len(calls),status='success',censored=False,feasible=True)
    def setup(*args):
        def validate(config):
            if not peptide_measure(config['sequence'],reference)['feasible']:raise ValueError('infeasible')
        return dict(reference_sequence=reference),initial,validate,oracle
    real_status=evaluate.status
    interrupted=False
    def interrupt_status(path,**value):
        nonlocal interrupted
        real_status(path,**value)
        if not interrupted and value.get('evaluations')==1:
            interrupted=True;raise InterruptedError('Simulated exit after committed evaluation')
    with tempfile.TemporaryDirectory(prefix='semanticopt_resume_',dir=ROOT/'logs') as tmp:
        root=Path(tmp);atomic(root/'recipe.json',recipe)
        with patch.object(evaluate,'ROOT',root),patch.object(evaluate,'verify',lambda:recipe),\
             patch.object(evaluate,'peptide_setup',setup),\
             patch('qwen_inference_transport.SharedQwenClient',Client):
            with patch.object(evaluate,'status',interrupt_status):
                try:evaluate.run('peptide','test_task',0)
                except InterruptedError:pass
            evaluate.run('peptide','test_task',0)
            done=load(root/'peptide/runs/test_task/status.json')
            assert done['evaluations']==3 and len(calls)==3 and len({canonical(c) for c in calls})==3
            evaluate.run('peptide','test_task',0)
            assert len(calls)==3
            db=evaluate.connect(root/'peptide/runs/test_task')
            with db:evaluate.put(db,'pending',dict(status='in_flight',answer={},number=3))
            # Explicitly incomplete oracle calls cannot silently consume a second call.
            recipe['peptide']['proposals']=4
            try:evaluate.run('peptide','test_task',0)
            except RuntimeError as exc:assert 'Unresolved oracle call' in str(exc)
            else:raise AssertionError('Ambiguous pending call replayed')
            assert len(calls)==3
    return dict(committed_resume_without_rescore=True,completed_task_idempotent=True,
        ambiguous_call_blocks_replay=True,exactly_one_oracle_call_per_selected_candidate=True)

def main():
    recipe=load(ROOT/'recipe.json');checks=recovery_checks()
    for domain in ['query','peptide']:
        settings=recipe[domain]
        assert len(set(settings['test']))==200 and len(set(settings['train']))==800
        assert not set(settings['test'])&set(settings['train'])
        assert settings['smoke_task'] in settings['train']
        assert model_identity(settings['base'],settings['adapter'])==settings['identity']
    checks['train_test_disjoint_and_exact200']=True
    checks['both_adapter_identities_match_completed_training']=True
    # Validate the full fixed query initialization through the production codec.
    for i,task in enumerate(recipe['query']['test']):
        context,initial,validate,_=evaluate.query_setup(task,recipe['query'],0)
        assert len(initial)==49
        for row in initial:validate(row['configuration'])
        try:validate(dict(plan=[999999,1,0,1,2]))
        except ValueError:pass
        else:raise AssertionError('Invalid table symbols accepted')
        if i%50==0:print('query validation',i,flush=True)
    checks['query_all200_initializations_canonical_and_valid']=True
    pool=load(recipe['peptide']['pool'])
    for task in recipe['peptide']['test']:
        data=pool[task]
        assert len(data['sequences'])==len(data['scores'])==100
        assert all(peptide_measure(s,data['reference'])['feasible'] for s in data['sequences'])
    checks['peptide_all200_initializations_feasible']=True
    # CPU/GPU kernels differ numerically. Measure the discrepancy on 20 archived
    # observations, keeping the predeclared 0.1% relative tolerance explicit.
    context,initial,validate,oracle=evaluate.peptide_setup(recipe['peptide']['test'][0],recipe['peptide'])
    agreement=[]
    for task in recipe['peptide']['test'][::10]:
        data=pool[task];config=dict(sequence=data['sequences'][0])
        # This task's feasibility is already checked above; the MIC oracle is task independent.
        actual=oracle(config)['objective'];expected=-float(data['scores'][0])
        agreement.append(dict(task=task,cpu=actual,archived=expected,
            relative_error=abs(actual-expected)/max(1.,abs(expected))))
    atomic(ROOT/'cpu_oracle_agreement.json',dict(samples=agreement,tolerance_relative=.001,
        maximum_relative_error=max(r['relative_error'] for r in agreement)))
    assert all(r['relative_error']<=.001 for r in agreement),agreement
    checks['cpu_APEX_agrees_with_archived_GPU_within_0p1_percent']=True
    assert select_candidate([dict(next_configuration=dict(sequence='A'),mean=8,std=0),
        dict(next_configuration=dict(sequence='B'),mean=1,std=0)],10,lambda c:None)['next_configuration']=={'sequence':'B'}
    checks['EI_selection_prefers_greater_expected_improvement']=True
    assert all(checks.values())
    atomic(ROOT/'offline_preflight.json',dict(passed=True,at=time.time(),checks=checks,
        heldout_production_evaluations=0,validation_rescores=len(agreement),recipe_sha256=digest(ROOT/'recipe.json')))
    print(canonical(dict(passed=True,checks=checks)),flush=True)

if __name__=='__main__':main()
