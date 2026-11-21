#!/usr/bin/env python3
"""Refresh test200 starting outcomes and spot-check every train800 task's oracle."""

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent
CAMPAIGN=Path('/root/data/logs/peptide_bo_train800_20261027_v5b')
RELEASE=Path('/root/data/logs/peptide_bo_release_20261027_v5')


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):value.update(block)
    return value.hexdigest()


def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server',default='http://127.0.0.1:18871')
    parser.add_argument('--cpu',type=int,default=16)
    parser.add_argument('--train-audit',action='store_true')
    args=parser.parse_args()
    os.sched_setaffinity(0,{args.cpu})
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    sys.path.insert(0,str(RELEASE))
    from mindware.domains.protein.peptide_rpc import RemoteObjective
    from mindware.domains.protein.verified_bo import similarity,validate_sequence
    import torch
    torch.set_num_threads(1)
    recipe=json.loads((CAMPAIGN/'recipe.json').read_text())
    pools=json.loads(Path(recipe['prepared_pool']).read_text())
    mode='train800_spotcheck' if args.train_audit else 'test200_rescore'
    state_path=ROOT/(mode+'_state.json')
    identity=dict(recipe_sha256=digest(CAMPAIGN/'recipe.json'),pool_sha256=digest(recipe['prepared_pool']),
                  script_sha256=digest(__file__),mode=mode,server=args.server)
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(identity=identity,scored={},tasks={})
    if state['identity']!=identity:raise ValueError('Rescore recovery identity changed')
    client=RemoteObjective(SimpleNamespace(vae_forward=None),args.server,mode+':'+identity['pool_sha256'])
    tasks=recipe['train'] if args.train_audit else recipe['test']
    train_results={r['task']:r for r in json.loads((ROOT/'train800_audit.json').read_text())['tasks']} if args.train_audit else None
    if train_results is not None and any(not r['passed'] for r in train_results.values()):
        raise ValueError('Train integrity audit must pass before outcome spot checks')
    started=time.time();mismatches=[];relative=[];absolute=[];fresh_calls=0
    for task in tasks:
        pool=pools[task]
        reference=pool['reference']
        if args.train_audit:
            entries=train_results[task]['rescore_samples']
            sequences=[row['sequence'] for row in entries]
            expected=[row['score'] for row in entries]
        else:
            sequences=pool['sequences'];expected=pool['scores']
        for seq in sequences:
            validate_sequence(seq)
            if not args.train_audit and similarity(seq,reference)<recipe['similarity']-1e-8:
                raise ValueError('Invalid test starting sequence: '+task)
        reference_errors=[]
        if not args.train_audit:
            try:validate_sequence(reference)
            except ValueError:reference_errors.append('reference_not_valid_evaluated_candidate')
        requested=sequences+([reference] if not args.train_audit and not reference_errors else [])
        missing=list(dict.fromkeys(seq for seq in requested if seq not in state['scored']))
        for start in range(0,len(missing),64):
            batch=missing[start:start+64]
            values=client.oracle(batch).tolist()
            if len(values)!=len(batch) or any(not math.isfinite(y) or y>=0 for y in values):
                raise ValueError('Invalid fresh APEX outcome')
            state['scored'].update(zip(batch,map(float,values)))
            fresh_calls+=len(batch)
            write(state_path,state)
        scores=[state['scored'][seq] for seq in sequences]
        task_mismatches=[]
        for index,(seq,old,new) in enumerate(zip(sequences,expected,scores)):
            delta=abs(new-old);rel=delta/max(abs(old),1e-12)
            absolute.append(delta);relative.append(rel)
            if not math.isclose(new,old,rel_tol=1e-4,abs_tol=1e-4):
                difference=dict(task=task,index=index,sequence=seq,old_score=old,new_score=new,
                                absolute_difference=delta,relative_difference=rel)
                task_mismatches.append(difference);mismatches.append(difference)
        state['tasks'][task]=dict(scores=scores,reference_candidate_errors=reference_errors,
                                 reference_score=state['scored'].get(reference) if not reference_errors else None,
                                 mismatches=task_mismatches)
        write(state_path,state)
        if len(state['tasks'])%10==0 or task==tasks[-1]:
            progress=dict(mode=mode,completed=len(state['tasks']),total=len(tasks),
                          unique_sequences_scored=len(state['scored']),significant_mismatches=len(mismatches),
                          seconds=time.time()-started)
            write(ROOT/(mode+'_progress.json'),progress)
            print(json.dumps(progress),flush=True)
    report=dict(passed=not mismatches,identity=identity,tasks=len(tasks),
                compared_outcomes=len(relative),unique_sequences_scored=len(state['scored']),
                fresh_calls_this_execution=fresh_calls,
                maximum_absolute_difference=max(absolute,default=0),
                maximum_relative_difference=max(relative,default=0),
                numeric_tolerance=dict(relative=1e-4,absolute=1e-4),
                significant_mismatches=mismatches,
                references_not_valid_candidates=[task for task,value in state['tasks'].items() if value['reference_candidate_errors']],
                seconds=time.time()-started)
    if not args.train_audit:
        verified={}
        for task in tasks:
            value=deepcopy(pools[task]);value['original_scores']=value['scores']
            value['scores']=state['tasks'][task]['scores']
            value['score_revalidation']=dict(identity=identity,tolerance=report['numeric_tolerance'])
            verified[task]=value
        write(ROOT/'test200_revalidated_initialization.json',verified)
        report['revalidated_pool_sha256']=digest(ROOT/'test200_revalidated_initialization.json')
    write(ROOT/(mode+'_report.json'),report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('identity','significant_mismatches')},allow_nan=False),flush=True)


if __name__=='__main__':main()
