#!/usr/bin/env python3
"""Read-only integrity, constraint, accounting and state audit of peptide V5."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tarfile
import time

CAMPAIGN = Path('/root/data/logs/peptide_bo_train800_20261027_v5b')
OUTPUT = Path(__file__).resolve().parent
ALPHABET = frozenset('ACDEFGHIKLMNPQRSTVWY')
RECIPE = None
POOLS = None
AUTHOR = None


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def candidate_errors(sequence, reference):
    from Levenshtein import distance
    errors = []
    if not sequence or len(sequence) > 30:
        errors.append('length')
    if set(sequence) - ALPHABET:
        errors.append('alphabet')
    if sequence.count('C') > 1:
        errors.append('cysteine_count')
    similarity = 1 - distance(sequence, reference) / len(reference)
    if similarity < 0.75 - 1e-8:
        errors.append('similarity')
    return errors, similarity


def pool_audit(recipe, pools, tasks):
    rows, problems = [], []
    for task in tasks:
        pool = pools[task]
        reference = recipe['references'][task]
        errors = []
        if pool['reference'] != reference:
            errors.append('reference_mismatch')
        if len(pool['sequences']) != 100 or len(pool['scores']) != 100:
            errors.append('pool_size')
        if len(set(pool['sequences'])) != 100:
            errors.append('duplicate_initialization')
        violations, similarities = [], []
        for index, (sequence, score) in enumerate(zip(pool['sequences'], pool['scores'])):
            invalid, similarity = candidate_errors(sequence, reference)
            if not math.isfinite(score) or score >= 0:
                invalid.append('nonnegative_or_nonfinite_score')
            if invalid:
                violations.append(dict(index=index, sequence=sequence, errors=invalid))
            similarities.append(similarity)
        reference_errors, _ = candidate_errors(reference, reference)
        mandatory = max(0, reference.count('C') - 1) + sum(a not in ALPHABET for a in reference)
        compatible = mandatory <= len(reference) // 4 and 1 <= len(reference) <= 30
        if not compatible:
            errors.append('no_decoder_compatible_neighborhood')
        row = dict(task=task, reference=reference, pool_size=len(similarities),
                   minimum_similarity=min(similarities), reference_candidate_errors=reference_errors,
                   mandatory_decoder_edits=mandatory, permitted_edits=len(reference)//4,
                   decoder_compatible_neighborhood=compatible, errors=errors, violations=violations)
        rows.append(row)
        if errors or violations:
            problems.append(task)
    return dict(passed=not problems, tasks=len(rows), candidates=sum(r['pool_size'] for r in rows),
                problem_tasks=problems, references_not_valid_candidates=sum(bool(r['reference_candidate_errors']) for r in rows),
                rows=rows)


def initialize(recipe, pools, cpu_ids):
    global RECIPE, POOLS, AUTHOR
    RECIPE, POOLS = recipe, pools
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    import multiprocessing
    index = (multiprocessing.current_process()._identity[0] - 1) % len(cpu_ids)
    os.sched_setaffinity(0, {cpu_ids[index]})
    sys.path.insert(0, recipe['cpu_upstream'])
    import torch
    torch.set_num_threads(1)
    AUTHOR = torch.load(recipe['vae'], map_location='cpu', weights_only=False)


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-5)


def task_audit(task):
    import numpy as np
    import torch
    recipe, pool = RECIPE, POOLS[task]
    identity = dict(format=recipe['format'], recipe_sha256=digest(CAMPAIGN/'recipe.json'), task=task)
    errors, examples, hashes, content = [], [], {}, {}
    archive_path = CAMPAIGN / 'runs' / (task + '.tar.gz')
    try:
        # A streaming pass avoids repeatedly decompressing the large checkpoint member.
        with tarfile.open(archive_path, 'r|gz') as archive:
            for member in archive:
                if not member.isfile() or member.name in hashes or member.name not in (
                        'checkpoint.pt', 'history.json', 'progress.json', 'manifest.json'):
                    raise ValueError('Invalid archive member: ' + member.name)
                value = hashlib.sha256()
                payload = io.BytesIO()
                with archive.extractfile(member) as stream:
                    for block in iter(lambda: stream.read(2**20), b''):
                        value.update(block)
                        payload.write(block)
                hashes[member.name] = value.hexdigest()
                if member.name == 'checkpoint.pt':
                    payload.seek(0)
                    content[member.name] = torch.load(payload, map_location='cpu', weights_only=False)
                else:
                    content[member.name] = json.loads(payload.getvalue())
        if set(hashes) != {'checkpoint.pt', 'history.json', 'progress.json', 'manifest.json'}:
            raise ValueError('Incomplete archive')
        manifest = content['manifest.json']
        if set(manifest['members']) != {'checkpoint.pt', 'history.json', 'progress.json'}:
            errors.append('manifest_member_inventory')
        if manifest['identity'] != identity:
            errors.append('manifest_identity')
        for name, expected in manifest['members'].items():
            if hashes[name] != expected:
                errors.append('checksum:' + name)
        history, saved, recorded = content['history.json'], content['checkpoint.pt'], content['progress.json']
        if any(value['identity'] != identity for value in (history, saved, recorded)):
            errors.append('task_recipe_identity')
        if history['reference'] != pool['reference'] or history['status'] != 'complete' or recorded['status'] != 'complete':
            errors.append('reference_or_completion')
        rows = history['observations']
        if saved['rows'] != rows:
            errors.append('saved_history_mismatch')
        if len(rows) != 100 + history['proposals'] + history['recenter_proposals']:
            errors.append('row_accounting')
        if history['fresh_evaluations'] + history['cache_hits'] != history['proposals'] + history['recenter_proposals']:
            errors.append('cache_accounting')
        if history['acquisition_fresh'] + history['recenter_fresh'] != history['fresh_evaluations']:
            errors.append('phase_call_accounting')
        if (history['proposals'] > recipe['proposals'] or history['fresh_evaluations'] > recipe['fresh_budget']
                or (history['proposals'] < recipe['proposals'] and history['fresh_evaluations'] < recipe['fresh_budget'])):
            errors.append('budget_or_early_completion')
        for key in ('rounds','proposals','fresh_evaluations','cache_hits','acquisition_fresh','recenter_fresh','recenter_proposals','events'):
            if saved['progress'][key] != history[key] or recorded[key] != history[key]:
                errors.append('progress_mismatch:' + key)
        seen, best, initial_best, new_feasible = {}, -math.inf, -math.inf, set()
        counts = Counter()
        constraints = []
        phases = Counter()
        previous_round, previous_time = 0, 0.0
        for i, row in enumerate(rows):
            sequence, score = row['sequence'], row['score']
            invalid, similarity = candidate_errors(sequence, pool['reference'])
            structural = [name for name in invalid if name != 'similarity']
            feasible = similarity >= recipe['similarity'] - 1e-8
            constraints.append(recipe['similarity'] - similarity)
            if structural:
                counts.update(structural)
            if bool(row['feasible']) != feasible:
                counts['feasibility_label'] += 1
            if not math.isfinite(score) or not close(row['mic'], -score) or score >= 0:
                counts['score_or_sign'] += 1
            if row['round'] < previous_round or row['elapsed'] < previous_time:
                counts['chronology'] += 1
            previous_round, previous_time = row['round'], row['elapsed']
            if i < 100:
                if (sequence != pool['sequences'][i] or not close(score,pool['scores'][i])
                        or row['phase'] != 'initialization' or not feasible or not row['cached']):
                    counts['initialization_alignment'] += 1
                initial_best = max(initial_best, score)
            else:
                phases[row['phase']] += 1
                cached = sequence in seen
                if bool(row['cached']) != cached:
                    counts['cache_label'] += 1
                if cached and not close(score,seen[sequence]):
                    counts['repeated_score_changed'] += 1
                if not cached:
                    counts['observed_fresh_calls'] += 1
                    if feasible:
                        new_feasible.add(sequence)
            seen[sequence] = score
            if feasible:
                best = max(best, score)
            if (structural or bool(row['feasible']) != feasible) and len(examples) < 10:
                examples.append(dict(index=i, sequence=sequence, errors=structural, similarity=similarity))
        if counts.pop('observed_fresh_calls',0) != history['fresh_evaluations']:
            errors.append('distinct_call_count')
        if phases['acquisition'] != history['proposals'] or phases['recenter'] != history['recenter_proposals']:
            errors.append('phase_proposal_count')
        if not close(-best,history['best_feasible_mic']):
            errors.append('incorrect_final_incumbent')
        if counts:
            errors.extend('observations:' + key for key in counts)
        state = saved['state']
        if state['train_x'] != [row['sequence'] for row in rows]:
            errors.append('state_sequence_alignment')
        n = len(rows)
        for name, expected in (('train_y',(n,1)),('train_z',(n,256)),('train_c',(n,1))):
            if tuple(state[name].shape) != expected or not torch.isfinite(state[name]).all():
                errors.append('state_shape_or_finiteness:' + name)
        if not np.allclose(state['train_y'].numpy().ravel(),[row['score'] for row in rows],rtol=1e-6,atol=1e-5):
            errors.append('state_score_alignment')
        if not np.allclose(state['train_c'].numpy().ravel(),constraints,rtol=1e-6,atol=1e-6):
            errors.append('state_constraint_alignment')
        if set(saved['scores']) != set(seen) or any(not close(saved['scores'][x],y) for x,y in seen.items()):
            errors.append('saved_score_cache')
        if saved['oracle_calls'] != history['fresh_evaluations'] or not close(state['best_score_seen'],best):
            errors.append('state_calls_or_incumbent')
        if state['best_x_seen'] not in seen or candidate_errors(state['best_x_seen'],pool['reference'])[0]:
            errors.append('invalid_saved_incumbent')
        if state['tot_num_e2e_updates'] != history['e2e_updates']:
            errors.append('e2e_counter')
        infinite_constraint_bounds = []
        for group,weights in [('vae',saved['vae']),('model',saved['model']),
                              *[(f'constraint{i}',weights) for i,weights in enumerate(saved['constraints'])]]:
            for key,value in weights.items():
                if not torch.is_tensor(value) or torch.isfinite(value).all():
                    continue
                if key.endswith('_constraint.upper_bound') and torch.isposinf(value).all():
                    infinite_constraint_bounds.append(group+':'+key)
                elif key.endswith('_constraint.lower_bound') and torch.isneginf(value).all():
                    infinite_constraint_bounds.append(group+':'+key)
                else:
                    errors.append('nonfinite_saved_weights:'+group+':'+key)
        if set(saved['vae']) != set(AUTHOR):
            errors.append('vae_weight_keys')
        changed = sum(not torch.equal(value,AUTHOR[key]) for key,value in saved['vae'].items())
        if history['e2e_updates'] and not changed:
            errors.append('vae_updates_without_changed_weights')
        from gpytorch.likelihoods import GaussianLikelihood
        from lolbo.utils.bo_utils.ppgpr import GPModelDKL
        posterior_checks=[]
        probe_indices=torch.linspace(0,n-1,16,dtype=torch.long)
        probe=state['train_z'][probe_indices]
        for group,weights in [('model',saved['model']),*[(f'constraint{i}',w) for i,w in enumerate(saved['constraints'])]]:
            inducing_count=weights['variational_strategy.inducing_points'].shape[0]
            model=GPModelDKL(state['train_z'][:inducing_count],GaussianLikelihood())
            model.load_state_dict(weights,strict=True)
            model.eval();model.likelihood.eval()
            with torch.no_grad():
                posterior=model.likelihood(model(probe))
                if (not torch.isfinite(posterior.mean).all() or not torch.isfinite(posterior.variance).all()
                        or (posterior.variance<0).any()):
                    errors.append('invalid_reloaded_posterior:'+group)
                posterior_checks.append(dict(group=group,mean_min=float(posterior.mean.min()),
                                             mean_max=float(posterior.mean.max()),
                                             variance_min=float(posterior.variance.min()),
                                             variance_max=float(posterior.variance.max())))
        sample_indices = [0,max(range(100),key=lambda i:rows[i]['score']),
                          next(i for i,row in enumerate(rows) if row['sequence']==state['best_x_seen'])]
        fresh_index=next((i for i,row in enumerate(rows[100:],100) if not row['cached']),None)
        if fresh_index is not None:sample_indices.append(fresh_index)
        return dict(task=task,passed=not errors,errors=errors,examples=examples,observations=n,
                    intentional_infinite_constraint_bounds=infinite_constraint_bounds,
                    reloaded_posterior_checks=posterior_checks,
                    rescore_samples=[dict(index=i,**rows[i]) for i in dict.fromkeys(sample_indices)],
                    distinct_sequences=len(seen),new_feasible_sequences=len(new_feasible),
                    proposals=history['proposals'],fresh_evaluations=history['fresh_evaluations'],
                    recenter_fresh=history['recenter_fresh'],cache_hits=history['cache_hits'],
                    initial_best_mic=-initial_best,final_best_mic=-best,
                    relative_improvement=(initial_best-best)/initial_best,
                    improved=best>initial_best+max(1e-6,abs(initial_best)*1e-6),
                    rounds=history['rounds'],e2e_updates=history['e2e_updates'],vae_changed_tensors=changed,
                    archive_bytes=archive_path.stat().st_size,member_checksums=hashes)
    except Exception as exc:
        return dict(task=task,passed=False,errors=[type(exc).__name__+': '+str(exc)])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--cpus',default='12,13,14,15')
    args=parser.parse_args()
    cpus=list(map(int,args.cpus.split(',')))
    if len(cpus)!=args.workers or len(set(cpus))!=len(cpus) or not set(cpus)<=os.sched_getaffinity(0):
        raise ValueError('Allocate one unique available CPU per audit worker')
    recipe=json.loads((CAMPAIGN/'recipe.json').read_text())
    pools=json.loads(Path(recipe['prepared_pool']).read_text())
    mismatch=[]
    for section in ('code','assets'):
        for name,expected in recipe[section].items():
            if digest(name)!=expected:mismatch.append(name)
    write(OUTPUT/'pinned_files.json',dict(passed=not mismatch,mismatches=mismatch))
    if mismatch:raise ValueError('Pinned campaign files changed')
    test=pool_audit(recipe,pools,recipe['test'])
    train=pool_audit(recipe,pools,recipe['train'])
    write(OUTPUT/'test200_pool_rules.json',test)
    write(OUTPUT/'train800_pool_rules.json',train)
    from Levenshtein import distance
    cross=max(max(1-distance(recipe['references'][a],recipe['references'][b])/len(recipe['references'][a]),
                  1-distance(recipe['references'][a],recipe['references'][b])/len(recipe['references'][b]))
              for a in recipe['train'] for b in recipe['test'])
    names=os.listdir(CAMPAIGN/'runs')
    inventory=dict(expected=800,completed=len([name for name in names if name.endswith('.tar.gz')]),
                   missing=sorted(set(recipe['train'])-{name[:-7] for name in names if name.endswith('.tar.gz')}),
                   unexpected=sorted(set(names)-{task+'.tar.gz' for task in recipe['train']}),
                   maximum_train_test_reference_similarity=cross)
    write(OUTPUT/'inventory.json',inventory)
    started=time.time();results=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(recipe,pools,cpus)) as executor:
        for result in executor.map(task_audit,recipe['train'],chunksize=1):
            results.append(result)
            if len(results)%10==0 or not result['passed']:
                progress=dict(completed=len(results),total=800,failed=sum(not r['passed'] for r in results),
                              last_task=result['task'],seconds=time.time()-started)
                write(OUTPUT/'progress.json',progress)
                print(json.dumps(progress),flush=True)
    valid=[r for r in results if r['passed']]
    summary=dict(passed=len(valid)==800 and test['passed'] and train['passed'] and not inventory['missing']
                        and not inventory['unexpected'] and cross<=0.75,
                 tasks=800,passed_tasks=len(valid),failed_tasks=[r for r in results if not r['passed']],
                 elapsed_seconds=time.time()-started)
    if valid:
        summary.update(improved_tasks=sum(r['improved'] for r in valid),
                       tasks_with_no_feasible_new_sequences=sum(not r['new_feasible_sequences'] for r in valid),
                       total_new_feasible_sequences=sum(r['new_feasible_sequences'] for r in valid),
                       total_fresh_evaluations=sum(r['fresh_evaluations'] for r in valid),
                       total_recenter_fresh=sum(r['recenter_fresh'] for r in valid),
                       total_proposals=sum(r['proposals'] for r in valid),
                       total_cache_hits=sum(r['cache_hits'] for r in valid),
                       median_relative_improvement=statistics.median(r['relative_improvement'] for r in valid),
                       median_initial_mic=statistics.median(r['initial_best_mic'] for r in valid),
                       median_final_mic=statistics.median(r['final_best_mic'] for r in valid),
                       total_distinct_pairs=sum(r['distinct_sequences'] for r in valid),
                       tasks_exhausting_fresh_budget=sum(r['fresh_evaluations']==5000 for r in valid),
                       tasks_exhausting_proposal_budget=sum(r['proposals']==5000 for r in valid))
    write(OUTPUT/'train800_audit.json',dict(summary=summary,tasks=results))
    print(json.dumps({k:v for k,v in summary.items() if k!='failed_tasks'}),flush=True)


if __name__=='__main__':main()
