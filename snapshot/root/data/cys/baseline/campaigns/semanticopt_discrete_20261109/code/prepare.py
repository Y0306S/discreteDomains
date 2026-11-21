"""CPU-only causal annotation of audited train800 corpora; never calls an oracle."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
import numpy as np
from protocol import *

CORPORA = dict(peptide=Path('/root/data/logs/peptide_transfer_train800_20261027_v2/corpus'),
               query=Path('/root/data/logs/query_transfer_train800_20261106_v1/corpus'))
QUERY_SOURCE = Path('/root/data/cys/logs/query_train800_author_fixed_20261004')
PEPTIDE_SOURCE = Path('/root/data/logs/peptide_bo_train800_20261027_v5b')
RECIPE = dict(peptide_warmup=50, peptide_decisions=50, query_window=32, query_warmup=5,
    teacher='sklearn RandomForestRegressor, 10 bootstrap trees, min leaf 3, max features 5/6',
    uncertainty='population standard deviation of tree predictions; no floor',
    label_policy='fit only preceding selected observations; predict before revealing next outcome',
    peptide_selection='first 50 feasible initialization points, then first 50 feasible post-initialization points in archive order',
    query_selection='all audited distinct uncensored plans, in chronological windows of at most 32; one-point tail remains context-only',
    failures='success-only audited corpus; no fabricated failure-probability labels',
    context='local causal window only; fixed frozen author VAE means used solely by offline RF teacher',
    no_test_data=True, oracle_calls=0)

def read_task(domain, task, manifest):
    c = sqlite3.connect('file:'+str(CORPORA[domain]/'corpus.sqlite')+'?mode=ro', uri=True)
    row = c.execute('select observations,source_sha256,payload_sha256,payload from tasks where id=?', (task,)).fetchone(); c.close()
    if row is None or hashlib.sha256(row[3]).hexdigest() != row[2] or row[2] != manifest['task_hashes'][task]:
        raise ValueError('Corpus payload checksum mismatch: '+task)
    with np.load(io.BytesIO(row[3]), allow_pickle=False) as z:
        data = {key:z[key].copy() for key in z.files}
    if len(data['y']) != row[0] or not np.isfinite(data['X']).all() or not np.isfinite(data['y']).all():
        raise ValueError('Invalid aligned features/outcomes')
    return data, row[1], row[2]

def annotate(features, objectives, records, warmup, context, task, source_indices):
    from sklearn.ensemble import RandomForestRegressor
    seed = (42 + int(hashlib.sha256(task.encode()).hexdigest()[:8], 16)) % (2**31-1)
    messages = [dict(role='system', content=SYSTEM), dict(role='user', content=canonical(dict(
        **context, total_evaluations_including_initial_observations=len(records),
        initial_evaluations=records[:warmup]))+'\nOutput the next configuration.')]
    attestations = []
    for index in range(warmup, len(records)):
        model = RandomForestRegressor(n_estimators=10, min_samples_leaf=3, max_features=5/6,
            bootstrap=True, random_state=seed, n_jobs=1)
        model.fit(features[:index], objectives[:index])
        predictions = np.asarray([t.predict(features[index:index+1])[0] for t in model.estimators_])
        answer = dict(next_configuration=records[index]['configuration'],
                      mean=sig4(predictions.mean()), std=sig4(predictions.std(ddof=0)))
        messages.append(dict(role='assistant', content=canonical(answer)))
        feedback = {k:v for k,v in records[index].items() if k != 'configuration'}
        messages.append(dict(role='user', content=canonical(dict(evaluation=index+1, **feedback))+
            ('\nOutput the next configuration.' if index+1 < len(records) else '\nEvaluation budget exhausted.')))
        attestations.append(dict(index=index, source_index=int(source_indices[index]),
            prefix_sha256=value_hash(records[:index]), fit_rows=index, **answer))
    return dict(task_id=task, messages=messages, annotations=attestations, warmup=warmup,
                selected_source_indices=[int(v) for v in source_indices])

def task_job(domain, task, limit=None):
    manifest = json.loads((CORPORA[domain]/'manifest.json').read_text())
    data, source_hash, payload_hash = read_task(domain, task, manifest)
    trajectories = []
    if domain == 'peptide':
        recipe = json.loads((PEPTIDE_SOURCE/'recipe.json').read_text())
        reference = data['reference'].item()
        if reference != recipe['references'][task]:
            raise ValueError('Wrong peptide reference')
        feasible = []
        for i, sequence in enumerate(data['sequences'].tolist()):
            measurement = peptide_measure(sequence, reference)
            if abs((.75-measurement['similarity'])-float(data['constraints'][i])) > 1e-6:
                raise ValueError('Measured similarity differs from recorded constraint')
            if measurement['feasible']:
                feasible.append(i)
        initial = [i for i in feasible if i < recipe['initialization']][:RECIPE['peptide_warmup']]
        actions = [i for i in feasible if i >= recipe['initialization']][:RECIPE['peptide_decisions']]
        if len(initial) != 50 or not actions:
            return dict(task_id=task, exclusion='Insufficient feasible initialization or post-initialization decisions')
        indices = initial+actions
        if limit: indices = indices[:50+limit]
        records = [dict(configuration=dict(sequence=str(data['sequences'][i])), objective=sig4(-data['y'][i]),
            status='success', **peptide_measure(str(data['sequences'][i]), reference)) for i in indices]
        trajectories.append(annotate(data['X'][indices], -data['y'][indices], records, 50,
            peptide_context(task, reference), task, indices))
    else:
        native = QUERY_SOURCE/'native'
        sys.path.insert(0, str(native))
        from codec.codec import AliasesCodec
        from workload.workloads import get_workload_set, WorkloadSpec, OracleCodec
        from oracle.oracle import _decode_query
        match = re.fullmatch(r'CEB_(\d+[A-Z])(\d+)', task)
        sql = native/'workload/ceb-3k'/match[1].lower()/((match[1]+match[2]).lower()+'.sql')
        if digest(sql) != manifest['sql_hashes'][task]:
            raise ValueError('Query SQL checksum changed')
        workloads = get_workload_set('CEB_3K')
        definition = workloads.extract_query(sql.read_text(), task)
        spec = WorkloadSpec.from_definition(definition, OracleCodec.Aliases)
        codec = AliasesCodec(definition.all_tables)
        path = QUERY_SOURCE/'outputs'/task/'lolbo/seed0/history.json'
        if digest(path) != source_hash:
            raise ValueError('Query history checksum changed')
        history = json.loads(path.read_text())
        records = []
        for i, source_row in enumerate(data['source_rows']):
            row = history['observations'][int(source_row)]
            if row['trial_state'] != 0 or row['extra_info']['censored'] or float(row['objectives'][0]) != data['y'][i]:
                raise ValueError('Query row objective/censoring mismatch')
            raw = row['extra_info']['plan']
            plan = codec.encode(codec.decode(definition.query_tables, raw.copy()))
            validate_query_plan(plan, definition, codec)
            if _decode_query(spec, raw.copy()) != _decode_query(spec, plan.copy()):
                raise ValueError('Canonicalization changed executed SQL')
            records.append(dict(configuration=dict(plan=plan), objective=sig4(data['y'][i]),
                                status='success', feasible=True))
        context = dict(task_id=task, domain='postgres_query_plan_optimization',
            sql=sql.read_text(), sql_sha256=digest(sql), query_tables=definition.query_tables,
            table_symbols={str(i):t for i,t in enumerate(definition.all_tables)},
            objective=dict(name='uncensored PostgreSQL runtime', unit='seconds', direction='minimize'),
            configuration_schema=dict(plan='integer list; each join is [left_table_symbol,left_alias,right_table_symbol,right_alias,operator]'),
            constraints=dict(operators={'1':'NestLoop','2':'HashJoin','3':'MergeJoin'},
                preserve='all query tables, aliases, predicates and SQL semantics',
                canonical_alias_codec=True, implicit_repair_allowed=False))
        size = RECIPE['query_window']
        for start in range(0, len(records)-1, size):
            end = min(start+size, len(records))
            warmup = min(5, end-start-1)
            if limit:end = min(end, start+warmup+limit)
            trajectories.append(annotate(data['X'][start:end], data['y'][start:end], records[start:end],
                warmup, context, task, data['source_rows'][start:end]))
    return dict(task_id=task, trajectories=trajectories, source_sha256=source_hash,
                payload_sha256=payload_hash, available_observations=len(data['y']))

def prepare(domain, workers, pilot):
    directory = ROOT/domain/'data'; directory.mkdir(parents=True, exist_ok=True)
    corpus_manifest = CORPORA[domain]/'manifest.json'
    manifest = json.loads(corpus_manifest.read_text())
    if manifest['status'] != 'complete':raise ValueError('Corpus not complete')
    if domain == 'peptide':
        recipe = json.loads((PEPTIDE_SOURCE/'recipe.json').read_text())
        train, test = recipe['train'], recipe['test']
        if digest(PEPTIDE_SOURCE/'recipe.json') != manifest['source_recipe_sha256']:raise ValueError('Recipe changed')
    else:
        plan = json.loads((QUERY_SOURCE/'plan.json').read_text())
        train_path, test_path = QUERY_SOURCE/'train800.txt', QUERY_SOURCE/'test200.txt'
        if digest(train_path) != manifest['train_split_sha256'] or digest(test_path) != manifest['test_split_sha256']:
            raise ValueError('Query split changed')
        train, test = train_path.read_text().split(), test_path.read_text().split()
        if digest(QUERY_SOURCE/'native/codec/codec.py') != manifest['codec_sha256']:raise ValueError('Codec changed')
    tasks = manifest['source_tasks']
    if len(train) != 800 or len(test) != 200 or set(train)&set(test) or not set(tasks)<=set(train):
        raise ValueError('Training/test split violation')
    if set(tasks)|set(manifest.get('excluded_tasks', {})) != set(train):raise ValueError('Unaccounted training tasks')
    identity = dict(domain=domain, recipe=RECIPE, script_sha256=digest(__file__),
        protocol_sha256=digest(Path(__file__).with_name('protocol.py')),
        corpus_manifest_sha256=digest(corpus_manifest), pilot=pilot)
    task_dir = directory/('pilot_tasks' if pilot else 'tasks'); task_dir.mkdir(exist_ok=True)
    selected = tasks[:2] if pilot else tasks
    started = time.time()
    def publish(n):
        atomic(directory/('pilot_status.json' if pilot else 'annotation_status.json'),
            dict(status='annotating',completed=n,total=len(selected),elapsed_seconds=time.time()-started))
    finished = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {}
        for task in selected:
            cached = task_dir/(task+'.json')
            if cached.exists():
                result = json.loads(cached.read_text())
                if result.get('identity') != identity:raise ValueError('Cached annotation identity changed')
                finished[task] = result;continue
            jobs[pool.submit(task_job, domain, task, 2 if pilot else None)] = task
        for job in as_completed(jobs):
            task = jobs[job];result = dict(job.result(),identity=identity)
            atomic(task_dir/(task+'.json'), result);finished[task] = result
            publish(len(finished))
            if len(finished)%20 == 0:print(domain,len(finished),len(selected),flush=True)
    target = directory/('pilot_trajectories.jsonl' if pilot else 'trajectories.jsonl')
    temp = target.with_suffix('.tmp')
    count = decisions = 0; excluded = dict(manifest.get('excluded_tasks', {}))
    with temp.open('w') as stream:
        for task in selected:
            result = finished[task]
            if 'exclusion' in result:excluded[task]=result['exclusion'];continue
            for trajectory in result['trajectories']:
                trajectory.update(source_sha256=result['source_sha256'],payload_sha256=result['payload_sha256'])
                stream.write(canonical(trajectory)+'\n');count+=1;decisions+=len(trajectory['annotations'])
    os.replace(temp,target)
    report = dict(identity,status='complete',trajectories=count,supervised_decisions=decisions,
        represented_tasks=[t for t in selected if 'exclusion' not in finished[t]],excluded_tasks=excluded,
        dataset_sha256=digest(target),train_test_disjoint_verified=True,oracle_calls=0,
        elapsed_seconds=time.time()-started)
    atomic(target.with_suffix('.manifest.json'),report);print(canonical(report),flush=True)

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('domain',choices=CORPORA);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--pilot',action='store_true');a=p.parse_args();prepare(a.domain,a.workers,a.pilot)
