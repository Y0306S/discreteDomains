"""Checkpointed direct discrete SemanticOPT; two candidates, one oracle call."""
import argparse
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time
from common import *
from protocol import SYSTEM, peptide_measure, peptide_context, validate_query_plan, select_candidate, sig4
from random_fallback import choose_fallback, query_proposal
from batch import ensure_batch, round_state
from peptide_batch import score_and_commit

def connect(directory):
    directory.mkdir(parents=True, exist_ok=True)
    db=sqlite3.connect(directory/'observations.sqlite',timeout=120)
    db.execute('PRAGMA synchronous=FULL')
    db.execute('PRAGMA journal_mode=TRUNCATE')
    db.execute('CREATE TABLE IF NOT EXISTS records (n INTEGER PRIMARY KEY, value TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    db.commit()
    return db

def get(db,key,default=None):
    row=db.execute('SELECT value FROM state WHERE key=?',(key,)).fetchone()
    return json.loads(row[0]) if row else default

def put(db,key,value):
    db.execute('INSERT OR REPLACE INTO state VALUES (?,?)',(key,canonical(value)))


def measure_pending(db, pending, oracle):
    from query_resource_recovery import QueryInfrastructureRetryExhausted
    pending['status']='in_flight'
    with db:put(db,'pending',pending)
    try:
        return oracle(pending['answer']['next_configuration'])
    except QueryInfrastructureRetryExhausted as exc:
        # All allowed retries returned explicit failures. Preserve the proposal
        # and audit trail without consuming budget or manufacturing a runtime.
        with db:
            put(db, f'infrastructure_failure_{pending["number"]}_{time.time_ns()}',
                dict(at=time.time(),failures=exc.failures))
            pending['status']='selected'
            put(db,'pending',pending)
        raise


def query_initializations(raw, definition, codec, saved=None):
    """Audit Bao measurements and reuse the exact checkpoint representation.

    AliasesCodec orders independent branches through a set, so an equivalent
    join tree can have different integer encodings in another process.
    """
    if saved is not None and (not isinstance(saved, list) or len(saved) != len(raw)):
        raise ValueError('Bao initialization count changed')
    initial=[]
    for index, row in enumerate(raw):
        tree=codec.decode(definition.query_tables,row['plan'].copy())
        metadata=dict(objective=row['runtime'],
            status='timeout' if row['censored'] else 'success',
            censored=bool(row['censored']),feasible=True)
        if saved is None:
            plan=codec.encode(tree)
            validate_query_plan(plan,definition,codec)
            initial.append(dict(configuration=dict(plan=plan),**metadata))
            continue
        old=saved[index]
        if (not isinstance(old,dict) or
                {k:v for k,v in old.items() if k!='configuration'} != metadata or
                not isinstance(old.get('configuration'),dict) or
                set(old['configuration'])!={'plan'}):
            raise ValueError(f'Bao initialization measurement changed at row {index}')
        plan=old['configuration']['plan']
        validate_query_plan(plan,definition,codec,require_canonical=False)
        if not tree.equal(codec.decode(definition.query_tables,plan.copy())):
            raise ValueError(f'Bao initialization join tree changed at row {index}')
        initial.append(old)
    return initial


def query_setup(task, settings, port, initial_snapshot=None):
    sys.path.insert(0,str(QUERY_SOURCE/'code'))
    import run_query_ceb_baselines as query
    native=QUERY_SOURCE/'native'; query.native_path(native)
    from workload.workloads import get_workload_set, WorkloadSpec, OracleCodec
    from codec.codec import AliasesCodec
    sql=query.sql_path(native,task)
    definition=get_workload_set('CEB_3K').extract_query(sql.read_text(),task)
    spec=WorkloadSpec.from_definition(definition,OracleCodec.Aliases)
    codec=AliasesCodec(definition.all_tables)
    raw=query.load_initializations(native/'initialization_data/ceb_3k_bao_initialization.csv',[task])[task]
    initial=query_initializations(raw,definition,codec,initial_snapshot)
    context=dict(task_id=task,domain='postgres_query_plan_optimization',sql=sql.read_text(),
        sql_sha256=digest(sql),query_tables=definition.query_tables,
        table_symbols={str(i):t for i,t in enumerate(definition.all_tables)},
        objective=dict(name='uncensored PostgreSQL runtime',unit='seconds',direction='minimize'),
        configuration_schema=dict(plan='integer list; each join is [left_table_symbol,left_alias,right_table_symbol,right_alias,operator]'),
        constraints=dict(operators={'1':'NestLoop','2':'HashJoin','3':'MergeJoin'},
            preserve='all query tables, aliases, predicates and SQL semantics',canonical_alias_codec=True,
            implicit_repair_allowed=False))
    def validate(config):
        try:
            if not isinstance(config,dict) or set(config)!={'plan'}:
                raise ValueError('Configuration must contain only plan')
            validate_query_plan(config['plan'],definition,codec)
        except Exception as exc:
            raise ValueError('Invalid canonical query configuration: '+str(exc)) from exc
    validate.sample_random = lambda rng: query_proposal(definition, codec, rng)
    os.environ.update(DB_HOST='127.0.0.1',DB_PORT=str(port),DB_NAME='imdb',DB_USER='imdb',DB_PASSWORD='imdb')
    oracle=query.RuntimeOracle(spec,max(r['objective'] for r in initial))
    def evaluate(config):
        from query_resource_recovery import evaluate_query
        result=evaluate_query(oracle,config['plan'],settings['timeout_seconds'])
        return dict(configuration=config,objective=result['runtime'],
            status='success' if not result['censored'] else result['status'],
            censored=bool(result['censored']),feasible=True,oracle=result)
    return context,initial,validate,evaluate

def peptide_setup(task, settings):
    data=load(settings['pool'])[task]
    reference=data['reference']
    initial=[dict(configuration=dict(sequence=s),objective=-float(y),status='success',
        censored=False,**peptide_measure(s,reference)) for s,y in zip(data['sequences'],data['scores'])]
    if len(initial)!=100 or not all(r['feasible'] and math.isfinite(r['objective']) for r in initial):
        raise ValueError('Invalid peptide initial pool')
    def validate(config):
        if not isinstance(config,dict) or set(config)!={'sequence'}:
            raise ValueError('Configuration must contain only sequence')
        if not peptide_measure(config['sequence'],reference)['feasible']:
            raise ValueError('Peptide violates reference similarity')
    # Same eight-model CPU APEX implementation as the audited baseline; no VAE needed.
    sys.path.insert(0,settings['cpu_upstream'])
    from apex_oracle.APEX_predict import apex_wrapper
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    def evaluate_many(configs):
        if not 1<=len(configs)<=50:raise ValueError('Peptide oracle batch must contain 1 to 50 items')
        for config in configs:validate(config)
        with torch.inference_mode():
            scores=apex_wrapper([c['sequence'] for c in configs])
        if scores.shape[0]!=len(configs):raise ValueError('APEX batch size mismatch')
        results=[]
        for config,score in zip(configs,scores):
            mic=float(score[0])
            if not math.isfinite(mic):raise ValueError('Nonfinite APEX MIC')
            results.append(dict(configuration=config,objective=mic,status='success',censored=False,
                **peptide_measure(config['sequence'],reference)))
        return results
    def evaluate(config):
        return evaluate_many([config])[0]
    evaluate.batch=evaluate_many
    return peptide_context(task,reference),initial,validate,evaluate

def prompt(context, initial, rows, total, errors, window=32):
    # Match causal training windows; retain incumbent alongside recent observations.
    successful=[r for r in initial+rows if not r.get('censored') and r['status']=='success']
    best=min(successful,key=lambda r:r['objective'])
    selected=(initial[:50]+rows[-window:]) if 'reference_sequence' in context else (initial+rows)[-window:]
    def compact(row):
        return {k:(sig4(v) if k=='objective' else v) for k,v in row.items() if k in
            ('configuration','objective','status','censored','feasible','similarity','proposal_source')}
    content=dict(**context,total_evaluations_including_initial_observations=len(initial)+total,
        completed_proposals=len(rows),initial_evaluations=[compact(r) for r in selected],
        best_observation=compact(best),recent_rejected_configurations=errors[-10:],
        instruction='Output a distinct, valid next configuration. Do not repeat any evaluated configuration.')
    # Replay measured decisions as the same assistant/action -> user/feedback
    # conversation used for SFT. Imported observations remain user context.
    replay=[r for r in rows[-window:] if 'prediction' in r]
    imported=[r for r in selected if 'prediction' not in r]
    content['initial_evaluations']=[compact(r) for r in imported]
    messages=[dict(role='system',content=SYSTEM),dict(role='user',content=canonical(content)+'\nOutput the next configuration.')]
    for row in replay:
        messages.append(dict(role='assistant',content=canonical(row['prediction'])))
        feedback={k:v for k,v in compact(row).items() if k!='configuration'}
        messages.append(dict(role='user',content=canonical(dict(evaluation=row['evaluation'],**feedback))+'\nOutput the next configuration.'))
    return messages

def parse_answer(text):
    text=text.strip()
    if text.startswith('```'):
        text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
    value=json.loads(text)
    if not isinstance(value,dict):raise ValueError('Expected a JSON object')
    return value

def run(domain, task, port, smoke=False):
    recipe=domain_recipe(verify(),domain);settings=recipe[domain]
    allowed=settings['smoke_task'] if smoke else None
    if task not in ([allowed] if smoke else settings['test']):raise ValueError('Task is outside allowed split')
    directory=ROOT/domain/('smoke' if smoke else 'runs')/task
    guard=lock(directory/'task.lock')
    db=connect(directory)
    identity=dict(recipe_sha256=digest(ROOT/'recipe.json'),domain=domain,task=task,smoke=smoke)
    if get(db,'identity') not in (None,identity):raise ValueError('Resume identity mismatch')
    with db:put(db,'identity',identity)
    runtime_sha = recipe.get('runtime_policy_sha256')
    if get(db, 'runtime_policy_sha256') not in (None, runtime_sha):
        raise ValueError('Runtime policy changed during evaluation; explicit audit required')
    with db:put(db, 'runtime_policy_sha256', runtime_sha)
    batch_sha=recipe.get('batch_policy_sha256')
    if get(db,'batch_policy_sha256') not in (None,batch_sha):
        raise ValueError('Batch policy changed; explicit audit required')
    with db:put(db,'batch_policy_sha256',batch_sha)
    budget_sha=recipe.get('budget_policy_sha256')
    if get(db,'budget_policy_sha256') not in (None,budget_sha):
        raise ValueError('Budget policy changed; explicit audit required')
    with db:put(db,'budget_policy_sha256',budget_sha)
    oracle_sha=recipe.get('peptide_oracle_policy_sha256') if domain=='peptide' else None
    if domain=='peptide':
        if get(db,'peptide_oracle_policy_sha256') not in (None,oracle_sha):
            raise ValueError('Peptide oracle policy changed; explicit audit required')
        with db:put(db,'peptide_oracle_policy_sha256',oracle_sha)
        journal=get(db,'oracle_batch')
        if journal and journal['status']=='in_flight':
            raise RuntimeError('Unresolved peptide batch oracle call; requires audit before replay')
    context,initial,validate,oracle=(query_setup(task,settings,port,initial_snapshot=get(db,'initial'))
        if domain=='query' else peptide_setup(task,settings))
    from qwen_inference_transport import SharedQwenClient
    import torch
    torch.manual_seed(recipe['seed'])
    client=SharedQwenClient(ROOT/domain/'services/client.json',settings['base'],settings['adapter'])
    os.environ['ML_QWEN_REQUEST_NAMESPACE']=str(directory)
    os.environ['ML_RUN_ID']=ROOT.name
    client.import_state(get(db,'inference_rng'))
    client.ensure_loaded()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(settings['adapter'],trust_remote_code=True,local_files_only=True)
    total=1 if smoke else settings['proposals']
    rows=[json.loads(r[0]) for r in db.execute('SELECT value FROM records ORDER BY n')]
    if len(rows)>total:
        raise ValueError('Existing evaluations exceed the effective budget; preserve and audit explicitly')
    seen={canonical(r['configuration']) for r in initial+rows}
    with db:
        if get(db,'initial') is None:put(db,'initial',initial)
        elif get(db,'initial')!=initial:raise ValueError('Initial observations changed')
    started=time.time()
    status(directory/'status.json',status='running',domain=domain,task=task,evaluations=len(rows),budget=total,
        budget_policy_sha256=budget_sha,recipe_sha256=identity['recipe_sha256'],adapter_sha256=settings['adapter_sha256'],
        runtime_policy_sha256=runtime_sha,batch_policy_sha256=batch_sha,**round_state(len(rows),recipe.get('batch_policy',{}).get('size',50)),
        peptide_oracle_policy_sha256=oracle_sha,
        random_fallback_evaluations=sum(r.get('proposal_source')=='random_fallback' for r in rows))
    while len(rows)<total:
        if (ROOT/'PAUSE').exists():raise RuntimeError('Campaign paused')
        yield_file=os.environ.get('SEMANTICOPT_YIELD_FILE')
        if yield_file and Path(yield_file).exists():
            status(directory/'status.json',status='yielded',domain=domain,task=task,evaluations=len(rows),
                budget=total,budget_policy_sha256=budget_sha,recipe_sha256=identity['recipe_sha256'],adapter_sha256=settings['adapter_sha256'],
                runtime_policy_sha256=runtime_sha,batch_policy_sha256=batch_sha,**round_state(len(rows),recipe.get('batch_policy',{}).get('size',50)),
                peptide_oracle_policy_sha256=oracle_sha)
            db.close();guard.close();return
        pending=get(db,'pending')
        if pending is None:
            if recipe.get('batch_policy'):
                batch=ensure_batch(db,get,put,recipe=recipe,domain=domain,context=context,
                    initial=initial,rows=rows,total=total,validate=validate,seen=seen,
                    tokenizer=tokenizer,client=client)
                if domain=='peptide' and recipe.get('peptide_oracle_policy',{}).get('enabled'):
                    results=score_and_commit(db,get,put,batch=batch,start=len(rows),total=total,
                        validate=validate,oracle=oracle.batch,seen=seen,
                        runtime_sha=runtime_sha,policy_sha=oracle_sha)
                    rows.extend(results);seen.update(canonical(r['configuration']) for r in results)
                    status(directory/'status.json',status='running',domain=domain,task=task,evaluations=len(rows),
                        budget=total,budget_policy_sha256=budget_sha,recipe_sha256=identity['recipe_sha256'],adapter_sha256=settings['adapter_sha256'],
                        runtime_policy_sha256=runtime_sha,batch_policy_sha256=batch_sha,**round_state(len(rows),recipe.get('batch_policy',{}).get('size',50)),
                        peptide_oracle_policy_sha256=oracle_sha,oracle_execution='cpu_batch',
                        last_oracle_batch_size=len(results),last_oracle_batch_seconds=results[-1]['oracle_batch_seconds'],
                        random_fallback_evaluations=sum(r.get('proposal_source')=='random_fallback' for r in rows),
                        best=min(r['objective'] for r in initial+rows if not r.get('censored') and r['status']=='success'))
                    continue
                item=batch['items'][len(rows)-batch['start']]
                pending=dict(**item,number=len(rows),status='selected',
                    runtime_policy_sha256=runtime_sha,batch_policy_sha256=batch_sha)
                with db:put(db,'pending',pending)
            else:
                errors=[];chosen=None
                best=min(r['objective'] for r in initial+rows if not r.get('censored') and r['status']=='success')
                had_failure=any(r.get('censored') for r in rows)
                def validate_fresh(config):
                    validate(config)
                    if canonical(config) in seen:raise ValueError('Duplicate configuration')
                for retry in range(recipe['candidate_retries']):
                    # Persist each sample/RNG together; durable broker replays interrupted requests.
                    sample_key=f'samples_{len(rows)}_{retry}'
                    samples=get(db,sample_key,[])
                    window=32
                    messages=prompt(context,initial,rows,total,errors,window)
                    while len(tokenizer(tokenizer.apply_chat_template(messages,tokenize=False,
                            add_generation_prompt=True,enable_thinking=False))['input_ids'])>15360:
                        if window<=1:raise ValueError('Task definition exceeds the validated inference context')
                        window=max(1,window//2)
                        messages=prompt(context,initial,rows,total,errors,window)
                    while len(samples)<2:
                        text=client.complete_messages(messages,use_s1b_template=False,settings=recipe['generation'])
                        try:answer=parse_answer(text)
                        except (ValueError,TypeError):answer={}
                        samples.append(dict(answer=answer,text=text))
                        with db:
                            put(db,sample_key,samples);put(db,'inference_rng',client.state)
                    answers=[s['answer'] for s in samples]
                    try:
                        chosen=select_candidate(answers,best,validate_fresh,had_failure=had_failure)
                        break
                    except (ValueError,KeyError,TypeError) as exc:
                        errors.extend(a.get('next_configuration',{}) for a in answers)
                        with db:put(db,f'rejection_{len(rows)}_{retry}',dict(error=str(exc),answers=answers))
                provenance=dict(source='llm', candidate_pairs=retry+1)
                if chosen is None:
                    policy=recipe.get('fallback')
                    if not isinstance(policy,dict) or not policy.get('enabled'):
                        raise RuntimeError('No valid distinct SemanticOPT proposal after candidate retries')
                    config,provenance=choose_fallback(domain,context,validate,seen,errors,
                        seed=recipe['seed'],number=len(rows),policy=policy)
                    chosen=dict(next_configuration=config)
                pending=dict(answer=chosen,number=len(rows),status='selected',provenance=provenance,
                    runtime_policy_sha256=runtime_sha)
                with db:put(db,'pending',pending)
        if pending['status']=='in_flight':
            raise RuntimeError('Unresolved oracle call; requires audit before replay')
        if pending['number']!=len(rows):raise ValueError('Pending evaluation cursor mismatch')
        validate(pending['answer']['next_configuration'])
        if canonical(pending['answer']['next_configuration']) in seen:raise ValueError('Pending duplicate configuration')
        result=measure_pending(db,pending,oracle)
        provenance=pending.get('provenance',dict(source='llm'))
        result.update(evaluation=len(rows)+1,at=time.time(),proposal_source=provenance['source'],
            proposal_provenance=provenance,runtime_policy_sha256=pending.get('runtime_policy_sha256'))
        if provenance['source']=='llm' and 'mean' in pending['answer']:result['prediction']=pending['answer']
        if pending.get('batch_policy_sha256'):
            result.update(batch_policy_sha256=pending['batch_policy_sha256'],
                proposal_round=len(rows)//recipe['batch_policy']['size'],
                proposal_round_offset=len(rows)%recipe['batch_policy']['size'])
        with db:
            db.execute('INSERT INTO records VALUES (?,?)',(len(rows),canonical(result)))
            put(db,'pending',None)
        rows.append(result);seen.add(canonical(result['configuration']))
        status(directory/'status.json',status='running',domain=domain,task=task,evaluations=len(rows),
            budget=total,budget_policy_sha256=budget_sha,recipe_sha256=identity['recipe_sha256'],adapter_sha256=settings['adapter_sha256'],
            runtime_policy_sha256=runtime_sha,batch_policy_sha256=batch_sha,**round_state(len(rows),recipe.get('batch_policy',{}).get('size',50)),
            random_fallback_evaluations=sum(r.get('proposal_source')=='random_fallback' for r in rows),
            best=min(r['objective'] for r in initial+rows if not r.get('censored') and r['status']=='success'))
    status(directory/'status.json',status='complete',domain=domain,task=task,evaluations=len(rows),budget=total,budget_policy_sha256=budget_sha,
        recipe_sha256=identity['recipe_sha256'],adapter_sha256=settings['adapter_sha256'],
        runtime_policy_sha256=runtime_sha,batch_policy_sha256=batch_sha,**round_state(len(rows),recipe.get('batch_policy',{}).get('size',50)),
        random_fallback_evaluations=sum(r.get('proposal_source')=='random_fallback' for r in rows),
        peptide_oracle_policy_sha256=oracle_sha,seconds_this_process=time.time()-started)
    db.close();guard.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('domain',choices=['query','peptide']);p.add_argument('task')
    p.add_argument('--port',type=int,default=0);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    run(args.domain,args.task,args.port,args.smoke)
