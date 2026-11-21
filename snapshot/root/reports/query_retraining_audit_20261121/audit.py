"""Read-only, completed-round comparison of original and retrained Query runs."""
import collections, csv, json, math, pickle, sqlite3, sys, time
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
BASE = Path('/root/data/cys/baseline/campaigns')
NEW = BASE/'query_our_fsbo_vae80_test200_20261117'
OLD = BASE/'query_our_fsbo_test200_20261116'
sys.path.insert(0, str(NEW/'batch_code'))
from query_data import plan_key

def load(root):
    tasks = {}
    errors = []
    for path in sorted((root/'orpo_batch10/runs').glob('*/runtime.sqlite')):
        try:
            with sqlite3.connect('file:'+str(path)+'?mode=ro', uri=True, timeout=5) as db:
                db.execute('PRAGMA query_only=ON')
                blob = db.execute("SELECT value FROM kv WHERE key='optimization_state'").fetchone()
                if not blob: continue
                state = pickle.loads(blob[0])
                times = {k:pickle.loads(v).get('at',0) for k,v in db.execute("SELECT key,value FROM kv WHERE key LIKE 'measurement/%'")}
            for event in state['events']:
                event['_last_measurement_at'] = max(times.get(f'measurement/{event["round"]}/{i}',float('inf')) for i in range(10))
            tasks[path.parent.name] = state
        except Exception as exc: errors.append(dict(task=path.parent.name,error=repr(exc)))
    return tasks, errors

def incumbent(rows):
    return min((r for r in rows if not r['censored']), key=lambda r:r['runtime'])

def diagnostics(events):
    c = collections.Counter(); sources=collections.Counter(); errors=collections.Counter()
    for e in events:
        c['rounds'] += 1
        c['selected_'+e['selected_arm']] += 1
        for result in e['results']:
            row=result['row']; source=result['actual_arm']; sources[source]+=1
            c['measurements']+=1
            c['timeouts']+=bool(row['censored'])
            c['timeouts_'+source]+=bool(row['censored'])
        d=e.get('fsbo_diagnostics')
        if not d:continue
        c['fsbo_calls']+=1;c['fsbo_requested']+=d['requested'];c['fsbo_accepted']+=d['accepted']
        c['fsbo_underfilled_calls']+=d['accepted']<d['requested']
        c['fsbo_rejected_decodes']+=len(d['rejected_decodes'])
        for rejected in d['rejected_decodes']:
            errors[rejected['error']]+=1
            decoded=rejected.get('decoded',{})
            c['decoder_explicit_fallbacks']+=bool(decoded.get('fallback'))
            c['rejected_decode_retries']+=decoded.get('decode_retries',0)
        c['canonicalized_accepted']+=sum(x.get('native_canonicalized',False) for x in d.get('accepted_decodes',[]))
    if c['fsbo_requested']:c['fsbo_requested_slot_fill_rate']=c['fsbo_accepted']/c['fsbo_requested']
    if c['fsbo_accepted']+c['fsbo_rejected_decodes']:c['fsbo_returned_decode_acceptance']=c['fsbo_accepted']/(c['fsbo_accepted']+c['fsbo_rejected_decodes'])
    return dict(counts=dict(c),sources=dict(sources),rejection_reasons=dict(errors))

def compare(new,old,cutoff=None):
    rows=[];new_events=[];old_events=[];same_plan_ratios=[]
    for task in sorted(new.keys()&old.keys()):
        ns,os=new[task],old[task]
        ne=[e for e in ns['events'] if cutoff is None or e['_last_measurement_at']<=cutoff]
        oe=os['events'];n=min(len(ne),len(oe))
        if not n:continue
        ne,oe=ne[:n],oe[:n]
        nr=ns['rows'][:30]+[r['row'] for e in ne for r in e['results']]
        orows=os['rows'][:30]+[r['row'] for e in oe for r in e['results']]
        assert [(r['plan'],r['runtime'],r['censored']) for r in nr[:30]]==[(r['plan'],r['runtime'],r['censored']) for r in orows[:30]]
        nb,ob=incumbent(nr),incumbent(orows);warm=incumbent(nr[:30])['runtime']
        ratio=nb['runtime']/ob['runtime']
        om={plan_key(r['plan']):r['runtime'] for r in orows[30:] if not r['censored']}
        pairs=[r['runtime']/om[plan_key(r['plan'])] for r in nr[30:] if not r['censored'] and plan_key(r['plan']) in om]
        same_plan_ratios.extend(pairs)
        rows.append(dict(task=task,optimization_rounds=n,optimization_measurements=n*10,new_best=nb['runtime'],old_best=ob['runtime'],bolt_best=warm,new_over_old=ratio,new_best_source=nb['source_phase'],old_best_source=ob['source_phase'],same_best_plan=plan_key(nb['plan'])==plan_key(ob['plan']),same_plan_timing_pairs=len(pairs),same_plan_geomean_ratio=math.exp(sum(map(math.log,pairs))/len(pairs)) if pairs else None))
        new_events.extend(ne);old_events.extend(oe)
    logs=np.log([r['new_over_old'] for r in rows]);rng=np.random.default_rng(123)
    interval=np.exp(np.quantile(np.mean(rng.choice(logs,(10000,len(logs))),axis=1),[.025,.975])).tolist() if len(logs) else None
    return dict(queries=len(rows),optimization_measurements_per_arm=sum(r['optimization_measurements'] for r in rows),geomean_new_over_old=float(np.exp(logs.mean())) if len(logs) else None,bootstrap_task_95pct_interval=interval,median_new_over_old=float(np.median(np.exp(logs))) if len(logs) else None,wins_over_5pct=sum(r['new_over_old']<.95 for r in rows),losses_over_5pct=sum(r['new_over_old']>1.05 for r in rows),within_5pct=sum(.95<=r['new_over_old']<=1.05 for r in rows),new_best_sources=dict(collections.Counter(r['new_best_source'] for r in rows)),same_plan_timing_pairs=len(same_plan_ratios),same_plan_timing_geomean_new_over_old=math.exp(sum(map(math.log,same_plan_ratios))/len(same_plan_ratios)) if same_plan_ratios else None,new_diagnostics=diagnostics(new_events),old_diagnostics=diagnostics(old_events),per_query=rows)

new,ne=load(NEW);old,oe=load(OLD)
# Recorded storage-change boundary, before optimizers resumed on the new setup.
boundary=json.loads(Path('/root/reports/query_ram_migration_20261118/storage_boundary.json').read_text())
cutoff=boundary['at']
result=dict(at=time.time(),method='Identical BOLT30; equal completed ten-plan rounds within each paired query; exact uncensored incumbents only. Descriptive comparison, not VAE-only causal ablation.',pre_migration_cutoff=cutoff,new_tasks=len(new),old_tasks=len(old),read_errors=dict(new=ne,old=oe),all_saved=compare(new,old),pre_migration=compare(new,old,cutoff),all_new_diagnostics=diagnostics([e for s in new.values() for e in s['events']]),all_old_diagnostics=diagnostics([e for s in old.values() for e in s['events']]))
(OUT/'comparison.json').write_text(json.dumps(result,indent=2))
for name in ['all_saved','pre_migration']:
    data=result[name]['per_query']
    with (OUT/(name+'.csv')).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
print(json.dumps({k:({a:b for a,b in v.items() if a!='per_query'} if k in ['all_saved','pre_migration'] else v) for k,v in result.items()},indent=2))
