"""Read-only evaluation telemetry, refreshed by the existing query controller."""
import json
import sqlite3
import time
from collections import Counter
from protocol import atomic
from common import domain_recipe


def update_progress(root,recipe,*,now=None,activate=False,force=False):
    recipe=domain_recipe(recipe,'query')
    now=time.time() if now is None else now
    folder=root/'query';private=folder/'timing_state.json';public=folder/'progress.json'
    previous=json.loads(private.read_text()) if private.exists() else {}
    if not force and now-previous.get('last_update',0)<60:return
    phase=previous.get('phase_started_at')
    if activate and phase is None:phase=now
    counts=dict(previous.get('counts',{}));recent=previous.get('recent',[]);errors=[];started=0
    settings=recipe['query'];budget=settings['proposals'];tasks=settings['test']
    for task in tasks:
        dbpath=folder/'runs'/task/'observations.sqlite'
        if not dbpath.exists():continue
        started+=1
        try:
            db=sqlite3.connect('file:'+str(dbpath)+'?mode=ro',uri=True,timeout=.2)
            try:
                # One read transaction keeps the cursor and new rows consistent.
                db.execute('BEGIN')
                count=db.execute('SELECT count(*) FROM records').fetchone()[0]
                old=counts.get(task,0)
                if count<old:raise ValueError('Committed count decreased; telemetry requires audit')
                pending=[]
                for number,raw in db.execute('SELECT n,value FROM records WHERE n>=? ORDER BY n',(old,)):
                    value=json.loads(raw);at=value.get('at',0)
                    if phase is not None and at>=phase:
                        pending.append([at,value.get('oracle',{}).get('elapsed_seconds'),
                            value.get('proposal_source','historical'),value.get('status')])
                recent.extend(pending);counts[task]=count
            finally:db.close()
        except (OSError,sqlite3.Error,ValueError) as exc:errors.append(dict(task=task,error=str(exc)))
    recent=[x for x in recent if now-3600<=x[0]<=now]
    total=sum(counts.get(task,0) for task in tasks);target=len(tasks)*budget;remaining=max(0,target-total)
    paused=(root/'PAUSE').exists();complete=all(counts.get(task)==budget for task in tasks)
    state='unverified' if errors else 'complete' if complete else 'paused' if paused else 'running' if phase else 'queued'
    windows={}
    for minutes in [15,30,60]:
        seconds=min(minutes*60,max(0,now-phase)) if phase is not None else 0
        selected=[x for x in recent if x[0]>=now-seconds] if seconds else []
        rate=len(selected)*3600/seconds if seconds else 0
        enough=seconds>=300 and len(selected)>=25
        eta=remaining/rate if enough and state=='running' else 0 if complete else None
        windows[str(minutes)]=dict(observed_seconds=seconds,committed=len(selected),evaluations_per_hour=rate,
            remaining_hours=eta,estimated_completion_at=now+eta*3600 if eta is not None else None,
            confidence='provisional' if enough and seconds<3600 else 'measured_window' if enough else 'insufficient_data')
    measured=[x[1] for x in recent if isinstance(x[1],(int,float))]
    value=dict(at=now,status=state,phase_started_at=phase,batch_size=recipe['batch_policy']['size'],
        rounds=budget//recipe['batch_policy']['size'],proposals_per_task=budget,target=target,evaluations=total,
        remaining=remaining,tasks_started=started,tasks_complete=sum(counts.get(t)==budget for t in tasks),
        windows=windows,recent_oracle_samples=len(measured),recent_oracle_mean_seconds=sum(measured)/len(measured) if measured else None,
        recent_proposal_sources=dict(Counter(x[2] for x in recent)),recent_outcomes=dict(Counter(x[3] for x in recent)),
        last_hour_commits=len(recent),last_committed_at=max((x[0] for x in recent),default=None),read_errors=errors,
        budget_policy_sha256=recipe.get('budget_policy_sha256'),refresh_seconds=60,
        mechanism='existing LLMOptimizer2 query controller; no separate scheduler or automatic budget reduction',
        timing_limit='ETA includes measured production overhead; prompt prefill and decoding are not separately timed')
    atomic(private,dict(last_update=now,phase_started_at=phase,counts=counts,recent=recent))
    atomic(public,value)
    return value


def maybe_update_progress(root,recipe):
    # Reporting failures must not interrupt an evaluation or invent an ETA.
    try:return update_progress(root,recipe,activate=True)
    except Exception as exc:
        try:atomic(root/'query/progress_error.json',dict(at=time.time(),error=repr(exc)))
        except OSError:pass
