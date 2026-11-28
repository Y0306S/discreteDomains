"""Queue known problem queries after normal work; run on LLMOptimizer3."""
import hashlib,json,os,pickle,shutil,signal,sqlite3,time
from pathlib import Path
import fleet

stamp=str(int(time.time()));evidence=fleet.OPS/f'deferred_queue_{stamp}';evidence.mkdir()
entries=[('semanticopt','CEB_11A122','PostgreSQL temporary-file I/O error; audit unfinished SQL before retry'),
         ('bolt','CEB_10A32','Low unique decoder yield; preserve proposal checkpoint and revisit after other queries')]
for kind,task,reason in entries:
 root=fleet.ROOTS[kind];marker=root/'task_pauses'/f'{task}.json';old=fleet.load(marker,{})
 if marker.exists():shutil.copy2(marker,evidence/f'{kind}_{task}.previous.json')
 fleet.atomic(marker,dict(**{k:v for k,v in old.items() if k not in ('at','status','reason')},
    status='deferred',reason=reason,queue_position='after_all_non_deferred_tasks',requires_audit_before_retry=True,user_instruction='Defer known issues until all ordinary queued work finishes'))
 queuepath=root/'deferred_tasks.json';queue=fleet.load(queuepath,dict(tasks=[]))
 queue['tasks']=[x for x in queue['tasks'] if x['task']!=task]+[dict(task=task,reason=reason,requires_audit_before_retry=True)]
 queue['policy']='Finish all non-deferred tasks first, then review and resume these saved tasks'
 fleet.atomic(queuepath,queue)

root=fleet.ROOTS['bolt'];target='CEB_10A32'
workers=[fleet.load(p) for p in (root/'services').glob('worker*.status.json')]
w=next((w for w in workers if w.get('task')==target and w.get('status')=='running'),None)
if w:
 assert w['host']=='LLMOptimizer3';pid=w['child_pid'];ticks=fleet.process_ticks(pid)
 assert ticks and str(fleet.OPS/'fleet.py').encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
 spec=next(s for s in fleet.configuration('bolt')[0] if s['slot']==w['slot'])
 for attempt in range(50):
  if fleet.process_ticks(pid)!=ticks:break
  os.kill(pid,signal.SIGSTOP)
  safe=False
  try:
   with sqlite3.connect('file:'+str(root/'runs'/target/'runtime.sqlite')+'?mode=ro',uri=True,timeout=.2) as db:
    values=list(db.execute('select key,value from kv'))
   pending=[k for k,v in values if k.startswith('measurement/') and pickle.loads(v).get('status') not in ('complete','safe_failed')]
   if not pending:
    fleet.database_check(spec);safe=True
  except Exception:pass
  if safe:
   fleet.atomic(evidence/'bolt_boundary.json',dict(worker=w,pending_sql=[],measurement_hashes={k:hashlib.sha256(v).hexdigest() for k,v in values if k.startswith('measurement/')}))
   os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
   print('Yielded BOLT slot',w['slot'],'with no pending SQL; proposal checkpoint retained',flush=True)
   break
  os.kill(pid,signal.SIGCONT);time.sleep(1)
 else:raise RuntimeError('No safe boundary found; deferral marker saved, worker still finishing current task')
fleet.atomic(evidence/'result.json',dict(deferred=[dict(kind=k,task=t,reason=r) for k,t,r in entries]))
print('Deferred queues saved:',[(k,t) for k,t,_ in entries],flush=True)
