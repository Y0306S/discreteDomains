import fcntl,json,pickle,shutil,sqlite3,time
from pathlib import Path
import fleet
archive=fleet.OPS/'transport_failures'
results=[]
for kind,root in fleet.ROOTS.items():
 for marker in sorted((root/'task_pauses').glob('*.json')):
  task=marker.stem;folder=fleet.directory(kind,task)
  leasepath=root/'leases'/f'{task}.lock' if kind=='semanticopt' else folder/'task.lock'
  try:lease=fleet.lock(leasepath)
  except BlockingIOError:raise RuntimeError('Quarantined task is still leased: '+task)
  try:
   error=fleet.load(marker);log=Path(error['log']).read_text()
   if 'Connection refused' not in log:raise RuntimeError('Not a transport failure: '+task)
   dbpath=folder/('observations.sqlite' if kind=='semanticopt' else 'runtime.sqlite')
   unresolved=[]
   if dbpath.exists():
    with sqlite3.connect('file:'+str(dbpath)+'?mode=ro',uri=True) as connection:
     if kind=='semanticopt':
      for key,value in connection.execute('select key,value from state'):
       value=json.loads(value)
       if key=='oracle_batch' and value and value.get('status')=='in_flight':unresolved.append(key)
     else:
      for key,value in connection.execute("select key,value from kv where key like 'measurement/%'"):
       if pickle.loads(value).get('status')!='complete':unresolved.append(key)
   if unresolved:raise RuntimeError('Unresolved measurement: '+str((kind,task,unresolved)))
   for path in [marker,folder/'error.json']:
    if not path.exists():continue
    dest=archive/kind/path.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(path,dest);path.unlink()
   results.append(dict(kind=kind,task=task,unresolved_measurements=unresolved))
  finally:lease.close()
fleet.atomic(fleet.OPS/'transport_tasks_released.json',dict(tasks=results))
print('Released',len(results),'tasks after checking no unresolved SQL measurements')
