"""Known-failure recovery only; preserve proposals, scores and experiment policy."""
import argparse,fcntl,hashlib,json,os,pickle,shlex,signal,socket,sqlite3,subprocess,sys,time,traceback
from pathlib import Path
OPS=Path(__file__).resolve().parent
ROOT=OPS.parents[1]
PYTHON='/root/data/cys/envs/bolt_clean_venv/bin/python'
sys.path.insert(0,str(ROOT/'batch_code'))
from query_protocol import atomic,digest
from resources import record_alive
MONITOR=Path('/root/.codex/loops/query-training-hourly-20261112')
ENV=dict(os.environ,PYTHONHASHSEED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')

def load(path,default=None):
 try:return json.loads(Path(path).read_text())
 except FileNotFoundError:return default

def command(args,timeout=120):
 p=subprocess.run(args,capture_output=True,text=True,timeout=timeout,env=ENV)
 if p.returncode:raise RuntimeError(f'Command failed {args[:3]}: '+(p.stdout+p.stderr)[-5000:])
 return p.stdout

def host_call(host,action):
 args=[PYTHON,'-u',str(OPS/'recovery.py'),action,'--host',host]
 if host!='LLMOptimizer2':args=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','LLMOptimizer4',shlex.join(args)]
 return json.loads(command(args,timeout=600))

def process_matches(record):
 if not record or not record_alive(record):return False
 try:
  cmd=Path(f"/proc/{record['pid']}/cmdline").read_bytes().split(b'\0')
  expected=record.get('command')
  if expected and cmd[:len(expected)]!=[str(v).encode() for v in expected]:return False
  return True
 except OSError:return False

def postgres_probe(w):
 import psycopg
 result=dict(index=w['index'],port=w['db_port'],pgdata=w['pgdata'])
 try:
  with psycopg.connect(host='127.0.0.1',port=w['db_port'],dbname='imdb',user='imdb',password='imdb',autocommit=True,connect_timeout=3,options='-c statement_timeout=5000') as db:
   directory=db.execute('SHOW data_directory').fetchone()[0]
   if directory!=w['pgdata']:raise ValueError('Database identity differs')
   active=db.execute("SELECT pid,backend_type,query_start,query FROM pg_stat_activity WHERE pid<>pg_backend_pid() AND state='active'").fetchall()
   result.update(ready=True,start=db.execute('SELECT pg_postmaster_start_time()').fetchone()[0].timestamp(),active=[dict(pid=p,backend_type=b,at=t.timestamp() if t else None,query=q) for p,b,t,q in active])
 except psycopg.OperationalError as exc:result.update(ready=False,error=str(exc))
 return result

def probe(host):
 cfg=load(ROOT/'resources.json');workers=[];databases=[]
 for w in cfg['workers']:
  if w['host']!=host:continue
  record=load(ROOT/'services'/f"batch_worker{w['index']}.process.json",{})
  matches=record_alive(record) if record.get('pid') else False
  if matches:
   cmd=Path(f"/proc/{record['pid']}/cmdline").read_bytes().split(b'\0')
   matches=str(ROOT/'batch_code/run.py').encode() in cmd and b'worker' in cmd
  workers.append(dict(index=w['index'],alive=matches,record=record,status=load(ROOT/'services'/f"batch_worker{w['index']}.status.json",{})))
  databases.append(postgres_probe(w))
 known={w['record'].get('pid') for w in workers};unrecorded=[]
 for proc in Path('/proc').glob('[0-9]*/cmdline'):
  try:
   args=proc.read_bytes().split(b'\0')
   if str(ROOT/'batch_code/run.py').encode() in args and b'worker' in args and int(proc.parent.name) not in known:unrecorded.append(int(proc.parent.name))
  except OSError:pass
 transport=None
 if host=='LLMOptimizer4':
  try:
   import urllib.request
   active=load(ROOT/'services/active_inference_stage.json');port=21810 if active['stage']=='warm30' else 21811
   with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=10) as response:h=json.load(response)
   transport=dict(ready=bool(h.get('ready')),replicas=h.get('replicas'))
  except Exception as exc:transport=dict(ready=False,error=repr(exc))
 return dict(host=host,at=time.time(),workers=workers,databases=databases,unrecorded_workers=unrecorded,transport=transport)

def pause_age():
 pause=ROOT/'PAUSE'
 if not pause.exists():return -1
 stat=pause.stat();owned=load(OPS/'recovery_pause.json',{})
 # Workers touch the existing pause file in their BaseException handler
 # when drained with SIGTERM. That does not begin a new pause interval.
 since=owned.get('pause_observed_at',stat.st_mtime) if owned.get('pause_inode')==stat.st_ino else stat.st_mtime
 return time.time()-since

def drain_generation(host):
 # The campaign has observed PAUSE for at least five minutes. Its SQL paths
 # check PAUSE before every new measurement, so idle databases cannot receive
 # another measurement from a generation-blocked worker.
 assert pause_age()>=290
 result=probe(host);assert not result['unrecorded_workers']
 assert all(d['ready'] and not d['active'] for d in result['databases'])
 stopped=[]
 for worker in result['workers']:
  if worker['alive'] and record_alive(worker['record']):
   os.kill(worker['record']['pid'],signal.SIGTERM);stopped.append(worker['record']['pid'])
 return dict(host=host,stopped_generation_workers=stopped,at=time.time())

def ensure_databases(host):
 cfg=load(ROOT/'resources.json');actions=[]
 for w in cfg['workers']:
  if w['host']!=host:continue
  status=postgres_probe(w)
  if status['ready']:
   if status['active']:raise RuntimeError('Database still has active SQL')
   continue
  d=Path(w['pgdata']);assert (d/'PG_VERSION').read_text().strip()=='16'
  pidfile=d/'postmaster.pid'
  if pidfile.exists():
   pid=pidfile.read_text().splitlines()[0];cmd=Path(f'/proc/{pid}/cmdline')
   if cmd.exists() and str(d).encode() in cmd.read_bytes():raise RuntimeError('Existing PostgreSQL is alive but unavailable; no blind restart')
  with socket.socket() as sock:
   if sock.connect_ex(('127.0.0.1',w['db_port']))==0:raise RuntimeError('Database port occupied by an unverified service')
  opts=shlex.split((d/'postmaster.opts').read_text())
  assert opts[0]=='/root/data/cys/envs/pg16/bin/postgres' and opts[opts.index('-D')+1]==str(d)
  assert int(opts[opts.index('-p')+1])==w['db_port']
  sockdir=Path(opts[opts.index('-k')+1]);sockdir.mkdir(mode=0o700,exist_ok=True);os.chown(sockdir,10000,10000)
  cmd=['setpriv','--reuid=10000','--regid=10000','--clear-groups','taskset','-c',','.join(map(str,w['postgres_cpus'])),'/root/data/cys/envs/pg16/bin/pg_ctl','-D',str(d),'-l',str(d/'semanticopt.log'),'-o',shlex.join(opts[1:]),'-w','-t','180','start']
  output=command(cmd,timeout=200);actions.append(dict(index=w['index'],action='restored_existing_database',output=output))
 from resources import host_prepare
 # host_prepare prints JSON; callers consume the last output line.
 import contextlib,io
 with contextlib.redirect_stdout(io.StringIO()):host_prepare(host)
 return dict(host=host,actions=actions,probe=probe(host))

def ensure_monitor():
 state=load(MONITOR/'state.json',{})
 if process_matches(dict(state,command=['taskset','-c','38','python3','-u',str(MONITOR/'monitor.py')])):return 'alive'
 # taskset execs python, so match its final command and recorded start time.
 if state.get('pid') and record_alive(state):
  args=Path(f"/proc/{state['pid']}/cmdline").read_bytes().split(b'\0')
  if str(MONITOR/'monitor.py').encode() in args:return 'alive'
 with (MONITOR/'monitor.log').open('ab') as log:p=subprocess.Popen(['taskset','-c','38','python3','-u',str(MONITOR/'monitor.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=ENV)
 return dict(started=p.pid)

def verify_pins():
 for p,h in load(ROOT/'evidence/batch_code_pins.json').items():
  if digest(p)!=h:raise RuntimeError('Pinned dependency changed; code repair needs review: '+p)

def error_allowed(text):
 return any(s in text for s in ['RemoteDisconnected','ConnectionResetError','ConnectionRefusedError','ConnectionTimeout','QueryInfrastructureRetryExhausted','UnresolvedMeasurement','URLError','Service process stopped','Remote transport is down','Remote worker startup failed'])

def classify(controller_alive,status,paused,errors,services_ok,worker_count,expected_count,now,managed_pause=False):
 if status.get('status')=='complete':return 'complete'
 if errors and any(not error_allowed(e.get('error','')) for e in errors):return 'held_unknown_error'
 if paused and not errors and not managed_pause:return 'held_manual_pause'
 if controller_alive and status.get('status')=='loading_inference' and now-status.get('at',0)<900:return 'loading'
 if controller_alive and not paused and not errors and status.get('status')=='running' and worker_count==expected_count:
  # A single deep HTTP probe can time out under load while every worker
  # continues to commit measurements. Workers have bounded transport retries
  # and will raise a durable error if the service is actually unavailable.
  return 'healthy' if services_ok else 'degraded_health'
 return 'recover'

def audit_journals(folder,probes):
 # Called only after every optimizer exited and all private databases report
 # no active SQL. Repeating an unscored read-only SELECT is then unambiguous.
 assert all(not w['alive'] for p in probes for w in p['workers'])
 assert all(not p.get('unrecorded_workers') for p in probes)
 assert all(d['ready'] and not d['active'] for p in probes for d in p['databases'])
 from query_data import canonicalize,plan_key
 audits=[];count=0
 for stage in ['warm30','orpo_batch10']:
  for path in (ROOT/stage/'runs').glob('*/runtime.sqlite'):
   with sqlite3.connect(path,timeout=120) as db:
    entries={k:pickle.loads(v) for k,v in db.execute('SELECT key,value FROM kv')}
    for state_key in ['warm_state','optimization_state']:
     if state_key not in entries:continue
     rows=entries.get(state_key,{}).get('rows',[]);seen=set()
     for row in rows:
      plan=canonicalize(path.parent.name,row['plan'],strict=True);key=plan_key(plan)
      if key in seen:raise ValueError('Saved rows contain duplicates')
      seen.add(key)
     if state_key=='warm_state' or stage=='orpo_batch10':count+=len(rows)-(30 if state_key=='optimization_state' else 0)
    pending=[(k,v) for k,v in entries.items() if 'measurement/' in k and isinstance(v,dict) and v.get('status')=='started']
    if not pending:continue
    backup=folder/'journals'/stage/path.parent.name/'runtime.sqlite';backup.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(backup) as target:db.backup(target)
    for key,entry in pending:
     plan=canonicalize(path.parent.name,entry['argument']['plan'],strict=True)
     assert plan==entry['argument']['plan']
     audit=dict(journal=str(path),key=key,original=entry,at=time.time(),decision='Exact frozen read-only plan may be remeasured: every old optimizer is gone and every database has no active SQL. No saved score is inferred.')
     audits.append(audit)
     recovered=dict(entry,status='safe_failed',recovery_audit=str(folder/'journal_audit.json'),at=time.time())
     db.execute('UPDATE kv SET value=? WHERE key=?',(pickle.dumps(recovered,protocol=4),key))
    db.commit()
 atomic(folder/'journal_audit.json',dict(at=time.time(),counted_saved_rows=count,operations=audits,quiescence=probes))
 return dict(saved_rows=count,audited_operations=len(audits))

def run_tick():
 lock=(OPS/'tick.lock').open('a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:return dict(result='recovery_already_running',at=time.time())
 cfg=load(OPS/'config.json');now=time.time()
 if not cfg['enabled'] or (OPS/'STOP').exists():return dict(result='disabled',at=now)
 status=load(ROOT/'status.json',{});controller=load(ROOT/'services/batch_controller.process.json',{})
 monitor=ensure_monitor();local=probe('LLMOptimizer2');remote=host_call('LLMOptimizer4','probe');probes=[local,remote]
 workers=[w for p in probes for w in p['workers']];alive_workers=sum(w['alive'] for w in workers)
 if any(p['unrecorded_workers'] for p in probes):return dict(at=now,result='held_unrecorded_worker',probes=probes)
 errors=[(p,load(p)) for stage in ['warm30','orpo_batch10'] for p in (ROOT/stage/'runs').glob('*/error.json')]
 resource_error=load(ROOT/'services/resource_error.json')
 relevant_errors=[v for _,v in errors]
 if resource_error and resource_error.get('at',0)>=controller.get('at',0):relevant_errors.append(resource_error)
 try:
  from resources import health
  health();services_ok=all(d['ready'] for p in probes for d in p['databases']) and bool(remote['transport']['ready']);service_error=None
 except Exception as exc:services_ok=False;service_error=repr(exc)
 managed_pause=load(OPS/'recovery_pause.json',{})
 pause=ROOT/'PAUSE'
 managed_pause=managed_pause.get('pause_inode')==pause.stat().st_ino if pause.exists() else False
 decision=classify(process_matches(controller),status,(ROOT/'PAUSE').exists(),relevant_errors,services_ok,alive_workers,len(workers),now,bool(managed_pause))
 base=dict(at=now,result=decision,counted_measurements=status.get('counted_measurements'),live_workers=alive_workers,expected_workers=len(workers),monitor=monitor,service_error=service_error)
 if decision!='recover':return base
 verify_pins()
 history=load(OPS/'failure_budget.json',dict(failures=0,progress=-1))
 progress=status.get('counted_measurements',0)
 failures=0 if progress>history['progress'] else history['failures']
 if failures>=cfg['max_failed_recoveries_without_progress']:return dict(base,result='held_repeated_failure',failure_budget=history)
 history=dict(failures=failures+1,progress=progress,at=now);atomic(OPS/'failure_budget.json',history)
 folder=OPS/'incidents'/str(time.time_ns());folder.mkdir(parents=True)
 atomic(folder/'before.json',dict(base,status=status,controller=controller,probes=probes,errors=relevant_errors))
 if not (ROOT/'PAUSE').exists():(ROOT/'PAUSE').touch()
 previous=load(OPS/'recovery_pause.json',{});pause_stat=(ROOT/'PAUSE').stat()
 observed=previous.get('pause_observed_at',now) if previous.get('pause_inode')==pause_stat.st_ino else now
 atomic(OPS/'recovery_pause.json',dict(incident=str(folder),at=now,reason=decision,pause_inode=pause_stat.st_ino,pause_observed_at=observed))
 # Let SQL finish within its existing 50-second cap, then leave at durable
 # boundaries. Never kill optimizers to force a retry of uncertain SQL.
 deadline=time.monotonic()+300
 while True:
  probes=[probe('LLMOptimizer2'),host_call('LLMOptimizer4','probe')]
  if any(p['unrecorded_workers'] for p in probes):raise RuntimeError('Unrecorded optimizer exists; refusing duplicate worker launch')
  still_alive=any(w['alive'] for p in probes for w in p['workers'])
  if not still_alive and not process_matches(controller):break
  if not still_alive and process_matches(controller):
   # A controller with no workers may be waiting to notice PAUSE; give it a tick.
   if time.monotonic()>deadline:
    if not all(d['ready'] and not d['active'] for p in probes for d in p['databases']):raise RuntimeError('Controller has not drained and database quiescence is unverified')
    os.kill(controller['pid'],signal.SIGTERM);deadline=time.monotonic()+30
  if time.monotonic()>deadline:
   if still_alive and all(d['ready'] and not d['active'] for p in probes for d in p['databases']):
    # A worker may touch PAUSE after this recovery started. Its most recent
    # touch must still age five minutes before we terminate generation.
    pause=ROOT/'PAUSE'
    if not pause.exists():raise RuntimeError('PAUSE disappeared during optimizer drain')
    remaining=290-pause_age()
    if remaining>0:
     deadline=time.monotonic()+remaining+2
     continue
    # Only generation-blocked workers are eligible; retain all pending SQL
    # entries for the journal audit after their verified exit.
    for host in ['LLMOptimizer2','LLMOptimizer4']:host_call(host,'drain')
    deadline=time.monotonic()+30
   else:return dict(base,result='draining',incident=str(folder))
  time.sleep(5)
 for host in ['LLMOptimizer2','LLMOptimizer4']:host_call(host,'prepare')
 probes=[probe('LLMOptimizer2'),host_call('LLMOptimizer4','probe')]
 journal_audit=audit_journals(folder,probes)
 from resources import vae_services,inference_stage,tunnel,start,json_get
 # After optimizer drain, replace only VAE processes whose HTTP health fails.
 # Require the saved process identity before sending any signal.
 for vae in load(ROOT/'resources.json')['vae']:
  record=load(ROOT/'services'/f"{vae['name']}_resources.process.json",{})
  if not process_matches(record):continue
  try:
   response=json_get(f"http://127.0.0.1:{vae['port']}/health")
   if response.get('status')!='ready':raise RuntimeError('VAE unavailable')
  except Exception:
   os.kill(record['pid'],signal.SIGTERM);deadline=time.monotonic()+30
   while process_matches(record):
    if time.monotonic()>deadline:raise RuntimeError('Unhealthy VAE did not stop')
    time.sleep(.2)
 vae_services()
 tunnel_record=load(ROOT/'services/resources_tunnel.process.json',{})
 if not process_matches(tunnel_record):tunnel()
 else:
  # Detect a live SSH process with failed forwarding. Do not disturb a healthy one.
  try:command(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','LLMOptimizer4',PYTHON+' -c '+shlex.quote("import urllib.request;urllib.request.urlopen('http://127.0.0.1:21820/health',timeout=10).read()")],timeout=30)
  except RuntimeError:
   if not process_matches(tunnel_record):raise
   os.kill(tunnel_record['pid'],signal.SIGTERM)
   deadline=time.monotonic()+30
   while process_matches(tunnel_record):
    if time.monotonic()>deadline:raise RuntimeError('Failed transport did not stop')
    time.sleep(.2)
   tunnel()
 stage='warm30'
 from run import validate_completion,recipe
 if all(validate_completion('warm30',t) for t in recipe()['tasks']):stage='orpo_batch10'
 inference_stage(stage)
 # Wait for all VAE loads and transport before allowing any new SQL.
 deadline=time.monotonic()+180
 while True:
  try:health();break
  except Exception:
   if time.monotonic()>deadline:raise
   time.sleep(2)
 command([PYTHON,str(ROOT/'batch_code/validate_launch.py')],timeout=180)
 for path,_ in errors:
  target=folder/'archived_errors'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);path.rename(target)
 if (ROOT/'services/resource_error.json').exists():(ROOT/'services/resource_error.json').rename(folder/'resource_error.json')
 # Holds are explicit and survive a watchdog check; a stop arriving during
 # recovery must prevent controller launch.
 if (OPS/'STOP').exists() or not load(OPS/'config.json')['enabled']:return dict(base,result='disabled_after_drain',incident=str(folder))
 (ROOT/'PAUSE').unlink()
 (OPS/'recovery_pause.json').unlink(missing_ok=True)
 pid=start([PYTHON,'-u',str(ROOT/'batch_code/run.py'),'controller'],'batch_controller')
 record=load(ROOT/'services/batch_controller.process.json')
 atomic(ROOT/'evidence/resources_resume.json',dict(controller=record,measurements_before=progress,evaluators=21,llm_replicas=2,vae_replicas=10,at=time.time(),automatic_recovery=str(folder)))
 # Launch acknowledgements and fresh controller status prove one controller
 # owns this continuation; the next check also compares measurement progress.
 deadline=time.monotonic()+120
 while time.monotonic()<deadline:
  if not process_matches(record):raise RuntimeError('Recovered controller exited; see logs/batch_controller.log')
  after=load(ROOT/'status.json',{})
  if after.get('at',0)>record['at'] and after.get('status')=='running' and after.get('worker_count')==21:
   result=dict(base,result='resumed',controller_pid=pid,incident=str(folder),journal_audit=journal_audit,after=after)
   atomic(folder/'after.json',result);return result
  time.sleep(2)
 return dict(base,result='restarted_loading',controller_pid=pid,incident=str(folder),journal_audit=journal_audit)

def tick():
 try:result=run_tick()
 except Exception as exc:
  result=dict(result='held_recovery_error',error=repr(exc),traceback=traceback.format_exc(),at=time.time())
 atomic(OPS/'local_latest.json',result)
 with (OPS/'local_history.jsonl').open('a') as log:log.write(json.dumps(result)+'\n');log.flush();os.fsync(log.fileno())
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['tick','probe','prepare','drain']);p.add_argument('--host',choices=['LLMOptimizer2','LLMOptimizer4']);a=p.parse_args()
 value=tick() if a.action=='tick' else probe(a.host) if a.action=='probe' else drain_generation(a.host) if a.action=='drain' else ensure_databases(a.host)
 print(json.dumps(value,allow_nan=False))
