"""Resource-only expansion. Algorithm identities and journals are unchanged."""
import argparse,concurrent.futures,fcntl,json,os,signal,subprocess,time,urllib.request,socket
from pathlib import Path
from query_protocol import atomic
ROOT=Path(__file__).resolve().parents[1]
PYTHON='/root/data/cys/envs/bolt_clean_venv/bin/python'
INFER='/root/data/cys/baseline/envs/taskset_qwen35_fast_20261030/bin/python'

def config():return json.loads((ROOT/'resources.json').read_text())
def start(command,name):
 env=dict(os.environ,PYTHONHASHSEED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
 if 'replica' in command and '--config' in command:
  replica=json.loads(Path(command[command.index('--config')+1]).read_text());env.update(CUDA_VISIBLE_DEVICES=str(replica['gpu']),HF_QWEN_DEVICE_MAP='cuda:0')
 with (ROOT/'logs'/f'{name}.log').open('ab') as log:
  p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
 atomic(ROOT/'services'/f'{name}.process.json',dict(pid=p.pid,host='LLMOptimizer2',command=command,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().rsplit(') ',1)[1].split()[19],at=time.time()))
 return p.pid

def json_get(url):
 # VAE /health shares the decoder mutex. A legitimate constrained batch can
 # outlast 15 seconds; allow it to finish within the 600-second check cadence.
 with urllib.request.urlopen(url,timeout=90) as response:return json.load(response)
def alive(pid,start_ticks=None):
 try:
  fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
  return fields[0]!='Z' and (start_ticks is None or fields[19]==str(start_ticks))
 except (OSError,ValueError):return False

def record_alive(record):
 # PIDs are reused after an execution environment restart. A record without
 # its original process start time is insufficient evidence of a live service.
 return record.get('start_ticks') is not None and alive(record['pid'],record['start_ticks'])

def host_prepare(host):
 import psycopg
 cfg=config();workers=[w for w in cfg['workers'] if w['host']==host]
 quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split();assert quota=='max' or int(quota)/int(period)>=cfg['hosts'][host]['cpu_quota']
 # The old pilot only exposes bounded readiness requests, never an optimizer.
 # Audit every database first; do not repurpose an instance with active SQL.
 rows=[]
 for w in workers:
  with psycopg.connect(host='127.0.0.1',port=w['db_port'],dbname='imdb',user='imdb',password='imdb',connect_timeout=10,autocommit=True) as db:
   assert db.execute("SELECT count(*) FROM pg_stat_activity WHERE pid<>pg_backend_pid() AND state='active'").fetchone()[0]==0
   assert db.execute('SHOW data_directory').fetchone()[0]==w['pgdata']
   db.execute("LOAD 'pg_hint_plan'");assert db.execute('SHOW pg_hint_plan.enable_hint').fetchone()[0]=='on'
   assert db.execute("SELECT reltuples::bigint FROM pg_class WHERE oid='title'::regclass").fetchone()[0]>1000000
   db.execute("ALTER SYSTEM SET temp_file_limit='8GB'");db.execute('SELECT pg_reload_conf()')
   if w.get('temp_tablespace_dir'):
    from psycopg import sql
    location=Path(w['temp_tablespace_dir']);location.mkdir(parents=True,exist_ok=True);os.chown(location,10000,10000)
    existing=db.execute("SELECT pg_tablespace_location(oid) FROM pg_tablespace WHERE spcname='query_hybrid_temp'").fetchone()
    if existing:assert existing[0]==str(location)
    else:db.execute(sql.SQL('CREATE TABLESPACE query_hybrid_temp LOCATION {}').format(sql.Literal(str(location))))
    db.execute("ALTER SYSTEM SET temp_tablespaces='query_hybrid_temp'");db.execute('SELECT pg_reload_conf()')
    # Existing standalone FSBO databases carry a database-level override,
    # which takes precedence over ALTER SYSTEM for new IMDb sessions.
    db.execute("ALTER DATABASE imdb SET temp_tablespaces='query_hybrid_temp'")
    with psycopg.connect(host='127.0.0.1',port=w['db_port'],dbname='imdb',user='imdb',password='imdb',connect_timeout=10,autocommit=True) as verify:
     assert verify.execute('SHOW temp_tablespaces').fetchone()[0]=='query_hybrid_temp'
   pid=int((Path(w['pgdata'])/'postmaster.pid').read_text().splitlines()[0])
   # Rebind all existing auxiliaries; future backends inherit the postmaster.
   ppid={}
   for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:ppid[int(p.name)]=int((p/'stat').read_text().rsplit(') ',1)[1].split()[1])
    except (OSError,ValueError):pass
   descendants={pid}
   while True:
    new={p for p,parent in ppid.items() if parent in descendants}-descendants
    if not new:break
    descendants.update(new)
   for p in descendants:
    try:subprocess.run(['setpriv','--reuid=10000','--regid=10000','--clear-groups','taskset','-pc',','.join(map(str,w['postgres_cpus'])),str(p)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    except ProcessLookupError:pass
    except subprocess.CalledProcessError:
     if alive(p):raise
   rows.append(dict(index=w['index'],port=w['db_port'],pid=pid,cpus=sorted(os.sched_getaffinity(pid)),hint=True,pgdata=w['pgdata'],temp_tablespace_dir=w.get('temp_tablespace_dir')))
 if host=='LLMOptimizer4':
  retired=[]
  for p in Path('/proc').iterdir():
   if not p.name.isdigit():continue
   try:
    command=(p/'cmdline').read_bytes().split(b'\0')
    if any(x.endswith(b'/query_fsbo_pilot_worker.py') for x in command):
     os.kill(int(p.name),signal.SIGTERM);retired.append(int(p.name))
   except (OSError,ProcessLookupError):pass
 else:retired=[]
 result=dict(host=host,postgres=rows,retired_idle_readiness_services=retired,at=time.time(),passed=True)
 atomic(ROOT/'evidence'/f'resources.{host}.json',result);print(json.dumps(result),flush=True)

def vae_services():
 assert (ROOT/'PAUSE').exists(),'Drain SQL workers before changing the VAE pool'
 for w in config()['vae']:
  record=ROOT/'services'/f"{w['name']}_resources.process.json"
  if record.exists() and record_alive(json.loads(record.read_text())):continue
  start([PYTHON,'-u','/root/data/cys/llmoptimizer_benchmark/tools/query_vae_service.py','--gpu',str(w['gpu']),'--cpu',str(w['cpu']),'--port',str(w['port']),'--token-file',str(ROOT/'services/token'),'--native','/root/data/cys/logs/query_train800_author_fixed_20261004/native','--memory-fraction','.02'],w['name']+'_resources')
 atomic(ROOT/'services/vae_pool.json',dict(domain='ceb_query_runtime',token_file=str(ROOT/'services/token'),workers=[dict(url=f"http://127.0.0.1:{w['port']}") for w in config()['vae']]))

def inference_stage(stage):
 name='warm_start' if stage=='warm30' else 'orpo'
 active=ROOT/'services/active_inference_stage.json'
 if active.exists() and json.loads(active.read_text()).get('stage')==stage and json.loads(active.read_text()).get('placement_verified'):
  try:health(deep=True);return
  except (OSError,ValueError,RuntimeError):
   if not (ROOT/'PAUSE').exists():raise
   # A stage marker survives a container restart; it does not prove that the
   # recorded processes survived. Explicit paused recovery reloads that stage.
   atomic(ROOT/'evidence'/f'inference_recovery_{time.time_ns()}.json',dict(previous=json.loads(active.read_text()),reason='paused stage service health failed',at=time.time()))
 # Only called before workers start or after every worker has completed a stage.
 retired=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   cmd=(p/'cmdline').read_bytes().split(b'\0')
   if any(str(ROOT).encode() in x for x in cmd) and any(x.endswith(b'/qwen_inference_server.py') for x in cmd):
    os.kill(int(p.name),signal.SIGTERM);retired.append(int(p.name))
  except (OSError,ProcessLookupError):pass
 deadline=time.monotonic()+40
 while any(alive(p) for p in retired):
  if time.monotonic()>deadline:raise RuntimeError('LLM stage handoff did not drain')
  time.sleep(.2)
 # Wait for the old listening sockets to be released, not just process exit.
 for port in sorted({21810,21811}|{w['port'] for w in config()['llm']}):
  deadline=time.monotonic()+90
  while True:
   try:
    with socket.socket() as sock:
     sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);sock.bind(('127.0.0.1',port))
    break
   except OSError:
    if time.monotonic()>deadline:raise RuntimeError('Inference port still occupied '+str(port))
    time.sleep(1)
 for w in config()['llm']:
  start([INFER,'-u',str(ROOT/'batch_code/qwen_inference_server.py'),'replica','--config',str(ROOT/f"services/{name}.{w['name']}.resources.json")],w['name']+'_resources')
 deadline=time.monotonic()+600
 while True:
  for w in config()['llm']:
   r=json.loads((ROOT/'services'/f"{w['name']}_resources.process.json").read_text())
   if not record_alive(r):raise RuntimeError('LLM startup failed '+w['name'])
  try:
   hs=[json_get(f"http://127.0.0.1:{w['port']}/health") for w in config()['llm']]
   if all(h.get('ready') and h.get('placement',{}).get('embedding_device')=='cuda:0' and h['placement']['language_devices']==['cuda:0'] for h in hs):break
  except (OSError,ValueError):pass
  if time.monotonic()>deadline:raise RuntimeError('LLM loading timed out for '+stage)
  time.sleep(2)
 start(['/usr/bin/python3','-u',str(ROOT/'batch_code/qwen_inference_server.py'),'broker','--config',str(ROOT/f'services/{name}.resources.broker.json')],name+'_resources_broker')
 deadline=time.monotonic()+60
 while True:
  try:
   h=json_get('http://127.0.0.1:'+('21810' if stage=='warm30' else '21811')+'/health')
   if h.get('ready') and h.get('replicas')==len(config()['llm']):break
  except (OSError,ValueError):pass
  if time.monotonic()>deadline:raise RuntimeError('LLM broker startup timed out')
  time.sleep(1)
 atomic(active,dict(stage=stage,role=name,replicas=len(config()['llm']),retired=retired,placement_verified=True,at=time.time()))
 print(str(len(config()['llm']))+' LLM replicas ready for '+stage,flush=True)

def tunnel():
 ports=[21810,21811]+[w['port'] for w in config()['vae']]
 cmd=['ssh','-NT','-o','BatchMode=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3']
 for port in ports:cmd+=['-R',f'127.0.0.1:{port}:127.0.0.1:{port}']
 cmd+=['LLMOptimizer4'];start(cmd,'resources_tunnel')

def health(require_workers=False,deep=True):
 cfg=config();results=[];active=json.loads((ROOT/'services/active_inference_stage.json').read_text())
 for name in [w['name']+'_resources' for w in cfg['llm']+cfg['vae']]+[active['role']+'_resources_broker']:
  r=json.loads((ROOT/'services'/f'{name}.process.json').read_text())
  if not record_alive(r):raise RuntimeError('Service process stopped or identity changed '+name)
 if deep:
  urls=[f"http://127.0.0.1:{w['port']}/health" for w in cfg['vae']]+['http://127.0.0.1:'+('21810' if active['stage']=='warm30' else '21811')+'/health']
  with concurrent.futures.ThreadPoolExecutor(max_workers=len(urls)) as executor:results=list(executor.map(json_get,urls))
  for h in results[:len(cfg['vae'])]:
   if h.get('status')!='ready' or h.get('checkpoint_sha256')!='83a4942701bffaaf8311887e530e3157b6843134e52e9f53d59f957c24aebab1':raise ValueError('Wrong VAE identity or unavailable VAE')
  for h in results[len(cfg['vae']):]:
   if not h.get('ready') or h.get('replicas')!=len(cfg['llm']):raise ValueError('LLM broker unavailable or incorrect count')
   if len(h.get('replica_health',[]))!=len(cfg['llm']) or not all(r.get('ready') for r in h['replica_health']):raise ValueError('An LLM replica is unavailable')
   for replica in h['replica_health']:
    if replica.get('placement',{}).get('embedding_device')!='cuda:0' or replica['placement']['language_devices']!=['cuda:0']:raise ValueError('Incorrect LLM GPU placement')
 tunnel_record=json.loads((ROOT/'services/resources_tunnel.process.json').read_text())
 if not record_alive(tunnel_record):raise RuntimeError('Remote transport is down or identity changed')
 result=dict(passed=True,stage=active['stage'],deep=deep,vae=results[:len(cfg['vae'])],llm=results[len(cfg['vae']):],service_processes_alive=len(cfg['llm'])+len(cfg['vae'])+1,tunnel_alive=True,at=time.time())
 atomic(ROOT/'services/resources_health.json',result)
 if require_workers:
  result['workers']=[json.loads((ROOT/'services'/f"batch_worker{w['index']}.status.json").read_text()) for w in cfg['workers']]
 return result

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('command',choices=['host_prepare','vae_services','tunnel','health','inference_stage']);parser.add_argument('--host');parser.add_argument('--stage',choices=['warm30','orpo_batch10']);a=parser.parse_args()
 if a.command=='host_prepare':host_prepare(a.host)
 elif a.command=='vae_services':vae_services()
 elif a.command=='tunnel':tunnel()
 elif a.command=='inference_stage':inference_stage(a.stage)
 else:print(json.dumps(health(),indent=2))
