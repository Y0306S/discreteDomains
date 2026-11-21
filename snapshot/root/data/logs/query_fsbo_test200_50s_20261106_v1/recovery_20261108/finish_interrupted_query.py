from pathlib import Path
import json,os,signal,subprocess,sys,time,sqlite3,zlib
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
w=next(w for w in json.loads((ROOT/'current_workers.json').read_text())['workers'] if w['name']=='w03')
evidence=json.loads((HERE/'CEB_8A220_recovery.json').read_text())
run=ROOT/'runs/CEB_8A220'
with sqlite3.connect(f'file:{run}/artifacts.sqlite3?mode=ro',uri=True) as c:
 hashes=dict(c.execute('select name,sha256 from artifacts'))
 row=c.execute("select payload from artifacts where name='proposal_0109.json'").fetchone();assert row
 result=json.loads(zlib.decompress(row[0]));assert result['database_attempts']==2 and result['status']=='timeout'
assert all(hashes.get(n)==v for n,v in evidence['prior_committed_result_hashes'].items())
p=Path('/proc')/str(w['pid']);f=(p/'stat').read_text().split();assert f[21]==str(w['process_start_ticks']) and f[2]=='T'
env=dict(os.environ,PGPASSWORD='imdb',PGCONNECT_TIMEOUT='5',PGOPTIONS='-c statement_timeout=5000')
cmd=['/root/data/cys/envs/pg16/bin/psql','-h','127.0.0.1','-p',str(w['db_port']),'-U','imdb','-d','imdb','-X','-A','-t','-c',"SELECT count(*) FROM pg_stat_activity WHERE datname='imdb' AND pid<>pg_backend_pid();"]
x=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=10,check=True);assert x.stdout.strip()=='0'
os.killpg(w['pid'],signal.SIGTERM);os.kill(w['pid'],signal.SIGCONT)
for _ in range(100):
 if not p.exists() or (p/'stat').read_text().split()[2]=='Z':break
 time.sleep(.1)
else:raise RuntimeError('Fenced worker did not stop')
sys.path.insert(0,'/root/data/cys/llmoptimizer_benchmark/tools')
import query_fsbo_eval as e
recipe=e.verify(ROOT)
old=e.QueryVAEBackend.__init__
def init(self,manifest,task,fallback,seed=123,timeout=300):old(self,manifest,task,fallback,seed=seed,timeout=1800)
e.QueryVAEBackend.__init__=init
try:
 result=e.run_task(ROOT,recipe,'CEB_8A220',w,stop_after_batches=1)
 e.atomic_json(HERE/'CEB_8A220_recovery_verified.json',dict(passed=True,at=time.time(),result=result,original_results_preserved=True,failed_proposal_replayed_once=True))
 print(json.dumps(result),flush=True)
finally:
 subprocess.run(['/root/data/cys/envs/bolt_clean_venv/bin/python','-B',str(HERE/'restart_workers.py'),'--host','LLMOptimizer2','--worker','w03'],check=True)
