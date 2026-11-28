import json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
spec=json.loads((ROOT/'allocation.json').read_text())['bolt_workers'][0]
os.sched_setaffinity(0,{spec['optimizer_cpu']})
os.environ.update(DB_HOST='127.0.0.1',DB_PORT=str(spec['db_port']),DB_NAME='imdb',DB_USER='imdb',DB_PASSWORD='imdb',CUDA_VISIBLE_DEVICES='',QUERY_LOLBO_VAE_OFFSET='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import run_lolbo as run
if __name__=='__main__':
 task=run.recipe()['tasks'][0];started=time.time()
 try:
  run.run_task(task,max_rounds=1)
  progress=json.loads((ROOT/'runs'/task/'progress.json').read_text());assert progress['completed_rounds']==1 and progress['rows']==40
  run.atomic(ROOT/'evidence/canary.json',dict(passed=True,task=task,progress=progress,seconds=time.time()-started,at=time.time()))
  print('BOLT canary passed: ten new evaluated plans',flush=True)
 except BaseException as e:
  run.atomic(ROOT/'evidence/canary.json',dict(passed=False,error=repr(e),traceback=traceback.format_exc(),seconds=time.time()-started,at=time.time()));raise
