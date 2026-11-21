import json,os,subprocess,time,sys
from pathlib import Path
from query_protocol import atomic
ROOT=Path(__file__).resolve().parents[1]
PYTHON='/root/data/cys/baseline/envs/taskset_qwen35_fast_20261030/bin/python'
QUERY='/root/data/cys/envs/bolt_clean_venv/bin/python'
def launch(name,cmd,extra=None):
 env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
 env.update(extra or {})
 with (ROOT/'logs'/f'{name}.log').open('ab') as log:
  p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT,start_new_session=True)
 atomic(ROOT/'services'/f'{name}.process.json',dict(pid=p.pid,command=cmd,started=time.time()))
 print(name,p.pid,flush=True)
if sys.argv[1]=='replicas':
 for gpu in range(4):
  launch(f'replica{gpu}',[PYTHON,'-u',str(ROOT/'code/qwen_inference_server.py'),'replica','--config',str(ROOT/'services'/f'replica{gpu}.json')],dict(CUDA_VISIBLE_DEVICES=str(gpu),HF_QWEN_DEVICE_MAP='cuda:0'))
 launch('vae',[QUERY,'-u','/root/data/cys/llmoptimizer_benchmark/tools/query_vae_service.py','--gpu','3','--cpu','36','--port','21820','--token-file',str(ROOT/'services/token'),'--native','/root/data/cys/logs/query_train800_author_fixed_20261004/native','--memory-fraction','.02'])
elif sys.argv[1]=='brokers':
 for stage in ['warm_start','orpo']:
  launch(stage+'_broker',['/usr/bin/python3','-u',str(ROOT/'code/qwen_inference_server.py'),'broker','--config',str(ROOT/'services'/f'{stage}.broker.json')],dict(CUDA_VISIBLE_DEVICES=''))
