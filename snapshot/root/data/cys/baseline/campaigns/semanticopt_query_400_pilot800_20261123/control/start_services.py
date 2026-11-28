import argparse,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/root/data/cys/baseline/campaigns/semanticopt_query_400_pilot800_20261123')
HY=ROOT.parent/'query_our_fsbo_vae80_test200_20261117'
INFER='/root/data/cys/baseline/envs/taskset_qwen35_fast_20261030/bin/python';ORACLE='/root/data/cys/envs/bolt_clean_venv/bin/python'
def start(cmd,label,env):
 p=ROOT/'query/services'/f'{label}.process.json'
 if p.exists():
  old=json.loads(p.read_text())
  try:
   stat=Path(f'/proc/{old["pid"]}/stat').read_text().rsplit(') ',1)[1].split()
   if stat[19]==old['start_ticks'] and stat[0]!='Z':return old
  except FileNotFoundError:pass
 log=(ROOT/'logs'/f'{label}.log').open('ab',buffering=0)
 proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True);log.close()
 record=dict(pid=proc.pid,start_ticks=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(') ',1)[1].split()[19],at=time.time(),command=cmd)
 p.write_text(json.dumps(record,indent=2)+'\n');return record
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('host');p.add_argument('--broker',action='store_true');a=p.parse_args()
 env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 if a.broker:
  print(json.dumps(start([INFER,'-u',str(ROOT/'services/qwen_inference_server.py'),'broker','--config',str(ROOT/'query/services/broker.json')],'broker',env)))
 else:
  cfg=json.loads((ROOT/'allocation.json').read_text())
  for r in cfg['replicas']:
   if r['host']==a.host:print(json.dumps(start([INFER,'-u',str(ROOT/'services/qwen_inference_server.py'),'replica','--config',r['config']],f'replica{r["index"]:02}',env)),flush=True)
  if a.host=='LLMOptimizer3':
   for i,gpu in enumerate([15,4,6,3,8,10,12,5]):
    cmd=[ORACLE,'-u',str(HY/'batch_code/query_vae_service.py'),'--gpu',str(gpu),'--cpu',str(20+i),'--port',str(23300+i),'--token-file',str(HY/'services/token'),'--native','/root/data/cys/logs/query_train800_author_fixed_20261004/native','--checkpoint',str(HY/'training/vae/best.ckpt'),'--checkpoint-sha256','eebbe1ee70209aec6b8236d3c01d43c0d3591e52f9bbc23e311c610712e0204d','--memory-fraction','.02']
    print(json.dumps(start(cmd,f'bolt_vae{i:02}',env)),flush=True)
