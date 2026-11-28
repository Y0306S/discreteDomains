"""Bounded GPU utilization and VAE request-counter sample on Optimizer3."""
from concurrent.futures import ThreadPoolExecutor
import statistics
import urllib.request
from common import *

require_host('LLMOptimizer3')
def health(i):
    with urllib.request.urlopen(f'http://127.0.0.1:{23300+i}/health',timeout=60) as r:return json.load(r)
def health_all():
    with ThreadPoolExecutor(max_workers=16) as executor:return list(executor.map(health,range(16)))
before=health_all();start=time.monotonic();samples=[]
for _ in range(15):
    output=subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total',
        '--format=csv,noheader,nounits'],text=True,timeout=10)
    samples.append([[float(x.strip()) for x in line.split(',')] for line in output.splitlines()])
    time.sleep(2)
after=health_all();elapsed=time.monotonic()-start
rows=[]
for i in range(16):
    vals=[next(r for r in sample if int(r[0])==i) for sample in samples]
    b=next(s for s in before if s['gpu']==i);a=next(s for s in after if s['gpu']==i)
    rows.append(dict(gpu=i,average_gpu_percent=statistics.mean(r[1] for r in vals),
        peak_gpu_percent=max(r[1] for r in vals),memory_mib=vals[-1][3],total_memory_mib=vals[-1][4],
        vae_requests=a['requests']-b['requests'],vae_errors=a['errors']-b['errors']))
result=dict(at=time.time(),seconds=elapsed,samples=len(samples),gpus=rows,
    mean_gpu_percent=statistics.mean(r['average_gpu_percent'] for r in rows),
    peak_gpu_percent=max(r['peak_gpu_percent'] for r in rows),
    vae_requests=sum(r['vae_requests'] for r in rows),vae_errors=sum(r['vae_errors'] for r in rows))
atomic(ROOT/'gpu_usage_latest.json',result);print(json.dumps(result))
