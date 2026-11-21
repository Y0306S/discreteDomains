"""Read-only campaign status; no inference or training."""
import json,time,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def read(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default

def main():
    recipe=read(ROOT/'rgpe/recipe.json')
    hosts=[read(p) for p in (ROOT/'dispatch').glob('*.status.json')]
    progress=[read(p) for p in (ROOT/'rgpe/runs').glob('*/progress.json')]
    progress=[p for p in progress if p]
    completed=[]
    for path in (ROOT/'rgpe/runs').glob('*.tar.gz'):
        with tarfile.open(path,'r:gz') as archive:
            completed.append(json.load(archive.extractfile('progress.json')))
    stages={p.parent.name:read(p) for p in (ROOT/'telemetry').glob('*/rgpe_gpu_stage.json')}
    gate=read(ROOT/'evidence/canary_passed.json')
    report=dict(at=time.time(),tasks=200,worker_cap=48,
        phase='production' if gate and gate.get('passed') else 'canary',
        complete=len(completed),started=len(progress)+len(completed),
        task_workers=sum(len(h['active']) for h in hosts if h),hosts=hosts,
        proposals=sum(p['proposals'] for p in progress+completed),
        fresh_evaluations=sum(p['fresh_evaluations'] for p in progress+completed),
        canary=read(ROOT/'evidence/canary.process.json'),gate=gate,
        canary_failure=read(ROOT/'evidence/canary_failed.json'),stages=stages,
        acquisition_benchmark=read(ROOT/'evidence/matched_speed.json'),
        vae_frozen=recipe['vae_policy']=='frozen_author' and recipe['e2e_every']==0)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
