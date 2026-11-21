"""Durable serialized peptide -> query training, gated on completed ML RL and smoke test."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from protocol import ROOT, atomic, digest

PYTHON='/root/data/cys/baseline/envs/taskset_qwen35_fast_20261030/bin/python'
RL=Path('/root/data/cys/baseline/artifacts/ml_orpo_grpo_cycle1_20261107/services/pipeline_status.json')

def main():
    lock=(ROOT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity=json.loads((ROOT/'queue_identity.json').read_text())
    for relative, expected in identity['files'].items():
        if digest(ROOT/relative)!=expected:raise ValueError('Queued recipe changed: '+relative)
    def publish(**values):
        atomic(ROOT/'queue_status.json',dict(pid=os.getpid(),at=time.time(),**values))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',DS_BUILD_OPS='0')
    for domain in ['peptide','query']:
        complete=ROOT/domain/'model/training_complete.json'
        if complete.exists():
            result=json.loads(complete.read_text())
            if result['status']!='completed' or not result['saved_adapter_matches_live'] or result['adapter_sha256']!=digest(complete.parent/'final_adapter/adapter_model.safetensors'):
                raise ValueError('Completed artifact audit failed')
            continue
        while True:
            if json.loads(RL.read_text()).get('status')!='complete':
                publish(status='waiting_for_ml_rl',next_domain=domain);time.sleep(15);continue
            smoke=ROOT/'query/smoke_status.json'
            if not smoke.exists() or json.loads(smoke.read_text()).get('status')!='completed':
                publish(status='waiting_for_distributed_update_test',next_domain=domain);time.sleep(15);continue
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
            free=[int(v) for v in raw.split()]
            if len(free)!=4 or min(free)<54*1024:
                publish(status='waiting_for_gpu_memory',next_domain=domain,free_mib=free);time.sleep(15);continue
            break
        preflight=json.loads((ROOT/domain/'data/preflight.json').read_text())
        if not preflight['passed'] or preflight['pilot']:raise ValueError('Production preflight not ready')
        command=[PYTHON,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(ROOT/'code/train.py'),'train',domain]
        with (ROOT/'logs'/(domain+'_training.log')).open('ab') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            publish(status='running',domain=domain,child_pid=process.pid,gpus=[0,1,2,3],
                queued_domains=['query'] if domain=='peptide' else [],expected_steps=preflight['optimizer_steps'])
            code=process.wait()
        if code!=0:
            publish(status='failed',domain=domain,returncode=code,log=str(ROOT/'logs'/(domain+'_training.log')));return 1
        if not complete.exists():raise ValueError('Training returned without verified completion artifact')
    publish(status='complete',domains=['peptide','query']);return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as error:
        atomic(ROOT/'queue_error.json',dict(error=repr(error),at=time.time()));raise
