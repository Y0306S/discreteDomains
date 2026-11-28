import hashlib
import json
import os
from pathlib import Path
import subprocess
import socket
import time

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
BOLT = BASE/'query_bolt30_lolbo_total800_20261123'
SEM = BASE/'semanticopt_query_400_pilot800_20261123'
HYB = BASE/'query_our_fsbo_vae80_test200_20261117'
OPS = BASE/'query_continuation_operations_20261124'
PYTHON = '/root/data/cys/envs/bolt_clean_venv/bin/python'
HOSTNAMES = {'LLMOptimizer2':'cs-94bac-0693e-server',
             'LLMOptimizer3':'cs-0b2cd-18deb-server',
             'LLMOptimizer4':'cs-fa498-8793f-server',
             'LLMOptimizer5':'cs-123f1-07363-server'}


def require_host(host):
    if socket.gethostname()!=HOSTNAMES[host]:
        raise RuntimeError('This operation must execute on '+host)


def load(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ticks(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
        return fields[19] if fields[0]!='Z' else None
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(record):
    return bool(record and record.get('start_ticks') and ticks(record['pid'])==record['start_ticks'])


def command(pid):
    return Path(f'/proc/{pid}/cmdline').read_bytes().replace(bytes([0]),b' ').decode()


def launch(cmd, label, env=None):
    log=ROOT/'logs'/f'{label}.log'
    log.parent.mkdir(exist_ok=True)
    with log.open('ab',buffering=0) as stream:
        proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=stream,
                              stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=proc.pid,start_ticks=ticks(proc.pid),command=cmd,log=str(log),at=time.time())
    atomic(ROOT/'processes'/f'{label}.json',record)
    return record


def environment():
    return dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
                HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
