import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

from resources import snapshot

ROOT = Path('/root/data/logs/peptide_bo_launch_20261027_v5')
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, required=True)
parser.add_argument('--cpu', type=int, required=True)
parser.add_argument('--port', type=int, required=True)
args = parser.parse_args()
assert args.gpu in (0, 2, 3) and args.cpu in snapshot()['free_physical_cpus']
command = ['taskset', '-c', str(args.cpu), '/root/data/cys/envs/bolt_clean_venv/bin/python',
    str(ROOT / 'peptide_vae_server.py'),
    '--gpu', str(args.gpu), '--cpu', str(args.cpu), '--port', str(args.port), '--vae',
    '/root/data/cys/BOLT/optimization/peptides/uniref_vae/saved_models/dim128_k1_kl0001_eff256_dff256_pious-sea-2_model_state_epoch_118_pretrained_authors.pkl']
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
with (ROOT / f'gpu_service_{args.port}.log').open('ab') as log:
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
        stderr=subprocess.STDOUT, env=env, start_new_session=True)
record = dict(pid=process.pid, command=command, gpu=args.gpu, cpu=args.cpu, port=args.port)
(ROOT / f'gpu_service_{args.port}.json').write_text(json.dumps(record, indent=2))
deadline = time.monotonic() + 180
while True:
    if process.poll() is not None:
        raise RuntimeError('GPU service exited during startup')
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health', timeout=3) as reply:
            health = json.load(reply)
        assert health['gpu'] == args.gpu and health['errors'] == 0
        print(json.dumps(dict(record, health=health)), flush=True)
        break
    except OSError:
        if time.monotonic() >= deadline:
            raise RuntimeError('GPU service did not become ready')
        time.sleep(2)
