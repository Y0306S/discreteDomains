"""Start only this pilot's dedicated inference services, recording process identities."""
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/root/data/cys/envs/bolt_clean_venv/bin/python'
OLD = ROOT.parent/'peptide_rgpe_screen200_test200_20261119'


def start(name, command, gpu):
    port = int(command[command.index('--port')+1])
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', port)) == 0:
            raise RuntimeError('Port already occupied: %s' % port)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    with (ROOT/'services'/(name+'.log')).open('ab') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    value = dict(pid=child.pid, command=command, started=time.time(), port=port,
                 start_ticks=Path('/proc/%d/stat'%child.pid).read_text().split()[21])
    (ROOT/'services'/(name+'.process.json')).write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    for gpu in range(4):
        start('math%d'%gpu, [PYTHON, '-u', str(ROOT/'code/math_service.py'),
              '--port', str(22540+gpu), '--cpu', str(96+gpu),
              '--token-file', str(ROOT/'services/token')], gpu)
    checkpoint = '/root/data/cys/BOLT/optimization/peptides/uniref_vae/saved_models/dim128_k1_kl0001_eff256_dff256_pious-sea-2_model_state_epoch_118_pretrained_authors.pkl'
    start('vae', [PYTHON, '-u', str(OLD/'code/vae_server.py'), '--gpu', '0', '--cpu', '100',
          '--port', '22560', '--vae', checkpoint, '--gpu-memory-fraction', '.04',
          '--max-weight-banks', '64', '--skip-identical-weights', '--http-keepalive'], 0)
