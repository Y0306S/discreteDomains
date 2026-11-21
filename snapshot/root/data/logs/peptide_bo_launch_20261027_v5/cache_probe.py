"""Exercise the task-bank count required by the expanded remote worker pool."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, required=True)
parser.add_argument('--require-fixed', action='store_true')
args = parser.parse_args()
os.sched_setaffinity(0, {0})
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
    MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
sys.path.insert(0, '/root/data/logs/peptide_bo_release_20261027_v5')
import torch
from mindware.domains.protein.peptide_rpc import RemoteObjective
torch.set_num_threads(1)
recipe = json.loads(Path('/root/data/logs/peptide_bo_train800_20261027_v5b/recipe.json').read_text())
weights = torch.load(recipe['vae'], map_location='cpu', weights_only=False)
sequences = json.loads(Path(recipe['prepared_pool']).read_text())['pep_000']['sequences'][:4]
clients = [RemoteObjective(SimpleNamespace(vae_forward=None),
    f'http://127.0.0.1:{args.port}', f'cache-capacity-probe:{i}') for i in range(48)]
try:
    for client in clients:
        client.request(dict(op='weights', version='probe', weights=weights))
    result = clients[0].request(dict(op='encode', version='probe', seed=12345, sequences=sequences))
    retained = not result.get('missing_weights', False)
    if retained:
        assert torch.isfinite(result['z']).all() and result['z'].numel() == 1024
        decoded = clients[0].request(dict(op='decode', version='probe', seed=54321,
            z=result['z'].reshape(4, -1)))['sequences']
        repeated = clients[0].request(dict(op='decode', version='probe', seed=54321,
            z=result['z'].reshape(4, -1)))['sequences']
        assert decoded == repeated and len(decoded) == 4
        scores = clients[0].oracle(decoded)
        assert len(scores) == 4 and torch.isfinite(torch.tensor(scores)).all()
        # Updating one task must not leave every old version resident.
        clients[0].request(dict(op='weights', version='probe-new', weights=weights))
        old = clients[0].request(dict(op='encode', version='probe', seed=12346, sequences=sequences))
        assert old.get('missing_weights')
    report = dict(port=args.port, capacity_48_retained=retained,
        real_encode_decode_oracle=retained, repeat_reply_verified=retained)
    Path(f'/root/data/logs/peptide_bo_launch_20261027_v5/cache_probe_{args.port}_{"fixed" if args.require_fixed else "before"}.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if args.require_fixed:
        assert retained, 'The oldest bank was evicted within the worker capacity'
finally:
    for client in clients:
        client.close()
