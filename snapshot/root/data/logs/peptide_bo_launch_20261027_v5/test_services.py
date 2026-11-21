import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

os.sched_setaffinity(0, {0})
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
sys.path.insert(0, '/root/data/logs/peptide_bo_release_20261027_v5')
import torch
from mindware.domains.protein.peptide_rpc import RemoteObjective
from mindware.domains.protein.verified_bo import validate_sequence

torch.set_num_threads(1)
recipe = json.loads(Path('/root/data/logs/peptide_bo_train800_20261027_v5b/recipe.json').read_text())
weights = torch.load(recipe['vae'], map_location='cpu', weights_only=False)
sequences = json.loads(Path(recipe['prepared_pool']).read_text())['pep_000']['sequences'][:4]
results = []
for port in (18871, 18872, 18873):
    objective = SimpleNamespace(vae_forward=None, vae=SimpleNamespace(state_dict=lambda: weights))
    client = RemoteObjective(objective, f'http://127.0.0.1:{port}', f'remote4-service-check:{port}')
    started = time.monotonic()
    with torch.no_grad():
        z, loss = client.forward(sequences)
        decoded = client.decode(z.reshape(4, -1))
        scores = client.oracle(decoded)
    assert z.numel() == 4 * 256 and torch.isfinite(z).all() and len(decoded) == 4
    assert len(scores) == 4 and torch.isfinite(torch.tensor(scores)).all()
    for sequence in decoded:
        validate_sequence(sequence)
    client.close()
    results.append(dict(port=port, passed=True, seconds=time.monotonic()-started,
        decoded=decoded, scores=scores.tolist()))
path = Path('/root/data/logs/peptide_bo_launch_20261027_v5/extra_services_remote_check.json')
path.write_text(json.dumps(results, indent=2))
print(json.dumps(results), flush=True)
