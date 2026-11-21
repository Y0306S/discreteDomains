#!/usr/bin/env python3
"""Audit all held-out initialization latent bounds and frozen mean stability."""

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
RELEASE = Path('/root/data/logs/peptide_bo_release_20261027_v5')


def main():
    os.sched_setaffinity(0, {17})
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(RELEASE))
    from mindware.domains.protein.peptide_rpc import RemoteObjective
    recipe = json.loads(Path('/root/data/logs/peptide_bo_train800_20261027_v5b/recipe.json').read_text())
    sys.path.insert(0, recipe['cpu_upstream'])
    os.chdir(recipe['cpu_upstream'])
    from uniref_vae.load_vae import load_vae
    vae, _ = load_vae(recipe['vae'], max_string_length=30)
    client = RemoteObjective(SimpleNamespace(vae=vae, vae_forward=None), 'http://127.0.0.1:18871', 'test200_bound_audit')
    pool = json.loads((ROOT / 'test200_revalidated_initialization.json').read_text())
    rows = []
    try:
        with torch.no_grad():
            for task in recipe['test']:
                torch.manual_seed(recipe['seed'])
                sequences = pool[task]['sequences']
                chunks = [client.inference('encode', sequences=sequences[i:i+64])['z'].reshape(-1, 256) for i in range(0, len(sequences), 64)]
                z = torch.cat(chunks)
                rows.append(dict(task=task, minimum=float(z.min()), maximum=float(z.max()), outside=int((z.abs() > 4).sum())))
                if len(rows) % 20 == 0:
                    print(json.dumps(dict(completed=len(rows), total=200, outside=sum(r['outside'] for r in rows))), flush=True)
            sequences = pool[recipe['test'][0]]['sequences'][:16]
            a = client.inference('encode_mean', sequences=sequences)['z'].reshape(16, 256)
            b = client.inference('encode_mean', sequences=sequences)['z'].reshape(16, 256)
            single = client.inference('encode_mean', sequences=sequences[:1])['z'].reshape(1, 256)
            if not torch.isfinite(a).all() or not torch.equal(a, b):
                raise ValueError('Frozen posterior mean is not stable')
        result = dict(passed=all(r['outside'] == 0 for r in rows), tasks=rows,
                      frozen_mean_repeated_identical=True, mean_single_batch_max_abs=float((a[:1] - single).abs().max()),
                      oracle_calls=0, representation='frozen_author_VAE_posterior_mean_256')
        temp = ROOT / 'test200_latent_bounds.json.tmp'
        temp.write_text(json.dumps(result, indent=2, allow_nan=False))
        temp.replace(ROOT / 'test200_latent_bounds.json')
    finally:
        client.close()


if __name__ == '__main__':
    main()
