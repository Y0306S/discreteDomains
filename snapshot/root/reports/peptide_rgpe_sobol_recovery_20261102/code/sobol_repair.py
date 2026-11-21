"""Preserve the legacy normal transform; extend only the uniform Sobol backend."""
import atexit
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
TOOLS = Path('/root/data/cys/llmoptimizer_benchmark/tools')
MODERN_PYTHON = '/root/data/cys/envs/bolt_clean_venv/bin/python'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(path):
    manifest = json.loads(Path(path).read_text())
    for name, expected in manifest['code'].items():
        if digest(name) != expected:
            raise ValueError('Sobol repair code changed: '+name)
    for name, expected in manifest['dependencies'].items():
        if digest(name) != expected:
            raise ValueError('Sobol repair dependency changed: '+name)
    report = json.loads(Path(manifest['validation']['path']).read_text())
    if (digest(manifest['validation']['path']) != manifest['validation']['sha256']
            or not report['passed'] or report['code'] != manifest['code']):
        raise ValueError('Sobol repair has not passed validation')
    return manifest


class UniformClient:
    def __init__(self):
        self.child = None
        self.cache = OrderedDict()
        self.draws = 0
        atexit.register(self.close)

    def draw(self, dimension, count, seed, offset):
        import numpy as np
        from sobol_helper import receive, send
        key = (dimension, count, seed, offset)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key].copy()
        if self.child is None:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            self.child = subprocess.Popen([MODERN_PYTHON, str(ROOT/'sobol_helper.py')],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        send(self.child.stdin, dict(dimension=dimension, count=count, seed=seed, offset=offset))
        reply = receive(self.child.stdout)
        if reply is None or 'error' in reply:
            raise RuntimeError('Sobol helper failed: '+repr(reply))
        values = reply['values']
        if values.shape != (count, dimension) or not np.isfinite(values).all():
            raise ValueError('Invalid extended Sobol draw')
        # Box-Muller requires strictly positive radii; rounding float64 uniforms
        # to float32 must not create zero or one.
        values = np.clip(values, np.nextafter(np.float32(0), np.float32(1)),
                         np.nextafter(np.float32(1), np.float32(0)))
        self.draws += 1
        if self.draws == 1:
            print('[sobol-repair] completed extended QMC draw: dimension=%d samples=%d' % (dimension,count), file=sys.stderr, flush=True)
        self.cache[key] = values.copy()
        if len(self.cache) > 4:
            self.cache.popitem(last=False)
        return values

    def close(self):
        child, self.child = self.child, None
        if child is None:
            return
        try:
            child.stdin.close()
            child.wait(timeout=10)
        except Exception:
            child.terminate()
            child.wait(timeout=10)


def install():
    import torch
    import botorch.sampling.qmc as module
    original = module.NormalQMCEngine
    client = UniformClient()

    class ExtendedSobol:
        def __init__(self, dimension, seed):
            self.dimension, self.seed, self.offset = dimension, seed, 0

        def draw(self, n, dtype=torch.float32):
            values = client.draw(self.dimension, n, self.seed, self.offset)
            self.offset += n
            return torch.from_numpy(values).to(dtype=dtype)

    class ExtendedNormal(original):
        def __init__(self, d, seed=None, inv_transform=False):
            dimension = d if inv_transform else 2*((d+1)//2)
            if dimension <= torch.quasirandom.SobolEngine.MAXDIM:
                super(ExtendedNormal, self).__init__(d, seed, inv_transform)
            else:
                if seed is None:
                    raise ValueError('Extended RGPE Sobol draws require an explicit seed')
                self._d, self._seed, self._inv_transform = d, seed, inv_transform
                self._sobol_engine = ExtendedSobol(dimension, seed)

    module.NormalQMCEngine = ExtendedNormal
    return client
