"""Exact GP arithmetic: reuse LOO factorizations and merge posteriors on CUDA."""
import importlib.util
import time
from pathlib import Path
import numpy as np
import torch

LEGACY = Path('/root/data/logs/peptide_gpu_acceleration_20261102/code/cuda_math.py')
spec = importlib.util.spec_from_file_location('rgpe_legacy_cuda_math', LEGACY)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


class Math(legacy.Math):
    def __init__(self):
        super().__init__()
        torch.cuda.set_per_process_memory_fraction(.25)
        self.cache_bytes = 0
        self.cache_limit_bytes = 12 * 2**30
        self.cache_hits = self.cache_misses = 0

    def source(self, task):
        if task in self.sources:
            self.cache_hits += 1
            self.sources.move_to_end(task)
            return self.sources[task]
        self.cache_misses += 1
        # Reuse the original pinned loader and its provenance verification, but
        # retain a byte-bounded cache instead of an unconditional 12-model limit.
        saved = self.sources
        self.sources = type(saved)()
        try:
            state = super().source(task)
        finally:
            self.sources = saved
        size = sum(v.numel()*v.element_size() for v in state.values() if torch.is_tensor(v))
        while saved and self.cache_bytes + size > self.cache_limit_bytes:
            _, old = saved.popitem(last=False)
            self.cache_bytes -= sum(v.numel()*v.element_size() for v in old.values() if torch.is_tensor(v))
        if size <= self.cache_limit_bytes:
            saved[task] = state
            self.cache_bytes += size
        return state

    @staticmethod
    def rank_one_cholesky(base, whitened):
        """L chol(I+u u.T), evaluated in O(n²), with positive diagonals."""
        cumulative = 1 + whitened.square().cumsum(0)
        previous = torch.cat((torch.ones_like(cumulative[:1]), cumulative[:-1]))
        diagonal = (cumulative/previous).sqrt()
        coefficient = whitened/(previous*cumulative).sqrt()
        products = base*whitened[None, :]
        tails = products.flip((1,)).cumsum(1).flip((1,)) - products
        return base*diagonal[None, :] + tails*coefficient[None, :]

    def loo_reuse(self, d):
        started = time.monotonic()
        x, y, theta, normal = (self.tensor(d[k]) for k in ('x','y','theta','normal'))
        n = len(y)
        if n < 3:
            return super().run(dict(d, op='loo'))
        kernel = self.kernel(x, x, theta, True)
        chol = torch.linalg.cholesky(kernel)
        inverse = torch.cholesky_inverse(chol)
        noise = theta[-1].exp()
        # S=K-noise*I, B=S K^-1. At these training locations the full
        # predictive covariance is 2*noise*I-noise²*K^-1. This form avoids
        # subtracting two almost equal dense signal covariance matrices.
        identity = torch.eye(n, dtype=torch.float64, device=x.device)
        influence = identity - noise*inverse
        full_covariance = 2*noise*identity - noise.square()*inverse
        base, base_jitter = self.factor(full_covariance)
        if base_jitter:
            # Preserve the old per-fold jitter policy if the shared base cannot
            # be factored without jitter. This fallback is intentionally exact.
            return super().run(dict(d, op='loo'))
        vectors = influence/inverse.diagonal().sqrt()[None, :]
        whitened = torch.linalg.solve_triangular(base, vectors, upper=False)
        alpha = inverse@y
        inverse_one = inverse.sum(1)
        mean_full = influence@y
        influence_one = influence.sum(1)
        total = torch.zeros(normal.shape[0], dtype=torch.int64, device=x.device)
        indices = torch.arange(n, device=x.device)
        for i in d.get('fold_indices', range(n)):
            keep = indices != i
            raw = y[keep]
            offset = raw.mean()
            scale = raw.std(unbiased=False)
            if float(scale) == 0:
                scale = torch.ones_like(scale)
            mean = (mean_full + offset*(1-influence_one)
                    - influence[:, i]*(alpha[i]-offset*inverse_one[i])/inverse[i,i])
            factor = self.rank_one_cholesky(base, whitened[:, i])
            draws = mean + (normal@factor.T)*scale
            total += ((draws[:,i:i+1] < draws) ^ (y[i] < y)[None,:]).sum(1)
        return dict(loss=total.cpu().numpy(), seconds=time.monotonic()-started,
                    implementation='factorization-reuse-rank-one-loo-v1',
                    observations=n, folds=len(d.get('fold_indices',range(n))),
                    shared_factorizations=2, fold_dense_factorizations=0)

    def merge(self, d):
        """Keep source units, weight² covariances and full joint batch semantics."""
        started = time.monotonic()
        x = self.tensor(d['x'])
        n = len(x)
        mean = torch.zeros(n,dtype=torch.float64,device=x.device)
        covariance = torch.zeros((n,n),dtype=torch.float64,device=x.device)
        weights = np.asarray(d['weights'],dtype=np.float64)
        if (len(weights) != len(d['sources'])+1 or not np.isfinite(weights).all()
                or np.any(weights < 0) or weights.sum() <= 0):
            raise ValueError('Invalid RGPE merge weights')
        weights = weights/weights.sum()
        before_hits, before_misses = self.cache_hits, self.cache_misses
        for index in np.flatnonzero(weights):
            state = (self.source(d['sources'][index]) if index < len(d['sources'])
                     else {k:self.tensor(v) if k in ('train','theta','L','alpha') else v
                           for k,v in d['target'].items()})
            weight = float(weights[index])
            train, theta = state['train'], state['theta']
            cross = self.kernel(x,train,theta)
            mean.add_(cross@state['alpha'],alpha=weight)
            projected = torch.linalg.solve_triangular(state['L'],cross.T,upper=False)
            # Use the original direct-distance kernel and signed covariance;
            # do not clip off-diagonal entries or drop low-weight models.
            source_covariance = self.kernel(x,x,theta,True)
            source_covariance.addmm_(projected.T,projected,beta=1.,alpha=-1.)
            covariance.add_(source_covariance,alpha=weight*weight)
            del cross, projected, source_covariance
        scale, offset = float(d['scale']),float(d['offset'])
        if not np.isfinite(scale) or scale <= 0 or not np.isfinite(offset):
            raise ValueError('Invalid target output scaling')
        mean.mul_(scale).add_(offset)
        covariance.mul_(scale*scale)
        if not torch.isfinite(mean).all() or not torch.isfinite(covariance).all():
            raise ValueError('Nonfinite merged posterior')
        torch.cuda.synchronize()
        return dict(mean=mean.cpu().numpy(),covariance=covariance.cpu().numpy(),
                    seconds=time.monotonic()-started,implementation='cuda-fp64-full-rgpe-merge-v1',
                    contributing_models=int(np.count_nonzero(weights)),
                    cache_hits=self.cache_hits-before_hits,cache_misses=self.cache_misses-before_misses)

    def run(self, d):
        if d['op'] == 'loo_reference':
            return super().run(dict(d,op='loo'))
        if d['op'] == 'loo':
            return self.loo_reuse(d)
        if d['op'] == 'merge':
            return self.merge(d)
        if d['op'] == 'health':
            return dict(device=torch.cuda.get_device_name(0),dtype='float64',
                        allocated_gib=torch.cuda.memory_allocated()/2**30,
                        peak_gib=torch.cuda.max_memory_allocated()/2**30,
                        source_cache_gib=self.cache_bytes/2**30,
                        source_cache_models=len(self.sources),cache_hits=self.cache_hits,
                        cache_misses=self.cache_misses,implementation='exact-rgpe-cuda-v2')
        return super().run(d)
