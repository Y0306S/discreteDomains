"""CUDA execution of the pinned peptide FSBO target-fitting algorithm."""

import hashlib
import io
import time
from pathlib import Path

import numpy as np

from peptide_transfer_models import FSBOBackend, checked_xy


def cpu_copy(value, torch):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item, torch) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_copy(item, torch) for item in value)
    return value


class CUDAFSBOBackend(FSBOBackend):
    def __init__(self, recipe, memory_fraction=0.08):
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError('Expose exactly one CUDA GPU to the FSBO service')
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        super().__init__(recipe)
        self.device = torch.device('cuda:0')
        self.model.to(self.device)
        self.model.device = self.device
        self.initial = cpu_copy(self.initial, torch)

    def fit(self, x, score, seed):
        from fsbo_target_refit import fit_target

        torch = self.torch
        x, score = checked_xy(x, score)
        self.offset = float(score.min())
        self.scale = float(score.max() - self.offset) or 1.0
        self.x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        self.y = torch.as_tensor((score - self.offset) / self.scale,
                                dtype=torch.float32, device=self.device)
        self.model.load_state_dict(self.initial, strict=True)
        settings = self.recipe.get('fsbo_target_stopping', {})
        torch.cuda.synchronize()
        started = time.monotonic()
        # The pinned likelihood seeds each probe. Also preserve the CUDA stream
        # around the complete fit; its inner CPU fork does not restore CUDA RNG.
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(seed)
            fit = fit_target(self.model, self.x, self.y,
                self.recipe['fsbo_target_epochs'],
                patience=settings.get('patience', self.recipe.get('fsbo_target_patience', 16)),
                min_delta=settings.get('min_delta', 0.0001),
                lr=self.recipe['fsbo_target_learning_rate'], seed=seed)
        torch.cuda.synchronize()
        self.fit_info = dict(fit, seconds=time.monotonic() - started,
            method='fsbo', target_observations=len(score), device='cuda',
            execution='shared_cuda_service', tf32_enabled=False,
            checkpoint_sha256=self.recipe['fsbo_weights_sha256'],
            label_scaling='target_minmax',
            stopping_code_sha256=hashlib.sha256(Path(__file__).with_name('fsbo_target_refit.py').read_bytes()).hexdigest(),
            amendment_sha256=self.recipe.get('target_fit_amendment_sha256'))
        return self.fit_info

    def sample(self, candidates, count, seed):
        torch = self.torch
        with torch.random.fork_rng(devices=[0]), torch.no_grad():
            torch.manual_seed(seed)
            x = torch.as_tensor(candidates, dtype=torch.float32, device=self.device)
            posterior = self.model.model(self.model.feature_extractor(x))
            samples = posterior.rsample(torch.Size([count])).cpu().numpy()
        if not np.isfinite(samples).all():
            raise ValueError('Nonfinite CUDA FSBO posterior draws')
        return dict(draws=samples * self.scale + self.offset,
            info=dict(method='fsbo', candidates=len(candidates), samples=count,
                      correlated=True, higher_is_better=True, device='cuda',
                      execution='shared_cuda_service', tf32_enabled=False))

    def snapshot(self):
        stream = io.BytesIO()
        self.torch.save(cpu_copy(dict(model=self.model.state_dict(), x=self.x, y=self.y,
            offset=self.offset, scale=self.scale, fit_info=self.fit_info), self.torch), stream)
        return stream.getvalue()

    def restore(self, payload):
        value = self.torch.load(io.BytesIO(payload), map_location='cpu')
        self.model.model.train()
        self.model.load_state_dict(value['model'], strict=True)
        self.x = value['x'].to(self.device)
        self.y = value['y'].to(self.device)
        self.offset, self.scale = value['offset'], value['scale']
        self.fit_info = value['fit_info']
        self.model.model.eval()
        self.model.likelihood.eval()
        self.model.feature_extractor.eval()
        with self.torch.no_grad():
            self.model.model.set_train_data(self.model.feature_extractor(self.x), self.y, strict=False)
        return self.fit_info
