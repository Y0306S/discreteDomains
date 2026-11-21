"""Experimental FP64 execution of the existing SMAC likelihood and priors."""
import contextlib
import os
from pathlib import Path
import subprocess
import time
from types import MethodType

import numpy as np

from peptide_rgpe_cuda_probe import send, receive
from peptide_rgpe_exact import kernel_parts


class CUDALikelihood:
    def __init__(self, python, gpu, log):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        self.child = subprocess.Popen([python, str(Path(__file__).with_name('peptide_rgpe_cuda_probe.py')), '--worker'],
                                     env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log)
        self.gp = None

    def evaluate(self, gp, theta):
        if self.gp is None:
            kernel_parts(gp.kernel_, gp.X_train_.shape[1])
            if gp.y_train_.ndim != 1 or gp.alpha != 0 or gp.normalize_y:
                raise ValueError('Unexpected SMAC likelihood setup')
            send(self.child.stdin, dict(x=gp.X_train_, y=gp.y_train_))
            self.gp = gp
        elif self.gp is not gp:
            raise ValueError('An objective worker cannot be shared between datasets')
        send(self.child.stdin, theta)
        reply = receive(self.child.stdout)
        return reply['value'], reply['gradient']

    def close(self):
        if self.child.poll() is None:
            if self.gp is None:
                self.child.terminate()
            else:
                send(self.child.stdin, None)
            self.child.wait(timeout=30)
        if self.child.returncode != 0:
            raise RuntimeError('CUDA likelihood worker failed')


def cuda_nll(model, theta):
    model._n_ll_evals += 1
    started = time.monotonic()
    state = model._cuda_objective_state
    value, gradient = state['backend'].evaluate(model.gp, theta)
    for dimension, priors in enumerate(model._all_priors):
        for prior in priors:
            value += prior.lnprob(theta[dimension])
            gradient[dimension] += prior.gradient(theta[dimension])
    state['evaluations'] += 1
    state['seconds'] += time.monotonic()-started
    if state['callback'] is not None and state['evaluations'] % 25 == 0:
        state['callback'](dict(evaluations=state['evaluations'], nll_seconds=state['seconds']))
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        return 1e25, np.zeros(theta.shape)
    return -value, -gradient


@contextlib.contextmanager
def cuda_objective(model, python, gpu, log, callback=None):
    kernel_parts(model.kernel, len(model.types))
    if '_nll' in model.__dict__ or '_cuda_objective_state' in model.__dict__:
        raise ValueError('Model already has an overridden objective')
    backend = CUDALikelihood(python, gpu, log)
    state = dict(backend=backend, evaluations=0, seconds=0., callback=callback)
    model._cuda_objective_state = state
    model._nll = MethodType(cuda_nll, model)
    try:
        yield state
    finally:
        del model._nll
        del model._cuda_objective_state
        backend.close()
