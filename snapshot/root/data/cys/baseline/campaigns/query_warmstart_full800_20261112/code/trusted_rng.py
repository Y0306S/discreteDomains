"""Restore only this run's exact, locally generated one-rank RNG checkpoint."""
import hashlib
import io
from pathlib import Path
import random


def restore_rng(checkpoint, expected_checkpoint, expected_sha256):
    import numpy as np
    import torch

    if Path(checkpoint).resolve() != Path(expected_checkpoint).resolve():
        raise ValueError('Unexpected RNG checkpoint')
    path = Path(checkpoint) / 'rng_state.pth'
    if path.stat().st_size > 2**20:
        raise ValueError('Unexpected RNG checkpoint size')
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError('RNG checkpoint checksum changed')
    # This is our own checkpoint, pinned before loading. Do not relax the
    # Transformers guard or alter deserialization of any other file.
    state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
    if set(state) - {'python', 'numpy', 'cpu', 'cuda'} or not {'python', 'numpy', 'cpu'} <= set(state):
        raise ValueError('Unexpected one-rank RNG state')
    if state['cpu'].dtype != torch.uint8 or state['cpu'].ndim != 1:
        raise ValueError('Invalid CPU RNG state')
    if 'cuda' in state:
        if not torch.cuda.is_available() or len(state['cuda']) != 1:
            raise ValueError('Require exactly one visible CUDA RNG state')
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.random.set_rng_state(state['cpu'])
    if 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])
