"""Framed CPU-only Sobol generator in the existing modern SciPy environment."""
import math
import pickle
import struct
import sys
import numpy as np


def receive(stream):
    header = stream.read(8)
    if not header:
        return None
    size = struct.unpack('!Q', header)[0]
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size-len(data))
        if not chunk:
            raise EOFError('Incomplete Sobol message')
        data.extend(chunk)
    return pickle.loads(data)


def send(stream, value):
    data = pickle.dumps(value, protocol=4)
    stream.write(struct.pack('!Q', len(data))+data)
    stream.flush()


def uniforms(dimension, count, seed, offset=0):
    from scipy.stats import qmc
    # Independent scrambled blocks only if a future history exceeds SciPy's
    # direction table. The caller still applies ONE full posterior Cholesky.
    blocks = []
    for index, start in enumerate(range(0, dimension, qmc.Sobol.MAXDIM)):
        width = min(dimension-start, qmc.Sobol.MAXDIM)
        block_seed = seed if index == 0 else (seed+104729*index) % (2**32)
        engine = qmc.Sobol(width, scramble=True, seed=block_seed)
        total = offset+count
        values = engine.random_base2(int(math.ceil(math.log2(max(1, total)))))
        blocks.append(values[offset:total].astype(np.float32))
    return np.concatenate(blocks, axis=1)


def main():
    while True:
        request = receive(sys.stdin.buffer)
        if request is None:
            break
        try:
            send(sys.stdout.buffer, {'values': uniforms(**request)})
        except Exception as exc:
            send(sys.stdout.buffer, {'error': repr(exc)})


if __name__ == '__main__':
    main()
