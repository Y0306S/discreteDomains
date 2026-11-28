"""Dedicated CUDA posterior service; existing services and code remain untouched."""
import sys
sys.path.insert(0, '/root/data/cys/rgpe_exact_acceleration_20261104/code')
import cuda_service
from fast_math import Math


class PilotMath(Math):
    def __init__(self):
        super().__init__()
        # Enough for the retained 80-source prediction states, within the same
        # 25% GPU allocator bound used by the parent experiment.
        self.cache_limit_bytes = 18 * 2**30


cuda_service.Math = PilotMath
if __name__ == '__main__':
    cuda_service.main()
