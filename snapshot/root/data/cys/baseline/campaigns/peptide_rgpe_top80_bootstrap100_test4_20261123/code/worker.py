"""Isolated top-80/bootstrap-100 model process with per-round evidence."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), '/root/data/cys/llmoptimizer_benchmark/tools',
    '/root/data/cys/rgpe_exact_acceleration_20261104/code',
    '/root/data/cys/baseline/repos/transfer-hpo-framework']
from source_screen import atomic, digest


def check_pins():
    for name, expected in json.loads((ROOT/'pins.json').read_text()).items():
        if digest(name) != expected:
            raise ValueError('Pinned dependency changed: '+name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--task', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only model worker required')
    check_pins()
    wire = sys.stdout.buffer
    sys.stdout = sys.stderr
    import peptide_transfer_models as models
    from rgpe_cuda_client import Client
    from backend import Backend
    from screening import shortlist
    client = Client(args.endpoint, ROOT/'services/token')
    directory = ROOT/'telemetry'/args.task
    directory.mkdir(parents=True, exist_ok=True)
    client.telemetry = directory/'rgpe_gpu_stage.json'
    backend = Backend(json.loads((ROOT/'rgpe/recipe.json').read_text()), args.task)
    fit_seed = None
    while True:
        req = models.receive(sys.stdin.buffer)
        if req is None:
            break
        try:
            op = req['op']
            if op == 'close':
                models.send(wire, dict(ok=True))
                break
            if op == 'fit':
                fit_seed = req['seed']
                result = backend.fit(req['x'], req['score'], fit_seed)
            elif op == 'screen':
                result = shortlist(backend.bank.target, req['candidates'], req['best_mic'],
                    backend.ensemble.Y_mean_, backend.ensemble.Y_std_, 200)
            elif op == 'sample':
                result = client.sample(backend.ensemble, req['candidates'], req['count'], req['seed'])
            elif op == 'snapshot':
                result = backend.snapshot()
            elif op == 'restore':
                result = backend.restore(req['payload'])
            else:
                raise ValueError('Unknown operation: '+op)
            if op in ('fit', 'screen', 'sample'):
                info = result.get('info', result)
                record = dict(at=time.time(), info=info)
                atomic(directory/('rgpe_'+op+'.json'), record)
                atomic(directory/('%s_seed%s.json'%(op, fit_seed)), record)
            models.send(wire, dict(result=result))
        except Exception:
            models.send(wire, dict(error=traceback.format_exc()))


if __name__ == '__main__':
    main()
