"""Exercise the actual framed worker, compact restore, sampling and restart."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

sys.path.insert(0, '/root/data/cys/llmoptimizer_benchmark/tools')
sys.path.insert(0, '/root/data/cys/baseline/repos/transfer-hpo-framework')
import numpy as np
from peptide_transfer_models import RGPEBackend, send, receive
from peptide_rgpe_merged import sample_ensemble

ROOT = Path(__file__).resolve().parent
TOOLS = Path('/root/data/cys/llmoptimizer_benchmark/tools')
CAMPAIGN = Path('/root/data/logs/peptide_frozen_local_recenter_test200_20261101_v1/rgpe')


def main():
    os.sched_setaffinity(0, {122})
    recipe = json.loads((CAMPAIGN/'recipe.json').read_text())
    fixture = pickle.loads(Path('/root/reports/peptide_transfer_wrapper_check_20261028_v1/rgpe_math_inputs.pkl').read_bytes())
    backend = RGPEBackend(recipe)
    backend.restore(fixture['model_snapshot'])
    backend.ensemble.weights_ = np.zeros(len(backend.bank))
    backend.ensemble.weights_[-1] = 1.
    snapshot = backend.snapshot()
    x = np.random.RandomState(11).uniform(size=(5000, 256))
    x[-1] = x[0]
    expected = sample_ensemble(backend.ensemble, x, 50, 413)
    results = []
    for restart in range(2):
        directory = ROOT/('wire'+str(restart)); directory.mkdir(exist_ok=True)
        command = [recipe['model_python'], str(TOOLS/'peptide_rgpe_merged_worker.py'),
            '--amendment', str(ROOT/'staging_amendment.json'),
            '--telemetry', str(directory/'rgpe_acquisition.json'),
            '--recipe', recipe['parent_recipe']]
        with (directory/'stderr.log').open('wb') as log:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log)
            def request(op, **kwargs):
                send(child.stdin, dict(kwargs, op=op))
                reply=receive(child.stdout)
                assert reply is not None and 'error' not in reply, reply
                return reply.get('result')
            try:
                request('restore', payload=snapshot)
                actual = request('sample', candidates=x, count=50, seed=413)
                np.testing.assert_array_equal(actual['draws'], expected['draws'])
                saved = request('snapshot')
                request('restore', payload=saved)
                repeated = request('sample', candidates=x, count=50, seed=413)
                np.testing.assert_array_equal(actual['draws'], repeated['draws'])
                telemetry = json.loads((directory/'rgpe_acquisition.json').read_text())
                marker = json.loads((directory/'rgpe_worker.json').read_text())
                assert telemetry['info']['covariance_factorizations'] == 1
                assert marker['implementation'] == 'tiled-independent-rgpe-v1'
                request('close')
            finally:
                child.stdin.close(); child.stdout.close()
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.terminate(); child.wait(timeout=10)
            assert child.returncode == 0
            results.append(dict(restart=restart, passed=True, telemetry=telemetry))
    report = dict(passed=True, worker_restarts=2, compact_restore_passed=True,
        seeded_draws_identical=True, candidate_count=5000, batch_size=50,
        oracle_calls=0, vae_updates=False, results=results)
    (ROOT/'wire.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
