"""Bounded real-checkpoint check: fresh CLI versus alternating HTTP studies."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'examples/benchmark'))
from mindware.components.optimizers.runtime_rpc import call, server, fingerprint


def traced_loader(original, trace):
    import jax
    import numpy as np
    def digest(tree):
        result = hashlib.sha256()
        for value in jax.tree_util.tree_leaves(tree):
            array = np.asarray(value)
            result.update(str((array.dtype, array.shape)).encode())
            result.update(array.tobytes())
        return result.hexdigest()
    def load(*args):
        model = original(*args)
        trace['params'] = digest(model._params)
        trace['logits'] = []
        compute = model._partitioned_compute_logits_from_batch
        def logits(params, batch, restricted):
            result = compute(params, batch, restricted)
            trace['logits'].append({'batch':digest(batch), 'result':digest(result)})
            return result
        model._partitioned_compute_logits_from_batch = logits
        return model
    return load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fresh', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        import optformer_worker as worker
        trace = {}
        worker.load_model = traced_loader(worker.load_model, trace)
        worker.configure_runtime()
        worker.cmd_suggest(argparse.Namespace(model_dir=args.model_dir,
            request=args.out/'fresh_request.json', out=args.out/'fresh_response.json'))
        (args.out/'fresh_trace.json').write_text(json.dumps(trace, indent=2)+'\n')
        return
    meta = json.loads((args.model_dir/'meta.json').read_text())
    records = []
    with open(meta['sequences']) as stream:
        for line in stream:
            record = json.loads(line)
            if len(record.get('configs', [])) >= 98:
                if not records or record['configspace'] != records[0]['configspace']:
                    records.append(record)
            if len(records) == 2:
                break
    if len(records) < 2:
        raise RuntimeError('Need two permitted histories with different schemas')
    requests = []
    for record, size in zip(records, (3, 98)):
        request = {k: record[k] for k in ('task_id', 'configspace', 'schema_fingerprint', 'algorithm_condition')}
        request.update(configs=record['configs'][:size], y_maximize=record['y_maximize'][:size],
                       seed=73, num_samples=100)
        requests.append(request)
    request_path = args.out/'fresh_request.json'
    request_path.write_text(json.dumps(requests[0]))
    start = time.monotonic()
    subprocess.run([sys.executable, str(Path(__file__)), '--fresh',
        '--model-dir', str(args.model_dir), '--out', str(args.out)], check=True, timeout=600)
    fresh_seconds = time.monotonic() - start
    from optformer_worker import configure_runtime, OptFormerEngine
    from mindware.components.optimizers.optformer_remote import OptFormerService
    configure_runtime()
    import jax
    if jax.default_backend() != 'gpu':
        raise RuntimeError('The OptFormer GPU check must not fall back to CPU')
    engine = OptFormerEngine(args.model_dir)
    trace = {}
    traced_loader(lambda: engine.model, trace)()
    app = OptFormerService(engine, args.out/'service.sqlite')
    httpd = server(app, 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    url = 'http://127.0.0.1:%d' % httpd.server_address[1]
    results, timings = [], []
    attempt = str(time.time_ns())
    try:
        for index in (0, 1, 0):
            started = time.monotonic()
            # Different request IDs force a genuine seeded replay after the other schema.
            envelope = {'identity': meta['checkpoint_sha256'], 'request_id': attempt + ':' + str(len(results)),
                        'request': requests[index]}
            results.append(call(url+'/rpc', envelope, timeout=600, retries=0))
            timings.append(time.monotonic()-started)
            (args.out/('persistent_trace%d.json'%len(results))).write_text(json.dumps(trace, indent=2)+'\n')
            trace['logits'].clear()
        fresh = json.loads((args.out/'fresh_response.json').read_text())
        if results[0] != results[2]:
            raise AssertionError('A different task changed the seeded persistent response')
        for key in fresh:
            if key != 'acquisition' and results[0][key] != fresh[key]:
                raise AssertionError('Persistent proposal differs from the fresh process: '+key)
        import numpy as np
        before = np.asarray(fresh['acquisition']['values'])
        after = np.asarray(results[0]['acquisition']['values'])
        np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(np.argsort(before), np.argsort(after))
        import jax
        stats = jax.local_devices()[0].memory_stats()
        report = {'passed': True, 'checkpoint': meta['checkpoint'], 'candidate_count': 100,
                  'histories': [3, 98, 3], 'fresh_seconds': fresh_seconds,
                  'persistent_seconds': timings, 'gpu_memory_stats': stats,
                  'identical_configurations': True, 'identical_persistent_replay': True,
                  'ei_max_absolute_difference': float(np.max(np.abs(before-after))),
                  'ei_max_relative_difference': float(np.max(np.abs(before-after)/np.maximum(np.abs(before),1e-30)))}
        (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2), flush=True)
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()
        app.connection.close()


if __name__ == '__main__':
    main()
