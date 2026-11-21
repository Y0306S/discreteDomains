#!/usr/bin/env python3
"""Bounded pipeline: wait for the export, then train both query model artifacts."""
import json
import hashlib
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
TOOLS = Path('/root/data/cys/llmoptimizer_benchmark/tools')
sys.path.insert(0, str(TOOLS))
from peptide_transfer import atomic_json, lock


def verify_artifacts():
    from query_transfer import load_corpus
    manifest, corpus = load_corpus(ROOT / 'corpus')
    corpus.close()
    status = json.loads((ROOT / 'fsbo/status.json').read_text())
    weights = ROOT / 'fsbo/weights'
    if (status['status'] != 'complete' or status['step'] != 100000
            or not status['reload_verified']
            or status['weights_sha256'] != hashlib.sha256(weights.read_bytes()).hexdigest()
            or status['identity']['domain'] != 'ceb_query_runtime'
            or status['identity']['acquisition_wrapper'] is not None):
        raise ValueError('FSBO final checkpoint validation failed')
    verified = []
    with sqlite3.connect('file:' + str(ROOT / 'rgpe/models.sqlite') + '?mode=ro', uri=True) as bank:
        for task, metadata, checksum, payload in bank.execute('SELECT task,metadata,sha256,payload FROM models'):
            meta = json.loads(metadata)
            if (task not in manifest['source_tasks'] or not meta['reload_verified']
                    or meta['domain'] != 'ceb_query_runtime' or meta['acquisition_wrapper'] is not None
                    or meta['source_payload_sha256'] != manifest['task_hashes'][task]
                    or hashlib.sha256(payload).hexdigest() != checksum):
                raise ValueError('RGPE final bank validation failed: ' + task)
            verified.append(task)
    if set(verified) != set(manifest['source_tasks']):
        raise ValueError('Incomplete query RGPE source bank')
    atomic_json(ROOT / 'final_validation.json', dict(passed=True, source_tasks=len(verified),
        observations=manifest['observations'], excluded_tasks=len(manifest['excluded_tasks']),
        fsbo_step=status['step'], fsbo_best_step=status['best_step'],
        fsbo_weights_sha256=status['weights_sha256'], acquisition_wrapper=None,
        corpus_sha256=hashlib.sha256((ROOT / 'corpus/manifest.json').read_bytes()).hexdigest()))


def main():
    with lock(ROOT / 'supervisor.lock'):
        state = dict(status='waiting_for_corpus', pid=os.getpid(), host=socket.gethostname(),
                     acquisition_wrapper=None, started=time.time())
        atomic_json(ROOT / 'training_pipeline.json', state)
        deadline = time.time() + 6 * 3600
        while not (ROOT / 'corpus/manifest.json').exists():
            failures = list((ROOT / 'corpus').glob('failure_*.json'))
            if failures:
                raise RuntimeError('Corpus export failed: ' + str(failures))
            if time.time() > deadline:
                raise RuntimeError('Corpus export did not finish within six hours')
            time.sleep(15)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                   PYTHONUNBUFFERED='1')
        script = str(TOOLS / 'query_transfer.py')
        commands = {
            'fsbo': ['/root/data/cys/baseline/envs/bl_fsbo_exact/bin/python', script,
                'fsbo', '--corpus', str(ROOT / 'corpus'), '--output', str(ROOT / 'fsbo'),
                '--steps', '100000', '--threads', '2'],
            'rgpe': ['/root/data/cys/baseline/envs/bl_rgpe/bin/python', script,
                'rgpe-dispatch', '--corpus', str(ROOT / 'corpus'), '--output', str(ROOT / 'rgpe'),
                '--workers', '24']}
        processes = {}
        for name, command in commands.items():
            with (ROOT / (name + '.log')).open('a') as stream:
                process = subprocess.Popen(['nice', '-n', '10'] + command, env=env,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            processes[name] = process
            atomic_json(ROOT / (name + '_process.json'), dict(pid=process.pid,
                host=socket.gethostname(), command=command, started=time.time()))
        state.update(status='training', pids={k: v.pid for k, v in processes.items()})
        atomic_json(ROOT / 'training_pipeline.json', state)
        returncodes = {name: process.wait() for name, process in processes.items()}
        if all(v == 0 for v in returncodes.values()):
            verify_artifacts()
        state.update(status='complete' if all(v == 0 for v in returncodes.values()) else 'failed',
                     returncodes=returncodes, finished=time.time())
        atomic_json(ROOT / 'training_pipeline.json', state)
        if state['status'] != 'complete':
            raise RuntimeError('Training failed: ' + str(returncodes))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        atomic_json(ROOT / 'supervisor_failure.json', dict(error=repr(exc), time=time.time()))
        raise
