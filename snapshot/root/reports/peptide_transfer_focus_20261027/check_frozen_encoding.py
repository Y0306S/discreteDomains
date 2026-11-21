#!/usr/bin/env python3
"""Verify saved train-only latents and cross-service frozen decoding without scoring."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import urllib.request

os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
os.sched_setaffinity(0, {12})
sys.path.insert(0, '/root/data/cys/llmoptimizer_benchmark/tools')
sys.path.insert(0, '/root/data/logs/peptide_bo_release_20261027_v5')

import numpy as np
import torch
import peptide_transfer as transfer
from mindware.domains.protein.peptide_rpc import dumps, loads
from mindware.domains.protein.verified_bo import validate_sequence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ports', default='18872,18874')
    args = parser.parse_args()
    ports = [int(p) for p in args.ports.split(',')]
    torch.set_num_threads(1)
    corpus = Path('/root/data/logs/peptide_transfer_train800_20261027_v2/corpus')
    identity = json.loads((corpus / 'status.json').read_text())['identity']
    recipe = json.loads((transfer.SOURCE / 'recipe.json').read_text())
    assert transfer.digest(recipe['vae']) == identity['vae_sha256']
    task = recipe['train'][0]
    assert task not in recipe['test']
    db = sqlite3.connect('file:' + str(corpus / 'corpus.sqlite') + '?mode=ro', uri=True)
    data = transfer.read_task(db, task)
    db.close()
    sequences = data['sequences'][:256].tolist()
    weights = torch.load(recipe['vae'], map_location='cpu', weights_only=False)
    version = hashlib.sha256(dumps(weights)).hexdigest()
    bank = 'frozen_export_consistency_check_' + str(time.time_ns())

    def request(port, **values):
        payload = dumps(dict(identity=bank, **values))
        req = urllib.request.Request('http://127.0.0.1:%d/rpc' % port, data=payload,
            headers={'X-Request-ID': hashlib.sha256(payload).hexdigest()})
        with urllib.request.urlopen(req, timeout=120) as response:
            result = loads(response.read())
        if result.get('error') or result.get('missing_weights'):
            raise RuntimeError(str(result))
        return result

    timings, encodings, decoded = {}, {}, {}
    try:
        for port in ports:
            request(port, op='weights', version=version, weights=weights)
            started = time.monotonic()
            encoded = request(port, op='encode_mean', version=version, seed=123, sequences=sequences)['z']
            timings[str(port)] = time.monotonic() - started
            assert encoded.shape == (256, 256) and torch.isfinite(encoded).all()
            encodings[port] = encoded
            decoded[port] = request(port, op='decode', version=version, seed=456, z=encoded[:4])['sequences']
            for sequence in decoded[port]:
                validate_sequence(sequence)
        saved = torch.from_numpy(data['X'][:256])
        differences = {str(p): float((encodings[p] - saved).abs().max()) for p in ports}
        assert max(differences.values()) <= 1e-6
        assert all(decoded[p] == decoded[ports[0]] for p in ports)
        first = encodings[ports[0]]
        # A nonce requests a real repeated inference instead of a cached reply.
        repeated = request(ports[0], op='encode_mean', version=version, seed=123,
                           sequences=sequences, check_nonce=1)['z']
        assert torch.equal(first, repeated)
        result = dict(passed=True, source_task=task, dimensions=256, batch_size=256,
            exported_latents_match_frozen_encoder=True, maximum_difference=differences,
            repeated_encoding_identical=True, seeded_cross_service_decoding_identical=True,
            decoded_sequences=decoded[ports[0]],
            decoding_is_not_guaranteed_sequence_roundtrip=True,
            similarity_feasibility_requires_separate_check=True, oracle_calls=0,
            encoder_sha256=identity['vae_sha256'], seconds_by_service=timings,
            test200_used_for_offline_training=False)
        transfer.atomic_json(Path(__file__).with_name('frozen_encoding_check.json'), result)
        print(json.dumps(result), flush=True)
    finally:
        for port in ports:
            request(port, op='release')


if __name__ == '__main__':
    main()
