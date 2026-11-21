#!/usr/bin/env python3
"""Check all prepared CEB windows with the original OptFormer tokenizer/converter."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from query_transfer_common import atomic_json, digest, load_corpus, window_indices
from query_optformer_bridge import study_from_window

ROOT = Path(__file__).resolve().parents[3]
OPT_REPO = Path('/root/data/cys/baseline/repos/optformer-neurips22')
VOCAB = Path('/root/data/cys/baseline/assets/optformer/sentencepiece.model')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('This preflight uses CPU only')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(OPT_REPO))
    from optformer.data import tasks as opt_tasks
    from mindware.components.optimizers.transfer_hpo.optformer_bridge import make_converter, checked_conversion
    meta = json.loads((args.windows / 'manifest.json').read_text())
    corpus = Path(meta['corpus'])
    manifest, sources = load_corpus(corpus)
    if (meta['corpus_sha256'] != digest(corpus / 'manifest.json')
            or meta['windows_sha256'] != digest(args.windows / 'windows.jsonl')
            or meta['source_tasks'] != manifest['source_tasks']):
        raise ValueError('Window provenance mismatch')
    vocabulary = opt_tasks.get_vocabulary(str(VOCAB), expected_vocab_size=33100)
    converter = make_converter()
    maximum = dict(inputs=0, targets=0, target_inputs=0)
    count, total, observed = 0, 0, set()
    expected_windows = [(task, indices) for task, source in sources.items()
                        for indices in window_indices(len(source['X']), meta['max_observations'])]
    with (args.windows / 'windows.jsonl').open() as stream:
        for line in stream:
            record = json.loads(line)
            if count >= len(expected_windows):
                raise ValueError('Unexpected extra window')
            task, indices = expected_windows[count]
            source = sources[task]
            if (record['task_id'] != task or not record['ordered'] or record['shuffled']
                    or record['source_rows'] != [source['source_rows'][i] for i in indices]
                    or not np.array_equal(record['X'], source['X'][indices])
                    or not np.array_equal(record['y_maximize'], source['y_maximize'][indices])):
                raise ValueError('Window does not preserve its original measurements and order')
            _, lengths = checked_conversion(converter, study_from_window(record), vocabulary)
            for key, value in lengths.items():
                maximum[key] = max(maximum[key], value)
            count += 1
            total += len(indices)
            observed.update((task, row) for row in record['source_rows'])
            if count % 200 == 0:
                print(json.dumps({'validated_windows': count, 'max_tokens': maximum}), flush=True)
    if count != meta['windows'] or count != len(expected_windows) or len(observed) != manifest['observations']:
        raise ValueError('Windows do not cover the complete eligible corpus')
    report = dict(status='converter_preflight_passed', training_started=False, windows=count,
                  distinct_observations=len(observed), window_observations=total,
                  max_tokens=maximum, vocabulary_sha256=digest(VOCAB),
                  windows_sha256=meta['windows_sha256'],
                  recommended_lengths=dict(inputs=((maximum['inputs'] + 255) // 128) * 128,
                                           targets=((maximum['targets'] + 67 + 127) // 128) * 128),
                  pending=['SeqIO/model feature preflight', 'checkpoint and inference adapter',
                           'GPU throughput and memory preflight', 'production launch'])
    atomic_json(args.windows / 'preflight.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
