#!/usr/bin/env python3
"""Write and check a separate full-query OptFormer training setup without training."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
OPT_REPO = Path('/root/data/cys/baseline/repos/optformer-neurips22')
T5X_REPO = Path('/root/data/cys/baseline/repos/t5x-2023-07')
VOCAB = Path('/root/data/cys/baseline/assets/optformer/sentencepiece.model')

from query_transfer_common import atomic_json, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('Preparation is CPU-only; set CUDA_VISIBLE_DEVICES empty')
    args.windows, args.output = args.windows.resolve(), args.output.resolve()
    if args.output.exists():
        parser.error('Select a new setup directory')
    sys.path[:0] = [str(ROOT), str(OPT_REPO), str(T5X_REPO)]
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    from query_optformer_bridge import load_windows, register_task
    from optformer.t5x import feature_converters

    window_meta, corpus, records = load_windows(args.windows)
    preflight = json.loads((args.windows / 'preflight.json').read_text())
    if (preflight['status'] != 'converter_preflight_passed'
            or preflight['windows_sha256'] != window_meta['windows_sha256']
            or preflight['vocabulary_sha256'] != digest(VOCAB)):
        raise ValueError('The exact windows must pass converter preflight first')
    target = preflight['recommended_lengths']['targets']
    lengths = dict(inputs=preflight['recommended_lengths']['inputs'],
                   **{name: target for name in ('targets', 'target_inputs', 'targets_types', 'targets_masks')})
    manifest = dict(
        status='prepared_not_started', method='optformer_query', model_size='t5_1_1_base',
        windows=str(args.windows), windows_manifest_sha256=digest(args.windows / 'manifest.json'),
        vocabulary=str(VOCAB), vocabulary_sha256=digest(VOCAB), lengths=lengths,
        vae_path=corpus['vae_path'], vae_sha256=corpus['vae_sha256'],
        train_tasks=corpus['source_tasks'], validation_tasks=[],
        monitor_split='train_monitor', monitor_is_heldout_validation=False,
        seed=123, steps=100000, effective_batch_size=256, num_microbatches=64,
        physical_microbatch_size=4, learning_rate_base=0.3, dropout_rate=0.3,
        source_weights='random initialization', selection='last completed training step',
        max_history_observations=32, recommendations_per_call=1, training_started=False,
        pending=['GPU memory/throughput check', 'production launch',
                 'trained checkpoint reload check', 'query inference/evaluation integration'])
    task = register_task(manifest)
    # Exercise upstream tokenization and model feature conversion without a model or optimizer.
    dataset = task.get_dataset(sequence_length=lengths, split='train', shuffle=False,
                               num_epochs=1, seed=123, use_cached=False)
    converter = feature_converters.VizierEncDecFeatureConverter(pack=False)
    example = next(iter(converter(dataset, task_feature_lengths=lengths).take(1)))
    features = {name: list(value.shape) for name, value in example.items()}
    if not example['decoder_loss_weights'].numpy().any():
        raise ValueError('Query training example has no supervised target tokens')
    args.output.mkdir(parents=True)
    manifest_path = args.output / 'manifest.json'
    model_dir = args.output / 'model'
    model_gin = [
        'from __gin__ import dynamic_registration',
        'from optformer.data import converters',
        'from optformer.data import tasks as t5_tasks',
        'from t5x.examples.t5 import network',
        "include 't5x/examples/t5/t5_1_1/base.gin'",
        "include 'optformer/t5x/configs/models/t5_1_1_overrides.gin'",
        'network.T5Config.dtype = %ml_optformer.bfloat16',
        'DROPOUT_RATE = 0.3', 'STEP_OFFSET = 0',
        'TASK_FEATURE_LENGTHS = ' + repr(lengths),
        'STUDY_CONVERTER = @converters.OptFormerConverter()',
        'converters.OptFormerConverter.max_trials = None',
        'converters.OptFormerConverter.discard_const_objective = False',
        'converters.OptFormerConverter.minimum_config = False',
        't5_tasks.get_vocabulary.sentencepiece_model_file = ' + repr(str(VOCAB)),
    ]
    (args.output / 'model.gin').write_text('\n'.join(model_gin) + '\n')
    training_gin = [
        'from __gin__ import dynamic_registration', 'import __main__ as train_script',
        'from t5x import utils', 'from t5x import trainer',
        "include 'optformer/t5x/configs/runs/train.gin'",
        'include ' + repr(str(args.output / 'model.gin')),
        "MIXTURE_OR_TASK_NAME = 'query_train800'", 'USE_CACHED_TASKS = False',
        'MODEL_DIR = ' + repr(str(model_dir)), 'TRAIN_STEPS = 100000',
        'BATCH_SIZE = 256', 'RANDOM_SEED = 123',
        'train/utils.DatasetConfig.seed = 123', 'train/utils.DatasetConfig.pack = False',
        "train_eval/utils.DatasetConfig.split = 'train_monitor'",
        'train_eval/utils.DatasetConfig.batch_size = 4', 'EVAL_STEPS = 20',
        'LEARNING_RATE = 0.3', 'trainer.Trainer.num_microbatches = 64',
        'train_script.train.eval_period = 1000', 'train_script.train.stats_period = 10',
        'utils.SaveCheckpointConfig.period = 1000', 'utils.SaveCheckpointConfig.keep = 2',
    ]
    (args.output / 'train.gin').write_text('\n'.join(training_gin) + '\n')
    manifest.update(seqio_task='query_train800', windows_count=len(records),
                    model_feature_preflight='passed', example_feature_shapes=features,
                    model_dir=str(model_dir),
                    model_gin_sha256=digest(args.output / 'model.gin'),
                    train_gin_sha256=digest(args.output / 'train.gin'))
    atomic_json(manifest_path, manifest)
    print(json.dumps({'status': manifest['status'], 'windows': len(records),
                      'lengths': lengths, 'model_feature_shapes': features,
                      'training_started': False}), flush=True)


if __name__ == '__main__':
    main()
