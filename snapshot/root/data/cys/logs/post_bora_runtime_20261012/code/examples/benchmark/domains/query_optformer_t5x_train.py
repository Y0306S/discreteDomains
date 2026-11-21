#!/usr/bin/env python3
"""Register the prepared query-only task; enter T5X only with explicit --execute."""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys

from query_transfer_common import atomic_json, digest, validate_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setup', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--check-config', action='store_true', help='Parse T5X configuration on CPU without training')
    args = parser.parse_args()
    if args.execute and args.check_config:
        parser.error('--check-config and --execute are separate operations')
    setup = args.setup.resolve()
    manifest = json.loads((setup / 'manifest.json').read_text())
    validate_assets(checkpoint=manifest['vae_path'])
    if (manifest['method'] != 'optformer_query'
            or manifest['model_feature_preflight'] != 'passed'
            or digest(setup / 'model.gin') != manifest['model_gin_sha256']
            or digest(setup / 'train.gin') != manifest['train_gin_sha256']):
        raise ValueError('Prepared query model configuration changed')
    print(json.dumps({'execute': args.execute, 'steps': manifest['steps'],
                      'effective_batch_size': manifest['effective_batch_size'],
                      'vae_checkpoint': manifest['vae_path'], 'model_dir': manifest['model_dir'],
                      'pending': manifest['pending']}), flush=True)
    if not args.execute and not args.check_config:
        return
    if args.check_config and os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('Configuration checks require CUDA_VISIBLE_DEVICES empty')
    if args.execute and (not os.environ.get('CUDA_VISIBLE_DEVICES') or os.environ['CUDA_VISIBLE_DEVICES'] == '-1'):
        raise ValueError('Explicit GPU assignment is required for full query OptFormer training')
    model_dir = Path(manifest['model_dir'])
    if model_dir.exists() and any(model_dir.iterdir()):
        raise ValueError('Model directory is not empty; inspect checkpoints before resuming')
    root = Path(__file__).resolve().parents[3]
    opt_repo = Path('/root/data/cys/baseline/repos/optformer-neurips22')
    t5x_repo = Path('/root/data/cys/baseline/repos/t5x-2023-07')
    sys.path[:0] = [str(root), str(opt_repo), str(t5x_repo)]
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    from mindware.components.optimizers.transfer_hpo.optformer_bridge import install_t5x_compat
    from query_optformer_bridge import register_task
    install_t5x_compat()
    register_task(manifest)
    if args.check_config:
        import gin
        from t5x import train as train_module
        sys.modules['__main__'].train = train_module.train
        gin.add_config_file_search_path(str(opt_repo))
        gin.add_config_file_search_path(str(t5x_repo))
        gin.parse_config_files_and_bindings([str(setup / 'train.gin')], [])
        report = dict(status='passed', training_started=False,
                      model_gin_sha256=manifest['model_gin_sha256'],
                      train_gin_sha256=manifest['train_gin_sha256'])
        atomic_json(setup / 'config_preflight.json', report)
        print(json.dumps(report), flush=True)
        return
    sys.argv = [sys.argv[0], '--gin_file=' + str(setup / 'train.gin'),
                '--gin_search_paths=' + str(opt_repo), '--gin_search_paths=' + str(t5x_repo)]
    runpy.run_module('t5x.train', run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    main()
