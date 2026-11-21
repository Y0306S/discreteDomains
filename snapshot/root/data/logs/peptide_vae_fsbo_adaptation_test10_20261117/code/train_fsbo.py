"""Use the original 100k-step CPU FSBO trainer with refreshed train800 features."""
import io
import json
import sys
from types import SimpleNamespace
from common import *


def main(component=False):
    import numpy as np
    # Keep the original FSBO environment and its original TensorBoard packages.
    sys.path.remove(str(ROOT/'deps'))
    sys.path.insert(0,str(REPO/'examples/benchmark/domains'))
    import train_taskset_transfer as trainer
    cfg=prepare()
    corpus=ROOT/('validation_corpus' if component else 'corpus')
    manifest=read(corpus/'manifest.json')
    if not component:
        assert len(manifest['source_tasks'])==800 and manifest['source_tasks']==cfg['train_tasks']
    db=sqlite3.connect('file:'+manifest['database']+'?mode=ro',uri=True)
    low,high=np.asarray(manifest['lower']),np.asarray(manifest['upper'])
    scale=np.where(high>low,high-low,1.)
    tasks={}
    for task in manifest['source_tasks']:
        expected,payload=db.execute('SELECT sha,payload FROM tasks WHERE id=?',(task,)).fetchone()
        assert hashlib.sha256(payload).hexdigest()==expected==manifest['task_hashes'][task]
        with np.load(io.BytesIO(payload),allow_pickle=False) as a:
            tasks[task]=dict(X=((a['X']-low)/scale).astype(np.float32),y_maximize=a['y'].copy())
    db.close()
    args=SimpleNamespace(output=ROOT/('validation_meta' if component else 'fsbo'),threads=2,seed=123,
        steps=100 if component else 100000,resume=True,
        purpose='component_validation' if component else 'train800_adapted_vae_pilot',method='fsbo')
    args.output.mkdir(exist_ok=True)
    def identity(args,manifest):
        return dict(domain='peptide',method='fsbo',purpose=args.purpose,
            corpus_sha256=sha(corpus/'manifest.json'),seed=123,
            trainer_sha256=sha(__file__),shared_training_loop_sha256=sha(trainer.__file__),
            representation=manifest['representation'],device='cpu',threads=2)
    trainer.identity=identity
    trainer.fsbo(args,manifest,tasks)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--component',action='store_true')
    main(parser.parse_args().component)
