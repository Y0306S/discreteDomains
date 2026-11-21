"""Initial FSBO scratch meta-training: all800 source tasks participate."""
from common import *


def main():
    import numpy as np
    import source_meta
    cfg=prepare();directory=ROOT/'fsbo';directory.mkdir(exist_ok=True)
    path=directory/'manifest.json';source=read(ROOT/'corpus/manifest.json')
    if not path.exists():
        def payload_for(t):return (Path(source['database'])/(t+'.npz')).read_bytes()
        counts={}
        for t in cfg['train_tasks']:
            payload=payload_for(t)
            assert hashlib.sha256(payload).hexdigest()==source['task_hashes'][t]
            with np.load(io.BytesIO(payload),allow_pickle=False) as a:counts[t]=len(a['y'])
        total=sum(counts.values());x=np.lib.format.open_memmap(directory/'features.npy',mode='w+',dtype='float32',shape=(total,256))
        y=np.lib.format.open_memmap(directory/'labels.npy',mode='w+',dtype='float32',shape=(total,))
        offsets={};cursor=0
        for t in cfg['train_tasks']:
            payload=payload_for(t)
            with np.load(io.BytesIO(payload),allow_pickle=False) as a:
                n=counts[t];x[cursor:cursor+n]=a['X'];y[cursor:cursor+n]=a['y']
            offsets[t]=[cursor,cursor+n];cursor+=n
        x.flush();y.flush();del x,y
        low,high=np.asarray(source['lower']),np.asarray(source['upper']);scale=np.where(high>low,high-low,1.)
        write(path,dict(identity=dict(component=False,purpose='initial_all800_scratch',config_sha256=sha(ROOT/'config.json')),
            source_tasks=cfg['train_tasks'],current_task=None,offsets=offsets,lower=low.tolist(),scale=scale.tolist(),
            normalization='source800_only',steps=cfg['fsbo_meta_steps'],seed=cfg['fsbo_meta_seed'],
            validate_every=100,checkpoint_every=1000,validation_tasks=source['fsbo_validation_tasks'],
            features_sha256=sha(directory/'features.npy'),labels_sha256=sha(directory/'labels.npy')))
    if not (directory/'meta_status.json').exists() or read(directory/'meta_status.json')['status']!='complete':
        source_meta.train(directory)
    assert read(directory/'meta_status.json')['tasks_sampled']==800
    assert read(directory/'meta_status.json')['initialization']=='scratch'
    for name in ('features.npy','labels.npy'):(directory/name).unlink(missing_ok=True)


if __name__=='__main__':main()
