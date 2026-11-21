"""Exercise all-source initial scratch trainer using four real source tasks."""
from common import *
import numpy as np
import source_meta
cfg=prepare();db=corpus_db();ids=cfg['train_tasks'][:4];rows=[];ys=[];offsets={};cursor=0
for t in ids:
    d=task_data(db,t);n=120;rows.append(d['X'][:n]);ys.append(d['y'][:n]);offsets[t]=[cursor,cursor+n];cursor+=n
db.close();directory=ROOT/'validation_initial_fsbo';directory.mkdir(exist_ok=True)
x=np.concatenate(rows);y=np.concatenate(ys);low=x.min(0);high=x.max(0);scale=np.where(high>low,high-low,1.)
np.save(directory/'features.npy',x);np.save(directory/'labels.npy',y)
write(directory/'manifest.json',dict(identity=dict(component=True,purpose='initial_all_source_component'),source_tasks=ids,current_task=None,
 offsets=offsets,lower=low.tolist(),scale=scale.tolist(),steps=50,seed=123,validate_every=25,checkpoint_every=25,
 validation_tasks=ids,features_sha256=sha(directory/'features.npy'),labels_sha256=sha(directory/'labels.npy')))
source_meta.train(directory)
value=read(directory/'meta_status.json');assert value['tasks_sampled']==4 and value['all_tasks_sampled'] and value['reload_verified']
