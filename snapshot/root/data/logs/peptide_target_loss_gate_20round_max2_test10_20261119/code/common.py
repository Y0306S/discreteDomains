"""Isolated peptide adaptation pilot; production artifacts are read-only."""
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = Path('/root/data/cys/llmoptimizer_benchmark')
UPSTREAM = Path('/root/data/cys/BOLT/optimization/peptides')
SOURCE = Path('/root/data/logs/peptide_bo_train800_20261027_v5b')
OLD = Path('/root/data/logs/peptide_transfer_train800_20261027_v2')
BASE = Path('/root/data/logs/peptide_batch50_gpu_test200_20261102_v1/fsbo')
for p in (ROOT/'deps', REPO/'tools', UPSTREAM):
    sys.path.insert(0, str(p))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    if (ROOT/'evaluation/adapted_online') in path.parents and path.name in ('status.json', 'result.json') and (ROOT/'stop_policy.json').exists():
        cap = read(ROOT/'stop_policy.json')
        value = dict(value, proposal_budget=cap['proposals'], fresh_budget=cap['fresh_budget'],
                     batch_size=cap['batch_size'], max_rounds=cap['max_rounds'],
                     stop_policy_sha256=sha(ROOT/'stop_policy.json'))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(tmp, path)


def save(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}


def setup(seed=123, gpu=True):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if gpu:
        allocation = read(ROOT/'execution_policy.json') if (ROOT/'execution_policy.json').exists() else {}
        torch.cuda.set_per_process_memory_fraction(allocation.get('gpu_memory_fraction', .22))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class PayloadCorpus:
    """Read immutable, hash-checked NPZ payloads without network SQLite locks."""
    def __init__(self):
        self.directory=ROOT/'source_snapshot/payloads'
        self.hashes=read(ROOT/'source_snapshot/storage_manifest.json')['payload_hashes']
    def execute(self,query,parameters):
        task,=parameters
        payload=(self.directory/(task+'.npz')).read_bytes();digest=self.hashes[task]
        if hashlib.sha256(payload).hexdigest()!=digest:raise ValueError('Source payload integrity: '+task)
        if query=='SELECT payload_sha256,payload FROM tasks WHERE id=?':self.row=(digest,payload)
        elif query=='SELECT payload FROM tasks WHERE id=?':self.row=(payload,)
        else:raise ValueError('Unsupported payload lookup: '+query)
        return self
    def fetchone(self):return self.row
    def close(self):pass


def corpus_db():
    return PayloadCorpus()


def task_data(db, task):
    import numpy as np
    row = db.execute('SELECT payload_sha256,payload FROM tasks WHERE id=?',(task,)).fetchone()
    if row is None or hashlib.sha256(row[1]).hexdigest() != row[0]:
        raise ValueError('Source payload integrity: '+task)
    with np.load(io.BytesIO(row[1]), allow_pickle=False) as a:
        return {k: a[k].copy() for k in a.files}


def load_vae(path):
    from uniref_vae.load_vae import load_vae as native_load
    previous = os.getcwd()
    try:
        os.chdir(UPSTREAM)
        return native_load(str(path), max_string_length=30)
    finally:
        os.chdir(previous)


def tokens(sequences, data):
    from uniref_vae.data import collate_fn
    encoded = [data.encode(s).unsqueeze(0) for s in data.tokenize_sequence(list(sequences))]
    return collate_fn(encoded).cuda()


def encode(model, data, sequences, mean=True, batch=128):
    import numpy as np
    import torch
    model.eval()
    out=[]
    with torch.no_grad():
        for i in range(0,len(sequences),batch):
            mu,sigma = model.encode(tokens(sequences[i:i+batch],data))
            z = mu if mean else model.sample_posterior(mu,sigma)
            out.append(z.reshape(-1,256).cpu().numpy())
    x = np.concatenate(out).astype(np.float32)
    if x.shape != (len(sequences),256) or not np.isfinite(x).all():
        raise ValueError('Invalid VAE embeddings')
    return x


def make_gp(x):
    import gpytorch
    import torch
    from lolbo.utils.bo_utils.ppgpr import GPModelDKL
    x=torch.as_tensor(x,dtype=torch.float32,device='cuda')
    m=GPModelDKL(x[:min(len(x),1024)],gpytorch.likelihoods.GaussianLikelihood()).cuda()
    return m,gpytorch.mlls.PredictiveLogLikelihood(m.likelihood,m,num_data=len(x))


def fit_gp(model,mll,x,y,epochs,total=None):
    import torch
    x=torch.as_tensor(x,dtype=torch.float32,device='cuda')
    y=torch.as_tensor(y,dtype=torch.float32,device='cuda').flatten()
    model.train()
    mll.num_data=len(y) if total is None else total
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    for _ in range(epochs):
        order=torch.randperm(len(y),device='cuda')
        for ids in order.split(128):
            optimizer.zero_grad(set_to_none=True)
            loss=-mll(model(x[ids]),y[ids])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite constraint/objective GP loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
    model.eval()
    model.likelihood.eval()
    return model


def training_indices(data):
    """Native recent-50 plus unique feasible top-1000, preserving duplicates across groups."""
    import numpy as np
    top=[i for i in np.argsort(-data['y'],kind='stable') if data['constraints'][i]<=0][:1000]
    return list(range(max(0,len(data['y'])-50),len(data['y'])))+top


def prepare():
    import numpy as np
    if (ROOT/'config.json').exists():
        cfg = read(ROOT/'config.json')
        # Preserve immutable training identity; audit execution caps separately.
        if (ROOT/'stop_policy.json').exists():
            cap = read(ROOT/'stop_policy.json')
            assert cap['tasks'] == cfg['tasks'] and cap['batch_size'] == cfg['batch_size'] == 50
            assert 0 < cap['proposals'] == cap['batch_size'] * cap['max_rounds'] <= cfg['proposals']
            assert 0 < cap['fresh_budget'] <= cfg['fresh_budget']
            cfg = dict(cfg, proposals=cap['proposals'], fresh_budget=cap['fresh_budget'],
                       stop_policy_sha256=sha(ROOT/'stop_policy.json'))
        return cfg
    source=read(SOURCE/'recipe.json')
    manifest=read(OLD/'corpus/manifest.json')
    assert len(source['train'])==800 and not set(source['train'])&set(source['test'])
    assert source['train']==manifest['source_tasks']
    # Prespecified random cohort, independent of outcomes and peptide length.
    rng=np.random.RandomState(20261117)
    tasks=sorted(rng.choice(source['test'],10,replace=False).tolist())
    cfg=dict(format='peptide-vae-fsbo-adaptation-pilot-v1', tasks=tasks,
        train_tasks=source['train'], seed=1, cohort_seed=20261117,
        initialization=100,batch_size=50,proposals=5000,fresh_budget=5000,
        candidates=5000,similarity=.75,stagnation_rounds=10,update_epochs=2,
        learning_rate=.001,fsbo_meta_steps=100000,fsbo_meta_seed=123,
        source_recipe=str(SOURCE/'recipe.json'),source_recipe_sha256=sha(SOURCE/'recipe.json'),
        source_corpus_sha256=sha(OLD/'corpus/manifest.json'),
        author_vae=source['vae'],author_vae_sha256=sha(source['vae']),
        offline_policy='one_shared_VAE_sequential_joint_LOLBO_replay_all800_recent50_top1000_two_epochs_per_task',
        offline_new_oracle_calls=0,
        online_policy='VAE_plus_FSBO_exact_MLL_plus_constraint_PLL_then_reencode_all_target_and_refit_FSBO',
        online_meta_retraining=False, fsbo_target_max_updates=1000,
        arms=['author_frozen','adapted_frozen','adapted_online'],
        limitation='Single optimizer seed; sequential shared offline VAE replay is a new transfer adaptation, not a fresh 800-task LOLBO campaign.')
    write(ROOT/'config.json',cfg)
    return cfg
