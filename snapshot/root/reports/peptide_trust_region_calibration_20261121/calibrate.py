"""Paired decoder-only radius/edit calibration. No MIC scoring or production writes."""
import csv
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
ROOT = Path('/root/data/logs/peptide_target_loss_gate_20round_max2_test10_20261119')
UP = Path('/root/data/cys/BOLT/optimization/peptides')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'deps'))
sys.path.insert(0, str(UP))
import torch
import numpy as np
import pandas as pd
from Levenshtein import distance
from uniref_vae import data as native_data
sys.modules['data'] = native_data
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
spec = importlib.util.spec_from_file_location('cpu_vae', ROOT/'reference/APEXGo/generation/vae.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
vocab = pd.read_csv(UP/'uniref_vae/1mer_vocab.csv', header=None).values.squeeze().tolist()
dataset = native_data.DatasetKmers(k=1, load_data=False, vocab=vocab)
model = module.InfoTransformerVAE(dataset=dataset, d_model=128, kl_factor=.0001,
    encoder_dim_feedforward=256, decoder_dim_feedforward=256, encoder_num_layers=6, decoder_num_layers=6)
model.max_string_length = 30
RADII = [.025, .05, .075, .1, .15, .2, .4, .8]
SEEDS = [31001, 31002, 31003]
PER_SEED = 64


def tokens(sequences):
    return native_data.collate_fn([dataset.encode(s).unsqueeze(0) for s in dataset.tokenize_sequence(sequences)])


def allowed(sequence, reference, center, local_edits):
    return (bool(sequence) and len(sequence) <= 30 and sequence.count('C') <= 1
            and not set(sequence)-set('ACDEFGHIKLMNPQRSTVWY')
            and distance(sequence, reference) <= len(reference)//4
            and distance(sequence, center) <= local_edits)


def main():
    start = time.monotonic()
    cfg = json.loads((ROOT/'config.json').read_text())
    pool_path = Path('/root/data/logs/peptide_bo_train800_20261027_v5b/initialization.json')
    pool = json.loads(pool_path.read_text())
    groups = {}
    for task in sorted(cfg['train_tasks']):
        groups.setdefault(len(pool[task]['reference'])//4, []).append(task)
    rng = random.Random(20261121)
    tasks = sorted(t for budget in sorted(groups) for t in rng.sample(groups[budget], 2))
    assert len(tasks) == 10 and not set(tasks)&set(cfg['tasks'])
    pep = json.loads((ROOT/'evaluation/adapted_online/pep_999/result.json').read_text())
    target_pool = dict(reference=pep['reference'], sequences=[r['sequence'] for r in pep['observations'] if r['phase']=='initialization'])
    cohorts = {'source_development': {t:pool[t] for t in tasks}, 'pep999_diagnostic': {'pep_999':target_pool}}
    models = [('author', Path(cfg['author_vae']), 'plain', ['source_development']),
              ('adapted_before_online', ROOT/'vae/weights.pt', 'plain', list(cohorts)),
              ('pep999_after_round20', ROOT/'evaluation/adapted_online/pep_999/vae_updates/round_020/retained_vae.pt', 'plain', ['pep999_diagnostic']),
              ('pep999_after_round40', ROOT/'evaluation/adapted_online/pep_999/vae_updates/round_040/retained_vae.pt', 'plain', ['pep999_diagnostic'])]
    report = dict(status='running', oracle_calls=0, training_performed=False, production_changes=False,
                  task_selection='Two source tasks per allowed-edit budget (2..6), seed20261121; no outcome selection.',
                  source_tasks=tasks, radii=RADII, seeds=SEEDS, draws_per_task_radius=PER_SEED*len(SEEDS),
                  anchor_selection='reference and first three feasible initialization sequences; no MIC ranking',
                  geometry='native absolute radius4*length; Sobol candidates; random mask20/256 with native fallback; all paired across radii/models',
                  models={}, rows=[], limitations=[
                      'Decoder calibration, not acquisition-selected proposals, BO convergence or evidence of better MIC.',
                      'Source development tasks were previously used for model training; not independent generalization results.',
                      'pep999 is an inspected test task; its diagnostics must not select global settings.',
                      'Decoded sequences cluster by task, anchor and seed; draws are not independent tasks.',
                      'Novelty means absent from initialization and reference; later production caches will differ.',
                      'Edit count includes substitutions, insertions and deletions. Radius units are latent, not edit counts.'
                  ])
    sequences_path = OUT/'draws.csv'
    with sequences_path.open('w', newline='') as f:
        writer = csv.DictWriter(f,fieldnames=['model','cohort','task','radius','seed','center','sequence','edits_reference','edits_center','global_valid','novel'])
        writer.writeheader()
        for name, path, _, selected_cohorts in models:
            state = torch.load(path,map_location='cpu',weights_only=False)
            model.load_state_dict(state,strict=True)
            model.eval()
            assert all(torch.isfinite(t).all() for t in model.state_dict().values() if t.is_floating_point())
            report['models'][name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            with torch.inference_mode():
                for cohort in selected_cohorts:
                    for task, item in cohorts[cohort].items():
                        ref = item['reference']; known = set(item['sequences'])|{ref}
                        assert all(distance(s,ref)<=len(ref)//4 for s in item['sequences'])
                        anchors = list(dict.fromkeys([ref]+item['sequences'][:3]))
                        means,_ = model.encode(tokens(anchors)); means=means.reshape(-1,256)
                        for radius in RADII:
                            draws=[]
                            for seed in SEEDS:
                                ids=torch.arange(PER_SEED)%len(anchors)
                                generator=torch.Generator().manual_seed(seed)
                                mask=torch.rand((PER_SEED,256),generator=generator)<=20/256
                                empty=torch.where(mask.sum(1)==0)[0]
                                mask[empty,torch.randint(0,255,(len(empty),),generator=generator)]=True
                                sobol=torch.quasirandom.SobolEngine(256,scramble=True,seed=seed).draw(PER_SEED)
                                z=means[ids].clone()+(sobol*2-1)*4*radius*mask
                                torch.manual_seed(seed+1000)
                                decoded=model.sample(z=z.reshape(-1,2,128))
                                for i in range(PER_SEED):
                                    s=dataset.decode(decoded[i]);center=anchors[int(ids[i])]
                                    row=dict(model=name,cohort=cohort,task=task,radius=radius,seed=seed,center=center,sequence=s,
                                             edits_reference=distance(s,ref),edits_center=distance(s,center),
                                             global_valid=allowed(s,ref,center,100),novel=s not in known)
                                    draws.append(row);writer.writerow(row)
                            counts={}
                            for edits in [1,2,3,100]:
                                good=[r for r in draws if r['global_valid'] and r['edits_center']<=edits]
                                fresh=[r for r in good if r['novel']]
                                counts[str(edits)]=dict(valid_draws=len(good),novel_valid_draws=len(fresh),unique_novel_valid=len({r['sequence'] for r in fresh}))
                            summary=dict(model=name,cohort=cohort,task=task,reference_length=len(ref),allowed_reference_edits=len(ref)//4,
                                         radius=radius,draws=len(draws),median_reference_edits=float(np.median([r['edits_reference'] for r in draws])),
                                         median_center_edits=float(np.median([r['edits_center'] for r in draws])),counts=counts)
                            report['rows'].append(summary)
                        print(name,cohort,task,'complete',flush=True)
                        (OUT/'results.partial.json').write_text(json.dumps(report,indent=2))
    report['status']='complete';report['elapsed_seconds']=time.monotonic()-start
    (OUT/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('COMPLETE',report['elapsed_seconds'],flush=True)


if __name__=='__main__':
    main()
