"""Read-only all800 historical source neighborhood coverage and outcome audit."""
import hashlib,json,time
from pathlib import Path
from collections import Counter
import numpy as np
from Levenshtein import distance
OUT=Path(__file__).resolve().parent
ROOT=Path('/root/data/logs/peptide_target_loss_gate_20round_max2_test10_20261119')
cfg=json.loads((ROOT/'config.json').read_text())
hashes=json.loads((ROOT/'source_snapshot/storage_manifest.json').read_text())['payload_hashes']
pool=json.loads(Path('/root/data/logs/peptide_bo_train800_20261027_v5b/initialization.json').read_text())
rows=[];start=time.monotonic()
for task in sorted(cfg['train_tasks']):
 path=ROOT/'source_snapshot/payloads'/(task+'.npz');raw=path.read_bytes()
 assert hashlib.sha256(raw).hexdigest()==hashes[task]
 a=np.load(path,allow_pickle=False);seqs=a['sequences'].tolist();y=a['y'];ref=str(a['reference']);constraints=a['constraints']
 assert np.isfinite(y).all() and len(seqs)==len(y) and ref==pool[task]['reference']
 allowed=len(ref)//4
 edits=np.array([distance(s,ref) for s in seqs]);feasible=edits<=allowed
 assert np.array_equal(feasible,constraints<=0)
 known=set(pool[task]['sequences']);novel=np.array([s not in known for s in seqs])
 seen=set();unique=[]
 for i,s in enumerate(seqs):
  if s not in seen:seen.add(s);unique.append(i)
 fresh=[i for i in unique if novel[i]]
 valid=[i for i in fresh if feasible[i]]
 initial=[i for i,s in enumerate(seqs) if s in known]
 initbest=float(-max(y[i] for i in initial));best=float(-max(y[feasible]))
 centers=list(dict.fromkeys([ref]+pool[task]['sequences'][:3]))
 distances=[min(distance(seqs[i],c) for c in centers) for i in valid]
 best_by_limit={}
 for local in [1,2,3,100]:
  selected=[i for i,di in zip(valid,distances) if di<=local]
  best_by_limit[str(local)]=dict(observed_candidates=len(selected),best_mic=min([initbest]+[-float(y[i]) for i in selected]))
 # Describe adaptive reference-feasible moves using source order; no reranking or new scoring.
 running=-initbest;center=pool[task]['sequences'][int(np.argmax(pool[task]['scores']))]
 moves=[]
 for i in range(len(seqs)):
  if feasible[i] and y[i]>running:
   moves.append(dict(row=i,edits_from_previous_best=distance(seqs[i],center),mic=-float(y[i])))
   running=float(y[i]);center=seqs[i]
 rows.append(dict(task=task,reference_length=len(ref),allowed_reference_edits=allowed,observations=len(seqs),distinct_sequences=len(unique),
  distinct_novel=len(fresh),distinct_novel_feasible=len(valid),novel_feasible_percent=100*len(valid)/len(fresh) if fresh else 0.,
  median_novel_reference_edits=float(np.median(edits[fresh])) if fresh else None,initial_mic=initbest,best_mic=best,
  initial_sequences_present=len(known&set(seqs)),source_sha256=hashes[task],
  observed_neighborhood_coverage=best_by_limit,improving_moves=moves))
 if len(rows)%100==0:print('audited',len(rows),round(time.monotonic()-start,1),flush=True)
report=dict(status='complete',tasks=len(rows),oracle_calls=0,elapsed_seconds=time.monotonic()-start,rows=rows,
 limitations=['Historical source observations reflect the old sampling policy; neighborhood coverage is not an unbiased sample of sequence space.',
 'Source corpus is development data previously used in representation/meta training, not a held-out benchmark.',
 'Nearest fixed center distances use only reference and first3initialization peptides; adaptive promising centers may cover different neighborhoods.',
 'Incumbent moves follow saved source-row order; source payload does not preserve all timing/acquisition phase metadata.'])
(OUT/'source_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False))
print('COMPLETE',len(rows),flush=True)
