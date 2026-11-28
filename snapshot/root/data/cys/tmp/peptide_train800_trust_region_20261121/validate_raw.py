"""Independent exact-edit and count audit of every saved decoder draw."""
import gzip,hashlib,json,time
from collections import Counter
from pathlib import Path
from Levenshtein import distance
ROOT=Path(__file__).resolve().parent
OUT=Path('/root/reports/peptide_train800_trust_region_20261121')
pool=json.loads(Path('/root/data/logs/peptide_bo_train800_20261027_v5b/initialization.json').read_text())
files=list((ROOT/'tasks/adapted').glob('*.json'))+list((ROOT/'tasks/author').glob('*.json'))
assert len(files)==1600
start=time.monotonic();total=0;manifest={}
for p in files:
 a=json.loads(p.read_text());task=a['task'];item=pool[task];ref=item['reference'];known=set(item['sequences'])|{ref}
 anchors=list(dict.fromkeys([ref]+item['sequences'][:3]))
 drawpath=p.with_suffix('.jsonl.gz');data=drawpath.read_bytes()
 rows=[json.loads(line) for line in gzip.decompress(data).decode().splitlines()]
 assert len(rows)==96*len(a['identity']['radii'])
 assert Counter((r['radius'],r['seed']) for r in rows)==Counter({(rad,seed):32 for rad in a['identity']['radii'] for seed in a['identity']['seeds']})
 for r in rows:
  seq=r['sequence'];center=anchors[r['anchor']];assert r['center']==center
  er=distance(seq,ref);ec=distance(seq,center)
  valid=bool(seq) and len(seq)<=30 and seq.count('C')<=1 and not set(seq)-set('ACDEFGHIKLMNPQRSTVWY') and er<=len(ref)//4
  assert r['edits_reference']==er and r['edits_center']==ec and r['valid']==valid and r['novel']==(seq not in known)
 for s in a['summary']:
  draw=[r for r in rows if r['radius']==s['radius']]
  for k in [1,2,3,100]:
   good=[r for r in draw if r['valid'] and r['edits_center']<=k];novel=[r for r in good if r['novel']]
   c=s['counts'][str(k)]
   assert c['valid_draws']==len(good) and c['novel_valid_draws']==len(novel) and c['unique_novel_valid']==len(set(r['sequence'] for r in novel))
 total+=len(rows);manifest[str(drawpath.relative_to(ROOT))]=hashlib.sha256(data).hexdigest()
assert total==844800
(OUT/'raw_validation.json').write_text(json.dumps(dict(status='passed',model_task_files=len(files),decoded_draws=total,elapsed_seconds=time.monotonic()-start,checks=['every exact reference and center edit distance','syntax and original reference feasibility','initial novelty flags','paired seed/radius draw counts','all local1/2/3/global summary counts'],sha256=manifest),indent=2))
print('PASS',len(files),'model-task files',total,'draws',round(time.monotonic()-start,2),'seconds')
