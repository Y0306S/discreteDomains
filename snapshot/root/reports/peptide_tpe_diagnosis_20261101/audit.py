#!/usr/bin/env python3
"""Offline diagnosis of completed peptide proposals; zero inference/oracle calls."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import collections,csv,hashlib,json,itertools,tarfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np

OUT=Path(__file__).resolve().parent
ROOTS={
 'TPE':Path('/root/data/logs/peptide_test200_baselines_20261027_v2/tpe'),
 'FSBO':Path('/root/data/logs/peptide_transfer_test200_20261028_v1/fsbo'),
 'LOLBO':Path('/root/data/logs/peptide_test200_baselines_20261027_v2/lolbo')}

def distances(reference):
    masks={}
    for i,c in enumerate(reference):masks[c]=masks.get(c,0)|(1<<i)
    m=len(reference);top=1<<(m-1)
    def distance(text):
        vp=(1<<m)-1;vn=0;score=m
        for c in text:
            eq=masks.get(c,0);xv=eq|vn;xh=(((eq&vp)+vp)^vp)|eq
            ph=vn|~(xh|vp);mh=vp&xh
            score+=bool(ph&top)-bool(mh&top)
            ph=(ph<<1)|1;mh<<=1
            vp=mh|~(xv|ph);vn=ph&xv
        return score
    return distance

def dp_distance(a,b):
    row=list(range(len(b)+1))
    for i,c in enumerate(a,1):
        new=[i]
        for j,d in enumerate(b,1):new.append(min(new[-1]+1,row[j]+1,row[j-1]+(c!=d)))
        row=new
    return row[-1]

def read_one(job):
    method,path,expected=job
    with tarfile.open(path,'r|gz') as tar:
        for member in tar:
            if member.name=='history.json':payload=tar.extractfile(member).read();break
    assert hashlib.sha256(payload).hexdigest()==expected
    h=json.loads(payload);post=h['observations'][100:];ref=h['reference']
    assert all(r['phase']=='initialization' for r in h['observations'][:100])
    feasible=[r for r in post if r['feasible']]
    fresh_feasible=[r for r in feasible if not r['cached']]
    start=min(r['mic'] for r in h['observations'][:100] if r['feasible'])
    result=dict(method=method,task=h['identity']['task'],proposals=len(post),
      acquisition_proposals=h['proposals'],fresh=h['fresh_evaluations'],
      feasible=len(feasible),fresh_feasible=len(fresh_feasible),
      unique_feasible=len({r['sequence'] for r in feasible}),
      unique_sequences=len({r['sequence'] for r in post}),
      initial_best_mic=start,final_best_mic=h['best_feasible_mic'],
      best_unconstrained_mic=min(r['mic'] for r in post),
      median_proposed_mic=float(np.median([r['mic'] for r in post])),
      feasible_better_than_initial=sum(r['mic']<start for r in feasible),
      unique_feasible_better_than_initial=len({r['sequence'] for r in feasible if r['mic']<start}))
    extra=None
    if method=='TPE':
        distance=distances(ref);cache={};sims=[]
        for r in h['observations']:
            seq=r['sequence']
            if seq not in cache:cache[seq]=distance(seq)
            similarity=1-cache[seq]/len(ref)
            assert r['feasible']==(similarity>=.75-1e-8),(path,seq)
            sims.append(similarity)
        sims=np.asarray(sims[100:])
        bins=[]
        for i in range(5):
            a=post[i*1000:(i+1)*1000];s=sims[i*1000:(i+1)*1000]
            bins.append(dict(proposals=len(a),feasible=sum(r['feasible'] for r in a),similarity_sum=float(s.sum())))
        result.update(mean_similarity=float(sims.mean()),max_similarity=float(sims.max()),
          median_similarity=float(np.median(sims)),feasibility_flags_independently_verified=True)
        # One concrete failure and, where available, one genuine improvement.
        extra=dict(task=result['task'],reference=ref,first_proposal=post[0],
            first_similarity=float(sims[0]),best_unconstrained=min(post,key=lambda r:r['mic']),
            best_unconstrained_similarity=1-distance(min(post,key=lambda r:r['mic'])['sequence'])/len(ref),
            best_new_feasible=min(feasible,key=lambda r:r['mic']) if feasible else None,bins=bins)
    return result,extra

def main():
    rng=np.random.default_rng(1)
    # Independently check the bit-parallel edit distance used in the full audit.
    for _ in range(300):
        a=''.join(rng.choice(list('ACDEFG'),size=int(rng.integers(1,31))))
        b=''.join(rng.choice(list('ACDEFG'),size=int(rng.integers(0,31))))
        assert distances(a)(b)==dp_distance(a,b)
    with Path('/root/reports/peptide_convergence_20261101/task_summary.csv').open() as f:
        expected={(r['method'],r['task']):r['history_sha256'] for r in csv.DictReader(f)}
    jobs=[(m,p,expected[m,p.name[:-7]]) for m,root in ROOTS.items() for p in sorted((root/'runs').glob('*.tar.gz'))]
    records=[];extras=[]
    with ThreadPoolExecutor(max_workers=2) as executor:
        for i,(record,extra) in enumerate(executor.map(read_one,jobs),1):
            records.append(record)
            if extra:extras.append(extra)
            if i%25==0:print(f'Audited {i}/600 histories',flush=True)
    summary={}
    for method in ROOTS:
        rs=[r for r in records if r['method']==method]
        d={k:sum(r[k] for r in rs) for k in ['proposals','fresh','feasible','fresh_feasible','unique_feasible','unique_sequences','feasible_better_than_initial','unique_feasible_better_than_initial']}
        d.update(tasks=len(rs),tasks_with_no_new_feasible=sum(r['feasible']==0 for r in rs),
            tasks_improved=sum(r['final_best_mic']<r['initial_best_mic'] for r in rs),
            feasible_percent=100*d['feasible']/d['proposals'],fresh_feasible_percent=100*d['fresh_feasible']/d['fresh'])
        summary[method]=d
    tr=[r for r in records if r['method']=='TPE']
    summary['TPE'].update(mean_similarity=float(np.mean([r['mean_similarity'] for r in tr])),
        tasks_with_no_fresh_feasible=sum(r['fresh_feasible']==0 for r in tr),
        feasible_proposals_per_task_range=[min(r['feasible'] for r in tr),max(r['feasible'] for r in tr)])
    bins=[]
    for i in range(5):
        n=sum(e['bins'][i]['proposals'] for e in extras);f=sum(e['bins'][i]['feasible'] for e in extras)
        bins.append(dict(proposal_range=[1000*i+1,1000*(i+1)],total_proposals=n,feasible=f,
            feasible_percent=100*f/n,mean_similarity=sum(e['bins'][i]['similarity_sum'] for e in extras)/n))
    keys=list(dict.fromkeys(k for r in records for k in r))
    with (OUT/'per_task.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)
    audit=dict(summary=summary,tpe_progress_bins=bins,examples=[e for e in extras if e['task']=='pep_925' or (e['best_new_feasible'] and e['best_new_feasible']['mic']<next(r['initial_best_mic'] for r in tr if r['task']==e['task']))],
        evidence='All 600 history hashes matched the prior full archive-checksum audit; TPE feasibility independently recomputed for all 1,020,000 initialization and proposal rows. No inference, oracle calls or experiment changes.',
        sources={m:str(p) for m,p in ROOTS.items()})
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(summary=summary,tpe_progress_bins=bins),indent=2),flush=True)

if __name__=='__main__':main()
