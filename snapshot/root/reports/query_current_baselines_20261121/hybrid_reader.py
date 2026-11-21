"""Read-only matched query comparison; all reported results come from saved measurements."""
import collections,csv,hashlib,json,math,pickle,sqlite3,sys,time
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent
B=Path('/root/data/cys/baseline/campaigns')
N=B/'query_our_fsbo_vae80_test200_20261117'
P=B/'query_our_fsbo_test200_20261116'
S=B/'semanticopt_eval200_20261110/query/runs'
sys.path.insert(0,str(N/'operations/recovery'));import recovery
from query_data import plan_key
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,j):(O/p).write_text(json.dumps(j,indent=2,allow_nan=False))
def load(root):
 out={}
 for p in sorted((root/'orpo_batch10/runs').glob('*/runtime.sqlite')):
  es=recovery.read_journal_entries(p)
  if 'optimization_state' not in es:continue
  st=es['optimization_state'];st['snapshot_at']=time.time();st['journal']=str(p)
  for e in st['events']:
   e['last_measurement_at']=max(es.get(f"measurement/{e['round']}/{i}",{}).get('at',float('inf')) for i in range(10))
  assert len(st['rows'])==30+10*len(st['events'])
  out[p.parent.name]=st
 return out
def best(rows):return min((r for r in rows if not r['censored']),key=lambda r:r['runtime'])
def curve(rows):
 v=float('inf');ans=[]
 for r in rows:
  if not r['censored']:v=min(v,r['runtime'])
  ans.append(v)
 return np.asarray(ans)
def comparison(rows,key):
 ratios=np.array([r[key] for r in rows],dtype=float);logs=np.log(ratios);rng=np.random.default_rng(123)
 if not len(rows):return dict(queries=0)
 ci=np.exp(np.quantile(rng.choice(logs,(10000,len(rows))).mean(axis=1),[.025,.975])).tolist()
 g=float(np.exp(logs.mean()))
 return dict(queries=len(rows),geomean_new_over_comparator=g,runtime_reduction_percent=100*(1-g),geomean_speedup=1/g,median_ratio=float(np.median(ratios)),wins_over_5pct=int((ratios<.95).sum()),losses_over_5pct=int((ratios>1.05).sum()),within_5pct=int(((ratios>=.95)&(ratios<=1.05)).sum()),descriptive_task_bootstrap_95pct_ratio_interval=ci)
def diagnostic(states,limits=None):
 c=collections.Counter();sources=collections.Counter();best_sources=collections.Counter()
 for task,st in states.items():
  n=(limits or {}).get(task,len(st['events']));ev=st['events'][:n];rows=st['rows'][:30+10*n]
  best_sources[best(rows)['source_phase']]+=1
  incumbent=best(rows[:30])['runtime']
  for e in ev:
   for r in e['results']:
    row=r['row'];a=r['actual_arm'];sources[a]+=1;c['timeouts_'+a]+=bool(row['censored'])
    if not row['censored'] and row['runtime']<incumbent:
     incumbent=row['runtime'];c['incumbent_improvements_'+a]+=1
   d=e.get('fsbo_diagnostics')
   if d:
    c['fsbo_requested']+=d['requested'];c['fsbo_accepted']+=d['accepted'];c['fsbo_decode_rejections']+=len(d['rejected_decodes'])
 return dict(counts=dict(c),proposal_sources=dict(sources),best_sources=dict(best_sources),fsbo_slot_fill_percent=100*c['fsbo_accepted']/c['fsbo_requested'] if c['fsbo_requested'] else None,fsbo_returned_decode_acceptance_percent=100*c['fsbo_accepted']/(c['fsbo_accepted']+c['fsbo_decode_rejections']) if c['fsbo_accepted'] else None)
def write_csv(name,rows):
 if not rows:return
 with (O/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 started=time.time();new=load(N);old=load(P);print('Loaded hybrid snapshots',len(new),len(old),flush=True)
 zpath=Path('/root/reports/query_convergence_four_methods_20261117/per_task_curves.npz');z=np.load(zpath)
 idx={str(t):i for i,t in enumerate(z['tasks'])}
 sources=json.loads(Path('/root/reports/query_convergence_fsbo_tpe_lolbo_20261109/audit.json').read_text())['sources']
 fsources={s['task']:s for s in sources if s['method']=='FSBO'}
 baselines={};source_evidence=[]
 for task in new:
  sr=fsources[task];p=Path(sr['source']);assert sha(p)==sr['sha256'];doc=json.loads(p.read_text());h=doc['observations'];assert len(h)==1049
  fc=[];v=float('inf')
  for i,r in enumerate(h):
   e=r['extra_info']
   if not e['censored']:v=min(v,float(e['runtime']))
   if i>=48:fc.append(v)
  np.testing.assert_allclose(fc,z['FSBO'][idx[task]],rtol=1e-12)
  dbpath=S/task/'observations.sqlite'
  with sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True,timeout=30) as db:
   db.execute('BEGIN');initialraw=db.execute("select value from state where key='initial'").fetchone()[0]
   initial=json.loads(initialraw);records=db.execute('select n,value from records order by n').fetchall()
  assert len(initial)==49 and len(records)==200
  sc=[];v=min(r['objective'] for r in initial if not r['censored'] and r['status']=='success');sc.append(v)
  for expected,(n,raw) in enumerate(records):
   assert n==expected;r=json.loads(raw)
   if not r['censored'] and r['status']=='success':v=min(v,float(r['objective']))
   sc.append(v)
  np.testing.assert_allclose(sc,z['SemanticOPT'][idx[task]],rtol=1e-12)
  baselines[task]=dict(FSBO=np.array(fc),SemanticOPT=np.array(sc))
  source_evidence.append(dict(task=task,frozen_fsbo_history=str(p),sha256=sr['sha256'],semanticopt_sqlite=str(dbpath),semanticopt_logical_sha256=hashlib.sha256((initialraw+''.join(raw for _,raw in records)).encode()).hexdigest()))
 print('Verified baseline raw histories and cached curves',flush=True)
 rows=[]
 for task,ns in new.items():
  if len(ns['rows'])<230:continue
  n=best(ns['rows'][:230]);ba=baselines[task];warm=best(ns['rows'][:30])['runtime']
  row=dict(task=task,new_optimization_proposals_available=len(ns['rows'])-30,new_best_at_200=n['runtime'],new_best_source_at_200=n['source_phase'],bolt30_best=warm,bao49_best=ba['FSBO'][0],frozen_fsbo_best_200=ba['FSBO'][200],semanticopt_best_200=ba['SemanticOPT'][200],new_over_bolt30=n['runtime']/warm,new_over_frozen200=n['runtime']/ba['FSBO'][200],new_over_semantic200=n['runtime']/ba['SemanticOPT'][200],new_over_frozen_equal_total230=n['runtime']/ba['FSBO'][181],new_over_semantic_equal_total230=n['runtime']/ba['SemanticOPT'][181])
  if task in old and len(old[task]['rows'])>=230:
   assert [(r['plan'],r['runtime'],r['censored']) for r in ns['rows'][:30]]==[(r['plan'],r['runtime'],r['censored']) for r in old[task]['rows'][:30]]
   row.update(old_hybrid_best_200=best(old[task]['rows'][:230])['runtime'])
   row['new_over_old_hybrid200']=n['runtime']/row['old_hybrid_best_200']
  rows.append(row)
 # Use explicit missing columns for tasks absent from the previous hybrid run.
 for r in rows:
  r.setdefault('old_hybrid_best_200',None);r.setdefault('new_over_old_hybrid200',None)
 write_csv('matched_200_proposals.csv',rows)
 equal_hybrid=[];pre=[];limits={};boundary=json.loads((N/'evidence/ram_storage_boundary_20261118.json').read_text())
 for task in sorted(new.keys()&old.keys()):
  ns,os=new[task],old[task];n=min(len(ns['events']),len(os['events']));limits[task]=n
  if n:
   nr=ns['rows'][:30+10*n];orr=os['rows'][:30+10*n]
   assert [(r['plan'],r['runtime'],r['censored']) for r in nr[:30]]==[(r['plan'],r['runtime'],r['censored']) for r in orr[:30]]
   equal_hybrid.append(dict(task=task,optimization_rounds_each=n,optimization_proposals_each=n*10,new_best=best(nr)['runtime'],old_best=best(orr)['runtime'],new_over_old=best(nr)['runtime']/best(orr)['runtime'],new_best_source=best(nr)['source_phase']))
  npre=min(sum(e['last_measurement_at']<=boundary['at'] for e in ns['events']),len(os['events']))
  if npre:
   nr=ns['rows'][:30+10*npre];orr=os['rows'][:30+10*npre]
   pre.append(dict(task=task,optimization_rounds_each=npre,new_over_old=best(nr)['runtime']/best(orr)['runtime']))
 write_csv('hybrid_equal_available_rounds.csv',equal_hybrid);write_csv('hybrid_before_ram_change.csv',pre)
 shared=[r for r in rows if r['new_over_old_hybrid200'] is not None]
 fixed_summary={k:comparison(rows,key) for k,key in [('versus_frozen_FSBO','new_over_frozen200'),('versus_SemanticOPT','new_over_semantic200'),('versus_BOLT_initialization','new_over_bolt30')]}
 fixed_summary['versus_previous_hybrid']=comparison(shared,'new_over_old_hybrid200')
 same_cohort={k:comparison(shared,key) for k,key in [('versus_frozen_FSBO','new_over_frozen200'),('versus_SemanticOPT','new_over_semantic200'),('versus_previous_hybrid','new_over_old_hybrid200')]}
 same_cohort_equal_total={k:comparison(shared,key) for k,key in [('versus_frozen_FSBO','new_over_frozen_equal_total230'),('versus_SemanticOPT','new_over_semantic_equal_total230')]}
 # Geometric relative-to-Bao incumbents on one common task cohort.
 methods=['Retrained hybrid','Previous hybrid','Frozen FSBO','SemanticOPT'];curves={m:[] for m in methods}
 for r in shared:
  t=r['task'];anchor=r['bao49_best']
  for name,st in [('Retrained hybrid',new[t]),('Previous hybrid',old[t])]:curves[name].append(curve(st['rows'][:230])[29:230]/anchor)
  for name,key in [('Frozen FSBO','FSBO'),('SemanticOPT','SemanticOPT')]:curves[name].append(baselines[t][key][:201]/anchor)
 mean_curves={m:np.exp(np.log(np.array(v)).mean(axis=0)) for m,v in curves.items()}
 curve_rows=[dict(proposals=i,**{m:float(v[i]) for m,v in mean_curves.items()}) for i in range(201)];write_csv('matched_convergence.csv',curve_rows)
 training=dict(vae=json.loads((N/'training/vae/status.json').read_text()),fsbo=json.loads((N/'training/fsbo/status.json').read_text()))
 status=json.loads((N/'status.json').read_text());monitor=json.loads(Path('/root/.codex/loops/query-training-hourly-20261112/state.json').read_text())
 result=dict(at=time.time(),snapshot_started=started,snapshot_finished=time.time(),status=status,monitor=monitor,snapshot_tasks=len(new),snapshot_optimization_measurements=sum(len(s['rows'])-30 for s in new.values()),
  fixed_200_proposals=fixed_summary,common_four_method_cohort_200=same_cohort,equal_total230=same_cohort_equal_total,
  equal_available_rounds_hybrid=comparison(equal_hybrid,'new_over_old'),pre_ram_equal_available_rounds_hybrid=comparison(pre,'new_over_old'),
  retrained_all_diagnostics=diagnostic(new),matched_retrained_diagnostics=diagnostic({t:new[t] for t in limits},limits),matched_previous_diagnostics=diagnostic({t:old[t] for t in limits},limits),training=training,
  paired_200_queries=len(shared),source_evidence=source_evidence,cache_curves_sha256=sha(zpath),storage_change_at=boundary['at'],storage_change_measurements=boundary['measurements_before'],
  limitations=['Partial nonrandom task cohort; no full-test conclusion.','Only uncensored successful measured outcomes can improve incumbents.','New and old hybrids share exact BOLT30 measurements; standalone frozen FSBO and SemanticOPT share Bao49 instead.','At equal 200 new proposals, total counts are 230 hybrid versus 249 baselines; equal-total230 sensitivity uses 181 baseline proposals.','Storage, hosts, dates and seeds differ. RAM/cache change can affect timing. This is not a controlled causal retraining ablation.','SemanticOPT has only 200 measured proposals; no flat extrapolation used.','Task bootstrap describes this partial cohort; it is not uncertainty over training/evaluation seeds.'])
 dump('summary.json',result)
 np.savez(O/'curves.npz',**mean_curves,tasks=np.array([r['task'] for r in shared]))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(9,5.3));colors=['#CC79A7','#777777','#009E73','#00A6BF']
 for m,c in zip(methods,colors):ax.plot(range(201),mean_curves[m],label=m,color=c,lw=2.2,linestyle='--' if m=='Previous hybrid' else '-')
 ax.set(xlabel='New measured proposals (initialization excluded)',ylabel='Geometric mean runtime / Bao49 best (lower is better)',title=f'Query: matched partial cohort of {len(shared)} tasks',xlim=(0,200));ax.grid(axis='y',alpha=.2);ax.legend(frameon=False)
 fig.text(.5,.015,'Hybrids: BOLT30 starts; baselines: Bao49. Historical timings; storage differs.',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.04,1,1))
 for ext in ['png','pdf']:fig.savefig(O/f'comparison.{ext}',dpi=180)
 print(json.dumps({k:result[k] for k in ['status','fixed_200_proposals','common_four_method_cohort_200','equal_total230','equal_available_rounds_hybrid','pre_ram_equal_available_rounds_hybrid','retrained_all_diagnostics']},indent=2),flush=True)
if __name__=='__main__':main()
