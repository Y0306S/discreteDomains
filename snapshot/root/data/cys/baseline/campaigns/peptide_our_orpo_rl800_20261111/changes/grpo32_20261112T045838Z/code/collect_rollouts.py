"""Fresh phase-policy peptide rollouts, exact tokens and durable measurements."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from campaign_io import TaskStore,canonical,verify_campaign
from rl_common import ROOT,read,parent_for,adapter_digest,seed_for,stop_sampling,select_training_members,group_improved,require_execution_enabled,digest
import peptide_context as peptide

def collect(job,client,parent,signature):
    base=ROOT/'rollouts'/job['phase']/job['task_id']/('k'+str(job['prefix_k']))/'r0'
    store=TaskStore(base/'group.sqlite',dict(campaign=signature,group=job['group_id'],adapter_sha256=adapter_digest(parent)))
    try:
        if store.get('complete'):return
        h,rows,valid=peptide.prefix(job);frozen=store.get('prompt')
        if frozen is None:
            messages,meta,_,_=peptide.prompt(job);frozen=dict(messages=messages,meta=meta);store.put('prompt',frozen)
        records=[]
        for i in range(10):
            if stop_sampling(records):break
            generation=store.get('generation_'+str(i))
            if generation is None:
                generation=client.generate(frozen['messages'],n=1,temperature=1.,seed=seed_for(job['group_id'],i))[0]
                store.put('generation_'+str(i),generation)
            sample=TaskStore(base/f'sample{i}.sqlite',dict(campaign=signature,group=job['group_id'],sample=i,completion_sha256=hashlib.sha256(generation['completion'].encode()).hexdigest()))
            try:
                record=sample.get('record')
                if record is None:
                    record=peptide.sample(generation['completion'],h['reference'],rows,valid,records,lambda seq:peptide.measured(seq,sample))
                    sample.put('record',record)
            finally:sample.close()
            records.append(dict(record,schema='peptide_rl_selected4.v1',group_id=job['group_id'],prefix_best_loss=min(r['mic'] for r in valid),
                sample_i=i,task_id=job['task_id'],dataset=job['task_id'],phase=job['phase'],scenario=job['scenario'],prefix_k=job['prefix_k'],
                messages=frozen['messages'],completion=generation['completion'],prompt_ids=generation['prompt_ids'],answer_ids=generation['answer_ids'],
                prompt_meta=frozen['meta'],response_format='json_only',policy_adapter=str(parent),policy_sha256=adapter_digest(parent),seed=seed_for(job['group_id'],i)))
        selection=select_training_members(records,job['group_id'],1)
        for row in records:row.update(train_member=row['sample_i'] in selection['train_members'],group_status=selection['reason'])
        target=base/('rollouts_'+job['task_id']+'.jsonl');temp=target.with_suffix('.tmp')
        with temp.open('w') as stream:
            for row in records:stream.write(canonical(row)+'\n')
            stream.flush();os.fsync(stream.fileno())
        os.replace(temp,target)
        store.put('complete',dict(rows=len(records),improved=group_improved(records),trainable=selection['trainable'],reason=selection['reason'],selection=selection,at=time.time(),output_sha256=digest(target)))
        print(canonical(dict(group=job['group_id'],samples=len(records),selection=selection)),flush=True)
    finally:store.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--execute',action='store_true')
    p.add_argument('--endpoint',required=True);p.add_argument('--token-file',type=Path,required=True);p.add_argument('--identity-file',type=Path,required=True)
    p.add_argument('--worker-index',type=int,default=0);p.add_argument('--worker-count',type=int,default=1);a=p.parse_args()
    require_execution_enabled();signature=verify_campaign(ROOT)
    if not a.execute:raise ValueError('Missing execution flag')
    os.sched_setaffinity(0,{read('settings.json')['cpu_affinity'][a.worker_index]})
    from rollout_client import ExactRolloutClient
    parent=parent_for(a.phase);identity=json.loads(a.identity_file.read_text());client=ExactRolloutClient(a.endpoint,a.token_file,expected_identity=identity['identity_sha256'],temperature=1.)
    served=client.model_identity
    if (served['files'].get(str((parent/'adapter_model.safetensors').resolve()))!=digest(parent/'adapter_model.safetensors') or not served['client'].get('adapter_verification',{}).get('exact') or served['client'].get('rollout_tokens')!='exact_v1'):raise ValueError('Wrong rollout adapter or exact tokens missing')
    assigned={t['task_id'] for t in read('assets/tasks.json')[a.worker_index::a.worker_count]}
    for job in read('assets/jobs.json'):
        if job['phase']==a.phase and job['task_id'] in assigned:collect(job,client,parent,signature)

if __name__=='__main__':main()
