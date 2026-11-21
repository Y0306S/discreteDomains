"""Compare four/eight replicas on identical training-only requests; no SQL."""
import concurrent.futures
import json
from pathlib import Path
import queue
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'batch_code'))
from query_data import build_prompt
from query_protocol import atomic
from qwen_inference_transport import digest,http_json


def bodies():
    cfg=json.loads((ROOT/'batch_recipe.json').read_text())
    folder=Path('/root/data/cys/baseline/campaigns/query_our_orpo_rl800_20261112/assets/histories')
    tasks=[t for t in ['CEB_10A19','CEB_11A1','CEB_11A10','CEB_12A1'] if t in cfg['train_tasks'] and (folder/(t+'.json')).exists()]
    if len(tasks)<3:
        tasks=sorted(t for t in cfg['train_tasks'] if (folder/(t+'.json')).exists())[:3]
    result=[]
    for i in range(32):
        task=tasks[i%len(tasks)]
        assert task not in cfg['tasks']
        records=json.loads((folder/(task+'.json')).read_text())['records'][:80+i%3*20]
        messages,_=build_prompt(task,records,scenario='normal',round_number=25)
        body=dict(identity=cfg['models']['orpo']['identity'],messages=messages,
                  settings=dict(cfg['sampling_by_stage']['orpo'],enable_thinking=False),
                  task='capacity_training_only/'+task,rng=dict(seed=191000+i,cuda_rng=None),use_s1b_template=True,
                  namespace=dict(capacity_probe=True,training_only=True,index=i))
        body['request_id']=digest(body)
        result.append(body)
    return result


def run(name,ports,requests):
    work=queue.Queue()
    for i,b in enumerate(requests):work.put((i,b))
    def worker(port):
        replies=[]
        while True:
            try:i,body=work.get_nowait()
            except queue.Empty:return replies
            r=http_json(f'http://127.0.0.1:{port}/generate',body,timeout=240)
            assert r['identity']==body['identity'] and r['request_id']==body['request_id']
            replies.append(dict(index=i,port=port,input_tokens=r['generation']['input_tokens'],output_tokens=r['generation']['output_tokens'],seconds=r['finished_at']-r['started_at'],peak_reserved_gib=r['peak_reserved_gib']))
    started=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ports)) as executor:
        values=[r for group in executor.map(worker,ports) for r in group]
    elapsed=time.time()-started
    result=dict(name=name,started=started,finished=time.time(),replicas=len(ports),requests=len(values),wall_seconds=elapsed,requests_per_hour=len(values)*3600/elapsed,output_tokens_per_second=sum(x['output_tokens'] for x in values)/elapsed,mean_input_tokens=sum(x['input_tokens'] for x in values)/len(values),responses=values,training_only=True,SQL_calls=0)
    atomic(HERE/(name+'.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='responses'}),flush=True)
    return result


def main():
    assert (ROOT/'PAUSE').exists()
    h=http_json('http://127.0.0.1:21811/health',timeout=30)
    assert h['queued']==0 and not any(r.get('busy') for r in h['replica_health'])
    requests=bodies()
    ports=[21800,21803,21801,21802,21804,21805,21806,21807]
    run('warmup_eight',ports,requests[:8])
    four=run('llm_four',ports[:4],requests)
    eight=run('llm_eight',ports,requests)
    four_repeat=run('llm_four_repeat',ports[:4],requests[:16])
    rate4=(four['requests']+four_repeat['requests'])*3600/(four['wall_seconds']+four_repeat['wall_seconds'])
    ratio=eight['requests_per_hour']/rate4
    result=dict(at=time.time(),four_replicas_per_hour=rate4,eight_replicas_per_hour=eight['requests_per_hour'],eight_over_four=ratio,selected_llm=8 if ratio>1.10 else 4,criterion='Keep eight only if identical training-only request throughput improves by more than 10%.',SQL_calls=0)
    atomic(HERE/'llm_benchmark.json',result)
    print('DECISION',json.dumps(result),flush=True)


if __name__=='__main__':main()
