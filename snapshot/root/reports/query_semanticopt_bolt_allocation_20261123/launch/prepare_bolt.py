from pathlib import Path
import json,shutil,hashlib,time
BASE=Path('/root/data/cys/baseline/campaigns')
SRC=BASE/'query_bolt_lolbo_test200_20261117';ROOT=BASE/'query_bolt30_lolbo_total800_20261123';SEM=BASE/'semanticopt_query_400_pilot800_20261123'
assert not ROOT.exists()
for sub in ['runs','services','logs','evidence','control']:(ROOT/sub).mkdir(parents=True,exist_ok=True)
cfg=json.loads((SRC/'recipe.json').read_text());cfg.update(name=ROOT.name,rounds=77,total_observations=800,new_evaluations=770,workers=json.loads((SEM/'allocation.json').read_text())['bolt_workers'],proposal_policy=dict(max_latent_batches=200,native_vae_execution_canonicalization=True,decoder_fallback='reject as failure; never evaluate substituted plans',decode_concurrency=8,seed_policy='per latent: base seed + vector index, retained under parallel dispatch'),authorized_at=time.time(),authorization='User authorized launch: reuse BOLT30, continue to 800 total evaluations, seven initial SQL workers, eight VAE services')
cfg['vae_pool']=str(ROOT/'services/vae_pool.json')
(ROOT/'recipe.json').write_text(json.dumps(cfg,indent=2)+'\n')
(ROOT/'services/vae_pool.json').write_text(json.dumps(dict(domain='ceb_query_runtime',token_file=str(BASE/'query_our_fsbo_vae80_test200_20261117/services/token'),workers=[dict(url=f'http://127.0.0.1:{23300+i}') for i in range(8)]),indent=2)+'\n')
s=(SRC/'run_lolbo.py').read_text()
s=s.replace("method='query_vae_lolbo_dkl_ts', rounds=80, batch_size=10)","method='query_vae_lolbo_dkl_ts_native_codec', rounds=77, batch_size=10)")
s=s.replace('for retry in range(20):',"for retry in range(recipe()['proposal_policy']['max_latent_batches']):")
s=s.replace("plan = canonicalize(task, item['plan'], strict=True)","plan = canonicalize(task, item['plan'])\n                plan = canonicalize(task, plan, strict=True)")
s=s.replace("accepted.append(dict(plan=plan, proposed_z=latent))","accepted.append(dict(plan=plan, proposed_z=latent, raw_vae_tokens=item['plan'], decode_retries=item.get('decode_retries'), native_canonicalized=item['plan'] != plan))")
s=s.replace("rejected.append(dict(retry=retry, error=str(exc)))","rejected.append(dict(retry=retry, error=str(exc), decoded=item))")
marker="            if len(accepted) == 10:\n                return dict(items=accepted, rejected=rejected,"
s=s.replace(marker,"            if len(accepted) == 10:\n                return dict(items=accepted, rejected=rejected,")
# Persist generation diagnostics even before a complete proposal batch is available.
marker="    raise RuntimeError(f'Could only decode {len(accepted)}/10 unique feasible plans')"
s=s.replace(marker,"        atomic(ROOT / 'runs' / task / 'proposal_progress.json', dict(task=task, round=round_number, latent_batches=retry+1, latent_attempts=(retry+1)*10, accepted=len(accepted), rejected=len(rejected), last_rejections=rejected[-10:], at=time.time()))\n"+marker)
start=s.index('    def decode_round_robin(');end=s.index("    folder = ROOT / 'runs' / task",start)
s=s[:start]+'''    from concurrent.futures import ThreadPoolExecutor
    def decode_parallel(self, vectors, seed=None):
        if seed is None:
            seed = self.seed + self.decode_index
            self.decode_index += 1
        endpoints = self.pool.endpoints
        offset = int(os.environ.get('QUERY_LOLBO_VAE_OFFSET', '0')) % len(endpoints)
        def one(pair):
            index, vector = pair
            endpoint = endpoints[(offset + index) % len(endpoints)]
            payload = json.dumps(dict(z=[vector], task=self.task, fallback=self.fallback,
                op='decode', seed=(seed+index) % (2**31-1)), allow_nan=False).encode()
            req = urllib.request.Request(endpoint+'/inference', data=payload,
                headers={'Content-Type':'application/json', 'Authorization':'Bearer '+self.pool.token})
            with urllib.request.urlopen(req, timeout=600) as response:
                result = json.load(response)
            if 'error' in result: raise RuntimeError(result['error'])
            return result['rows'][0]
        with ThreadPoolExecutor(max_workers=min(8,len(vectors))) as executor:
            return list(executor.map(one, enumerate(vectors)))
    vae.decode = types.MethodType(decode_parallel, vae)
''' + s[end:]
s=s.replace('target = min(80, max_rounds) if max_rounds is not None else 80','target = min(77, max_rounds) if max_rounds is not None else 77')
s=s.replace('identity=identity(task), attempts=800,','identity=identity(task), attempts=770,')
(ROOT/'run_lolbo.py').write_text(s)
shutil.copy2(SRC/'evidence/gpu3_canary_report.json',ROOT/'evidence/previous_failed_canary.json')
shutil.copy2(SEM/'allocation.json',ROOT/'allocation.json')
(ROOT/'status.json').write_text(json.dumps(dict(status='ready_for_canary',warm_evaluations=6000,new_evaluations=0,new_evaluation_target=154000,at=time.time()),indent=2))
print(ROOT)
