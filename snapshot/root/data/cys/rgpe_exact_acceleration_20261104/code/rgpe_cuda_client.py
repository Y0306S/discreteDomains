"""Validated v2 CUDA client; original optimizer, RNG and joint sampling retained."""
import importlib.util
import pickle
import time
import urllib.request
from pathlib import Path
import numpy as np

LEGACY=Path('/root/data/logs/peptide_gpu_acceleration_20261102/code/rgpe_cuda_client.py')
spec=importlib.util.spec_from_file_location('rgpe_legacy_cuda_client',LEGACY)
legacy=importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


class Client(legacy.Client):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.queue_seconds=self.compute_seconds=0.
        self.last_timing={}

    def call(self,op,**values):
        req=urllib.request.Request(self.endpoint+'/rpc',
            data=pickle.dumps(dict(op=op,**values),protocol=4),
            headers={'Content-Type':'application/octet-stream','Authorization':'Bearer '+self.token})
        with urllib.request.urlopen(req,timeout=6*3600) as response:
            reply=pickle.loads(response.read())
        self.calls+=1
        self.last_timing={k:reply.get(k,0.) for k in ('queue_seconds','compute_seconds')}
        self.queue_seconds+=self.last_timing['queue_seconds']
        self.compute_seconds+=self.last_timing['compute_seconds']
        if 'error' in reply:
            raise RuntimeError('CUDA math: '+reply['error'])
        return reply['result']

    def sample(self,ensemble,candidates,count,seed):
        from peptide_rgpe_merged import sample_posterior, VERSION
        started=time.monotonic()
        if ensemble.variance_mode!='average':
            raise ValueError('Expected independent RGPE average variance')
        x=np.asarray(candidates,dtype=np.float64)
        if x.ndim!=2 or not len(x) or not np.isfinite(x).all():
            raise ValueError('Invalid candidates')
        bank=ensemble.model_list_
        target=self.payload(bank.target,x,source_output='standardized')
        target.pop('x')
        self.write_stage('cuda_covariance_merge',candidates=len(x))
        result=self.call('merge',x=x,sources=bank.tasks,weights=ensemble.weights_,target=target,
                         scale=ensemble.Y_std_,offset=ensemble.Y_mean_)
        merge_seconds=time.monotonic()-started
        self.write_stage('joint_sampling_cpu',candidates=len(x),merge_seconds=merge_seconds,
                         **self.last_timing)
        draws,sampling=sample_posterior(result['mean'],result['covariance'],count,seed)
        return dict(draws=draws,info=dict(sampling,method='rgpe',posterior='merged_independent_GP',
            higher_is_better=True,implementation=VERSION,arithmetic_implementation=result['implementation'],
            contributing_models=result['contributing_models'],candidates=len(x),covariance_mode='full',
            covariance_bytes=result['covariance'].nbytes,source_output='standardized',
            independence_assumption=True,source_covariance_modes=['cuda_exact'],retained_source_covariances=1,
            merge_seconds=merge_seconds,merge_compute_seconds=result['seconds'],
            merge_queue_seconds=self.last_timing['queue_seconds'],
            sample_seconds=time.monotonic()-started-merge_seconds,
            source_cache_hits=result['cache_hits'],source_cache_misses=result['cache_misses']))


def install(client):
    legacy.install(client)
    import peptide_rgpe_merged_worker as worker
    worker.MergedRGPEBackend.sample=lambda self,candidates,count,seed: client.sample(self.ensemble,candidates,count,seed)
