import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;TOOLS=Path('/root/data/cys/llmoptimizer_benchmark/tools');REPAIR=Path('/root/reports/peptide_rgpe_sobol_recovery_20261102/code')
sys.path[:0]=[str(TOOLS),str(REPAIR)]
from rgpe_cuda_client import Client,install

def main():
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--gpu-config',required=True);p.add_argument('--endpoint',required=True);a,rest=p.parse_known_args()
 config=json.loads(Path(a.gpu_config).read_text())
 validation=json.loads(Path(config['validation']).read_text())
 if not validation['passed'] or hashlib.sha256(Path(config['validation']).read_bytes()).hexdigest()!=config['validation_sha256']:raise ValueError('GPU validation changed or failed')
 for name in config['numerical_code']:
  if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=validation['code'][name]:raise ValueError('GPU numerical code differs from validation')
 for n,h in config['code'].items():
  if hashlib.sha256(Path(n).read_bytes()).hexdigest()!=h:raise ValueError('GPU execution code changed: '+n)
 sys.path.insert(0,'/root/data/cys/baseline/repos/transfer-hpo-framework')
 import peptide_transfer_models as models
 from sobol_repair import install as sobol_install,verify
 verify(config['sobol_repair']);sobol_install()
 client=Client(a.endpoint,config['token_file']);client.telemetry=Path(rest[rest.index('--telemetry')+1]).with_name('rgpe_gpu_stage.json');install(client)
 import peptide_rgpe_merged_worker as worker
 base=worker.MergedRGPEBackend
 class GPUBackend(base):
  def fit(self,x,score,seed):
   client.y=(-__import__('numpy').asarray(score)-(-__import__('numpy').asarray(score)).mean())/((-__import__('numpy').asarray(score)).std() or 1.)
   client.source_seconds=client.loo_seconds=0.;client.source_count=0;client.write_stage('target_fit_cpu',observations=len(score))
   result=super().fit(x,score,seed)
   result.update(device='hybrid_cpu_fit_cuda_ranking_covariance_fp64',endpoint=a.endpoint,source_rank_seconds=client.source_seconds,loo_seconds=client.loo_seconds)
   telemetry=Path(rest[rest.index('--telemetry')+1]).with_name('rgpe_gpu_fit.json');worker.atomic_json(telemetry,dict(at=time.time(),info=result))
   return result
 worker.MergedRGPEBackend=GPUBackend
 telemetry=Path(rest[rest.index('--telemetry')+1]).with_name('rgpe_gpu_worker.json');worker.atomic_json(telemetry,dict(at=time.time(),endpoint=a.endpoint,config=a.gpu_config,device='cuda_fp64'))
 sys.argv=[sys.argv[0]]+rest;worker.main()
if __name__=='__main__':main()
