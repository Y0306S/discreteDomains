"""Frozen local/recenter FSBO, batch 50, using validated CUDA services."""
import argparse,json,sys,time
from pathlib import Path
TOOLS=Path('/root/data/cys/llmoptimizer_benchmark/tools')
sys.path.insert(0,str(TOOLS))
from peptide_fsbo_gpu_eval import GPUModelClient,runtime
import peptide_local_bo as runner

def main():
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--gpu-config',required=True);a,rest=p.parse_known_args()
 config=runtime(a.gpu_config)
 cpu=int(rest[rest.index('--cpu')+1]);task=rest[rest.index('--task')+1]
 endpoints=[config['endpoints'][i] for i in (0,2,3)]
 def client(path,recipe):return GPUModelClient(path,recipe,config,endpoints[(cpu-64)%len(endpoints)])
 runner.transfer.ModelClient=client
 campaign=Path(rest[rest.index('--campaign')+1]);dest=campaign/'dispatch'/'execution';dest.mkdir(parents=True,exist_ok=True)
 (dest/(task+'.json')).write_text(json.dumps(dict(at=time.time(),batch_size=50,device='cuda',backend='GPyTorch',coordinator_cpu=cpu,endpoint=endpoints[(cpu-64)%len(endpoints)],model_runtime=a.gpu_config,adapter=str(TOOLS/'peptide_local_bo.py')),indent=2))
 sys.argv=[runner.__file__]+rest;runner.main()
if __name__=='__main__':main()
