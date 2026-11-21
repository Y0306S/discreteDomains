import argparse,json,runpy,subprocess,sys
from pathlib import Path
TOOLS=Path('/root/data/cys/llmoptimizer_benchmark/tools');ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--gpu-config',required=True);p.add_argument('--host',required=True);a,rest=p.parse_known_args();config=json.loads(Path(a.gpu_config).read_text())
 cpu=int(rest[rest.index('--cpu')+1]);endpoint=config['hosts'][a.host]['endpoints'][cpu%len(config['hosts'][a.host]['endpoints'])]
 original=subprocess.Popen
 def popen(command,*positional,**keywords):
  if isinstance(command,list) and len(command)>1 and command[1]==str(TOOLS/'peptide_rgpe_merged_worker.py'):
   command=[command[0],str(ROOT/'rgpe_worker.py'),'--gpu-config',a.gpu_config,'--endpoint',endpoint]+command[2:]
  return original(command,*positional,**keywords)
 subprocess.Popen=popen;sys.path.insert(0,str(TOOLS));sys.argv=[str(TOOLS/'peptide_rgpe_merged_launch.py')]+rest;runpy.run_path(sys.argv[0],run_name='__main__')
if __name__=='__main__':main()
