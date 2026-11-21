"""Load and verify pinned SMAC sources in their original environment."""
import json,sys
from pathlib import Path
sys.path.insert(0,'/root/data/cys/llmoptimizer_benchmark/tools')
from peptide_transfer_models import SourceBank,receive,send
recipe=json.loads(Path(sys.argv[1]).read_text());bank=SourceBank(recipe)
while True:
 req=receive(sys.stdin.buffer)
 if req is None:break
 try:
  m=bank[recipe['rgpe_sources'].index(req['task'])];g=m.gp
  payload=dict(train=g.X_train_,L=g.L_,alpha=g.alpha_,theta=g.kernel_.theta,scale=float(m.std_y_),offset=float(m.mean_y_))
  send(sys.stdout.buffer,dict(result=payload));del payload,m,g
 except Exception:
  import traceback
  send(sys.stdout.buffer,dict(error=traceback.format_exc()))
