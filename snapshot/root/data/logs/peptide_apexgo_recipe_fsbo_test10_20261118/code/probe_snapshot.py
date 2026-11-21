from common import *
import torch
from reconstruction import audit
setup();cfg=prepare();state=torch.load(ROOT/'vae/state.pt',map_location='cpu',weights_only=False)
assert state['config_sha256']==sha(ROOT/'config.json')
model,data=load_vae(cfg['author_vae']);model.load_state_dict(state['vae'])
value=audit(model,data);value.update(step=state['step'],epoch=state['epoch'],checked_at=time.time())
write(ROOT/'monitor/snapshot_latest.json',value)
