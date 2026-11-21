"""Read an atomic VAE checkpoint and measure source-only reconstruction."""
import sys
sys.path.insert(0,'/root/data/logs/peptide_vae_fsbo_source_replay_test10_20261118/code')
from common import *
from reconstruction import audit


def main():
    import torch
    setup();cfg=prepare()
    stage=read(ROOT/'vae/status.json')
    model,data=load_vae(cfg['author_vae'])
    saved=torch.load(ROOT/'vae/state.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(saved['vae'])
    value=dict(audit(model,data),source_tasks_processed=saved['index'],attempt=saved['attempt'],
        checked_at=time.time(),config_sha256=saved['config_sha256'])
    write(sys.argv[1],value);print(json.dumps(value),flush=True)


if __name__=='__main__':main()
