"""Exercise the new gate with real saved pep_999 weights; no training/oracle."""
import importlib.util
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
import common
import torch
import pandas as pd
from uniref_vae import data as native_data
import sys
sys.modules['data']=native_data
from target_gate import cpu_state, guarded_retrain


def main():
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    parent=Path(common.read(common.ROOT/'config.json')['parent_experiment'])
    spec=importlib.util.spec_from_file_location('author_cpu_vae',common.ROOT/'reference/APEXGo/generation/vae.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    vocab=pd.read_csv(common.UPSTREAM/'uniref_vae/1mer_vocab.csv',header=None).values.squeeze().tolist()
    data=native_data.DatasetKmers(k=1,load_data=False,vocab=vocab)
    model=module.InfoTransformerVAE(dataset=data,d_model=128,kl_factor=.0001,
        encoder_dim_feedforward=256,decoder_dim_feedforward=256,encoder_num_layers=6,decoder_num_layers=6)
    model.max_string_length=30
    initial=torch.load(parent/'vae/weights.pt',map_location='cpu',weights_only=False)
    candidate=torch.load(parent/'evaluation/adapted_online/pep_999/checkpoint.pt',map_location='cpu',weights_only=False)['vae']
    model.load_state_dict(initial,strict=True)
    result=common.read(parent/'evaluation/adapted_online/pep_999/result.json')
    state=dict(rows=[r for r in result['observations'] if r['phase']=='initialization'])
    assert len(state['rows'])==100
    # Isolate the target gate: source checks are a passing fixture here, not a
    # claimed rerun of the source corpus probe. This replays checkpoint weights,
    # not intermediate historical online training decisions.
    def source_fixture(model,data):
        return dict(feasible=128,reconstruction_loss=1.,validation_fixture=True)
    def candidate_swap(records):
        model.load_state_dict(candidate,strict=True)
    started=time.monotonic()
    event=guarded_retrain(model,data,state,candidate_swap,source_fixture,source_fixture(None,None))
    restored=cpu_state(model)
    identical=all(torch.equal(v,initial[k]) for k,v in restored.items())
    checks=dict(real_checkpoint_target_degradation_rejected=not event['accepted'],
        all_parameters_and_buffers_restored=identical,
        recorded_weight_delta_zero=event['vae_change_l2']==0.,
        first100_initializations_only=event['target_loss_before']['sequences']==100,
        no_training_steps=event['optimizer_steps']==0)
    report=dict(passed=all(checks.values()),checks=checks,event=event,oracle_calls=0,
        training_performed=False,device='cpu',elapsed_seconds=time.monotonic()-started,
        caveat='Final-versus-initial checkpoint replay validates rejection and restoration, not specific intermediate historical update decisions. Source probe uses a passing fixture to isolate target gate.')
    assert report['passed'],report
    common.write(common.ROOT/'validation/saved_pep999_gate.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
