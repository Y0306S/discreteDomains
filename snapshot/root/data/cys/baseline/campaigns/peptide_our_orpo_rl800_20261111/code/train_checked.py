"""Guarded entry to the copied, audited ML LLaMA-Factory ORPO trainer."""
import argparse
import json
from pathlib import Path
from peptide_prompt import file_sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);a=p.parse_args()
    root=a.campaign.resolve()
    audit=json.loads((root/'reports/independent_audit.json').read_text())
    assert audit['passed'] and audit['tasks_audited']==800
    assert audit['data_sha256']==file_sha(root/'data/orpo_train.jsonl')
    for filename,sha in audit['code_sha256'].items():
        assert file_sha(root/'code'/filename)==sha, f'Audited code changed: {filename}'
    config=json.loads((root/'orpo_config.json').read_text())
    assert config['model_name_or_path']=='/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B'
    assert not config.get('adapter_name_or_path') and config['pref_loss']=='orpo'
    assert config['flash_attn']=='sdpa'
    import torch
    assert torch.cuda.device_count()==1, 'This entry uses the ML single-GPU recipe; set one visible GPU'
    free,total=torch.cuda.mem_get_info(0)
    if free<54*2**30:
        raise RuntimeError(f'Need at least 54 GiB free before training; available {free/2**30:.2f}')
    from ml_orpo_train import train
    train(root)


if __name__=='__main__':main()
