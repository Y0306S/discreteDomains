import json
import os
import torch
import torch.distributed as dist
from datetime import timedelta
from protocol import ROOT, atomic

rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
dist.init_process_group('nccl',timeout=timedelta(seconds=180))
x=torch.tensor([rank+1.],device='cuda');dist.all_reduce(x)
if x.item()!=10:raise ValueError('Four-GPU collective failed')
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
q=torch.randn(1,4,64,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
    result=torch.nn.functional.scaled_dot_product_attention(q,q,q,is_causal=True)
    result.float().square().mean().backward()
if not torch.isfinite(q.grad).all():raise ValueError('FlashAttention backward failed')
record=dict(rank=rank,gpu=torch.cuda.get_device_name(rank),flash_forward_backward=True,collective_sum=x.item())
records=[None]*4;dist.all_gather_object(records,record)
if rank==0:
    atomic(ROOT/'distributed_preflight.json',dict(passed=True,world_size=4,ranks=records))
    print(json.dumps(records),flush=True)
dist.barrier();dist.destroy_process_group()
