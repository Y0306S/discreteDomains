"""Deterministic phase passes and observable adapter changes."""
import math
import random
import torch


def training_order(count, epochs, seed):
    order=[]
    for epoch in range(epochs):
        indices=list(range(count));random.Random(seed+epoch).shuffle(indices);order.extend(indices)
    return order


def snapshot(named_parameters):
    return {name:p.detach().cpu().clone() for name,p in named_parameters}


def weight_change(named_parameters, before):
    delta_squared=0.;before_squared=0.
    with torch.no_grad():
        for name,p in named_parameters:
            old=before[name].to(p.device,dtype=torch.float32)
            delta_squared+=(p.detach().float()-old).square().sum().item()
            before_squared+=old.square().sum().item()
    l2=math.sqrt(delta_squared)
    return dict(l2=l2,relative_l2=l2/max(math.sqrt(before_squared),1e-12))


def policy_metrics(logp,old,reference,epsilon):
    with torch.no_grad():
        delta=reference.float()-logp.detach().float()
        ratio=(logp.detach().float()-old.float()).exp()
        return dict(reference_kl=(delta.expm1()-delta).mean().item(),
                    clip_fraction=((ratio<1-epsilon)|(ratio>1+epsilon)).float().mean().item())
