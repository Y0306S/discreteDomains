"""Group-relative, clipped token policy loss with a differentiable reference KL."""
import torch


def advantages(rewards):
    values = torch.as_tensor(rewards, dtype=torch.float32)
    if values.ndim != 1 or values.numel() != 4 or not torch.isfinite(values).all():
        raise ValueError('GRPO requires exactly four finite selected sibling rewards')
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)


def token_loss(logp, old_logp, ref_logp, advantage, *, beta=.05, epsilon=.2):
    if logp.shape != old_logp.shape or logp.shape != ref_logp.shape or not logp.numel():
        raise ValueError('Policy, rollout and reference token log probabilities must align')
    logp, old, ref = logp.float(), old_logp.detach().float(), ref_logp.detach().float()
    ratio = torch.exp(logp - old)
    pg = -torch.minimum(ratio * advantage, ratio.clamp(1 - epsilon, 1 + epsilon) * advantage)
    delta = ref - logp
    kl = torch.expm1(delta) - delta
    return (pg + beta * kl).mean()


def duplicate_rejection_loss(logp, old_logp):
    """Reduce the repeated answer's average token likelihood relative to its rollout policy.

    Unlike a constant negative reward, this still has a gradient for an all-repeat
    group. Duplicates bypass the GRPO policy term, so normalization can never
    accidentally reward a repeated answer that outscored a poor novel candidate.
    """
    return torch.nn.functional.softplus(logp.float().mean() - old_logp.detach().float().mean())


def response_loss(logp,old,ref,advantage,duplicate,*,beta=.05,epsilon=.2,duplicate_weight=2.):
    loss=token_loss(logp,old,ref,0. if duplicate else advantage,beta=beta,epsilon=epsilon)
    reject=duplicate_rejection_loss(logp,old) if duplicate else logp.new_zeros(())
    return loss+duplicate_weight*reject,reject
