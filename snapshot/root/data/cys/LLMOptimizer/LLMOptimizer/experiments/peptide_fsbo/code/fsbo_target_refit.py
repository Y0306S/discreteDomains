"""Joint deep-kernel target adaptation with deterministic best-loss patience."""

import copy
import time

import gpytorch
import torch


def fit_target(model, x, y, max_updates, patience=16, min_delta=0.0001, lr=0.001, seed=0):
    if max_updates < 1 or patience < 1 or min_delta <= 0:
        raise ValueError('Invalid FSBO target stopping settings')
    model.model.train()
    model.likelihood.train()
    model.feature_extractor.train()
    optimizer = torch.optim.Adam([
        {'params': model.model.parameters()},
        {'params': model.feature_extractor.parameters()}], lr=lr)
    best, anchor, stale, selected = float('inf'), float('inf'), 0, None
    best_step, trace = None, []
    started = time.monotonic()
    def likelihood_loss():
        # Repeat the native probe/preconditioner draws for a fixed set of weights.
        # Restoring the RNG also preserves the caller's acquisition random stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            z = model.feature_extractor(x)
            model.model.set_train_data(z, y, strict=False)
            return -model.mll(model.model(z), y)

    with gpytorch.settings.max_cholesky_size(800), \
            gpytorch.settings.deterministic_probes(False), \
            gpytorch.settings.num_trace_samples(10):
        for updates in range(max_updates + 1):
            optimizer.zero_grad()
            loss = likelihood_loss()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite FSBO target likelihood')
            value = float(loss.detach())
            trace.append(value)
            # Save before stepping: these are the weights whose loss was measured.
            if value < best:
                best, best_step = value, updates
                selected = copy.deepcopy(model.state_dict())
            if value < anchor - min_delta:
                anchor, stale = value, 0
            else:
                stale += 1
            if stale >= patience:
                reason = 'best_loss_patience'
                break
            if updates == max_updates:
                reason = 'maximum_updates'
                break
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all()
                   for p in model.parameters()):
                raise ValueError('Nonfinite FSBO target gradient')
            optimizer.step()
        model.load_state_dict(selected, strict=True)
        # RBFKernel takes a different numerical path for inputs without gradients.
        # Verify the selected loss in the same mode used during optimization.
        restored = float(likelihood_loss().detach())
        if abs(restored - best) > 0.00001:
            raise ValueError('Restored FSBO weights do not reproduce the selected loss: %.9g vs %.9g' % (restored, best))
    model.model.eval()
    model.likelihood.eval()
    model.feature_extractor.eval()
    with torch.no_grad():
        model.model.set_train_data(model.feature_extractor(x), y, strict=False)
    return dict(adaptation_steps=updates, optimizer_updates=updates,
        likelihood_evaluations=len(trace), best_step=best_step, best_loss=best,
        initial_loss=trace[0], last_loss=trace[-1], restored_loss=restored,
        stop_reason=reason, stale_evaluations=stale, patience=patience,
        min_delta=min_delta, maximum_updates=max_updates,
        likelihood='cholesky' if len(y) <= 800 else 'seeded_probe_lanczos',
        likelihood_policy='deterministic_seeded_likelihood', probe_seed=seed,
        trace_samples=10, cholesky_threshold=800, joint_feature_and_gp=True,
        seconds=time.monotonic() - started)
