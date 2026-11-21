import math
from dataclasses import dataclass
import torch
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from torch.quasirandom import SobolEngine
from .constrained_max_posterior_sampling import MaxPosteriorSampling

@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 32
    success_counter: int = 0
    success_tolerance: int = 10
    best_value: float = -float('inf')
    restart_triggered: bool = False
    best_constraint_values: torch.Tensor = torch.ones(2) * torch.inf

def update_tr_length(state):
    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0
    if state.length < state.length_min:
        state.restart_triggered = True
    return state

def update_state_constrained(state, Y_next, C_next):
    """Method used to update the TuRBO state after each
    step of optimization.

    Success and failure counters are updated accoding to
    the objective values (Y_next) and constraint values (C_next)
    of the batch of candidate points evaluated on the optimization step.

    As in the original TuRBO paper, a success is counted whenver
    any one of the new candidate points imporves upon the incumbent
    best point. The key difference for SCBO is that we only compare points
    by their objective values when both points are valid (meet all constraints).
    If exactly one of the two points beinc compared voliates a constraint, the
    other valid point is automatically considered to be better. If both points
    violate some constraints, we compare them inated by their constraint values.
    The better point in this case is the one with minimum total constraint violation
    (the minimum sum over constraint values)"""
    bool_tensor = C_next <= 0
    bool_tensor = torch.all(bool_tensor, dim=-1)
    Valid_Y_next = Y_next[bool_tensor]
    Valid_C_next = C_next[bool_tensor]
    if Valid_Y_next.numel() == 0:
        sum_violation = C_next.sum(dim=-1)
        min_violation = sum_violation.min()
        if min_violation < state.best_constraint_values.sum():
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = Y_next[sum_violation.argmin()].item()
            state.best_constraint_values = C_next[sum_violation.argmin()]
        else:
            state.success_counter = 0
            state.failure_counter += 1
    else:
        imporved_obj = max(Valid_Y_next) > state.best_value + 0.001 * math.fabs(state.best_value)
        obtained_validity = torch.all(state.best_constraint_values > 0)
        if imporved_obj or obtained_validity:
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = max(Valid_Y_next).item()
            state.best_constraint_values = Valid_C_next[Valid_Y_next.argmax()]
        else:
            state.success_counter = 0
            state.failure_counter += 1
    state = update_tr_length(state)
    return state

def update_state(state, Y_next, C_next):
    if C_next is None:
        return update_state_unconstrained(state, Y_next)
    else:
        return update_state_constrained(state, Y_next, C_next)

def update_state_unconstrained(state, Y_next):
    if max(Y_next) > state.best_value + 0.001 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1
    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0
    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state

def generate_batch(state, model, X, Y, batch_size, n_candidates=None, num_restarts=10, raw_samples=256, acqf='ts', dtype=torch.float32, device=torch.device('cpu'), absolute_bounds=None, constraint_model_list=None):
    assert acqf in ('ts', 'ei')
    if constraint_model_list is not None:
        assert acqf == 'ts'
        constrained = True
    else:
        constrained = False
    assert torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))
    x_center = X[Y.argmax(), :].clone()
    weights = torch.ones_like(x_center)
    if absolute_bounds is None:
        weights = weights * 8
        tr_lb = x_center - weights * state.length / 2.0
        tr_ub = x_center + weights * state.length / 2.0
    else:
        lb, ub = absolute_bounds
        weights = weights * (ub - lb)
        tr_lb = torch.clamp(x_center - weights * state.length / 2.0, lb, ub)
        tr_ub = torch.clamp(x_center + weights * state.length / 2.0, lb, ub)
    if acqf == 'ei':
        try:
            ei = qExpectedImprovement(model.cpu(), Y.max().cpu())
            X_next, _ = optimize_acqf(ei, bounds=torch.stack([tr_lb, tr_ub]).cpu(), q=batch_size, num_restarts=num_restarts, raw_samples=raw_samples)
        except:
            acqf = 'ts'
    if acqf == 'ts':
        dim = X.shape[-1]
        tr_lb = tr_lb.cpu()
        tr_ub = tr_ub.cpu()
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=dtype).cpu()
        pert = tr_lb + (tr_ub - tr_lb) * pert
        tr_lb = tr_lb.cpu()
        tr_ub = tr_ub.cpu()
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1
        mask = mask.cpu()
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand = X_cand.cpu()
        X_cand[mask] = pert[mask]
        thompson_sampling = MaxPosteriorSampling(model=model, constraint_models=constraint_model_list, replacement=False, constrained=constrained)
        with torch.no_grad():
            X_next = thompson_sampling(X_cand.cpu(), num_samples=batch_size)
    return X_next
