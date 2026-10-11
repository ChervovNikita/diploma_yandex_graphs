"""Inactive source-level Rank-1 BNN acquisition contract; no consumer or CLI.

Torch is supplied by a future admitted owner. This module is not imported or
numerically executed during preparation. The native model/FactorLinear sampling
scope, registered scales, checkpoint custody and complete reader are not supplied
here and still require source qualification. No existing source is modified.
"""
import math

TRAIN_ROWS = 580
PRIOR_STD = 0.1
INITIAL_LOG_STD = math.log(PRIOR_STD)


def draw_route(torch, means, log_stds, generator):
    """Draw once per complete route; caller reuses these tensors at every site.

    Keys identify actual registered affine r/s vectors of the active route.
    Do not copy draws into parameters or resample within a graph/layer occurrence.
    Original route identity persists across sites; noises are independent by key.
    """
    assert set(means) == set(log_stds) and means
    result = {}
    for key in sorted(means):
        mu, rho = means[key], log_stds[key]
        assert mu.ndim == 1 and mu.shape == rho.shape and mu.device == rho.device and mu.dtype == rho.dtype
        epsilon = torch.randn(mu.shape, dtype=mu.dtype, device=mu.device, generator=generator)
        result[key] = mu + rho.exp() * epsilon
    return result


def conditional_factor_KL(torch, means, log_stds, initial_centres):
    """Exact diagonal-Gaussian conditional-component KL, summed over one route.

    The prior centres are immutable original data-free factor initializations.
    This is KL(q(phi|m)||p(phi|m)), not a marginal Gaussian-mixture KL.
    """
    assert set(means) == set(log_stds) == set(initial_centres) and means
    terms = []
    for key in sorted(means):
        mu, rho, centre = means[key], log_stds[key], initial_centres[key]
        assert mu.ndim == 1 and mu.shape == rho.shape == centre.shape
        assert mu.device == rho.device == centre.device and mu.dtype == rho.dtype == centre.dtype
        terms.append(0.5 * ((rho.mul(2).exp() + (mu-centre).square()) / PRIOR_STD**2
                           - 1 + 2 * (INITIAL_LOG_STD-rho)).sum())
    return torch.stack(terms).sum()


def acquisition_loss(torch, mean_TRAIN_own_CEs, conditional_KLs):
    """One draw per route, one native AdamW update; no pool CE or extra loss.

    TRAIN own CEs already use reduction mean over580 labels. Four-route banks
    average four complete terms. Each independent/single body uses one term.
    """
    assert len(mean_TRAIN_own_CEs) == len(conditional_KLs) and len(conditional_KLs) in (1, 4)
    return torch.stack(mean_TRAIN_own_CEs).mean() + torch.stack(conditional_KLs).mean() / TRAIN_ROWS
