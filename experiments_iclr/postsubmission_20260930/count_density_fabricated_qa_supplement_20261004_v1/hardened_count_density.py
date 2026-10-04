"""Exact two-sided unary/count-potential law; prepared, never executed here.

Unary plus cardinality potentials and ESP inference follow Tarlow et al.
(arXiv:1210.4899); this is an attributed implementation, not a novelty claim.
All feasible count pairs and native slots participate. FP64 arithmetic.
"""
import torch


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate_unaries(eta):
    require(eta.ndim == 1 and eta.dtype == torch.float64
            and bool(torch.isfinite(eta).all()), "Finite FP64 unary logits required")


def validate_law(left, right, potential):
    validate_unaries(left)
    validate_unaries(right)
    require(left.device == right.device == potential.device, "Density devices differ")
    require(potential.shape == (len(left) + 1, len(right) + 1)
            and potential.dtype == torch.float64 and bool(torch.isfinite(potential).all()),
            "Finite complete FP64 potential required")


def log_coefficients(eta):
    """Autograd-safe log ESP: update reachable cells only."""
    require(eta.ndim == 1 and eta.dtype == torch.float64
            and bool(torch.isfinite(eta).all()), "Finite FP64 unary logits required")
    row = eta.new_zeros(1)
    for value in eta.unbind():
        row = torch.cat((row[:1], torch.logaddexp(row[1:], value + row[:-1]),
                         value + row[-1:]))
    return row


def log_prefix(eta):
    """Detached quadratic workspace, valid prefix coefficients only."""
    validate_unaries(eta)
    n = len(eta)
    table = eta.new_full((n + 1, n + 1), float("-inf"))
    table[0, 0] = 0.
    for j in range(1, n + 1):
        table[j, 0] = 0.
        table[j, j] = eta[j - 1] + table[j - 1, j - 1]
        if j > 1:
            table[j, 1:j] = torch.logaddexp(table[j - 1, 1:j],
                                          eta[j - 1] + table[j - 1, :j - 1])
    return table


def reverse_inclusion(eta, table, count_probability):
    """ESP reverse with a count-law seed: exact unary inclusion probabilities.

    Inputs are independent unary logits. Context and g are already detached;
    differentiating a neural g(eta) or centered eta would give the wrong mu.
    The per-slot mass ratio is algebraically the same unit-mass reverse.
    It imposes no clipping, epsilon, truncation or count approximation.
    """
    n = len(eta)
    require(count_probability.shape == (n + 1,), "All count states required")
    adjoint = count_probability.clone()
    result = torch.zeros_like(eta)
    for j in range(n, 0, -1):
        previous = eta.new_zeros(j)
        previous[0] = adjoint[0]
        previous[j - 1] += adjoint[j]
        included, excluded = adjoint[j], adjoint[0]
        if j > 1:
            include = eta[j - 1] + table[j - 1, :j - 1]
            exclude = table[j - 1, 1:j]
            probability = torch.sigmoid(include - exclude)
            mass_in = adjoint[1:j] * probability
            mass_out = adjoint[1:j] * (1. - probability)
            previous[:j - 1] += mass_in
            previous[1:j] += mass_out
            included, excluded = included + mass_in.sum(), excluded + mass_out.sum()
        total = included + excluded
        require(bool(torch.isfinite(total)) and float(total) > 0., "Missing reverse mass")
        result[j - 1] = included / total
        adjoint = previous
    require(bool(torch.isfinite(result).all()) and bool(((result >= 0) & (result <= 1)).all()),
            "Unary marginal outside probability support")
    return result


def log_normalizer(left, right, potential):
    require(potential.shape == (len(left) + 1, len(right) + 1)
            and potential.dtype == torch.float64 and bool(torch.isfinite(potential).all()),
            "Finite complete two-dimensional count potential required")
    states = log_coefficients(left)[:, None] + log_coefficients(right)[None, :] + potential
    return torch.logsumexp(states.flatten(), 0)


def query_nll(left, right, potential, labels):
    validate_law(left, right, potential)
    require(labels.shape == (len(left) + len(right),)
            and bool(((labels == 0) | (labels == 1)).all()), "Aligned observation bits required")
    size = len(labels)
    if size == 0:
        # g(0,0) cancels exactly. Empty queries retain their all-query zero.
        return (left.sum() + right.sum() + potential.sum()) * 0.
    z = labels.detach().to(dtype=torch.float64, device=left.device)
    kl, kr = int(z[:len(left)].sum()), int(z[len(left):].sum())
    logz = log_normalizer(left, right, potential)
    numerator = (left * z[:len(left)]).sum() + (right * z[len(left):]).sum()
    return (logz - numerator - potential[kl, kr]) / size


@torch.no_grad()
def unary_marginals(left, right, potential):
    """E[Z_i|fixed C], not a total derivative through the context network."""
    validate_law(left, right, potential)
    left, right, potential = left.detach(), right.detach(), potential.detach()
    require(potential.shape == (len(left) + 1, len(right) + 1)
            and potential.dtype == torch.float64 and bool(torch.isfinite(potential).all()),
            "Full finite fixed potential required")
    lt, rt = log_prefix(left), log_prefix(right)
    state = lt[len(left), :len(left) + 1, None] + rt[len(right), None, :len(right) + 1] + potential
    logz = torch.logsumexp(state.flatten(), 0)
    probability = torch.softmax(state.flatten(), 0).reshape_as(state)
    lm = reverse_inclusion(left, lt, probability.sum(1))
    rm = reverse_inclusion(right, rt, probability.sum(0))
    expected_l = (probability.sum(1) * torch.arange(len(left) + 1, device=left.device)).sum()
    expected_r = (probability.sum(0) * torch.arange(len(right) + 1, device=right.device)).sum()
    require(abs(float(lm.sum() - expected_l)) <= 1e-9 * max(len(left), 1)
            and abs(float(rm.sum() - expected_r)) <= 1e-9 * max(len(right), 1),
            "Exact marginal/count identity failed")
    return torch.cat((lm, rm)), {"log_normalizer": logz, "count_probability": probability,
                                "expected_left": expected_l, "expected_right": expected_r}
