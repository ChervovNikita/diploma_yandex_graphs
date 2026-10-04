"""Exact padded-batch ESP/count law. Query chunks limit workspace, not support.

Same unary+two-sided count potential as the scalar C_mu source. FP64; every
actual slot and count pair participates. No neural context derivative in mu.
"""
import torch


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate(left, right, nl, nr, potential):
    require(left.ndim == right.ndim == 2 and len(left) == len(right), "Padded unary batch shapes differ")
    require(left.dtype == right.dtype == potential.dtype == torch.float64
            and bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all())
            and bool(torch.isfinite(potential).all()), "Finite FP64 inputs required")
    require(nl.dtype == nr.dtype == torch.long and nl.shape == nr.shape == (len(left),), "Int64 side lengths required")
    require(bool(((nl >= 0) & (nl <= left.shape[1])).all())
            and bool(((nr >= 0) & (nr <= right.shape[1])).all()), "Side length outside padded shape")
    require(potential.shape == (len(left), left.shape[1] + 1, right.shape[1] + 1), "Complete padded count grid required")
    require(left.device == right.device == nl.device == nr.device == potential.device, "Input devices differ")


def _coefficients(eta, length, *, prefixes=False):
    """Reachable-cell update vectorized across queries and count states.

    Safe dummy operands prevent logaddexp(-inf,-inf) even on masked branches;
    impossible outputs stay -inf. Their derivatives are never substituted.
    """
    batch, width = eta.shape
    column = torch.arange(width + 1, device=eta.device)[None, :]
    row = torch.cat((eta.new_zeros((batch, 1)), eta.new_full((batch, width), float("-inf"))), 1)
    table = [row] if prefixes else None
    for j in range(1, width + 1):
        active = length[:, None] >= j
        interior = active & (column > 0) & (column < j)
        previous = torch.cat((row.new_zeros((batch, 1)), row[:, :-1]), 1)
        include = eta[:, j - 1, None] + previous
        safe_out = torch.where(interior, row, torch.zeros_like(row))
        safe_in = torch.where(interior, include, torch.zeros_like(include))
        candidate = torch.logaddexp(safe_out, safe_in)
        updated = torch.where(interior, candidate, row)
        boundary = active & (column == j)
        row = torch.where(boundary, include, updated)
        if prefixes:
            table.append(row)
    return torch.stack(table, 1) if prefixes else row


def log_normalizers(left, right, nl, nr, potential):
    validate(left, right, nl, nr, potential)
    a, b = _coefficients(left, nl), _coefficients(right, nr)
    state = a[:, :, None] + b[:, None, :] + potential
    result = torch.logsumexp(state.flatten(1), 1)
    require(bool(torch.isfinite(result).all()), "Nonfinite FP64 partition arithmetic")
    return result


def nll(left, right, nl, nr, potential, zl, zr):
    validate(left, right, nl, nr, potential)
    require(zl.shape == left.shape and zr.shape == right.shape
            and bool(((zl == 0) | (zl == 1)).all()) and bool(((zr == 0) | (zr == 1)).all()), "Binary padded labels required")
    ml = torch.arange(left.shape[1], device=left.device)[None, :] < nl[:, None]
    mr = torch.arange(right.shape[1], device=right.device)[None, :] < nr[:, None]
    require(not bool(zl[~ml].any()) and not bool(zr[~mr].any()), "Padding cannot be an observation slot")
    zl, zr = zl.detach().to(torch.float64), zr.detach().to(torch.float64)
    kl, kr = zl.sum(1).long(), zr.sum(1).long()
    logz = log_normalizers(left, right, nl, nr, potential)
    rows = torch.arange(len(left), device=left.device)
    numerator = (left * zl).sum(1) + (right * zr).sum(1) + potential[rows, kl, kr]
    result = (logz - numerator) / (nl + nr).clamp_min(1)
    result = torch.where(nl + nr > 0, result, result * 0.)
    require(bool(torch.isfinite(result).all()), "Nonfinite FP64 NLL arithmetic")
    return result


def _reverse(eta, length, prefix, probability):
    batch, width = eta.shape
    adjoint = probability
    output = torch.zeros_like(eta)
    good_mass = torch.ones(batch, dtype=torch.bool, device=eta.device)
    for j in range(width, 0, -1):
        active = length >= j
        current = adjoint[:, 1:j]
        if j > 1:
            include = eta[:, j - 1, None] + prefix[:, j - 1, :j - 1]
            exclude = prefix[:, j - 1, 1:j]
            safe_in = torch.where(active[:, None], include, torch.zeros_like(include))
            safe_out = torch.where(active[:, None], exclude, torch.zeros_like(exclude))
            p = torch.sigmoid(safe_in - safe_out)
            included, excluded = current * p, current * (1. - p)
        else:
            included, excluded = current, current
        previous = torch.zeros_like(adjoint)
        previous[:, 0] = adjoint[:, 0]
        previous[:, j - 1] += adjoint[:, j]
        if j > 1:
            previous[:, :j - 1] += included
            previous[:, 1:j] += excluded
        in_mass = adjoint[:, j] + included.sum(1)
        out_mass = adjoint[:, 0] + excluded.sum(1)
        total = in_mass + out_mass
        # Accumulate device-side checks; avoid a host synchronization per slot.
        good_mass = good_mass & (~active | (torch.isfinite(total) & (total > 0)))
        divisor = torch.where(active, total, torch.ones_like(total))
        output[:, j - 1] = torch.where(active, in_mass / divisor, torch.zeros_like(total))
        adjoint = torch.where(active[:, None], previous, adjoint)
    require(bool(good_mass.all()), "Reverse mass failed")
    return output


@torch.no_grad()
def marginals(left, right, nl, nr, potential):
    validate(left, right, nl, nr, potential)
    left, right, potential = left.detach(), right.detach(), potential.detach()
    a, b = _coefficients(left, nl, prefixes=True), _coefficients(right, nr, prefixes=True)
    state = a[:, -1, :, None] + b[:, -1, None, :] + potential
    probability = torch.softmax(state.flatten(1), 1).reshape_as(state)
    pl, pr = probability.sum(2), probability.sum(1)
    mu_l, mu_r = _reverse(left, nl, a, pl), _reverse(right, nr, b, pr)
    expected_l = (pl * torch.arange(left.shape[1] + 1, device=left.device)).sum(1)
    expected_r = (pr * torch.arange(right.shape[1] + 1, device=right.device)).sum(1)
    require(bool(torch.isfinite(mu_l).all()) and bool(torch.isfinite(mu_r).all())
            and bool(((mu_l >= 0) & (mu_l <= 1)).all()) and bool(((mu_r >= 0) & (mu_r <= 1)).all()), "Marginal probability failed")
    require(bool(((mu_l.sum(1) - expected_l).abs() <= 1e-9 * nl.clamp_min(1)).all())
            and bool(((mu_r.sum(1) - expected_r).abs() <= 1e-9 * nr.clamp_min(1)).all()), "Marginal/count identity failed")
    return mu_l, mu_r, torch.logsumexp(state.flatten(1), 1)
