"""Full-count FP64 log ESP, recomputed auxiliary backward and exact draws.

SOURCE ONLY: this module has not been imported or numerically qualified.
Forward/backward retain neural slot tensors, never all queries' DP tables.
One query at a time uses a full (R+1)^2 prefix workspace. No support cap,
count truncation, exp(t), probability epsilon or dtype fallback is present.
"""
from dataclasses import dataclass
import torch

ROUNDING_LOG_INCLUDE = 1e-10


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def offsets_as_list(offsets, slot_count):
    require(offsets.dtype == torch.long and offsets.ndim == 1 and len(offsets) >= 1, "Invalid slot offsets")
    values = offsets.detach().cpu().tolist()  # paid host transfer/synchronization
    require(values[0] == 0 and values[-1] == slot_count and all(a <= b for a, b in zip(values, values[1:])), "Invalid contiguous support partition")
    return values


def log_esp_prefix(u):
    """Independent FP64 inputs; reachable cells only; no autograd retention."""
    require(u.ndim == 1 and u.dtype == torch.float64 and bool(torch.isfinite(u).all()), "Finite FP64 logits required")
    r = len(u)
    table = u.new_full((r + 1, r + 1), float("-inf"))
    table[0, 0] = 0.
    for j in range(1, r + 1):
        table[j, 0] = 0.
        table[j, j] = u[j - 1] + table[j - 1, j - 1]
        if j > 1:
            table[j, 1:j] = torch.logaddexp(table[j - 1, 1:j], u[j - 1] + table[j - 1, :j - 1])
    require(bool(torch.isfinite(table[r, :r + 1]).all()), "Nonfinite full-count DP")
    return table


def reverse_esp(u, table, final_adjoint, *, probability_readout=False):
    """Derivative wrt independent u, including K0/KR boundaries exactly.

    Seed with one-hot K to get conditional marginals, or detached pi to get
    actual mixture marginals. Branch probabilities use stable two-term
    softmax derivatives, never exp(t) or a probability normalizer. The
    probability readout divides included by included+excluded reverse mass;
    algebraically this is the same derivative for a unit-mass pi seed. It
    maintains probability boundaries without an epsilon or clipping rule.
    """
    r = len(u)
    require(final_adjoint.shape == (r + 1,) and final_adjoint.dtype == torch.float64, "Invalid full-count reverse seed")
    adjoint = final_adjoint.clone()
    gradient = torch.zeros_like(u)
    for j in range(r, 0, -1):
        previous = u.new_zeros(j)
        previous[0] = adjoint[0]
        previous[j - 1] += adjoint[j]
        included_mass, excluded_mass = adjoint[j], adjoint[0]
        if j > 1:
            current = adjoint[1:j]
            include_branch = u[j - 1] + table[j - 1, :j - 1]
            exclude_branch = table[j - 1, 1:j]
            # Both terms are reachable, so there is no -inf-minus--inf.
            # A two-branch softmax from their difference is the stable
            # logaddexp derivative. Complementary weights conserve mass.
            include_probability = torch.sigmoid(include_branch - exclude_branch)
            included = current * include_probability
            excluded = current * (1. - include_probability)
            previous[:j - 1] += included
            previous[1:j] += excluded
            included_mass = included_mass + included.sum()
            excluded_mass = excluded_mass + excluded.sum()
        if probability_readout:
            total_mass = included_mass + excluded_mass
            require(bool(torch.isfinite(total_mass)) and float(total_mass) > 0., "Missing reverse probability mass")
            gradient[j - 1] = included_mass / total_mass
        else:
            gradient[j - 1] = included_mass
        adjoint = previous
    require(bool(torch.isfinite(gradient).all()), "Nonfinite DP reverse derivative")
    return gradient


class LogESPAtObservedCount(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, offsets, observed_count):
        require(u.dtype == torch.float64 and u.ndim == 1, "FP64 centered slot logits required")
        boundaries = offsets_as_list(offsets, len(u))
        count = observed_count.detach().cpu().tolist()
        require(observed_count.dtype == torch.long and len(count) == len(boundaries) - 1, "Observed-count partition differs")
        result = u.new_zeros(len(count))
        for query, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            r, k = end - start, count[query]
            require(0 <= k <= r, "Observed count outside full support")
            table = log_esp_prefix(u[start:end])
            result[query] = table[r, k]
            del table  # last table must not overlap the next workspace
        ctx.save_for_backward(u, offsets, observed_count)
        return result

    @staticmethod
    def backward(ctx, output_gradient):
        u, offsets, observed_count = ctx.saved_tensors
        boundaries = offsets_as_list(offsets, len(u))
        count = observed_count.detach().cpu().tolist()
        gradient = torch.zeros_like(u)
        for query, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            r, k = end - start, count[query]
            table = log_esp_prefix(u[start:end])  # paid recomputation
            seed = u.new_zeros(r + 1)
            seed[k] = 1.
            gradient[start:end] = output_gradient[query] * reverse_esp(u[start:end], table, seed)
            del table
        # u is centered OUTSIDE this function. Autograd therefore applies the
        # mean-subtraction chain; full numerator and normalizer shifts cancel.
        return gradient, None, None


def center_by_query(t, rows, offsets):
    require(t.ndim == rows.ndim == 1 and t.shape == rows.shape and rows.dtype == torch.long, "Invalid ordered slot logits")
    q = len(offsets) - 1
    t64 = t.to(torch.float64)
    count = (offsets[1:] - offsets[:-1]).to(torch.float64)
    mean = t64.new_zeros(q).index_add(0, rows, t64) / count.clamp_min(1)
    return t64 - mean[rows]


def segmented_log_softmax(logits, rows, queries):
    require(logits.ndim == rows.ndim == 1 and logits.shape == rows.shape and logits.dtype == torch.float64, "FP64 full count logits required")
    require(bool(torch.isfinite(logits).all()), "Nonfinite count logits")
    # Detached maximum is a numerical shift only. Logsumexp derivatives are
    # unchanged, including at ties; the head's q max uses torch.amax below.
    maximum = logits.detach().new_full((queries,), float("-inf"))
    maximum.scatter_reduce_(0, rows, logits.detach(), reduce="amax", include_self=True)
    shifted = logits - maximum[rows]
    denominator = logits.new_zeros(queries).index_add(0, rows, shifted.exp()).log()
    result = shifted - denominator[rows]
    require(bool(torch.isfinite(result).all()), "Nonfinite full count log-softmax")
    return result


@dataclass(frozen=True)
class PredictedLaw:
    t: torch.Tensor
    centered_t: torch.Tensor
    affinity: torch.Tensor
    log_pi: torch.Tensor
    count_logits: torch.Tensor
    rows: torch.Tensor
    offsets: torch.Tensor
    class_offsets: torch.Tensor


def pattern_nll(law, labels):
    """Label-only loss after law and target forwards; all-query R0 zeros."""
    require(labels.shape == law.t.shape and labels.ndim == 1, "Teacher cannot alter support")
    require(bool(((labels == 0) | (labels == 1)).all()), "Observation teacher must supply binary labels")
    q = len(law.offsets) - 1
    z = labels.to(dtype=torch.float64, device=law.t.device).detach()
    count = torch.zeros(q, dtype=torch.long, device=z.device).index_add(0, law.rows, z.long())
    normalizer = LogESPAtObservedCount.apply(law.centered_t, law.offsets, count)
    numerator = z.new_zeros(q).index_add(0, law.rows, law.centered_t * z)
    logpi_k = law.log_pi[law.class_offsets[:-1] + count]
    r = law.offsets[1:] - law.offsets[:-1]
    nll = (normalizer - numerator - logpi_k) / r.clamp_min(1).to(torch.float64)
    # Empty pi0 is constant0. Explicit zeros also block empty-head learning.
    nll = torch.where(r > 0, nll, torch.zeros_like(nll))
    require(bool(torch.isfinite(nll).all()), "Nonfinite full-pattern NLL")
    return nll, count


def sample_count(log_pi, log_uniform):
    require(log_pi.ndim == 1 and len(log_pi) > 0 and bool(torch.isfinite(log_pi).all()), "Finite complete count law required")
    terminal = torch.logsumexp(log_pi, dim=0)
    require(abs(float(terminal)) <= 1e-10, "Count law is not normalized")
    # Subtract the same terminal mass: an exact log-CDF normalization, no
    # epsilon or class truncation. The terminal class closes mass at log1=0.
    cdf = torch.logcumsumexp(log_pi, dim=0) - terminal
    cdf[-1] = 0.
    require(bool((cdf <= 0).all()) and bool((cdf[1:] >= cdf[:-1]).all()), "Inconsistent count log-CDF")
    require(log_uniform < 0., "Counter uniform must remain open")
    selected = torch.nonzero(log_uniform < cdf, as_tuple=False).flatten()
    require(len(selected) > 0, "Open uniform escaped complete count mass")
    return int(selected[0])


def sample_subset(u, table, count, slot_log_uniforms):
    r = len(u)
    require(0 <= count <= r and len(slot_log_uniforms) == r, "Conditional sample support differs")
    bits = torch.zeros(r, dtype=torch.bool, device=u.device)
    remaining = count
    for j in range(r, 0, -1):
        if remaining == 0:
            break
        if remaining == j:
            bits[:j] = True
            remaining = 0
            break
        log_include = float(u[j - 1] + table[j - 1, remaining - 1] - table[j, remaining])
        require(log_include <= ROUNDING_LOG_INCLUDE, "Positive log-inclusion exceeds prebound FP64 roundoff")
        if log_include > 0.:
            log_include = 0.
        require(slot_log_uniforms[j - 1] < 0., "Counter uniform must remain open")
        if slot_log_uniforms[j - 1] < log_include:
            bits[j - 1] = True
            remaining -= 1
    require(remaining == 0 and int(bits.sum()) == count, "Conditional sample violated exact count")
    return bits


@torch.no_grad()
def actual_marginals(law):
    """Paid diagnostic readout without samples; derivative wrt independent u."""
    boundaries = offsets_as_list(law.offsets, len(law.t))
    classes = offsets_as_list(law.class_offsets, len(law.log_pi))
    mu = law.t.new_empty(len(law.t), dtype=torch.float64)
    expected = law.t.new_zeros(len(boundaries) - 1, dtype=torch.float64)
    for query, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        r = end - start
        u = law.centered_t[start:end].detach()
        pi = law.log_pi[classes[query]:classes[query + 1]].detach().exp()
        require(len(pi) == r + 1, "Marginal readout lost feasible count classes")
        table = log_esp_prefix(u)
        current = reverse_esp(u, table, pi, probability_readout=True)
        expected[query] = (pi * torch.arange(r + 1, dtype=torch.float64, device=u.device)).sum()
        require(bool(((current >= 0.) & (current <= 1.)).all()), "Actual marginal outside [0,1]")
        require(abs(float(current.sum() - expected[query])) <= 1e-9 * max(r, 1), "Marginal/count identity failed")
        mu[start:end] = current
        del table
    return mu, expected


@torch.no_grad()
def completion_draws(law, queries, counterpart_pairs, keys, *, route):
    """Detached D4 or M4; no teacher, target-label or model-state input.

    Returns four full-support bit vectors, marginal/count readouts and work
    counts. M4 reverse seeds pi wrt independent u, so mu is NOT sigmoid(t)
    and does not lose the E[K]/R centering correction.
    """
    require(route in ("D4", "M4"), "Unknown frozen serving route")
    boundaries = offsets_as_list(law.offsets, len(law.t))
    classes = offsets_as_list(law.class_offsets, len(law.log_pi))
    query_host = queries.detach().cpu().tolist()
    counterpart_host = counterpart_pairs.detach().cpu().tolist()
    require(len(query_host) == len(boundaries) - 1 and len(counterpart_host) == len(law.t), "Key coordinate/support partition differs")
    bits = torch.zeros((4, len(law.t)), dtype=torch.bool, device=law.t.device)
    sampled_counts = torch.zeros((len(query_host), 4), dtype=torch.long, device=law.t.device)
    expected_counts = law.t.new_zeros(len(query_host), dtype=torch.float64)
    actual_mu = law.t.new_empty(len(law.t), dtype=torch.float64) if route == "M4" else None
    cells = 0
    for query, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        r = end - start
        u = law.centered_t[start:end].detach()
        logpi = law.log_pi[classes[query]:classes[query + 1]].detach()
        require(len(logpi) == r + 1, "Count classes truncated")
        table = log_esp_prefix(u)
        pi = logpi.exp()
        expected_counts[query] = (pi * torch.arange(r + 1, device=u.device, dtype=torch.float64)).sum()
        cells += (r + 1) * (r + 2) // 2
        if route == "M4":
            mu = reverse_esp(u, table, pi, probability_readout=True)  # independent u inputs
            require(bool(((mu >= 0.) & (mu <= 1.)).all()), "Actual marginal outside [0,1]")
            require(abs(float(mu.sum() - expected_counts[query])) <= 1e-9 * max(r, 1), "Marginal/count identity failed")
            actual_mu[start:end] = mu
            marginal_host = mu.cpu().tolist()
        for draw in range(4):
            slot_uniforms = [keys.log_uniform(query_host[query], query, draw, "slot", counterpart_host[slot]) for slot in range(start, end)]
            if route == "D4":
                count_uniform = keys.log_uniform(query_host[query], query, draw, "count")
                k = sample_count(logpi, count_uniform)
                current = sample_subset(u, table, k, slot_uniforms)
            else:
                # Each bit uses ONLY its actual marginal and same slot key.
                # No sampled count or other-bit input enters this product law.
                import math
                values = [False if mu_i == 0. else True if mu_i == 1. else log_u < math.log(mu_i)
                          for mu_i, log_u in zip(marginal_host, slot_uniforms)]
                current = torch.tensor(values, dtype=torch.bool, device=u.device)
                k = int(current.sum())
            bits[draw, start:end] = current
            sampled_counts[query, draw] = k
        del table
    work = {"queries": len(query_host), "residual_slots": len(law.t), "all_count_classes": len(law.log_pi),
            "sampled_draws": 4, "DP_forward_queries": len(query_host), "DP_forward_reachable_cells": cells,
            "DP_marginal_reverse_queries": len(query_host) if route == "M4" else 0,
            "retained_all_query_DP_tables": False, "query_slot_coordinate_host_transfers": 2}
    return {"bits": bits.detach(), "sampled_counts": sampled_counts, "expected_count": expected_counts,
            "actual_marginal": actual_mu, "work": work}
