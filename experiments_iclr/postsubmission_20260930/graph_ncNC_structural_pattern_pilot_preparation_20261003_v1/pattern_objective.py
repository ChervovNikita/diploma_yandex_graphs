"""GRAN-attributed uniform joint mixture vs equally supervised marginals."""
from math import log
import torch
from torch.nn import functional as F
from pilot_common import require


def losses(t, rows, labels, queries):
    require(t.ndim == 2 and t.shape[0] == 4 and t.shape[1] == len(rows) == len(labels), "Require aligned four-member residual scores")
    require(rows.dtype == torch.long and labels.dtype == t.dtype and bool(((labels == 0) | (labels == 1)).all()), "Observation bits/rows invalid")
    require(queries > 0 and (not len(rows) or (int(rows.min()) >= 0 and int(rows.max()) < queries)), "Query grouping invalid")
    require(bool(torch.isfinite(t).all()), "Nonfinite likelihood logits")
    # Use log-sigmoid of t: exact algebraic normalized native clamp, without
    # clipping native weights or introducing epsilon boundary semantics.
    a = labels[None, :] * F.logsigmoid(t) + (1 - labels[None, :]) * F.logsigmoid(-t)
    counts = torch.bincount(rows, minlength=queries)
    denom = counts.clamp(min=1).to(t.dtype)
    summed = t.new_zeros((4, queries)).index_add(1, rows, a)
    joint = -(torch.logsumexp(summed, dim=0) - log(4)) / denom
    factorial = t.new_zeros(queries).index_add(0, rows, -(torch.logsumexp(a, dim=0) - log(4))) / denom
    joint = torch.where(counts > 0, joint, torch.zeros_like(joint))
    factorial = torch.where(counts > 0, factorial, torch.zeros_like(factorial))
    return {"J": joint, "F": factorial, "counts": counts, "rho": torch.softmax(summed, dim=0)}


def diagnostic_sums(t, native_q, rows, labels, queries, values, *, common_rows=None):
    """Detached sufficient totals; no query/mask is treated as independent graph."""
    with torch.no_grad():
        q = torch.sigmoid(t)
        component_entropy = -(q * F.logsigmoid(t) + (1 - q) * F.logsigmoid(-t)).mean(0)
        spread = (q - q.mean(0, keepdim=True)).square().mean(0)
        sums = t.new_zeros((2, queries)).index_add(1, rows, torch.stack((component_entropy, spread)))
        denom = values["counts"].clamp(min=1).to(t.dtype)
        active = values["counts"] > 0
        entropy_query, spread_query = sums / denom
        rho = values["rho"]
        responsibility_entropy = -(rho * torch.log(rho.clamp(min=torch.finfo(rho.dtype).tiny))).sum(0)
        # tiny is only a detached entropy diagnostic boundary convention; it
        # never enters either likelihood or its gradient.
        positive_queries = t.new_zeros(queries).index_add(0, rows, labels)
        strata = {"all": torch.ones(queries, device=t.device, dtype=torch.bool),
                  "has_synthetic_removal": positive_queries > 0,
                  "no_synthetic_removal": (positive_queries == 0) & active}
        if common_rows is not None:
            common_count = torch.bincount(common_rows, minlength=queries)
            strata.update(cn0=common_count == 0, has_common=common_count > 0)
        bit_a = labels[None,:]*F.logsigmoid(t) + (1-labels[None,:])*F.logsigmoid(-t)
        marginal_bit_nll = -(torch.logsumexp(bit_a,dim=0)-log(4))
        mean_q = q.mean(0)
        result = {"queries": queries, "active_queries": int(active.sum()), "residual_slots": len(rows),
                  "synthetic_removed_observed_slots": int(labels.sum()), "source_unobserved_slots": int((1-labels).sum()),
                  "normalized_native_q_vs_stable_sigmoid_max_abs": float((native_q-q).abs().max()) if q.numel() else 0.,
                  "component_entropy_query_sum": float(entropy_query.sum()), "between_component_spread_query_sum": float(spread_query.sum()),
                  "active_responsibility_entropy_sum": float(responsibility_entropy[active].sum()),
                  "active_responsibility_max_sum": float(rho.max(0).values[active].sum())}
        result["bit_marginal"] = {str(bit): {"slots": int((labels == bit).sum()),
                "nll_sum": float(marginal_bit_nll[labels == bit].sum()),
                "brier_sum": float((mean_q[labels == bit]-bit).square().sum()),
                "predicted_observation_probability_sum": float(mean_q[labels == bit].sum())} for bit in (0,1)}
        result["strata"] = {name: {"queries": int(mask.sum()), "joint_query_sum": float(values["J"][mask].sum()),
                                   "factorial_query_sum": float(values["F"][mask].sum()),
                                   "joint_minus_factorial_query_sum": float((values["J"]-values["F"])[mask].sum()),
                                   "between_component_spread_query_sum": float(spread_query[mask].sum())}
                            for name, mask in strata.items()}
        return result
