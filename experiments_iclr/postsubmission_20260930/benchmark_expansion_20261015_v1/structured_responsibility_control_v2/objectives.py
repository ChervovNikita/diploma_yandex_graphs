"""Torch-injected standalone native reduction and explicit block criteria.

No frozen link imports, model, optimizer, native sampler, or fit entry point.
Caller supplies four live native losses and authenticated actual route rows.
"""
from .sampler import authenticate, binary_groups, raw_responsibilities


def _family(family):
    if family not in ("NCN", "NCNC", "BUDDY"):
        raise ValueError("Explicit family NCN, NCNC, or BUDDY is required")
    return family


def native_record_losses(torch, logits, groups, family):
    _family(family)
    if logits.ndim not in (1, 2) or not logits.is_floating_point():
        raise ValueError("Floating logits must have shape [query] or [member,query]")
    labels = binary_groups(groups, logits.shape[-1])
    target = torch.tensor(labels, dtype=logits.dtype, device=logits.device).expand_as(logits)
    if family == "BUDDY":
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return torch.where(target.bool(), -torch.nn.functional.logsigmoid(logits),
                       -torch.nn.functional.logsigmoid(-logits))


def reduce_native(torch, records, groups, family, weights=None):
    """NCN/NCNC: mean(pos)+mean(neg); BUDDY: one original joint batch mean.

    Positive fixed weights are divided by their mean within each actual native
    source reduction group in THIS batch. This ratio changes their marginals.
    """
    _family(family)
    if records.ndim not in (1, 2) or not records.is_floating_point():
        raise ValueError("Floating records must have shape [query] or [member,query]")
    labels = binary_groups(groups, records.shape[-1])
    if not labels or not bool(torch.isfinite(records.detach()).all()):
        raise ValueError("Nonempty finite native loss records are required")
    labels = torch.tensor(labels, dtype=torch.long, device=records.device)
    if weights is not None:
        if (weights.shape != records.shape or weights.device != records.device
                or weights.dtype != records.dtype or weights.requires_grad
                or not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all())):
            raise ValueError("Weights must be fixed, finite, positive, and exactly record-aligned")
    masks = (labels == 1, labels == 0) if family != "BUDDY" else (torch.ones_like(labels, dtype=torch.bool),)
    parts = []
    for mask in masks:
        if not bool(mask.any()):
            raise ValueError("The native loss reduction group is empty")
        current = records[..., mask]
        if weights is not None:
            selected = weights[..., mask]
            current = current * (selected / selected.mean(dim=-1, keepdim=True))
        parts.append(current.mean(dim=-1))
    return sum(parts)


def build_objectives(torch, *, batch, groups, family, logits, native_member_losses,
                     route_query_signatures, seed, mode, floor=0.25, role="TRAIN"):
    """F = mean unchanged source scalar losses; private W = mean weighted risks.

    Return two criteria. Assign shared gradients from F and private gradients
    from W at the same old state in any separately reviewed future integration.
    This helper performs no backward pass or optimizer transition.
    """
    _family(family)
    signature = authenticate(batch, groups, role=role)
    if list(route_query_signatures) != [signature] * 4:
        raise ValueError("Four signatures from actual ordered native route captures are required")
    if logits.ndim == 3 and logits.shape[-1] == 1:
        logits = logits[..., 0]
    if tuple(logits.shape) != (4, len(batch.keys)) or len(native_member_losses) != 4:
        raise ValueError("Four aligned live native routes and native scalar losses are required")
    if any(loss.ndim != 0 or loss.device != logits.device or loss.dtype != logits.dtype
           for loss in native_member_losses):
        raise ValueError("Native losses must be live scalars with the logits dtype/device")
    records = native_record_losses(torch, logits, groups, family)
    reconstructed = reduce_native(torch, records, groups, family)
    source = torch.stack(tuple(native_member_losses))
    if not torch.allclose(reconstructed.detach(), source.detach(), rtol=1e-6, atol=1e-7):
        raise ValueError("Per-query records do not recover the supplied native scalar losses")
    raw = raw_responsibilities(batch, seed, mode=mode, floor=floor)
    weights = torch.tensor(raw, dtype=logits.dtype, device=logits.device)
    F = source.mean()
    W = F if mode == "unit" else reduce_native(torch, records, groups, family, weights).mean()
    return {
        "shared_objective": F,
        "private_objective": W,
        "native_own_loss": F,
        "weighted_private_task_loss": W,
        "raw_weights": weights,
        "diagnostics": {
            "schema": "structured-responsibility-v2", "family": family, "mode": mode,
            "role": role, "seed": int(seed), "query_alignment_sha256": signature, "scope": batch.scope,
            "directed": batch.directed, "query_count": len(batch.keys),
            "native_positive_count": sum(int(value) == 1 for value in groups),
            "native_negative_count": sum(int(value) == 0 for value in groups),
            "hash": "BLAKE2b64/link-resp-v2/52bit-midpoint", "positive_floor": floor,
            "normalization": "per member, per original source group in current batch",
            "native_reduction": "joint batch mean" if family == "BUDDY" else "positive mean plus negative mean",
            "raw_marginal_control": "two query Exp halves for nonself; one query Exp for loop",
            "normalized_marginals_matched": False,
            "unbiased_original_risk_claim": False,
            "source_reduction_parity_checked": True,
            "extra_source_forwards": 0, "optimizer_transitions": 0,
        },
    }
