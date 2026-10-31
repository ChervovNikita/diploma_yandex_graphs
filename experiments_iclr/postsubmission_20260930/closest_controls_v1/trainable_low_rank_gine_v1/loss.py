"""Exact inherited graph/member BCE helper; no dataset imports."""
import torch
import torch.nn.functional as F


def graph_member_bce(logits, targets):
    if logits.ndim != 3 or targets.shape != logits.shape[1:]:
        raise ValueError("Expect member logits[M,B,T] and graph targets[B,T]")
    finite = torch.isfinite(targets)
    observed = finite.sum(-1) > 0
    if not bool(observed.any()):
        raise ValueError("Minibatch has no observed graph labels")
    safe = torch.where(finite, targets, torch.zeros_like(targets))
    element = F.binary_cross_entropy_with_logits(logits, safe[None].expand_as(logits), reduction="none")
    graph_loss = (element * finite[None]).sum(-1) / finite.sum(-1).clamp_min(1)[None]
    return graph_loss[:, observed].mean()
