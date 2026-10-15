"""Fixed factual-context donor targets for existing native route preheads.

No native model, inference feature, label loader or optimizer is created here.
The caller authenticates TRAIN identities and builds the declared factual
contexts. This tensor helper is not a qualified native task integration.
"""
from __future__ import annotations
import math

MODES = ("common", "split")
REDUCTIONS = ("class_balanced", "binary_separate", "joint")
TEMPERATURE = 0.2
COEFFICIENT = 0.05
EPSILON = 1e-12


def context_auxiliary(torch, *, preheads, labels, object_ids, context_a,
                      context_b, mode, reduction, relation_ids=None):
    """Return .05A and exposure diagnostics, without altering native risk F.

    preheads are live [4,Q,H]. Contexts are detached factual [Q,D] objects
    already computed from permitted native inputs/roles. Labels are observed
    TRAIN classes/source groups, not inference labels or mined latent truth.
    Every compatible distinct donor in the supplied bounded cohort is used.
    """
    if mode not in MODES or reduction not in REDUCTIONS:
        raise ValueError("Explicit context assignment and task reduction required")
    if preheads.ndim != 3 or preheads.shape[0] != 4 or min(preheads.shape[1:]) < 1:
        raise ValueError("Four aligned native preheads [4,Q,H] required")
    if preheads.dtype not in (torch.float32, torch.float64) or not torch.isfinite(preheads).all():
        raise ValueError("Finite native FP32/FP64 preheads required")
    count = preheads.shape[1]
    if count > 1024:
        raise ValueError("Predeclared bounded auxiliary cohort exceeded")
    for name, value in (("labels", labels), ("object_ids", object_ids)):
        if value.ndim != 1 or len(value) != count or value.dtype != torch.long or value.device != preheads.device:
            raise ValueError(name+" must be aligned int64 on native device")
    if len(torch.unique(object_ids)) != count:
        raise ValueError("Distinct authenticated auxiliary objects required")
    if (labels < 0).any():
        raise ValueError("Only observed classes/groups may enter the auxiliary")
    if reduction in ("binary_separate", "joint") and ((labels != 0) & (labels != 1)).any():
        raise ValueError("Native link source groups are binary")
    if relation_ids is None:
        relation_ids = torch.zeros_like(labels)
    if relation_ids.shape != labels.shape or relation_ids.dtype != torch.long or relation_ids.device != labels.device:
        raise ValueError("Aligned native relation IDs required")
    if (relation_ids < 0).any():
        raise ValueError("Invalid relation ID")
    for name, value in (("context_a", context_a), ("context_b", context_b)):
        if value.ndim != 2 or value.shape[0] != count or value.shape[1] < 1:
            raise ValueError(name+" must be an aligned factual [Q,D] tensor")
        if value.dtype != preheads.dtype or value.device != preheads.device or not torch.isfinite(value).all():
            raise ValueError("Context/native dtype, device or finiteness differs")
    compatible = (labels[:, None] == labels[None, :]) & (relation_ids[:, None] == relation_ids[None, :])
    compatible = compatible & (object_ids[:, None] != object_ids[None, :])
    support = compatible.sum(-1)
    eligible = support > 0
    classes = torch.unique(labels, sorted=True)
    details = dict(mode=mode, reduction=reduction, temperature=TEMPERATURE,
                   coefficient=COEFFICIENT, input_objects=count,
                   eligible_anchors=int(eligible.sum()), omitted_anchors=int((~eligible).sum()),
                   compatible_donors=int(support.sum()),
                   by_class={str(int(c)):dict(objects=int((labels == c).sum()),
                       supported=int((eligible & (labels == c)).sum())) for c in classes},
                   same_object_donors=0, target_gradient="detached",
                   native_task_population_changed=False)
    if not eligible.any():
        zero = preheads.sum()*0
        return dict(loss=zero, A=zero, diagnostics=details)

    F = torch.nn.functional
    z = F.normalize(preheads, dim=-1, eps=EPSILON)
    mask = compatible[eligible]
    targets = []
    with torch.no_grad():
        for name, context in (("a", context_a), ("b", context_b)):
            norm = context.detach().norm(dim=-1)
            details["zero_context_"+name] = int((norm <= EPSILON).sum())
            t = F.normalize(context.detach(), dim=-1, eps=EPSILON)
            scores = (t[eligible] @ t.T) / TEMPERATURE
            targets.append(scores.masked_fill(~mask, -torch.inf).softmax(-1))
    per_member = []
    active_labels = labels[eligible]
    for member in range(4):
        scores = (z[member, eligible] @ z[member].T) / TEMPERATURE
        logp = scores.masked_fill(~mask, -torch.inf).log_softmax(-1)
        # Avoid0*(-inf), retaining zero derivatives for excluded donor entries.
        logp = logp.masked_fill(~mask, 0)
        target = .5*(targets[0]+targets[1]) if mode == "common" else targets[0 if member < 2 else 1]
        rows = -(target*logp).sum(-1)
        if reduction == "class_balanced":
            terms = [rows[active_labels == c].mean() for c in classes if (active_labels == c).any()]
            value = torch.stack(terms).mean()
        elif reduction == "binary_separate":
            # NCN native task convention is a SUM of two group means.
            # An unsupported group has zero auxiliary, with omission recorded.
            terms = [rows[active_labels == c].mean() for c in (0, 1) if (active_labels == c).any()]
            value = torch.stack(terms).sum()
        else:
            value = rows.mean()
        per_member.append(value)
    A = torch.stack(per_member).mean()
    if not torch.isfinite(A):
        raise ValueError("Nonfinite factual context objective")
    details["per_member_A"] = [float(v.detach()) for v in per_member]
    details["A"] = float(A.detach())
    return dict(loss=COEFFICIENT*A, A=A, diagnostics=details)
