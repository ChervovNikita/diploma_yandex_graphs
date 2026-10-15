"""Explicit multiclass adaptation of the pinned molecular C1 objective.

Same binary formula; observed categorical classes replace hardcoded(0,1).
No route-repulsion, model, forward, projector, augmentation or optimizer.
"""
from dataclasses import asdict, dataclass
import math
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    mode: str
    temperature: float = .2
    coefficient: float = .05
    std_fraction: float = .5
    variance_epsilon: float = 1e-8
    normalization_epsilon: float = 1e-12

    def __post_init__(self):
        if self.mode not in ("alignment", "alignment_variance") or self.temperature != .2 \
                or self.coefficient != .05 or self.std_fraction != .5 \
                or self.variance_epsilon != 1e-8 or self.normalization_epsilon != 1e-12:
            raise ValueError("Two explicitly named fixed C1 modes; no adjustable numerical settings")


def _inputs(prehead, labels, ids):
    if prehead.ndim != 3 or min(prehead.shape) < 1 or prehead.dtype not in (torch.float32, torch.float64) \
            or not torch.isfinite(prehead).all():
        raise ValueError("Finite live native [M,N,H] FP32 prehead, or FP64 semantic tensor, required")
    n = prehead.shape[1]
    if labels.shape != (n,) or ids.shape != (n,) or ids.dtype != torch.long or len(torch.unique(ids)) != n:
        raise ValueError("Exact unique native node IDs and aligned single-task categorical labels required")
    labels, ids = labels.to(prehead.device), ids.to(prehead.device)
    if labels.dtype not in (torch.long, torch.float32, torch.float64) or torch.isinf(labels).any():
        raise ValueError("Observed nonnegative integer classes; only NaN means missing in helper semantics")
    observed = ~torch.isnan(labels) if labels.is_floating_point() else torch.ones_like(labels, dtype=torch.bool)
    y = labels[observed]
    if ((y < 0) | (y != y.floor())).any():
        raise ValueError("Observed categorical targets must be nonnegative integers")
    return prehead[:, observed], y.to(torch.long), ids[observed], observed


def _alignment(z, labels, ids, temperature):
    members, count, width = z.shape
    classes = torch.unique(labels).tolist()
    counts = {str(c): int((labels == c).sum()) for c in classes}
    details = {"class_node_counts": counts, "at_least_two_observed_classes": len(classes) >= 2,
        "eligible_anchors": 0, "eligible_anchor_classes": 0, "same_node_positive_fraction": None,
        "different_node_positive_anchor_fraction": 0., "singleton_classes": [c for c in classes if counts[str(c)] == 1]}
    if count < 2 or len(classes) < 2:
        return z.sum()*0, details
    flat, y, ii = z.reshape(-1, width), labels.repeat(members), ids.repeat(members)
    scores = flat @ flat.T / temperature
    nonself = ~torch.eye(len(flat), dtype=torch.bool, device=z.device)
    positive = nonself & y[:, None].eq(y[None, :])
    positive_count = positive.sum(-1); valid = positive_count > 0
    if not valid.any(): return z.sum()*0, details
    scores, nonself, positive = scores[valid], nonself[valid], positive[valid]
    positive_count = positive_count[valid]; anchor_y = y[valid]
    denominator_classes = []
    for cls in classes:
        candidates = nonself & y[None, :].eq(cls)
        n = candidates.sum(-1)
        denominator_classes.append(torch.logsumexp(scores.masked_fill(~candidates, -torch.inf), -1)
            - n.clamp_min(1).to(scores).log())
    denominator = torch.logsumexp(torch.stack(denominator_classes), dim=0)
    per_anchor = denominator - scores.masked_fill(~positive, 0).sum(-1)/positive_count
    class_losses = [per_anchor[anchor_y == c].mean() for c in classes if (anchor_y == c).any()]
    same_node = positive & ii[valid, None].eq(ii[None, :])
    different_node = positive & ~ii[valid, None].eq(ii[None, :])
    details.update(eligible_anchors=int(valid.sum()), eligible_anchor_classes=len(class_losses),
        same_node_positive_fraction=float((same_node.sum(-1)/positive_count.to(scores)).detach().mean()),
        different_node_positive_anchor_fraction=float(different_node.any(-1).float().mean()),
        positive_count_min=int(positive_count.min()), positive_count_max=int(positive_count.max()))
    return torch.stack(class_losses).mean(), details


def _variance(z, labels, config):
    gamma = config.std_fraction/math.sqrt(z.shape[-1])
    if config.variance_epsilon >= gamma*gamma:
        raise ValueError("Variance epsilon masks the declared normalized-feature floor")
    penalties, details = [], {}
    for cls in torch.unique(labels).tolist():
        mask = labels == cls; count = int(mask.sum())
        details[str(cls)] = {"distinct_nodes": count, "supported": count >= 2}
        if count >= 2:
            std = (z[:, mask].var(dim=1, unbiased=False)+config.variance_epsilon).sqrt()
            penalties.append(F.relu(gamma-std).mean()/gamma)
            details[str(cls)].update(mean_std=float(std.detach().mean()),
                mean_std_over_target=float(std.detach().mean()/gamma),
                fraction_dimensions_below_target=float((std.detach() < gamma).float().mean()),
                per_member_mean_std=std.detach().mean(-1).cpu().tolist())
    return (torch.stack(penalties).mean() if penalties else z.sum()*0,
        {"target_std": gamma, "eligible_classes": len(penalties), "by_class": details})


class NodeGeometry:
    def __init__(self, mode): self.config = Config(mode=mode)

    def state_dict(self):
        return {"schema": "task-node-class-geometry-v1", "config": asdict(self.config), "learned_or_queue_state": None}

    def load_state_dict(self, value):
        if value != self.state_dict(): raise ValueError("Node class geometry objective configuration changed")

    def __call__(self, *, prehead, labels, node_ids, role):
        if role != "TRAIN": raise ValueError("Only authenticated TRAIN preheads enter this objective")
        h, y, ids, observed = _inputs(prehead, labels, node_ids)
        z = F.normalize(h, dim=-1, eps=self.config.normalization_epsilon)
        alignment, alignment_info = _alignment(z, y, ids, self.config.temperature)
        variance, variance_info = _variance(z, y, self.config)
        weighted = self.config.coefficient*(alignment + (variance if self.config.mode == "alignment_variance" else 0))
        if not torch.isfinite(weighted): raise FloatingPointError("Nonfinite node geometry auxiliary")
        return {"loss": weighted, "alignment": alignment, "variance": variance,
            "diagnostics": {"role": role, "mode": self.config.mode, "members": prehead.shape[0],
                "selected_nodes": prehead.shape[1], "observed_nodes": int(observed.sum()),
                "missing_labels": int((~observed).sum()), "prehead_width": prehead.shape[2],
                "alignment": alignment_info, "conditional_variance": variance_info,
                "coefficient": self.config.coefficient, "temperature": self.config.temperature,
                "variance_in_update": self.config.mode == "alignment_variance", "new_native_forward_calls": 0,
                "route_repulsion": False, "TEST_accessed": False}}
