"""Graph-level supervised alignment using existing native molecular preheads.

No forward, native module, optimizer, sampler, augmentation, projector or memory
bank is constructed here. This is an attributed experimental composition, not a
reproduction of SupCon, BCL, VICReg, GraphCL or MolCLR.
"""
from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch.nn import functional as F


SCHEMA = "molecular-graph-alignment-v1"


@dataclass(frozen=True)
class ObjectiveConfig:
    mode: str = "alignment_variance"
    temperature: float = 0.2
    coefficient: float = 0.05
    std_fraction: float = 0.5
    variance_epsilon: float = 1e-8
    normalization_epsilon: float = 1e-12
    gradient_scope: str = "all"

    def __post_init__(self):
        if self.mode not in ("none", "alignment", "alignment_variance"):
            raise ValueError("Unknown molecular objective mode")
        if self.gradient_scope not in ("all", "private"):
            raise ValueError("Gradient scope must be explicit: all or private")
        for name in ("temperature", "std_fraction", "variance_epsilon", "normalization_epsilon"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        if self.std_fraction > 1:
            raise ValueError("The normalized-feature std fraction must not exceed one")
        if not math.isfinite(self.coefficient) or self.coefficient < 0:
            raise ValueError("A finite nonnegative auxiliary coefficient is required")


@dataclass
class AuxiliaryResult:
    loss: Any
    terms: dict
    diagnostics: dict
    gradient_scope: str

    def scalar_total(self, native_own_loss):
        """The ordinary all-parameter case needs one native optimizer step."""
        if self.gradient_scope != "all":
            raise ValueError("Private auxiliary credit requires explicit block-gradient routing")
        return native_own_loss + self.loss


def _validate_labels(labels, batch, device):
    if labels.ndim not in (1, 2) or (labels.ndim == 2 and labels.shape[1] != 1):
        raise ValueError("Native single-task binary labels [B] or [B,1] required")
    if labels.numel() != batch:
        raise ValueError("Labels and graph-prehead rows differ")
    labels = labels.reshape(-1).to(device)
    if torch.isinf(labels).any():
        raise ValueError("NaN denotes missing labels; infinity is invalid")
    observed = ~torch.isnan(labels)
    if not ((labels[observed] == 0) | (labels[observed] == 1)).all():
        raise ValueError("Observed labels must be native binary 0/1")
    return labels, observed


def _validate_inputs(prehead, labels, graph_ids):
    if prehead.ndim != 3 or min(prehead.shape) < 1:
        raise ValueError("Live native graph-prehead tensors must be [M,B,D]")
    if prehead.dtype not in (torch.float32, torch.float64):
        raise ValueError("Native FP32 is required; FP64 is accepted for semantic checks")
    if not torch.isfinite(prehead).all():
        raise ValueError("Nonfinite native prehead")
    members, batch, width = prehead.shape
    y, observed = _validate_labels(labels, batch, prehead.device)
    if graph_ids.ndim != 1 or len(graph_ids) != batch or graph_ids.dtype != torch.long:
        raise ValueError("Aligned native graph IDs must be int64 [B]")
    ids = graph_ids.to(prehead.device)
    if len(torch.unique(ids)) != batch:
        raise ValueError("Native no-replacement minibatches require unique graph IDs")
    return members, batch, width, y, observed, ids


def _conditional_variance(z, labels, config):
    """Variance is across distinct graphs, never across member identities."""
    gamma = config.std_fraction / math.sqrt(z.shape[-1])
    if config.variance_epsilon >= gamma * gamma:
        raise ValueError("Variance epsilon would mask the declared variance floor")
    penalties, details = [], {}
    for cls in (0, 1):
        mask = labels == cls
        count = int(mask.sum())
        detail = {"distinct_graphs": count, "supported": count >= 2}
        if count >= 2:
            # Mean and residuals stay live: this is not a detached-centroid trick.
            std = (z[:, mask].var(dim=1, unbiased=False) + config.variance_epsilon).sqrt()
            penalties.append(F.relu(gamma - std).mean() / gamma)
            detail.update({"mean_std": float(std.detach().mean()),
                           "mean_std_over_target": float(std.detach().mean() / gamma),
                           "fraction_dimensions_below_target": float((std.detach() < gamma).float().mean()),
                           "per_member_mean_std": std.detach().mean(-1).cpu().tolist()})
        details[str(cls)] = detail
    loss = torch.stack(penalties).mean() if penalties else z.sum() * 0
    return loss, {"target_std": gamma, "eligible_classes": len(penalties), "by_class": details}


def _balanced_alignment(z, labels, graph_ids, temperature):
    members, graphs, width = z.shape
    counts = {str(c): int((labels == c).sum()) for c in (0, 1)}
    details = {"class_graph_counts": counts, "two_class_gate": all(counts.values()),
               "eligible_anchors": 0, "eligible_anchor_classes": 0,
               "same_graph_positive_fraction": None,
               "different_graph_positive_anchor_fraction": 0.0}
    # A one-class batch has no observed task-negative. Do not replace it with
    # instance negatives or a fabricated class prototype.
    if graphs < 2 or not all(counts.values()):
        return z.sum() * 0, details
    flat, y, ids = z.reshape(-1, width), labels.repeat(members), graph_ids.repeat(members)
    scores = flat @ flat.T / temperature
    nonself = ~torch.eye(len(flat), dtype=torch.bool, device=z.device)
    positives = nonself & (y[:, None] == y[None, :])
    positive_count = positives.sum(-1)
    valid = positive_count > 0
    if not valid.any():
        return z.sum() * 0, details
    # Remove unsupported anchors before logsumexp, so an M1 singleton class
    # cannot create an all-minus-infinity row and a NaN backward derivative.
    scores, nonself, positives = scores[valid], nonself[valid], positives[valid]
    positive_count = positive_count[valid]
    anchor_y = y[valid]
    denominator_classes = []
    for cls in (0, 1):
        candidates = nonself & (y[None, :] == cls)
        n = candidates.sum(-1)
        denominator_classes.append(torch.logsumexp(scores.masked_fill(~candidates, -torch.inf), -1)
                                   - n.clamp_min(1).to(scores).log())
    log_denominator = torch.logsumexp(torch.stack(denominator_classes), dim=0)
    mean_positive = scores.masked_fill(~positives, 0).sum(-1) / positive_count.clamp_min(1)
    per_anchor = log_denominator - mean_positive
    class_losses = [per_anchor[anchor_y == c].mean() for c in (0, 1)
                    if (anchor_y == c).any()]
    if not class_losses:
        return z.sum() * 0, details
    same_graph = positives & (ids[valid, None] == ids[None, :])
    different_graph = positives & (ids[valid, None] != ids[None, :])
    details.update({"eligible_anchors": int(valid.sum()), "eligible_anchor_classes": len(class_losses),
                    "same_graph_positive_fraction": float((same_graph.sum(-1)
                        / positive_count.to(scores)).detach().mean()),
                    "different_graph_positive_anchor_fraction": float(different_graph.any(-1).float().mean()),
                    "positive_count_min": int(positive_count.min()),
                    "positive_count_max": int(positive_count.max())})
    return torch.stack(class_losses).mean(), details


class MolecularObjective:
    """Stateless optional hook for the new full-epoch adapter, never frozen smoke."""
    def __init__(self, config=None):
        self.config = ObjectiveConfig() if config is None else config

    def state_dict(self):
        return {"schema": SCHEMA, "config": asdict(self.config), "learned_or_queue_state": None}

    def load_state_dict(self, state):
        if state != self.state_dict():
            raise ValueError("Checkpoint molecular objective configuration differs")

    def __call__(self, *, prehead, labels, graph_ids, role):
        if role != "TRAIN":
            raise ValueError("This training objective accepts authenticated TRAIN batches only")
        members, batch, width, y, observed, ids = _validate_inputs(prehead, labels, graph_ids)
        cfg = self.config
        zero = prehead.sum() * 0
        h, yy, ii = prehead[:, observed], y[observed], ids[observed]
        z = F.normalize(h, dim=-1, eps=cfg.normalization_epsilon)
        alignment, pair_info = _balanced_alignment(z, yy, ii, cfg.temperature)
        variance, variance_info = _conditional_variance(z, yy, cfg)
        if cfg.mode == "none":
            loss = zero
        elif cfg.mode == "alignment":
            loss = cfg.coefficient * alignment
        else:
            loss = cfg.coefficient * (alignment + variance)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite molecular auxiliary")
        diagnostics = {"schema": SCHEMA, "role": role, "members": members,
            "batch_graphs": batch, "observed_graphs": int(observed.sum()),
            "missing_labels": int((~observed).sum()), "prehead_width": width,
            "native_prehead_norm_mean": float(h.detach().norm(dim=-1).mean()) if h.numel() else None,
            "mode": cfg.mode, "coefficient": cfg.coefficient, "alignment": pair_info,
            "conditional_variance": variance_info,
            "native_forward_calls_added": 0, "TEST_accessed": False}
        return AuxiliaryResult(loss=loss, terms={"alignment": alignment, "variance": variance},
                               diagnostics=diagnostics, gradient_scope=cfg.gradient_scope)


@torch.no_grad()
def embedding_diagnostics(*, prehead, labels, graph_ids, role, config=None):
    if role not in ("TRAIN", "VALID"):
        raise ValueError("Embedding analysis is restricted to TRAIN/VALID")
    cfg = ObjectiveConfig() if config is None else config
    _, _, _, y, observed, _ = _validate_inputs(prehead, labels, graph_ids)
    z = F.normalize(prehead[:, observed], dim=-1, eps=cfg.normalization_epsilon)
    _, info = _conditional_variance(z, y[observed], cfg)
    return {"role": role, "observed_graphs": int(observed.sum()), "conditional_variance": info}


@torch.no_grad()
def prediction_diagnostics(*, raw_logits, labels, graph_ids, role, evaluator):
    """Call the supplied source OGB evaluator; never substitute a custom AUC."""
    if role not in ("TRAIN", "VALID"):
        raise ValueError("Prediction analysis is restricted to TRAIN/VALID")
    if raw_logits.ndim != 3 or raw_logits.shape[-1] != 1:
        raise ValueError("Aligned native raw logits [M,B,1] required")
    if not torch.isfinite(raw_logits).all():
        raise ValueError("Nonfinite native predictions")
    members, batch, _ = raw_logits.shape
    if members < 1 or graph_ids.ndim != 1 or len(graph_ids) != batch \
            or graph_ids.dtype != torch.long or len(torch.unique(graph_ids)) != batch:
        raise ValueError("Exact aligned graph identities required")
    y, observed = _validate_labels(labels, batch, raw_logits.device)
    if not observed.any():
        raise ValueError("No observed labels for prediction analysis")
    truth = y[:, None].detach().cpu().numpy()
    pooled = raw_logits.mean(0)
    outputs = [*raw_logits, pooled]
    scores = []
    have_both_classes = len(torch.unique(y[observed])) == 2
    for logits in outputs:
        auc = float(evaluator.eval({"y_true": truth, "y_pred": logits.detach().cpu().numpy()})["rocauc"]) \
              if have_both_classes else None
        bce = float(F.binary_cross_entropy_with_logits(logits[observed, 0], y[observed].to(logits)))
        scores.append({"rocauc": auc, "BCE": bce})
    correct = (raw_logits[:, observed, 0] >= 0) == y[observed].bool()[None, :]
    pool_correct = (pooled[observed, 0] >= 0) == y[observed].bool()
    pair_disagreement, pair_double_fault = [], []
    for m in range(members):
        for k in range(m):
            pair_disagreement.append(float((correct[m] != correct[k]).float().mean()))
            pair_double_fault.append(float((~correct[m] & ~correct[k]).float().mean()))
    per_member_accuracy = correct.float().mean(-1)
    summary = {"role": role, "graphs": batch, "observed_graphs": int(observed.sum()),
        "source_metric_scale": "fraction", "member_metrics": scores[:-1], "pooled_metrics": scores[-1],
        "mean_member_BCE": sum(x["BCE"] for x in scores[:-1]) / members,
        "worst_member_BCE": max(x["BCE"] for x in scores[:-1]),
        "member_threshold_accuracy": per_member_accuracy.cpu().tolist(),
        "pooled_threshold_accuracy": float(pool_correct.float().mean()),
        "all_members_wrong": int((~correct.any(0)).sum()),
        "any_member_correct": int(correct.any(0).sum()),
        "some_but_not_all_members_correct": int((correct.any(0) & ~correct.all(0)).sum()),
        "pool_repairs_relative_member0": int((pool_correct & ~correct[0]).sum()),
        "pool_harms_relative_member0": int((~pool_correct & correct[0]).sum()),
        "per_member_unique_correct": [int((correct[m] & (correct.sum(0) == 1)).sum()) for m in range(members)],
        "pair_threshold_disagreement_mean": sum(pair_disagreement) / len(pair_disagreement) if pair_disagreement else 0.0,
        "pair_double_fault_mean": sum(pair_double_fault) / len(pair_double_fault) if pair_double_fault else 0.0,
        "TEST_accessed": False}
    if have_both_classes:
        aucs = [x["rocauc"] for x in scores[:-1]]
        summary.update({"mean_member_rocauc": sum(aucs) / members,
                        "worst_member_rocauc": min(aucs),
                        "pool_rocauc_minus_mean_member": scores[-1]["rocauc"] - sum(aucs) / members})
    return summary
