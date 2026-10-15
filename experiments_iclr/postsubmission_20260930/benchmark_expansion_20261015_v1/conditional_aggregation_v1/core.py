"""Four-route raw-logit readouts and caller-owned native global-J risk.

Torch is injected. No model/data import, optimizer, label-derived gate feature,
teacher, extra backbone, or fitting loop is defined here.
"""
from __future__ import annotations
import copy
import math

MEMBERS = 4
MODES = ("mean", "global", "conditional")
TASKS = ("multiclass", "binary")
UNIFORM_MASS = 0.2
LEARNED_MASS = 0.8
HIDDEN = 4
LOG_EPSILON = 1e-12
GATE_SEED_OFFSET = 1301009
SCHEMA = "conditional-raw-logit-aggregator-v1"


def _aligned(torch, raw_logits, task_kind):
    values = tuple(raw_logits)
    if len(values) != MEMBERS or any(not isinstance(v, torch.Tensor) for v in values):
        raise ValueError("Exactly four aligned raw-logit tensors required")
    first = values[0]
    if first.dtype not in (torch.float32, torch.float64) or any(
            v.shape != first.shape or v.dtype != first.dtype or v.device != first.device for v in values):
        raise ValueError("Aligned native FP32/FP64 raw-logit shape/dtype/device required")
    if task_kind == "multiclass":
        if first.ndim != 2 or first.shape[0] == 0 or first.shape[1] < 2:
            raise ValueError("Multiclass route logits must have shape [queries,classes>=2]")
    elif task_kind == "binary":
        if first.ndim not in (1, 2) or first.shape[0] == 0 or (first.ndim == 2 and first.shape[1] != 1):
            raise ValueError("Single-task binary route logits must have shape [queries] or [queries,1]")
    else:
        raise ValueError("Explicit multiclass or single-task binary logit domain required")
    if any(not torch.isfinite(v).all().item() for v in values):
        raise ValueError("Every native raw logit must be finite")
    return torch.stack(values)


def _descriptors(torch, logits, task_kind):
    # Gate inputs are predictions only. Detach the complete descriptor path;
    # prediction logits stay live in the final raw-logit readout.
    with torch.no_grad():
        if task_kind == "multiclass":
            probability = logits.detach().softmax(-1)
        else:
            positive = logits.detach().reshape(MEMBERS, -1).sigmoid()
            probability = torch.stack((1 - positive, positive), dim=-1)
        mean_probability = probability.mean(0, keepdim=True).expand_as(probability)
        midpoint = (probability + mean_probability) * 0.5
        logp = probability.clamp_min(LOG_EPSILON).log()
        logq = mean_probability.clamp_min(LOG_EPSILON).log()
        logr = midpoint.clamp_min(LOG_EPSILON).log()
        entropy = -(probability * logp).sum(-1)
        confidence = (1 - entropy / math.log(probability.shape[-1])).clamp(0, 1)
        js = 0.5 * ((probability * (logp - logr)).sum(-1)
                    + (mean_probability * (logq - logr)).sum(-1))
        disagreement = (js / math.log(2)).clamp(0, 1)
        result = torch.stack((confidence, disagreement), dim=-1)
        if not torch.isfinite(result).all().item():
            raise ValueError("Nonfinite detached prediction descriptor")
        return result


def make_aggregator(torch, *, mode, task_kind, seed, reference):
    """Build 0/4/20-parameter mean/global/conditional readouts.

    reference supplies only native dtype/device. Conditional W is a fixed-law
    Gaussian draw from an isolated CPU generator; no global RNG is consumed.
    """
    if mode not in MODES or task_kind not in TASKS:
        raise ValueError("Unlisted aggregation mode/task")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63 - GATE_SEED_OFFSET:
        raise ValueError("Explicit nonnegative, bounded gate seed required")
    if not isinstance(reference, torch.Tensor) or reference.dtype not in (torch.float32, torch.float64):
        raise TypeError("Native FP32/FP64 reference tensor required")
    config = {"schema": SCHEMA, "mode": mode, "task_kind": task_kind,
              "routes": MEMBERS, "uniform_mass": UNIFORM_MASS, "learned_mass": LEARNED_MASS,
              "route_floor": UNIFORM_MASS / MEMBERS, "hidden": HIDDEN,
              "features": ["one_minus_normalized_entropy", "normalized_JS_to_route_mean_probability"],
              "descriptor_gradient": "detached", "log_epsilon": LOG_EPSILON,
              "seed": seed, "conditional_W_seed": seed + GATE_SEED_OFFSET,
              "conditional_W_law": "iid_normal_mean0_std1/sqrt2_CPU_FP64_then_native_cast",
              "output_bias": False, "native_dtype": str(reference.dtype)}

    class Aggregator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("_anchor", reference.detach().new_empty(0), persistent=False)
            if mode != "mean":
                self.member_bias = torch.nn.Parameter(reference.detach().new_zeros(MEMBERS))
            if mode == "conditional":
                generator = torch.Generator(device="cpu").manual_seed(seed + GATE_SEED_OFFSET)
                initial = torch.randn(HIDDEN, 2, generator=generator, device="cpu", dtype=torch.float64) / math.sqrt(2)
                self.W = torch.nn.Parameter(initial.to(reference))
                self.a = torch.nn.Parameter(reference.detach().new_zeros(HIDDEN))
                self.v = torch.nn.Parameter(reference.detach().new_zeros(HIDDEN))

        def get_extra_state(self):
            return copy.deepcopy(config)

        def set_extra_state(self, state):
            if state != config:
                raise ValueError("Aggregator checkpoint method/task/seed/config differs")

        def metadata(self):
            return {**copy.deepcopy(config), "parameters": sum(p.numel() for p in self.parameters()),
                    "initial_weights": "exactly 1/4", "prediction_pool": "convex raw logits",
                    "binary_interpretation": "source-label confidence surrogate; no calibrated natural edge probability claim",
                    "novelty_claim": False}

        def forward(self, raw_logits):
            z = _aligned(torch, raw_logits, task_kind)
            if z.device != self._anchor.device or z.dtype != self._anchor.dtype:
                raise ValueError("Aggregator/native raw-logit dtype or device differs")
            descriptors = None
            if mode == "mean":
                weights = z.new_full((MEMBERS, z.shape[1]), 1 / MEMBERS)
            else:
                scores = self.member_bias[:, None].expand(MEMBERS, z.shape[1])
                if mode == "conditional":
                    descriptors = _descriptors(torch, z, task_kind)
                    hidden = torch.nn.functional.gelu(torch.nn.functional.linear(descriptors, self.W, self.a))
                    scores = scores + (hidden * self.v).sum(-1)
                weights = UNIFORM_MASS / MEMBERS + LEARNED_MASS * scores.softmax(dim=0)
            # Algebraically sum(alpha*z). The centered form also preserves the
            # source's exact mean arithmetic at uniform initialization while
            # keeping derivatives to gate parameters live at that point.
            correction = weights - 1 / MEMBERS
            if z.ndim == 3:
                correction = correction.unsqueeze(-1)
            served = z.mean(0) + (correction * z).sum(0)
            if not torch.isfinite(served).all().item() or not torch.isfinite(weights).all().item():
                raise ValueError("Nonfinite served readout/weights")
            return {"served_raw_logits": served, "weights": weights, "descriptors": descriptors}

    return Aggregator()


def _scalar(torch, value, name, *, live=False):
    if not isinstance(value, torch.Tensor) or value.ndim != 0 or not torch.isfinite(value).item():
        raise ValueError(name + " must be a finite native scalar tensor")
    if live and not value.requires_grad:
        raise ValueError(name + " must retain the native training computation graph")
    return value


def native_objective(torch, *, native_member_losses, raw_logits, aggregator,
                     served_native_loss, native_regularizer):
    """Use caller-supplied live source reductions, including source groups.

    Per-member inputs are task losses without regularization. The unchanged
    source scalar regularizer occurs once in F and L, hence once in J. The
    callback consumes the ACTUAL served raw logits and owns masks, activations,
    labels, positive/negative groups, and native within-group reductions.
    """
    losses = tuple(native_member_losses)
    if len(losses) != MEMBERS:
        raise ValueError("Four caller-owned native per-route task losses required")
    losses = torch.stack([_scalar(torch, loss, "Native route loss", live=True) for loss in losses])
    regularizer = _scalar(torch, native_regularizer, "Native regularizer")
    readout = aggregator(raw_logits)
    served_loss = _scalar(torch, served_native_loss(readout["served_raw_logits"]), "Actual served native task loss", live=True)
    F, L = losses.mean() + regularizer, served_loss + regularizer
    return {**readout, "native_member_losses": losses, "F": F, "L": L, "J": 0.5 * F + 0.5 * L}
