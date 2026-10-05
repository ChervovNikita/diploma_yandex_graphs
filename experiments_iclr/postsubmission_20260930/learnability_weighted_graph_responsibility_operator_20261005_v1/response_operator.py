"""Disabled, model-agnostic finite-response operator; no trainer or native import.

Only episode_step is a release-gated entry point. Underscored math functions are
available for a separately authorized engineering oracle. This module does not
import Torch until one of those functions is deliberately called. No I/O, data
loading, RNG, optimizer history, checkpointing, or command-line entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from torch import Tensor

Params = dict[str, "Tensor"]
Forward = Callable[[Params, Params], "Tensor"]
SOURCE_RELEASED = False
CONTROLS = frozenset({"live", "uniform", "graph_free", "margins", "stop_q"})


@dataclass(frozen=True)
class Config:
    """G0: declared gate conventions, not fitted/native optimizer settings."""

    eta_probe: float = 0.01
    eta_private: float = 0.01
    eta_core: float = 0.001
    extra_margin: float = 0.1
    pool_fraction: float = 0.5
    response_epsilon: float = 0.001
    entropy: float = 1.0
    graph: float = 1.0
    assignment_steps: int = 8
    assignment_rate: float = 1.0
    ratio_smoothing: float = 0.01
    reciprocal_smoothing: float = 0.01


@dataclass(frozen=True)
class Pair:
    classes: tuple[int, int]
    nodes: Tensor                 # Local positions in innerS, both label directions.
    targets: Tensor
    competitors: Tensor
    laplacian: Tensor             # Fixed public affinity, spectral norm <= 2.


def _backend():
    import torch
    import torch.nn.functional as functional
    return torch, functional


def _sgd(params: Params, grads: Params, rate: float) -> Params:
    return {name: value - rate * grads[name] for name, value in params.items()}


def _detach(params: Params) -> Params:
    return {name: value.detach() for name, value in params.items()}


def _own_ce(theta, phi, forward, indices, labels):
    _, functional = _backend()
    return functional.cross_entropy(forward(theta, phi)[indices], labels)


def _margin(logits, pair):
    """[members, innerS, classes] -> [pair items, members]."""
    _, functional = _backend()
    gap = (logits[:, pair.nodes, pair.competitors]
           - logits[:, pair.nodes, pair.targets])
    return functional.softplus(gap).transpose(0, 1)


def _pair_blocks(labels, public_affinity, num_classes):
    """All unordered pairs; no same-label/homophily edge filter."""
    torch, _ = _backend()
    blocks = []
    for left in range(num_classes):
        for right in range(left + 1, num_classes):
            nodes = torch.nonzero((labels == left) | (labels == right)).flatten()
            targets = labels[nodes]
            competitors = torch.where(targets == left, right, left)
            raw = public_affinity[nodes][:, nodes]
            affinity = raw / (1.0 + raw.sum(dim=1).max())
            laplacian = torch.diag(affinity.sum(dim=1)) - affinity
            blocks.append(Pair((left, right), nodes, targets, competitors, laplacian))
    return tuple(blocks)


def _normalized_cost(raw, epsilon):
    """Center across members per item; ONE smooth RMS scale per pair."""
    centered = raw - raw.mean(dim=1, keepdim=True)
    scale = (epsilon * epsilon + centered.square().mean()).sqrt()
    return centered / scale, centered, scale


def _assignment_map(cost, laplacian, cfg: Config, graph_coefficient=None):
    """Exactly T smooth feasible descent steps, NOT an exact convex optimizer.

    Objective: <cost,Q> + tau*sum Q*log(M*Q) + gamma/2*tr(Q^T L Q).
    Q starts uniform. H is the orthogonal row/column tangent projection of G.
    Smooth upper bounds on max(H/Q) and max(1/Q) control positivity/curvature.
    Real-arithmetic invariants: rows sum 1; columns sum n/M; Q > 0.
    No clamping, balancing repair, warm start, tolerance stop, or implicit VJP.
    """
    torch, _ = _backend()
    members = cost.shape[1]
    gamma = cfg.graph if graph_coefficient is None else graph_coefficient
    q = torch.full_like(cost, 1.0 / members)
    for _ in range(cfg.assignment_steps):
        gradient = (cost + cfg.entropy * ((members * q).log() + 1.0)
                    + gamma * (laplacian @ q))
        tangent = (gradient - gradient.mean(dim=1, keepdim=True)
                   - gradient.mean(dim=0, keepdim=True) + gradient.mean())
        ratio_bound = cfg.ratio_smoothing * torch.logsumexp(
            (tangent / q).flatten() / cfg.ratio_smoothing, dim=0)
        reciprocal_bound = cfg.reciprocal_smoothing * torch.logsumexp(
            q.reciprocal().flatten() / cfg.reciprocal_smoothing, dim=0)
        step = 1.0 / (1.0 / cfg.assignment_rate + 2.0 * ratio_bound
                      + 2.0 * cfg.entropy * reciprocal_bound + 2.0 * gamma)
        q = q - step * tangent
    return q


def _main_loss(theta, phi, q_blocks, member, forward, inner_indices,
               inner_labels, pairs, num_classes, cfg):
    """Q IS AN INDEPENDENT FUNCTION ARGUMENT for the private partial."""
    _, functional = _backend()
    logits = forward(theta, phi)[inner_indices]
    own = functional.cross_entropy(logits, inner_labels)
    allocated = sum((q[:, member] * _margin(logits.unsqueeze(0), pair)[:, 0]).sum()
                    for q, pair in zip(q_blocks, pairs))
    count = inner_labels.shape[0] * (num_classes - 1)
    members = q_blocks[0].shape[1]
    return own + cfg.extra_margin * (members / count) * allocated


def _private_response(theta, phis, forward, inner_indices, inner_labels,
                      pairs, num_classes, cfg: Config, control="live",
                      collect_diagnostics=False):
    """Probe is discarded; only the independent-Q main response is returned."""
    torch, functional = _backend()
    before = torch.stack([forward(theta, phi)[inner_indices] for phi in phis])
    own_partial = torch.func.grad(_own_ce, argnums=1)
    probes = tuple(_sgd(phi, own_partial(theta, phi, forward,
                                        inner_indices, inner_labels), cfg.eta_probe)
                   for phi in phis)
    after = torch.stack([forward(theta, phi)[inner_indices] for phi in probes])
    q_blocks, diagnostics = [], []
    gamma = 0.0 if control == "graph_free" else cfg.graph
    for pair in pairs:
        before_margin, after_margin = _margin(before, pair), _margin(after, pair)
        response = after_margin - before_margin
        raw = before_margin if control == "margins" else response
        cost, centered, scale = _normalized_cost(raw, cfg.response_epsilon)
        q = _assignment_map(cost, pair.laplacian, cfg, gamma)
        if control == "uniform":
            q = torch.full_like(q, 1.0 / len(phis))
        if control == "stop_q":
            q = q.detach()       # ONLY this control stops the outer Q chain.
        q_blocks.append(q)
        if collect_diagnostics:
            qd, cd, sd = q.detach(), centered.detach(), scale.detach()
            response_centered = (response - response.mean(dim=1, keepdim=True)).detach()
            delta = qd - 1.0 / len(phis)
            diagnostics.append({
                "classes": pair.classes,
                "item_count": pair.nodes.shape[0],
                "left_count": (pair.targets == pair.classes[0]).sum(),
                "right_count": (pair.targets == pair.classes[1]).sum(),
                "cost_kind": "current_margins" if control == "margins" else "response",
                "centered_response_rms": response_centered.square().mean().sqrt(),
                "centered_cost_rms": cd.square().mean().sqrt(),
                "smooth_cost_scale": sd,
                "epsilon_over_scale": cfg.response_epsilon / sd,
                "normalized_cost_rms": (cd / sd).square().mean().sqrt(),
                "assignment_relative_rms": len(phis) * delta.square().mean().sqrt(),
                "assignment_max_deviation": delta.abs().max(),
                "mean_entropy": -(qd * qd.log()).sum(dim=1).mean(),
                "min_assignment": qd.min(),
                "row_residual": (qd.sum(dim=1) - 1.0).abs().max(),
                "column_residual": (qd.sum(dim=0) - qd.shape[0] / len(phis)).abs().max(),
            })
    q_blocks = tuple(q_blocks)
    # torch.func.grad differentiates the private ARGUMENT, not a prebuilt graph.
    # It treats Q as independent here, then preserves the returned gradient's
    # dependence on the supplied Q(theta, ALL phi) for the outer core derivative.
    main_partial = torch.func.grad(_main_loss, argnums=1)
    adapted = tuple(_sgd(phi, main_partial(theta, phi, q_blocks, member, forward,
                                           inner_indices, inner_labels, pairs,
                                           num_classes, cfg), cfg.eta_private)
                    for member, phi in enumerate(phis))
    if collect_diagnostics:
        probe_ce_change = torch.stack([
            functional.cross_entropy(after[m], inner_labels)
            - functional.cross_entropy(before[m], inner_labels)
            for m in range(len(phis))]).detach()
        info = {"pairs": tuple(diagnostics), "probe_own_ce_change": probe_ce_change,
                "assignments": tuple(q.detach() for q in q_blocks)}
    else:
        info = None
    return adapted, info


def _query_objective(theta, adapted, forward, query_indices, query_labels, cfg):
    torch, functional = _backend()
    log_probs = torch.stack([
        functional.log_softmax(forward(theta, phi)[query_indices], dim=-1)
        for phi in adapted])
    targets = query_labels[None, :, None].expand(len(adapted), -1, 1)
    own = -log_probs.gather(2, targets).mean()
    pooled_log_probs = torch.logsumexp(log_probs, dim=0) - log_probs.new_tensor(
        float(len(adapted))).log()
    pool = -pooled_log_probs.gather(1, query_labels[:, None]).mean()
    return (1.0 - cfg.pool_fraction) * own + cfg.pool_fraction * pool


def _validate(inner_indices, inner_labels, query_indices, query_labels,
              public_affinity, num_classes, phis, cfg, control):
    torch, _ = _backend()
    if not phis or num_classes < 2 or control not in CONTROLS:
        raise ValueError("Need members, >=2 classes and a declared control")
    if cfg.assignment_steps != 8:
        raise ValueError("This source binds exactly eight assignment steps")
    positive = (cfg.eta_probe, cfg.eta_private, cfg.eta_core, cfg.response_epsilon,
                cfg.entropy, cfg.assignment_rate, cfg.ratio_smoothing,
                cfg.reciprocal_smoothing)
    if not all(isfinite(value) and value > 0.0 for value in positive):
        raise ValueError("Strictly positive step/entropy/smoothing constants required")
    if (not all(isfinite(value) for value in (cfg.graph, cfg.extra_margin, cfg.pool_fraction))
            or cfg.graph < 0.0 or cfg.extra_margin < 0.0
            or not 0.0 <= cfg.pool_fraction <= 1.0):
        raise ValueError("Invalid graph, loss or pool coefficient")
    for indices, labels in ((inner_indices, inner_labels), (query_indices, query_labels)):
        if (indices.ndim != 1 or labels.shape != indices.shape or not indices.numel()
                or indices.dtype != torch.long or labels.dtype != torch.long):
            raise ValueError("Nonempty aligned one-dimensional long indices/labels required")
        if (indices.unique().numel() != indices.numel() or bool((indices < 0).any())
                or bool(((labels < 0) | (labels >= num_classes)).any())):
            raise ValueError("Duplicate/negative indices or out-of-range labels")
    if bool(torch.isin(inner_indices, query_indices).any()):
        raise ValueError("Current innerS and outerR must be disjoint")
    if not bool((torch.bincount(inner_labels, minlength=num_classes) > 0).all()):
        raise ValueError("innerS must contain both directions of every class pair")
    size = inner_labels.shape[0]
    if (public_affinity.shape != (size, size) or public_affinity.requires_grad
            or not bool(torch.isfinite(public_affinity).all())
            or bool((public_affinity < 0).any())
            or not torch.equal(public_affinity, public_affinity.transpose(0, 1))
            or bool((public_affinity.diagonal() != 0).any())):
        raise ValueError("Fixed symmetric finite nonnegative zero-diagonal public affinity required")


def episode_step(theta: Params, phis: tuple[Params, ...], forward: Forward,
                 inner_indices: Tensor, inner_labels: Tensor,
                 query_indices: Tensor, query_labels: Tensor,
                 public_affinity: Tensor, num_classes: int,
                 cfg: Config = Config(), control: str = "live"):
    """One pure episode; no label outside innerS/outerR is accepted.

    Caller must attest heldA exclusion from ALL previous acquisition/selection,
    label-free affinity provenance, complete private partition and deterministic
    pure forward returning [all graph nodes, num_classes] with actual full context.
    State truncates between episodes. There are no Adam moments or RNG changes.
    """
    if not SOURCE_RELEASED:
        raise RuntimeError("Disabled source-only candidate: SOURCE_RELEASED is False")
    torch, _ = _backend()
    _validate(inner_indices, inner_labels, query_indices, query_labels,
              public_affinity, num_classes, phis, cfg, control)
    theta_start = _detach(theta)
    private_start = tuple(_detach(phi) for phi in phis)
    pairs = _pair_blocks(inner_labels, public_affinity, num_classes)

    def outer(core):
        adapted, _ = _private_response(core, private_start, forward,
                                       inner_indices, inner_labels, pairs,
                                       num_classes, cfg, control)
        return _query_objective(core, adapted, forward,
                                query_indices, query_labels, cfg)

    core_gradient, virtual_query_loss = torch.func.grad_and_value(outer)(theta_start)
    theta_next = _detach(_sgd(theta_start, core_gradient, cfg.eta_core))
    # Recompute EVERYTHING at theta+, starting from original private phi.
    # The preliminary probe and theta-time virtual main response are discarded.
    recomputed, diagnostics = _private_response(
        theta_next, private_start, forward, inner_indices, inner_labels, pairs,
        num_classes, cfg, control, collect_diagnostics=True)
    private_next = tuple(_detach(phi) for phi in recomputed)
    diagnostics["virtual_query_loss"] = virtual_query_loss.detach()
    return theta_next, private_next, diagnostics
