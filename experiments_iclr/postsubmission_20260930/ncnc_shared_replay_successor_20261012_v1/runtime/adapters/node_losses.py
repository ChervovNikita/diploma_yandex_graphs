"""Source-attributed node-loss formulas; Torch/F are injected by the caller.

These tensor primitives do not reproduce a training recipe or qualify a new
backbone. Inputs must already contain the declared ordered TRAIN rows. No model,
label reader, optimizer, RNG draw, detached teacher, or universal recipe lives
here. SOURCE_CONTRACTS records the actual archived applicability boundaries.
"""
from __future__ import annotations

import math


SOURCE_CONTRACTS = {
    "native_sage_gncl_supcon": {
        "source": "native_SAGE_GNCL_SupCon_2x2_source_20261010_v1/run_family.py::Family",
        "backbone": "native SAGE", "dataset": "WikiCS", "train_rows": 580,
        "members": (1, 4), "gncl_members": 4, "pool_weight": .5,
        "supcon_weight": .05, "temperature": .2,
        "hidden": "live native preclassifier H, all ordered TRAIN",
        "gncl": "convex own-CE/probability-pool CE; no covariance penalty",
    },
    "portable_internal_be": {
        "source": "portable_internal_be_public_interface_20261007_v2/core/objectives.py",
        "backbones": ("WikiPolynormer", "CollabGCN+NCN", "bond-aware residual GINE+virtual node"),
        "tasks": ("WikiCS node CE", "Collab balanced edge BCE", "molecule BCE"),
        "alignment_weight": .05, "residual_weight": .05, "temperature": .2,
        "own_views": 2, "auxiliary_panel": "source max_objects and ordered subsampling",
        "independent_scaling": "sum own + M*aux; each member retains unscaled own gradients",
        "status": "explicit graph adaptation; not DICE/CDLG reproduction or conditional MI",
    },
    "portable_wikics_supcon_comparison": {
        "source": "portable_wikics_supcon_loss_comparison_20261008_v1/losses.py and train.py::ObjectivesFacade",
        "backbone": "shared four-route Polynormer, unit factors", "dataset": "WikiCS",
        "members": 4, "auxiliary_rows": 512, "complete_train_rows": 580,
        "coefficient": .05, "temperature": .2, "residual_weight": 0.,
        "gradient_scope": "upstream representation; captured task head own CE only",
    },
    "context_targets": {
        "sources": ("native_SAGE_context_positive_exclusion_source_20261010_v1/context_objective.py",
                    "native_SAGE_continuous_context_affinity_source_20261011_v2/context_affinity_objective.py"),
        "backbone": "native SAGE", "dataset": "WikiCS", "train_rows": 580,
        "members": (1, 4), "affinity_hidden_width": 128,
        "coefficient": .05, "temperature": .2,
        "required": "prospectively prepared frozen target/support banks and row custody",
        "mixture": "mean of complete target/support losses; never union denominators",
        "row_mass": "literal Q row mass multiplies logZ; no renormalization",
        "cross_route_repulsion": False,
    },
}

REQUIRES_SPECIAL_ADAPTER = {
    "O/I/P/G": "public_internal_be_allocation_controls_20261007_v1/method.py: parameter-role gradient assignment; no global scalar objective",
    "private_CMCL": "private_cmcl_polyformer_source_20261010_v1/objective.py: stopped same-state owners plus private/all-block gradient permissions",
    "FoRDE": "conditional_graph_response_dice_forde_source_recipes_20261003_v2/core/forde_identity.py: input-gradient profiles, bandwidth and first-argument ownership surrogate",
    "fixed_context_key": "native_SAGE_fixed_context_key_source_20261011_v1: registered bare query projection and frozen key/moment/provider custody",
}

VARIANT_COEFFICIENTS = {
    "own": set(),
    "gncl": {"pool_weight"},
    "within_route_supcon": {"supcon_weight", "temperature"},
    "gncl_within_route_supcon": {"pool_weight", "supcon_weight", "temperature"},
    "two_view_class_full_cross_view": {"alignment_weight", "temperature"},
    "two_view_supcon_eq2": {"supcon_weight", "temperature"},
    "two_view_internal_be": {"alignment_weight", "residual_weight", "temperature"},
}


def _number(value, name, *, positive=False, unit=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name + " must be a declared finite Python coefficient")
    if value < 0 or (positive and value == 0) or (unit and value > 1):
        raise ValueError("Invalid " + name)
    return float(value)


def _finite(torch, tensor, name):
    if not bool(torch.isfinite(tensor).all().item()):
        raise FloatingPointError("Nonfinite " + name)


def _labels(torch, labels, rows, device, classes=None):
    if labels.ndim != 1 or labels.shape[0] != rows or labels.dtype != torch.long or labels.device != device:
        raise ValueError("Exact ordered TRAIN int64 labels on the tensor device required")
    if bool((labels < 0).any().item()) or (classes is not None and bool((labels >= classes).any().item())):
        raise ValueError("TRAIN class index outside output domain")


def _logits(torch, logits, labels):
    if logits.ndim != 3 or min(logits.shape) < 1 or not logits.is_floating_point():
        raise ValueError("Finite floating [route, TRAIN row, class] logits required")
    _labels(torch, labels, logits.shape[1], logits.device, logits.shape[2])
    _finite(torch, logits, "TRAIN logits")


def _hidden(torch, hidden, labels):
    if hidden is None or hidden.ndim != 3 or min(hidden.shape) < 1 or not hidden.is_floating_point():
        raise ValueError("Live floating [route, TRAIN row, feature] hidden states required")
    _labels(torch, labels, hidden.shape[1], hidden.device)
    _finite(torch, hidden, "TRAIN hidden states")


def _pair(torch, a, b, labels, temperature):
    _hidden(torch, a, labels)
    _hidden(torch, b, labels)
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        raise ValueError("Two exactly aligned representation views required")
    return _number(temperature, "temperature", positive=True)


def own_ce(torch, functional, logits, labels):
    """Native SAGE source's single flattened mean CE across all routes/rows."""
    _logits(torch, logits, labels)
    return functional.cross_entropy(logits.flatten(0, 1), labels.repeat(logits.shape[0]))


def probability_pool_ce(torch, logits, labels):
    """CE of mean softmax probabilities, using the exact stable source algebra."""
    _logits(torch, logits, labels)
    log_pool = torch.logsumexp(logits.log_softmax(-1), dim=0) - math.log(logits.shape[0])
    result = -log_pool.gather(1, labels[:, None]).mean()
    _finite(torch, result, "probability-pool CE")
    return result


def gncl(torch, functional, logits, labels, *, pool_weight):
    """Archived generalized negative-correlation mixture; not output covariance."""
    weight = _number(pool_weight, "pool_weight", unit=True)
    own = own_ce(torch, functional, logits, labels)
    if weight == 0:
        return own
    pool = probability_pool_ce(torch, logits, labels)
    return (1 - weight) * own + weight * pool


def within_route_supcon(torch, functional, hidden, labels, *, temperature):
    """One view per route; every anchor needs a nonself same-class positive.

    Source: native SAGE Family.within_route_supcon. Returns mean and route losses.
    This primitive accepts arbitrary TRAIN size; archived recipe requires580.
    """
    _hidden(torch, hidden, labels)
    temperature = _number(temperature, "temperature", positive=True)
    normalized = functional.normalize(hidden, p=2, dim=-1, eps=1e-12)
    scores = (normalized @ normalized.transpose(-2, -1)) / temperature
    allowed = ~torch.eye(len(labels), dtype=torch.bool, device=hidden.device)
    positives = labels[:, None].eq(labels[None, :]) & allowed
    positive_counts = positives.sum(-1)
    if bool((positive_counts == 0).any().item()):
        raise ValueError("Every TRAIN anchor requires a nonself same-class positive")
    log_z = torch.logsumexp(scores.masked_fill(~allowed[None], -float("inf")), dim=-1)
    positive_score = scores.masked_fill(~positives[None], 0).sum(-1) / positive_counts[None]
    routes = (log_z - positive_score).mean(-1)
    _finite(torch, routes, "within-route SupCon")
    return routes.mean(), routes


def class_full_cross_view(torch, functional, a, b, labels, *, temperature):
    """Portable alignment: same-class opposite-view positives incl. counterpart.

    All opposite-view TRAIN objects form the denominator, separately per route.
    No cross-route pair is introduced. Preserves the source's B<2 zero guard.
    """
    temperature = _pair(torch, a, b, labels, temperature)
    if a.shape[1] < 2:
        return a.sum() * 0
    a, b = functional.normalize(a, dim=-1), functional.normalize(b, dim=-1)
    scores = torch.einsum("mbd,mcd->mbc", a, b) / temperature
    positives = labels[:, None] == labels[None, :]

    def direction(value):
        return -(functional.log_softmax(value, -1) * positives).sum(-1) / positives.sum(-1)

    result = .5 * (direction(scores).mean() + direction(scores.transpose(1, 2)).mean())
    _finite(torch, result, "class-full cross-view alignment")
    return result


def canonical_supcon_eq2(torch, functional, a, b, labels, *, temperature):
    """Archived paired Eq2: outside-log positive mean, exact anchor excluded."""
    temperature = _pair(torch, a, b, labels, temperature)
    z = torch.cat((functional.normalize(a, dim=-1), functional.normalize(b, dim=-1)), dim=1)
    scores = z @ z.transpose(1, 2) / temperature
    _finite(torch, scores, "paired SupCon raw scores")
    self_mask = torch.eye(2 * a.shape[1], dtype=torch.bool, device=a.device)
    y = labels.repeat(2)
    positive = (y[:, None] == y[None, :]) & ~self_mask
    log_z = torch.logsumexp(scores.masked_fill(self_mask, -float("inf")), dim=-1)
    positive_mean = (scores * positive).sum(-1) / positive.sum(-1)
    result = (log_z - positive_mean).mean()
    _finite(torch, result, "paired SupCon Eq2")
    return result


def residual_member_contrast(torch, functional, a, b, labels, *, temperature):
    """Portable differentiable class-centering, same-object cross-route InfoNCE.

    Same route across views is positive; other routes at that same object are
    negatives. Singleton classes are skipped; M1 is exactly the source zero.
    Class centers remain live. This is not conditional mutual information.
    """
    temperature = _pair(torch, a, b, labels, temperature)
    members, rows, _ = a.shape
    if members == 1:
        return a.sum() * 0
    ra, rb = a.clone(), b.clone()
    valid = torch.zeros(rows, dtype=torch.bool, device=a.device)
    for cls in torch.unique(labels):
        mask = labels == cls
        if int(mask.sum()) >= 2:
            valid |= mask
            ra[:, mask] = a[:, mask] - a[:, mask].mean(1, keepdim=True)
            rb[:, mask] = b[:, mask] - b[:, mask].mean(1, keepdim=True)
    if not bool(valid.any().item()):
        return a.sum() * 0
    ra, rb = functional.normalize(ra[:, valid], dim=-1), functional.normalize(rb[:, valid], dim=-1)
    scores = torch.einsum("mbd,kbd->bmk", ra, rb) / temperature
    targets = torch.arange(members, device=a.device).expand(scores.shape[0], -1).reshape(-1)
    result = .5 * (functional.cross_entropy(scores.reshape(-1, members), targets)
                   + functional.cross_entropy(scores.transpose(1, 2).reshape(-1, members), targets))
    _finite(torch, result, "class-centered residual contrast")
    return result


def node_objective(torch, functional, logits, hidden, labels, *, variant, coefficients,
                   paired_logits=None, paired_hidden=None, auxiliary_labels=None, independent=False):
    """Live objective/component tensors for adapter-owned training/backward.

    Explicit coefficients are mandatory; no source strengths are silently
    adopted. Tensor primitives generalize algebra only. The caller must bind
    recipe fidelity/applicability using SOURCE_CONTRACTS and its data authority.
    """
    if variant not in VARIANT_COEFFICIENTS or set(coefficients) != VARIANT_COEFFICIENTS[variant]:
        raise ValueError("Declare one supported loss variant and its exact coefficient fields")
    values = {key: _number(value, key, positive=key == "temperature", unit=key == "pool_weight")
              for key, value in coefficients.items()}
    if type(independent) is not bool:
        raise ValueError("Declare independent/shared reduction explicitly")
    _logits(torch, logits, labels)
    paired = variant.startswith("two_view_")
    auxiliary_labels = labels if auxiliary_labels is None else auxiliary_labels
    components = {}
    if paired:
        if paired_logits is None:
            raise ValueError("Paired variants retain two own-supervised TRAIN views")
        _logits(torch, paired_logits, labels)
        if logits.shape != paired_logits.shape or logits.dtype != paired_logits.dtype or logits.device != paired_logits.device:
            raise ValueError("Matched paired TRAIN logits required")
        own_routes = .5 * (torch.stack([functional.cross_entropy(z, labels) for z in logits])
                           + torch.stack([functional.cross_entropy(z, labels) for z in paired_logits]))
        ce = own_routes.sum() if independent else own_routes.mean()
        components.update(own_per_route=own_routes, own_mean_ce=own_routes.mean())
    else:
        if independent or paired_logits is not None or paired_hidden is not None:
            raise ValueError("Independent sum reduction/paired tensors belong to paired source variants")
        ce = own_ce(torch, functional, logits, labels)
        components["own_mean_ce"] = ce
    if values.get("pool_weight", 0):
        pool = probability_pool_ce(torch, logits, labels)
        components["probability_pool_ce"] = pool
        base_objective = (1 - values["pool_weight"]) * ce + values["pool_weight"] * pool
    else:
        base_objective = ce
    aux = ce * 0
    if values.get("supcon_weight", 0):
        if paired:
            sc = canonical_supcon_eq2(torch, functional, hidden, paired_hidden, auxiliary_labels,
                                      temperature=values["temperature"])
        else:
            _hidden(torch, hidden, labels)
            if hidden.shape[:2] != logits.shape[:2] or hidden.dtype != logits.dtype or hidden.device != logits.device:
                raise ValueError("Hidden states must align exactly with TRAIN logits")
            sc, routes = within_route_supcon(torch, functional, hidden, labels, temperature=values["temperature"])
            components["supcon_per_route"] = routes
        components["supcon"] = sc
        aux = aux + values["supcon_weight"] * sc
    if values.get("alignment_weight", 0):
        alignment = class_full_cross_view(torch, functional, hidden, paired_hidden, auxiliary_labels,
                                           temperature=values["temperature"])
        components["alignment"] = alignment
        aux = aux + values["alignment_weight"] * alignment
    if values.get("residual_weight", 0):
        residual = residual_member_contrast(torch, functional, hidden, paired_hidden, auxiliary_labels,
                                            temperature=values["temperature"])
        components["residual"] = residual
        aux = aux + values["residual_weight"] * residual
    if paired and (values.get("supcon_weight", 0) or values.get("alignment_weight", 0) or values.get("residual_weight", 0)):
        if hidden.shape[0] != logits.shape[0] or hidden.dtype != logits.dtype or hidden.device != logits.device:
            raise ValueError("Paired auxiliary panel must have the same routes/dtype/device as own logits")
    aux = logits.shape[0] * aux if independent else aux
    objective = base_objective + aux
    _finite(torch, objective, "complete node objective")
    return {"objective": objective, "ce": base_objective, "regularization": aux, "components": components}


def context_auxiliary(torch, functional, hidden, labels, *, source, mode, targets, temperature,
                      context_index=None):
    """Exact native prepared-target peer formulas; target preparation stays external.

    Preserves full580 source constraints and literal float32 Q row masses.
    Returns mean, route and route/context pair losses. No cross-route repulsion.
    """
    _hidden(torch, hidden, labels)
    temperature = _number(temperature, "temperature", positive=True)
    members, rows, width = hidden.shape
    if members not in (1, 4) or rows != 580 or hidden.dtype != torch.float32:
        raise ValueError("Archived native context losses require float32 [1or4,580,D] H")
    if any(value.requires_grad or value.device != hidden.device for value in targets.values()):
        raise ValueError("Frozen same-device prepared context bank required")
    weights, allowed = targets["weights"], targets["full_allowed"]
    if weights.shape != (4, rows, rows) or weights.dtype != hidden.dtype or allowed.shape != (rows, rows) or allowed.dtype != torch.bool:
        raise ValueError("Four frozen aligned Q matrices and Boolean full support required")
    _finite(torch, weights, "prepared context weights")
    if bool((weights < 0).any().item()) or bool((weights.sum(-1) <= 0).any().item()):
        raise ValueError("Nonnegative context weights with positive row masses required")
    normalized = functional.normalize(hidden, p=2, dim=-1, eps=1e-12)
    scores = (normalized @ normalized.transpose(-2, -1)) / temperature

    def complete(q, support):
        if q.requires_grad or support.requires_grad or q.dtype != hidden.dtype or support.dtype != torch.bool:
            raise ValueError("Detached same-dtype target weights and Boolean support required")
        if q.device != hidden.device or support.device != hidden.device:
            raise ValueError("Aligned context target/support devices required")
        _finite(torch, q, "context Q")
        if bool((q < 0).any().item()) or bool((q.masked_select(~support) != 0).any().item()):
            raise ValueError("Context target mass must lie inside its own denominator")
        if bool((~support.any(-1)).any().item()):
            raise ValueError("Every context anchor needs denominator support")
        log_z = torch.logsumexp(scores.masked_fill(~support, -float("inf")), dim=-1)
        return (q.sum(-1) * log_z - (q * scores).sum(-1)).mean(-1)

    if source == "positive_exclusion":
        if context_index is not None:
            raise ValueError("Positive-exclusion source has no assigned context index")
        if mode not in {"private", "mixture", "private_full", "shuffled_private"}:
            raise ValueError("Unknown archived positive-exclusion mode")
        if mode != "mixture" and members != 4:
            raise ValueError("Private context allocation requires four routes")
        q = targets["shuffled_weights"] if mode == "shuffled_private" else weights
        support = targets["shuffled_retained"] if mode == "shuffled_private" else targets["retained"]
        if q.shape != (4, rows, rows) or support.shape != q.shape:
            raise ValueError("Four aligned complete target/support pairs required")
        if mode == "mixture":
            pairs = torch.stack([complete(q[c][None], support[c][None]) for c in range(4)], dim=-1)
            routes = pairs.mean(-1)
        else:
            routes = complete(q, allowed[None] if mode == "private_full" else support)
            pairs = routes[:, None]
    elif source == "continuous_affinity":
        if width != 128 or mode not in {"weighted_private", "weighted_mixture", "uniform_allclassmate", "weighted_assigned"}:
            raise ValueError("Archived affinity requires native width128 and a fixed mode")
        uniform = targets["uniform_weights"]
        if uniform.shape != (rows, rows):
            raise ValueError("Aligned all-classmate target matrix required")
        if mode != "weighted_assigned" and context_index is not None:
            raise ValueError("Only assigned affinity bodies accept a context index")
        if mode == "weighted_mixture":
            pairs = torch.stack([complete(weights[c][None], allowed[None]) for c in range(4)], dim=-1)
            routes = pairs.mean(-1)
        elif mode == "weighted_assigned":
            if type(context_index) is not int or members != 1 or not 0 <= context_index < 4:
                raise ValueError("Assigned context index must lie in0..3")
            routes = complete(weights[context_index][None], allowed[None])
            pairs = routes[:, None]
        else:
            if members != 4:
                raise ValueError("Private/uniform affinity allocation requires four routes")
            routes = complete(uniform[None] if mode == "uniform_allclassmate" else weights, allowed[None])
            pairs = routes[:, None]
    else:
        raise ValueError("Only archived positive-exclusion/continuous-affinity sources supported")
    _finite(torch, routes, "context auxiliary")
    return routes.mean(), routes, pairs
