"""Objective-specific centered coefficient gradients; no optimizer or selector."""
import itertools

import torch
from torch.nn import functional as F

from spectral_privacy.v2_abi_fix_01.collector import PreservedProbe, unit_fingerprint
from spectral_privacy.v2_abi_fix_01.contracts import pack_float64, unpack_float64

GROUPS = ("selected", "random", "raw_gradient")
OBJECTIVES = ("mean_member_bce", "pooled_logit_bce")
TARGETS = tuple(f"residual_modules.{i}.module.conv.lin_l.weight" for i in range(3))
TOLERANCE = {"float32_chain_atol": 2e-6, "float32_chain_rtol": 2e-5,
             "float64_chain_atol": 1e-10, "float64_chain_rtol": 1e-8,
             "float32_null_max_abs": 2e-6, "float64_null_max_abs": 1e-10}
NUMERIC_RULE = {
    "source_forward": "original source and adapter dtype, CPU eval; no conversion of checkpoints",
    "pooling": "native-dtype arithmetic mean of raw member logits before BCE",
    "BCE_elementwise": "native source dtype; binary_cross_entropy_with_logits reduction=none",
    "loss_reductions": "promote elementwise BCE to float64 before observed-entry/member means",
    "gradients": "native parameter autograd promoted to float64 for all diagnostics",
    "graph_weighting": "each whole graph gets equal weight; graph gradients remain separate",
    "mean_member_scaling": "mean of four own-member graph-mean BCE losses; includes 1/M",
    "pooled_response_scaling": "H_m=M times pooled-loss branch-m source-matrix gradient",
}


def observation_mask(unit, logits):
    if unit.split != "train" or logits.shape != unit.targets.shape:
        raise ValueError("Training-only matching target/logit shapes required")
    observed = torch.isfinite(unit.targets)
    mask = unit.observed_mask
    if mask is None or mask.dtype != torch.bool or mask.device != unit.targets.device:
        raise ValueError("Explicit boolean observed-node mask is required")
    if mask.shape == unit.targets.shape:
        visible_mask = mask
        observed &= mask
    elif mask.ndim == 1 and len(mask) == len(unit.targets):
        visible_mask = mask.reshape((-1,) + (1,) * (unit.targets.ndim - 1)).expand_as(unit.targets)
        observed &= visible_mask
    else:
        raise ValueError("Bad observed-mask shape")
    if bool(torch.isfinite(unit.targets[~visible_mask]).any()):
        raise ValueError("Unobserved training targets must be physically NaN")
    if not int(observed.sum()):
        raise ValueError("No observed labels in training graph")
    target = unit.targets[observed]
    if not bool(((target == 0) | (target == 1)).all()):
        raise ValueError("Observed labels must be binary")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Nonfinite model output")
    return observed


def losses(member_logits, unit):
    if len(member_logits) != 4:
        raise ValueError("Frozen four-member diagnostic required")
    observed = observation_mask(unit, member_logits[0])
    if any(value.shape != member_logits[0].shape for value in member_logits):
        raise ValueError("Member output shape mismatch")
    stacked = torch.stack(member_logits)
    target = unit.targets[observed].to(stacked.dtype)
    own_entries = F.binary_cross_entropy_with_logits(
        stacked[:, observed], target.unsqueeze(0).expand(4, -1), reduction="none")
    own = own_entries.double().mean(dim=1)
    pooled_logits = stacked.mean(dim=0)  # Exact deployed pooling order and native dtype.
    pooled_entries = F.binary_cross_entropy_with_logits(
        pooled_logits[observed], target, reduction="none")
    return {"mean_member_bce": own.mean(), "pooled_logit_bce": pooled_entries.double().mean()}, own, stacked, observed


def as_double(value):
    result = value.detach().cpu().double().contiguous()
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Nonfinite gradient evidence")
    return result


def collect_one(model, unit, *, matrices=False):
    """Read-only gradients at zero A; rank-zero coefficients are explicit empties."""
    if model.members != 4 or not model.is_tied_at_state():
        raise ValueError("Require original tied source with zero centered coefficients")
    coefficients = model.named_private_coefficients()
    if set(coefficients) != set(TARGETS):
        raise ValueError("Require exactly the three frozen neighbor targets")
    active = tuple(key for key in TARGETS if coefficients[key].numel())
    before_inputs = unit_fingerprint(unit)
    snapshot = PreservedProbe(model)
    old_modes = [module.training for module, _ in snapshot.modes]
    try:
        model.eval()
        with torch.enable_grad():
            member_logits = [model.forward_member(unit.graph, unit.features, m) for m in range(4)]
            objectives, own, stacked, observed = losses(member_logits, unit)
            gradients = {}
            for name in OBJECTIVES:
                values = (torch.autograd.grad(objectives[name], tuple(coefficients[key] for key in active),
                                              retain_graph=True, allow_unused=False) if active else ())
                by_key = dict(zip(active, values))
                gradients[name] = {key: (as_double(by_key[key]) if key in by_key else
                                         torch.empty(coefficients[key].shape, dtype=torch.float64))
                                   for key in TARGETS}
            source_matrices = None
            if matrices:
                targets = model.named_neighbor_matrices()
                source_parameters = tuple(targets[key] for key in TARGETS)
                own_rows = {key: [] for key in TARGETS}
                pool_rows = {key: [] for key in TARGETS}
                pooled_output_gradients = torch.autograd.grad(objectives["pooled_logit_bce"],
                                                             tuple(member_logits), retain_graph=True)
                for member in range(4):
                    values = torch.autograd.grad(own[member], source_parameters,
                                                 retain_graph=True, allow_unused=False)
                    for key, value in zip(TARGETS, values):
                        own_rows[key].append(as_double(value))
                    values = torch.autograd.grad(member_logits[member], source_parameters,
                                                 grad_outputs=pooled_output_gradients[member].detach(),
                                                 retain_graph=True, allow_unused=False)
                    for key, value in zip(TARGETS, values):
                        pool_rows[key].append(as_double(4 * value))
                source_matrices = {"mean_member_bce": {key: torch.stack(own_rows[key]) for key in TARGETS},
                                   "pooled_logit_bce": {key: torch.stack(pool_rows[key]) for key in TARGETS}}
        if not snapshot.matches() or before_inputs != unit_fingerprint(unit):
            raise RuntimeError("Gradient collection changed state, grads, RNG, or training inputs")
        return {"coefficient_gradients": gradients, "source_response_matrices": source_matrices,
                "member_logits": stacked.detach().clone(), "observed_entries": int(observed.sum()),
                "losses": {name: float(value.detach()) for name, value in objectives.items()},
                "own_member_losses": own.detach().tolist(),
                "jensen_gap": float(objectives["mean_member_bce"] - objectives["pooled_logit_bce"])}
    finally:
        snapshot.restore()
        if [module.training for module, _ in snapshot.modes] != old_modes:
            raise RuntimeError("Probe modes were not restored")


def projected_chain(source_responses, bases, contrasts):
    return {name: {key: torch.einsum("mq,or,moi->qri", contrasts.double(),
                                    bases[key].double(), source_responses[name][key]) / 4
                   for key in TARGETS} for name in OBJECTIVES}


def check_chain(actual, expected, dtype):
    prefix = "float32" if dtype == torch.float32 else "float64"
    atol, rtol = TOLERANCE[prefix + "_chain_atol"], TOLERANCE[prefix + "_chain_rtol"]
    records = {}
    for name in OBJECTIVES:
        records[name] = {}
        for key in TARGETS:
            left, right = actual[name][key], expected[name][key]
            if left.shape != right.shape or not torch.allclose(left, right, atol=atol, rtol=rtol):
                raise ValueError("Objective-specific coefficient chain rule failed: " + name + " " + key)
            records[name][key] = {"max_abs_error": float((left - right).abs().max()) if left.numel() else 0,
                                  "reference_max_abs": float(right.abs().max()) if right.numel() else 0,
                                  "atol": atol, "rtol": rtol}
    return records


def vector(values, keys=TARGETS):
    return torch.cat([values[key].reshape(-1) for key in keys])


def relation(left, right):
    if left.shape != right.shape:
        raise ValueError("Directions must use matching parameter coordinates")
    l2, r2, inner = left.square().sum(), right.square().sum(), (left * right).sum()
    denominator = torch.sqrt(l2 * r2)
    return {"left_norm2": float(l2), "right_norm2": float(r2), "inner_product": float(inner),
            "cosine": None if not float(denominator) else float(inner / denominator),
            "zero_direction": not float(denominator)}


def lift(coefficients, bases, contrasts):
    return {key: torch.einsum("mq,or,qri->moi", contrasts.double(), bases[key].double(), coefficients[key])
            for key in TARGETS}


def diagnostics(gradients, basis_groups, contrasts):
    """All reductions float64. Cross-basis comparisons use lifted weight tangents."""
    result, lifted = {}, {}
    for group in GROUPS:
        rows = gradients[group]
        lifted[group] = {name: [lift(row, basis_groups[group], contrasts) for row in rows[name]]
                         for name in OBJECTIVES}
        targets = {}
        for tag, keys in [("all_targets", TARGETS)] + [(key, (key,)) for key in TARGETS]:
            own = [vector(row, keys) for row in rows["mean_member_bce"]]
            pool = [vector(row, keys) for row in rows["pooled_logit_bce"]]
            targets[tag] = {
                "capacity_scalars": own[0].numel(),
                "own_vs_pool_by_graph": [relation(own[i], pool[i]) for i in range(2)],
                "own_cross_graph": relation(own[0], own[1]),
                "pool_cross_graph": relation(pool[0], pool[1]),
                "own_graph09_vs_pool_graph18": relation(own[0], pool[1]),
                "pool_graph09_vs_own_graph18": relation(pool[0], own[1]),
                "symmetric_cross_objective_recurrence": float(((own[0] * pool[1]).sum() +
                                                              (pool[0] * own[1]).sum()) / 2),
                "equal_graph_mean_own_vs_pool": relation((own[0] + own[1]) / 2, (pool[0] + pool[1]) / 2),
            }
        result[group] = targets
    cross_basis = {}
    for left, right in itertools.combinations(GROUPS, 2):
        cross_basis[left + "_vs_" + right] = {
            name: [relation(vector(lifted[left][name][i]), vector(lifted[right][name][i])) for i in range(2)]
            for name in OBJECTIVES}
    return {"within_basis_coefficient_directions": result,
            "cross_basis_lifted_weight_directions": cross_basis,
            "cross_basis_coordinate_warning": "Different coefficient bases are compared only after lifting into common centered neighbor-weight space",
            "interpretation": "local coefficient-gradient recurrence and cross-objective alignment; not denoising T, AdamW benefit, or predictive improvement"}


def pack_gradients(gradients):
    return {group: {name: {key: pack_float64(torch.stack([row[key] for row in rows])) for key in TARGETS}
                    for name, rows in values.items()} for group, values in gradients.items()}


def unpack_gradients(payload):
    if set(payload) != set(GROUPS):
        raise ValueError("Fixed basis groups differ")
    result = {}
    for group in GROUPS:
        if set(payload[group]) != set(OBJECTIVES):
            raise ValueError("Objective keys differ")
        result[group] = {}
        for name in OBJECTIVES:
            if set(payload[group][name]) != set(TARGETS):
                raise ValueError("Coefficient target keys differ")
            values = {key: unpack_float64(payload[group][name][key]) for key in TARGETS}
            if any(value.ndim != 4 or value.shape[0] != 2 or value.shape[1] != 3 or value.shape[3] != 128
                   for value in values.values()):
                raise ValueError("Coefficient evidence dimensions differ")
            result[group][name] = [{key: value[i] for key, value in values.items()} for i in range(2)]
    return result
