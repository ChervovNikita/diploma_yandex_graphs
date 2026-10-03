"""Exact member class-probability response geometry; Torch is deferred."""
from dataclasses import dataclass
import math
from typing import Any, Mapping
from .groups import FixedPlan

VIEWS = ("native", "same_removal", "other_removal")
MEMBERS = 4


class ResponseFailure(ValueError):
    pass


def _torch():
    import torch
    return torch


def validate_logits(logits: Mapping[str, Any]):
    torch = _torch()
    if set(logits) != set(VIEWS):
        raise ValueError("Exactly native/same_removal/other_removal logits required")
    z0 = logits["native"]
    if z0.ndim != 3 or z0.shape[0] != MEMBERS or z0.shape[2] < 2:
        raise ValueError("Complete member logits must have shape [4, nodes, classes>=2]")
    for view in VIEWS:
        z = logits[view]
        if z.shape != z0.shape or z.dtype != z0.dtype or z.device != z0.device:
            raise ValueError("All view logits need identical shape/dtype/device")
        if not z.is_floating_point() or not torch.isfinite(z).all().item():
            raise ResponseFailure("Nonfinite or nonfloating logits")


def responses(logits: Mapping[str, Any]):
    """A_m(v)=[p_native-p_same, p_native-p_other], before any serving pool.

    Frozen common/classifier maps remain in the differentiable forward. This
    function neither detaches nor enters no_grad. Softmax removes common logit
    shifts; response coordinates are the identified class probabilities.
    """
    validate_logits(logits)
    torch = _torch()
    p = {v: torch.softmax(logits[v], dim=-1) for v in VIEWS}
    return torch.cat((p["native"] - p["same_removal"],
                      p["native"] - p["other_removal"]), dim=-1)


def centered_statistics(a: Any, plan: FixedPlan, names: tuple[str, ...] | None = None):
    torch = _torch()
    if (a.ndim != 3 or a.shape[0] != MEMBERS or a.shape[-1] < 4 or
            a.shape[-1] % 2 or not a.is_floating_point() or plan.role != "fit"):
        raise ValueError("Fit response shape [4, complete nodes, 2*classes] required")
    if not torch.isfinite(a).all().item():
        raise ResponseFailure("Nonfinite response")
    selected = tuple(name for name, _ in plan.cells) if names is None else names
    if len(set(selected)) != len(selected) or not set(selected) <= dict(plan.cells).keys():
        raise ValueError("Unknown or repeated fixed response group")
    out = {}
    for name in selected:
        node_ids = plan.indices(name)
        if max(node_ids) >= a.shape[1]:
            raise ValueError("Membership exceeds complete-node logits")
        index = torch.tensor(node_ids, dtype=torch.long, device=a.device)
        group = a.index_select(1, index)
        r = (group - group.mean(dim=1, keepdim=True)).flatten(start_dim=1)
        out[name] = (r, torch.linalg.vector_norm(r, dim=1))
    return out


@dataclass(frozen=True)
class ActiveReference:
    plan_fingerprint: str
    names: tuple[str, ...]
    reference_norms: tuple[tuple[str, tuple[float, ...]], ...]
    excluded: tuple[tuple[str, str], ...]
    coverage: float
    warm_function_receipt: str
    floor: float = 1e-5

    def __post_init__(self):
        if not self.warm_function_receipt or self.floor != 1e-5:
            raise ValueError("Copied warm receipt and sealed reference floor required")
        refs = dict(self.reference_norms)
        if not self.names or set(refs) != set(self.names):
            raise ResponseFailure("No active response groups or mismatched references")
        for norms in refs.values():
            if len(norms) != MEMBERS or any(not math.isfinite(x) or x < self.floor for x in norms):
                raise ResponseFailure("Active warm reference fails member floor")
        if not 0.5 <= self.coverage <= 1.0:
            raise ResponseFailure("Active fit coverage below sealed 50% screen floor")

    def validate_plan(self, plan: FixedPlan):
        if plan.fingerprint != self.plan_fingerprint:
            raise ValueError("Group membership/resolution changed after warm reference")


def build_active_reference(a: Any, plan: FixedPlan, *, warm_function_receipt: str):
    """Called once on a copied warm function. Coverage counts unique fit nodes."""
    torch = _torch()
    with torch.no_grad():
        stats = centered_statistics(a, plan)
        refs, excluded, covered = [], [], set()
        for name, (_, norm) in stats.items():
            values = tuple(float(x) for x in norm.detach().cpu().tolist())
            if any(not math.isfinite(x) for x in values):
                raise ResponseFailure("Nonfinite warm reference is a qualification failure")
            if any(x < 1e-5 for x in values):
                excluded.append((name, "at_least_one_member_warm_norm_below_1e-5"))
            else:
                refs.append((name, values))
                covered.update(plan.indices(name))
        coverage = len(covered) / len(plan.target_ids)
        return ActiveReference(plan.fingerprint, tuple(n for n, _ in refs),
                               tuple(refs), tuple(excluded), coverage,
                               warm_function_receipt)


def redundancy(a: Any, plan: FixedPlan, active: ActiveReference):
    """Equal group and unordered member-pair mean of squared normalized dot.

    No epsilon is added. Current zero response fails; warm active sets never
    change in response to a candidate. Mean responses are centered, not repelled.
    Energy feasibility is a separate guard on each active group/member.
    """
    torch = _torch()
    active.validate_plan(plan)
    penalties = []
    for _, (r, norm) in centered_statistics(a, plan, active.names).items():
        if not torch.isfinite(norm).all().item() or (norm <= 0).any().item():
            raise ResponseFailure("Current active response norm is zero/nonfinite")
        u = r / norm[:, None]
        pairs = [(u[m] * u[n]).sum().square()
                 for m in range(MEMBERS) for n in range(m + 1, MEMBERS)]
        penalties.append(torch.stack(pairs).mean())
    return torch.stack(penalties).mean()


def energy_failures(a: Any, plan: FixedPlan, active: ActiveReference) -> tuple[str, ...]:
    """Inclusive [0.5,2.0] warm norm band, no tolerance relaxation."""
    torch = _torch()
    active.validate_plan(plan)
    failures = []
    with torch.no_grad():
        for name, (_, norm) in centered_statistics(a, plan, active.names).items():
            ref = torch.tensor(dict(active.reference_norms)[name], dtype=norm.dtype, device=norm.device)
            bad = ~torch.isfinite(norm) | (norm <= 0) | (norm < 0.5 * ref) | (norm > 2.0 * ref)
            for m in bad.nonzero(as_tuple=False).flatten().tolist():
                failures.append(f"energy:{name}:member:{m}")
    return tuple(failures)


def mean_member_ce(logits: Any, roles: Any, role: str = "fit"):
    torch = _torch()
    indices = torch.tensor(roles.targets(role), dtype=torch.long, device=logits.device)
    labels = torch.tensor(roles.labels_for(role), dtype=torch.long, device=logits.device)
    if logits.ndim != 3 or logits.shape[0] != MEMBERS or logits.shape[-1] != roles.class_count:
        raise ValueError("CE requires complete [4,nodes,declared_classes] logits")
    selected = logits.index_select(1, indices)
    return torch.nn.functional.cross_entropy(selected.reshape(-1, roles.class_count),
                                            labels.repeat(MEMBERS), reduction="mean")


def candidate_objective(logits: Mapping[str, Any], roles: Any,
                        plan: FixedPlan, active: ActiveReference):
    """Native mean member CE + mean two-probe member CE + sealed 0.1 D."""
    validate_logits(logits)
    if plan.target_ids != roles.fit:
        raise ValueError("Objective fit targets differ from the fixed response plan")
    native = mean_member_ce(logits["native"], roles)
    probes = (mean_member_ce(logits["same_removal"], roles) +
              mean_member_ce(logits["other_removal"], roles)) / 2
    d = redundancy(responses(logits), plan, active)
    return native + probes + 0.1 * d, {"native_ce": native, "probe_ce": probes, "D": d}
