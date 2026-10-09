"""Inactive source-supply kernels. No model, data, optimizer or runner is constructed.

PyTorch is imported lazily only after an explicit enabled configuration. The
stdlib analytical kernels are independently usable for synthetic verification.
Native source views, replay tokens, architecture and data roles remain caller
responsibilities; declarations below are not proof of native view correctness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import exp, fsum, hypot, isfinite, log, log1p
from typing import Any, Callable, Mapping, Sequence


class ContractError(RuntimeError):
    pass


class NonfiniteScores(ContractError):
    pass


@dataclass(frozen=True)
class Config:
    enabled: bool = False
    # Bernoulli is a prospective proper-score amendment, never an inferred mode.
    score_mode: str = "categorical_logits"
    armijo: float = 1e-4
    correction_fraction: float = 0.1
    guard_atol: float = 0.0
    replay_atol: float = 0.0
    logp_atol: float = 5e-5
    cone_tol: float = 1e-9

    def require_enabled(self) -> None:
        if not self.enabled:
            raise ContractError("Inactive helper: explicit root-qualified enablement required")
        if self.score_mode not in {
            "categorical_logits", "categorical_log_probs", "bernoulli_marginal_logits"
        }:
            raise ContractError("Unknown explicit score mode")
        for value in (self.guard_atol, self.replay_atol, self.logp_atol, self.cone_tol):
            if not isfinite(value) or value < 0:
                raise ContractError("Invalid fixed numerical tolerance")
        if not 0 < self.armijo < 1 or not 0 < self.correction_fraction <= 1:
            raise ContractError("Invalid fixed update constants")


def _torch(config: Config) -> Any:
    config.require_enabled()
    import torch  # Deliberately absent from import-time and stdlib verification.
    return torch


@dataclass(frozen=True)
class OutputKey:
    kind: str  # train_full, train_probe, eval_full
    member: int
    source: str | None = None


@dataclass(frozen=True)
class SourceView:
    source: str
    binding: str
    removed_families: tuple[str, ...]
    required_reverse_relations_removed: bool
    other_families_retained: bool
    query_own_features_retained: bool
    native_support_and_paths_rebuilt: bool

    def validate(self) -> None:
        if not self.source or not self.binding or self.removed_families != (self.source,):
            raise ContractError("A probe must declare exactly its own removed family")
        if not all((self.required_reverse_relations_removed,
                    self.other_families_retained, self.query_own_features_retained,
                    self.native_support_and_paths_rebuilt)):
            raise ContractError("Incomplete native removed-family view contract")


def _unique(values: Sequence[Any]) -> tuple[Any, ...]:
    result, seen = [], set()
    for value in values:
        if id(value) not in seen:
            seen.add(id(value))
            result.append(value)
    return tuple(result)


def _storage_key(parameter: Any) -> tuple[str, int]:
    if parameter.numel() == 0 or str(parameter.device) == "meta":
        raise ContractError("Ownership requires material nonempty parameters")
    return str(parameter.device), int(parameter.untyped_storage().data_ptr())


@dataclass
class Ownership:
    shared: tuple[Any, ...]
    private: tuple[tuple[Any, ...], ...]
    all_parameters: tuple[Any, ...]
    member_owned_nonsteered: tuple[tuple[Any, ...], ...]

    def stamp(self) -> tuple[tuple[int, int, bool], ...]:
        return tuple((id(p), int(p._version), bool(p.requires_grad))
                     for p in self.all_parameters)

    def grad_stamp(self) -> tuple[Any, ...]:
        return tuple(None if p.grad is None else (id(p.grad), int(p.grad._version))
                     for p in self.all_parameters)


def verify_ownership(members: Sequence[Any], shared_names: Sequence[str],
                     private_names: Sequence[str],
                     member_owned_nonsteered_names: Sequence[str] = (),
                     *, require_shared: bool = True) -> Ownership:
    """Complete name/identity/storage declaration; supports a genuine untied control.

    In the shared candidate every shared site must reference the same object.
    Untied bodies declare native weights as member-owned/nonsteered, rather than
    pretending a shared implementation is an independent ensemble.
    """
    if len(members) < 2 or not private_names or (require_shared and not shared_names):
        raise ContractError("Need a complete committee and declared private/shared sites")
    groups = [set(shared_names), set(private_names), set(member_owned_nonsteered_names)]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ContractError("Parameter ownership declarations overlap")
    declared = set().union(*groups)
    maps = []
    for member in members:
        named = dict(member.named_parameters(remove_duplicate=False))
        if set(named) != declared:
            raise ContractError("Every native parameter path must have explicit ownership")
        maps.append(named)
    for name in shared_names:
        if any(m[name] is not maps[0][name] for m in maps[1:]):
            raise ContractError("Shared names are not genuinely shared parameter objects")
    shared = _unique([maps[0][name] for name in shared_names])
    private = tuple(_unique([m[name] for name in private_names]) for m in maps)
    nonsteered = tuple(_unique([m[name] for name in member_owned_nonsteered_names])
                       for m in maps)
    shared_ids, shared_storage = {id(p) for p in shared}, {_storage_key(p) for p in shared}
    if len(shared_storage) != len(shared):
        raise ContractError("Distinct shared parameter objects alias one storage")
    owned_ids, owned_storage = set(), set()
    for private_block, other_block in zip(private, nonsteered):
        if {id(p) for p in private_block} & {id(p) for p in other_block}:
            raise ContractError("A private recipient aliases a nonsteered native weight")
        block = _unique(private_block + other_block)
        ids, storage = {id(p) for p in block}, {_storage_key(p) for p in block}
        if ids & (shared_ids | owned_ids) or storage & (shared_storage | owned_storage):
            raise ContractError("Member-owned parameters/storage overlap another owner")
        if len(storage) != len(block):
            raise ContractError("Distinct member-owned objects alias one storage")
        owned_ids |= ids
        owned_storage |= storage
    all_parameters = _unique([p for m in maps for p in m.values()])
    return Ownership(shared, private, all_parameters, nonsteered)


def _observed_logp(scores: Any, targets: Any, config: Config) -> Any:
    torch = _torch(config)
    if scores.ndim != 2 or not scores.is_floating_point() or min(scores.shape) < 1:
        raise ContractError("Native scores require a nonempty floating query-by-output tensor")
    if not torch.isfinite(scores).all().item():
        raise NonfiniteScores("Nonfinite native scores")
    if targets.device != scores.device:
        raise ContractError("Targets and native scores must use the same device")
    if config.score_mode == "bernoulli_marginal_logits":
        if targets.shape != scores.shape or not torch.isfinite(targets).all().item():
            raise ContractError("Bernoulli targets require a complete binary entry matrix")
        if not ((targets == 0) | (targets == 1)).all().item():
            raise ContractError("Bernoulli amendment requires observed hard binary outcomes")
        # Each entry remains separate until the final mean. No joint-label product.
        return -torch.nn.functional.binary_cross_entropy_with_logits(
            scores, targets.to(dtype=scores.dtype), reduction="none")
    if targets.ndim != 1 or targets.shape[0] != scores.shape[0] or targets.dtype != torch.long:
        raise ContractError("Categorical targets require one integer class per query")
    if not ((targets >= 0) & (targets < scores.shape[1])).all().item():
        raise ContractError("Categorical class index outside native output")
    if config.score_mode == "categorical_log_probs":
        if torch.logsumexp(scores, dim=1).abs().max().item() > config.logp_atol:
            raise ContractError("Declared native log probabilities are not normalized")
        logp = scores
    else:
        logp = torch.log_softmax(scores, dim=1)
    return logp.gather(1, targets[:, None]).squeeze(1)


def _mean_pool_nll(observed: Sequence[Any], config: Config) -> Any:
    torch = _torch(config)
    # For Bernoulli observed has [member,query,label], not summed labels.
    return -(torch.logsumexp(torch.stack(tuple(observed), dim=0), dim=0)
             - log(len(observed))).mean()


def _terms(outputs: Mapping[OutputKey, Any], peers: Mapping[OutputKey, Any],
           targets: Any, assignments: Sequence[str | None], config: Config) -> dict[str, Any]:
    count, terms, js = len(assignments), {}, []
    evaluated = []
    for member in range(count):
        value = _observed_logp(outputs[OutputKey("eval_full", member)], targets, config)
        terms[f"full_own:{member}"] = -value.mean()
        evaluated.append(value)
    terms["full_pool"] = _mean_pool_nll(evaluated, config)
    for member, source in enumerate(assignments):
        if source is None:
            continue
        factual = _observed_logp(outputs[OutputKey("train_full", member)], targets, config)
        probe = _observed_logp(outputs[OutputKey("train_probe", member, source)], targets, config)
        # All peers lose THIS recipient's source, irrespective of their assignments.
        others = [_observed_logp(peers[OutputKey("train_probe", k, source)].detach(),
                                  targets, config) for k in range(count) if k != member]
        supplied = _mean_pool_nll([factual, *others], config)
        absent = _mean_pool_nll([probe, *others], config)
        terms[f"supply:{member}"] = supplied
        terms[f"absent:{member}"] = absent
        terms[f"probe_own:{member}"] = -probe.mean()
        terms[f"source_j:{member}"] = supplied - absent
        js.append(supplied - absent)
    if not js:
        raise ContractError("At least one active recipient is required")
    terms["J"] = sum(js) / len(js)
    return terms


@dataclass
class CreditPlan:
    config: Config
    ownership: Ownership
    assignments: tuple[str | None, ...]
    views: Mapping[str, SourceView]
    full_view_binding: str
    targets: Any
    outputs: dict[OutputKey, Any]
    tokens: Mapping[OutputKey, Any]
    values: dict[str, float]
    risks: tuple[str, ...]
    cotangents: dict[str, dict[OutputKey, Any]]
    expected_stamp: tuple[Any, ...]
    origin_private: tuple[tuple[Any, ...], ...]
    gradients: dict[str, dict[int, list[Any]]] = field(default_factory=dict)
    replayed: set[OutputKey] = field(default_factory=set)
    consumed: bool = False

    def active(self) -> tuple[int, ...]:
        return tuple(m for m, source in enumerate(self.assignments) if source is not None)

    def require_current(self) -> None:
        self.config.require_enabled()
        if self.consumed or self.ownership.stamp() != self.expected_stamp:
            raise ContractError("Stale or consumed reference: parameters/ownership changed")

    def replay_keys(self) -> tuple[OutputKey, ...]:
        m = len(self.assignments)
        return (tuple(OutputKey("train_full", k) for k in range(m))
                + tuple(OutputKey("train_probe", k, self.assignments[k]) for k in self.active())
                + tuple(OutputKey("eval_full", k) for k in range(m)))


def make_credit_plan(config: Config, ownership: Ownership,
                     assignments: Sequence[str | None], views: Mapping[str, SourceView],
                     full_view_binding: str, outputs: Mapping[OutputKey, Any],
                     targets: Any, replay_tokens: Mapping[OutputKey, Any],
                     *, target_role: str) -> CreditPlan:
    """Consume no-grad old-state native scores, creating only output-leaf tapes."""
    torch = _torch(config)
    assignment = tuple(assignments)
    if target_role != "TRAIN" or len(assignment) != len(ownership.private) or not full_view_binding:
        raise ContractError("Require explicit TRAIN role and complete full-view ownership")
    assigned = set(source for source in assignment if source is not None)
    sources = set(views)  # Retain every family for equal COMMON/reference opportunity.
    if not assigned or not assigned <= sources:
        raise ContractError("Missing assigned native source view")
    for source in sources:
        views[source].validate()
        if views[source].source != source or views[source].binding == full_view_binding:
            raise ContractError("Probe binding mismatches its source or factual view")
    count = len(assignment)
    keys = ({OutputKey("train_full", m) for m in range(count)}
            | {OutputKey("eval_full", m) for m in range(count)}
            | {OutputKey("train_probe", m, a) for a in sources for m in range(count)})
    if set(outputs) != keys or set(replay_tokens) != keys:
        raise ContractError("Reference requires every peer under every assigned source")
    native = {}
    shape, device, dtype = None, None, None
    for key in keys:
        score = outputs[key]
        if score.requires_grad or score.grad_fn is not None:
            raise ContractError("Reference must not retain a native model tape")
        _observed_logp(score, targets, config)
        if shape is None:
            shape, device, dtype = score.shape, score.device, score.dtype
        if (score.shape, score.device, score.dtype) != (shape, device, dtype):
            raise ContractError("Reference scores must share query/output order, dtype and device")
        native[key] = score.detach().clone()
    target_copy = targets.detach().clone()
    leaves = {key: score.clone().requires_grad_(True) for key, score in native.items()}
    terms = _terms(leaves, native, target_copy, assignment, config)
    risks = tuple([f"full_own:{m}" for m in range(count)] + ["full_pool"]
                  + [f"probe_own:{m}" for m, a in enumerate(assignment) if a is not None]
                  + [f"absent:{m}" for m, a in enumerate(assignment) if a is not None])
    if len(risks) > 11:
        raise ContractError("Declared bounded cone has more than11 risk constraints")
    order, labels = tuple(sorted(keys, key=lambda k: (k.kind, k.member, k.source or ""))), ("J", *risks)
    credits = {}
    for i, label in enumerate(labels):
        derivatives = torch.autograd.grad(terms[label], [leaves[k] for k in order],
                                          allow_unused=True, retain_graph=i < len(labels)-1)
        credits[label] = {key: derivative.detach().clone() for key, derivative in zip(order, derivatives)
                          if derivative is not None}
    values = {name: float(value.detach().item()) for name, value in terms.items()}
    if not all(isfinite(value) for value in values.values()):
        raise ContractError("Nonfinite reference proper score")
    return CreditPlan(config, ownership, assignment, dict(views), full_view_binding,
                      target_copy, native, dict(replay_tokens), values, risks, credits,
                      ownership.stamp(), tuple(tuple(p.detach().clone() for p in block)
                                               for block in ownership.private))


def accumulate_replay(plan: CreditPlan, key: OutputKey, native_scores: Any) -> None:
    """One native path at a time; VJPs never populate shared/private .grad fields."""
    torch = _torch(plan.config)
    plan.require_current()
    if key not in plan.replay_keys() or key in plan.replayed:
        raise ContractError("Unexpected or repeated native replay")
    reference = plan.outputs[key]
    if native_scores.shape != reference.shape or not torch.isfinite(native_scores).all().item():
        raise ContractError("Invalid replay output")
    if not torch.allclose(native_scores.detach(), reference, rtol=0, atol=plan.config.replay_atol):
        raise ContractError("Replay differs from credited old-state output")
    before = plan.ownership.grad_stamp()
    eligible = plan.assignments[key.member] is not None
    labels = [label for label, credit in plan.cotangents.items() if key in credit] if eligible else []
    parameters = plan.ownership.private[key.member]
    if labels and any(not p.requires_grad for p in parameters):
        raise ContractError("Private recipient is not differentiable")
    if labels and (not torch.is_grad_enabled() or not native_scores.requires_grad):
        raise ContractError("Native replay must construct its differentiable tape")
    for i, label in enumerate(labels):
        derivatives = torch.autograd.grad(native_scores, parameters,
                                          grad_outputs=plan.cotangents[label][key],
                                          allow_unused=True, retain_graph=i < len(labels)-1)
        buffers = plan.gradients.setdefault(label, {}).setdefault(
            key.member, [torch.zeros_like(p) for p in parameters])
        with torch.no_grad():
            for buffer, derivative in zip(buffers, derivatives):
                if derivative is not None:
                    if not torch.isfinite(derivative).all().item():
                        raise ContractError("Nonfinite private derivative")
                    buffer.add_(derivative.detach())
    if plan.ownership.grad_stamp() != before:
        raise ContractError("Native backward modified parameter .grad fields")
    plan.require_current()
    plan.replayed.add(key)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ContractError("Dot-product shape mismatch")
    try:
        value = fsum(x*y for x, y in zip(a, b))
    except (OverflowError, ValueError) as error:
        raise ContractError("Nonfinite dot-product arithmetic") from error
    if not isfinite(value):
        raise ContractError("Nonfinite dot-product result")
    return value


def validate_descent_slope(value: float) -> None:
    if not isfinite(value) or value >= 0:
        raise ContractError("A nonzero step requires a finite strictly negative slope")


def _solve(matrix: Sequence[Sequence[float]], rhs: Sequence[float], tol: float) -> list[float] | None:
    size = len(rhs)
    rows = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]
    if not all(isfinite(value) for row in rows for value in row):
        raise ContractError("Nonfinite cone linear system")
    for column in range(size):
        pivot = max(range(column, size), key=lambda i: abs(rows[i][column]))
        if abs(rows[pivot][column]) <= tol:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        value = rows[column][column]
        rows[column] = [x/value for x in rows[column]]
        for row in range(size):
            if row != column:
                value = rows[row][column]
                rows[row] = [x-value*y for x, y in zip(rows[row], rows[column])]
        if not all(isfinite(value) for row in rows for value in row):
            raise ContractError("Nonfinite cone elimination intermediate")
    return [rows[i][-1] for i in range(size)]


def project_nonincrease_cone(target: Sequence[float], risks: Sequence[Sequence[float]],
                             tol: float = 1e-9) -> tuple[float, ...]:
    """Same normalized cone projection; active subsets use one cached Gram.

    Enumerate at most11 rows with the unchanged _solve/pivot tolerance. Rank and
    test candidates in Gram space, then reconstruct the chosen full vector once.
    Any numerical inconsistency fails explicitly; never retolerate or return an
    invented zero direction. Config, gradient pullback and finite-step guards
    remain unchanged from the frozen predecessor.
    """
    if not target or len(risks) > 11 or not isfinite(tol) or tol <= 0:
        raise ContractError("Invalid bounded cone problem")
    if not all(isfinite(x) for x in target):
        raise ContractError("Nonfinite projection target")
    rows = []
    for row in risks:
        if len(row) != len(target) or not all(isfinite(x) for x in row):
            raise ContractError("Invalid constraint gradient")
        norm = hypot(*row)
        if not isfinite(norm):
            raise ContractError("Nonfinite constraint norm")
        if norm:
            normalized = tuple(x/norm for x in row)
            if not all(isfinite(x) for x in normalized):
                raise ContractError("Nonfinite normalized constraint")
            rows.append(normalized)

    count = len(rows)
    gram = [[0.0] * count for _ in range(count)]
    for i, a in enumerate(rows):
        for j in range(i, count):
            gram[i][j] = gram[j][i] = _dot(a, rows[j])
    target_inner = [_dot(a, target) for a in rows]
    best_indices, best_multipliers, best_distance = None, None, float("inf")
    for size in range(count+1):
        for indices in combinations(range(count), size):
            active_gram = [[gram[i][j] for j in indices] for i in indices]
            multipliers = _solve(active_gram, [target_inner[i] for i in indices], tol) if size else []
            if multipliers is None or any(x < -tol for x in multipliers):
                continue
            residuals = [target_inner[i] - _dot([gram[i][j] for j in indices], multipliers)
                         for i in range(count)]
            if not all(isfinite(x) for x in residuals):
                raise ContractError("Nonfinite Gram-space feasibility arithmetic")
            if any(x > tol for x in residuals):
                continue
            normal_inner = [_dot(row, multipliers) for row in active_gram]
            distance = _dot(multipliers, normal_inner)
            if distance < 0:
                raise ContractError("Negative Gram-space squared distance; no roundoff clamp")
            if distance < best_distance:
                best_indices, best_multipliers, best_distance = indices, multipliers, distance
    if best_indices is None:
        raise ContractError("Cone solver failed; no zero-direction substitution")

    active = [rows[i] for i in best_indices]
    value = tuple(x-fsum(lam*a[j] for lam, a in zip(best_multipliers, active))
                  for j, x in enumerate(target))
    if not all(isfinite(x) for x in value):
        raise ContractError("Nonfinite projected direction")
    if any(_dot(row, value) > tol for row in rows):
        raise ContractError("Gram/full-space feasibility disagreement; no fallback or retolerance")
    difference = [x-y for x, y in zip(value, target)]
    _dot(difference, difference)  # Preserve the original full-space finite-distance guard once.
    return value


@dataclass(frozen=True)
class Direction:
    by_member: Mapping[int, tuple[Any, ...]]
    g_dot_d: float
    zero: bool
    reference_identity: int


def build_private_direction(plan: CreditPlan,
                            own_displacements: Mapping[int, Sequence[Any]]) -> Direction:
    torch = _torch(plan.config)
    plan.require_current()
    if plan.replayed != set(plan.replay_keys()):
        raise ContractError("Complete audited one-path replay is required")
    active = plan.active()
    def blocks(label: str) -> list[Any]:
        return [plan.gradients.get(label, {}).get(m, [torch.zeros_like(p) for p in plan.ownership.private[m]])[i]
                for m in active for i in range(len(plan.ownership.private[m]))]
    def flatten(values: Sequence[Any]) -> list[float]:
        return torch.cat([value.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
                          for value in values]).tolist()
    own = []
    for m in active:
        values = own_displacements[m]
        if len(values) != len(plan.ownership.private[m]):
            raise ContractError("Own-step private displacement scope mismatch")
        for value, parameter in zip(values, plan.ownership.private[m]):
            if value.shape != parameter.shape:
                raise ContractError("Own-step displacement shape mismatch")
            own.append(value)
    gradient, own_flat = flatten(blocks("J")), flatten(own)
    if not all(isfinite(x) for x in own_flat):
        raise ContractError("Nonfinite own displacement")
    own_norm = hypot(*own_flat)
    if not isfinite(own_norm):
        raise ContractError("Nonfinite own-displacement norm")
    projected = project_nonincrease_cone([-x for x in gradient],
                                        [flatten(blocks(label)) for label in plan.risks], plan.config.cone_tol)
    norm = hypot(*projected)
    if not isfinite(norm):
        raise ContractError("Nonfinite projected norm")
    scale = plan.config.correction_fraction * own_norm / norm if norm and own_norm else 0.0
    if not isfinite(scale):
        raise ContractError("Nonfinite correction scale")
    flat = [scale*x for x in projected]
    if not all(isfinite(x) for x in flat):
        raise ContractError("Nonfinite proposed displacement")
    result, offset = {}, 0
    for m in active:
        values = []
        for parameter in plan.ownership.private[m]:
            size = parameter.numel()
            values.append(torch.tensor(flat[offset:offset+size], dtype=parameter.dtype,
                                       device=parameter.device).reshape(parameter.shape))
            offset += size
        result[m] = tuple(values)
    # Armijo uses the displacement actually representable in native parameter dtype.
    actual = flatten([value for m in active for value in result[m]])
    if not all(isfinite(x) for x in actual):
        raise ContractError("Native dtype cannot represent proposed displacement")
    dot = _dot(gradient, actual)
    if any(actual):
        validate_descent_slope(dot)
    return Direction(result, dot, not any(actual), id(plan))


@dataclass(frozen=True)
class GuardDecision:
    accepted: bool
    violations: tuple[str, ...]
    values: Mapping[str, float]


def check_guard_values(reference: Mapping[str, float], trial: Mapping[str, float],
                       risk_names: Sequence[str], recipient_names: Sequence[str],
                       *, armijo_rhs: float, atol: float = 0.0) -> GuardDecision:
    """Pure scalar guard, also usable in the stdlib analytical audit."""
    needed = set(risk_names) | set(recipient_names) | {"J"}
    if not needed <= set(reference) or not needed <= set(trial) or not isfinite(atol) or atol < 0:
        raise ContractError("Incomplete guard values")
    if not all(isfinite(reference[name]) for name in needed):
        raise ContractError("Nonfinite reference guard values")
    violations = []
    if not isfinite(armijo_rhs) or not all(isfinite(trial[name]) for name in needed):
        violations.append("nonfinite")
    else:
        if trial["J"] > armijo_rhs + atol:
            violations.append("Armijo:J")
        violations.extend(name for name in risk_names if trial[name] > reference[name] + atol)
        violations.extend(name for name in recipient_names if trial[name] > reference[name] + atol)
    return GuardDecision(not violations, tuple(violations), dict(trial))


def try_private_step(plan: CreditPlan, direction: Direction, trial_scale: float,
                     native_forward: Callable[[int, str | None, str, Any], Any]) -> GuardDecision:
    """One finite trial only. Caller supplies native forward/view/RNG replay.

    No own optimizer, training loop, retry schedule or checkpoint selector exists
    here. Only private recipients are temporarily copied; a failed trial restores
    exact old private values. Accepted steps consume the reference. Optimizer
    objects/moments are neither received nor modified.
    """
    torch = _torch(plan.config)
    plan.require_current()
    if direction.zero or trial_scale not in (1.0, 0.5, 0.25, 0.125):
        raise ContractError("Zero correction or undeclared trial scale")
    if (direction.reference_identity != id(plan)
            or set(direction.by_member) != set(plan.active())):
        raise ContractError("Invalid private direction scope")
    validate_descent_slope(direction.g_dot_d)
    if any(len(direction.by_member[m]) != len(plan.ownership.private[m]) for m in plan.active()):
        raise ContractError("Incomplete private candidate")
    before_grad = plan.ownership.grad_stamp()
    accepted, decision, fatal = False, None, False
    try:
        with torch.no_grad():
            for m in plan.active():
                for p, old, d in zip(plan.ownership.private[m], plan.origin_private[m], direction.by_member[m]):
                    if p.shape != d.shape or p.device != d.device or p.dtype != d.dtype:
                        raise ContractError("Candidate parameter representation mismatch")
                    if not torch.isfinite(d).all().item():
                        raise ContractError("Nonfinite private candidate displacement")
                    p.copy_(old + trial_scale*d)
            candidate_stamp = plan.ownership.stamp()
            trial_outputs = {}
            for key in plan.replay_keys():
                mode = "eval" if key.kind == "eval_full" else "train"
                score = native_forward(key.member, key.source, mode, plan.tokens[key])
                if score.requires_grad or score.grad_fn is not None:
                    raise ContractError("Trial must not retain native model tapes")
                reference = plan.outputs[key]
                if (score.shape, score.device, score.dtype) != (reference.shape, reference.device, reference.dtype):
                    raise ContractError("Trial query/output representation changed")
                _observed_logp(score, plan.targets, plan.config)
                trial_outputs[key] = score.detach().clone()
            if plan.ownership.stamp() != candidate_stamp or plan.ownership.grad_stamp() != before_grad:
                raise ContractError("Native forward mutated parameters or .grad fields")
            # Peers always come from the original frozen reference, never trial probes.
            terms = _terms(trial_outputs, plan.outputs, plan.targets, plan.assignments, plan.config)
            values = {name: float(value.item()) for name, value in terms.items()}
            decision = check_guard_values(plan.values, values, plan.risks,
                                          [f"source_j:{m}" for m in plan.active()],
                                          armijo_rhs=plan.values["J"] + plan.config.armijo*trial_scale*direction.g_dot_d,
                                          atol=plan.config.guard_atol)
            accepted = decision.accepted
    except NonfiniteScores:
        if plan.ownership.stamp() != candidate_stamp or plan.ownership.grad_stamp() != before_grad:
            fatal = True
            plan.consumed = True
            raise ContractError("Invalid native trial also mutated parameters or .grad fields")
        decision = GuardDecision(False, ("nonfinite_native_trial",), {})
    except BaseException:
        fatal = True
        plan.consumed = True
        raise
    finally:
        if not accepted:
            with torch.no_grad():
                for m in plan.active():
                    for p, old in zip(plan.ownership.private[m], plan.origin_private[m]):
                        p.copy_(old)
                        if not torch.equal(p, old):
                            raise ContractError("Failed exact private rollback")
            # copy_ increments private versions even after value-exact rollback.
            if not fatal:
                plan.expected_stamp = plan.ownership.stamp()
        else:
            plan.consumed = True
    if decision is None:
        raise ContractError("Trial did not produce a guard decision")
    return decision


def full_input_probability_pool(config: Config, full_scores: Sequence[Any]) -> Any:
    """Serving helper accepts ONLY factual scores; source-supply mixtures are absent."""
    torch = _torch(config)
    if not full_scores:
        raise ContractError("Empty full-input serving committee")
    first = full_scores[0]
    if first.ndim != 2 or min(first.shape) < 1:
        raise ContractError("Invalid full-input serving shape")
    for value in full_scores:
        if (value.shape, value.dtype, value.device) != (first.shape, first.dtype, first.device):
            raise ContractError("Inconsistent full-input serving query/output representation")
        if not torch.isfinite(value).all().item():
            raise NonfiniteScores("Nonfinite full-input native scores")
        if (config.score_mode == "categorical_log_probs"
                and torch.logsumexp(value, dim=1).abs().max().item() > config.logp_atol):
            raise ContractError("Full-input native log probabilities are not normalized")
    if config.score_mode == "bernoulli_marginal_logits":
        probabilities = [torch.sigmoid(value) for value in full_scores]
    elif config.score_mode == "categorical_log_probs":
        probabilities = [value.exp() for value in full_scores]
    else:
        probabilities = [torch.softmax(value, dim=1) for value in full_scores]
    if not all(torch.isfinite(value).all().item() for value in probabilities):
        raise ContractError("Nonfinite full-input serving")
    return torch.stack(probabilities, dim=0).mean(dim=0)


def serve_full_input(config: Config, ownership: Ownership,
                     native_forward: Callable[[int, str | None, str, Any], Any],
                     eval_tokens: Sequence[Any], *, full_view_binding: str) -> tuple[Any, tuple[Any, ...]]:
    """Explicit native all-full-input serving; never calls a removed-family view."""
    torch = _torch(config)
    if not full_view_binding or len(eval_tokens) != len(ownership.private):
        raise ContractError("Serving needs all complete factual native paths")
    stamp, gradients = ownership.stamp(), ownership.grad_stamp()
    with torch.no_grad():
        scores = tuple(native_forward(m, None, "eval", eval_tokens[m]).detach().clone()
                       for m in range(len(ownership.private)))
        if ownership.stamp() != stamp or ownership.grad_stamp() != gradients:
            raise ContractError("Serving modified learned parameters or .grad fields")
        return full_input_probability_pool(config, scores), scores


def _lse(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + log(fsum(exp(x-maximum) for x in values))


def _logsigmoid(value: float) -> float:
    return -log1p(exp(-value)) if value >= 0 else value-log1p(exp(value))


def analytic_source_supply(factual: Sequence[float], probe: Sequence[float],
                          frozen_peers: Sequence[Sequence[float]], target: Any,
                          *, mode: str) -> dict[str, Any]:
    """Stdlib single-query synthetic kernel and exact native-score derivatives.

    Bernoulli keeps one observed outcome per label and averages label scores;
    it never mixes factorized joint-label probabilities. Peers are fixed values.
    """
    scores = [tuple(factual), tuple(probe), *map(tuple, frozen_peers)]
    if not factual or any(len(row) != len(factual) for row in scores):
        raise ContractError("Synthetic score shape mismatch")
    if mode == "categorical_logits":
        if not isinstance(target, int) or not 0 <= target < len(factual):
            raise ContractError("Invalid categorical synthetic target")
        observed = [[row[target]-_lse(row)] for row in scores]
        ce_grad = [[exp(x-_lse(row))-(i == target) for i, x in enumerate(row)] for row in scores[:2]]
    elif mode == "bernoulli_marginal_logits":
        if len(target) != len(factual) or any(y not in (0, 1) for y in target):
            raise ContractError("Invalid Bernoulli synthetic outcome vector")
        observed = [[_logsigmoid(x if y else -x) for x, y in zip(row, target)] for row in scores]
        ce_grad = [[exp(_logsigmoid(x))-y for x, y in zip(row, target)] for row in scores[:2]]
    else:
        raise ContractError("Synthetic audit supports logits modes explicitly")
    count, entries = 1+len(frozen_peers), len(observed[0])
    supply_log, absent_log, rho_s, rho_a = [], [], [], []
    for entry in range(entries):
        peers = [row[entry] for row in observed[2:]]
        s = _lse([observed[0][entry], *peers])
        a = _lse([observed[1][entry], *peers])
        supply_log.append(s-log(count))
        absent_log.append(a-log(count))
        rho_s.append(exp(observed[0][entry]-s))
        rho_a.append(exp(observed[1][entry]-a))
    supplied, absent = -fsum(supply_log)/entries, -fsum(absent_log)/entries
    if mode == "categorical_logits":
        factual_gradient = [rho_s[0]*x for x in ce_grad[0]]
        probe_gradient = [-rho_a[0]*x for x in ce_grad[1]]
    else:
        factual_gradient = [r*x/entries for r, x in zip(rho_s, ce_grad[0])]
        probe_gradient = [-r*x/entries for r, x in zip(rho_a, ce_grad[1])]
    return {"J": supplied-absent, "supply": supplied, "absent": absent,
            "factual_gradient": tuple(factual_gradient), "probe_gradient": tuple(probe_gradient),
            "rho_supply": tuple(rho_s), "rho_absent": tuple(rho_a)}
