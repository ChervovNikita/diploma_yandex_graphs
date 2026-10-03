"""Synthetic numerical witnesses PREPARED, NOT EXECUTED on 2026-10-03.

Import uses stdlib only. Execution requires a future explicit authorization file.
This runner has no data/training/remote/launch path. Fixture resolutions do not
resolve the immutable proposal's native grouping, masks or optimizer semantics.
Float64 allclose tolerances below are witness equality tolerances only; the
sealed acceptance guards use exact inclusive comparisons without relaxation.
"""
from dataclasses import replace
from pathlib import Path
import argparse
import json
from time import perf_counter
from typing import Any
from core.groups import FixedPlan, LabelRoles, ResolutionRequired, raw_group_keys
from core.response import (ActiveReference, ResponseFailure, responses,
                           build_active_reference, redundancy, energy_failures,
                           centered_statistics, candidate_objective)
from core.guards import (CostLedger, GuardDecision, CompetenceReference,
                         build_competence_reference, assess_bundle, complete_views)
from core.transaction import (Custody, ExternalStateHook, StateSnapshot,
                             MOMENT_POLICY, ZERO_POLICY, guarded_adamw_displacement)

WITNESS_NAMES = (
    "label_role_boundaries_and_resolution_gate",
    "common_logit_shifts_and_member_permutation",
    "compensated_hidden_reparameterization_and_classifier_nullspace",
    "conditional_vs_class_only_grouping_and_functional_kernel_equivalence",
    "fixed_active_set_zero_large_and_inclusive_energy",
    "supported_and_fallback_control_competence",
    "private_only_gradient_flow_through_frozen_maps",
    "accepted_scaled_moments_state_replay_and_full_rejection_rollback",
    "guard_mutation_exception_rollback_and_cost_survival",
    "original_feature_gradient_exact_cached_token_adjoint",
)


def _torch():
    import torch
    return torch


def _close(a: Any, b: Any, atol: float = 1e-11, rtol: float = 1e-10):
    torch = _torch()
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError("Prepared numerical equality witness failed")


def _must_raise(kind: type[Exception], fn):
    try:
        fn()
    except kind:
        return
    raise AssertionError(f"Expected {kind.__name__}")


def _fixture():
    torch = _torch()
    fit, control = tuple(range(128)), tuple(range(128, 192))
    labels = tuple((n, int(64 <= n < 128 or n >= 160)) for n in range(192))
    roles = LabelRoles(fit, control, labels, 2, forbidden_nodes=(192, 193))
    resolution = (("fallback_membership", "synthetic_explicit_disjoint_rare_union"),
                  ("no_neighbor_fallback", "synthetic_explicit_class_then_global"),
                  ("undersized_global", "stop_if_global_unsupported"),
                  ("overlap_weighting", "disjoint_partition"),
                  ("control_redundant_cells", "retain_declared_cells"))
    fit_cells = tuple((f"fit_{g}", tuple(range(32 * g, 32 * (g + 1)))) for g in range(4))
    fit_plan = FixedPlan("fit", fit, fit_cells,
                         tuple((n, "fine") for n, _ in fit_cells), resolution,
                         "synthetic_fixture_only_not_native_rule_resolution")
    control_cells = (("global", control), ("class0", tuple(range(128, 160))),
                     ("class1_fallback_including_rare8", tuple(range(160, 192))),
                     ("class0_low", tuple(range(128, 144))),
                     ("class0_high", tuple(range(144, 160))),
                     ("class1_high_supported24", tuple(range(168, 192))))
    control_kinds = (("global", "global"), ("class0", "class"),
                     ("class1_fallback_including_rare8", "class"),
                     ("class0_low", "fine"), ("class0_high", "fine"),
                     ("class1_high_supported24", "fine"))
    control_resolution = tuple((k, "equal_cells_with_declared_overlap" if k == "overlap_weighting" else v)
                               for k, v in resolution)
    control_plan = FixedPlan("control", control, control_cells, control_kinds,
                             control_resolution, "synthetic_explicit_supported_and_fallback_cells")
    neighbors = {}
    for n, y in labels:
        is_high = (32 <= n < 64) or (96 <= n < 128) or (144 <= n < 160) or n >= 168
        same, other = ((0, 1), (64, 65)) if y == 0 else ((64, 65), (0, 1))
        neighbors[n] = (same + other[:1]) if is_high else (other + same[:1])
    raw_fit = raw_group_keys(roles, "fit", neighbors, adjacency_semantics_receipt="synthetic_directed_units")
    raw_control = raw_group_keys(roles, "control", neighbors, adjacency_semantics_receipt="synthetic_directed_units")
    fit_plan.validate_labels(roles, raw_fit)
    control_plan.validate_labels(roles, raw_control)
    node = torch.arange(192, dtype=torch.float64)[None, :, None]
    member = torch.arange(4, dtype=torch.float64)[:, None, None]
    cls = torch.arange(2, dtype=torch.float64)[None, None, :]
    target = torch.tensor([y for _, y in labels], dtype=torch.long)
    base = 1.5 * torch.nn.functional.one_hot(target, 2).to(torch.float64)[None, :, :]
    native = base + 0.1 * torch.sin(0.13 * node + 0.4 * member + 0.9 * cls)
    logits = {"native": native,
              "same_removal": native - 0.06 * torch.cos(0.19 * node + 0.3 * member + 0.8 * cls),
              "other_removal": native - 0.09 * torch.sin(0.11 * node + 0.7 * member + 1.1 * cls)}
    a = responses(logits)
    active = build_active_reference(a, fit_plan, warm_function_receipt="synthetic_warm_function")
    competence = build_competence_reference(logits, roles, control_plan,
                                           warm_function_receipt="synthetic_warm_function")
    return roles, fit_plan, control_plan, logits, active, competence, neighbors


def witness_label_boundaries():
    roles, fit_plan, _, _, _, _, neighbors = _fixture()
    _must_raise(ValueError, lambda: replace(roles, train_labels=roles.train_labels + ((192, 1),)))
    _must_raise(ValueError, lambda: replace(roles, control=roles.control + (192,)))
    _must_raise(ResolutionRequired, lambda: replace(fit_plan, resolution=()))
    _must_raise(ValueError, lambda: replace(fit_plan, cells=fit_plan.cells[:-1], kinds=fit_plan.kinds[:-1]))
    before = raw_group_keys(roles, "fit", neighbors, adjacency_semantics_receipt="synthetic")
    altered = replace(roles, train_labels=tuple((n, 1 - y if n in roles.control else y)
                                               for n, y in roles.train_labels))
    after = raw_group_keys(altered, "fit", neighbors, adjacency_semantics_receipt="synthetic")
    assert before == after
    # Even neighbors carrying control/forbidden ids cannot expose their labels.
    added = {n: tuple(v) + (128, 192, 193) for n, v in neighbors.items()}
    assert before == raw_group_keys(roles, "fit", added, adjacency_semantics_receipt="synthetic")
    isolated = dict(neighbors)
    isolated[0] = (128, 192)
    assert raw_group_keys(roles, "fit", isolated, adjacency_semantics_receipt="synthetic")[0][1] == "no_fit_labeled_neighbors"


def witness_logit_shifts_and_permutation():
    torch = _torch()
    roles, fit_plan, control_plan, logits, active, competence, _ = _fixture()
    shift = torch.linspace(-9, 13, 4 * 192, dtype=torch.float64).reshape(4, 192, 1)
    shifted = {view: z + (i + 1) * shift for i, (view, z) in enumerate(logits.items())}
    _close(responses(logits), responses(shifted))
    _close(redundancy(responses(logits), fit_plan, active), redundancy(responses(shifted), fit_plan, active))
    assert assess_bundle(shifted, roles, control_plan, competence, fit_plan, active).accepted
    order = [2, 0, 3, 1]
    permuted = {v: z[order] for v, z in logits.items()}
    pactive = replace(active, reference_norms=tuple((g, tuple(ns[i] for i in order))
                                                   for g, ns in active.reference_norms))
    pcompetence = replace(competence, values=tuple((v, g, tuple(ns[i] for i in order))
                                                 for v, g, ns in competence.values))
    _close(redundancy(responses(logits), fit_plan, active), redundancy(responses(permuted), fit_plan, pactive))
    before, _ = candidate_objective(logits, roles, fit_plan, active)
    after, _ = candidate_objective(permuted, roles, fit_plan, pactive)
    _close(before, after)
    assert assess_bundle(permuted, roles, control_plan, pcompetence, fit_plan, pactive).accepted


def witness_hidden_reparameterization():
    torch = _torch()
    _, fit_plan, _, _, _, _, _ = _fixture()
    x = torch.arange(192 * 3, dtype=torch.float64).reshape(192, 3) / 79
    w = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=torch.float64)
    null = torch.tensor([-1., -1., 1.], dtype=torch.float64)
    original, transformed = {}, {}
    for j, view in enumerate(("native", "same_removal", "other_removal")):
        z, znew = [], []
        for m in range(4):
            h = torch.sin(x * (1 + .1 * m)) + .03 * j * torch.cos(x * (1.2 + .03 * m))
            q = torch.tensor([[1.2 + .1 * m, .2, 0.], [0., .7, .1], [.1, 0., 1.4]], dtype=torch.float64)
            hnull = h + .2 * torch.cos(x[:, :1]) * null[None, :]
            z.append(h @ w)
            znew.append((hnull @ q) @ torch.linalg.solve(q, w))
        original[view], transformed[view] = torch.stack(z), torch.stack(znew)
        _close(original[view], transformed[view])
    a, anew = responses(original), responses(transformed)
    ref = build_active_reference(a, fit_plan, warm_function_receipt="synthetic_reparameterization")
    _close(a, anew)
    _close(redundancy(a, fit_plan, ref), redundancy(anew, fit_plan, ref))


def witness_conditional_grouping_and_kernel():
    torch = _torch()
    _, conditional, _, _, _, _, _ = _fixture()
    a = torch.zeros(4, 192, 4, dtype=torch.float64)
    nodes = torch.arange(32, dtype=torch.float64)
    for group in range(4):
        mean = .08 if group % 2 else -.08
        for m in range(4):
            delta = mean + .002 * torch.sin(2 * torch.pi * (m + 1) * nodes / 32)
            a[m, 32 * group:32 * (group + 1)] = torch.stack((delta, -delta, .5 * delta, -.5 * delta), dim=1)
    class_cells = (("class0", tuple(range(64))), ("class1", tuple(range(64, 128))))
    class_plan = replace(conditional, cells=class_cells, kinds=(("class0", "class"), ("class1", "class")))
    conditional_ref = build_active_reference(a, conditional, warm_function_receipt="synthetic_conditional")
    class_ref = build_active_reference(a, class_plan, warm_function_receipt="synthetic_class")
    dc, dclass = redundancy(a, conditional, conditional_ref), redundancy(a, class_plan, class_ref)
    assert float(dc) < 1e-20 and float(dclass) > .9
    # Probability-consistent measurement realization: native .5/.5, probes subtract A.
    p0 = torch.full((4, 192, 2), .5, dtype=torch.float64)
    logits = {"native": p0.log(), "same_removal": (p0 - a[:, :, :2]).log(),
              "other_removal": (p0 - a[:, :, 2:]).log()}
    _close(a, responses(logits))
    kernel_terms = []
    for r, norm in centered_statistics(a, conditional, conditional_ref.names).values():
        u = r / norm[:, None]
        kernel = (u @ u.T).square()  # homogeneous degree-two polynomial functional kernel
        ix = torch.triu_indices(4, 4, offset=1)
        kernel_terms.append(kernel[ix[0], ix[1]].mean())
    _close(dc, torch.stack(kernel_terms).mean())
    # This is exact operator duplication on identical measurements, not efficacy.


def witness_active_energy():
    torch = _torch()
    _, plan, _, logits, active, _, _ = _fixture()
    a = responses(logits)
    zero = torch.zeros_like(a)
    _must_raise(ResponseFailure, lambda: redundancy(zero, plan, active))
    assert len(energy_failures(zero, plan, active)) == 4 * len(active.names)
    assert len(energy_failures(3 * a, plan, active)) == 4 * len(active.names)
    assert not energy_failures(.5 * a, plan, active)
    assert not energy_failures(2 * a, plan, active)
    _close(redundancy(a, plan, active), redundancy(3 * a, plan, active))
    reduced = a.clone()
    reduced[0, :32] = 0
    ref = build_active_reference(reduced, plan, warm_function_receipt="synthetic_exclude_one")
    assert ref.names == tuple(n for n, _ in plan.cells)[1:] and ref.coverage == .75
    assert len(ref.excluded) == 1
    assert len(energy_failures(torch.zeros_like(reduced), plan, ref)) == 12
    assert ref.names == tuple(n for n, _ in plan.cells)[1:]  # current collapse never changes active set
    insufficient = torch.zeros_like(a)
    insufficient[:, :32] = a[:, :32]
    _must_raise(ResponseFailure, lambda: build_active_reference(insufficient, plan, warm_function_receipt="synthetic_insufficient"))
    _must_raise(ValueError, lambda: redundancy(a, replace(plan, source_receipt="changed"), active))


def witness_competence_cells():
    torch = _torch()
    roles, fit_plan, cells, logits, active, competence, _ = _fixture()
    assert assess_bundle(logits, roles, cells, competence, fit_plan, active).accepted
    # Damage a supported fine group, with member/view identity preserved.
    supported = {v: z.clone() for v, z in logits.items()}
    supported["same_removal"][2, 128:144, 0] -= 2
    decision = assess_bundle(supported, roles, cells, competence, fit_plan, active)
    assert not decision.accepted
    assert "competence:same_removal:class0_low:member:2" in decision.failures
    # Rare8 has no fine guard; its explicit class fallback still sees the failure.
    fallback = {v: z.clone() for v, z in logits.items()}
    fallback["native"][1, 160:168, 1] -= 4
    decision = assess_bundle(fallback, roles, cells, competence, fit_plan, active)
    assert "competence:native:class1_fallback_including_rare8:member:1" in decision.failures
    no_response = {v: logits["native"].clone() for v in logits}
    decision = assess_bundle(no_response, roles, cells, competence, fit_plan, active)
    assert not decision.accepted and any(f.startswith("energy:") for f in decision.failures)
    bad = {v: z.clone() for v, z in logits.items()}
    bad["native"][0, 0, 0] = torch.nan
    assert not assess_bundle(bad, roles, cells, competence, fit_plan, active).accepted


def _tiny_model_and_views():
    torch = _torch()

    class TinyPaths(torch.nn.Module):
        """Fixture only; not a PolyFormer port or backbone qualification."""
        def __init__(self):
            super().__init__()
            self.stem = torch.nn.Parameter(torch.arange(12, dtype=torch.float64).reshape(4, 3) / 25 + .1, requires_grad=False)
            self.common = torch.nn.Parameter(torch.eye(3, dtype=torch.float64) + .03, requires_grad=False)
            self.classifier = torch.nn.Parameter(torch.tensor([[.3, -.2], [-.1, .4], [.2, .1]], dtype=torch.float64), requires_grad=False)
            self.private_attention = torch.nn.Parameter(.9 + .01 * torch.arange(12, dtype=torch.float64).reshape(4, 3))
            self.private_ffn = torch.nn.Parameter(1.0 + .02 * torch.arange(12, dtype=torch.float64).reshape(4, 3))
            self.register_buffer("nonpersistent_custody", torch.tensor([7.], dtype=torch.float64), persistent=False)
            self.external_counter = {"calls": 0}

        def forward(self, x):
            h = torch.tanh((x @ self.stem)[None, :, :] * self.private_attention[:, None, :])
            h = torch.tanh(h @ self.common) * self.private_ffn[:, None, :]
            return h @ self.classifier

    model = TinyPaths().eval()
    x = torch.arange(192 * 4, dtype=torch.float64).reshape(192, 4) / 59
    views = {"native": torch.sin(x), "same_removal": torch.sin(x) - .06 * torch.cos(1.3 * x),
             "other_removal": torch.sin(x) + .07 * torch.sin(.7 * x + .4)}
    return model, views


def _custody(model):
    return Custody(("private_attention", "private_ffn"), "synthetic_sites_only",
                   "synthetic_all_mutable_state_audit", "synthetic_optimizer_interpretation_only",
                   MOMENT_POLICY, ZERO_POLICY,
                   (ExternalStateHook("external_counter", "synthetic_attribute_audit",
                                      lambda: model.external_counter,
                                      lambda value: setattr(model, "external_counter", value)),))


def witness_private_gradient_flow():
    torch = _torch()
    roles, fit_plan, _, _, _, _, _ = _fixture()
    model, views = _tiny_model_and_views()
    ledger = CostLedger()
    logits = complete_views(model, lambda mdl, v: mdl(views[v]), ledger=ledger,
                            stage="synthetic_objective", complete_nodes=192, gradients=True)
    active = build_active_reference(responses(logits), fit_plan, warm_function_receipt="synthetic_tiny")
    objective, _ = candidate_objective(logits, roles, fit_plan, active)
    backward_start = perf_counter()
    objective.backward()
    ledger.record_work("synthetic_objective_backward", elapsed_seconds=perf_counter() - backward_start,
                       details={"higher_order": False, "synchronized_gpu_timing": False})
    for name, p in model.named_parameters():
        if name in ("private_attention", "private_ffn"):
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        else:
            assert not p.requires_grad and p.grad is None
    assert ledger.snapshot()["forward_attempts"] == 3
    assert ledger.snapshot()["member_path_evaluation_attempts"] == 12


def _transaction_fixture():
    torch = _torch()
    roles, fit_plan, control_plan, _, _, _, _ = _fixture()
    model, views = _tiny_model_and_views()
    custody = _custody(model)
    optimizer = torch.optim.AdamW([model.private_attention, model.private_ffn], lr=1e-4, weight_decay=.1)
    # Initialize nonempty AdamW moments before the rollback witness. This is a
    # future synthetic fixture operation, not a warm-bank/data training run.
    for p in optimizer.param_groups[0]["params"]:
        p.grad = torch.full_like(p, .03)
    optimizer.step()
    for p in optimizer.param_groups[0]["params"]:
        p.grad = torch.full_like(p, .04)
    with torch.no_grad():
        warm = {v: model(x) for v, x in views.items()}
    active = build_active_reference(responses(warm), fit_plan, warm_function_receipt="synthetic_transaction")
    competence = build_competence_reference(warm, roles, control_plan, warm_function_receipt="synthetic_transaction")
    base = StateSnapshot.capture(model, optimizer, custody)
    optimizer.step()
    proposed = StateSnapshot.capture(model, optimizer, custody)
    base.restore(model, optimizer, custody)

    def natural_guard(mdl, ledger):
        logits = complete_views(mdl, lambda m, v: m(views[v]), ledger=ledger,
                                stage="synthetic_guard", complete_nodes=192, gradients=False)
        return assess_bundle(logits, roles, control_plan, competence, fit_plan, active)

    return model, optimizer, custody, base, proposed, natural_guard


def witness_transaction_replay():
    torch = _torch()
    model, optimizer, custody, base, proposed, natural = _transaction_fixture()

    def staged_guard():
        calls = 0
        def guard(mdl, ledger):
            nonlocal calls
            calls += 1
            actual = natural(mdl, ledger)
            assert actual.accepted  # tiny predetermined displacement, not threshold tuning
            # Deliberate fixture rejection forces scaled replay. This does not
            # claim that the real competence/energy guard needed backtracking.
            if calls == 1:
                return GuardDecision(False, ("fixture_reject_full_displacement",),
                                     actual.evaluated_ce_entries, actual.evaluated_energy_entries)
            return actual
        return guard

    ledger = CostLedger()
    result = guarded_adamw_displacement(model, optimizer, custody, staged_guard(), ledger)
    assert result.accepted_nonzero and result.scale == .5
    actual = StateSnapshot.capture(model, optimizer, custody)
    base.restore(model, optimizer, custody)
    with torch.no_grad():
        for name in custody.private_parameter_names:
            dict(model.named_parameters())[name].copy_(base.parameters[name] + .5 *
                                                       (proposed.parameters[name] - base.parameters[name]))
    proposed.restore_optimizer(model, optimizer)
    expected = StateSnapshot.capture(model, optimizer, custody)
    assert expected.exact_equal(actual)
    base.restore(model, optimizer, custody)
    replay_ledger = CostLedger()
    replay = guarded_adamw_displacement(model, optimizer, custody, staged_guard(), replay_ledger)
    assert replay.scale == .5 and actual.exact_equal(StateSnapshot.capture(model, optimizer, custody))
    assert ledger.snapshot()["optimizer_step_attempts"] == 1
    assert ledger.snapshot()["committed_proposals"] == 1
    assert ledger.snapshot()["forward_attempts"] == 6  # rejected full + accepted half
    assert ledger.snapshot()["member_path_evaluation_attempts"] == 24
    base.restore(model, optimizer, custody)
    rejected_ledger = CostLedger()
    def reject_all(mdl, cost):
        actual = natural(mdl, cost)
        return GuardDecision(False, actual.failures + ("fixture_reject_every_scale",),
                             actual.evaluated_ce_entries, actual.evaluated_energy_entries)
    zero = guarded_adamw_displacement(model, optimizer, custody, reject_all, rejected_ledger)
    assert zero.scale == 0 and not zero.accepted_nonzero
    assert base.exact_equal(StateSnapshot.capture(model, optimizer, custody))
    assert rejected_ledger.snapshot()["forward_attempts"] == 12
    assert rejected_ledger.snapshot()["member_path_evaluation_attempts"] == 48
    assert rejected_ledger.snapshot()["discarded_proposals"] == 1
    # Full parameter displacement can be zero while moments would change.
    # Explicit prepared zero policy discards those optimizer updates as well.
    optimizer.param_groups[0]["lr"] = 0.0
    zero_base = StateSnapshot.capture(model, optimizer, custody)
    zero_ledger = CostLedger()
    def no_guard_for_zero(mdl, cost):
        raise AssertionError("Bitwise-zero displacement should restore without a guard")
    zero_displacement = guarded_adamw_displacement(model, optimizer, custody, no_guard_for_zero, zero_ledger)
    assert zero_displacement.scale == 0 and zero_base.exact_equal(StateSnapshot.capture(model, optimizer, custody))
    assert zero_ledger.snapshot()["optimizer_step_attempts"] == 1
    assert zero_ledger.snapshot()["forward_attempts"] == 0 and zero_ledger.snapshot()["discarded_proposals"] == 1


def witness_exception_rollback():
    torch = _torch()
    model, optimizer, custody, base, _, natural = _transaction_fixture()
    ledger = CostLedger()
    def impure(mdl, cost):
        decision = natural(mdl, cost)
        mdl.nonpersistent_custody.add_(1)
        mdl.external_counter["calls"] += 1
        torch.rand(1)  # explicit future synthetic RNG-custody violation
        return decision
    _must_raise(RuntimeError, lambda: guarded_adamw_displacement(model, optimizer, custody, impure, ledger))
    assert base.exact_equal(StateSnapshot.capture(model, optimizer, custody))
    assert ledger.snapshot()["forward_attempts"] == 3
    assert ledger.snapshot()["discarded_proposals"] == 1
    # A failing forward is counted as attempted work, even when no bundle returns.
    failing = CostLedger()
    _must_raise(RuntimeError, lambda: failing.forward(
        lambda: (_ for _ in ()).throw(RuntimeError("fixture_forward_failure")),
        stage="synthetic_failure", view="native", complete_nodes=192))
    assert failing.snapshot()["forward_attempts"] == 1 and failing.snapshot()["forward_failures"] == 1


def witness_feature_adjoint():
    torch = _torch()
    n, features, orders = 6, 3, 3
    p = .7 * torch.eye(n, dtype=torch.float64) + .3 * torch.roll(torch.eye(n, dtype=torch.float64), 1, dims=1)
    operators = [torch.linalg.matrix_power(p, k) for k in range(orders)]
    x = (torch.arange(n * features, dtype=torch.float64).reshape(n, features) / 17).requires_grad_()
    weights = (.05 + torch.arange(4 * orders * features * 2, dtype=torch.float64).reshape(4, orders, features, 2) / 100).requires_grad_()
    tokens = [operator @ x for operator in operators]
    cached = [t.detach().clone().requires_grad_() for t in tokens]
    labels = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)

    def scores(ts):
        return sum(torch.tanh(torch.einsum("nf,mfc->mnc", t, weights[:, k]))
                   for k, t in enumerate(ts))

    z, zcached = scores(tokens), scores(cached)
    _close(z, zcached)
    originals, pulled, unpulled = [], [], []
    for m in range(4):
        for v in range(n):
            # Explicit true-label raw-logit score for this adapted graph fixture.
            gx = torch.autograd.grad(z[m, v, labels[v]], x, create_graph=True, retain_graph=True)[0]
            gt = torch.autograd.grad(zcached[m, v, labels[v]], cached, create_graph=True, retain_graph=True)
            adjoint = sum(op.T @ g for op, g in zip(operators, gt))
            _close(gx, adjoint)
            originals.append(gx)
            pulled.append(adjoint)
            unpulled.append(sum(gt))
    originals, pulled, unpulled = map(torch.stack, (originals, pulled, unpulled))
    assert not torch.allclose(originals, unpulled, atol=1e-11, rtol=1e-10)
    # Preserve M and target axes; higher-order graph must reach private weights.
    gx = originals.reshape(4, n, -1)
    u = gx / torch.linalg.vector_norm(gx, dim=-1, keepdim=True)
    sensitivity = torch.stack([(u[m] * u[j]).sum(dim=-1).square().mean()
                               for m in range(4) for j in range(m + 1, 4)]).mean()
    second = torch.autograd.grad(sensitivity, weights)[0]
    assert torch.isfinite(second).all() and second.abs().sum() > 0


def run_prepared_witnesses(*, authorization_file: str):
    authorization = json.loads(Path(authorization_file).read_text())
    if (authorization.get("scope") != "source_and_synthetic_numerical_qualification_only" or
            authorization.get("explicit_numerical_execution_authorized") is not True or
            not authorization.get("human_or_root_authorization_reference")):
        raise PermissionError("Future explicit numerical qualification authorization is required")
    functions = (witness_label_boundaries, witness_logit_shifts_and_permutation,
                 witness_hidden_reparameterization, witness_conditional_grouping_and_kernel,
                 witness_active_energy, witness_competence_cells, witness_private_gradient_flow,
                 witness_transaction_replay, witness_exception_rollback, witness_feature_adjoint)
    completed = []
    for name, function in zip(WITNESS_NAMES, functions):
        function()
        completed.append(name)
    return {"status": "synthetic_witnesses_passed", "completed": completed,
            "native_source_qualification": False, "complete_data_resource_measured": False,
            "pilot_frozen_or_launched": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-prepared-numerical-witnesses", action="store_true")
    parser.add_argument("--authorization-file")
    args = parser.parse_args()
    if not args.execute_prepared_numerical_witnesses:
        print(json.dumps({"status": "prepared_not_executed", "witnesses": WITNESS_NAMES}, indent=2))
    elif not args.authorization_file:
        parser.error("--authorization-file is required before any Torch import")
    else:
        print(json.dumps(run_prepared_witnesses(authorization_file=args.authorization_file), indent=2))
