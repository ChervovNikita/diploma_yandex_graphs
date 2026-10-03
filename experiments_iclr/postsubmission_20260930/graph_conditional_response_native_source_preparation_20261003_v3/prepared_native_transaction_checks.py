"""Actual native three-view transaction checks, prepared and unexecuted."""
from dataclasses import replace
from time import perf_counter
from core.guards import CostLedger, GuardDecision, complete_views, build_competence_reference, assess_bundle
from core.response import candidate_objective, responses, build_active_reference, VIEWS
from core.transaction import StateSnapshot, guarded_adamw_displacement, _equal
from native_source.adapter import build_all_layer_polyformer
from prepared_native_reference_checks import unequal_factors
from prepared_native_view_checks import fixture_views, input_hook


def _audit(model, optimizer):
    """Bound synthetic object audit; no generic Python-state promise."""
    from pathlib import Path
    expected = Path(__file__).resolve().parent / "native_source/adapter.py"
    for name in ("forward", "forward_member", "begin_private_continuation", "stage_b_custody"):
        method = getattr(model, name)
        assert name not in model.__dict__ and Path(method.__func__.__code__.co_filename).resolve() == expected
    for module in model.modules():
        for name in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks"):
            assert not getattr(module, name, {})
        assert "forward" not in module.__dict__
    assert "step" not in optimizer.__dict__


def _charge_step(optimizer, ledger, kind):
    ledger.optimizer_step_attempts += 1
    start = perf_counter()
    optimizer.step()
    ledger.record_work(kind, elapsed_seconds=perf_counter() - start,
                       details={"actual_ordinary_adamw_steps": 1, "synthetic_fixture_only": True})


def _objective(model, complete, roles, fit, active, optimizer, ledger, stage):
    import torch
    optimizer.zero_grad(set_to_none=True)
    start = perf_counter()
    bundle = complete_views(model, complete.forward_view, ledger=ledger, stage=stage,
                            complete_nodes=complete.complete_nodes, gradients=True)
    objective, parts = candidate_objective(bundle, roles, fit, active)
    indices = torch.tensor(roles.fit, dtype=torch.long, device="cpu")
    labels = torch.tensor(roles.labels_for("fit"), dtype=torch.long, device="cpu")
    explicit = {}
    for view in VIEWS:
        explicit[view] = torch.stack([torch.nn.functional.cross_entropy(
            bundle[view][member].index_select(0, indices), labels) for member in range(4)]).mean()
    assert torch.allclose(parts["native_ce"], explicit["native"], atol=1e-7, rtol=1e-6)
    assert torch.allclose(parts["probe_ce"], (explicit["same_removal"] + explicit["other_removal"]) / 2,
                          atol=1e-7, rtol=1e-6)
    assert torch.equal(objective, parts["native_ce"] + parts["probe_ce"] + .1 * parts["D"])
    assert objective.requires_grad and all(value.requires_grad for value in parts.values())
    # Every actual view contributes through the full native autograd graph.
    view_gradients = torch.autograd.grad(objective, tuple(bundle[v] for v in VIEWS), retain_graph=True)
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in view_gradients)
    private = tuple(p for p in model.parameters() if p.requires_grad)
    d_gradients = torch.autograd.grad(parts["D"], private, retain_graph=True)
    assert all(torch.isfinite(g).all() for g in d_gradients)
    assert sum(float(g.abs().sum()) for g in d_gradients) > 0
    objective.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in private)
    assert sum(float(p.grad.abs().sum()) for p in private) > 0
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    ledger.record_work("native_three_view_objective_and_backward", elapsed_seconds=perf_counter() - start,
                       details={"three_complete_views": True, "additional_view_and_D_gradient_witnesses": True,
                                "nested_forward_spans_included": True, "synthetic_fixture_only": True})


def native_three_view_transaction(spec):
    import torch
    x, edges, edge_views, complete, roles, fit, control = fixture_views(spec.K)
    model = build_all_layer_polyformer(spec, seed=17, composition_receipt="synthetic_native_transaction_only").eval()
    unequal_factors(model)
    model.begin_private_continuation(warm_bank_receipt="synthetic_nonone_not_a_fitted_warm_bank")
    # Preserve nondefault selections as well as ordinary None restoration.
    for i, name in enumerate(model.site_names):
        model.core.get_submodule(name).active_member = i % 4
    native_custody = model.stage_b_custody(audit_receipt="synthetic_bound_native_object_audit",
                                         optimizer_resolution_receipt="parent_modified_step_successor_rule")
    custody = replace(native_custody, external_hooks=native_custody.external_hooks +
                      (input_hook(x, edges, edge_views, complete),))
    private = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(private, lr=1e-5, weight_decay=.01)
    optimizer._engineering_marker = {"nested": ["bound_python_state", 17]}
    ledger = CostLedger()
    _audit(model, optimizer)
    # Initialize actual optimizer state with fixed synthetic gradients, not a
    # model fit. The complete step is explicitly charged, then cleared.
    for p in private:
        p.grad = torch.full_like(p, .01)
    _charge_step(optimizer, ledger, "engineering_moment_initialization")
    assert len(optimizer.state) == len(private) and all(
        {"step", "exp_avg", "exp_avg_sq"} <= state.keys() for state in optimizer.state.values())
    optimizer.zero_grad(set_to_none=True)
    warm_receipt = "one_copied_synthetic_native_function_for_all_three_views"
    warm = complete_views(model, complete.forward_view, ledger=ledger, stage="synthetic_warm_reference",
                          complete_nodes=96, gradients=False)
    active = build_active_reference(responses(warm), fit, warm_function_receipt=warm_receipt)
    competence = build_competence_reference(warm, roles, control, warm_function_receipt=warm_receipt)
    assert active.names and active.coverage >= .5
    assert assess_bundle(warm, roles, control, competence, fit, active).accepted

    def actual_guard(current, costs, stage):
        logits = complete_views(current, complete.forward_view, ledger=costs, stage=stage,
                                complete_nodes=96, gradients=False)
        decision = assess_bundle(logits, roles, control, competence, fit, active)
        assert decision.evaluated_ce_entries == 3 * len(control.cells) * 4
        assert decision.evaluated_energy_entries == len(active.names) * 4
        return decision

    # Full and fractional acceptance: derive the exact independent ordinary
    # proposal first, restore the base, then witness a single actual transaction.
    for target_scale in (1.0, .5):
        _objective(model, complete, roles, fit, active, optimizer, ledger, f"objective_accept_{target_scale}")
        base = StateSnapshot.capture(model, optimizer, custody)
        _charge_step(optimizer, ledger, "engineering_exact_proposal_reference")
        proposal = StateSnapshot.capture(model, optimizer, custody)
        base.restore(model, optimizer, custody)
        with torch.no_grad():
            named = dict(model.named_parameters())
            for name in custody.private_parameter_names:
                named[name].copy_(proposal.parameters[name] if target_scale == 1 else
                                  base.parameters[name] + target_scale * (proposal.parameters[name] - base.parameters[name]))
        proposal.restore_optimizer(model, optimizer)
        expected = StateSnapshot.capture(model, optimizer, custody)
        base.restore(model, optimizer, custody)
        assert base.exact_equal(StateSnapshot.capture(model, optimizer, custody))
        calls = []

        def guard(current, costs):
            decision = actual_guard(current, costs, f"accept_{target_scale}_trial")
            calls.append(decision)
            if target_scale == .5 and len(calls) == 1:
                return GuardDecision(False, decision.failures + ("engineering_force_first_scale_rejection",),
                                     decision.evaluated_ce_entries, decision.evaluated_energy_entries)
            return decision

        old = ledger.snapshot()
        result = guarded_adamw_displacement(model, optimizer, custody, guard, ledger)
        assert result.accepted_nonzero and result.scale == target_scale
        assert len(result.trials) == (1 if target_scale == 1 else 2)
        assert expected.exact_equal(StateSnapshot.capture(model, optimizer, custody))
        new = ledger.snapshot()
        assert new["forward_attempts"] - old["forward_attempts"] == 3 * len(result.trials)
        assert new["member_path_evaluation_attempts"] - old["member_path_evaluation_attempts"] == 12 * len(result.trials)
        assert new["optimizer_step_attempts"] - old["optimizer_step_attempts"] == 1
        assert new["committed_proposals"] - old["committed_proposals"] == 1
        assert all(_equal(base.parameters[n], expected.parameters[n]) for n in base.parameters
                   if n not in custody.private_parameter_names)

    # Exhausted scales: every bundle runs the actual guard, with an additional
    # fixed engineering rejection to force restoration independently of feasibility.
    _objective(model, complete, roles, fit, active, optimizer, ledger, "objective_all_rejected")
    base = StateSnapshot.capture(model, optimizer, custody)
    old = ledger.snapshot()

    def reject(current, costs):
        decision = actual_guard(current, costs, "all_rejected_trial")
        return GuardDecision(False, decision.failures + ("engineering_forced_rejection",),
                             decision.evaluated_ce_entries, decision.evaluated_energy_entries)

    result = guarded_adamw_displacement(model, optimizer, custody, reject, ledger)
    assert not result.accepted_nonzero and result.scale == 0
    assert tuple(s for s, _ in result.trials) == (1.0, .5, .25, .125)
    assert base.exact_equal(StateSnapshot.capture(model, optimizer, custody))
    new = ledger.snapshot()
    assert new["forward_attempts"] - old["forward_attempts"] == 12
    assert new["member_path_evaluation_attempts"] - old["member_path_evaluation_attempts"] == 48
    assert new["discarded_proposals"] - old["discarded_proposals"] == 1

    # Exact full displacement zero, with initialized moments: no zero guard.
    for group in optimizer.param_groups:
        group["lr"] = 0.0
    base = StateSnapshot.capture(model, optimizer, custody)
    old = ledger.snapshot()

    def forbidden_zero_guard(current, costs):
        raise AssertionError("Adopted restore_only_no_guard forbids this callback")

    result = guarded_adamw_displacement(model, optimizer, custody, forbidden_zero_guard, ledger)
    assert not result.accepted_nonzero and result.scale == 0 and result.trials == ()
    assert base.exact_equal(StateSnapshot.capture(model, optimizer, custody))
    assert ledger.snapshot()["forward_attempts"] == old["forward_attempts"]
    assert ledger.discarded_proposals == old["discarded_proposals"] + 1
    for group in optimizer.param_groups:
        group["lr"] = 1e-5

    # Native primitive, tensors, permissions, optimizer Python/moments and RNG
    # are deliberately corrupted only after a complete real guard bundle.
    base = StateSnapshot.capture(model, optimizer, custody)
    old = ledger.snapshot()

    def impure(current, costs):
        decision = actual_guard(current, costs, "impure_trial")
        first = current.core.get_submodule(current.intermediate_sites[0])
        with torch.no_grad():
            first.R.add_(.1)
            first.R.grad.add_(.2)
            current.core.lin1.shared.bias.add_(.3)
            current.core.attn[0].attnmodule.bias.add_(.4)
            dict(complete.banks)["same_removal"].tokens[1].add_(.5)
            next(iter(optimizer.state.values()))["exp_avg"].add_(.6)
        current.core.lin1.shared.weight.requires_grad_(True)
        current.core.attn[0].ffnmodule.ffn_net.gelu.approximate = "tanh"
        current.core.dropout = .99
        current.core.training = True
        first.active_member = None
        current._in_forward = True
        current.stage = "engineering_corruption"
        optimizer._engineering_marker["nested"].append("mutated")
        optimizer.param_groups[0]["lr"] = .9
        torch.rand(5, device="cpu")
        return decision

    try:
        guarded_adamw_displacement(model, optimizer, custody, impure, ledger)
    except RuntimeError as error:
        assert "Guard mutated" in str(error)
    else:
        raise AssertionError("Impure native/input guard was accepted")
    assert base.exact_equal(StateSnapshot.capture(model, optimizer, custody))
    assert ledger.snapshot()["forward_returns"] - old["forward_returns"] == 3
    assert ledger.discarded_proposals == old["discarded_proposals"] + 1

    # Raised native guard forward: all nondefault selected members and in-forward
    # flag must restore; attempted failed work remains charged outside rollback.
    base = StateSnapshot.capture(model, optimizer, custody)
    old = ledger.snapshot()

    def native_failure(current, costs):
        def wrong_dtype(mod, view):
            bank = dict(complete.banks)[view]
            return mod(torch.stack(bank.tokens, dim=1).double())
        complete_views(current, wrong_dtype, ledger=costs, stage="native_exception_trial",
                       complete_nodes=96, gradients=False)
        raise AssertionError("Expected native dtype failure")

    try:
        guarded_adamw_displacement(model, optimizer, custody, native_failure, ledger)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Native guard exception did not propagate")
    assert base.exact_equal(StateSnapshot.capture(model, optimizer, custody))
    new = ledger.snapshot()
    assert new["forward_attempts"] - old["forward_attempts"] == 1
    assert new["forward_failures"] - old["forward_failures"] == 1
    assert new["member_path_evaluation_attempts"] - old["member_path_evaluation_attempts"] == 4
    assert new["discarded_proposals"] - old["discarded_proposals"] == 1
    _audit(model, optimizer)
    assert all(model.core.get_submodule(name).active_member == i % 4 for i, name in enumerate(model.site_names))
    assert model._in_forward is False
    return ledger.snapshot()
