"""Actual pinned native-composition checks PREPARED, NOT EXECUTED.

Separate root authorization is required; old ten generic witness admission does
not authorize these checks. No data, training fit, outcome selection or launch.
"""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
import json
from time import perf_counter
from native_source.adapter import NativeSpec, build_all_layer_polyformer, parameter_algebra
from native_source.tokens import CacheIdentity, build_uncached_mono_tokens, pullback_mono_gradients
from core.transaction import StateSnapshot

CHECK_NAMES = ("native_one_member_and_all_identical_affine_parity",
               "native_member_isolation_private_gradient_and_state_custody",
               "native_mono_normalization_and_original_feature_adjoint",
               "native_all_site_heterogeneous_materialized_reference_and_gradients",
               "native_split_reset_and_nonone_checkpoint_reconstruction",
               "native_irregular_three_view_normalization_identity_and_input_custody",
               "native_three_view_objective_and_initialized_adamw_transaction",
               "native_leading_axis_transpose_and_all_private_higher_order_adjoint")


def _spec():
    return NativeSpec(dataset="engineering_fixture_only", num_features=3, num_classes=2,
                      hidden=8, d_ffn=5, K=2, nlayer=2, n_head=2, q=1.4,
                      multi=1.5, dropout=.3, dprate=.4, base="mono",
                      recipe_receipt="synthetic_source_fixture_not_Amazon_recipe")


def _tokens():
    import torch
    x = torch.sin(torch.arange(24, dtype=torch.float32, device="cpu").reshape(8, 3) / 7)
    return torch.stack((x, .8 * x + .1, .6 * x - .05), dim=1)


def native_parity():
    import torch
    from native_source.native_polyformer_outer import PolyFormer
    spec = _spec()
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError("Native CPU engineering checks require float32 default dtype")
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(17)
        native = PolyFormer(None, SimpleNamespace(**asdict(spec))).eval()
    x = _tokens()
    expected = native(SimpleNamespace(list_mat=x.unbind(1)))
    for members in (1, 4):
        model = build_all_layer_polyformer(spec, seed=17, members=members,
                                          composition_receipt="engineering_new_composition_only").eval()
        actual = model(x)
        assert actual.shape == (members, 8, 2)
        for m in range(members):
            assert torch.equal(actual[m], expected)
        algebra = parameter_algebra(spec, members)
        assert sum(p.numel() for p in model.parameters()) == algebra["complete_parameters"]
        assert len(model.site_names) == algebra["affine_sites"]
        assert all(model.core.get_submodule(s).active_member is None for s in model.site_names)
        assert all(f"core.attn.{i}.attnmodule.bias" in dict(model.named_buffers()) for i in range(spec.nlayer))


def native_private_custody():
    import torch
    model = build_all_layer_polyformer(_spec(), seed=17,
                                      composition_receipt="engineering_new_composition_only").eval()
    x = _tokens()
    baseline = model(x).detach().clone()
    # Intermediate member factor perturbation must not affect the other members.
    site = model.intermediate_sites[0]
    with torch.no_grad():
        model.core.get_submodule(site).R[2].mul_(1.2)
    changed = model(x)
    assert not torch.equal(changed[2], baseline[2])
    for m in (0, 1, 3):
        assert torch.equal(changed[m], baseline[m])
    model.begin_private_continuation(warm_bank_receipt="engineering_fixture_not_actual_warm_bank")
    custody = model.stage_b_custody(audit_receipt="engineering_source_audit_only",
                                  optimizer_resolution_receipt="parent_modified_step_successor_rule")
    private = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(private, lr=.001)
    before = StateSnapshot.capture(model, optimizer, custody)
    with torch.no_grad():
        first, second = model(x), model(x)
    assert torch.equal(first, second)  # actual dropout probabilities are nonzero
    assert before.exact_equal(StateSnapshot.capture(model, optimizer, custody))
    try:
        model(x.double())  # native dtype mismatch fails after member selection
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected native dtype failure")
    assert before.exact_equal(StateSnapshot.capture(model, optimizer, custody))
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long, device="cpu")
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 2), labels.repeat(4))
    loss.backward()
    expected = set(custody.private_parameter_names)
    assert {n for n, p in model.named_parameters() if p.requires_grad} == expected
    for name, parameter in model.named_parameters():
        if name in expected:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        else:
            assert parameter.grad is None
    assert sum(float(p.grad.abs().sum()) for p in private) > 0
    assert len(private) == 2 * len(model.intermediate_sites)
    # Stem/readout/classifier R/S, every shared affine bias/weight, LayerNorm and
    # bias_scale are frozen. No no_grad/detach encloses the objective forward.


def native_normalization_adjoint():
    import torch
    import torch_geometric
    from native_source.native_preprocess import mono_base
    if torch_geometric.__version__ != "2.7.0":
        raise RuntimeError("Pinned PyG2.7 source/runtime qualification required")
    x = torch.sin(torch.arange(24, dtype=torch.float32, device="cpu").reshape(8, 3) / 7).requires_grad_()
    pairs = [(i, (i + 1) % 7) for i in range(7)]
    records = [(u, v) for u, v in pairs] + [(v, u) for u, v in pairs] + [(0, 0)]
    edge_index = torch.tensor(records, dtype=torch.long, device="cpu").T.contiguous()
    identity = CacheIdentity("fixture", "fixture_X", "fixture_edges", "none", "none", "fixture_fit",
                             "native", 2, "mono", "pinned_source_only", "float32_PyG2.7_fixture")
    bank = build_uncached_mono_tokens(x, edge_index, None, identity)
    source = mono_base(2, x, edge_index, None)
    assert bank.operator.dtype == torch.float32 and len(bank.tokens) == 3
    for left, right in zip(source, bank.tokens):
        assert torch.equal(left, right)  # same native normalization/arithmetic
    model = build_all_layer_polyformer(_spec(), seed=17,
                                      composition_receipt="engineering_new_composition_only").eval()
    model.begin_private_continuation(warm_bank_receipt="engineering_fixture_only")
    original = model(torch.stack(bank.tokens, dim=1))
    cached = tuple(t.detach().clone().requires_grad_() for t in bank.tokens)
    cached_scores = model(torch.stack(cached, dim=1))
    assert torch.equal(original, cached_scores)
    pulled = []
    for m in range(4):
        for target in range(8):
            y = target % 2
            gx = torch.autograd.grad(original[m, target, y], x, create_graph=True, retain_graph=True)[0]
            gt = torch.autograd.grad(cached_scores[m, target, y], cached, create_graph=True, retain_graph=True)
            adjoint = pullback_mono_gradients(gt, bank.operator)
            assert torch.allclose(gx, adjoint, atol=1e-5, rtol=1e-4)
            pulled.append(adjoint)
    # This is the explicitly synthetic true-label raw-logit adjoint. It does not
    # choose or qualify FoRDE's published score/objective. Preserve M/target axes.
    derivatives = torch.stack(pulled).reshape(4, 8, 8, 3)
    energy = derivatives.square().sum()
    second = torch.autograd.grad(energy, model.core.get_submodule(model.intermediate_sites[0]).R)[0]
    assert torch.isfinite(second).all() and second.abs().sum() > 0


def run_authorized_native_checks(authorization_file: str, *, receipt_file: str):
    receipt = json.loads(Path(authorization_file).read_text())
    if receipt.get("explicit_native_cpu_engineering_checks_authorized") is not True or not receipt.get("root_authorization_reference"):
        raise PermissionError("Old generic-witness authorization does not admit these native checks")
    from prepared_native_reference_checks import native_heterogeneous_reference, native_split_reset_reconstruction
    from prepared_native_view_checks import require_runtime_receipt, native_view_identity
    from prepared_native_transaction_checks import native_three_view_transaction
    from prepared_native_adjoint_checks import native_adjoint_axes_and_higher_order
    from engineering_receipts import EngineeringReceipt
    saved = EngineeringReceipt(receipt_file, receipt, CHECK_NAMES)
    start = perf_counter()
    try:
        runtime = require_runtime_receipt(receipt)
    except BaseException as error:
        saved.data["runtime_gate_elapsed_CPU_seconds"] = perf_counter() - start
        saved.finish(error=error)
        raise
    saved.data["runtime_gate_elapsed_CPU_seconds"] = perf_counter() - start
    saved.data["bound_synthetic_runtime"] = runtime
    saved.persist()
    functions = (native_parity, native_private_custody, native_normalization_adjoint,
                 lambda: native_heterogeneous_reference(_spec(), _tokens()),
                 lambda: native_split_reset_reconstruction(_spec(), _tokens()),
                 native_view_identity, lambda: native_three_view_transaction(_spec(), progress_callback=saved.family_progress),
                 lambda: native_adjoint_axes_and_higher_order(_spec()))
    if len(functions) != len(CHECK_NAMES):
        raise AssertionError("Prepared check inventory differs from dispatcher")
    check_receipts = []
    for name, function in zip(CHECK_NAMES, functions):
        saved.family_start(name)
        start = perf_counter()
        try:
            result = function()
        except BaseException as error:
            saved.family_end(elapsed_seconds=perf_counter() - start, error=error)
            saved.finish(runtime=runtime, error=error)
            raise  # stop on the same failure; no retries or fixture selection
        saved.family_end(elapsed_seconds=perf_counter() - start, result=result)
        check_receipts.append({"family": name, "elapsed_CPU_seconds": perf_counter() - start,
                               "engineering_result": result, "representative_resource_qualification": False})
    saved.finish(runtime=runtime)
    return {"prepared_native_checks_passed": list(CHECK_NAMES),
            "durable_receipt_file": str(saved.path),
            "synthetic_engineering_check_receipts": check_receipts,
            "bound_synthetic_runtime": runtime, "PyG2_3_runtime_equivalence_qualified": False,
            "Amazon_recipe_qualified": False, "DICE_FoRDE_qualified": False,
            "data_or_fit_performed": False, "pilot_launched": False}
