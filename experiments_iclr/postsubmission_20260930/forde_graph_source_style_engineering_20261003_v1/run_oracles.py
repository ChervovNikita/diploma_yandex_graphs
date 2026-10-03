"""Synthetic CPU-only derivative engineering; no labels, data, checkpoints or fit."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import torch
from forde_graph import (BoundGraphBatch, setup_all_private, source_sgd, tokens_from_x,
                         selected_coefficients, evaluate_coefficients, repulsion,
                         tensor_digest)


def metric(a, b, atol, rtol):
    a, b = a.detach(), b.detach()
    finite = bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
    error = (a - b).abs()
    scaled = error / (atol + rtol * b.abs())
    return {"finite": finite, "max_absolute_error": float(error.max()) if finite else None,
            "max_scaled_error": float(scaled.max()) if finite else None,
            "within_predeclared_tolerance": finite and bool((scaled <= 1).all())}


def summaries(values):
    finite = bool(torch.isfinite(values).all())
    return {"shape": list(values.shape), "finite": finite,
            "minimum": float(values.detach().min()) if finite else None,
            "maximum": float(values.detach().max()) if finite else None}


def gradient_map(loss, names, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return {name: gradient for name, gradient in zip(names, gradients)}


def compare_gradients(a, b, atol, rtol):
    records = {}
    for name in a:
        if a[name] is None or b[name] is None:
            records[name] = {"direct_present": b[name] is not None,
                             "gram_present": a[name] is not None,
                             "within_predeclared_tolerance": False}
        else:
            records[name] = metric(a[name], b[name], atol, rtol)
            records[name]["shape"] = list(a[name].shape)
            records[name]["direct_max_abs"] = float(b[name].detach().abs().max())
    return {"tensor_count": len(records), "every_tensor_passed": all(
        row["within_predeclared_tolerance"] for row in records.values()), "tensors": records}


def fixed_operator(dtype, case):
    if case == "nonsymmetric_duplicate":
        # Duplicate uncoalesced records: two 1/4 entries represent the first 1/2.
        indices = torch.tensor([[0, 0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 0, 2]])
        values = torch.tensor([1/4, 1/4, 1/2, 1/2, 1/2, 1/3, 2/3], dtype=dtype)
        return torch.sparse_coo_tensor(indices, values, (3, 3))
    if case == "identity_repeated_eigen":
        return torch.eye(3, dtype=dtype).to_sparse()
    raise ValueError(case)


def binding_checks(batch):
    def rejection(call):
        try:
            call()
        except ValueError as error:
            return {"rejected": True, "message": str(error)}
        return {"rejected": False}
    swapped = rejection(lambda: batch.validate(batch.targets.flip(0)))
    with torch.no_grad():
        original = batch.grams[0, 0, 0].clone()
        batch.grams[0, 0, 0] += 1
    modified = rejection(batch.validate)
    with torch.no_grad():
        batch.grams[0, 0, 0].copy_(original)
    batch.validate()
    return {"swapped_target_order": swapped, "modified_gram": modified}


def native_case(dtype, operator_case, tolerances):
    from native_source.adapter import NativeSpec, build_all_layer_polyformer
    spec = NativeSpec(dataset="synthetic_no_labels", num_features=2, num_classes=3,
                      hidden=4, d_ffn=6, K=2, nlayer=1, n_head=2, q=1., multi=1.5,
                      dropout=0.3, dprate=0.2, base="mono", recipe_receipt="synthetic_native_derivative_only")
    model = build_all_layer_polyformer(spec, seed=6143, members=4,
                                      composition_receipt="unchanged_native_v4_synthetic")
    model.to(dtype=dtype)
    # Predeclared deterministic distinct factors, not an outcome-adapted warm fit.
    with torch.no_grad():
        for site_index, site in enumerate(model.site_names):
            factor = model.core.get_submodule(site)
            for member in range(model.members):
                for name_index, name in enumerate(("R", "S")):
                    value = getattr(factor, name)[member]
                    coordinates = torch.arange(value.numel(), dtype=dtype)
                    value.copy_(1 + 0.17 * torch.sin((member + 1) * (coordinates + 1)
                                                   + (site_index + 1) * (name_index + 1)))
        # Ensure synthetic classifier readout's ReLU is away from zero.
        model.core.lin2.shared.bias.fill_(1.4)
    names = setup_all_private(model, bank_receipt="distinct_factor_synthetic_bank_v1")
    parameters = [dict(model.named_parameters())[name] for name in names]
    optimizer = source_sgd(model, learning_rate=0.01)
    assert {id(p) for p in optimizer.param_groups[0]["params"]} == {id(p) for p in parameters}
    assert not optimizer.state
    p = fixed_operator(dtype, operator_case)
    x = torch.tensor([[.4, -.8], [.7, .2], [-.3, .6]], dtype=dtype)
    targets = torch.tensor([2, 0])
    # Constant output channels are synthetic scalar definitions, not supervision.
    channels = torch.tensor([0, 1])
    tokens = tokens_from_x(x, p, 2).detach()
    batch = BoundGraphBatch.create(p, tokens, targets, maximum_power=2,
                                  feature_identity=tensor_digest(x),
                                  precision_receipt="CPU_1thread_highest_autocast_off",
                                  diagnostic_float64=dtype == torch.float64)
    source_before = {n: tensor_digest(value) for n, value in model.state_dict().items()}
    selected_logits = torch.stack([model.forward_member(tokens[targets], m) for m in range(4)])
    full_logits = torch.stack([model.forward_member(tokens, m)[targets] for m in range(4)])
    q, _ = selected_coefficients(model, batch, channels)
    gram_r, gram_details = evaluate_coefficients(q, batch, backend="gram")
    coefficient_direct_r, coefficient_direct_details = evaluate_coefficients(q, batch, backend="direct")
    direct_gradients = []
    for member in range(4):
        feature_leaf = x.clone().requires_grad_(True)
        live_tokens = tokens_from_x(feature_leaf, p, 2)
        z = model.forward_member(live_tokens, member)
        member_gradients = []
        for row, node in enumerate(targets):
            member_gradients.append(torch.autograd.grad(z[node, channels[row]], feature_leaf,
                                                        create_graph=True, retain_graph=True)[0])
        direct_gradients.append(torch.stack(member_gradients))
    full = torch.stack(direct_gradients)
    norm2 = full.square().sum(dim=(-1, -2))
    normalized = full / torch.sqrt(norm2[..., None, None] + 1e-24)
    d = (normalized[:, None] - normalized.detach()[None, :]).square().sum(dim=(-1, -2))
    similarity = (normalized[:, None] * normalized.detach()[None, :]).sum(dim=(-1, -2))
    direct_r, direct_h = repulsion(d)
    direct_kernel = torch.exp(-d / direct_h.unsqueeze(0)).mean(dim=2)
    atol, rtol = tolerances["value_absolute"], tolerances["value_relative"]
    checks = {
        "selected_full_logits": metric(selected_logits, full_logits, atol, rtol),
        "full_X_pullback": metric(torch.einsum("bkn,mbkf->mbnf", batch.powers, q), full, atol, rtol),
        "squared_norm": metric(gram_details["norm2"], norm2, atol, rtol),
        "normalized_similarity": metric(gram_details["similarity"], similarity, atol, rtol),
        "distance": metric(gram_details["distance"], d, atol, rtol),
        "bandwidth": metric(gram_details["bandwidth"], direct_h, atol, rtol),
        "kernel": metric(gram_details["kernel"], direct_kernel, atol, rtol),
        "repulsion": metric(gram_r.reshape(1), direct_r.reshape(1), atol, rtol),
        "coefficient_direct_repulsion": metric(coefficient_direct_r.reshape(1), direct_r.reshape(1), atol, rtol),
    }
    grad_direct = gradient_map(direct_r, names, parameters)
    grad_gram = gradient_map(gram_r, names, parameters)
    grad_coefficient_direct = gradient_map(coefficient_direct_r, names, parameters)
    mixed = compare_gradients(grad_gram, grad_direct, tolerances["gradient_absolute"], tolerances["gradient_relative"])
    mixed_direct = compare_gradients(grad_coefficient_direct, grad_direct,
                                     tolerances["gradient_absolute"], tolerances["gradient_relative"])
    direction_distances = d.detach()
    off_diagonal = direction_distances[~torch.eye(4, dtype=torch.bool)].reshape(12, -1)
    readout = model.core.lin2
    # Use passive calculation via ordinary source hooks, then discard.
    activations = []
    hook = model.core.lin3.register_forward_pre_hook(lambda module, args: activations.append(args[0].detach().clone()))
    try:
        for m in range(4): model.forward_member(tokens[targets], m)
    finally:
        hook.remove()
    native_self_zero = bool((d.diagonal(dim1=0, dim2=1) == 0).all())
    custody = binding_checks(batch)
    unchanged = source_before == {n: tensor_digest(value) for n, value in model.state_dict().items()}
    return {"kind": "native_v4", "dtype": str(dtype), "operator": operator_case,
            "checks": checks, "all_trainable_mixed_derivatives": mixed,
            "explicit_direct_backend_mixed_derivatives": mixed_direct,
            "trainable_names": list(names), "trainable_tensor_count": len(names),
            "trainable_scalars": sum(p.numel() for p in parameters),
            "boundary_sites_included": all(f"core.{site}.{factor}" in names for site in ("lin1", "lin2", "lin3") for factor in ("R", "S")),
            "optimizer_ownership_exact_and_fresh": True,
            "frozen_state_unchanged": unchanged, "all_module_modes_eval": all(not m.training for m in model.modules()),
            "member_selection_restored": all(model.core.get_submodule(s).active_member is None for s in model.site_names),
            "parameter_grads_not_written": all(p.grad is None for p in model.parameters()),
            "distinct_parameter_members": all(not torch.equal(parameters[0][i], parameters[0][j]) for i in range(4) for j in range(i)),
            "minimum_off_diagonal_direction_distance": float(off_diagonal.min()),
            "noncollapsed_directions": bool((off_diagonal > 0).all()),
            "raw_gradient_norm2": summaries(norm2),
            "penultimate_relu_minimum": float(torch.stack(activations).min()),
            "self_distance_exact_zero": native_self_zero, "binding_negative_witnesses": custody,
            "qualified_for_this_fixture": all(c["within_predeclared_tolerance"] for c in checks.values()) and mixed["every_tensor_passed"] and mixed_direct["every_tensor_passed"] and unchanged and native_self_zero and bool((off_diagonal > 0).all())}


def coefficient_case(dtype, name, tolerances):
    p = fixed_operator(dtype, "identity_repeated_eigen")
    x = torch.zeros((3, 2), dtype=dtype)
    targets = torch.tensor([0, 2])
    batch = BoundGraphBatch.create(p, tokens_from_x(x, p, 2), targets, maximum_power=2,
                                  feature_identity=tensor_digest(x), precision_receipt="CPU_fixture",
                                  diagnostic_float64=dtype == torch.float64)
    q = torch.zeros((4, 2, 3, 2), dtype=dtype)
    for member in range(4):
        angle = torch.tensor((member + 1) * .47, dtype=dtype)
        direction = torch.stack((angle.cos(), angle.sin()))
        for b in range(2):
            if name == "ordinary_collinear":
                q[member, b, 0] = direction
                q[member, b, 1] = .3 * direction
                q[member, b, 2] = -.1 * direction
            elif name == "severe_cancellation":
                q[member, b, 0] = 8192 * direction
                q[member, b, 1] = -8192 * direction
                q[member, b, 2] = .001 * direction
            elif name == "exact_null_cancellation":
                q[member, b, 0] = (member + 1) * 8192 * direction
                q[member, b, 1] = -q[member, b, 0]
            elif name == "all_zero":
                pass
            else:
                raise ValueError(name)
    q.requires_grad_(True)
    result = {"kind": "coefficient_fixture", "name": name, "dtype": str(dtype),
              "repeated_eigenvalues": [1, 1, 1], "collinear_row_powers": True,
              "author_norm_epsilon": 1e-24, "author_bandwidth_floor": 1e-12}
    direct_r, direct = evaluate_coefficients(q, batch, backend="direct")
    direct_grad = torch.autograd.grad(direct_r, q, retain_graph=True)[0]
    result["direct"] = {"repulsion": float(direct_r.detach()), "norm2": summaries(direct["norm2"]),
                        "bandwidth": summaries(direct["bandwidth"]),
                        "gradient": summaries(direct_grad),
                        "self_distance_exact_zero": bool((direct["distance"].diagonal(dim1=0, dim2=1) == 0).all())}
    try:
        gram_r, gram = evaluate_coefficients(q, batch, backend="gram")
        gram_grad = torch.autograd.grad(gram_r, q, retain_graph=True)[0]
        comparisons = {key: metric(gram[key], direct[key], tolerances["value_absolute"], tolerances["value_relative"])
                       for key in ("norm2", "distance", "similarity", "bandwidth", "kernel")}
        comparisons["repulsion"] = metric(gram_r.reshape(1), direct_r.reshape(1), tolerances["value_absolute"], tolerances["value_relative"])
        comparisons["all_q_mixed_derivatives"] = metric(gram_grad, direct_grad, tolerances["gradient_absolute"], tolerances["gradient_relative"])
        result["gram"] = {"completed": True, "repulsion": float(gram_r.detach()),
                          "gradient": summaries(gram_grad), "comparisons": comparisons,
                          "self_distance_exact_zero": bool((gram["distance"].diagonal(dim1=0, dim2=1) == 0).all()),
                          "qualified_for_this_fixture": all(c["within_predeclared_tolerance"] for c in comparisons.values())}
    except (FloatingPointError, RuntimeError) as error:
        result["gram"] = {"completed": False, "failure_type": type(error).__name__,
                          "message": str(error), "qualified_for_this_fixture": False}
    try:
        repulsion(direct["distance"], frozen_bandwidth=torch.zeros_like(direct["bandwidth"]))
        result["explicit_zero_bandwidth_rejected"] = False
    except FloatingPointError as error:
        result["explicit_zero_bandwidth_rejected"] = True
        result["zero_bandwidth_failure_message"] = str(error)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.native_source.is_dir()
    sys.path.insert(0, str(args.native_source))
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    args.output.mkdir(exist_ok=False)
    started = time.monotonic()
    receipt = {"schema": "forde_synthetic_CPU_derivative_receipt_v1",
               "UTC_started": datetime.now(timezone.utc).isoformat(),
               "torch_version": torch.__version__, "python_version": sys.version,
               "interpreter": sys.executable, "native_source": str(args.native_source),
               "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "source_sha256": hashlib.sha256((Path(__file__).parent / 'forde_graph.py').read_bytes()).hexdigest(),
               "runtime_settings": {"device": "cpu", "threads": torch.get_num_threads(),
                                    "interop_threads": torch.get_num_interop_threads(),
                                    "float32_matmul_precision": torch.get_float32_matmul_precision(),
                                    "autocast": False, "distributed": False},
               "no_dataset_labels_checkpoints_or_quality_scoring": True,
               "no_optimizer_steps_or_predictive_fit": True,
               "float64_scope": "explicit synthetic diagnostic; not native float32 substitution",
               "tolerances_predeclared_before_execution": {
                   "torch.float32": {"value_absolute": 3e-6, "value_relative": 3e-4,
                                      "gradient_absolute": 1e-4, "gradient_relative": 3e-3},
                   "torch.float64": {"value_absolute": 1e-11, "value_relative": 1e-8,
                                      "gradient_absolute": 1e-9, "gradient_relative": 1e-7}},
               "native_cases": [], "coefficient_cases": [], "unexpected_failures": []}
    for dtype in (torch.float32, torch.float64):
        tolerance = receipt["tolerances_predeclared_before_execution"][str(dtype)]
        for operator in ("nonsymmetric_duplicate", "identity_repeated_eigen"):
            try:
                receipt["native_cases"].append(native_case(dtype, operator, tolerance))
            except Exception as error:
                receipt["unexpected_failures"].append({"fixture": "native", "dtype": str(dtype),
                                                       "operator": operator, "type": type(error).__name__,
                                                       "message": str(error), "traceback": traceback.format_exc()})
        for name in ("ordinary_collinear", "severe_cancellation", "exact_null_cancellation", "all_zero"):
            try:
                receipt["coefficient_cases"].append(coefficient_case(dtype, name, tolerance))
            except Exception as error:
                receipt["unexpected_failures"].append({"fixture": name, "dtype": str(dtype),
                                                       "type": type(error).__name__, "message": str(error),
                                                       "traceback": traceback.format_exc()})
    receipt["wall_seconds"] = time.monotonic() - started
    receipt["CUDA_initialized_after"] = torch.cuda.is_initialized()
    receipt["UTC_finished"] = datetime.now(timezone.utc).isoformat()
    receipt["all_native_cases_qualified"] = len(receipt["native_cases"]) == 4 and all(row["qualified_for_this_fixture"] for row in receipt["native_cases"])
    receipt["all_Gram_coefficient_cases_qualified"] = len(receipt["coefficient_cases"]) == 8 and all(row["gram"]["qualified_for_this_fixture"] for row in receipt["coefficient_cases"])
    receipt["status"] = "NUMERICAL_BLOCKERS_PRESERVED" if receipt["unexpected_failures"] or not receipt["all_native_cases_qualified"] or not receipt["all_Gram_coefficient_cases_qualified"] else "SYNTHETIC_SCOPE_QUALIFIED"
    (args.output / "EXECUTION_RECEIPT.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: receipt[key] for key in ("status", "wall_seconds", "all_native_cases_qualified", "all_Gram_coefficient_cases_qualified", "CUDA_initialized_after")}))


if __name__ == "__main__":
    main()
