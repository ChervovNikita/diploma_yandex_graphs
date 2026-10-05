"""Root-run engineering worker. Disabled unless --execute-authorized is passed.

Synthetic targets are generated algebraic inputs, never dataset labels. Full
mode uses only a separately reviewed enabled public+B accessor. No fit, selected
checkpoint, optimizer, persistent update, A scoring, VALID/TEST or SSH exists.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import resource
import signal
import socket
import sys
import time
import traceback

PINS = {
    "operator": ("learnability_weighted_graph_responsibility_operator_20261005_v2/response_operator.py",
                 "fba3ca3d4bb35da0438923941d97bc4833004dd9fd385735c2352f343693b3e3"),
    "port": ("learnability_responsibility_native_amazon_sparse_port_20261005_v1/native_sparse_port.py",
             "a8ee35afda97afcb745c365a51f8e8647fe5e6a1350f654a5321c81e12abad86"),
    "native": ("amazon_polynormer_paired_family_source_preparation_20261003_v6/reused_models/native_polynormer.py",
               "9b4e533f46ae7f91a23a552f88bbb47a01359e224b53996cf1a68c10a865f6a8"),
    "boundary": ("amazon_polynormer_paired_family_source_preparation_20261003_v6/reused_models/backbone_boundary_adapter.py",
                 "699b606ead00cb7bdd9be6cd58730a0687157c40cf594af620d1edc4c93b6bac"),
}


def load_module(name, path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Source bytes differ: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def close(a, b, *, atol=1e-10, rtol=1e-8):
    error = float((a.detach() - b.detach()).abs().max()) if a.numel() else 0.0
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError("Tensor mismatch, max abs error=" + str(error))
    return error


def tree_error(actual, expected, *, atol=1e-8, rtol=1e-6):
    if actual.keys() != expected.keys():
        raise AssertionError("Parameter dictionaries differ")
    return max(close(actual[n], expected[n], atol=atol, rtol=rtol) for n in actual)


def zero_unused(theta_gradient, phis_before=None, phis_after=None):
    unused = [n for n in theta_gradient if n.startswith("local_head.")]
    if not unused or any(bool(torch.count_nonzero(theta_gradient[n])) for n in unused):
        raise AssertionError("Global-stage unused local head has nonzero shared tangent")
    if phis_before is not None:
        for before, after in zip(phis_before, phis_after):
            for name in before:
                if name.startswith("local_head.") and not torch.equal(before[name], after[name]):
                    raise AssertionError("Unused private local head moved")
    return {"unused_theta_names": unused, "unused_local_head_exact_zero": True}


def snapshot(family, x, edges, device):
    return {
        "parameters": {n: p.detach().clone() for n, p in family.named_parameters()},
        "parameter_ids": {n: id(p) for n, p in family.named_parameters(remove_duplicate=False)},
        "buffers": {n: b.detach().clone() for n, b in family.named_buffers()},
        "modes": {n: m.training for n, m in family.named_modules()},
        "attributes": {n: {k: repr(v) for k, v in vars(m).items()
                           if k not in {"_parameters", "_buffers", "_modules"}}
                       for n, m in family.named_modules()},
        "global": family.core._global,
        "x": x.clone(), "edges": edges.clone(),
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state().clone(),
        "cuda_rng": torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None,
    }


def unchanged(before, family, x, edges, device):
    after = snapshot(family, x, edges, device)
    for key in ("parameters", "buffers"):
        if before[key].keys() != after[key].keys() or any(
                not torch.equal(before[key][n], after[key][n]) for n in before[key]):
            raise AssertionError("Native state changed: " + key)
    for key in ("parameter_ids", "modes", "attributes", "global", "python_rng"):
        if before[key] != after[key]:
            raise AssertionError("Native metadata/RNG changed: " + key)
    for key in ("x", "edges", "torch_rng"):
        if not torch.equal(before[key], after[key]):
            raise AssertionError("Input/RNG changed: " + key)
    a, b = before["numpy_rng"], after["numpy_rng"]
    if a[0] != b[0] or not np.array_equal(a[1], b[1]) or a[2:] != b[2:]:
        raise AssertionError("NumPy RNG changed")
    if device.type == "cuda" and not torch.equal(before["cuda_rng"], after["cuda_rng"]):
        raise AssertionError("CUDA RNG changed")
    return {"native_parameters_modes_attributes_buffers_inputs_and_rng_unchanged": True}


def build_family(native, boundary, device, dtype, seed):
    torch.manual_seed(seed)
    model = native.Polynormer(300, 256, 5, local_layers=10, global_layers=1,
                             in_dropout=0.2, dropout=0.3, global_dropout=0.3,
                             heads=2, beta=-1, pre_ln=False).to(device=device, dtype=dtype)
    model.reset_parameters()
    family = boundary.PolynormerBoundaryFamily(model, members=4)
    family.set_global_stage(True)
    family.eval()
    return family


def outer(operator, core, phis, forward, pairs, s, ys, r, yr, control="live"):
    adapted, _ = operator._private_response(core, phis, forward, s, ys, pairs,
                                           5, operator.Config(), control)
    return operator._query_objective(core, adapted, forward, r, yr, operator.Config())


def reference_products_and_assignment(operator, pairs, *, dtype, device):
    generator = torch.Generator(device=device).manual_seed(106005)
    maximum, checks = 0.0, []
    for pair in pairs:
        lap = pair.laplacian
        count = pair.nodes.shape[0]
        q = torch.randn(count, 4, dtype=dtype, device=device, generator=generator)
        adjacency = torch.sparse_coo_tensor(torch.stack((lap.row, lap.column)), lap.weight,
                                            (count, count)).coalesce()
        expected = lap.degree[:, None] * q - torch.sparse.mm(adjacency, q)
        maximum = max(maximum, close(lap @ q, expected,
                                    atol=1e-10 if dtype == torch.float64 else 2e-6,
                                    rtol=1e-8 if dtype == torch.float64 else 2e-5))
        raw = torch.randn(count, 4, dtype=dtype, device=device, generator=generator)
        cost, _, _ = operator._normalized_cost(raw, operator.Config().response_epsilon)
        assigned = operator._assignment_map(cost, lap, operator.Config())
        def objective(q):
            cfg = operator.Config()
            return ((cost * q).sum() + cfg.entropy * (q * (4 * q).log()).sum()
                    + cfg.graph / 2 * (q * (lap @ q)).sum())
        previous = torch.full_like(cost, 0.25)
        descent_tolerance = 2e-11 if dtype == torch.float64 else 2e-3
        for prefix in range(1, 9):
            current = operator._assignment_map(cost, lap, replace(operator.Config(), assignment_steps=prefix))
            if float(objective(current) - objective(previous)) > descent_tolerance or current.min() <= 0:
                raise AssertionError("An exact solver prefix lost positivity/descent")
            previous = current
        row_error = float((assigned.sum(1) - 1).abs().max())
        col_error = float((assigned.sum(0) - count / 4).abs().max())
        row_tolerance = 2e-12 if dtype == torch.float64 else 3e-6
        col_tolerance = 2e-10 if dtype == torch.float64 else 2e-3
        if assigned.min() <= 0 or row_error > row_tolerance or col_error > col_tolerance:
            raise AssertionError("Sparse responsibility feasibility failed")
        checks.append({"classes": pair.classes, "items": count,
                       "row_residual": row_error, "column_residual": col_error,
                       "minimum": float(assigned.min())})
    return {"sparse_product_max_error": maximum, "assignment_checks": checks}


def pair_map_equivalence(operator, dense, sparse):
    generator = torch.Generator().manual_seed(136005)
    maximum_value, maximum_gradient = 0.0, 0.0
    for a, b in zip(dense, sparse):
        if (a.classes != b.classes or not torch.equal(a.nodes, b.nodes)
                or not torch.equal(a.targets, b.targets)
                or not torch.equal(a.competitors, b.competitors)):
            raise AssertionError("Dense/sparse task support differs")
        cost = torch.randn(a.nodes.numel(), 4, dtype=torch.float64, generator=generator)
        cotangent = torch.randn(cost.shape, dtype=torch.float64, generator=generator)
        for gamma in (1.0, 0.0):
            qa = operator._assignment_map(cost, a.laplacian, operator.Config(), gamma)
            qb = operator._assignment_map(cost, b.laplacian, operator.Config(), gamma)
            maximum_value = max(maximum_value, close(qa, qb))
            ga = torch.func.grad(lambda c: (operator._assignment_map(
                c, a.laplacian, operator.Config(), gamma) * cotangent).sum())(cost)
            gb = torch.func.grad(lambda c: (operator._assignment_map(
                c, b.laplacian, operator.Config(), gamma) * cotangent).sum())(cost)
            maximum_gradient = max(maximum_gradient, close(ga, gb))
    return {"assignment_value_max_error": maximum_value,
            "assignment_cost_gradient_max_error": maximum_gradient,
            "gamma0_checked": True}


def reconstruct_members(family, forward, theta, phis, x, edges, s, ys):
    results = []
    for member in range(4):
        reference = family.forward_member(x, edges, member)
        actual = forward(theta, phis[member])
        error = close(actual, reference)
        expected_params = dict(family.named_parameters())
        loss = torch.nn.functional.cross_entropy(reference[s], ys)
        expected_values = torch.autograd.grad(loss, tuple(expected_params.values()), allow_unused=True)
        missing = [n for (n, _), g in zip(expected_params.items(), expected_values)
                   if g is None and not n.startswith("local_head.")]
        if missing:
            raise AssertionError("Disconnected active native parameter: " + str(missing))
        expected = {n: torch.zeros_like(p) if g is None else g
                    for (n, p), g in zip(expected_params.items(), expected_values)}
        grads_theta, grads_phi = torch.func.grad(
            lambda core, private: torch.nn.functional.cross_entropy(forward(core, private)[s], ys),
            argnums=(0, 1))(theta, phis[member])
        shared_error = tree_error(grads_theta, {n: expected[n] for n in theta})
        private_error = tree_error(grads_phi, {n: expected[n][member] for n in phis[member]})
        for name in phis[member]:
            other_rows = [m for m in range(4) if m != member]
            if bool(torch.count_nonzero(expected[name][other_rows])):
                raise AssertionError("Selected reference touches another member row")
        zero_unused(grads_theta)
        results.append({"member": member, "full_logit_max_error": error,
                        "shared_gradient_max_error": shared_error,
                        "private_gradient_max_error": private_error})
        del reference, actual, expected, expected_values, grads_theta, grads_phi
    return {"all_four_members": results}


def dense_sparse_native(operator, forward, theta, phis, dense, sparse, s, ys, r, yr):
    cfg = operator.Config()
    a, _ = operator._private_response(theta, phis, forward, s, ys, dense, 5, cfg)
    b, _ = operator._private_response(theta, phis, forward, s, ys, sparse, 5, cfg)
    response_error = max(tree_error(left, right) for left, right in zip(a, b))
    value_error = close(outer(operator, theta, phis, forward, dense, s, ys, r, yr),
                        outer(operator, theta, phis, forward, sparse, s, ys, r, yr))
    dense_gradient = torch.func.grad(
        lambda core: outer(operator, core, phis, forward, dense, s, ys, r, yr))(theta)
    sparse_gradient = torch.func.grad(
        lambda core: outer(operator, core, phis, forward, sparse, s, ys, r, yr))(theta)
    gradient_error = tree_error(dense_gradient, sparse_gradient)
    zero_unused(sparse_gradient, phis, b)
    return {"private_response_max_error": response_error, "outer_value_error": value_error,
            "all_shared_coordinate_gradient_max_error": gradient_error}


def directional_native(operator, forward, theta, phis, pairs, s, ys, r, yr,
                       *, dtype, device):
    live = lambda core: outer(operator, core, phis, forward, pairs, s, ys, r, yr)
    stopped = lambda core: outer(operator, core, phis, forward, pairs, s, ys, r, yr, "stop_q")
    gradient = torch.func.grad(live)(theta)
    zero_unused(gradient)
    if any(not bool(torch.isfinite(g).all()) for g in gradient.values()):
        raise AssertionError("Nonfinite shared higher-order derivative")
    norm = sum(g.square().sum() for g in gradient.values()).sqrt()
    if not bool(norm > 0):
        raise AssertionError("Complete shared derivative vanished")
    generator = torch.Generator(device=device).manual_seed(116005)
    direction = {n: torch.zeros_like(v) if n.startswith("local_head.") else
                 torch.randn(v.shape, dtype=dtype, device=device, generator=generator)
                 for n, v in theta.items()}
    direction_norm = sum(v.square().sum() for v in direction.values()).sqrt()
    direction = {n: v / direction_norm for n, v in direction.items()}
    analytic = float(sum((gradient[n] * direction[n]).sum() for n in theta))
    scales = (1e-3, 3e-4, 1e-4) if dtype == torch.float64 else (0.1, 0.03, 0.01)
    differences = []
    for epsilon in scales:
        plus = {n: value + epsilon * direction[n] for n, value in theta.items()}
        minus = {n: value - epsilon * direction[n] for n, value in theta.items()}
        numerical = float((live(plus) - live(minus)) / (2 * epsilon))
        error = abs(numerical - analytic)
        tolerance = 5e-6 + 1e-3 * abs(analytic) if dtype == torch.float64 else 2e-4 + 0.03 * abs(analytic)
        if error > tolerance:
            raise AssertionError("Full shared directional FD failed: " +
                                 str((epsilon, numerical, analytic, error, tolerance)))
        differences.append({"epsilon": epsilon, "numerical": numerical,
                            "analytic": analytic, "absolute_error": error,
                            "fixed_tolerance": tolerance})
    # Completely unused shared local head: finite tangent MUST be exactly zero.
    unused_plus = {n: v + 0.1 if n.startswith("local_head.") else v for n, v in theta.items()}
    unused_minus = {n: v - 0.1 if n.startswith("local_head.") else v for n, v in theta.items()}
    if not torch.equal(live(unused_plus), live(unused_minus)):
        raise AssertionError("Unused local-head finite tangent is nonzero")
    live_response, info = operator._private_response(theta, phis, forward, s, ys, pairs,
                                                    5, operator.Config(), collect_diagnostics=True)
    stop_response, _ = operator._private_response(theta, phis, forward, s, ys, pairs,
                                                 5, operator.Config(), "stop_q")
    for a, b in zip(live_response, stop_response):
        if any(not torch.equal(a[n], b[n]) for n in a):
            raise AssertionError("Stopped Q changed the forward private response")
    anchors = info["assignments"]

    def fixed_q(core):
        partial = torch.func.grad(operator._main_loss, argnums=1)
        adapted = tuple(operator._sgd(phi, partial(core, phi, anchors, member, forward,
                                                  s, ys, pairs, 5, operator.Config()),
                                      operator.Config().eta_private)
                        for member, phi in enumerate(phis))
        return operator._query_objective(core, adapted, forward, r, yr, operator.Config())

    stopped_gradient = torch.func.grad(stopped)(theta)
    anchored_gradient = torch.func.grad(fixed_q)(theta)
    stop_error = tree_error(stopped_gradient, anchored_gradient,
                           atol=1e-8 if dtype == torch.float64 else 2e-5,
                           rtol=1e-6 if dtype == torch.float64 else 5e-4)
    chain_norm = float(sum((gradient[n] - stopped_gradient[n]).square().sum() for n in theta).sqrt())
    return {"full_active_shared_directional_finite_differences": differences,
            "unused_local_head_finite_tangent_exact_zero": True,
            "stop_q_vs_fixed_q_all_coordinate_error": stop_error,
            "live_q_chain_gradient_norm": chain_norm,
            "full_shared_gradient_norm": float(norm)}


def public_episode(operator, port, forward, theta, phis, pairs, s, ys, r, yr):
    """Execute the complete public entry; discard returned states, restore gate."""
    cfg = operator.Config()
    if port.PORT_RELEASED is not False:
        raise AssertionError("Unexpected native port entry gate")
    port.PORT_RELEASED = True  # Authorized process-local engineering release only.
    try:
        next_core, recomputed, info = port.native_sparse_episode(
            operator, PINS["operator"][1], theta, phis, forward, pairs, s, ys, r, yr,
            control="live")
    finally:
        port.PORT_RELEASED = False
    tangent = {n: (theta[n] - next_core[n]) / cfg.eta_core for n in theta}
    if not any(bool(torch.count_nonzero(v)) for n, v in tangent.items()
               if not n.startswith("local_head.")):
        raise AssertionError("Public episode produced no active shared change")
    zero_unused(tangent, phis, recomputed)
    fresh, _ = operator._private_response(next_core, phis, forward, s, ys,
                                         pairs, 5, cfg)
    tolerance = 1e-10 if next(iter(theta.values())).dtype == torch.float64 else 2e-6
    commit_error = max(tree_error(a, b, atol=tolerance, rtol=2e-5)
                       for a, b in zip(recomputed, fresh))
    if any(not bool(torch.isfinite(v).all()) for phi in recomputed for v in phi.values()):
        raise AssertionError("Nonfinite recomputed private state")
    return {"persistent_parameter_updates": 0, "constructed_states_discarded": True,
            "complete_public_nine_M_forward_episode_called": True,
            "extra_independent_response_recompute_checked": True,
            "post_core_original_private_recompute_max_error": commit_error,
            "process_local_entry_release_gate_restored": port.PORT_RELEASED is False,
            "all_shared_derivative_coordinates": sum(g.numel() for g in tangent.values()),
            "pair_item_counts": [p.nodes.shape[0] for p in pairs],
            "min_committed_assignment": min(float(q.min()) for q in info["assignments"])}


def run(args, check):
    global torch, np
    repo = Path(args.repository).resolve()
    site = repo / ".venv/lib/python3.11/site-packages"
    expected_python = repo / "experiments_iclr/postsubmission_20260930/native_ncn_runtime_20261005_v1/.venv/bin/python"
    if socket.gethostname() != "anogena-2-0" or Path(sys.executable).absolute() != expected_python:
        raise RuntimeError("Require the reviewed normal allocation/runtime, not another host/interpreter")
    if not sys.dont_write_bytecode:
        raise RuntimeError("Use -B to preserve source directories")
    if not site.is_dir():
        raise RuntimeError("Normal repository site-packages unavailable; no installation fallback")
    sys.path.insert(0, str(site))
    import torch
    import numpy as np
    if not Path(torch.__file__).resolve().is_relative_to(repo):
        raise RuntimeError("Torch comes from another runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if args.mode == "synthetic" and device.type != "cpu":
        raise ValueError("Synthetic fixture is bound to CPU float64")
    source_root = Path(args.source_root).resolve()
    modules = {n: load_module("qualification_" + n, source_root / p, h)
               for n, (p, h) in PINS.items()}
    op, port = modules["operator"], modules["port"]
    if op.SOURCE_RELEASED is not False or port.PORT_RELEASED is not False:
        raise AssertionError("Immutable math/port source flags changed")
    if args.mode == "synthetic":
        dtype, count = torch.float64, 15
        generator = torch.Generator().manual_seed(126005)
        x = torch.randn(count, 300, dtype=dtype, generator=generator)
        support = set((i, i) for i in range(count))
        for i in range(count - 1):
            support.update({(i, i + 1), (i + 1, i)})
        for i, j in [(0, 4), (1, 6), (2, 7), (3, 8), (4, 9), (0, 9)]:
            support.update({(i, j), (j, i)})
        edges = torch.tensor(sorted(support), dtype=torch.long).transpose(0, 1).contiguous()
        s, r = torch.arange(10), torch.arange(10, 15)
        ys, yr = torch.arange(10) % 5, torch.arange(5)
        seed = 106005
    else:
        if not args.synthetic_pass or not args.accessor or not args.accessor_sha or not args.public_b_dir:
            raise ValueError("Full mode requires passed synthetic receipt and reviewed enabled safe accessor")
        prior = json.loads(Path(args.synthetic_pass).read_text())
        if (prior.get("status") != "PASS_SYNTHETIC_ENGINEERING_ONLY"
                or prior.get("source_pins") != {n: list(v) for n, v in PINS.items()}):
            raise ValueError("Synthetic prerequisite differs or failed")
        accessor = load_module("qualification_safe_public_b", Path(args.accessor), args.accessor_sha)
        if accessor.SOURCE_RELEASED is not True:
            raise RuntimeError("Accessor revision is disabled; root must review a separate enabled revision")
        payload = accessor.load_public_b(repo, Path(args.public_b_dir), device=str(device))
        x, edges = payload["features"], payload["edge_index"]
        s, ys = payload["inner_indices"], payload["inner_labels"]
        r, yr = payload["query_indices"], payload["query_labels"]
        if (x.shape != (24492, 300) or x.dtype != torch.float32 or s.numel() != 2449
                or r.numel() != 2450 or payload["W_ids"].numel() != 4898
                or payload["B_ids"].numel() != 9797
                or bool(torch.isin(torch.cat((s, r)), payload["A_ids"]).any())):
            raise ValueError("Safe accessor/role dimensions differ")
        dtype, count, seed = torch.float32, 24492, 17
    family = build_family(modules["native"], modules["boundary"], device, dtype, seed)
    aliases = {}
    for name, parameter in family.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    gat_aliases = [names for names in aliases.values()
                  if any(".local_convs." in name for name in names)]
    check("native_GAT_alias_inventory", lambda: {
        "parameter_alias_groups": gat_aliases,
        "class_names": [type(module).__name__ for module in family.core.local_convs],
        "local_layers": len(family.core.local_convs), "global_stage": family.core._global})
    forward, theta, phis, shared_names = port._native_callback_and_state(
        family, x, edges, expected_nodes=count, global_stage=True)
    pairs = port._sparse_pairs(op, s, ys, edges, node_count=count, dtype=dtype)
    before = snapshot(family, x, edges, device)
    check("sparse_products_and_feasibility", lambda: reference_products_and_assignment(
        op, pairs, dtype=dtype, device=device))
    if args.mode == "synthetic":
        check("four_full_native_member_reconstructions", lambda: reconstruct_members(
            family, forward, theta, phis, x, edges, s, ys))
        affinity = torch.zeros(s.numel(), s.numel(), dtype=dtype)
        for i, j in sorted(support):
            if i < s.numel() and j < s.numel() and i != j:
                affinity[i, j] = 1
        dense = op._pair_blocks(ys, affinity, 5)
        check("dense_sparse_assignment_values_and_cost_gradients", lambda: pair_map_equivalence(op, dense, pairs))
        permutation = (torch.arange(s.numel()) + 5) % s.numel()
        permuted_sparse = port._sparse_pairs(op, s, ys, edges, node_count=count,
                                            dtype=dtype, affinity_permutation=permutation)
        permuted_dense = op._pair_blocks(ys, affinity[permutation][:, permutation], 5)
        check("fixed_within_class_affinity_permutation_orientation", lambda: pair_map_equivalence(
            op, permuted_dense, permuted_sparse))
        check("dense_sparse_native_output_and_full_shared_gradient", lambda: dense_sparse_native(
            op, forward, theta, phis, dense, pairs, s, ys, r, yr))
        check("full_native_mixed_shared_directional_fd_and_unused_tangent", lambda: directional_native(
            op, forward, theta, phis, pairs, s, ys, r, yr, dtype=dtype, device=device))
    elif args.full_fd:
        check("full_graph_mixed_shared_directional_fd_and_unused_tangent", lambda: directional_native(
            op, forward, theta, phis, pairs, s, ys, r, yr, dtype=dtype, device=device))
    check("complete_public_episode_higher_order_recompute_and_resource", lambda: public_episode(
        op, port, forward, theta, phis, pairs, s, ys, r, yr))
    check("native_state_input_and_rng_restoration", lambda: unchanged(before, family, x, edges, device))
    return {"torch_version": torch.__version__, "torch_module_path": torch.__file__,
            "dtype": str(dtype), "device": str(device), "nodes": count,
            "full_shared_parameter_names": list(shared_names),
            "native_GAT_parameter_alias_groups": gat_aliases,
            "real_dataset_label_access": args.mode == "full",
            "B_labels_only": args.mode == "full", "A_label_access": False,
            "full_graph_directional_fd_performed": args.mode == "full" and args.full_fd,
            "scientific_fit_or_predictive_utility_qualified": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-authorized", action="store_true")
    parser.add_argument("--mode", choices=("synthetic", "full"), default="synthetic")
    parser.add_argument("--source-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--repository", default="/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    parser.add_argument("--synthetic-pass")
    parser.add_argument("--accessor")
    parser.add_argument("--accessor-sha")
    parser.add_argument("--public-b-dir")
    parser.add_argument("--full-fd", action="store_true")
    args = parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({"status": "DISABLED", "numeric_imports": False,
                          "instruction": "Root reviews source before explicit execution flag"}))
        return
    if not args.output:
        parser.error("Authorized runs require a fresh --output directory")
    if args.mode == "synthetic" and any((args.accessor, args.public_b_dir, args.synthetic_pass)):
        parser.error("Synthetic fixture accepts no data/accessor/prerequisite inputs")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "RUNNING", "UTC": datetime.now(timezone.utc).isoformat(),
               "mode": args.mode, "source_pins": PINS, "checks": [],
               "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "hostname": socket.gethostname(), "python": sys.executable,
               "model_fits": 0, "persistent_updates": 0, "A_scoring": False,
               "VALID_TEST_access": False, "predictive_evidence": False}
    deadline_seconds = 180 if args.mode == "synthetic" else 900
    receipt["fixed_total_deadline_seconds"] = deadline_seconds
    started = time.monotonic()

    def save():
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        (output / "RESULT.json").write_text(json.dumps(receipt, indent=2) + "\n")

    def check(name, function):
        receipt["last_started_check"] = name
        save()
        beginning = time.monotonic()
        result = function()
        receipt["checks"].append({"name": name, "elapsed_seconds": time.monotonic() - beginning,
                                  "result": result})
        save()

    save()
    def timeout_handler(signum, frame):
        raise TimeoutError("Fixed engineering deadline exceeded: " + str(deadline_seconds))
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(deadline_seconds)
    try:
        receipt.update(run(args, check))
        receipt["status"] = ("PASS_SYNTHETIC_ENGINEERING_ONLY" if args.mode == "synthetic"
                             else "PASS_FULL_B_NO_UPDATE_RESOURCE_HIGHER_ORDER_ONLY")
    except BaseException as error:
        receipt.update(status="FAIL_ENGINEERING", error_type=type(error).__name__,
                       error=str(error), traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
        if "torch" in globals() and args.device.startswith("cuda") and torch.cuda.is_initialized():
            receipt["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
            receipt["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(args.device)
        save()
    print(json.dumps(receipt))
    raise SystemExit(0 if receipt["status"].startswith("PASS_") else 1)


if __name__ == "__main__":
    main()
