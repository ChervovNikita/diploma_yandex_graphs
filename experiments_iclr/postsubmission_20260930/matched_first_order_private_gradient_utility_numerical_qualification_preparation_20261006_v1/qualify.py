"""Disabled CPU float64 synthetic utility qualification, never a fit.

The graph/targets are algebraically generated; no dataset/native-model loader.
The independent oracle explicitly differentiates each scalar margin to private
coordinates and keeps both factors live inside the monolithic outer objective.
Native double-AD/full-context FP32 support is outside this synthetic scope.
"""
import time
STARTED = time.monotonic()
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import resource
import signal
import sys
import traceback

SOURCE_RELEASED = False
PINS = {
    "operator": ("learnability_weighted_graph_responsibility_operator_20261005_v2/response_operator.py", "fba3ca3d4bb35da0438923941d97bc4833004dd9fd385735c2352f343693b3e3"),
    "port": ("learnability_responsibility_native_amazon_sparse_port_20261005_v1/native_sparse_port.py", "a8ee35afda97afcb745c365a51f8e8647fe5e6a1350f654a5321c81e12abad86"),
    "sequential": ("learnability_responsibility_sequential_autograd_vjp_preparation_20261006_v1/sequential_vjp.py", "1f9e704a1b95a98cec688d695094058594758a99bac2f057e2fe90e7f9315167"),
    "utility": ("matched_first_order_private_gradient_utility_control_source_preparation_20261006_v1/utility_control.py", "83967dc045db903205ce6c67df963fb557a858175e107e7de205651ad0b28545"),
}
VALUE_ATOL, VALUE_RTOL = 1e-10, 1e-8
GRAD_ATOL, GRAD_RTOL = 1e-8, 1e-6
FD_SCALES = (1e-6, 1e-7)
FD_ATOL, FD_RTOL = 5e-6, 1e-3
NONTRIVIAL_MIN = 1e-12


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load(name, path, expected, loaded):
    require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, "Source differs: " + str(path))
    module_name = "_utility_synthetic_qualification_" + name
    require(module_name not in sys.modules, "Fresh source identity required")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    loaded.append(module_name)
    spec.loader.exec_module(module)
    return module


def compare(actual, expected, *, gradient=False):
    if isinstance(expected, torch.Tensor):
        require(isinstance(actual, torch.Tensor) and actual.shape == expected.shape
                and actual.dtype == expected.dtype and actual.device == expected.device,
                "Tensor structure differs")
        require(bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()), "Nonfinite compared tensor")
        if not expected.is_floating_point():
            require(torch.equal(actual, expected), "Integer tensor differs")
            return 0.0
        atol, rtol = (GRAD_ATOL, GRAD_RTOL) if gradient else (VALUE_ATOL, VALUE_RTOL)
        error = float((actual.detach() - expected.detach()).abs().max()) if expected.numel() else 0.0
        require(torch.allclose(actual, expected, atol=atol, rtol=rtol), "All-coordinate mismatch: " + str(error))
        return error
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), "Dictionary keys differ")
        return max((compare(actual[k], expected[k], gradient=gradient) for k in expected), default=0.0)
    if isinstance(expected, (tuple, list)):
        require(isinstance(actual, type(expected)) and len(actual) == len(expected), "Sequence differs")
        return max((compare(a, b, gradient=gradient) for a, b in zip(actual, expected)), default=0.0)
    require(actual == expected, "Scalar/metadata differs")
    return 0.0


class OracleMeter:
    """Actual Python callback/derivative API attempts, not FLOPs/backward kernels."""
    def __init__(self):
        self.total, self.stages = {}, {}

    def add(self, kind, stage):
        self.total[kind] = self.total.get(kind, 0) + 1
        row = self.stages.setdefault(stage, {})
        row[kind] = row.get(kind, 0) + 1

    def grad(self, fn, args, *, argnums, kind, stage):
        self.add(kind, stage)
        return torch.func.grad(fn, argnums=argnums)(*args)

    def snapshot(self):
        return {"total": dict(self.total), "stages": {k: dict(v) for k, v in self.stages.items()},
                "semantics": "Attempted Python APIs; nested transforms/AD kernels/FLOPs are not equal-cost units."}


def fixture(port):
    """Identical prior 15-node edge/role pattern; new smooth graph callback."""
    dtype = torch.float64
    x = (0.3 * torch.sin(torch.arange(15 * 300, dtype=dtype) * 0.071)).reshape(15, 300)
    support = {(i, i) for i in range(15)}
    for i in range(14):
        support.update({(i, i + 1), (i + 1, i)})
    for i, j in ((0, 4), (1, 6), (2, 7), (3, 8), (4, 9), (0, 9)):
        support.update({(i, j), (j, i)})
    edges = torch.tensor(sorted(support), dtype=torch.long).transpose(0, 1).contiguous()
    s, r = torch.arange(10), torch.arange(10, 15)
    ys, yr = torch.arange(10) % 5, torch.arange(5)
    adjacency = torch.zeros(15, 15, dtype=dtype)
    adjacency[edges[0], edges[1]] = 1.0
    aggregation = adjacency / adjacency.sum(dim=1, keepdim=True)

    def values(shape, scale, offset=0.0):
        return offset + scale * torch.sin(torch.arange(math.prod(shape), dtype=dtype).reshape(shape) * 0.37 + 0.2)

    theta = {"stem.weight": values((6, 7), 0.19), "stem.bias": values((7,), 0.08),
             "stem.residual": values((6, 7), 0.11), "global_head.weight": values((7, 5), 0.21),
             "global_head.bias": values((5,), 0.07), "local_head.weight": values((7, 5), 0.13),
             "local_head.bias": values((5,), 0.05)}
    phis = []
    for member in range(4):
        phi = {}
        for name, shape in port.PRIVATE_SHAPES.items():
            offset = 1.0 if name.endswith(".S") or name == "global_head.R" else 0.0
            coordinate = torch.arange(math.prod(shape), dtype=dtype).reshape(shape)
            phi[name] = offset + (0.03 + 0.01 * member) * torch.cos(coordinate * 0.13 + 0.7 * member)
        phis.append(phi)

    def forward(core, private):
        mixed = 0.65 * x[:, :6] + 0.35 * (aggregation @ x[:, :6])
        z = mixed * (1.0 + private["stem.R"][:6])
        hidden = torch.tanh((z @ core["stem.weight"] + core["stem.bias"])
                            * private["stem.S"][:7] + private["stem.B"][:7])
        hidden = hidden + 0.15 * torch.tanh(z @ core["stem.residual"])
        return ((hidden * private["global_head.R"][:7]) @ core["global_head.weight"]
                + core["global_head.bias"]) * private["global_head.S"] + private["global_head.B"]

    return theta, tuple(phis), forward, (x, edges, s, ys, r, yr, aggregation)


class DirectReference:
    """Direct per-item gradient-dot oracle; never calls utility/sequential code."""
    def __init__(self, op, forward, pairs, s, ys, r, yr, meter):
        self.op, self.forward, self.pairs = op, forward, pairs
        self.s, self.ys, self.r, self.yr, self.meter = s, ys, r, yr, meter
        self.cfg = op.Config()

    def callback(self, stage):
        def call(core, private):
            self.meter.add("synthetic_forward_callbacks", stage)
            return self.forward(core, private)
        return call

    def grad(self, fn, args, *, argnums, kind, stage):
        return self.meter.grad(fn, args, argnums=argnums, kind=kind, stage=stage)

    def bank(self, core, rows, stage):
        raw_columns, pre_columns, post_columns = ([[] for _ in self.pairs] for _ in range(3))
        changes, probes = [], []
        for phi in rows:
            call = self.callback(stage)
            g = self.grad(self.op._own_ce, (core, phi, call, self.s, self.ys), argnums=1,
                          kind="own_CE_private_grad_transforms", stage=stage)
            before = call(core, phi)[self.s]
            probe = {n: phi[n] - self.cfg.eta_probe * g[n] for n in phi}
            after = call(core, probe)[self.s]
            for bank_index, pair in enumerate(self.pairs):
                item_values = []
                for item in range(pair.nodes.numel()):
                    def scalar_margin(theta_arg, phi_arg, pair=pair, item=item):
                        return self.op._margin(call(theta_arg, phi_arg)[self.s].unsqueeze(0), pair)[item, 0]
                    h = self.grad(scalar_margin, (core, phi), argnums=1,
                                  kind="explicit_item_margin_private_grad_transforms", stage=stage)
                    item_values.append(-self.cfg.eta_probe * sum((h[n] * g[n]).sum() for n in phi))
                raw_columns[bank_index].append(torch.stack(item_values))
                pre_columns[bank_index].append(self.op._margin(before.unsqueeze(0), pair)[:, 0])
                post_columns[bank_index].append(self.op._margin(after.unsqueeze(0), pair)[:, 0])
            probes.append(g)
            changes.append(torch.nn.functional.cross_entropy(after, self.ys)
                           - torch.nn.functional.cross_entropy(before, self.ys))
        raw = tuple(torch.stack(columns, dim=1) for columns in raw_columns)
        finite = tuple(torch.stack(post, dim=1) - torch.stack(pre, dim=1)
                       for pre, post in zip(pre_columns, post_columns))
        q = self.maps(raw, stage)
        return {"raw": raw, "q": q, "finite": finite, "probe": tuple(probes),
                "ce_change": torch.stack(changes)}

    def maps(self, raw, stage):
        result = []
        for pair, cost in zip(self.pairs, raw):
            self.meter.add("pair_primal_maps", stage)
            normalized = self.op._normalized_cost(cost, self.cfg.response_epsilon)[0]
            result.append(self.op._assignment_map(normalized, pair.laplacian, self.cfg, self.cfg.graph))
        return tuple(result)

    def adapt(self, core, rows, q, stage):
        partials, states = [], []
        call = self.callback(stage)
        for member, phi in enumerate(rows):
            g = self.grad(self.op._main_loss,
                (core, phi, q, member, call, self.s, self.ys, self.pairs, 5, self.cfg),
                argnums=1, kind="independent_Q_main_private_grad_transforms", stage=stage)
            partials.append(g)
            states.append({n: phi[n] - self.cfg.eta_private * g[n] for n in phi})
        return tuple(states), tuple(partials)

    def query(self, core, states, stage):
        return self.op._query_objective(core, states, self.callback(stage), self.r, self.yr, self.cfg)

    def from_q(self, core, rows, q, stage):
        states, _ = self.adapt(core, rows, q, stage)
        return self.query(core, states, stage)

    def from_raw(self, core, rows, raw, stage):
        return self.from_q(core, rows, self.maps(raw, stage), stage)

    def outer(self, core, rows, stage):
        bank = self.bank(core, rows, stage)
        states, main = self.adapt(core, rows, bank["q"], stage)
        return self.query(core, states, stage), bank, states, main

    def factor_credit(self, core, rows, raw_t, mode, stage):
        total = torch.zeros((), dtype=torch.float64)
        call = self.callback(stage)
        for member, phi in enumerate(rows):
            g = self.grad(self.op._own_ce, (core, phi, call, self.s, self.ys), argnums=1,
                          kind="credit_own_private_grad_transforms", stage=stage)
            def weighted(theta_arg, phi_arg):
                logits = call(theta_arg, phi_arg)[self.s].unsqueeze(0)
                return sum((t[:, member].detach() * self.op._margin(logits, pair)[:, 0]).sum()
                           for t, pair in zip(raw_t, self.pairs))
            h = self.grad(weighted, (core, phi), argnums=1,
                          kind="credit_weighted_margin_private_grad_transforms", stage=stage)
            if mode == "stop_g":
                g = {n: v.detach() for n, v in g.items()}
            elif mode == "stop_h":
                h = {n: v.detach() for n, v in h.items()}
            else:
                require(mode == "live", "Unknown product-rule check")
            total = total - self.cfg.eta_probe * sum((g[n] * h[n]).sum() for n in phi)
        return total


def tree_op(a, b, sign=1.0):
    if isinstance(a, dict):
        return {k: a[k] + sign * b[k] for k in a}
    return tuple(tree_op(x, y, sign) for x, y in zip(a, b))


def max_abs(tree):
    if isinstance(tree, dict):
        return max((float(v.detach().abs().max()) for v in tree.values()), default=0.0)
    return max((max_abs(v) for v in tree), default=0.0)


def diagnostic_reference(op, bank, pairs):
    rows = []
    for pair, raw, q, finite in zip(pairs, bank["raw"], bank["q"], bank["finite"]):
        normalized, centered, scale = op._normalized_cost(raw, op.Config().response_epsilon)
        finite_centered = finite - finite.mean(dim=1, keepdim=True)
        delta = q - 0.25
        rows.append({"classes": pair.classes, "item_count": pair.nodes.shape[0],
            "left_count": (pair.targets == pair.classes[0]).sum(), "right_count": (pair.targets == pair.classes[1]).sum(),
            "cost_kind": "first_order_private_gradient_utility", "observed_response_kind": "paid_finite_softplus_margin_response",
            "centered_response_rms": finite_centered.square().mean().sqrt(),
            "centered_cost_rms": centered.square().mean().sqrt(), "smooth_cost_scale": scale,
            "epsilon_over_scale": op.Config().response_epsilon / scale,
            "normalized_cost_rms": normalized.square().mean().sqrt(),
            "assignment_relative_rms": 4 * delta.square().mean().sqrt(),
            "assignment_max_deviation": delta.abs().max(), "mean_entropy": -(q * q.log()).sum(dim=1).mean(),
            "min_assignment": q.min(), "row_residual": (q.sum(dim=1) - 1.0).abs().max(),
            "column_residual": (q.sum(dim=0) - q.shape[0] / 4).abs().max()})
    return {"pairs": tuple(rows), "probe_own_ce_change": bank["ce_change"], "assignments": bank["q"],
            "utility_raw_costs": bank["raw"], "observed_finite_response_costs": bank["finite"]}


def inactive(core_gradient, private_gradient, before, after):
    for name, value in core_gradient.items():
        if name.startswith("local_head."):
            require(torch.count_nonzero(value).item() == 0, "Inactive shared coordinate has credit")
    active_length = {"stem.R": 6, "stem.S": 7, "stem.B": 7, "global_head.R": 7,
                     "global_head.S": 5, "global_head.B": 5}
    for g, old, new in zip(private_gradient, before, after):
        for name in old:
            length = active_length.get(name, 0)
            require(torch.count_nonzero(g[name][length:]).item() == 0, "Inactive private coordinate has credit")
            require(torch.equal(old[name][length:], new[name][length:]), "Inactive private coordinate moved")
    return {"local_shared_and_all_unused_private_coordinates_exact_zero": True,
            "inactive_committed_coordinates_exactly_preserved": True}


def parity(modules, data, meter, oracle_meter, receipt):
    op, port, seq, utility = (modules[k] for k in ("operator", "port", "sequential", "utility"))
    theta, phis, forward, inputs, pairs = data
    x, edges, s, ys, r, yr, aggregation = inputs
    physical = {"synthetic_forward_callbacks": 0, "ordinary_autograd_grad_API_attempts": 0}
    receipt["candidate_independent_observations"] = physical
    def counted(core, private):
        physical["synthetic_forward_callbacks"] += 1
        return forward(core, private)
    original_grad = torch.autograd.grad
    def counted_grad(*args, **kwargs):
        physical["ordinary_autograd_grad_API_attempts"] += 1
        return original_grad(*args, **kwargs)
    torch.autograd.grad = counted_grad
    try:
        nt, np_, info, inspection = utility._engineering_episode(
            op, PINS["operator"][1], port, PINS["port"][1], seq, PINS["sequential"][1],
            theta, phis, counted, pairs, s, ys, r, yr, engineering_authorized=True,
            counters=meter, collect_inspection=True)
    finally:
        torch.autograd.grad = original_grad
    require(torch.autograd.grad is original_grad, "Derivative API wrapper not restored")
    require(meter.total == utility.EXPECTED_EPISODE_COUNTS, "Actual candidate internal accounting differs")
    require(physical == {"synthetic_forward_callbacks": 40, "ordinary_autograd_grad_API_attempts": 63},
            "Independent candidate callback/AD counts differ")
    outer, committed = inspection["outer"], inspection["committed"]
    ref = DirectReference(op, forward, pairs, s, ys, r, yr, oracle_meter)
    shared, private = ref.grad(lambda core, rows: ref.outer(core, rows, "monolithic_gradient")[0],
        (theta, phis), argnums=(0, 1), kind="complete_outer_shared_private_grad_transforms", stage="monolithic_gradient")
    loss, bank, states, main = ref.outer(theta, phis, "independent_original_values")
    errors = {"all_shared_gradient_coordinates": compare(outer["shared_gradient"], shared, gradient=True),
              "all_inspection_private_gradient_coordinates": compare(outer["outer_private_gradients"], private, gradient=True),
              "joint_query_loss": compare(outer["virtual_query_loss"], loss),
              "all_signed_utility_costs": compare(outer["response"].raw_costs, bank["raw"]),
              "all_normalized_Q_values": compare(outer["response"].q_blocks, bank["q"]),
              "distinct_paid_finite_response_costs": compare(outer["response"].finite_response_costs, bank["finite"]),
              "all_probe_partials": compare(outer["response"].probe_private_gradients, bank["probe"], gradient=True),
              "all_main_partials": compare(outer["main_private_gradients"], main, gradient=True),
              "recomputed_main_partials": compare(outer["recomputed_main_private_partials"], main, gradient=True),
              "virtual_private_states": compare(outer["adapted"], states),
              "paid_finite_CE_change": compare(outer["response"].diagnostics["probe_own_ce_change"], bank["ce_change"])}
    errors["complete_initial_diagnostics"] = compare(outer["response"].diagnostics, diagnostic_reference(op, bank, pairs))
    expected_query = torch.stack([ref.callback("query_logit_values")(theta, state)[r] for state in states])
    errors["query_logits"] = compare(outer["query_logits"], expected_query)
    def small_query(logits):
        return op._query_objective(None, tuple(range(4)), lambda unused, member: logits[member],
                                   torch.arange(r.numel()), yr, op.Config())
    query_t = ref.grad(small_query, (expected_query,), argnums=0,
        kind="joint_query_logit_grad_transforms", stage="coupled_credit")
    errors["complete_coupled_query_cotangents"] = compare(outer["query_logit_cotangents"], query_t, gradient=True)
    q_t = ref.grad(lambda core, rows, q: ref.from_q(core, rows, q, "Q_cotangents"),
        (theta, phis, bank["q"]), argnums=2, kind="all_member_Q_grad_transforms", stage="Q_cotangents")
    raw_t = ref.grad(lambda core, rows, raw: ref.from_raw(core, rows, raw, "raw_cotangents"),
        (theta, phis, bank["raw"]), argnums=2, kind="complete_raw_to_query_grad_transforms", stage="raw_cotangents")
    errors["all_member_coupled_Q_cotangents"] = compare(outer["q_cotangents"], q_t, gradient=True)
    errors["complete_normalization_solver_raw_cotangents"] = compare(outer["raw_cost_cotangents"], raw_t, gradient=True)
    direct = ref.grad(lambda core, rows: ref.from_q(core, rows, tuple(q.detach() for q in bank["q"]), "fixed_Q_direct"),
        (theta, phis), argnums=(0, 1), kind="direct_shared_private_grad_transforms", stage="fixed_Q_direct")
    factors = {}
    for mode in ("live", "stop_g", "stop_h"):
        factors[mode] = ref.grad(lambda core, rows, mode=mode: ref.factor_credit(core, rows, raw_t, mode, "factor_" + mode),
            (theta, phis), argnums=(0, 1), kind="product_rule_shared_private_grad_transforms", stage="factor_" + mode)
    errors["both_product_rule_terms_sum_to_live"] = compare(factors["live"], tree_op(factors["stop_g"], factors["stop_h"]), gradient=True)
    errors["direct_plus_complete_utility_equals_monolithic"] = compare((shared, private), tree_op(direct, factors["live"]), gradient=True)
    errors["candidate_complete_cost_credit"] = compare(
        tree_op((outer["shared_gradient"], outer["outer_private_gradients"]), direct, -1.0), factors["live"], gradient=True)
    require(max_abs(factors["stop_g"]) > NONTRIVIAL_MIN and max_abs(factors["stop_h"]) > NONTRIVIAL_MIN,
            "Fixture failed to exercise both mixed derivative factors")
    require(max_abs(tree_op(factors["live"], factors["stop_g"], -1.0)) > NONTRIVIAL_MIN
            and max_abs(tree_op(factors["live"], factors["stop_h"], -1.0)) > NONTRIVIAL_MIN,
            "Detached-factor negative controls are not discriminating")
    ref_nt = {n: theta[n] - op.Config().eta_core * shared[n] for n in theta}
    plus_bank = ref.bank(ref_nt, phis, "original_phi_theta_plus")
    ref_np, plus_main = ref.adapt(ref_nt, phis, plus_bank["q"], "original_phi_theta_plus")
    errors.update(theta_plus_every_coordinate=compare(nt, ref_nt), committed_private_every_coordinate=compare(np_, ref_np),
        fresh_original_phi_utility_costs=compare(committed["response"].raw_costs, plus_bank["raw"]),
        fresh_original_phi_Q=compare(committed["response"].q_blocks, plus_bank["q"]),
        fresh_original_phi_main=compare(committed["main_private_gradients"], plus_main, gradient=True),
        fresh_original_phi_paid_finite_costs=compare(committed["response"].finite_response_costs, plus_bank["finite"]))
    plus_diagnostics = diagnostic_reference(op, plus_bank, pairs)
    errors["complete_committed_diagnostics"] = compare({k: info[k] for k in plus_diagnostics}, plus_diagnostics)
    for name, value in theta.items():
        if name.startswith("local_head."):
            require(torch.equal(nt[name], value), "Inactive shared committed coordinate moved")
    wrong_bank = ref.bank(ref_nt, states, "wrong_virtual_phi_negative_control")
    wrong_np, _ = ref.adapt(ref_nt, states, wrong_bank["q"], "wrong_virtual_phi_negative_control")
    require(max_abs(tree_op(ref_np, wrong_np, -1.0)) > NONTRIVIAL_MIN, "Wrong virtual-phi commitment control is not discriminating")
    unused = inactive(shared, private, phis, np_)
    inactive(outer["shared_gradient"], outer["outer_private_gradients"], phis, np_)
    require(any(float(pair.laplacian.weight.sum()) > 0.0 for pair in pairs), "Synthetic graph affinity vanished")
    require(max(float((q - 0.25).abs().max()) for q in bank["q"]) > NONTRIVIAL_MIN, "Synthetic assignment is trivial")
    require(max(float((a - b).abs().max()) for a, b in zip(bank["raw"], bank["finite"])) > NONTRIVIAL_MIN,
            "Utility and observed finite response were not distinguished")
    for q in (*bank["q"], *plus_bank["q"]):
        require(bool((q > 0).all()), "Assignment is not positive")
        compare(q.sum(dim=1), torch.ones(q.shape[0], dtype=q.dtype))
        compare(q.sum(dim=0), torch.full((4,), q.shape[0] / 4, dtype=q.dtype))
    for pair in pairs:
        require(torch.equal(pair.targets, ys[pair.nodes]), "Pair targets differ")
        expected = torch.where(pair.targets == pair.classes[0], pair.classes[1], pair.classes[0])
        require(torch.equal(pair.competitors, expected), "Both pair orientations differ")
    return {"max_absolute_errors": errors, "inactive": unused, "candidate_actual_internal_counts": meter.snapshot(),
            "candidate_independent_observations": physical, "native_reverse_constructions": 52,
            "product_rule_term_maxima": {k: max_abs(v) for k, v in factors.items()},
            "wrong_virtual_phi_commit_max_difference": max_abs(tree_op(ref_np, wrong_np, -1.0)),
            "private_inspection_never_committed": True, "native_architecture_or_resource_pass": False}, (ref, shared, private, raw_t)


def zero_gradient_factor_case(meter):
    """g=0 still retains h*Dtheta(g), with analytic scalar expectations."""
    theta = torch.zeros((), dtype=torch.float64, requires_grad=True)
    phi = torch.zeros((), dtype=torch.float64, requires_grad=True)
    own = 0.5 * (phi - theta).square()
    b = torch.nn.functional.softplus(phi + 2.0 * theta)
    meter.add("scalar_autograd_grad_API_attempts", "zero_own_gradient_factor")
    g, = torch.autograd.grad(own, (phi,), create_graph=True)
    meter.add("scalar_autograd_grad_API_attempts", "zero_own_gradient_factor")
    h, = torch.autograd.grad(b, (phi,), create_graph=True)
    meter.add("scalar_autograd_grad_API_attempts", "zero_own_gradient_factor")
    shared, private = torch.autograd.grad(-0.01 * g * h, (theta, phi))
    require(float(g) == 0.0, "Analytic case own gradient is not zero")
    return {"shared_error": compare(shared, torch.tensor(0.005, dtype=torch.float64), gradient=True),
            "private_error": compare(private, torch.tensor(-0.005, dtype=torch.float64), gradient=True),
            "nonzero_Dtheta_own_gradient_term_at_g_zero": True}


def finite_difference(data, reference, shared, private):
    theta, phis, forward, inputs, pairs = data
    rows = []
    private_lengths = {"stem.R": 6, "stem.S": 7, "stem.B": 7, "global_head.R": 7,
                       "global_head.S": 5, "global_head.B": 5}
    for which, bank, gradient in (("shared", theta, shared), ("member0_private", phis[0], private[0])):
        direction = {}
        for name, value in bank.items():
            d = torch.sin(torch.arange(value.numel(), dtype=value.dtype).reshape(value.shape) * 0.17 + 0.4)
            if name.startswith("local_head."):
                d = torch.zeros_like(value)
            elif which == "member0_private":
                d = torch.cat((d[:private_lengths[name]], torch.zeros_like(d[private_lengths[name]:])))
            direction[name] = d
        norm = sum(d.square().sum() for d in direction.values()).sqrt()
        direction = {n: d / norm for n, d in direction.items()}
        analytic = float(sum((gradient[n] * direction[n]).sum() for n in bank))
        require(abs(analytic) > NONTRIVIAL_MIN, "Fixed smooth synthetic FD direction is not discriminating")
        for epsilon in FD_SCALES:
            plus = {n: v + epsilon * direction[n] for n, v in bank.items()}
            minus = {n: v - epsilon * direction[n] for n, v in bank.items()}
            if which == "shared":
                a = reference.outer(plus, phis, "FD_shared_plus")[0]
                b = reference.outer(minus, phis, "FD_shared_minus")[0]
            else:
                a = reference.outer(theta, (plus, *phis[1:]), "FD_private_plus")[0]
                b = reference.outer(theta, (minus, *phis[1:]), "FD_private_minus")[0]
            numerical = float((a - b) / (2.0 * epsilon))
            tolerance = FD_ATOL + FD_RTOL * abs(analytic)
            error = abs(numerical - analytic)
            require(math.isfinite(numerical) and error <= tolerance, "Fixed smooth synthetic directional FD failed")
            rows.append({"coordinates": which, "epsilon": epsilon, "analytic": analytic,
                         "numerical": numerical, "absolute_error": error, "fixed_tolerance": tolerance})
    return {"spots": rows, "smooth_synthetic_only_not_native_kink_or_coarse_FD_rehabilitation": True}


def snapshot(data):
    theta, phis, forward, inputs, pairs = data
    named = {"theta." + n: v for n, v in theta.items()}
    named.update({"phi." + str(m) + "." + n: v for m, phi in enumerate(phis) for n, v in phi.items()})
    named.update({"input." + str(i): v for i, v in enumerate(inputs)})
    for i, pair in enumerate(pairs):
        for name in ("nodes", "targets", "competitors"):
            named["pair." + str(i) + "." + name] = getattr(pair, name)
        for name in ("row", "column", "weight", "degree"):
            named["laplacian." + str(i) + "." + name] = getattr(pair.laplacian, name)
    return {k: (id(v), v.data_ptr(), v.requires_grad, v.detach().clone()) for k, v in named.items()}


def unchanged(before, data):
    after = snapshot(data)
    require(before.keys() == after.keys(), "Fixture tensor set changed")
    for name, old in before.items():
        current = after[name]
        require(old[:3] == current[:3] and torch.equal(old[3], current[3]), "Fixture input/state/alias/flags changed: " + name)
    return {"fixture_values_objects_storage_aliases_and_requires_grad_unchanged": True}


def run(args, receipt, save, check):
    global torch
    loaded, old_path, old_argv = [], list(sys.path), list(sys.argv)
    python_rng, old_handler, old_timer = random.getstate(), signal.getsignal(signal.SIGALRM), signal.getitimer(signal.ITIMER_REAL)
    require(old_timer == (0.0, 0.0), "Fresh child with no active timer required")
    previous_threads = previous_interop = previous_backend = previous_rng = None
    before = data = None
    modules = {}
    candidate_meter = None
    oracle_meter = OracleMeter()
    try:
        phase = Path(args.source_root).resolve()
        modules = {k: load(k, phase / path, pin, loaded) for k, (path, pin) in PINS.items()}
        require(modules["operator"].SOURCE_RELEASED is False and modules["port"].PORT_RELEASED is False
                and modules["sequential"].SOURCE_RELEASED is False and modules["utility"].SOURCE_RELEASED is False,
                "Every original/candidate source gate must remain false")
        def expired(signum, frame):
            raise TimeoutError("Root-fixed synthetic whole-process deadline exceeded")
        signal.signal(signal.SIGALRM, expired)
        remaining = args.max_elapsed_seconds - (time.monotonic() - STARTED)
        require(remaining > 0, "Root synthetic deadline exhausted before numeric import")
        signal.setitimer(signal.ITIMER_REAL, remaining)
        if args.site_packages:
            site = Path(args.site_packages).resolve()
            require(site.is_dir(), "Explicit root runtime site-packages missing; no fallback")
            sys.path.insert(0, str(site))
        import torch
        require(not torch.cuda.is_initialized() and torch.empty(0).device.type == "cpu",
                "CPU synthetic scope requires an uninitialized CUDA runtime and CPU default device")
        previous_rng = torch.get_rng_state().clone()
        previous_threads = torch.get_num_threads()
        previous_interop = torch.get_num_interop_threads()
        previous_backend = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.set_num_threads(1)
        theta, phis, forward, inputs = fixture(modules["port"])
        x, edges, s, ys, r, yr, aggregation = inputs
        pairs = modules["port"]._sparse_pairs(modules["operator"], s, ys, edges, node_count=15, dtype=torch.float64)
        data = theta, phis, forward, inputs, pairs
        before = snapshot(data)
        candidate_meter = modules["utility"].Counters()
        result, auxiliary = check("complete_signed_cost_gradient_credit_and_original_phi_commit_parity",
            lambda: parity(modules, data, candidate_meter, oracle_meter, receipt), strip_auxiliary=True)
        reference, shared, private, raw_t = auxiliary
        check("both_factors_at_zero_own_gradient_analytic_case", lambda: zero_gradient_factor_case(oracle_meter))
        check("fixed_smooth_synthetic_shared_and_private_directional_FD",
              lambda: finite_difference(data, reference, shared, private))
        require(not torch.cuda.is_initialized(), "Synthetic qualification initialized CUDA")
        receipt["runtime"] = {"torch_version": str(torch.__version__), "torch_module_path": str(Path(torch.__file__).resolve()),
                              "device": "cpu", "dtype": "torch.float64", "intra_op_during": 1,
                              "interop_threads_unchanged": torch.get_num_interop_threads(), "CUDA_initialized": False}
    except BaseException as error:
        receipt["primary_exception"] = {"error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        receipt["candidate_attempted_counters"] = candidate_meter.snapshot() if candidate_meter is not None else None
        receipt["reference_attempted_API_counters"] = oracle_meter.snapshot()
        errors = []
        def restore(name, function):
            try:
                function()
                receipt[name] = True
            except BaseException as error:
                errors.append({"restoration": name, "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        if before is not None:
            restore("fixture_state_restoration_verified", lambda: unchanged(before, data))
        def gates():
            for k, module in modules.items():
                attr = "PORT_RELEASED" if k == "port" else "SOURCE_RELEASED"
                require(getattr(module, attr) is False and hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == PINS[k][1],
                        "Source bytes/gate changed: " + k)
        restore("all_original_and_candidate_sources_and_false_gates_unchanged", gates)
        if previous_rng is not None:
            def rng():
                require(torch.equal(torch.get_rng_state(), previous_rng), "Synthetic work changed global Torch CPU RNG")
                torch.set_rng_state(previous_rng)
                require(torch.equal(torch.get_rng_state(), previous_rng), "Torch CPU RNG not restored")
            restore("Torch_CPU_rng_unchanged_and_restored", rng)
        if previous_threads is not None:
            def threads():
                torch.set_num_threads(previous_threads)
                require(torch.get_num_threads() == previous_threads and torch.get_num_interop_threads() == previous_interop,
                        "Intra-op threads not restored or unchanged interop count differs")
            restore("intra_op_threads_restored", threads)
        if previous_backend is not None:
            def backend():
                torch.use_deterministic_algorithms(previous_backend[0], warn_only=previous_backend[1])
                require((torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()) == previous_backend,
                        "Deterministic backend state not restored")
            restore("deterministic_backend_state_restored", backend)
        def python_state():
            require(random.getstate() == python_rng, "Synthetic work changed Python RNG")
            random.setstate(python_rng)
            sys.path[:], sys.argv[:] = old_path, old_argv
            for name in loaded:
                sys.modules.pop(name, None)
            require(sys.path == old_path and sys.argv == old_argv and not any(n in sys.modules for n in loaded),
                    "Path/argv/prepared-module bindings not restored")
        restore("Python_rng_path_argv_and_prepared_module_bindings_restored", python_state)
        def timer():
            signal.signal(signal.SIGALRM, old_handler)
            signal.setitimer(signal.ITIMER_REAL, *old_timer)
            require(signal.getsignal(signal.SIGALRM) == old_handler and signal.getitimer(signal.ITIMER_REAL) == old_timer,
                    "Signal state not restored")
        restore("signal_handler_timer_restored", timer)
        receipt["restoration_errors"] = errors
        save()
    require(not receipt["restoration_errors"], "Synthetic qualifier restoration failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-authorized", action="store_true")
    parser.add_argument("--source-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--site-packages")
    parser.add_argument("--output")
    parser.add_argument("--max-elapsed-seconds", type=float)
    parser.add_argument("--max-rss-bytes", type=int)
    args = parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({"status": "DISABLED_UTILITY_SYNTHETIC_SOURCE_ONLY", "SOURCE_RELEASED": SOURCE_RELEASED, "numeric_imports": False}))
        return
    require(SOURCE_RELEASED is False, "Default-disabled qualifier source marker stays false")
    if not args.output or args.max_elapsed_seconds is None or args.max_rss_bytes is None:
        parser.error("Fresh output and root-fixed prospective elapsed/RSS limits required")
    require(math.isfinite(args.max_elapsed_seconds) and args.max_elapsed_seconds > 0 and args.max_rss_bytes > 0,
            "Positive finite root resource limits required")
    output = Path(args.output).resolve()
    output.mkdir(parents=False, exist_ok=False)
    receipt = {"schema": "utility_CPU_float64_synthetic_qualification_result_v1", "status": "RUNNING",
        "UTC": datetime.now(timezone.utc).isoformat(), "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_pins": PINS, "checks": [], "root_fixed_max_elapsed_seconds": args.max_elapsed_seconds,
        "root_fixed_max_rss_bytes": args.max_rss_bytes, "model_fits": 0, "persistent_updates": 0,
        "A_scoring": False, "VALID_TEST_access": False, "dataset_labels_read": False,
        "synthetic_targets_generated_algebraically": True, "original_six_arm_pilot_changed": False,
        "native_support_resource_or_predictive_pass_claim": False}
    def save():
        receipt["elapsed_seconds_including_imports_setup_checks_cleanup"] = time.monotonic() - STARTED
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt["process_peak_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
        tmp = output / "RESULT.tmp"
        tmp.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        tmp.replace(output / "RESULT.json")
    def check(name, function, strip_auxiliary=False):
        receipt["last_started_check"] = name
        save()
        started = time.monotonic()
        value = function()
        reported = value[0] if strip_auxiliary else value
        receipt["checks"].append({"name": name, "elapsed_seconds": time.monotonic() - started, "result": reported})
        save()
        return value
    code = 1
    try:
        save()
        run(args, receipt, save, check)
        save()
        require(receipt["elapsed_seconds_including_imports_setup_checks_cleanup"] <= args.max_elapsed_seconds
                and receipt["process_peak_rss_bytes"] <= args.max_rss_bytes, "Root synthetic whole-process resource limit exceeded")
        receipt["status"] = "PASS_CPU_FLOAT64_UTILITY_SYNTHETIC_PARITY_ONLY"
        code = 0
    except BaseException as error:
        receipt.update(status="FAIL_UTILITY_SYNTHETIC_QUALIFICATION", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        save()
        (output / "RESULT.json").chmod(0o444)
    print(json.dumps({"status": receipt["status"], "checks_completed": len(receipt["checks"]), "error": receipt.get("error")}))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
