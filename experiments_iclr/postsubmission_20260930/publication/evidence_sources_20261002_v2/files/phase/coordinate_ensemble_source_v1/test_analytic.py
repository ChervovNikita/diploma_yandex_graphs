"""Draft analytic qualification tests. DO NOT RUN before root admission.

The fixtures are prescribed tiny algebra/graph cases, not study datasets. Nothing
is instantiated at import beyond definitions; unittest constructs fixtures only
after this script is explicitly executed under a separate qualification gate.
"""

from __future__ import annotations

import io
import json
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from models import (ContextLinear, _load_frozen, build_model,
                    parameter_count_formula, storage_report)


def prescribed_graph():
    # Includes repeated edges and an isolated node: mean aggregation matters.
    return SimpleNamespace(edge_index=torch.tensor(
        [[0, 0, 1, 2, 2], [1, 1, 0, 1, 2]], dtype=torch.long,
    ))


def prescribed_features(dtype=torch.float64):
    return torch.tensor([[1., -2., .5], [.2, 1., -1.], [-.7, .1, 2.], [0., .4, -.2]], dtype=dtype)


def linear_reference(module, x, member):
    """Independent folded matrix implementation, including post-transform bias."""
    if not isinstance(module, ContextLinear):
        return F.linear(x, module.weight, module.bias)
    w = module.weight
    incoming = torch.eye(module.in_features, dtype=w.dtype, device=w.device)
    outgoing = torch.eye(module.out_features, dtype=w.dtype, device=w.device)
    if module.input_index is not None:
        incoming = incoming[module.input_index[member]]
    if module.output_index is not None:
        outgoing = outgoing[module.output_index[member]]
    r = torch.ones(module.in_features, dtype=w.dtype, device=w.device) if module.R is None else module.R[member]
    s = torch.ones(module.out_features, dtype=w.dtype, device=w.device) if module.S is None else module.S[member]
    effective = torch.diag(s) @ outgoing @ w @ incoming @ torch.diag(r)
    bias = None if module.bias is None else module.bias[member]
    return F.linear(x, effective, bias)


def layernorm_reference(module, x):
    return F.layer_norm(x, module.normalized_shape, module.weight, module.bias, module.eps)


def boundary_reference(block, x, member):
    return F.linear(x * block.R[member], block.W.weight) * block.S[member] + block.B[member]


def dense_reference_member(model, graph, x, member):
    """No PyG propagate or model forward: independently forms the mean operator."""
    base = model.base
    x = F.gelu(boundary_reference(base.input_be_block, x, member))
    n = x.shape[0]
    adjacency = x.new_zeros(n, n)
    for source, target in graph.edge_index.t().tolist():
        adjacency[target, source] += 1
    adjacency = adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1)
    for residual in base.residual_modules:
        normalized = layernorm_reference(residual.normalization, x)
        module = residual.module
        message = (linear_reference(module.conv.lin_l, adjacency @ normalized, member)
                   + linear_reference(module.conv.lin_r, normalized, member))
        ff = module.feed_forward_module
        branch = linear_reference(ff.linear_1, torch.cat([normalized, message], dim=1), member)
        branch = F.gelu(branch)
        branch = linear_reference(ff.linear_2, branch, member)
        x = x + branch
    return boundary_reference(base.output_be_block,
                              layernorm_reference(base.output_normalization, x), member)


def set_indices(module, incoming, outgoing):
    with torch.no_grad():
        module.input_index.copy_(torch.tensor([incoming], dtype=torch.long))
        module.output_index.copy_(torch.tensor([outgoing], dtype=torch.long))


class AnalyticQualification(unittest.TestCase):
    def test_gather_convention_and_post_output_bias(self):
        source = nn.Linear(3, 2).double()
        with torch.no_grad():
            source.weight.copy_(torch.tensor([[1., 2., 4.], [-3., 5., 7.]], dtype=torch.float64))
            source.bias.copy_(torch.tensor([11., -13.], dtype=torch.float64))
        context = ContextLinear(source, 1, "coordinate", torch.Generator().manual_seed(9))
        set_indices(context, [2, 0, 1], [1, 0])
        with torch.no_grad():
            context.R.copy_(torch.tensor([[2., 3., 5.]], dtype=torch.float64))
            context.S.copy_(torch.tensor([[7., 11.]], dtype=torch.float64))
        x = torch.tensor([[1., 2., 3.]], dtype=torch.float64)
        torch.testing.assert_close(context(x, 0), linear_reference(context, x, 0), rtol=1e-12, atol=1e-12)
        # D_r x=(2,6,15), B gathers (15,2,6), W gives (43,7).
        # P_out swaps, D_s gives (49,473), then the private bias is added.
        torch.testing.assert_close(context(x, 0), torch.tensor([[60., 460.]], dtype=torch.float64))

    def test_coherent_gauge_cancellation_with_gelu(self):
        one_source, two_source = nn.Linear(2, 2).double(), nn.Linear(2, 2).double()
        with torch.no_grad():
            one_source.weight.copy_(torch.tensor([[1., 2.], [3., 5.]], dtype=torch.float64))
            one_source.bias.copy_(torch.tensor([.2, -.4], dtype=torch.float64))
            two_source.weight.copy_(torch.tensor([[2., 1.], [1., 3.]], dtype=torch.float64))
            two_source.bias.copy_(torch.tensor([.1, .3], dtype=torch.float64))
        generator = torch.Generator().manual_seed(21)
        one = ContextLinear(one_source, 1, "coordinate", generator)
        two = ContextLinear(two_source, 1, "coordinate", generator)
        set_indices(one, [0, 1], [1, 0])
        set_indices(two, [1, 0], [0, 1])
        r1, s1 = torch.tensor([1.2, -.7], dtype=torch.float64), torch.tensor([.8, 1.1], dtype=torch.float64)
        r2, s2 = torch.tensor([.6, 1.3], dtype=torch.float64), torch.tensor([1.4, -.9], dtype=torch.float64)
        with torch.no_grad():
            one.R[0].copy_(r1)
            one.S[0].copy_(s1.flip(0))
            one.bias[0].copy_(one_source.bias.flip(0))
            two.R[0].copy_(r2.flip(0))
            two.S[0].copy_(s2)
        x = torch.tensor([[1., -.2], [-.4, .7]], dtype=torch.float64)
        baseline = F.gelu(F.linear(x * r1, one_source.weight) * s1 + one_source.bias)
        baseline = F.gelu(F.linear(baseline * r2, two_source.weight) * s2 + two_source.bias)
        candidate = F.gelu(two(F.gelu(one(x, 0)), 0))
        torch.testing.assert_close(candidate, baseline, rtol=1e-12, atol=1e-12)

    def test_noncancelling_witness(self):
        source1, source2 = nn.Linear(2, 2, bias=False).double(), nn.Linear(2, 2, bias=False).double()
        with torch.no_grad():
            source1.weight.copy_(torch.tensor([[1., 2.], [3., 5.]], dtype=torch.float64))
            source2.weight.copy_(torch.tensor([[2., 1.], [1., 3.]], dtype=torch.float64))
        generator = torch.Generator().manual_seed(33)
        one = ContextLinear(source1, 1, "permutation", generator)
        two = ContextLinear(source2, 1, "permutation", generator)
        set_indices(one, [0, 1], [1, 0])
        set_indices(two, [0, 1], [0, 1])
        x = torch.tensor([[1., 0.]], dtype=torch.float64)
        actual = F.relu(two(F.relu(one(x, 0)), 0))[0, 0]
        baseline = F.relu(F.linear(F.relu(F.linear(x, source1.weight)), source2.weight))[0, 0]
        self.assertEqual(actual.item(), 7.)
        self.assertEqual(baseline.item(), 5.)
        # GELU also commutes with permutations, but does not turn every C into I.
        self.assertFalse(torch.allclose(F.gelu(two(F.gelu(one(x, 0)), 0)),
                                        F.gelu(F.linear(F.gelu(F.linear(x, source1.weight)), source2.weight))))

    def test_scalar_loop_and_independent_graph_reference(self):
        graph, x = prescribed_graph(), prescribed_features()
        for arm in ("original", "factor", "coordinate", "permutation", "bias_only"):
            model = build_model(arm, 3, 4, 2, 71, 901, members=3,
                                num_layers=2, dropout=0., hidden_dim_multiplier=1.5).double().eval()
            actual = model(graph, x)
            reference = torch.stack([dense_reference_member(model, graph, x, m) for m in range(3)])
            scalar = torch.stack([model.forward_member(graph, x, m) for m in range(3)])
            torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-10)
            torch.testing.assert_close(actual, scalar, rtol=1e-12, atol=1e-12)

    def test_exact_original_control_and_matched_initialization(self):
        frozen = _load_frozen()
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(17)
            original = frozen.TABMModel("SAGE", 2, 3, 4, 2, 1., 4,
                                        "LayerNorm", 0., 3, device="cpu").double().eval()
        wrapper = build_model("original", 3, 4, 2, 17, 401, members=3,
                              num_layers=2, dropout=0.).double().eval()
        graph, x = prescribed_graph(), prescribed_features()
        expected = torch.stack([original(graph, x, tabm_seed=m) for m in range(3)])
        torch.testing.assert_close(wrapper(graph, x), expected, rtol=1e-12, atol=1e-12)
        arms = {arm: build_model(arm, 3, 4, 2, 17, 401, members=3,
                                 num_layers=2, dropout=0.).double().eval()
                for arm in ("factor", "coordinate", "permutation", "bias_only")}
        for arm, model in arms.items():
            for name, value in wrapper.named_parameters():
                if name.endswith("weight"):
                    torch.testing.assert_close(dict(model.named_parameters())[name], value)
        torch.testing.assert_close(arms["factor"](graph, x), arms["bias_only"](graph, x))
        torch.testing.assert_close(arms["coordinate"](graph, x), arms["permutation"](graph, x))
        torch.testing.assert_close(arms["factor"](graph, x), wrapper(graph, x))
        # Permutation generator choice never changes shared or private init.
        alternative = build_model("coordinate", 3, 4, 2, 17, 402, members=3,
                                  num_layers=2, dropout=0.).double().eval()
        for name, value in arms["coordinate"].named_parameters():
            torch.testing.assert_close(dict(alternative.named_parameters())[name], value)

    def test_shared_gradient_is_member_gradient_sum(self):
        model = build_model("coordinate", 3, 4, 2, 25, 301, members=3,
                            num_layers=2, dropout=0.).double().eval()
        graph, x = prescribed_graph(), prescribed_features()
        shared = model.base.residual_modules[0].module.conv.lin_l.weight
        self.assertEqual(sum(id(p) == id(shared) for p in model.parameters()), 1)
        full_loss = model(graph, x).square().sum() / model.members
        full_grad = torch.autograd.grad(full_loss, shared)[0]
        member_grads = [torch.autograd.grad(model.forward_member(graph, x, m).square().sum(), shared)[0]
                        for m in range(model.members)]
        torch.testing.assert_close(full_grad, torch.stack(member_grads).mean(0), rtol=1e-10, atol=1e-10)
        model.zero_grad(set_to_none=True)
        model.forward_member(graph, x, 1).square().sum().backward()
        for module in model.modules():
            if isinstance(module, ContextLinear):
                for parameter in module.private_parameters():
                    if parameter.grad is not None:
                        torch.testing.assert_close(parameter.grad[0], torch.zeros_like(parameter.grad[0]))
                        torch.testing.assert_close(parameter.grad[2], torch.zeros_like(parameter.grad[2]))

    def test_one_member_identity_custom_sage_against_frozen_residual(self):
        # A direct independent implementation comparison, not two new forwards.
        graph = prescribed_graph()
        hidden = torch.tensor([[1., -2., .5, .7], [.2, 1., -1., -.3],
                               [-.7, .1, 2., .4], [0., .4, -.2, .9]], dtype=torch.float64)
        original = build_model("original", 3, 4, 2, 83, 801, members=1,
                               num_layers=1, dropout=0., hidden_dim_multiplier=1.5).double().eval()
        identity = build_model("bias_only", 3, 4, 2, 83, 801, members=1,
                               num_layers=1, dropout=0., hidden_dim_multiplier=1.5).double().eval()
        original_residual = original.base.residual_modules[0]
        custom_residual = identity.base.residual_modules[0]
        normalized = custom_residual.normalization(hidden)
        module = custom_residual.module
        neighbors = module.conv.propagate(graph.edge_index, x=(normalized, normalized), size=None)
        message = module.conv.lin_l(neighbors, 0) + module.conv.lin_r(normalized, 0)
        ff = module.feed_forward_module
        branch = ff.linear_1(torch.cat([normalized, message], dim=1), 0)
        custom = hidden + ff.dropout_2(ff.linear_2(ff.act(ff.dropout_1(branch)), 0))
        torch.testing.assert_close(custom, original_residual(graph, hidden), rtol=1e-12, atol=1e-12)

    def test_semantic_boundaries_checkpoint_indices_and_class_axis(self):
        model = build_model("coordinate", 3, 4, 1, 31, 501, members=3,
                            num_layers=2, dropout=0.).double().eval()
        frozen = _load_frozen()
        self.assertIsInstance(model.base.input_be_block, frozen.BEBlock)
        self.assertIsInstance(model.base.output_be_block, frozen.BEBlock)
        self.assertIsInstance(model.base.input_be_block.W, nn.Linear)
        self.assertIsInstance(model.base.output_be_block.W, nn.Linear)
        self.assertEqual(len(model.hidden_context_paths), 8)
        for path in model.hidden_context_paths:
            self.assertNotIn("input_be_block", path)
            self.assertNotIn("output_be_block", path)
        graph, x = prescribed_graph(), prescribed_features()
        self.assertEqual(tuple(model(graph, x).shape), (3, 4, 1))
        state = model.state_dict()
        self.assertEqual(sum(name.endswith("input_index") for name in state), 8)
        self.assertEqual(sum(name.endswith("output_index") for name in state), 8)
        target = build_model("coordinate", 3, 4, 1, 32, 502, members=3,
                             num_layers=2, dropout=0.).double().eval()
        # In-memory checkpoint only; no research data/artifact is created.
        payload = io.BytesIO()
        torch.save(state, payload)
        payload.seek(0)
        target.load_state_dict(torch.load(payload, map_location="cpu", weights_only=True))
        torch.testing.assert_close(target(graph, x), model(graph, x))
        for m in (-1, 3):
            with self.assertRaises(IndexError):
                model.forward_member(graph, x, m)

    def test_complete_parameter_and_index_storage_counts(self):
        for arm in ("original", "factor", "coordinate", "permutation", "bias_only",
                    "single", "untied", "heads"):
            model = build_model(arm, 3, 4, 2, 47, 601, members=3,
                                num_layers=2, dropout=0., hidden_dim_multiplier=1.5)
            report = storage_report(model)
            json.dumps(report)  # Interface must be JSON serializable.
            expected = parameter_count_formula(arm, 3, 4, 2, 3, 2, 1.5)
            self.assertEqual(report["trainable_parameters"], expected)
            self.assertEqual(report["trainable_parameters"],
                             report["shared_trainable_parameters"] + report["private_trainable_parameters"])
            expected_indices = 3 * 2 * (7 * 4 + 2 * 6) if arm in ("coordinate", "permutation") else 0
            self.assertEqual(report["index_buffer_values"], expected_indices)
            self.assertEqual(report["index_buffer_bytes"], expected_indices * 8)
            self.assertEqual(report["buffer_bytes"], expected_indices * 8)
            self.assertEqual(report["total_model_tensor_bytes"], 4 * expected + expected_indices * 8)
            self.assertEqual(report["unique_storage_bytes"], report["total_model_tensor_bytes"])
            for module in model.modules():
                if isinstance(module, ContextLinear) and module.input_index is not None:
                    for row in module.input_index:
                        torch.testing.assert_close(row.sort().values, torch.arange(module.in_features))
                    for row in module.output_index:
                        torch.testing.assert_close(row.sort().values, torch.arange(module.out_features))

    def test_caller_rng_is_unchanged_by_construction(self):
        state = torch.get_rng_state().clone()
        build_model("coordinate", 3, 4, 2, 59, 701, members=3, dropout=0.)
        torch.testing.assert_close(torch.get_rng_state(), state)


if __name__ == "__main__":
    unittest.main()
