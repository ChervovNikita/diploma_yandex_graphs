"""Pending synthetic CPU gate; no datasets, checkpoints, fits, or GPU context.

Execute only in the declared repository runtime with CUDA_VISIBLE_DEVICES=''.
This file was prepared during a source-only audit; preparation is not PASS.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def run():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Require hidden CUDA devices")
    import torch
    import torch.nn.functional as F
    import torch_geometric
    from torch_geometric.nn import SAGEConv
    from types import SimpleNamespace

    expected = json.loads((HERE / "EXPECTED_RUNTIME.json").read_text())
    actual = {"python": sys.version.split()[0], "torch": torch.__version__,
              "torch_geometric": torch_geometric.__version__,
              "SAGEConv_sha256": hashlib.sha256(inspect.getsource(SAGEConv).encode()).hexdigest(),
              "LayerNorm_sha256": hashlib.sha256(inspect.getsource(torch.nn.LayerNorm).encode()).hexdigest()}
    require(actual == expected["runtime"], "Runtime ABI differs from the recorded repository runtime")
    manifest = json.loads((HERE / "artifact_manifest.json").read_text())
    for path, digest in manifest["sha256"].items():
        require(hashlib.sha256((HERE / path).read_bytes()).hexdigest() == digest,
                "Preserved source or prototype changed: " + path)
    spec = importlib.util.spec_from_file_location("all_layer_sage_gate_adapter", HERE / "all_layer_sage_adapter.py")
    a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    checks = []

    def same(x, y, message, tol=1e-5):
        require(torch.allclose(x, y, atol=tol, rtol=tol), message)
        return float((x - y).abs().max())

    def effective(layer, x, member):
        weight = layer.shared.weight * layer.S[member, :, None] * layer.R[member, None, :]
        bias = layer.B[member]
        if layer.shared.bias is not None:
            bias = bias + layer.shared.bias * layer.S[member]
        return F.linear(x, weight, bias)

    # Explicit nonidentity factors test weight orientation and scaled shared bias.
    linear = torch.nn.Linear(5, 3).double()
    factor = a.MemberFactorLinear(linear, 4).double()
    with torch.no_grad():
        factor.R.copy_(torch.linspace(0.4, 1.6, factor.R.numel()).reshape_as(factor.R))
        factor.S.copy_(torch.linspace(-0.7, 1.1, factor.S.numel()).reshape_as(factor.S))
        factor.B.copy_(torch.linspace(-0.2, 0.3, factor.B.numel()).reshape_as(factor.B))
    x = torch.linspace(-0.9, 1.2, 35, dtype=torch.float64).reshape(7, 5)
    for member in range(4):
        factor.active_member = member
        actual_value, expected_value = factor(x), effective(factor, x, member)
        same(actual_value, expected_value, "Effective affine matrix/bias mismatch", 1e-10)
        params = tuple(factor.parameters())
        g1 = torch.autograd.grad(actual_value.square().mean(), params)
        g2 = torch.autograd.grad(expected_value.square().mean(), params)
        for left, right in zip(g1, g2):
            same(left, right, "Factor/shared derivative mismatch", 1e-10)
    checks.append("effective_weight_bias_and_gradients")

    # Directed edges include repeated entries, a self entry, and zero-in-degree nodes.
    edges = torch.tensor([[0, 0, 1, 2, 4, 5, 5], [1, 1, 1, 0, 4, 3, 0]])
    graph = SimpleNamespace(edge_index=edges, num_nodes=6)
    shape_reports = []
    for task, inputs, outputs, layers in [("ppi", 50, 121, args.ppi_layers),
                                           ("mnist", 3, 10, args.mnist_layers)]:
        options = a.BASELINE.model_options(inputs, outputs, hidden_dim=128, layers=layers, dropout=0.2)
        features = torch.linspace(-1, 1, 6 * inputs).reshape(6, inputs)
        torch.manual_seed(19)
        reference = a.BASELINE.ProjectorEnsemble(options, 4).eval()
        reference_rng = torch.get_rng_state().clone()
        torch.manual_seed(19)
        candidate = a.AllLayerSAGEAdapter(options, 4).eval()
        require(torch.equal(reference_rng, torch.get_rng_state()), "Construction RNG differs from ProjectorEnsemble")
        require(len(candidate.factor_layers) == 4 * layers, "Hidden map coverage differs")
        require(candidate.added_parameters == 13 * 4 * 128 * layers, "Added factor parameter count differs")
        parameters = list(candidate.parameters())
        require(len({p.untyped_storage().data_ptr() for p in parameters}) == len(parameters), "Aliased or repeated parameter storage")
        mapped = {}
        for name, value in candidate.model.state_dict().items():
            if "residual_modules" in name and name.endswith((".R", ".S", ".B")) and ".shared." not in name:
                continue
            mapped[name.replace(".shared.", ".")] = value
        expected_state = reference.model.state_dict()
        require(set(mapped) == set(expected_state) and all(torch.equal(mapped[k], expected_state[k]) for k in mapped),
                "Preserved shared/boundary state differs")
        with torch.no_grad():
            logits = candidate(graph, features)
            require(torch.equal(logits, reference(graph, features)), "Identity-initialized member logits differ")
        require(tuple(logits.shape) == (4, 6, outputs), "Node-logit ABI differs")

        # Matched train-mode dropout sequence and shared first-step gradients.
        reference.train(); candidate.train()
        target = torch.linspace(0, 1, 6 * outputs).reshape(6, outputs) if task == "ppi" else torch.arange(6) % outputs
        def member_loss(logits):
            if task == "ppi":
                return F.binary_cross_entropy_with_logits(logits, target[None].expand_as(logits))
            return F.cross_entropy(logits.reshape(-1, outputs), target[None].expand(4, -1).reshape(-1))
        torch.manual_seed(783)
        member_loss(reference(graph, features)).backward()
        torch.manual_seed(783)
        member_loss(candidate(graph, features)).backward()
        ref_params = dict(reference.model.named_parameters())
        shared_diffs = []
        for name, p in candidate.model.named_parameters():
            if "residual_modules" in name and name.endswith((".R", ".S", ".B")) and ".shared." not in name:
                require(p.grad is not None and torch.isfinite(p.grad).all(), "Missing/nonfinite hidden-factor gradient")
                continue
            shared_diffs.append(same(p.grad, ref_params[name.replace(".shared.", ".")].grad,
                                     "Matched shared first-step gradient differs"))
        require(any(torch.count_nonzero(p.grad) for layer in candidate.factor_layers for p in (layer.R, layer.S, layer.B)),
                "All added gradients vanish")

        # One member contributes only to its own factor rows; all shared maps remain trainable.
        candidate.eval(); candidate.zero_grad(set_to_none=True)
        candidate.forward_member(graph, features, 2).square().mean().backward()
        for layer in candidate.factor_layers:
            for p in (layer.R, layer.S, layer.B):
                require(p.grad is not None and torch.isfinite(p.grad).all(), "Missing member-specific gradient")
                require(torch.count_nonzero(p.grad[[0, 1, 3]]) == 0, "Cross-member factor gradient leakage")
        for layer in candidate.factor_layers:
            require(layer.shared.weight.grad is not None and torch.isfinite(layer.shared.weight.grad).all(),
                    "Shared weight gradient missing")

        # Native SAGE mean aggregation is retained, including multiplicity and isolates.
        conv = candidate.model.residual_modules[0].module.conv
        require(conv.aggr == "mean" and conv.root_weight and not conv.normalize and not conv.project,
                "Unexpected SAGE configuration")
        require(conv.lin_l.shared.bias is not None and conv.lin_r.shared.bias is None, "Unexpected SAGE bias ABI")
        h = torch.linspace(-0.8, 1.3, 6 * 128).reshape(6, 128)
        incoming = torch.zeros_like(h)
        counts = torch.zeros(6)
        for src, dst in edges.t().tolist():
            incoming[dst] += h[src]
            counts[dst] += 1
        incoming /= counts.clamp_min(1)[:, None]
        with torch.no_grad():
            conv.lin_l.R[2, 0] = 1.37
            conv.lin_l.S[2, 1] = -0.41
            conv.lin_l.B[2, 2] = 0.29
            conv.lin_r.B[2, 3] = -0.16
        for layer in candidate.factor_layers:
            layer.active_member = 2
        scatter_diff = same(conv(h, edges), effective(conv.lin_l, incoming, 2) + effective(conv.lin_r, h, 2),
                            "Native incoming-neighbor mean/scatter differs")

        # Distinct hidden member states arise from hidden factors even with equal boundaries.
        with torch.no_grad():
            candidate.model.input_be_block.R.fill_(1)
        hidden = []
        handle = candidate.model.residual_modules[0].register_forward_hook(lambda module, ins, out: hidden.append(out.detach().clone()))
        candidate(graph, features)
        handle.remove()
        require(len(hidden) == 4 and not torch.equal(hidden[0], hidden[2]), "Hidden member state did not differ")
        shape_reports.append({"task": task, "layers": layers, "shape": list(logits.shape),
                              "parameters": a.BASELINE.parameter_count(candidate),
                              "added_parameters": candidate.added_parameters,
                              "max_shared_gradient_error": max(shared_diffs), "max_scatter_error": scatter_diff})
    checks.append("map_coverage_identity_rng_member_gradients_native_scatter_and_hidden_states")

    # Exact existing graph readout: mean each member's normalized node logits per graph.
    batched = SimpleNamespace(edge_index=edges[:, :5], num_nodes=6, num_graphs=2,
                              batch=torch.tensor([0, 0, 0, 0, 0, 1]))
    graph_model, _ = a.make_mnist_model(layers=args.mnist_layers)
    graph_model.eval()
    features = torch.linspace(-1, 1, 18).reshape(6, 3)
    with torch.no_grad():
        node_logits = graph_model.node_model(batched, features)
        graph_logits = graph_model(batched, features)
        expected_graph = torch.stack([node_logits[:, :5].mean(1), node_logits[:, 5:].mean(1)], dim=1)
        same(graph_logits, expected_graph, "Per-member graph readout differs")
        require(tuple(graph_logits.shape) == (4, 2, 10), "Graph-logit ABI differs")
        same(graph_model.pooled_forward(batched, features), graph_logits.softmax(-1).mean(0), "MNIST pooling differs")
        ppi = torch.tensor([[[-12.0]], [[2.0]], [[2.0]], [[2.0]]])
        same(a.pooled_probabilities(ppi, "ppi"), ppi.sigmoid().mean(0), "PPI probability pooling differs")
        require(bool(a.pooled_probabilities(ppi, "ppi")[0, 0] >= 0.5) and bool(ppi.mean(0)[0, 0] < 0),
                "Probability vs raw-logit pooling regression fixture is ineffective")
    checks.append("mnist_readout_and_probability_pooling")
    require(not torch.cuda.is_initialized(), "CUDA context was created")
    return {"status": "PASS", "scope": "synthetic CPU engineering only", "runtime": actual,
            "checks": checks, "shapes": shape_reports, "dataset_loads": 0,
            "optimizer_steps": 0, "historical_score_reads": 0, "CUDA_context_created": False,
            "performance_evidence": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--ppi-layers", type=int, default=3)
    parser.add_argument("--mnist-layers", type=int, default=3)
    args = parser.parse_args()
    require(args.ppi_layers >= 1 and args.mnist_layers >= 1, "Layer counts must be positive")
    require(not args.report.exists(), "Refusing to overwrite a gate report")
    require(args.report.resolve().is_relative_to(HERE), "Gate report must remain inside this audit sibling")
    try:
        report = run()
    except Exception as exc:
        report = {"status": "FAIL", "error": type(exc).__name__ + ": " + str(exc),
                  "scope": "synthetic CPU engineering only", "performance_evidence": False}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        raise
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
