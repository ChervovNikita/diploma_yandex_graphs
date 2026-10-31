"""CPU-only structural checks. These do not report a dataset score."""

import argparse
import copy
import hashlib
import json
import platform
from pathlib import Path

import torch

from centered_paths import (CenteredPathEnsemble, IndependentNarrowPaths, MeanOperator,
                            PackedIndependentSAGEEnsemble, graph_mean_logits,
                            helmert_contrasts, mean_member_loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("STRUCTURAL_CHECKS.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite an existing check record")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(240930)
    members, nodes, inputs, dim, rank, classes, layers = 4, 23, 7, 16, 3, 5, 3
    contrast = helmert_contrasts(members)
    assert torch.allclose(contrast.T @ contrast, torch.eye(members - 1), atol=1e-12, rtol=0)
    assert contrast.sum(dim=0).abs().max().item() < 1e-12
    # The last node is isolated; a duplicate edge and self-loop test semantics.
    edges = torch.randint(nodes - 1, (2, 101))
    edges = torch.cat((edges, edges[:, :1], torch.tensor([[0], [0]])), dim=1)
    operator = MeanOperator.from_edges(edges, nodes)
    features = torch.randn(nodes, inputs)
    manual = torch.zeros_like(features)
    for target in range(nodes):
        incoming = edges[0, edges[1] == target]
        if incoming.numel():
            manual[target] = features[incoming].mean(dim=0)
    assert torch.allclose(operator(features), manual, atol=1e-12, rtol=0)
    operator.reset_counts()

    model = CenteredPathEnsemble(inputs, classes, dim, rank, members, layers, dropout=0).cpu()
    reference = copy.deepcopy(model)
    logits, trace = model(operator, features, return_trace=True)
    packed_widths = list(operator.widths)
    assert packed_widths == [dim + (members - 1) * rank] * layers
    operator.reset_counts()
    explicit_logits = reference(operator, features, explicit=True)
    explicit_widths = list(operator.widths)
    assert explicit_widths == [width for _ in range(layers) for width in (dim, members * rank)]
    error = (logits - explicit_logits).abs().max().item()
    assert error < 1e-10
    recycling_error = 0.0
    for item in trace:
        assert item["after_private"].mean(dim=1).abs().max().item() < 1e-12
        left = item["after_common"][:, None, :] + item["after_private"] @ model.expansion
        right = (item["before_common"] + item["common_delta"])[:, None, :]
        right = right + item["raw_private"] @ model.expansion
        recycling_error = max(recycling_error, (left - right).abs().max().item())
    assert recycling_error < 1e-10
    member_spread = (logits[0] - logits[1]).abs().max().item()
    assert member_spread > 1e-6
    labels = torch.randint(classes, (nodes,))
    mask = torch.arange(nodes) < 11
    mean_member_loss(logits, labels, mask).backward()
    mean_member_loss(explicit_logits, labels, mask).backward()
    gradient_error = 0.0
    for (name, p), (other_name, q) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == other_name
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        gradient_error = max(gradient_error, (p.grad - q.grad).abs().max().item())
    assert gradient_error < 1e-9

    # Suppressing private messages is a real execution and prediction ablation.
    ablated = copy.deepcopy(model)
    ablated.private_edges = False
    operator.reset_counts()
    ablated_logits = ablated(operator, features)
    ablated_widths = list(operator.widths)
    assert ablated_widths == [dim] * layers
    edge_effect = (ablated_logits - logits.detach()).abs().max().item()
    assert edge_effect > 1e-6

    saved_shapes = []
    def pack(tensor):
        saved_shapes.append(list(tensor.shape))
        return tensor
    def unpack(tensor):
        return tensor
    model.zero_grad(set_to_none=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        result = model(operator, features)
        mean_member_loss(result, labels, mask).backward()
    assert [nodes, members, dim] not in saved_shapes
    assert [members, nodes, dim] not in saved_shapes
    assert [nodes, members, inputs] not in saved_shapes
    assert [members, nodes, inputs] not in saved_shapes
    assert torch.allclose(model.mean_logits(operator, features), result.mean(dim=0), atol=1e-12, rtol=0)
    binary = CenteredPathEnsemble(inputs, 1, dim, rank, members, layers, dropout=0).cpu()
    binary_logits = binary(operator, features)
    binary_loss = mean_member_loss(binary_logits, torch.randint(2, (nodes,)), mask)
    assert binary_logits.shape == (members, nodes, 1)
    binary_loss.backward()
    multilabel = CenteredPathEnsemble(inputs, 6, dim, rank, members, layers, dropout=0).cpu()
    multilabel_loss = mean_member_loss(multilabel(operator, features),
                                      torch.randint(2, (nodes, 6)), mask)
    multilabel_loss.backward()
    packed_model = PackedIndependentSAGEEnsemble(inputs, classes, dim=6, members=4,
                                                layers=layers, dropout=0).cpu()
    operator.reset_counts()
    packed_logits = packed_model(operator, features)
    assert operator.widths == [24] * layers
    packed_explicit = packed_model(operator, features, explicit=True)
    packed_error = (packed_logits - packed_explicit).abs().max().item()
    assert packed_error < 1e-10
    mean_member_loss(packed_logits, labels, mask).backward()
    uncentered = IndependentNarrowPaths(inputs, classes, dim, rank, members,
                                        layers, dropout=0).cpu()
    operator.reset_counts()
    uncentered_logits = uncentered(operator, features)
    assert operator.widths == [dim + members * rank] * layers
    mean_member_loss(uncentered_logits, labels, mask).backward()
    batch = torch.arange(nodes) % 3
    graph_logits = graph_mean_logits(logits.detach(), batch)
    manual_graph_logits = torch.stack([logits.detach()[:, batch == graph, :].mean(dim=1)
                                      for graph in range(3)], dim=1)
    assert torch.allclose(graph_logits, manual_graph_logits, atol=1e-12, rtol=0)
    production = CenteredPathEnsemble(inputs, classes, 128, 16, 4, 2, dropout=0).cpu()
    costs = production.structural_cost()
    report = {
        "scope": "CPU synthetic graph structural checks; no dataset or GPU training",
        "status": "passed", "python": platform.python_version(), "torch": torch.__version__,
        "device": "cpu", "dtype": "float64", "packed_widths": packed_widths,
        "explicit_widths": explicit_widths, "private_edge_ablation_widths": ablated_widths,
        "max_logit_error": error, "max_gradient_error": gradient_error,
        "max_recycling_identity_error": recycling_error,
        "initial_member_max_logit_difference": member_spread,
        "private_edge_ablation_max_logit_effect": edge_effect,
        "packed_independent_max_logit_error": packed_error,
        "saved_tensor_shapes": sorted({tuple(shape) for shape in saved_shapes}),
        "proposed_128_16_costs": costs,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            Path(__file__), Path(__file__).with_name("centered_paths.py")
        ]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "saved_tensor_shapes"}, indent=2))


if __name__ == "__main__":
    main()
