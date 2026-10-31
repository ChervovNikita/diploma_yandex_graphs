"""CPU eval and gradient equivalence to a provided frozen source models.py.

Requires Torch/PyG already installed. No training step, data score, or GPU use.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from centered_paths import (MeanOperator, PackedIndependentSAGEEnsemble,
                            PackedSharedSAGEEnsemble, PlainSAGE, mean_member_loss,
                            nearest_parameter_width, source_sage_parameter_count)


def common_pairs(source, packed):
    pairs = []
    for layer, residual in enumerate(source.residual_modules):
        for name in ("weight", "bias"):
            pairs.append((getattr(residual.normalization, name), getattr(packed.norm[layer], name)))
        ff = residual.module.feed_forward_module
        for original, destination in ((residual.module.conv.lin_l, packed.blocks[layer].neighbor),
                                      (residual.module.conv.lin_r, packed.blocks[layer].root),
                                      (ff.linear_1, packed.blocks[layer].first),
                                      (ff.linear_2, packed.blocks[layer].second)):
            pairs.append((original.weight, destination.weight))
            if original.bias is not None:
                pairs.append((original.bias, destination.bias))
    for name in ("weight", "bias"):
        pairs.append((getattr(source.output_normalization, name), getattr(packed.output_norm, name)))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-models", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("SOURCE_TRANSFER_CHECKS.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite a prior check record")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(240930)
    spec = importlib.util.spec_from_file_location("frozen_source_models", args.source_models)
    source_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source_module)
    nodes, inputs, dim, classes, members, layers = 29, 9, 16, 5, 4, 3
    features = torch.randn(nodes, inputs)
    edges = torch.randint(nodes - 1, (2, 111))
    edges = torch.cat((edges, edges[:, :2], torch.tensor([[0], [0]])), dim=1)
    graph = SimpleNamespace(edge_index=edges)
    operator = MeanOperator.from_edges(edges, nodes)
    mask = torch.arange(nodes) < 13
    labels = torch.randint(classes, (nodes,))
    source_kw = dict(model_name="SAGE", num_layers=layers, input_dim=inputs,
                     hidden_dim=dim, output_dim=classes, hidden_dim_multiplier=1,
                     num_heads=1, normalization="LayerNorm", dropout=0)
    source = source_module.TABMModel(**source_kw, tabm_inits=members, device="cpu").cpu()
    with torch.no_grad():
        for boundary in (source.input_be_block, source.output_be_block):
            boundary.S.add_(0.2 * torch.randn_like(boundary.S))
            boundary.B.add_(0.1 * torch.randn_like(boundary.B))
        source.output_be_block.R.add_(0.2 * torch.randn_like(source.output_be_block.R))
    packed = PackedSharedSAGEEnsemble(inputs, classes, dim, members, layers, dropout=0).cpu()
    packed.copy_from_source_tabm(source)
    source_logits = torch.stack([source(graph, features, member) for member in range(members)])
    packed_logits = packed(operator, features)
    shared_error = (source_logits - packed_logits).abs().max().item()
    assert shared_error < 1e-10
    assert operator.widths == [members * dim] * layers
    mean_member_loss(source_logits, labels, mask).backward()
    mean_member_loss(packed_logits, labels, mask).backward()
    pairs = common_pairs(source, packed)
    for original, r, s, b, weight in (
            (source.input_be_block, packed.input_r, packed.input_s, packed.input_b, packed.input_weight),
            (source.output_be_block, packed.output_r, packed.output_s, packed.output_b, packed.output_weight)):
        pairs.extend([(original.R, r), (original.S, s), (original.B, b)])
        pairs.append((original.W.weight, weight))
    shared_gradient_error = 0.0
    for original, destination in pairs:
        expected = original.grad
        actual = destination.grad
        # Shared boundary matrices use the input-major stored convention.
        if destination is packed.input_weight or destination is packed.output_weight:
            actual = actual.T
        assert expected is not None and actual is not None
        shared_gradient_error = max(shared_gradient_error, (expected - actual).abs().max().item())
    assert shared_gradient_error < 1e-9
    assert len(pairs) == len(list(source.parameters())) == len(list(packed.parameters()))

    individual_models = [source_module.Model(**source_kw).cpu() for _ in range(members)]
    independent = PackedIndependentSAGEEnsemble(inputs, classes, dim, members, layers, dropout=0).cpu()
    independent.copy_from_source_members(individual_models)
    individual_logits = torch.stack([model(graph, features) for model in individual_models])
    independent_logits = independent(operator, features)
    independent_error = (individual_logits - independent_logits).abs().max().item()
    assert independent_error < 1e-10
    mean_member_loss(individual_logits, labels, mask).backward()
    mean_member_loss(independent_logits, labels, mask).backward()
    independent_gradient_error = 0.0
    for member, model in enumerate(individual_models):
        def check_linear(original, destination):
            nonlocal independent_gradient_error
            independent_gradient_error = max(independent_gradient_error,
                (original.weight.grad - destination.weight.grad[member].T).abs().max().item())
            if original.bias is not None:
                independent_gradient_error = max(independent_gradient_error,
                    (original.bias.grad - destination.bias.grad[member]).abs().max().item())
        def check_norm(original, destination):
            nonlocal independent_gradient_error
            for name in ("weight", "bias"):
                independent_gradient_error = max(independent_gradient_error,
                    (getattr(original, name).grad - getattr(destination, name).grad[member]).abs().max().item())
        check_linear(model.input_linear, independent.input)
        for layer, residual in enumerate(model.residual_modules):
            check_norm(residual.normalization, independent.norm[layer])
            ff = residual.module.feed_forward_module
            for original, destination in ((residual.module.conv.lin_l, independent.neighbor[layer]),
                                          (residual.module.conv.lin_r, independent.root[layer]),
                                          (ff.linear_1, independent.first[layer]),
                                          (ff.linear_2, independent.second[layer])):
                check_linear(original, destination)
        check_norm(model.output_normalization, independent.output_norm)
        check_linear(model.output_linear, independent.output)
    assert independent_gradient_error < 1e-9
    assert sum(x.numel() for x in independent.parameters()) == sum(
        x.numel() for model in individual_models for x in model.parameters())
    assert sum(x.numel() for x in independent.parameters()) == source_sage_parameter_count(
        inputs, classes, dim, layers, members)

    # Binary source squeezes its channel; the packed API retains it explicitly.
    binary_kw = {**source_kw, "output_dim": 1}
    binary_source = source_module.TABMModel(**binary_kw, tabm_inits=members, device="cpu").cpu()
    binary_packed = PackedSharedSAGEEnsemble(inputs, 1, dim, members, layers, dropout=0).cpu()
    binary_packed.copy_from_source_tabm(binary_source)
    binary_actual = binary_packed(operator, features)
    binary_expected = torch.stack([binary_source(graph, features, member)
                                   for member in range(members)])[..., None]
    binary_error = (binary_actual - binary_expected).abs().max().item()
    assert binary_error < 1e-10
    for width in (8, 44, 128, 136):
        plain = PlainSAGE(inputs, classes, width, layers=layers, dropout=0).cpu()
        assert sum(x.numel() for x in plain.parameters()) == source_sage_parameter_count(
            inputs, classes, width, layers)
    report = {
        "scope": "CPU synthetic graph; frozen source evaluation/gradient equivalence, no dataset scores",
        "status": "passed", "torch": torch.__version__, "source_models": str(args.source_models),
        "source_sha256": hashlib.sha256(args.source_models.read_bytes()).hexdigest(),
        "shared_max_logit_error": shared_error, "shared_max_gradient_error": shared_gradient_error,
        "independent_max_logit_error": independent_error,
        "independent_max_gradient_error": independent_gradient_error,
        "binary_max_logit_error": binary_error,
        "parameter_count_formula": "passed for source independent and ordinary reference widths",
        "source_files_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ["centered_paths.py", "verify_source_transfer.py"]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
