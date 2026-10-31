"""CPU-only synthetic functional/gradient checks; no dataset or model scoring."""
import copy
import json
import torch
from torch import nn
from torch_geometric.data import Batch, Data
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

from gine_adapters import (
    MolecularIndependentEnsemble, MolecularProjectorEnsemble,
    MolecularSharedMultihead, MolecularSingle, copied_private,
    gine_options, parameter_count,
)


def synthetic_batch(output_dim):
    graphs = []
    for n in [5, 3, 1]:
        atom = torch.stack([torch.randint(dim, (n,)) for dim in get_atom_feature_dims()], dim=1)
        if n > 1:
            src = torch.arange(n - 1)
            dst = src + 1
            edges = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
            bond = torch.stack([torch.randint(dim, (n - 1,))
                                for dim in get_bond_feature_dims()], dim=1)
            bond = torch.cat([bond, bond], dim=0)
        else:
            edges, bond = torch.empty((2, 0), dtype=torch.long), torch.empty((0, 3), dtype=torch.long)
        y = torch.randint(2, (1, output_dim)).double()
        graphs.append(Data(x=atom, edge_index=edges, edge_attr=bond, y=y))
    return Batch.from_data_list(graphs)


def assert_close(a, b, name, atol=1e-10, rtol=1e-9):
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(f"{name}: max error {(a-b).abs().max().item()}")


def run_checks(output_dim):
    torch.manual_seed(20260930 + output_dim)
    batch = synthetic_batch(output_dim)
    options = gine_options(output_dim=output_dim, hidden_dim=16, num_layers=3, dropout=0.0)
    tied = MolecularProjectorEnsemble(options, members=4).double()
    private = copied_private(tied)
    selective = copied_private(tied, [1])
    logits = tied(batch, batch.x)
    assert logits.shape == (4, 3, output_dim)
    assert_close(logits, private(batch, batch.x), "initial all-private logits")
    assert_close(logits, selective(batch, batch.x), "initial selected-private logits")
    assert_close(tied.pooled_forward(batch), logits.mean(0), "raw-logit pooling")
    for member in range(4):
        assert_close(logits[member], tied.forward_member(batch, batch.x, member), "member interface")
    target = batch.y.unsqueeze(0).expand_as(logits)
    nn.functional.binary_cross_entropy_with_logits(logits, target).backward()
    nn.functional.binary_cross_entropy_with_logits(private(batch, batch.x), target).backward()
    tied_parameters, private_parameters = dict(tied.named_parameters()), dict(private.named_parameters())
    for name, parameter in tied_parameters.items():
        if name.startswith("blocks."):
            _, block, tail = name.split(".", 2)
            summed = sum(private_parameters[f"private_blocks.{block}.{member}.{tail}"].grad
                         for member in range(4))
            assert_close(parameter.grad, summed, "shared gradient equals sum of private gradients: " + name)
        else:
            assert_close(parameter.grad, private_parameters[name].grad, "boundary/encoder gradient: " + name)
    blocks = tied.probe_blocks()
    assert len(blocks) == 3
    for i in range(3):
        bonds = set(id(p) for p in tied.blocks[i].bond_encoder.parameters())
        assert bonds <= set(id(p) for p in blocks[str(i)])
    for model in [private, selective]:
        try:
            model.probe_blocks()
        except ValueError:
            pass
        else:
            raise AssertionError("Private model must not expose a fully shared warmup probe")
    atom_count = sum(p.numel() for p in tied.encoder.parameters())
    for model in [private, selective]:
        assert sum(p.numel() for p in model.encoder.parameters()) == atom_count
        assert not any(name.startswith("blocks.1.") for name, _ in model.named_parameters())
    # AtomEncoder stays identical while only the chosen block becomes private.
    for a, b in zip(tied.encoder.parameters(), selective.encoder.parameters()):
        assert_close(a, b, "atom encoder unchanged after privatization")
    other_models = [MolecularSingle(options).double(), MolecularSharedMultihead(options, 4).double(),
                    MolecularIndependentEnsemble(options, 4).double()]
    for model in other_models:
        result = model(batch, batch.x)
        assert result.shape == (model.members, 3, output_dim)
        assert_close(model.pooled_forward(batch), result.mean(0), "control pooling")
        nn.functional.binary_cross_entropy_with_logits(result, batch.y.unsqueeze(0).expand_as(result)).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    # Evaluation-mode forward and member interfaces agree for multihead control.
    shared_heads = other_models[1].eval()
    assert_close(shared_heads(batch), torch.stack([shared_heads.forward_member(batch, batch.x, i)
                                                  for i in range(shared_heads.members)]), "one-path multihead interface")
    assert parameter_count(private) > parameter_count(tied)
    assert parameter_count(tied) < parameter_count(other_models[2])
    return {"targets": output_dim, "initial_and_gradient_checks": "passed",
            "synthetic_parameters": {"TIED": parameter_count(tied), "copied_PRIVATE": parameter_count(private),
                                      "BASE": parameter_count(other_models[0]),
                                      "shared_multihead": parameter_count(other_models[1]),
                                      "ENS": parameter_count(other_models[2])}}


if __name__ == "__main__":
    torch.set_num_threads(1)
    print(json.dumps({"device": "cpu", "dataset_accessed": False,
                      "checks": [run_checks(1), run_checks(2)]}, indent=2))
