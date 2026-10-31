"""Controlled node-task adapters around the immutable GNNM source snapshot.

All models return [members, nodes, classes]. These are new-study controls;
they do not recreate or change any historical reported score.
"""
import copy
import importlib.util
from pathlib import Path

import torch
from torch import nn


_SOURCE = Path(__file__).resolve().parents[1] / "source_snapshot" / "models.py"
_SPEC = importlib.util.spec_from_file_location("gnnm_frozen_models_20260930", _SOURCE)
_MODELS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODELS)


def model_options(input_dim, output_dim, hidden_dim=128, layers=2, dropout=0.2):
    return dict(model_name="SAGE", num_layers=layers, input_dim=input_dim,
                hidden_dim=hidden_dim, output_dim=output_dim,
                hidden_dim_multiplier=1, num_heads=4,
                normalization="LayerNorm", dropout=dropout)


def _matrix(logits, output_dim):
    return logits.unsqueeze(-1) if output_dim == 1 else logits


class SingleModel(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.model = _MODELS.Model(**options)
        self.output_dim = options["output_dim"]

    def forward(self, graph, features):
        return _matrix(self.model(graph, features), self.output_dim).unsqueeze(0)


class IndependentEnsemble(nn.Module):
    """Independent initialization, explicitly distinct from copied-stack control."""
    def __init__(self, options, members=4):
        super().__init__()
        self.models = nn.ModuleList([_MODELS.Model(**options) for _ in range(members)])
        self.output_dim = options["output_dim"]

    def forward(self, graph, features):
        return torch.stack([_matrix(m(graph, features), self.output_dim) for m in self.models])


class ProjectorEnsemble(nn.Module):
    def __init__(self, options, members=4):
        super().__init__()
        self.model = _MODELS.TABMModel(**options, tabm_inits=members, device="cpu")
        self.members = members
        self.output_dim = options["output_dim"]

    def forward(self, graph, features):
        return torch.stack([_matrix(self.model(graph, features, i), self.output_dim)
                            for i in range(self.members)])


class SelectiveProjectorEnsemble(nn.Module):
    """Share boundary projectors; optionally copy selected residual blocks.

Construct from an existing ProjectorEnsemble to preserve its current state.
Privatization removes unused shared blocks from the registered parameters.
Call privatize before constructing the optimizer, or rebuild the optimizer
and explicitly record how warmup state is transferred. No optimizer policy
is implicit in this adapter.
"""
    def __init__(self, original, private_indices=()):
        super().__init__()
        self.model = copy.deepcopy(original.model)
        self.members = original.members
        self.output_dim = original.output_dim
        self.private_blocks = nn.ModuleDict()
        self.privatize(private_indices)

    def privatize(self, indices):
        for index in sorted(set(indices)):
            key = str(index)
            if index < 0 or index >= len(self.model.residual_modules):
                raise ValueError("Residual block index out of range")
            if key in self.private_blocks:
                raise ValueError("Block already private")
            module = self.model.residual_modules[index]
            self.private_blocks[key] = nn.ModuleList([
                copy.deepcopy(module) for _ in range(self.members)
            ])
            self.model.residual_modules[index] = nn.Identity()

    def forward_member(self, graph, features, member):
        x = self.model.input_be_block(features, tabm_seed=member)
        x = self.model.act(self.model.dropout(x))
        for index, shared in enumerate(self.model.residual_modules):
            key = str(index)
            block = self.private_blocks[key][member] if key in self.private_blocks else shared
            x = block(graph, x)
        x = self.model.output_normalization(x)
        x = self.model.output_be_block(x, tabm_seed=member)
        return x

    def forward(self, graph, features):
        return torch.stack([self.forward_member(graph, features, i)
                            for i in range(self.members)])

    def probe_blocks(self):
        if self.private_blocks:
            raise ValueError("Probe a fully shared warmup state")
        return {str(i): tuple(block.parameters())
                for i, block in enumerate(self.model.residual_modules)}


class SharedTrunkMultihead(nn.Module):
    """One propagation path, independently initialized linear output heads."""
    def __init__(self, options, members=4):
        super().__init__()
        self.trunk = _MODELS.Model(**options)
        self.trunk.output_linear = nn.Identity()
        self.heads = nn.ModuleList([nn.Linear(options["hidden_dim"], options["output_dim"])
                                    for _ in range(members)])

    def forward(self, graph, features):
        x = self.trunk.input_linear(features)
        x = self.trunk.act(self.trunk.dropout(x))
        for block in self.trunk.residual_modules:
            x = block(graph, x)
        x = self.trunk.output_normalization(x)
        return torch.stack([head(x) for head in self.heads])


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())
