"""Where the BatchEnsemble factors go: the variants compared in this study."""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn.dense.linear import Linear as PyGLinear

from models import TABMModel


VARIANTS = {'gnnm': 4, 'input_only': 4, 'output_only_signs': 4, 'all_layer_be': 4, 'gnnm_m1': 1}
LINEAR_TYPES = (nn.Linear, PyGLinear)


class SharedProjector(nn.Module):
    """Ordinary affine map, starting from the shared weight of the block it replaces."""

    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone())
        self.bias = nn.Parameter(torch.zeros(weight.shape[0], device=weight.device))

    def forward(self, x, tabm_seed):
        return F.linear(x, self.weight, self.bias)


class MemberContext:
    """Holds the member whose forward pass is running."""

    member = None


class MemberFactorLinear(nn.Module):
    """Identity-initialized member factors around an existing shared linear map."""

    def __init__(self, shared, members, context):
        super().__init__()
        outputs, inputs = shared.weight.shape
        self.shared = shared
        self.R = nn.Parameter(shared.weight.new_ones((members, inputs)))
        self.S = nn.Parameter(shared.weight.new_ones((members, outputs)))
        self.B = nn.Parameter(shared.weight.new_zeros((members, outputs)))
        self.context = context

    def forward(self, x):
        member = self.context.member
        return self.shared(x * self.R[member]) * self.S[member] + self.B[member]


def factorize_linear_layers(root, members, context):
    """Wrap every linear layer under ``root``; attention vectors stay shared."""
    targets = [
        (parent, name, child)
        for parent in root.modules()
        for name, child in parent.named_children()
        if isinstance(child, LINEAR_TYPES)
    ]
    wrappers = {}
    for parent, name, child in targets:
        if id(child) not in wrappers:  # One layer can be registered under two names.
            wrappers[id(child)] = MemberFactorLinear(child, members, context)
        setattr(parent, name, wrappers[id(child)])
    return list(wrappers.values())


class AllLayerBEModel(nn.Module):
    """GNNM with extra factors on the linear layers inside every residual block."""

    def __init__(self, base, members):
        super().__init__()
        self.base = base
        self.context = MemberContext()
        self.factor_layers = factorize_linear_layers(base.residual_modules, members, self.context)

    def forward(self, graph, x, tabm_seed):
        self.context.member = tabm_seed
        return self.base(graph, x, tabm_seed=tabm_seed)


def build_model(variant, model_name, num_layers, input_dim, hidden_dim, output_dim, m, device,
                hidden_dim_multiplier=1, num_heads=8, normalization='LayerNorm', dropout=0.2):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant {variant!r}, expected one of {list(VARIANTS)}')
    model = TABMModel(
        model_name=model_name, num_layers=num_layers, input_dim=input_dim,
        hidden_dim=hidden_dim, output_dim=output_dim,
        hidden_dim_multiplier=hidden_dim_multiplier, num_heads=num_heads,
        normalization=normalization, dropout=dropout, tabm_inits=m, device=device,
    )
    if variant == 'input_only':
        model.output_be_block = SharedProjector(model.output_be_block.W.weight)
    elif variant == 'output_only_signs':
        model.input_be_block = SharedProjector(model.input_be_block.W.weight)
        with torch.no_grad():
            model.output_be_block.R.bernoulli_(0.5).mul_(2).sub_(1)
    elif variant == 'all_layer_be':
        model = AllLayerBEModel(model, m)
    return model.to(device)
