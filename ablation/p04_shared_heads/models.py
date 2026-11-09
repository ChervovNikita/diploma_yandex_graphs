"""Shared backbone with ordinary per-member matrices instead of BatchEnsemble."""

import torch
from torch import nn

from ablation.p01_placement.models import SharedProjector
from models import TABMModel


VARIANTS = {'independent_projectors': 4, 'heads_single_pass': 4}


class IndependentProjectors(nn.Module):
    """One ordinary affine map per member, as in a plain ensemble."""

    def __init__(self, input_dim, output_dim, members):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(input_dim, output_dim) for _ in range(members))
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, tabm_seed):
        return self.layers[tabm_seed](x)


class HeadsSinglePassModel(nn.Module):
    """One backbone pass and one dropout mask per step, then one readout per member."""

    def __init__(self, base, members):
        super().__init__()
        self.base = base
        self.members = members

    def forward_all(self, graph, x):
        base = self.base
        x = base.input_be_block(x, tabm_seed=0)
        x = base.dropout(x)
        x = base.act(x)
        for residual_module in base.residual_modules:
            x = residual_module(graph, x)
        x = base.output_normalization(x)
        return torch.stack([base.output_be_block(x, tabm_seed=i).squeeze(1)
                            for i in range(self.members)])

    def forward(self, graph, x, tabm_seed):
        return self.forward_all(graph, x)[tabm_seed]


def build_model(variant, model_name, num_layers, input_dim, hidden_dim, output_dim, m, device):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant {variant!r}, expected one of {list(VARIANTS)}')
    base = TABMModel(
        model_name=model_name, num_layers=num_layers, input_dim=input_dim,
        hidden_dim=hidden_dim, output_dim=output_dim, hidden_dim_multiplier=1, num_heads=8,
        normalization='LayerNorm', dropout=0.2, tabm_inits=m, device=device,
    )
    base.output_be_block = IndependentProjectors(hidden_dim, output_dim, m)
    if variant == 'independent_projectors':
        base.input_be_block = IndependentProjectors(input_dim, hidden_dim, m)
        return base.to(device)
    base.input_be_block = SharedProjector(base.input_be_block.W.weight)
    return HeadsSinglePassModel(base, m).to(device)
