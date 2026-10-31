"""TABM model variants used by the ablation studies.

Reuses the residual-block stack from ``models.Model`` and only re-implements
the BatchEnsemble projector, since the ablation knobs (``tabm_inits`` and the
output-layer initialization scheme) only affect the projectors.
"""

import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MODULES, NORMALIZATION, ResidualModuleWrapper

INIT_SCHEMES = ('default', 'xavier_all', 'ones_all')


class BEBlockAblation(nn.Module):
    """BatchEnsemble linear layer with a configurable init scheme."""

    def __init__(self, in_shape, out_shape, is_first, device, tabm_inits, init_scheme):
        super().__init__()
        if init_scheme not in INIT_SCHEMES:
            raise ValueError(f'Unknown init_scheme={init_scheme!r}, expected one of {INIT_SCHEMES}')

        self.R = nn.Parameter(torch.empty(tabm_inits, in_shape, device=device))
        self.S = nn.Parameter(torch.empty(tabm_inits, out_shape, device=device))
        self.B = nn.Parameter(torch.empty(tabm_inits, out_shape, device=device))
        self.W = nn.Linear(in_shape, out_shape, bias=False)

        self.tabm_inits = tabm_inits
        self.is_first = is_first
        self.init_scheme = init_scheme
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        nn.init.xavier_uniform_(self.W.weight)
        if self.init_scheme == 'xavier_all':
            nn.init.xavier_uniform_(self.R)
        elif self.init_scheme == 'ones_all':
            self.R.fill_(1.0)
        elif self.is_first:
            self.R.bernoulli_(0.5).mul_(2).sub_(1)
        else:
            self.R.fill_(1.0)
        self.S.fill_(1.0)
        self.B.zero_()

    def forward(self, x, tabm_seed):
        x = x * self.R[tabm_seed]
        x = self.W(x)
        x = x * self.S[tabm_seed]
        x = x + self.B[tabm_seed]
        return x


class TABMAblationModel(nn.Module):
    """``models.TABMModel`` with ``tabm_inits`` and ``init_scheme`` knobs."""

    def __init__(self, model_name, num_layers, input_dim, hidden_dim, output_dim,
                 hidden_dim_multiplier, num_heads, normalization, dropout,
                 tabm_inits, init_scheme='default', device='cpu'):
        super().__init__()
        self.model_name = model_name
        self.tabm_inits = tabm_inits
        self.init_scheme = init_scheme

        norm = NORMALIZATION[normalization]
        self.input_be_block = BEBlockAblation(
            in_shape=input_dim, out_shape=hidden_dim,
            is_first=True, device=device,
            tabm_inits=tabm_inits, init_scheme=init_scheme,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

        self.residual_modules = nn.ModuleList()
        for _ in range(num_layers):
            for module in MODULES[model_name]:
                self.residual_modules.append(ResidualModuleWrapper(
                    module=module, normalization=norm, dim=hidden_dim,
                    hidden_dim_multiplier=hidden_dim_multiplier,
                    num_heads=num_heads, dropout=dropout,
                ))

        self.output_normalization = norm(hidden_dim)
        self.output_be_block = BEBlockAblation(
            in_shape=hidden_dim, out_shape=output_dim,
            is_first=False, device=device,
            tabm_inits=tabm_inits, init_scheme=init_scheme,
        )

    def forward(self, graph, x, tabm_seed):
        x = self.input_be_block(x, tabm_seed=tabm_seed)
        x = self.dropout(x)
        x = self.act(x)
        for residual_module in self.residual_modules:
            x = residual_module(graph, x)
        x = self.output_normalization(x)
        x = self.output_be_block(x, tabm_seed=tabm_seed).squeeze(1)
        return x
