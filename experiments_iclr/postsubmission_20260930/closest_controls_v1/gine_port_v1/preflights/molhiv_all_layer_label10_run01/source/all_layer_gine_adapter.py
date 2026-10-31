"""Declared chemical adaptation of graph-layer BatchEnsemble, not a reproduction.

Preserves the prospective molecular GNNM's encoders, boundaries, residual
GINE blocks and graph readout, adding the exact validated hidden-affine
wrapper to both MLP maps in every block. Sequential member execution only.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn

from member_factor_linear import MemberFactorLinear

HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "closest_gine_preserved_molecular_adapter", HERE / "new_tasks/gine_adapters.py")
GINE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(GINE)
ALL_LAYER_ARM = "all_layer_be_gine_adaptation_prob"


class AllLayerGINEAdapter(GINE.MolecularProjectorEnsemble):
    """Member logits[M,graphs,1]; inherited boundaries and six hidden maps.

    Categorical atom lookup, each layer's categorical bond lookup, LayerNorm
    and scalar GINE epsilon remain shared. Their application is per member;
    node hidden states are never averaged. This fixed comparator uses M4,
    width128, three residual layers and MLP multiplier2.
    """
    def __init__(self, options=None, members=4):
        options = GINE.gine_options() if options is None else dict(options)
        expected = {"hidden_dim": 128, "num_layers": 3,
                    "hidden_dim_multiplier": 2, "output_dim": 1, "dropout": 0.1}
        if members != 4 or options != expected:
            raise ValueError("Comparator supports the fixed MolHIV M4,width128,L3 recipe only")
        super().__init__(options, members)
        factors = []
        for block in self.blocks:
            if not isinstance(block, GINE.GINEBlock):
                raise TypeError("Unexpected molecular residual block")
            mlp = block.conv.nn
            if len(mlp) != 5 or not isinstance(mlp[0], nn.Linear) or not isinstance(mlp[3], nn.Linear):
                raise TypeError("Unexpected two-affine GINE MLP")
            for index in (0, 3):
                wrapped = MemberFactorLinear(mlp[index], members)
                mlp[index] = wrapped
                factors.append(wrapped)
        # These references are already registered in blocks, not registered twice.
        self.factor_layers = tuple(factors)

    @property
    def added_parameters(self):
        return sum(layer.added_parameters for layer in self.factor_layers)

    def forward_member(self, graph, features=None, member=0):
        if not 0 <= member < self.members:
            raise IndexError("Member index out of range")
        for layer in self.factor_layers:
            layer.active_member = member
        return super().forward_member(graph, features, member)


def make_model():
    model = AllLayerGINEAdapter()
    definition = {
        "arm": ALL_LAYER_ARM, "members": 4, "hidden_dim": 128,
        "layers": 3, "MLP_multiplier": 2, "dropout": 0.1,
        "parameters": GINE.parameter_count(model),
        "added_hidden_factor_parameters": model.added_parameters,
        "factorized_hidden_maps": ["each GINE MLP affine0", "each GINE MLP affine3"],
        "initialization": "inherited GNNM boundary draws; hidden R=S=1,B=0; no extra RNG draws",
        "AtomEncoder": "shared categorical lookup",
        "BondEncoder": "one shared categorical lookup per GINE layer",
        "normalization": "shared affine LayerNorm applied separately to every member/node",
        "GINE_eps": "one shared trainable scalar per layer",
        "graph_readout": "mean node embeddings after output LayerNorm, then member boundary projector",
        "execution": "sequential full member hidden states; no hidden pooling or propagation savings",
        "primary_pooling": "mean member sigmoid probabilities after graph readout",
        "prior_method": "Kim-inspired graph-layer BatchEnsemble; GINE bond-feature adaptation",
        "thesis_reproduction": False, "novelty_claim": False,
    }
    return model, definition
