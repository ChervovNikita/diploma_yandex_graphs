"""Native group factory; zero-decay predicate restricted to reachable GAT modules."""
from __future__ import annotations
import itertools
from typing import Any
import torch
import torch.nn as nn
from torch.nn import Parameter
from native_model import Model
from native_layers import BaseGraphBackbone,ResidualModule,GATModule,FFNModule

def default_zero_weight_decay_condition(module_name,module,parameter_name,parameter):
    del module_name,parameter
    return parameter_name.endswith("bias") or isinstance(module,nn.LayerNorm)

def make_parameter_groups(
    module: nn.Module,
    zero_weight_decay_condition=default_zero_weight_decay_condition,
    custom_groups: None | list[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    if custom_groups is None:
        custom_groups = []
    custom_params = frozenset(
        itertools.chain.from_iterable(group["params"] for group in custom_groups)
    )
    assert len(custom_params) == sum(len(group["params"]) for group in custom_groups), (
        "Parameters in custom_groups must not intersect"
    )

    # NOTE: We need deterministic order in zero_wd_params to load checkpoints.
    #       So, using zero_wd_params_set as zero_wd_params can be bad.
    zero_wd_params = []
    zero_wd_params_set = set()
    for mn, m in module.named_modules():
        for pn, p in m.named_parameters():
            if p not in custom_params and zero_weight_decay_condition(mn, m, pn, p):
                if p not in zero_wd_params_set:
                    zero_wd_params_set.add(p)
                    zero_wd_params.append(p)

    default_group = {
        "params": [
            p
            for p in module.parameters()
            if p not in custom_params and p not in zero_wd_params_set
        ]
    }
    return [
        default_group,
        {"params": list(zero_wd_params), "weight_decay": 0.0},
        *custom_groups,
    ]


def checked_parameter_groups(model):
    allowed=(Model,BaseGraphBackbone,ResidualModule,GATModule,FFNModule,nn.Linear,
             nn.LayerNorm,nn.Dropout,nn.GELU,nn.LeakyReLU,nn.Sequential,nn.ModuleList)
    if any(type(m) not in allowed for m in model.modules()):
        raise ValueError("Unreviewed module would change native zero-decay predicate")
    if model.num_embedding is not None or model.pearl is not None:
        raise ValueError("Embeddings/PEARL are outside native closure")
    return make_parameter_groups(model)

def make_optimizer(model,config):
    if config["type"] != "AdamW":raise ValueError("Native AdamW required")
    return torch.optim.AdamW(checked_parameter_groups(model),**{
        k:(tuple(v) if k=="betas" else v) for k,v in config.items() if k!="type"})
