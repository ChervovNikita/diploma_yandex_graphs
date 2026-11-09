"""Every architecture whose per-step cost is measured."""

import torch

from run_common import set_seed

from ablation.p01_placement.models import build_model as build_placement
from ablation.p04_shared_heads.models import build_model as build_heads
from ablation.p06_member_diversity.models import build_model as build_main


ENSEMBLE_MEMBERS = 4
LEARNING_RATE = 3e-5
VARIANTS = {
    'base': 1, 'ens': 4, 'gnnm': 4, 'gnnm_m2': 2, 'gnnm_m8': 8,
    'input_only': 4, 'output_only_signs': 4, 'all_layer_be': 4,
    'independent_projectors': 4, 'heads_single_pass': 4,
}
GNNM_SIZES = {'gnnm_m2': 2, 'gnnm': 4, 'gnnm_m8': 8}


def build_units(variant, model_name, num_layers, input_dim, hidden_dim, output_dim, split, device):
    """Return (model, members, optimizer) units; only ENS has more than one."""
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant {variant!r}, expected one of {list(VARIANTS)}')
    common = (model_name, num_layers, input_dim, hidden_dim, output_dim)
    members = VARIANTS[variant]
    models = []
    if variant in ('base', 'ens'):
        for k in range(ENSEMBLE_MEMBERS if variant == 'ens' else 1):
            set_seed(split * 1000 + k if variant == 'ens' else split)
            models.append((build_main('ens_m0', *common, 1, device), 1))
    else:
        set_seed(split)
        if variant in GNNM_SIZES:
            model = build_main('gnnm', *common, members, device)
        elif variant in ('independent_projectors', 'heads_single_pass'):
            model = build_heads(variant, *common, members, device)
        else:
            model = build_placement(variant, *common, members, device)
        models.append((model, members))
    return [
        (model, count, torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0))
        for model, count in models
    ]
