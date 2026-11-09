"""Xavier-all with both projector factors rescaled to the activation scale of +-1."""

import torch

from ablation.p02_full_ablations.models import build_ablation_model


VARIANTS = {'xavier_rms': 4}


@torch.no_grad()
def normalize_to_unit_rms(factor):
    factor.div_(factor.pow(2).mean().sqrt())


def build_model(variant, model_name, num_layers, input_dim, hidden_dim, output_dim, m, device):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant {variant!r}, expected one of {list(VARIANTS)}')
    model = build_ablation_model(model_name, num_layers, input_dim, hidden_dim, output_dim, m,
                                 'xavier_all', device)
    normalize_to_unit_rms(model.input_be_block.R)
    normalize_to_unit_rms(model.output_be_block.R)
    return model
