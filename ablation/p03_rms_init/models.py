"""Controlled projector initialization and explicit factor-RMS matching.

Unit factor RMS controls factor magnitude. Actual activation RMS is measured
separately because covariance and input directions also affect activations.
"""
import torch
from ablation.p02_full_ablations.models import build_ablation_model
from ablation.p01_placement.models import build_model as build_original

VARIANTS = {
    'xavier_rms': 4, 'xavier_row_rms': 4,
    'identity_both': 4, 'signs_both': 4, 'identity_input_signs_output': 4,
}

@torch.no_grad()
def normalize_to_unit_rms(factor, per_member=False):
    dims = (1,) if per_member else tuple(range(factor.ndim))
    factor.div_(factor.square().mean(dim=dims, keepdim=True).sqrt().clamp_min(1e-12))

def build_model(variant, model_name, num_layers, input_dim, hidden_dim, output_dim, m, device):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant {variant!r}')
    if variant.startswith('xavier'):
        model = build_ablation_model(model_name, num_layers, input_dim, hidden_dim,
                                     output_dim, m, 'xavier_all', device)
        for factor in (model.input_be_block.R, model.output_be_block.R):
            normalize_to_unit_rms(factor, per_member=variant == 'xavier_row_rms')
        return model
    model = build_original('gnnm', model_name, num_layers, input_dim,
                           hidden_dim, output_dim, m, device)
    with torch.no_grad():
        if variant in ('identity_both', 'identity_input_signs_output'):
            model.input_be_block.R.fill_(1)
        if variant in ('signs_both', 'identity_input_signs_output'):
            model.output_be_block.R.bernoulli_(0.5).mul_(2).sub_(1)
    return model
