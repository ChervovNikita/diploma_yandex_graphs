"""Paired Xavier controls with fixed shared weights and optional endpoint RMS matching."""

import copy

import torch
from torch import nn

from ablation.p01_placement.models import build_model as build_original


VARIANTS = {'xavier_same_weights', 'xavier_activation_rms'}


def build_model(variant, model_name, num_layers, input_dim, hidden_dim, output_dim, m, device):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown paired initialization variant {variant}')
    model = build_original('gnnm', model_name, num_layers, input_dim, hidden_dim,
                           output_dim, m, device)
    model.original_initial_factors = {
        'input': model.input_be_block.R.detach().clone(),
        'output': model.output_be_block.R.detach().clone(),
    }
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    # The paired controls preserve the original post-construction dropout RNG.
    with torch.random.fork_rng(devices=devices):
        nn.init.xavier_uniform_(model.input_be_block.R)
        nn.init.xavier_uniform_(model.output_be_block.R)
    return model


def rms(tensor):
    value = tensor.double().square().mean().sqrt()
    if not torch.isfinite(value):
        raise FloatingPointError('Nonfinite activation RMS during initialization')
    return float(value)


@torch.no_grad()
def calibrate_model(model, data, members):
    """Match training-node input-projector and final-logit RMS route by route."""
    mask = data.train_mask
    if mask.dtype != torch.bool or not mask.any():
        raise ValueError('Initialization calibration requires a nonempty boolean TRAIN mask')
    reference = copy.deepcopy(model)
    reference.input_be_block.R.copy_(model.original_initial_factors['input'])
    reference.output_be_block.R.copy_(model.original_initial_factors['output'])
    training = model.training
    model.eval()
    reference.eval()
    rows = []

    def scale(factor, target, current):
        if current == 0:
            if target != 0:
                raise ValueError('Cannot match a positive RMS from zero activations')
            return 1.0
        multiplier = target / current
        factor.mul_(multiplier)
        return multiplier

    for member in range(members):
        target_input = rms(reference.input_be_block(data.x, tabm_seed=member)[mask])
        target_output = rms(reference(data, data.x, tabm_seed=member)[mask])
        raw_input = rms(model.input_be_block(data.x, tabm_seed=member)[mask])
        input_multiplier = scale(model.input_be_block.R[member], target_input, raw_input)
        input_matched_output = rms(model(data, data.x, tabm_seed=member)[mask])
        output_multiplier = scale(model.output_be_block.R[member], target_output, input_matched_output)
        rows.append({
            'member': member, 'target_input_rms': target_input,
            'raw_input_rms': raw_input,
            'matched_input_rms': rms(model.input_be_block(data.x, tabm_seed=member)[mask]),
            'target_logit_rms': target_output,
            'input_matched_logit_rms': input_matched_output,
            'matched_logit_rms': rms(model(data, data.x, tabm_seed=member)[mask]),
            'input_R_multiplier': input_multiplier, 'output_R_multiplier': output_multiplier,
        })
    model.train(training)
    model.initialization_calibration = {
        'role': 'initialization', 'data_role': 'train', 'labels_used': False,
        'training_nodes': int(mask.sum()), 'training_updates': 0, 'members': rows,
        'reference': 'same-seed original GNNM with identical W/backbone/S/B',
        'matched_locations': ['input projector before GELU/dropout', 'final logits'],
        'intermediate_activations': 'measured separately; their RMS is unconstrained',
    }
    return model
