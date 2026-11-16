"""Measure actual initialization magnitudes on training nodes before any update."""

from pathlib import Path

import pandas as pd
import torch

from datasets import load_dataset
from run_common import set_seed
from ablation.common.resumable import atomic_write_json
from ablation.p05_hparam_grid.selected import build_selected
from .analyze import capture_member, stage_similarity


def initialization_diagnostics(spec, out, data_dir='data', device='cpu'):
    device = torch.device(device)
    data, train, _, _, _, targets, _ = load_dataset(
        spec['dataset'], add_self_loops=True, device=device, data_dir=data_dir)
    set_seed(spec['seed'])
    mask = train[:, spec['split']].to(device)
    data.train_mask = mask
    model, members = build_selected(spec, data, targets, device, calibrate=True)
    outputs = [capture_member(model, data, member, mask) for member in range(members)]
    activations = []
    for member, (logits, stages) in enumerate(outputs):
        for stage, value in {**stages, 'logits': logits}.items():
            activations.append({'member': member, 'stage': stage,
                                'rms': value.double().square().mean().sqrt().item(),
                                'mean': value.double().mean().item(),
                                'std': value.double().std(unbiased=False).item()})
    factors = []
    for name, parameter in model.named_parameters():
        if name.rsplit('.', 1)[-1] in {'R', 'S', 'B'} and parameter.ndim == 2:
            for member, row in enumerate(parameter):
                factors.append({'parameter': name, 'member': member,
                                'rms': row.double().square().mean().sqrt().item()})
    result = {
        'role': 'initialization_diagnostic', 'data_role': 'train',
        'dataset': spec['dataset'], 'model': spec['model'], 'split': spec['split'],
        'seed': spec['seed'], 'num_layers': spec['num_layers'],
        'method': spec.get('variant') or spec.get('method'), 'members': members,
        'nodes': int(mask.sum()), 'training_updates': 0,
        'interpretation': 'Factor RMS and activation RMS are separate observations. Matching factor RMS does not guarantee equal activation RMS.',
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(activations).to_csv(out / 'activation_rms.csv', index=False)
    pd.DataFrame(factors).to_csv(out / 'factor_rms.csv', index=False)
    pd.DataFrame(stage_similarity([item[1] for item in outputs])).to_csv(
        out / 'representation_similarity.csv', index=False)
    if hasattr(model, 'initialization_calibration'):
        atomic_write_json(model.initialization_calibration, out / 'activation_calibration.json')
    atomic_write_json(result, out / 'initialization.json')
    return result
