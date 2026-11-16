"""Prediction and representation diagnostics for already selected node checkpoints."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from ablation.common.resumable import atomic_write_json
from ablation.p05_hparam_grid.selected import load_selected_group
from .analyze import capture_member, factor_similarity, prediction_metrics, stage_similarity


def analyse_selected(job_dirs, out, role='val', data_dir='data', device='cpu'):
    config, units, data, masks, binary = load_selected_group(job_dirs, data_dir, device)
    outputs = [capture_member(model, data, member, masks[role])
               for model, members in units for member in range(members)]
    logits = torch.stack([output[0] for output in outputs])
    metrics = prediction_metrics(logits, data.y[masks[role]], binary, config['dataset'])
    stages = stage_similarity([output[1] for output in outputs])
    parameters = []
    for unit, (model, members) in enumerate(units):
        if members > 1:
            for name, parameter in model.named_parameters():
                if name.rsplit('.', 1)[-1] in {'R', 'S', 'B'} and parameter.ndim == 2:
                    parameters.append({'unit': unit, 'parameter': name,
                                       **factor_similarity(parameter.detach().cpu())})
    result = {
        'dataset': config['dataset'], 'model': config['model'],
        'method': 'ens' if len(units) > 1 else config.get('variant', config.get('method')),
        'split': config['split'], 'seed': config['seed'], 'num_layers': config['num_layers'],
        'role': role, 'nodes': int(masks[role].sum()), 'members': logits.shape[0],
        'job_dirs': [str(Path(path).resolve()) for path in job_dirs],
        **metrics,
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(stages).to_csv(out / 'representation_similarity.csv', index=False)
    pd.DataFrame(parameters).to_csv(out / 'factor_similarity.csv', index=False)
    atomic_write_json(result, out / 'diversity.json')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--role', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    print(json.dumps(analyse_selected(args.jobs, args.out, args.role, args.data_dir, args.device),
                     indent=2))


if __name__ == '__main__':
    main()
