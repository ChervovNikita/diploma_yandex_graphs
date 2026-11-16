"""Profile the exact selected architecture on an exclusively reserved GPU.

Profiling changes an in-memory copy of the selected model. Checkpoint files are
read-only. Fixed-step measurements describe execution cost rather than accuracy.
"""

import argparse
import gc
import json
from pathlib import Path

import torch

from ablation.common.resumable import atomic_write_json
from ablation.p05_hparam_grid.selected import load_selected_group
from .run import (INFERENCE_REPEATS, clock, measure, memory_of_others_mb, peak_mb,
                  pooled_logits, reset_peak, summary)


def profile_selected(job_dirs, out, data_dir='data', device='cuda:0',
                     warmup=50, measured=200, tick=None):
    if warmup < 0 or measured < 1:
        raise ValueError('Profiling requires nonnegative warmup and positive measured steps')
    device = torch.device(device)
    config, loaded, data, masks, binary = load_selected_group(job_dirs, data_dir, device)
    data.train_mask, data.val_mask = masks['train'], masks['val']
    units = [
        (model, members, torch.optim.AdamW(model.parameters(), lr=config['lr'],
                                           weight_decay=config.get('weight_decay', 0)))
        for model, members in loaded
    ]
    before = memory_of_others_mb(device)
    result = measure(units, data, binary, config['dataset'], device,
                     warmup, measured, tick or (lambda step: None))
    result['inference_with_training_state_peak_mb'] = result.pop('inference_peak_mb')
    for model, _, optimizer in units:
        optimizer.zero_grad(set_to_none=True)
        optimizer.state.clear()
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    reset_peak(device)
    inference = []
    for _ in range(INFERENCE_REPEATS):
        started = clock(device)
        pooled_logits(units, data)
        inference.append((clock(device) - started) * 1000)
    result.update({
        'inference_ms': summary(inference)[0],
        'inference_std_ms': summary(inference)[1],
        'inference_peak_mb': peak_mb(device),
        'parameter_bytes': sum(parameter.numel() * parameter.element_size()
                               for model, _, _ in units for parameter in model.parameters()),
        'state_bytes': sum(tensor.numel() * tensor.element_size()
                           for model, _, _ in units for tensor in model.state_dict().values()),
        'dataset': config['dataset'], 'model': config['model'],
        'num_layers': config['num_layers'], 'split': config['split'],
        'method': 'ens' if len(units) > 1 else config.get('variant', config.get('method')),
        'members': sum(members for _, members, _ in units),
        'job_dirs': [str(Path(path).resolve()) for path in job_dirs],
        'initial_state': 'validation_selected_checkpoint',
        'measurement_scope': 'update plus full validation every step; ensemble units execute sequentially',
        'other_gpu_memory_before_mb': before,
        'other_gpu_memory_after_mb': memory_of_others_mb(device),
        'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
    })
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(result, out / 'profile.json')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--measured', type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(profile_selected(args.jobs, args.out, args.data_dir, args.device,
                                      args.warmup, args.measured), indent=2))


if __name__ == '__main__':
    main()
