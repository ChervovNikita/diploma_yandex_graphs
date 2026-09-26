"""Choose a wider ordinary SAGE by A100 training-step time alone.

The predeclared candidates are widths 896, 1024, and 1152. The target is
GNNM with four members at width 512. All measurements use Roman Empire split 0,
five graph layers, learning rate 3e-5, 50 warmup and 200 CUDA-synchronized
training steps. No validation or test score is read to select a width.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ablation.run_cost import profile_variant
from datasets import load_dataset

OUT = ROOT / 'experiments_iclr' / 'strong_base_results'
WIDTHS = (896, 1024, 1152)
FIELDS = ['dataset', 'model', 'variant', 'hidden_dim', 'num_layers', 'lr',
          'num_params', 'mean_step_ms', 'std_step_ms', 'peak_allocated_mib',
          'device']


def main():
    device = torch.device('cuda:0')
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for runtime matching')
    data, train, _, _, _, output_dim, is_binary = load_dataset(
        'roman-empire', add_self_loops=True, device=device, data_dir='data')
    data.train_mask = train[:, 0].to(device)
    device_name = torch.cuda.get_device_name(device)
    rows = []
    for variant, width in [('TABM_k4', 512), ('BASE', 512)] + [
            ('BASE', width) for width in WIDTHS]:
        params, times, peak = profile_variant(
            variant, 'SAGE', 5, width, 3e-5, data, output_dim, is_binary, device,
            weight_decay=0.0)
        row = {
            'dataset': 'roman-empire', 'model': 'SAGE', 'variant': variant,
            'hidden_dim': width, 'num_layers': 5, 'lr': 3e-5,
            'num_params': params, 'mean_step_ms': float(np.mean(times)),
            'std_step_ms': float(np.std(times, ddof=0)),
            'peak_allocated_mib': peak, 'device': device_name,
        }
        rows.append(row)
        print('MEASURED', json.dumps(row, sort_keys=True), flush=True)
    target = rows[0]['mean_step_ms']
    candidates = [r for r in rows if r['variant'] == 'BASE'
                  and r['hidden_dim'] in WIDTHS]
    chosen = min(candidates,
                 key=lambda r: (abs(math.log(r['mean_step_ms'] / target)),
                                r['hidden_dim']))
    selection = {
        'selection_rule': 'Among predeclared widths 896, 1024, 1152, choose the ordinary SAGE whose synchronized A100 training-step time is closest on a multiplicative scale to four-member GNNM at width 512. No validation or test metric participates.',
        'profile_protocol': 'Roman Empire official split 0, depth 5, lr 3e-5, AdamW weight decay 0, 50 warmup then 200 CUDA-synchronized training steps',
        'device': device_name, 'torch_version': torch.__version__,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'target_step_ms': target,
        'selected_width': int(chosen['hidden_dim']),
        'selected_step_ms': chosen['mean_step_ms'],
        'selected_to_target_step_ratio': chosen['mean_step_ms'] / target,
        'selected_num_params': int(chosen['num_params']),
        'gnnm_num_params': int(rows[0]['num_params']),
        'measurements': rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'strong_base_profile.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (OUT / 'strong_base_selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    print('SELECTED', json.dumps(selection, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
