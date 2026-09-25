"""Add member logits to older saved predictions without retraining."""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset
from projector_controls import all_logits, make_model, test_analysis

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / 'experiments_iclr' / 'results' / 'projector_controls.csv'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    device = torch.device(args.device)
    with RESULT.open(newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        pred_file = ROOT / row['prediction_file']
        if not pred_file.is_file():
            continue
        with np.load(pred_file) as z:
            if 'member_logits' in z:
                continue
        data, _, _, test_masks, _, output_dim, is_binary = load_dataset(
            row['dataset'], add_self_loops=True, device=device, data_dir='data'
        )
        if is_binary:
            continue
        split = int(row['split'])
        data.test_mask = test_masks[:, split].to(device)
        config = argparse.Namespace(
            model=row['model'], num_layers=int(row['num_layers']),
            hidden_dim=int(row['hidden_dim']), m=int(row['m'])
        )
        model = make_model(config, row['variant'], data.x.size(1), output_dim, device)
        state = torch.load(ROOT / row['checkpoint'], map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            logits = all_logits(model, row['variant'], data, int(row['m']))
            _, pred = test_analysis(logits, data.y, data.test_mask, data.edge_index)
        np.savez_compressed(pred_file, **pred)
        print('Re-exported', pred_file.relative_to(ROOT))
        del data, model, logits
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
