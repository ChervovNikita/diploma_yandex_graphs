"""Init-scheme ablation: train ``TABMAblationModel`` with k=4 under the chosen
output-projector init scheme on a (num_layers, hidden_dim, lr) grid for every
split. Compared in aggregation against the main-table TABM (k=4, default scheme).
"""

import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import DATASET_CHOICES
from run_common import ALL_MODELS, prefixed_metrics_to_str

from ._runner import NUM_SPLITS, NUM_STEPS, load_dataset, train_one_split
from .models_ablation import INIT_SCHEMES
from .run_k_ablation import FIELDNAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, required=True, choices=DATASET_CHOICES)
    parser.add_argument('--model', type=str, required=True, choices=ALL_MODELS)
    parser.add_argument('--init_scheme', type=str, required=True, choices=INIT_SCHEMES)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--layers', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument('--hidden_dim', nargs='+', type=int, default=[512])
    parser.add_argument('--lr', nargs='+', type=float, default=[3e-5])
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--save_dir', type=str, default='ablation/checkpoints_init')
    parser.add_argument('--log_path', type=str, required=True)
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS)
    parser.add_argument('--num_splits', type=int, default=NUM_SPLITS)
    parser.add_argument('--stdout_log', type=str, default=None)
    args = parser.parse_args()

    if args.stdout_log and os.path.exists(args.stdout_log):
        with open(args.stdout_log) as f:
            if 'Done.' in f.read():
                print(f'Skipping: "Done." found in {args.stdout_log}')
                return

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    if os.path.dirname(args.log_path):
        os.makedirs(os.path.dirname(args.log_path), exist_ok=True)

    pyg_data, train_masks, val_masks, test_masks, num_classes, num_targets, is_binary = load_dataset(
        args.dataset, add_self_loops=True, device=device, data_dir=args.data_dir
    )

    write_header = not os.path.exists(args.log_path)
    log_file = open(args.log_path, 'a', newline='')
    writer = csv.DictWriter(log_file, fieldnames=FIELDNAMES, extrasaction='ignore')
    if write_header:
        writer.writeheader()

    for split_idx in range(args.num_splits):
        for num_layers in args.layers:
            for hidden_dim in args.hidden_dim:
                for lr in args.lr:
                    tag = (f'init={args.init_scheme} {args.dataset} {args.model} '
                           f'L={num_layers} h={hidden_dim} lr={lr} split={split_idx}')
                    print(f'\n{"="*60}\nTraining {tag}\n{"="*60}')
                    result = train_one_split(
                        model_name=args.model, num_layers=num_layers, split_idx=split_idx,
                        data=pyg_data, train_masks=train_masks, val_masks=val_masks,
                        test_masks=test_masks, num_targets=num_targets, is_binary=is_binary,
                        device=device, save_dir=args.save_dir, dataset_name=args.dataset,
                        hidden_dim=hidden_dim, lr=lr, num_steps=args.num_steps,
                        k=args.k, init_scheme=args.init_scheme,
                    )
                    result['dataset'] = args.dataset
                    writer.writerow(result)
                    log_file.flush()
                    print(f'  >> {prefixed_metrics_to_str("test", result)}')

    log_file.close()
    print(f'\nDone. Results saved to {args.log_path}')


if __name__ == '__main__':
    main()
