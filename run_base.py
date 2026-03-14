import argparse
import os
import csv
import random
import numpy as np
import torch
import torch.nn.functional as F
from models import Model
from datasets import DATASET_CHOICES, load_dataset, compute_metrics


ALL_MODELS = ['ResNet', 'GCN', 'SAGE', 'GAT', 'GAT-sep', 'GT', 'GT-sep', 'TAG']
NUM_LAYERS_RANGE = range(1, 6)
NUM_SPLITS = 10
NUM_STEPS = 1000
PATIENCE = 3


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_step(model, optimizer, data, is_binary):
    model.train()
    optimizer.zero_grad()
    out = model(data, data.x)
    if is_binary:
        loss = F.binary_cross_entropy_with_logits(out[data.train_mask].squeeze(-1), data.y[data.train_mask])
    else:
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask].long())
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask, is_binary):
    model.eval()
    out = model(data, data.x)
    return compute_metrics(out, data.y, mask, is_binary)


def train_single(model_name, num_layers, split_idx, data, train_masks, val_masks, test_masks,
                 num_targets, is_binary, device, save_dir):
    data.train_mask = train_masks[:, split_idx].to(device)
    data.val_mask = val_masks[:, split_idx].to(device)
    data.test_mask = test_masks[:, split_idx].to(device)

    set_seed(split_idx)

    model = Model(
        model_name=model_name,
        num_layers=num_layers,
        input_dim=data.x.size(1),
        hidden_dim=512,
        output_dim=num_targets,
        hidden_dim_multiplier=1,
        num_heads=8,
        normalization='LayerNorm',
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0)

    best_val_metric = -float('inf')
    best_step = 0
    steps_without_improvement = 0
    ckpt_path = os.path.join(save_dir, f'{model_name}_{num_layers}L_split{split_idx}.pt')

    for step in range(1, NUM_STEPS + 1):
        train_loss = train_step(model, optimizer, data, is_binary)
        val_metrics = evaluate(model, data, data.val_mask, is_binary)

        if val_metrics['metric'] > best_val_metric:
            best_val_metric = val_metrics['metric']
            best_val_acc = val_metrics['acc']
            best_step = step
            steps_without_improvement = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            steps_without_improvement += 1

        if step % 50 == 0 or step == 1:
            print(f'Step {step:4d} | loss {train_loss:.4f} | val_acc {val_metrics["acc"]:.4f}')

        if steps_without_improvement >= PATIENCE * 100:
            print(f'Early stopping at step {step}')
            break

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    test_m = evaluate(model, data, data.test_mask, is_binary)
    train_m = evaluate(model, data, data.train_mask, is_binary)

    print(f'\n--- Final (best ckpt @ step {best_step}) ---')
    print(f'  train_acc={train_m["acc"]:.4f}  val_acc={best_val_acc:.4f}')
    print(f'  test_acc={test_m["acc"]:.4f}  test_loss={test_m["loss"]:.4f}')
    if not is_binary:
        print(f'  test_f1m={test_m["f1_macro"]:.4f}  test_f1w={test_m["f1_weighted"]:.4f}')
    else:
        print(f'  test_roc_auc={test_m["roc_auc"]:.4f}')

    del model, optimizer
    torch.cuda.empty_cache()

    result = {
        'model': model_name,
        'num_layers': num_layers,
        'split': split_idx,
        'best_step': best_step,
        'train_acc': train_m['acc'],
        'val_acc': best_val_acc,
        'test_acc': test_m['acc'],
        'test_loss': test_m['loss'],
    }
    if is_binary:
        result['test_roc_auc'] = test_m['roc_auc']
    else:
        result['test_f1_macro'] = test_m['f1_macro']
        result['test_f1_weighted'] = test_m['f1_weighted']
        result['test_prec_macro'] = test_m['prec_macro']
        result['test_rec_macro'] = test_m['rec_macro']
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='roman-empire', choices=DATASET_CHOICES)
    parser.add_argument('--models', nargs='+', default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument('--layers', nargs='+', type=int, default=None)
    parser.add_argument('--split', type=int, default=None, choices=range(NUM_SPLITS))
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--save_dir', type=str, default='checkpoints_base')
    parser.add_argument('--log_path', type=str, default='results_base.csv')
    parser.add_argument('--stdout_log', type=str, default=None)
    args = parser.parse_args()

    if args.stdout_log and os.path.exists(args.stdout_log):
        with open(args.stdout_log) as f:
            if 'Done.' in f.read():
                print(f'Skipping: "Done." found in {args.stdout_log}')
                return

    device = torch.device(args.device)
    add_self_loops = True

    save_dir = args.save_dir
    log_path = args.log_path
    os.makedirs(save_dir, exist_ok=True)
    if os.path.dirname(log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fieldnames = ['dataset', 'model', 'num_layers', 'split', 'best_step', 'train_acc', 'val_acc',
                  'test_acc', 'test_loss', 'test_f1_macro', 'test_f1_weighted',
                  'test_prec_macro', 'test_rec_macro', 'test_roc_auc']

    write_header = not os.path.exists(log_path)
    log_file = open(log_path, 'a', newline='')
    writer = csv.DictWriter(log_file, fieldnames=fieldnames, extrasaction='ignore')
    if write_header:
        writer.writeheader()

    print('Loading data...')
    pyg_data, train_masks, val_masks, test_masks, num_classes, num_targets, is_binary = load_dataset(
        args.dataset, add_self_loops=add_self_loops, device=device, data_dir=args.data_dir
    )

    splits = [args.split] if args.split is not None else range(NUM_SPLITS)
    layers = args.layers if args.layers is not None else list(NUM_LAYERS_RANGE)

    for split_idx in splits:
        for model_name in args.models:
            for num_layers in layers:
                tag = f'{args.dataset} {model_name} {num_layers}L split={split_idx}'
                print(f'\n{"="*60}\nTraining {tag}\n{"="*60}')

                result = train_single(
                    model_name=model_name,
                    num_layers=num_layers,
                    split_idx=split_idx,
                    data=pyg_data,
                    train_masks=train_masks,
                    val_masks=val_masks,
                    test_masks=test_masks,
                    num_targets=num_targets,
                    is_binary=is_binary,
                    device=device,
                    save_dir=save_dir,
                )
                result['dataset'] = args.dataset

                writer.writerow(result)
                log_file.flush()

                print(f'  >> best_step={result["best_step"]}  val_acc={result["val_acc"]:.4f}  test_acc={result["test_acc"]:.4f}')

    log_file.close()
    print(f'\nDone. Results saved to {log_path}, checkpoints in {save_dir}/')


if __name__ == '__main__':
    main()
