import argparse
import os
import csv
import torch
import torch.nn.functional as F
from models import Model
from datasets import DATASET_CHOICES, load_dataset, compute_metrics, get_validation_metric_name
from run_common import (
    ALL_MODELS,
    NUM_LAYERS_RANGE,
    NUM_SPLITS,
    NUM_STEPS,
    PATIENCE,
    COMMON_RESULT_FIELDNAMES,
    combo_slug,
    read_best_hparams_from_split0_log,
    set_seed,
    metrics_to_str,
    metrics_to_prefixed_dict,
    prefixed_metrics_to_str,
)


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
def evaluate(model, data, mask, is_binary, dataset_name):
    model.eval()
    out = model(data, data.x)
    return compute_metrics(out, data.y, mask, is_binary, dataset_name=dataset_name)


def train_single(model_name, num_layers, split_idx, data, train_masks, val_masks, test_masks,
                 num_targets, is_binary, device, save_dir, dataset_name, hidden_dim, lr, num_steps):
    data.train_mask = train_masks[:, split_idx].to(device)
    data.val_mask = val_masks[:, split_idx].to(device)
    data.test_mask = test_masks[:, split_idx].to(device)

    set_seed(split_idx)

    slug = combo_slug(hidden_dim, lr)
    model = Model(
        model_name=model_name,
        num_layers=num_layers,
        input_dim=data.x.size(1),
        hidden_dim=hidden_dim,
        output_dim=num_targets,
        hidden_dim_multiplier=1,
        num_heads=8,
        normalization='LayerNorm',
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

    best_val_metric = -float('inf')
    best_val_acc = 0.0
    best_step = 0
    steps_without_improvement = 0
    ckpt_path = os.path.join(save_dir, f'{model_name}_{num_layers}L_{slug}_split{split_idx}.pt')

    val_metric_name = get_validation_metric_name(dataset_name, is_binary)
    print(f'Training for {num_steps} steps')
    for step in range(1, num_steps + 1):
        train_loss = train_step(model, optimizer, data, is_binary)
        val_metrics = evaluate(model, data, data.val_mask, is_binary, dataset_name)
        
        if val_metrics['metric'] > best_val_metric:
            steps_without_improvement = 0
            if step % 10 == 0:
                best_val_metric = val_metrics['metric']
                best_val_acc = val_metrics['acc']
                best_step = step
                torch.save(model.state_dict(), ckpt_path)
        else:
            steps_without_improvement += 1

        if step % 50 == 0 or step == 1:
            print(
                f'Step {step:4d} | loss {train_loss:.4f} | '
                f'{metrics_to_str("val", val_metrics)}'
            )

        if steps_without_improvement >= PATIENCE * 100:
            print(f'Early stopping at step {step}')
            break

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    val_m = evaluate(model, data, data.val_mask, is_binary, dataset_name)
    test_m = evaluate(model, data, data.test_mask, is_binary, dataset_name)
    train_m = evaluate(model, data, data.train_mask, is_binary, dataset_name)

    print(f'\n--- Final (best ckpt @ step {best_step}) ---')
    print(f'  {metrics_to_str("train", train_m)}')
    print(f'  {metrics_to_str("val", val_m)}')
    print(f'  {metrics_to_str("test", test_m)}')
    print(f'  test_{val_metric_name}={test_m[val_metric_name]:.4f}')

    del model, optimizer
    torch.cuda.empty_cache()

    result = {
        'model': model_name,
        'num_layers': num_layers,
        'hidden_dim': hidden_dim,
        'lr': lr,
        'split': split_idx,
        'best_step': best_step,
        'val_metric_name': val_metric_name,
        'val_metric': val_m['metric'],
        'train_acc': train_m['acc'],
        'val_acc': val_m['acc'],
        'test_acc': test_m['acc'],
        'test_loss': test_m['loss'],
    }
    result.update(metrics_to_prefixed_dict('val', val_m))
    result.update(metrics_to_prefixed_dict('test', test_m))
    result.update(metrics_to_prefixed_dict('train', train_m))
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
    parser.add_argument(
        '--hidden_dim', nargs='+', type=int, default=[256, 512], help='Hidden dimension(s) to try'
    )
    parser.add_argument(
        '--lr', nargs='+', type=float, default=[3e-5, 2e-5], help='Learning rate(s) to try'
    )
    parser.add_argument('--split', type=int, default=None, choices=range(NUM_SPLITS))
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--save_dir', type=str, default='checkpoints_base')
    parser.add_argument('--log_path', type=str, default='results_base.csv')
    parser.add_argument('--stdout_log', type=str, default=None)
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS)
    parser.add_argument(
        '--search_each_split', action='store_true',
        help='Search the supplied depth/width/LR grid on every official split, as in the archived main table.'
    )
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
    fieldnames = (
        ['dataset', 'model', 'num_layers', 'hidden_dim', 'lr', 'split', 'best_step']
        + COMMON_RESULT_FIELDNAMES
    )

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
    hidden_dims = args.hidden_dim
    lrs = args.lr

    for split_idx in splits:
        for model_name in args.models:
            combo_results = []
            if split_idx == 0 or args.search_each_split:
                layers_use, hidden_use, lrs_use = layers, hidden_dims, lrs
            else:
                nl, hd, lr0 = read_best_hparams_from_split0_log(log_path, args.dataset, model_name)
                layers_use, hidden_use, lrs_use = [nl], [hd], [lr0]
                print(
                    f'[split {split_idx}] using best split-0 hparams: '
                    f'layers={nl} hidden_dim={hd} lr={lr0}'
                )

            for num_layers in layers_use:
                for hidden_dim in hidden_use:
                    for lr in lrs_use:
                        tag = (
                            f'{args.dataset} {model_name} {num_layers}L h={hidden_dim} lr={lr} '
                            f'split={split_idx}'
                        )
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
                            dataset_name=args.dataset,
                            hidden_dim=hidden_dim,
                            lr=lr,
                            num_steps=args.num_steps,
                        )
                        result['dataset'] = args.dataset
                        combo_results.append(result)

            best_idx = max(range(len(combo_results)), key=lambda i: combo_results[i]['val_metric'])
            for result in combo_results:
                writer.writerow(result)
                log_file.flush()

                test_primary_key = f'test_{result["val_metric_name"]}'
                print(
                    f'  >> num_layers={result["num_layers"]} hidden_dim={result["hidden_dim"]} '
                    f'lr={result["lr"]} best_step={result["best_step"]} '
                    f'val_{result["val_metric_name"]}={result["val_metric"]:.4f} '
                    f'test_{result["val_metric_name"]}={result[test_primary_key]:.4f}'
                )
                print(f'     {prefixed_metrics_to_str("val", result)}')
                print(f'     {prefixed_metrics_to_str("test", result)}')

            best_result = combo_results[best_idx]
            test_primary_key = f'test_{best_result["val_metric_name"]}'
            print(
                f'  >> selected best (by val): num_layers={best_result["num_layers"]} '
                f'hidden_dim={best_result["hidden_dim"]} lr={best_result["lr"]} '
                f'val_{best_result["val_metric_name"]}={best_result["val_metric"]:.4f} '
                f'test_{best_result["val_metric_name"]}={best_result[test_primary_key]:.4f}'
            )
            print(f'     {prefixed_metrics_to_str("val", best_result)}')
            print(f'     {prefixed_metrics_to_str("test", best_result)}')

    log_file.close()
    print(f'\nDone. Results saved to {log_path}, checkpoints in {save_dir}/')


if __name__ == '__main__':
    main()
