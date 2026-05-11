"""Shared training/eval helpers for the TABM ablation studies."""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import compute_metrics, get_validation_metric_name, load_dataset
from run_common import (
    NUM_SPLITS,
    NUM_STEPS,
    PATIENCE,
    combo_slug,
    metrics_to_prefixed_dict,
    metrics_to_str,
    read_best_hparams_from_split0_log,
    set_seed,
)

from .models_ablation import TABMAblationModel


def train_step_tabm(model, data, optimizer, is_binary, k):
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    for i in range(k):
        out = model(data, data.x, tabm_seed=i)
        if is_binary:
            loss = F.binary_cross_entropy_with_logits(
                out[data.train_mask].squeeze(-1), data.y[data.train_mask]
            )
        else:
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask].long())
        loss = loss / k
        loss.backward()
        total_loss += loss.item()
    optimizer.step()
    return total_loss


@torch.no_grad()
def evaluate_tabm(model, data, mask, is_binary, dataset_name, k):
    model.eval()
    outs = [model(data, data.x, tabm_seed=i) for i in range(k)]
    avg_out = torch.stack(outs).mean(dim=0)
    return compute_metrics(avg_out, data.y, mask, is_binary, dataset_name=dataset_name)


def train_one_split(model_name, num_layers, split_idx, data, train_masks, val_masks, test_masks,
                    num_targets, is_binary, device, save_dir, dataset_name, hidden_dim, lr,
                    num_steps, k, init_scheme):
    data.train_mask = train_masks[:, split_idx].to(device)
    data.val_mask = val_masks[:, split_idx].to(device)
    data.test_mask = test_masks[:, split_idx].to(device)

    set_seed(split_idx)

    model = TABMAblationModel(
        model_name=model_name,
        num_layers=num_layers,
        input_dim=data.x.size(1),
        hidden_dim=hidden_dim,
        output_dim=num_targets,
        hidden_dim_multiplier=1,
        num_heads=8,
        normalization='LayerNorm',
        dropout=0.2,
        tabm_inits=k,
        init_scheme=init_scheme,
        device=device,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

    best_val_metric = -float('inf')
    best_step = 0
    steps_without_improvement = 0
    slug = combo_slug(hidden_dim, lr)
    ckpt_path = os.path.join(
        save_dir, f'{model_name}_{num_layers}L_{slug}_split{split_idx}_k{k}_{init_scheme}.pt'
    )
    val_metric_name = get_validation_metric_name(dataset_name, is_binary)

    EVAL_EVERY = 10
    for step in range(1, num_steps + 1):
        train_loss = train_step_tabm(model, data, optimizer, is_binary, k)

        if step % EVAL_EVERY != 0 and step != 1:
            continue

        val_metrics = evaluate_tabm(model, data, data.val_mask, is_binary, dataset_name, k)

        if val_metrics['metric'] > best_val_metric:
            best_val_metric = val_metrics['metric']
            best_step = step
            torch.save(model.state_dict(), ckpt_path)
            steps_without_improvement = 0
        else:
            steps_without_improvement += EVAL_EVERY

        if step % 100 == 0 or step == 1:
            print(f'Step {step:4d} | loss {train_loss:.4f} | {metrics_to_str("val", val_metrics)}')

        if steps_without_improvement >= PATIENCE * 100:
            print(f'Early stopping at step {step}')
            break

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    val_m = evaluate_tabm(model, data, data.val_mask, is_binary, dataset_name, k)
    test_m = evaluate_tabm(model, data, data.test_mask, is_binary, dataset_name, k)
    train_m = evaluate_tabm(model, data, data.train_mask, is_binary, dataset_name, k)

    del model, optimizer
    torch.cuda.empty_cache()

    result = {
        'model': model_name,
        'num_layers': num_layers,
        'hidden_dim': hidden_dim,
        'lr': lr,
        'split': split_idx,
        'best_step': best_step,
        'tabm_inits': k,
        'init_scheme': init_scheme,
        'val_metric_name': val_metric_name,
        'val_metric': val_m['metric'],
    }
    result.update(metrics_to_prefixed_dict('val', val_m))
    result.update(metrics_to_prefixed_dict('test', test_m))
    result.update(metrics_to_prefixed_dict('train', train_m))
    return result


__all__ = [
    'NUM_SPLITS', 'NUM_STEPS', 'load_dataset', 'read_best_hparams_from_split0_log',
    'train_one_split',
]
