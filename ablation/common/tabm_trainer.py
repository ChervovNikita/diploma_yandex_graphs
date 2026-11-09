"""Resumable version of the training loop in ``run_tabm.py``.

The optimizer, per-step validation, checkpoint rule and patience counter
deliberately match ``run_tabm.train_single``, including two details of it: the
best model only updates on steps divisible by ten, while any step that beats
the current best value resets the patience counter. Results are therefore
directly comparable with the main table. ``test_p01`` checks the equivalence.
"""

import time
from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import torch
import torch.nn.functional as F

from datasets import compute_metrics
from run_common import metrics_to_prefixed_dict, set_seed

from .resumable import atomic_torch_save, capture_rng, restore_rng
from .worker import Preempted


@dataclass
class TrainSettings:
    dataset: str
    split: int
    seed: int
    members: int
    lr: float
    num_steps: int
    patience: int
    save_every_sec: float = 120.0
    save_every_steps: Optional[int] = None
    heartbeat_every_sec: float = 30.0


def member_loss(out, data, is_binary):
    if is_binary:
        return F.binary_cross_entropy_with_logits(
            out[data.train_mask].squeeze(-1), data.y[data.train_mask])
    return F.cross_entropy(out[data.train_mask], data.y[data.train_mask].long())


def train_step(model, data, optimizer, is_binary, members):
    model.train()
    optimizer.zero_grad()
    if hasattr(model, 'forward_all'):  # One backbone pass shared by all members.
        logits = model.forward_all(data, data.x)
        (sum(member_loss(out, data, is_binary) for out in logits) / members).backward()
    else:
        for member in range(members):
            out = model(data, data.x, tabm_seed=member)
            (member_loss(out, data, is_binary) / members).backward()
    optimizer.step()


@torch.no_grad()
def member_logits(model, data, members):
    model.eval()
    if hasattr(model, 'forward_all'):
        return model.forward_all(data, data.x)
    return torch.stack([model(data, data.x, tabm_seed=i) for i in range(members)])


def evaluate(logits, data, mask, is_binary, dataset):
    return compute_metrics(logits.mean(dim=0), data.y, mask, is_binary, dataset_name=dataset)


def member_statistics(logits, data, mask, is_binary, dataset):
    """Quality of single members and how much they differ, on one mask."""
    members = logits.shape[0]
    y = data.y[mask].long()
    if is_binary:
        predictions = (logits[:, mask].squeeze(-1) > 0).long()
    else:
        predictions = logits[:, mask].argmax(dim=-1)
    disagreements = []
    jaccards = []
    for first, second in combinations(range(members), 2):
        disagreements.append(predictions[first].ne(predictions[second]).float().mean().item())
        first_errors, second_errors = predictions[first].ne(y), predictions[second].ne(y)
        union = (first_errors | second_errors).sum().item()
        jaccards.append((first_errors & second_errors).sum().item() / union if union else 1.0)
    member_metric = sum(
        compute_metrics(logits[i], data.y, mask, is_binary, dataset_name=dataset)['metric']
        for i in range(members)
    ) / members
    return {
        'test_mean_member_metric': member_metric,
        'test_pair_disagreement': sum(disagreements) / len(disagreements) if disagreements else 0.0,
        'test_error_jaccard': sum(jaccards) / len(jaccards) if jaccards else None,
    }


def train_resumable(settings, store, bundle, device, build_model, stop=None, on_step=None,
                    logits_path=None, model_path=None):
    """Train one job, continuing from ``store`` when a saved state exists."""
    data, train_masks, val_masks, test_masks, is_binary = bundle
    data.train_mask = train_masks[:, settings.split].to(device)
    data.val_mask = val_masks[:, settings.split].to(device)
    data.test_mask = test_masks[:, settings.split].to(device)

    set_seed(settings.seed)
    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=0)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    progress = {
        'step': 0,
        'best_val_metric': -float('inf'),
        'best_step': 0,
        'steps_without_improvement': 0,
        'best_model': None,
        'train_seconds': 0.0,
        'resumes': 0,
    }
    saved = store.load_state()
    if saved is not None:
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        progress = saved['progress']
        progress['resumes'] += 1
        restore_rng(saved['rng'], device)
        print(f'[resume] {store.dir} continues after step {progress["step"]}', flush=True)

    def save_state():
        store.save_state({
            'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
            'optimizer': optimizer.state_dict(),
            'progress': progress,
            'rng': capture_rng(device),
        })

    last_save = last_heartbeat = time.monotonic()
    for step in range(progress['step'] + 1, settings.num_steps + 1):
        started = time.monotonic()
        train_step(model, data, optimizer, is_binary, settings.members)
        progress['train_seconds'] += time.monotonic() - started

        logits = member_logits(model, data, settings.members)
        val_metric = evaluate(logits, data, data.val_mask, is_binary, settings.dataset)['metric']
        del logits

        if val_metric > progress['best_val_metric']:
            progress['steps_without_improvement'] = 0
            if step % 10 == 0:
                progress['best_val_metric'] = val_metric
                progress['best_step'] = step
                progress['best_model'] = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
        else:
            progress['steps_without_improvement'] += 1
        progress['step'] = step

        if step % 500 == 0:
            print(f'{store.dir} step={step} val={val_metric:.4f} '
                  f'best={progress["best_val_metric"]:.4f}', flush=True)

        if progress['steps_without_improvement'] >= settings.patience:
            break

        now = time.monotonic()
        stopping = stop is not None and stop.requested
        due_by_steps = (settings.save_every_steps is not None
                        and step % settings.save_every_steps == 0)
        if stopping or due_by_steps or now - last_save >= settings.save_every_sec:
            save_state()
            last_save = now
        if stopping:
            raise Preempted()
        if now - last_heartbeat >= settings.heartbeat_every_sec:
            store.heartbeat()
            last_heartbeat = now
        if on_step is not None:
            on_step(step)

    if progress['best_model'] is None:
        raise RuntimeError('No checkpoint was selected: need at least ten steps and a finite metric')
    store.heartbeat()
    model.load_state_dict(progress['best_model'])
    logits = member_logits(model, data, settings.members)
    val_metrics = evaluate(logits, data, data.val_mask, is_binary, settings.dataset)
    test_metrics = evaluate(logits, data, data.test_mask, is_binary, settings.dataset)
    train_metrics = evaluate(logits, data, data.train_mask, is_binary, settings.dataset)
    if logits_path is not None:
        atomic_torch_save(logits.detach().cpu(), logits_path)
    if model_path is not None:
        atomic_torch_save(progress['best_model'], model_path)

    result = {
        'best_step': progress['best_step'],
        'stopped_step': progress['step'],
        'num_params': num_params,
        'train_seconds': progress['train_seconds'],
        'resumes': progress['resumes'],
    }
    result.update(metrics_to_prefixed_dict('val', val_metrics))
    result.update(metrics_to_prefixed_dict('test', test_metrics))
    result.update(metrics_to_prefixed_dict('train', train_metrics))
    result.update(member_statistics(logits, data, data.test_mask, is_binary, settings.dataset))
    return result
