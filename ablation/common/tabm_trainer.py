"""Restartable node training with an explicit checkpoint-selection protocol.

The default ten-step selector preserves the submitted training implementation.
Prospective campaigns set selection_interval=1, a minimum training horizon and
a predeclared patience. Held-out scoring can be deferred until selection is frozen.
"""

import math
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import torch
import torch.nn.functional as F

from datasets import compute_metrics
from run_common import metrics_to_prefixed_dict, set_seed

from .resumable import atomic_torch_save, atomic_write_json, capture_rng, restore_rng
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
    min_steps: int = 0
    selection_interval: int = 10
    weight_decay: float = 0.0
    defer_test: bool = False
    save_selected: bool = False

    def __post_init__(self):
        if self.num_steps < 1 or self.patience < 1 or self.selection_interval < 1:
            raise ValueError('Training horizon, patience and selection interval must be positive')
        if not 0 <= self.min_steps <= self.num_steps:
            raise ValueError('min_steps must lie between zero and num_steps')
        if self.lr <= 0 or self.weight_decay < 0 or self.members < 1:
            raise ValueError('Invalid learning rate, weight decay or member count')


def member_loss(out, data, is_binary):
    if is_binary:
        return F.binary_cross_entropy_with_logits(
            out[data.train_mask].squeeze(-1), data.y[data.train_mask])
    return F.cross_entropy(out[data.train_mask], data.y[data.train_mask].long())


def train_step(model, data, optimizer, is_binary, members):
    model.train()
    optimizer.zero_grad()
    if hasattr(model, 'contrastive_train_step'):
        return float(model.contrastive_train_step(data, optimizer, is_binary, members))
    if hasattr(model, 'forward_all'):
        logits = model.forward_all(data, data.x)
        loss = sum(member_loss(out, data, is_binary) for out in logits) / members
        loss.backward()
        total_loss = loss.detach()
    else:
        losses = []
        for member in range(members):
            out = model(data, data.x, tabm_seed=member)
            loss = member_loss(out, data, is_binary) / members
            loss.backward()
            losses.append(loss.detach())
        total_loss = torch.stack(losses).sum()
    if not torch.isfinite(total_loss):
        raise FloatingPointError('Non-finite training loss')
    optimizer.step()
    return float(total_loss)


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


def _synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def train_resumable(settings, store, bundle, device, build_model, stop=None, on_step=None,
                    logits_path=None, model_path=None):
    """Train one fit and retain the selected checkpoint without changing its protocol."""
    data, train_masks, val_masks, test_masks, is_binary = bundle
    data.train_mask = train_masks[:, settings.split].to(device)
    data.val_mask = val_masks[:, settings.split].to(device)
    data.test_mask = test_masks[:, settings.split].to(device)

    set_seed(settings.seed)
    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr,
                                  weight_decay=settings.weight_decay)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    progress = {
        'step': 0, 'best_val_metric': -float('inf'), 'best_step': 0,
        'steps_without_improvement': 0, 'best_model': None,
        'train_seconds': 0.0, 'validation_seconds': 0.0,
        'fit_seconds': 0.0, 'resumes': 0, 'curve': [],
    }
    saved = store.load_state()
    if saved is not None:
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        progress.update(saved['progress'])
        progress['resumes'] += 1
        restore_rng(saved['rng'], device)
        print(f'[resume] {store.dir} continues after step {progress["step"]}', flush=True)

    def save_state():
        store.save_state({
            'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
            'optimizer': optimizer.state_dict(), 'progress': progress,
            'rng': capture_rng(device),
        })
        atomic_write_json(progress['curve'], store.dir / 'curve.json')

    last_save = last_heartbeat = time.monotonic()
    already_converged = (progress['step'] >= settings.min_steps
                         and progress['steps_without_improvement'] >= settings.patience)
    last_step = progress['step'] if already_converged else settings.num_steps
    for step in range(progress['step'] + 1, last_step + 1):
        _synchronize(device)
        started = time.monotonic()
        try:
            loss = train_step(model, data, optimizer, is_binary, settings.members)
            _synchronize(device)
            trained = time.monotonic()
            progress['train_seconds'] += trained - started
            logits = member_logits(model, data, settings.members)
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f'Non-finite predictions at step {step}')
            val_metric = evaluate(logits, data, data.val_mask, is_binary, settings.dataset)['metric']
            del logits
            _synchronize(device)
            validated = time.monotonic()
            progress['validation_seconds'] += validated - trained
            if not math.isfinite(val_metric):
                raise FloatingPointError(f'Non-finite validation metric at step {step}')

        except FloatingPointError as error:
            error.attempted_steps = step
            error.completed_steps = progress['step']
            raise

        if val_metric > progress['best_val_metric']:
            progress['steps_without_improvement'] = 0
            if step % settings.selection_interval == 0:
                progress['best_val_metric'] = val_metric
                progress['best_step'] = step
                progress['best_model'] = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
        else:
            progress['steps_without_improvement'] += 1
        progress['step'] = step
        progress['fit_seconds'] += time.monotonic() - started
        progress['curve'].append({
            'step': step, 'train_loss': loss, 'val_metric': val_metric,
            'best_val_metric': progress['best_val_metric'] if progress['best_model'] else None,
            'train_seconds': progress['train_seconds'],
            'validation_seconds': progress['validation_seconds'],
            'fit_seconds': progress['fit_seconds'],
        })

        if step % 500 == 0:
            print(f'{store.dir} step={step} val={val_metric:.4f} '
                  f'best={progress["best_val_metric"]:.4f}', flush=True)

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
        if step >= settings.min_steps and progress['steps_without_improvement'] >= settings.patience:
            break
        if on_step is not None:
            on_step(step)

    if progress['best_model'] is None:
        raise RuntimeError('No checkpoint selected; horizon must cover selection_interval')
    save_state()
    store.heartbeat()
    model.load_state_dict(progress['best_model'])
    logits = member_logits(model, data, settings.members)
    val_metrics = evaluate(logits, data, data.val_mask, is_binary, settings.dataset)
    train_metrics = evaluate(logits, data, data.train_mask, is_binary, settings.dataset)
    if logits_path is not None:
        if settings.defer_test:
            raise ValueError('Full-node logits export is disabled while held-out scoring is deferred')
        atomic_torch_save(logits.detach().cpu(), logits_path)
    if model_path is not None:
        atomic_torch_save(progress['best_model'], model_path)
    if settings.save_selected:
        atomic_torch_save(progress['best_model'], store.dir / 'model.pt')
        roles = {'train': data.train_mask, 'val': data.val_mask}
        if not settings.defer_test:
            roles['test'] = data.test_mask
        predictions = {
            'schema': 1, 'dataset': settings.dataset, 'split': settings.split,
            'seed': settings.seed, 'members': settings.members,
            'best_step': progress['best_step'], 'is_binary': is_binary, 'roles': {},
        }
        for role, mask in roles.items():
            predictions['roles'][role] = {
                'indices': mask.nonzero(as_tuple=False).flatten().cpu(),
                'labels': data.y[mask].detach().cpu(),
                'logits': logits[:, mask].detach().cpu(),
            }
        atomic_torch_save(predictions, store.dir / 'predictions.pt')

    patience_reached = progress['steps_without_improvement'] >= settings.patience
    converged = progress['step'] >= settings.min_steps and patience_reached
    result = {
        'best_step': progress['best_step'], 'stopped_step': progress['step'],
        'num_params': num_params, 'train_seconds': progress['train_seconds'],
        'validation_seconds': progress['validation_seconds'],
        'fit_seconds': progress['fit_seconds'], 'resumes': progress['resumes'],
        'stop_reason': 'patience' if converged else 'step_cap',
        'needs_convergence_extension': not converged,
        'steps_without_improvement': progress['steps_without_improvement'],
        'test_deferred': settings.defer_test,
        'selection_interval': settings.selection_interval,
    }
    if hasattr(model, 'initialization_calibration'):
        atomic_write_json(model.initialization_calibration, store.dir / 'activation_calibration.json')
        result['initialization_calibrated'] = True
    result.update(metrics_to_prefixed_dict('val', val_metrics))
    result.update(metrics_to_prefixed_dict('train', train_metrics))
    if not settings.defer_test:
        test_metrics = evaluate(logits, data, data.test_mask, is_binary, settings.dataset)
        result.update(metrics_to_prefixed_dict('test', test_metrics))
        result.update(member_statistics(logits, data, data.test_mask, is_binary, settings.dataset))
    return result
