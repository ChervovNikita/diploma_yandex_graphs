"""Minibatch-exact recovery for the fixed OGB molecular training recipe."""

import hashlib
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from torch_geometric.data import Batch

from ablation.common.resumable import atomic_torch_save, capture_rng, restore_rng
from ablation.common.worker import Preempted
from .models import build_model, route_logits



def graph_inputs_identity(dataset):
    digest = hashlib.sha256()
    for index in range(len(dataset)):
        graph = dataset[index]
        digest.update(str(index).encode())
        for name in ('x', 'edge_index', 'edge_attr'):
            tensor = getattr(graph, name).detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_dataset(name, data_dir):
    dataset = PygGraphPropPredDataset(name=name, root=str(data_dir))
    if dataset.meta_info['split'] != 'scaffold' or dataset.eval_metric != 'rocauc':
        raise ValueError('This recipe requires official scaffold splits and ROC-AUC')
    if dataset.num_tasks != 1 or 'classification' not in dataset.task_type:
        raise ValueError('This recipe requires one binary molecular task')
    splits = {name: index.long().cpu() for name, index in dataset.get_idx_split().items()}
    all_indices = torch.cat(list(splits.values()))
    if len(all_indices) != len(dataset) or len(torch.unique(all_indices)) != len(dataset):
        raise ValueError('Official train/valid/test indices must partition the dataset')
    digest = hashlib.sha256()
    for split_name, index in sorted(splits.items()):
        digest.update(split_name.encode())
        digest.update(index.numpy().tobytes())
    digest.update(dataset.y.cpu().numpy().tobytes())
    binding = {'dataset': name, 'graphs': len(dataset), 'num_tasks': dataset.num_tasks,
               'splits_and_labels_sha256': digest.hexdigest(),
               'graph_inputs_sha256': graph_inputs_identity(dataset),
               'split_counts': {name: len(index) for name, index in splits.items()}}
    return dataset, splits, Evaluator(name=dataset.name), binding


def score(evaluator, labels, logits):
    return float(evaluator.eval({'y_true': labels.numpy(),
                                 'y_pred': logits.numpy()})['rocauc'])


def batches(dataset, indices, batch_size, start=0):
    """Explicit collation retains the saved permutation and next-batch cursor."""
    for offset in range(start, len(indices), batch_size):
        selected = indices[offset:offset + batch_size]
        yield offset + len(selected), Batch.from_data_list([dataset[int(i)] for i in selected])


@torch.no_grad()
def predict(model, dataset, indices, batch_size, device, members, tick):
    model.eval()
    labels, logits = [], []
    for _, batch in batches(dataset, indices, batch_size):
        batch = batch.to(device)
        logits.append(route_logits(model, batch, members).detach().cpu())
        labels.append(batch.y.detach().cpu())
        tick()
    return torch.cat(labels, dim=0), torch.cat(logits, dim=1)


def model_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train(config, store, stop, dataset, splits, evaluator, data_binding, device,
          save_every_sec=120.0, on_update=None):
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config['seed'])
    model = build_model(config, dataset.num_tasks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                 weight_decay=config['weight_decay'])
    # The reference has no scheduler. An explicit null in state/config binds it.
    shuffle = torch.Generator().manual_seed(config['seed'])
    progress = {'epoch': 0, 'completed_epochs': 0, 'updates': 0, 'cursor': 0,
                'order': None, 'phase': 'train', 'best_val_metric': -math.inf,
                'best_epoch': None, 'best_updates': None, 'best_model': None,
                'curves': [], 'epoch_loss_sum': 0.0, 'epoch_updates': 0,
                'train_seconds': 0.0, 'resumes': 0, 'stale_checks': 0, 'continuations': []}
    saved = store.load_state()
    if saved is not None:
        if saved['config'] != config or saved['data_binding'] != data_binding:
            raise ValueError('Checkpoint source/config/data binding differs from this job')
        if saved['scheduler'] is not None:
            raise ValueError('The native recipe has no scheduler')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        progress = saved['progress']
        progress['resumes'] += 1
        shuffle.set_state(saved['shuffle_rng'])
        restore_rng(saved['rng'], device)

    def save():
        store.save_state({'schema': 1, 'config': config, 'data_binding': data_binding,
                          'model': model_state(model), 'optimizer': optimizer.state_dict(),
                          'scheduler': None, 'progress': progress,
                          'rng': capture_rng(device), 'shuffle_rng': shuffle.get_state()})

    last_save = last_heartbeat = time.monotonic()

    def tick():
        nonlocal last_save, last_heartbeat
        now = time.monotonic()
        if now - last_heartbeat >= 30:
            store.heartbeat()
            last_heartbeat = now
        if stop.requested or now - last_save >= save_every_sec:
            save()
            last_save = now
        if stop.requested:
            raise Preempted()

    epoch_limit = config['smoke_epochs'] or config.get('cap_schedule', [config['epochs']])[-1]
    update_limit = config['smoke_updates'] or math.inf
    while progress['epoch'] < epoch_limit and (progress['updates'] < update_limit
                                               or progress['order'] is not None):
        if (config.get('patience') and progress['stale_checks'] >= config['patience']
                and not config['smoke']):
            break
        if progress['order'] is None:
            permutation = torch.randperm(len(splits['train']), generator=shuffle)
            progress['order'] = splits['train'][permutation]
        if progress['phase'] == 'train' and progress['updates'] < update_limit:
            model.train()
            for next_cursor, batch in batches(dataset, progress['order'], config['batch_size'],
                                               progress['cursor']):
                batch = batch.to(device)
                # Match the official training guard for BatchNorm-sensitive batches.
                if batch.x.shape[0] > 1 and batch.num_graphs > 1:
                    started = time.monotonic()
                    optimizer.zero_grad(set_to_none=True)
                    labeled = batch.y == batch.y
                    losses = []
                    if 'contrast' in config['method']:
                        from ablation.common.contrastive import unit_contrastive_loss, private_gradient_step
                        outputs = [model(batch, member=member, return_prehead=True)
                                   for member in range(config['members'])]
                        task_losses = [F.binary_cross_entropy_with_logits(out.float()[labeled],
                                                                          batch.y.float()[labeled])
                                       for out, _ in outputs]
                        features = torch.stack([feature[labeled.reshape(-1)] for _, feature in outputs])
                        auxiliary, info = unit_contrastive_loss(
                            features, batch.y.reshape(-1)[labeled.reshape(-1)],
                            config['seed'] + 104729 * progress['updates'])
                        block = model.input_be_block
                        private_gradient_step(model, optimizer, torch.stack(task_losses).mean(),
                                              auxiliary, (block.R, block.S, block.B))
                        progress['last_contrast_diagnostic'] = dict(info, loss=float(auxiliary.detach()))
                        losses = [float(loss.detach()) for loss in task_losses]
                    else:
                        for member in range(config['members']):
                            logits = model(batch, member=member) if config['members'] > 1 else model(batch)
                            loss = F.binary_cross_entropy_with_logits(logits.float()[labeled],
                                                                      batch.y.float()[labeled])
                            if not torch.isfinite(loss):
                                raise FloatingPointError('Non-finite molecular training loss')
                            (loss / config['members']).backward()
                            losses.append(float(loss.detach()))
                        optimizer.step()
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    progress['train_seconds'] += time.monotonic() - started
                    progress['updates'] += 1
                    progress['epoch_updates'] += 1
                    progress['epoch_loss_sum'] += sum(losses) / len(losses)
                progress['cursor'] = next_cursor
                if on_update is not None:
                    on_update(progress['updates'])
                tick()
                if progress['updates'] >= update_limit:
                    break
        progress['phase'] = 'evaluate'
        labels, logits = predict(model, dataset, splits['valid'], config['batch_size'],
                                 device, config['members'], tick)
        validation = score(evaluator, labels, logits.mean(dim=0))
        if not math.isfinite(validation):
            raise FloatingPointError('Non-finite official validation ROC-AUC')
        epoch = progress['epoch'] + 1
        if validation > progress['best_val_metric']:
            progress.update(best_val_metric=validation, best_epoch=epoch,
                            best_updates=progress['updates'], best_model=model_state(model),
                            stale_checks=0)
        else:
            progress['stale_checks'] += 1
        full_epoch = progress['cursor'] == len(progress['order'])
        progress['completed_epochs'] += int(full_epoch)
        progress['curves'].append({'epoch': epoch, 'updates': progress['updates'],
                                   'full_epoch': full_epoch, 'val_rocauc': validation,
                                   'mean_batch_loss': progress['epoch_loss_sum'] /
                                   max(1, progress['epoch_updates'])})
        if (not config['smoke'] and epoch in config.get('cap_schedule', [])[:-1]
                and progress['stale_checks'] < config['patience']):
            progress['continuations'].append({
                'completed_epoch': epoch, 'next_cap': epoch_limit,
                'reason': 'patience_plateau_not_reached',
                'optimizer_and_rng_preserved': True})
        if epoch == 100 and full_epoch and not config['smoke']:
            atomic_torch_save({'config': config, 'data_binding': data_binding,
                               'model': progress['best_model'],
                               'selected_epoch': progress['best_epoch'],
                               'val_metric': progress['best_val_metric']},
                              store.dir / 'reference100_model.pt')
        print(f'{store.dir} epoch={epoch} updates={progress["updates"]} '
              f'val={validation:.6f} best={progress["best_val_metric"]:.6f}', flush=True)
        progress.update(epoch=epoch, cursor=0, order=None, phase='train',
                        epoch_loss_sum=0.0, epoch_updates=0)
        save()
        tick()

    # An interruption immediately after the final smoke update still needs its
    # pending validation pass. Production runs always finish complete epochs.
    if progress['best_model'] is None:
        raise RuntimeError('No finite validation checkpoint selected')
    save()
    model.load_state_dict(progress['best_model'])

    def final_tick():
        store.heartbeat()
        if stop.requested:
            # The saved state still pairs the latest training model and optimizer.
            raise Preempted()

    predictions = {'config': config, 'data_binding': data_binding, 'splits': {}}
    metrics = {}
    final_splits = ['valid'] if config['smoke'] else ['train', 'valid']
    if not config['smoke'] and not config.get('defer_test'):
        final_splits.append('test')
    for split in final_splits:
        labels, logits = predict(model, dataset, splits[split], config['batch_size'],
                                 device, config['members'], final_tick)
        predictions['splits'][split] = {'indices': splits[split], 'y_true': labels,
                                        'member_logits': logits}
        prefix = 'val' if split == 'valid' else split
        metrics[f'{prefix}_metric'] = score(evaluator, labels, logits.mean(dim=0))
    atomic_torch_save({'config': config, 'data_binding': data_binding,
                       'model': progress['best_model'], 'selected_epoch': progress['best_epoch']},
                      store.dir / 'model.pt')
    atomic_torch_save(predictions, store.dir / 'predictions.pt')
    return dict(config, **metrics, data_binding=data_binding,
                best_epoch=progress['best_epoch'], best_updates=progress['best_updates'],
                stopped_epoch=progress['epoch'], completed_epochs=progress['completed_epochs'],
                updates=progress['updates'], train_seconds=progress['train_seconds'],
                resumes=progress['resumes'], validation_curve=progress['curves'],
                selected_in_final_quarter=progress['best_epoch'] > 0.75 * epoch_limit,
                num_params=sum(p.numel() for p in model.parameters()),
                device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                stop_reason=('smoke' if config['smoke'] else
                             'patience' if config.get('patience') and
                             progress['stale_checks'] >= config['patience'] else 'epoch_cap'),
                cap_hit=not config['smoke'] and progress['epoch'] >= epoch_limit,
                stale_validation_checks=progress['stale_checks'],
                continuations=progress['continuations'],
                reference100_available=(store.dir / 'reference100_model.pt').exists(),
                reference100_checkpoint_sha256=(
                    hashlib.sha256((store.dir / 'reference100_model.pt').read_bytes()).hexdigest()
                    if (store.dir / 'reference100_model.pt').exists() else None),
                selected_checkpoint_sha256=hashlib.sha256((store.dir / 'model.pt').read_bytes()).hexdigest(),
                last_contrast_diagnostic=progress.get('last_contrast_diagnostic'),
                 test_evaluated='test' in final_splits)


def score_selected(config, training_dir, store, stop, dataset, splits, evaluator,
                   data_binding, device):
    """Score frozen selected states after the campaign selection dependencies."""
    from ablation.common.resumable import JobStore
    completed = JobStore(training_dir).read_done()
    actual = hashlib.sha256((training_dir / 'model.pt').read_bytes()).hexdigest()
    if completed.get('selected_checkpoint_sha256') != actual:
        raise ValueError('Selected checkpoint differs from the completed training result')
    selected = torch.load(training_dir / 'model.pt', map_location='cpu', weights_only=False)
    if selected['config'] != config or selected['data_binding'] != data_binding:
        raise ValueError('Selected checkpoint source/config/data binding differs')
    model = build_model(config, dataset.num_tasks).to(device)

    def tick():
        store.heartbeat()
        if stop.requested:
            raise Preempted()

    result = dict(completed, test_evaluated=True, selected_epoch=selected['selected_epoch'],
                  data_binding=data_binding)
    reference = training_dir / 'reference100_model.pt'
    if reference.exists() and hashlib.sha256(reference.read_bytes()).hexdigest() != completed.get(
            'reference100_checkpoint_sha256'):
        raise ValueError('Reference100 checkpoint changed after completed training')
    for checkpoint, filename, prefix in (
            (selected, 'predictions.pt', ''),
            (torch.load(training_dir / 'reference100_model.pt', map_location='cpu',
                        weights_only=False) if (training_dir / 'reference100_model.pt').exists()
             else None, 'reference100_predictions.pt', 'reference100_')):
        if checkpoint is None:
            continue
        if checkpoint['config'] != config or checkpoint['data_binding'] != data_binding:
            raise ValueError('Reference checkpoint source/config/data binding differs')
        model.load_state_dict(checkpoint['model'])
        output = {'config': config, 'data_binding': data_binding, 'splits': {}}
        for name in ('train', 'valid', 'test'):
            labels, logits = predict(model, dataset, splits[name], config['batch_size'],
                                     device, config['members'], tick)
            output['splits'][name] = {'indices': splits[name], 'y_true': labels,
                                     'member_logits': logits}
            key = 'val' if name == 'valid' else name
            result[prefix + key + '_metric'] = score(evaluator, labels, logits.mean(0))
        atomic_torch_save(output, store.dir / filename)
    return result
