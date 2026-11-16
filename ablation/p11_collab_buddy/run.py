"""Resumable native BUDDY, independent ensemble and original four-route GNNM."""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from ogb.linkproppred import Evaluator

from ablation.common.resumable import (
    JobStore, atomic_torch_save, capture_rng, restore_rng,
)
from ablation.common.worker import Job, Preempted, print_status, run_worker
from .protocol import RECIPE, UPSTREAM_COMMIT, dependency_versions, inside_repository, source_identity


def make_jobs(args, source_sha256, dataset_sha256):
    smoke = None
    if args.smoke:
        smoke = {'epochs': args.smoke_epochs, 'train_edges': args.smoke_train_edges,
                 'max_batches': args.smoke_max_batches}
    recipe = dict(RECIPE)
    if getattr(args, 'convergence', False):
        recipe.update(epochs=1000, cap_schedule=[1000, 2000], minimum_epochs=0, patience=50,
                      controller='min0_caps1000_2000_patience50')
    common = {
        'study': 'p11_collab_buddy', 'dataset': 'ogbl-collab',
        'upstream_commit': UPSTREAM_COMMIT, 'source_sha256': source_sha256,
        'dataset_sha256': dataset_sha256, 'recipe': recipe,
        'defer_test': getattr(args, 'defer_test', False), 'dependencies': dependency_versions(),
        'smoke': smoke, 'role': 'smoke_train_valid' if smoke else 'full',
        'evaluation': 'official_complete_valid_hits50',
    }
    jobs = []
    for cohort in args.cohorts:
        for member in range(4):
            config = dict(common, variant='native', cohort=cohort, member=member,
                          seed=cohort * 1000 + member, reports_base=member == 0)
            jobs.append(Job(f'cohort{cohort}/native{member}', config))
        config = dict(common, variant='gnnm', cohort=cohort, seed=cohort * 1000, members=4)
        jobs.append(Job(f'cohort{cohort}/gnnm', config))
        if getattr(args, 'extensions', False) or getattr(args, 'only', None) in (
                'gnnm_contrast', 'gnnm_private_bn', 'gnnm_private_bn_contrast'):
            for variant in ('gnnm_contrast', 'gnnm_private_bn', 'gnnm_private_bn_contrast'):
                jobs.append(Job(f'cohort{cohort}/{variant}',
                                dict(common, variant=variant, cohort=cohort,
                                     seed=cohort * 1000, members=4)))
    return jobs


def inputs_for(data, indices, device):
    links = data['links'][indices]
    degree = data['degrees'][links].to(device)
    return (data['subgraph_features'][indices].to(device), data['x'][links].to(device),
            degree[:, 0], degree[:, 1])


def evaluate_scores(scores, num_positive):
    evaluator = Evaluator(name='ogbl-collab')
    evaluator.K = 50
    if not torch.isfinite(scores).all():
        raise FloatingPointError('BUDDY prediction contains a non-finite score')
    return float(evaluator.eval({'y_pred_pos': scores[:num_positive],
                                 'y_pred_neg': scores[num_positive:]})['hits@50'])


@torch.no_grad()
def predict(model, data, variant, device, batch_size, store, stop):
    from .models import route_logits
    model.eval()
    predictions = []
    for indices in torch.arange(len(data['links'])).split(batch_size):
        predictions.append(route_logits(model, inputs_for(data, indices, device), variant).cpu())
        store.heartbeat()
        if stop.requested:
            raise Preempted()
    member_scores = torch.cat(predictions, dim=1)
    return {'scores': member_scores.mean(0), 'member_scores': member_scores, 'num_positive': data['num_positive'],
            'edge_order_sha256': data['edge_order_sha256']}


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train_job(job, store, stop, args, bundle, cache_root):
    from .data import prepare_test
    from .models import build_model, route_logits, contrastive_step
    config = job.config
    device = torch.device(args.device)
    recipe = config['recipe']
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = build_model(config['variant'], bundle['train']['x'].size(1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=recipe['lr'], weight_decay=recipe['weight_decay'])
    progress = {
        'epoch': 0, 'next_batch': 0, 'order': None, 'global_step': 0,
        'best_val_metric': -float('inf'), 'best_epoch': None, 'best_model': None,
        'epoch_loss_sum': 0.0, 'epoch_examples': 0, 'epochs_without_improvement': 0,
        'train_seconds': 0.0, 'resumes': 0, 'curves': [], 'continuations': [],
    }
    saved = store.load_state()
    if saved is not None:
        if saved.get('config', config) != config or saved['source_sha256'] != config['source_sha256']:
            raise RuntimeError('Checkpoint source identity differs from this job')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        progress = saved['progress']
        progress['resumes'] += 1
        restore_rng(saved['rng'], device)
        print(f'[buddy] resume {job.key} epoch={progress["epoch"]} batch={progress["next_batch"]}', flush=True)

    def save():
        store.save_state({'config': config, 'source_sha256': config['source_sha256'], 'model': cpu_state(model),
                          'optimizer': optimizer.state_dict(), 'scheduler': None,
                          'progress': progress, 'rng': capture_rng(device)})

    epochs = config['smoke']['epochs'] if config['smoke'] else recipe.get('cap_schedule', [recipe['epochs']])[-1]
    train = bundle['train']
    num_positive = train['num_positive']
    labels = torch.cat((torch.ones(num_positive), torch.zeros(len(train['links']) - num_positive)))
    last_save = time.monotonic()
    # Save before the first stochastic batch, including initial parameters/RNG.
    if saved is None:
        save()
    while progress['epoch'] < epochs:
        if (recipe.get('patience') and progress['epochs_without_improvement'] >= recipe['patience']
                and not config['smoke']):
            break
        if progress['order'] is None:
            progress['order'] = torch.randperm(len(labels))
        batches = list(progress['order'].split(recipe['batch_size']))
        if config['smoke'] and config['smoke']['max_batches']:
            batches = batches[:config['smoke']['max_batches']]
        model.train()
        for batch_index in range(progress['next_batch'], len(batches)):
            indices = batches[batch_index]
            started = time.monotonic()
            optimizer.zero_grad()
            inputs = inputs_for(train, indices, device)
            targets = labels[indices].to(device)
            if 'contrast' in config['variant']:
                loss, diagnostic = contrastive_step(
                    model, inputs, targets, train['links'][indices], config['seed'], optimizer)
                progress['last_contrast_diagnostic'] = diagnostic
            else:
                logits = route_logits(model, inputs, config['variant'])
                loss = F.binary_cross_entropy_with_logits(logits, targets.expand_as(logits))
                if not torch.isfinite(loss):
                    raise FloatingPointError('BUDDY training loss is non-finite')
                loss.backward()
                optimizer.step()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            progress['train_seconds'] += time.monotonic() - started
            progress['global_step'] += 1
            progress['next_batch'] = batch_index + 1
            progress['epoch_loss_sum'] += loss.item() * len(indices)
            progress['epoch_examples'] += len(indices)
            store.heartbeat()
            if stop.requested or time.monotonic() - last_save >= args.save_every_sec:
                save()
                last_save = time.monotonic()
            if stop.requested:
                raise Preempted()
        # Commit the completed epoch before evaluation, so an interruption replays
        # only deterministic evaluation and never repeats an optimizer update.
        save()
        validation = predict(model, bundle['valid'], config['variant'], device,
                             args.eval_batch_size, store, stop)
        val_metric = evaluate_scores(validation['scores'], validation['num_positive'])
        epoch = progress['epoch'] + 1
        improved = val_metric > progress['best_val_metric']
        if improved:
            progress['best_val_metric'] = val_metric
            progress['best_epoch'] = epoch
            progress['best_model'] = cpu_state(model)
            progress['epochs_without_improvement'] = 0
        else:
            progress['epochs_without_improvement'] += 1
        if (not config['smoke'] and epoch in recipe.get('cap_schedule', [])[:-1]
                and progress['epochs_without_improvement'] < recipe['patience']):
            progress['continuations'].append({
                'completed_epoch': epoch, 'next_cap': epochs,
                'reason': 'patience_plateau_not_reached',
                'optimizer_and_rng_preserved': True})
        if epoch == 100 and not config['smoke']:
            atomic_torch_save({'config': config, 'model': progress['best_model'],
                               'selected_epoch': progress['best_epoch'],
                               'val_metric': progress['best_val_metric']},
                              store.dir / 'reference100_model.pt')
        progress['curves'].append({'epoch': epoch, 'validation_hits50': val_metric,
                                   'loss': progress['epoch_loss_sum'] / progress['epoch_examples'],
                                   'global_step': progress['global_step'],
                                   'train_seconds': progress['train_seconds']})
        progress['epoch'] = epoch
        progress['next_batch'] = 0
        progress['order'] = None
        progress['epoch_loss_sum'] = 0.0
        progress['epoch_examples'] = 0
        save()
        print(f'[buddy] {job.key} epoch={epoch} valid={val_metric:.6f} best={progress["best_val_metric"]:.6f}', flush=True)
    model.load_state_dict(progress['best_model'])
    atomic_torch_save(progress['best_model'], store.dir / 'best_model.pt')
    predictions = {
        'config': config,
        'valid': predict(model, bundle['valid'], config['variant'], device,
                         args.eval_batch_size, store, stop),
    }
    result = dict(config, best_epoch=progress['best_epoch'], stopped_epoch=progress['epoch'],
                  global_step=progress['global_step'], train_seconds=progress['train_seconds'],
                  resumes=progress['resumes'], val_metric=progress['best_val_metric'],
                  num_params=sum(parameter.numel() for parameter in model.parameters()),
                  metric='Hits@50', curves=progress['curves'], test_evaluated=False,
                  predictions_file='predictions.pt', best_model_file='best_model.pt',
                  selected_checkpoint_sha256=hashlib.sha256((store.dir / 'best_model.pt').read_bytes()).hexdigest(),
                  cap_hit=not config['smoke'] and progress['epoch'] >= epochs,
                  stale_validation_checks=progress['epochs_without_improvement'],
                  continuations=progress['continuations'],
                  stop_reason=('smoke' if config['smoke'] else 'patience'
                               if recipe.get('patience') and progress['epochs_without_improvement']
                               >= recipe['patience'] else 'epoch_cap'),
                  reference100_available=(store.dir / 'reference100_model.pt').exists(),
                  reference100_checkpoint_sha256=(
                      hashlib.sha256((store.dir / 'reference100_model.pt').read_bytes()).hexdigest()
                      if (store.dir / 'reference100_model.pt').exists() else None),
                  last_contrast_diagnostic=progress.get('last_contrast_diagnostic'))
    if not config['smoke'] and not config.get('defer_test'):
        test = prepare_test(config, args, cache_root, store, stop)
        predictions['test'] = predict(model, test, config['variant'], device,
                                      args.eval_batch_size, store, stop)
        result['test_metric'] = evaluate_scores(predictions['test']['scores'], test['num_positive'])
        result['test_evaluated'] = True
    atomic_torch_save(predictions, store.dir / 'predictions.pt')
    return result



def score_selected(job, training_dir, store, stop, args, bundle, cache_root):
    from .data import prepare_test
    from .models import build_model
    config = job.config
    parent = JobStore(training_dir)
    if json.loads(parent.config_path.read_text()) != config:
        raise ValueError('Selected fit config differs')
    model = build_model(config['variant'], bundle['train']['x'].size(1)).to(args.device)
    result = parent.read_done()
    actual = hashlib.sha256((training_dir / 'best_model.pt').read_bytes()).hexdigest()
    if result.get('selected_checkpoint_sha256') != actual:
        raise ValueError('Selected checkpoint differs from completed training')
    test = prepare_test(config, args, cache_root, store, stop)
    checkpoints = [(torch.load(training_dir / 'best_model.pt', map_location='cpu',
                               weights_only=False), 'predictions.pt', '')]
    reference = training_dir / 'reference100_model.pt'
    if reference.exists():
        if hashlib.sha256(reference.read_bytes()).hexdigest() != result.get('reference100_checkpoint_sha256'):
            raise ValueError('Reference100 checkpoint changed after completed training')
        payload = torch.load(reference, map_location='cpu', weights_only=False)
        if payload['config'] != config:
            raise ValueError('Reference100 checkpoint config differs')
        checkpoints.append((payload['model'], 'reference100_predictions.pt', 'reference100_'))
    for state, filename, prefix in checkpoints:
        model.load_state_dict(state)
        predictions = {'config': config}
        for name, data in (('valid', bundle['valid']), ('test', test)):
            predictions[name] = predict(model, data, config['variant'], torch.device(args.device),
                                        args.eval_batch_size, store, stop)
            metric_name = 'val' if name == 'valid' else name
            result[prefix + metric_name + '_metric'] = evaluate_scores(
                predictions[name]['scores'], data['num_positive'])
        atomic_torch_save(predictions, store.dir / filename)
    result['test_evaluated'] = True
    return result

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data_dir', type=Path, default=Path('data'))
    parser.add_argument('--cache_dir', type=Path, default=Path('ablation/p11_collab_buddy/cache'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cohorts', nargs='+', type=int, default=list(range(5)))
    parser.add_argument('--eval_batch_size', type=int, default=8192)
    parser.add_argument('--preprocess_batch_size', type=int, default=8192)
    parser.add_argument('--save_every_sec', type=float, default=120)
    parser.add_argument('--max_failures', type=int, default=3)
    parser.add_argument('--stale_seconds', type=float, default=600)
    parser.add_argument('--no_wait', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke_epochs', type=int, default=1)
    parser.add_argument('--smoke_train_edges', type=int, default=0)
    parser.add_argument('--smoke_max_batches', type=int, default=2)
    parser.add_argument('--require-cache', action='store_true')
    parser.add_argument('--convergence', action='store_true')
    parser.add_argument('--defer-test', action='store_true')
    parser.add_argument('--score-only', action='store_true')
    parser.add_argument('--extensions', action='store_true')
    parser.add_argument('--only', choices=['native', 'gnnm', 'gnnm_contrast', 'gnnm_private_bn',
                                          'gnnm_private_bn_contrast'])
    parser.add_argument('--member', type=int, choices=range(4))
    args = parser.parse_args(argv)
    for name in ('root', 'data_dir', 'cache_dir'):
        setattr(args, name, inside_repository(getattr(args, name)))
    for name in ('eval_batch_size', 'preprocess_batch_size', 'smoke_epochs'):
        if getattr(args, name) < 1:
            parser.error(f'--{name} must be positive')
    if args.smoke_max_batches < 0 or args.smoke_train_edges < 0 or any(cohort < 0 for cohort in args.cohorts):
        parser.error('Batch cap and cohort identifiers must be nonnegative')
    return args


def main(argv=None):
    args = parse_args(argv)
    from .data import dataset_identity
    from .cache import prepare_training
    source = source_identity()
    identity = dataset_identity(args.data_dir) if not args.status else 'status_only'
    jobs = make_jobs(args, source, identity)
    if args.only:
        jobs = [job for job in jobs if job.config['variant'] == args.only]
    if args.member is not None:
        jobs = [job for job in jobs if job.config.get('member') == args.member]
    if args.status:
        print_status(args.root, jobs, args.max_failures, args.stale_seconds)
        return 0
    # Verify finished jobs too, because the generic worker skips them early.
    for job in jobs:
        store = JobStore(args.root / job.key)
        if store.config_path.exists():
            store.ensure_config(job.config)
    cached = None

    if args.score_only:
        if not args.defer_test:
            raise ValueError('--score-only requires the training --defer-test binding')
        for job in jobs:
            parent = JobStore(args.root / job.key)
            if not parent.is_done() or parent.read_done().get('smoke'):
                raise ValueError('Scoring requires completed scientific training')
        def score_job(job, store, stop):
            nonlocal cached
            if cached is None:
                cached = prepare_training(job.config, args, store, stop)
            return score_selected(job, args.root / job.key, store, stop, args, *cached)
        return run_worker(args.root / 'scoring', jobs, score_job,
                          max_failures=args.max_failures, stale_seconds=args.stale_seconds,
                          wait_for_others=not args.no_wait)

    def execute(job, store, stop):
        nonlocal cached
        if cached is None:
            cached = prepare_training(job.config, args, store, stop)
        return train_job(job, store, stop, args, *cached)

    return run_worker(args.root, jobs, execute, max_failures=args.max_failures,
                      stale_seconds=args.stale_seconds, wait_for_others=not args.no_wait)


if __name__ == '__main__':
    sys.exit(main())
