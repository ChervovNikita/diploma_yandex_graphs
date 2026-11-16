"""Resumable native GCN/GraphSAGE, independent ensemble members and GNNM on Collab.

    python -m ablation.p14_collab_ogb.run --root ROOT --backbone gcn --device cuda:0
    python -m ablation.p14_collab_ogb.run --root ROOT --backbone gcn --defer-test --score-only

Each cohort holds four independent native fits seeded ``1000 * cohort + member``
(member 0 is BASE, all four form ENS) and one four-route GNNM fit seeded
``1000 * cohort``; ``gnnm_contrast`` adds the shared contrastive term. TEST is
read only by ``--score-only``, after the validation-selected checkpoint is
frozen. Prediction files have the same layout as the BUDDY runs, so the campaign
scoring and ``rescore.py`` treat both studies the same way.
"""

import argparse
import hashlib
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ablation.common.contrastive import unit_contrastive_loss
from ablation.common.resumable import JobStore, atomic_torch_save, capture_rng, restore_rng
from ablation.common.worker import Job, Preempted, print_status, run_worker
from .protocol import HITS, RECIPE, ROUTES, dependency_versions, inside_repository, source_identity


def make_jobs(args, source_sha256, dataset_sha256):
    common = {
        'study': 'p14_collab_ogb', 'dataset': 'ogbl-collab', 'backbone': args.backbone,
        'source_sha256': source_sha256, 'dataset_sha256': dataset_sha256, 'recipe': RECIPE,
        'defer_test': args.defer_test, 'dependencies': dependency_versions(),
        'smoke': args.smoke_updates, 'role': 'smoke_train_valid' if args.smoke_updates else 'full',
        'evaluation': 'official_complete_valid_hits50',
    }
    jobs = []
    for cohort in args.cohorts:
        for member in range(4):
            jobs.append(Job(f'cohort{cohort}/native{member}', dict(
                common, variant='native', cohort=cohort, member=member,
                seed=cohort * 1000 + member, reports_base=member == 0)))
        for variant in ('gnnm', 'gnnm_contrast'):
            jobs.append(Job(f'cohort{cohort}/{variant}', dict(
                common, variant=variant, cohort=cohort, seed=cohort * 1000, members=ROUTES)))
    return jobs


def hits(scores, num_positive, k):
    from ogb.linkproppred import Evaluator
    if not torch.isfinite(scores).all():
        raise FloatingPointError('Collab prediction contains a non-finite score')
    evaluator = Evaluator(name='ogbl-collab')
    evaluator.K = k
    return float(evaluator.eval({'y_pred_pos': scores[:num_positive],
                                 'y_pred_neg': scores[num_positive:]})[f'hits@{k}'])


def evaluate_scores(scores, num_positive):
    return hits(scores, num_positive, 50)


def all_hits(scores, num_positive):
    return {f'hits@{k}': hits(scores, num_positive, k) for k in HITS}


def to_device(graph, device):
    return {key: value.to(device) for key, value in graph.items()}


@torch.no_grad()
def predict(model, x, graph, links, batch_size):
    model.eval()
    member_scores = []
    for route in range(model.routes):
        h = model.embed(x, graph, route)
        member_scores.append(torch.cat([model.score(h, chunk.to(x.device), route).cpu()
                                        for chunk in links['links'].split(batch_size)]))
    member_scores = torch.stack(member_scores)
    return {'scores': member_scores.mean(0), 'member_scores': member_scores,
            'num_positive': links['num_positive'], 'edge_order_sha256': links['edge_order_sha256']}


def clip(model):
    """The example clips the encoder and the predictor separately."""
    encoder = list(model.convs.parameters())
    predictor = list(model.lins.parameters()) + list(model.final.parameters())
    if model.routes > 1:
        encoder += [model.R, model.S, model.B]
    torch.nn.utils.clip_grad_norm_(encoder, RECIPE['grad_clip'])
    torch.nn.utils.clip_grad_norm_(predictor, RECIPE['grad_clip'])


def train_batch(model, optimizer, variant, x, graph, positive, num_nodes, seed, step):
    from .models import link_loss
    optimizer.zero_grad()
    negative = torch.randint(0, num_nodes, positive.size(), dtype=torch.long, device=x.device)
    if variant == 'gnnm_contrast':
        links = torch.cat((positive, negative))
        labels = torch.cat((torch.ones(len(positive)), torch.zeros(len(negative))))
        losses, preheads = [], []
        for route in range(model.routes):
            logits, prehead = model.score(model.embed(x, graph, route), links, route, return_prehead=True)
            losses.append(link_loss(logits[:len(positive)], logits[len(positive):]))
            preheads.append(prehead)
        task = torch.stack(losses).mean()
        contrast = RECIPE['contrast']
        auxiliary, _ = unit_contrastive_loss(torch.stack(preheads), labels, seed + 104729 * step,
                                             contrast['temperature'], contrast['anchors'],
                                             contrast['alternatives'])
        if not torch.isfinite(task) or not torch.isfinite(auxiliary):
            raise FloatingPointError('Non-finite contrastive or task objective')
        # Shared weights learn from the task loss only; route factors also from the contrast.
        private = model.private_parameters()
        private_ids = {id(parameter) for parameter in private}
        shared = [parameter for parameter in model.parameters() if id(parameter) not in private_ids]
        shared_grad = torch.autograd.grad(task, shared, retain_graph=True, allow_unused=True)
        private_grad = torch.autograd.grad(task + contrast['coefficient'] * auxiliary, private,
                                           allow_unused=True)
        for parameter, gradient in zip(list(shared) + list(private), shared_grad + private_grad):
            parameter.grad = gradient
        loss = task
    else:
        losses = []
        for route in range(model.routes):
            h = model.embed(x, graph, route)
            losses.append(link_loss(model.score(h, positive, route), model.score(h, negative, route)))
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError('Non-finite gradient')
    clip(model)
    optimizer.step()
    return float(loss.detach())


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def seed_everything(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def train_job(job, store, stop, args, bundle):
    from .models import build_model
    config, device = job.config, torch.device(args.device)
    seed_everything(config['seed'], device)
    model = build_model(config['backbone'], config['variant'], bundle['x'].size(1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=RECIPE['lr'], weight_decay=RECIPE['weight_decay'])
    caps = RECIPE['cap_schedule']
    progress = {'epoch': 0, 'cap': caps[0], 'global_step': 0, 'best_val_metric': -float('inf'),
                'best_epoch': None, 'best_model': None, 'train_seconds': 0.0, 'resumes': 0,
                'curves': [], 'extensions': []}
    saved = store.load_state()
    if saved is not None:
        if saved['config'] != config:
            raise RuntimeError('Checkpoint config differs from this job')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        progress = saved['progress']
        progress['resumes'] += 1
        restore_rng(saved['rng'], device)

    def save():
        store.save_state({'config': config, 'model': cpu_state(model), 'optimizer': optimizer.state_dict(),
                          'progress': progress, 'rng': capture_rng(device)})

    x = bundle['x'].to(device)
    graph = to_device(bundle['train_graph'], device)
    positive_pool = bundle['train_positive']
    smoke = config['smoke']
    if saved is None:
        save()
    while progress['epoch'] < (1 if smoke else progress['cap']):
        started = time.monotonic()
        model.train()
        batches = torch.randperm(len(positive_pool)).split(RECIPE['batch_size'])
        if smoke:
            batches = batches[:smoke]
        loss_sum = 0.0
        for batch in batches:
            loss_sum += train_batch(model, optimizer, config['variant'], x, graph,
                                    positive_pool[batch].to(device), bundle['num_nodes'],
                                    config['seed'], progress['global_step'])
            progress['global_step'] += 1
            store.heartbeat()
            if stop.requested:
                raise Preempted()  # Resume replays this epoch from the last saved state.
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        progress['train_seconds'] += time.monotonic() - started
        validation = predict(model, x, graph, bundle['valid'], args.eval_batch_size)
        val_metric = evaluate_scores(validation['scores'], validation['num_positive'])
        epoch = progress['epoch'] + 1
        if val_metric > progress['best_val_metric']:
            progress.update(best_val_metric=val_metric, best_epoch=epoch, best_model=cpu_state(model))
        progress['curves'].append({'epoch': epoch, 'validation_hits50': val_metric,
                                   'loss': loss_sum / max(len(batches), 1),
                                   'train_seconds': progress['train_seconds']})
        progress['epoch'] = epoch
        within = RECIPE['extend_if_best_within']
        if (not smoke and epoch == progress['cap'] and progress['cap'] != caps[-1]
                and progress['best_epoch'] > epoch - within):
            following = caps[caps.index(progress['cap']) + 1]
            progress['extensions'].append({'at_epoch': epoch, 'best_epoch': progress['best_epoch'],
                                           'next_cap': following})
            progress['cap'] = following
        save()
        print(f'[collab-{config["backbone"]}] {job.key} epoch={epoch} valid={val_metric:.5f} '
              f'best={progress["best_val_metric"]:.5f}', flush=True)
        if stop.requested:
            raise Preempted()

    model.load_state_dict(progress['best_model'])
    atomic_torch_save(progress['best_model'], store.dir / 'best_model.pt')
    atomic_torch_save({'config': config,
                       'valid': predict(model, x, graph, bundle['valid'], args.eval_batch_size)},
                      store.dir / 'predictions.pt')
    best_near_cap = progress['best_epoch'] > progress['epoch'] - RECIPE['extend_if_best_within']
    return dict(config, best_epoch=progress['best_epoch'], stopped_epoch=progress['epoch'],
                global_step=progress['global_step'], train_seconds=progress['train_seconds'],
                resumes=progress['resumes'], val_metric=progress['best_val_metric'],
                num_params=sum(parameter.numel() for parameter in model.parameters()),
                metric='Hits@50', curves=progress['curves'], extensions=progress['extensions'],
                test_evaluated=False, predictions_file='predictions.pt', best_model_file='best_model.pt',
                selected_checkpoint_sha256=hashlib.sha256((store.dir / 'best_model.pt').read_bytes()).hexdigest(),
                cap_hit=not smoke and best_near_cap,
                stop_reason='smoke' if smoke else 'epoch_cap' if best_near_cap else 'fixed_epochs')


def score_selected(job, training_dir, store, args, bundle):
    """Score the frozen checkpoint on VALID and TEST, with Hits@20/50/100."""
    from .models import build_model
    config, device = job.config, torch.device(args.device)
    parent = JobStore(training_dir)
    if parent.read_done().get('selected_checkpoint_sha256') != hashlib.sha256(
            (training_dir / 'best_model.pt').read_bytes()).hexdigest():
        raise ValueError('Selected checkpoint differs from completed training')
    model = build_model(config['backbone'], config['variant'], bundle['x'].size(1)).to(device)
    model.load_state_dict(torch.load(training_dir / 'best_model.pt', map_location='cpu'))
    x = bundle['x'].to(device)
    predictions = {'config': config}
    result = parent.read_done()
    for name, graph in (('valid', bundle['train_graph']), ('test', bundle['test_graph'])):
        predictions[name] = predict(model, x, to_device(graph, device), bundle[name], args.eval_batch_size)
        prefix = 'val' if name == 'valid' else name
        result[f'{prefix}_metric'] = evaluate_scores(predictions[name]['scores'], bundle[name]['num_positive'])
        for key, value in all_hits(predictions[name]['scores'], bundle[name]['num_positive']).items():
            result[f'{prefix}_{key}'] = value
        store.heartbeat()
    atomic_torch_save(predictions, store.dir / 'predictions.pt')
    result['test_evaluated'] = True
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--backbone', choices=['gcn', 'sage'], required=True)
    parser.add_argument('--data_dir', type=Path, default=Path('data'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cohorts', nargs='+', type=int, default=list(range(5)))
    parser.add_argument('--only', choices=['native', 'gnnm', 'gnnm_contrast'])
    parser.add_argument('--member', type=int, choices=range(4))
    parser.add_argument('--eval_batch_size', type=int, default=64 * 1024)
    parser.add_argument('--defer-test', action='store_true')
    parser.add_argument('--score-only', action='store_true')
    parser.add_argument('--smoke_updates', type=int, default=0)
    parser.add_argument('--max_failures', type=int, default=3)
    parser.add_argument('--stale_seconds', type=float, default=600)
    parser.add_argument('--no_wait', action='store_true')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args(argv)
    args.root = inside_repository(args.root)
    args.data_dir = inside_repository(args.data_dir)
    if args.score_only and args.smoke_updates:
        parser.error('Smoke runs never read TEST')
    return args


def main(argv=None):
    args = parse_args(argv)
    from .data import dataset_identity, load_bundle
    identity = 'status_only' if args.status else dataset_identity(args.data_dir)
    jobs = make_jobs(args, source_identity(), identity)
    if args.only:
        jobs = [job for job in jobs if job.config['variant'] == args.only]
    if args.member is not None:
        jobs = [job for job in jobs if job.config.get('member') == args.member]
    if args.status:
        print_status(args.root, jobs, args.max_failures, args.stale_seconds)
        return 0
    for job in jobs:
        store = JobStore(args.root / job.key)
        if store.config_path.exists():
            store.ensure_config(job.config)
    cached = {}

    def bundle(with_test):
        if with_test not in cached:
            cached[with_test] = load_bundle(args.data_dir, with_test)
        return cached[with_test]

    if args.score_only:
        if not args.defer_test:
            raise ValueError('--score-only requires the training --defer-test binding')
        for job in jobs:
            parent = JobStore(args.root / job.key)
            if not parent.is_done() or parent.read_done().get('smoke'):
                raise ValueError('Scoring requires completed scientific training')
        return run_worker(args.root / 'scoring', jobs,
                          lambda job, store, stop: score_selected(job, args.root / job.key, store,
                                                                  args, bundle(True)),
                          max_failures=args.max_failures, stale_seconds=args.stale_seconds,
                          wait_for_others=not args.no_wait)
    return run_worker(args.root, jobs,
                      lambda job, store, stop: train_job(job, store, stop, args, bundle(False)),
                      max_failures=args.max_failures, stale_seconds=args.stale_seconds,
                      wait_for_others=not args.no_wait)


if __name__ == '__main__':
    sys.exit(main())
