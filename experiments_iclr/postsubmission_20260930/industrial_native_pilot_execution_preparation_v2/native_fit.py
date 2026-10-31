"""Bounded native first-stage fit only; no raw labels, extension or final metric."""
from __future__ import annotations
import copy
from pathlib import Path
import resource
import time
import delu
import numpy as np
import scipy.special
import torch
from sklearn.metrics import average_precision_score
from closed_inputs import sha, write_json
from native_optimizer import make_optimizer
from public_preprocessor import load_public, transform_public
from qualify_native import make_model


def atomic_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.pending')
    if temporary.exists():
        raise ValueError('Earlier incomplete checkpoint must be preserved')
    torch.save(value, temporary)
    temporary.replace(path)


def fit_cell(inputs, roles, compact, config, seed, device, output):
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    delu.random.seed(seed)
    public = transform_public(load_public(inputs / 'tolokers-2', roles, compact), config['transform']['features'])
    graph, numerical, other = public.tensors(device)
    model = make_model(public, config, device)
    optimizer = make_optimizer(model, config['optimizer'])
    train_rows = torch.tensor(compact['train']['rows'], dtype=torch.long, device=device)
    train_labels = torch.tensor(compact['train']['labels'], dtype=torch.long, device=device)
    val_rows = np.array(compact['val']['rows'], dtype=np.int64)
    val_labels = np.array(compact['val']['labels'], dtype=np.int64)
    stopper = delu.tools.EarlyStopping(config['patience'], mode='max')
    best_ap, best_step = None, None
    trace = []
    warm50_written = False
    progress = {'completed_optimizer_updates': 0, 'attempted_step': 0, 'stage': 'constructed'}
    write_json(output / 'progress.json', progress)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    def snapshot(step):
        return {'step': step, 'model': copy.deepcopy(model.state_dict()),
                'optimizer': copy.deepcopy(optimizer.state_dict()),
                'random_state': delu.random.get_state(), 'config': copy.deepcopy(config),
                'seed': seed, 'classes': 2, 'test_label_access': False}

    for step in range(1, config['n_steps'] + 1):
        progress.update(attempted_step=step, stage='training_forward_backward_step')
        write_json(output / 'progress.json', progress)
        model.train()
        optimizer.zero_grad()
        logits = model(graph, numerical, other).squeeze(-1).float()
        if logits.shape != (public.node_count, 2):
            raise ValueError('Native full-graph two-logit shape differs')
        loss = torch.nn.functional.cross_entropy(logits[train_rows], train_labels)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite source training loss')
        loss.backward()
        optimizer.step()
        progress.update(completed_optimizer_updates=step, stage='validation_forward')
        write_json(output / 'progress.json', progress)
        model.eval()
        with torch.inference_mode():
            raw = model(graph, numerical, other).squeeze(-1).float().cpu().numpy()
        # Exact native scipy softmax and source validation AP; no test slicing.
        probability = scipy.special.softmax(raw, axis=-1)[val_rows, 1]
        if not np.isfinite(probability).all():
            raise ValueError('Nonfinite source validation probability')
        ap = float(average_precision_score(val_labels, probability))
        if not np.isfinite(ap):
            raise ValueError('Nonfinite source validation AP')
        if best_ap is None or ap > best_ap:
            best_ap, best_step = ap, step
            atomic_checkpoint(output / 'best.pt', snapshot(step))
            np.save(output / 'best_validation_probability.npy', probability)
        if step == 50:
            atomic_checkpoint(output / 'warm50.pt', snapshot(step))
            warm50_written = True
        trace.append({'step': step, 'train_ce': float(loss.detach()), 'validation_ap': ap})
        write_json(output / 'trace.json', trace)
        progress.update(stage='selection_and_stopping_complete', best_step=best_step, best_validation_ap=best_ap)
        write_json(output / 'progress.json', progress)
        stopper.update(ap)
        if stopper.should_stop():
            break
    if not warm50_written or best_ap is None:
        raise ValueError('Native fit failed to produce the fixed warm50 donor')
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    result = {'status': 'SUCCEEDED', 'seed': seed, 'best_step': best_step, 'best_validation_ap': best_ap,
              'updates': len(trace), 'warm50_selector': 'fixed_last', 'test_labels': 'CLOSED',
              'nonconstant_selected_validation_probability': bool(np.ptp(np.load(output / 'best_validation_probability.npy')) > 0),
              'encoded_width': public.width, 'node_count': public.node_count, 'edge_count': graph.num_edges(),
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'seconds': time.perf_counter() - started,
              'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
              'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
              'checkpoint_sha256': sha(output / 'best.pt'), 'warm50_sha256': sha(output / 'warm50.pt'),
              'validation_probability_sha256': sha(output / 'best_validation_probability.npy')}
    write_json(output / 'trace.json', trace)
    write_json(output / 'result.json', result)
    return result


def run_nine(inputs, roles, compact, matrix, device, output, input_identity, runtime_identity, source_identity):
    registry = [{'cell_id': row['cell_id'], 'config': row['config'], 'seed': row['seed'], 'status': 'PLANNED'}
                for row in matrix['fit_cells']]
    write_json(output / 'registry.json', registry)
    configurations = {row['id']: row['native_config'] for row in matrix['configurations']}
    results = {}
    for index, cell in enumerate(registry):
        cell['status'] = 'RUNNING'
        write_json(output / 'registry.json', registry)
        started = time.perf_counter()
        try:
            result = fit_cell(inputs, roles, compact, configurations[cell['config']], cell['seed'],
                              device, output / cell['cell_id'])
            result['config_id'] = cell['config']
            results[cell['cell_id']] = result
            cell.update(status='SUCCEEDED', result=result)
        except BaseException as error:
            cell.update(status='FAILED', error_type=type(error).__name__, error=str(error),
                        elapsed_seconds=time.perf_counter() - started)
            progress_path = output / cell['cell_id'] / 'progress.json'
            if progress_path.is_file():
                from closed_inputs import load
                cell['paid_update_progress'] = load(progress_path)
            for pending in registry[index + 1:]:
                pending['status'] = 'NOT_STARTED_DEPENDENCY_FAILED'
            write_json(output / 'registry.json', registry)
            raise
        write_json(output / 'registry.json', registry)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if len(results) != 9 or any(row['status'] != 'SUCCEEDED' for row in registry):
        raise ValueError('Incomplete matrix forbids selection')
    scores = {name: sum(result['best_validation_ap'] for result in results.values()
                        if result['config_id'] == name) / 3 for name in matrix['configuration_order']}
    selected = matrix['configuration_order'][0]
    for name in matrix['configuration_order'][1:]:
        if scores[name] > scores[selected]:
            selected = name
    y = np.array(compact['val']['labels'], dtype=np.int64)
    p_train = float(np.mean(compact['train']['labels']))
    constant_ap = float(average_precision_score(y, np.full(len(y), p_train)))
    selected_results = [results[selected + '__s' + str(seed)] for seed in matrix['seeds']]
    uplifts = [result['best_validation_ap'] - constant_ap for result in selected_results]
    competent = all(uplift > 0 for uplift in uplifts) and sum(uplifts) / 3 >= 0.05
    competent &= all(result['nonconstant_selected_validation_probability'] for result in selected_results)
    probabilities = [np.load(output / (selected + '__s' + str(seed)) / 'best_validation_probability.npy')
                     for seed in matrix['seeds']]
    ensemble = np.mean(np.stack(probabilities), axis=0)
    np.save(output / 'native_three_member_validation_probability.npy', ensemble)
    return {'status': 'COMPLETED_NINE_NATIVE_FITS', 'selected_config': selected, 'mean_best_validation_AP': scores,
            'tie_rule': 'fixed c0,c1,c2 order,strict improvement', 'source_validation_competence': {
                'passed': bool(competent), 'constant_train_probability': p_train,
                'constant_validation_AP': constant_ap, 'selected_seed_uplifts': uplifts,
                'mean_uplift': sum(uplifts) / 3, 'threshold': 0.05,
                'interpretation': 'Prespecified basic learning screen only; no competitive or reproduction claim.'},
            'native_three_member_validation_AP': float(average_precision_score(y, ensemble)),
            'selected_checkpoints': selected_results, 'total_useful_updates': sum(r['updates'] for r in results.values()),
            'maximum_useful_updates': 9000, 'input_identity': input_identity, 'runtime_identity': runtime_identity,
            'source_identity': source_identity, 'test_labels': 'CLOSED', 'extension_fit': False,
            'final_test_evaluation': False, 'all_native_fit_costs_retained': True,
            'failed_competence_consequence': 'No initializer extension may proceed; preserve complete first-stage result.'}
