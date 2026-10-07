"""Disabled public Mol18 selected-state collector; no numerical import at module scope."""
import argparse
import copy
import gc
from importlib import metadata
import math
import os
from pathlib import Path
import random
import resource
import sys
import time
from protocol import (ALLOCATION, CONDITIONS, HERE, PHASE, PUBLIC, SEEDS, binding, bound,
                      inside, module, read, release, require, sha, write)
from rank_repair import analyze, freeze_baseline


def usage():
    value = resource.getrusage(resource.RUSAGE_SELF)
    return {'CPU_user_seconds': value.ru_utime, 'CPU_system_seconds': value.ru_stime,
            'process_cumulative_peak_RSS_bytes': value.ru_maxrss * (1 if sys.platform == 'darwin' else 1024)}


def identity(run, condition, seed, adoption, public, allocation):
    arm = condition if condition in ('single', 'independent4') else 'be_init__allocation_' + condition
    require(run.get('task') == 'molhiv' and run.get('arm') == arm and run.get('seed') == seed
            and run.get('TEST_scoring') is False, 'Exact public task/condition/seed with TEST closed required')
    expected_core = {Path(row['path']).name: row['sha256']
                     for row in read(PHASE / PUBLIC / 'MANIFEST.json')['files']
                     if row['path'] in ('core/factors.py', 'core/models.py', 'core/objectives.py', 'core/selection.py')}
    require(run.get('core') == expected_core and run.get('native') == {}
            and run.get('driver_sha256') == sha(PHASE / PUBLIC / 'train.py'), 'Exact public MolHIV core/driver required')
    roles = adoption['frozen_role_projection']['roles']
    require(run['data']['train_npz_sha256'] == roles['train']['sha256']
            and run['data']['valid_npz_sha256'] == roles['valid']['sha256']
            and run['data']['schema'] == 'portable-user-supplied-train-valid-v1'
            and run['data']['array_shape_domain_role_checks_passed'] is True, 'Frozen TRAIN/development roles required')
    config = copy.deepcopy(public.recipe('molhiv'))
    if condition in ('O', 'I', 'P', 'G'):
        expected = allocation.control_identity(condition, 'be_init', .5)
        require(run.get('method_identity') == arm and run.get('underlying_constructor_arm') == 'be_init'
                and all(run['allocation_control'].get(k) == v for k, v in expected.items()), 'Exact O/I/P/G source policy required')
        config.update(arms=[arm], allocation_control=expected)
    return arm, config


def fit_metadata(row, pins, adoption, public, allocation):
    """Bind full fit closure; never recompute selector scores or choose epochs."""
    directory = inside(PHASE, row['fit_output'])
    require(not (directory / 'FAILURE.json').exists(), 'Complete row also has a FAILURE receipt')
    artifacts = row['artifacts']
    required = ['RUN.json', 'COMPLETE.json', 'PROGRESS.json', 'VALID_TRACE.json', 'selected.pt']
    if row['condition'] == 'independent4':
        required.append('OWN_BEST_BANK.json')
    paths = {}
    for name in required:
        paths[name] = bound(PHASE, artifacts[name])
        require(paths[name] == directory / name, 'Exact public output artifact path required')
    run, complete, progress = (read(paths[name]) for name in ('RUN.json', 'COMPLETE.json', 'PROGRESS.json'))
    arm, config = identity(run, row['condition'], row['seed'], adoption, public, allocation)
    require(complete.get('complete') is True and complete.get('task') == 'molhiv'
            and complete.get('arm') == arm and complete.get('seed') == row['seed']
            and complete.get('epochs') == 100 and complete.get('steps') == 25800
            and complete.get('TEST_scoring') is False
            and complete.get('selected_sha256') == artifacts['selected.pt']['sha256'], 'Complete100-epoch/25800-update fit required')
    require(progress.get('epoch') == 100 and progress.get('steps') == 25800, 'Complete full-horizon progress required')
    trace = read(paths['VALID_TRACE.json'])
    require(isinstance(trace, list) and len(trace) == 100 and [x['epoch'] for x in trace] == list(range(1, 101)),
            'Every original full-development selector event must be retained')
    if row['condition'] in ('O', 'I', 'P', 'G'):
        expected = allocation.control_identity(row['condition'], 'be_init', .5)
        meta = complete['allocation_control']
        require(all(meta.get(k) == v for k, v in expected.items()) and meta.get('completed_source_steps') == 25800,
                'Complete policy identity/work differs')
        counts = meta['adapter_runtime_metadata']['counters']
        require(all(counts[k] == 25800 for k in ('two_view_updates', 'Adam_calls', 'Adam_steps'))
                and counts['Session_forward_calls'] == 51600 and counts['member_forwards'] == 206400
                and counts['autograd_grad_calls'] == 25800 * {'O': 2, 'I': 2, 'P': 3, 'G': 2}[row['condition']]
                and meta['actual_external_work']['complete_VALID_evaluations'] == 100
                and meta['actual_external_work']['complete_VALID_forward_calls'] == 3300
                and meta['actual_external_work']['complete_VALID_member_forwards'] == 13200, 'Full policy training/evaluation work differs')
    return paths, config, complete


def serve(row, config, paths, output, public, allocation, adoption, data, train, valid, device, ledger):
    import numpy as np
    import torch
    from ogb.graphproppred import Evaluator
    began = time.monotonic()
    seed, condition = row['seed'], row['condition']
    identifier = condition + '_' + str(seed)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    stages = {}
    checkpoint = paths['selected.pt']
    ledger.update(stage='deserialize', checkpoint_bytes=checkpoint.stat().st_size)
    write(output / 'compact' / 'CELL_PROGRESS.json', ledger)
    start = time.monotonic()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    require(sha(checkpoint) == row['artifacts']['selected.pt']['sha256'], 'Selected checkpoint changed during deserialization')
    stages['checkpoint_deserialization_seconds'] = time.monotonic() - start
    _, expected = identity(saved['run'], condition, seed, adoption, public, allocation)
    require(saved['config'] == expected == config, 'Exact selected public source/task configuration required')
    historical = {'selected_VALID': saved['selected_VALID']}
    require(isinstance(historical['selected_VALID'], (int, float)) and math.isfinite(historical['selected_VALID']),
            'Stored selected scalar must be finite; no parity comparison is performed')
    if condition == 'independent4':
        require(saved.get('evaluation_only') is True and saved.get('candidate') == 'individual_best_bank_only'
                and saved.get('body_global') == [False] * 4, 'Original final assembled own-selected bank required')
        bank = read(paths['OWN_BEST_BANK.json'])
        require(isinstance(bank.get('members'), list) and len(bank['members']) == 4
                and all(isinstance(x, (int, float)) and math.isfinite(x) for x in bank['members'])
                and isinstance(bank.get('VALID'), (int, float)) and math.isfinite(bank['VALID']), 'Complete original final-bank scalar metadata required')
        historical['final_assembled_bank'] = bank
        streams = None  # This selected endpoint never serialized live dropout streams.
    else:
        require(saved.get('checkpoint_kind') == 'strict-first-maximum complete VALID joint snapshot'
                and type(saved.get('epoch')) is int and 1 <= saved['epoch'] <= 100
                and saved.get('global') is False, 'Coherent original MolHIV joint selected state required')
        historical.update(epoch=saved['epoch'], member_VALID=saved['member_VALID'])
        count = 1 if condition == 'single' else 4
        require(isinstance(saved['member_VALID'], list) and len(saved['member_VALID']) == count
                and all(isinstance(x, (int, float)) and math.isfinite(x) for x in saved['member_VALID'])
                and len(saved['optimizers']) == 1, 'Original coherent joint scalar/optimizer-bank structure required')
        if condition in ('O', 'I', 'P', 'G'):
            require(saved.get('method_identity') == 'be_init__allocation_' + condition
                    and all(saved['allocation_control'].get(k) == v for k, v in config['allocation_control'].items())
                    and saved['allocation_control']['completed_source_steps'] == saved['epoch'] * 258,
                    'Exact selected policy/epoch identity required')
        streams = saved['streams']
    constructor = condition if condition in ('single', 'independent4') else 'be_init'
    start = time.monotonic()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
    core = public._core()
    # Deliberately avoid Session.__init__: no Adam bank or training state is created.
    view = public.Session.__new__(public.Session)
    view.torch, view.np, view.task, view.arm = torch, np, 'molhiv', constructor
    view.device, view.cuda_index = device, (0 if device.type == 'cuda' else None)
    view.model = core['models'].Ensemble('molhiv', constructor, seed, config['model'], (None, None)).to(device)
    view.model.load_state_dict(saved['model'], strict=True)
    core['selection'].finite_state(view.model, [])
    view.model.eval()
    require(all(not child.training for child in view.model.modules()), 'Original serving eval mode required')
    if streams is not None:
        require(len(streams) == view.model.members and all('cpu' in stream and
                (device.type != 'cuda' or 'cuda' in stream) for stream in streams), 'Joint selected member RNG bank differs')
        require(all(isinstance(stream['cpu'], torch.Tensor) and stream['cpu'].dtype == torch.uint8
                    and stream['cpu'].ndim == 1 for stream in streams), 'Original member CPU RNG state required')
        view.streams = streams
    del saved
    stages['constructor_strict_restore_eval_seconds'] = time.monotonic() - start
    native_forward = view.model.member_forward

    def charged_forward(batch, member):
        ledger['attempted_member_calls'] += 1
        ledger.update(stage='serving', member=member)
        # An external kill between this receipt and invocation leaves a conservative attempt ledger.
        write(output / 'compact' / 'CELL_PROGRESS.json', ledger)
        value = native_forward(batch, member)
        ledger['returned_member_calls'] += 1
        return value

    view.model.member_forward = charged_forward
    chunks, pools, labels = [], [], []
    partial = output / 'raw' / (identifier + '.partial.npz')
    start = time.monotonic()
    with torch.no_grad():
        for index, (batch, target) in enumerate(data.validation_batches('molhiv', train, valid, device), 1):
            ledger['batch'] = index
            # Joint uses the original saved RNG contexts. The assembled bank has no streams;
            # original eval-only GINE is deterministic with every dropout module disabled.
            logits, _ = view.forward(batch) if streams is not None else view.model(batch)
            pool = view.serving(logits)
            core['selection'].finite_predictions(logits, pool)
            require(logits.dtype == pool.dtype == torch.float32 and logits.shape == (view.model.members, len(target), 1),
                    'Original finite float32 binary member/serving shape required')
            chunks.append(logits[:, :, 0].cpu()); pools.append(pool[:, 0].cpu()); labels.append(target.cpu())
            start_save = time.monotonic()
            np.savez(partial, member_logits=torch.cat(chunks, 1).numpy(), pool_logits=torch.cat(pools).numpy(),
                     labels=torch.cat(labels).numpy(), ids=valid['ids'][:sum(map(len, labels))].numpy())
            stages['partial_storage_seconds'] = stages.get('partial_storage_seconds', 0.) + time.monotonic() - start_save
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    stages['one_serving_pass_including_batch_transfer_and_partial_storage_seconds'] = time.monotonic() - start
    members, pool, truth = torch.cat(chunks, 1), torch.cat(pools), torch.cat(labels)
    require(len(pool) == 4113 and index == 33 and len(labels[-1]) == 17
            and torch.equal(truth, valid['y']), 'All4113 development molecules/33 batches/tail17 required')
    metric = float(Evaluator(name='ogbg-molhiv').eval({'y_true': truth.numpy().reshape(-1, 1),
                                                     'y_pred': pool.numpy().reshape(-1, 1)})['rocauc'])
    require(math.isfinite(metric), 'Separate reevaluation source ROC AUC must be finite')
    raw = {'ids': valid['ids'].numpy(), 'labels': truth.numpy(),
           'member_logits': members.numpy(), 'pool_logits': pool.numpy()}
    destination = output / 'raw' / (identifier + '.npz')
    start = time.monotonic(); np.savez(destination, **raw); partial.unlink()
    stages['final_storage_seconds'] = time.monotonic() - start
    peaks = ({'CUDA_peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
              'CUDA_peak_reserved_bytes': torch.cuda.max_memory_reserved(device)} if device.type == 'cuda' else
             {'CUDA_peak_allocated_bytes': None, 'CUDA_peak_reserved_bytes': None})
    result = {'status': 'collected', 'source_served_AUC': metric, 'historical_selected_scalars': historical,
              'raw': binding(PHASE, destination), 'selected': row['artifacts']['selected.pt'],
              'member_count': len(members), 'serving_passes': 1, 'attempted_member_calls': ledger['attempted_member_calls'],
              'returned_member_calls': ledger['returned_member_calls'], 'stage_seconds': stages,
              'inclusive_cell_seconds': time.monotonic() - began, **usage(), **peaks,
              'reevaluation_event': True, 'historical_score_parity_gate': False}
    del view.model.member_forward  # Restore class lookup; do not leave a bound-method self-cycle.
    del native_forward, charged_forward
    del view, chunks, pools, members, pool, truth, batch, target, logits, _
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result, raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    require(os.environ.get('INTERNAL_BE_MOLHIV_READOUT_RELEASED') == '1', 'DISABLED: separately release this readout event')
    began, initial = time.monotonic(), usage()
    cfg, closure, pins, adoption, output = release(args.release, args.release_sha256)
    output.mkdir(mode=0o700)
    (output / 'raw').mkdir(mode=0o700); (output / 'compact').mkdir()
    collection, baselines, stages, total_calls = [], {}, {}, {'attempted': 0, 'returned': 0}
    analysis, error, torch, device, active_ledger = None, None, None, None, None
    write(output / 'compact' / 'EVENT.json', {'event_id': cfg['event_id'], 'release': binding(PHASE, args.release),
          'engineering_closure': cfg['engineering_closure'], 'terminal_evidence': cfg['terminal_evidence'],
          'TEST_access': False, 'source_ready_only_before_this_authorized_invocation': True})
    try:
        start = time.monotonic()
        import numpy as np
        import torch
        from ogb.graphproppred import Evaluator
        public = module(PHASE / PUBLIC / 'portable.py', '_mol18_public_readout')
        previous = sys.modules.get('portable')
        try:
            sys.modules['portable'] = public
            data = module(PHASE / PUBLIC / 'data_interface.py', '_mol18_public_readout_data')
        finally:
            if previous is None:
                sys.modules.pop('portable', None)
            else:
                sys.modules['portable'] = previous
        allocation = module(PHASE / ALLOCATION / 'train.py', '_mol18_policy_identity')  # stdlib source identity only.
        device = torch.device(cfg['authorized_work']['device'])
        if device.type == 'cuda':
            require(os.environ.get('CUDA_VISIBLE_DEVICES') == cfg['authorized_work']['gpu_uuid']
                    and torch.cuda.is_available() and torch.cuda.device_count() == 1,
                    'One separately authorized physical CUDA device required')
            torch.cuda.set_device(device)
        stages['numerical_source_runtime_loading_seconds'] = time.monotonic() - start
        start = time.monotonic()
        roles = adoption['frozen_role_projection']
        paths = {name: inside(PHASE, str(Path(roles['root_relative_directory']) / row['path']))
                 for name, row in roles['roles'].items()}
        require(all(sha(paths[k]) == roles['roles'][k]['sha256'] for k in ('train', 'valid')), 'Frozen role bytes differ')
        train, valid, origin = data.load_train_valid('molhiv', paths['train'], paths['valid'])
        stages['complete_TRAIN_and_development_role_load_validation_seconds'] = time.monotonic() - start
        write(output / 'compact' / 'ROLE_CUSTODY.json', {'train': binding(PHASE, paths['train']),
              'valid': binding(PHASE, paths['valid']), 'source_loader': origin, 'TEST_access': False})
        rows = {(r['condition'], r['seed']): r for r in closure['cells']}
        # Freeze all O baselines before the first candidate or comparator serving attempt.
        order = [('O', s) for s in SEEDS] + [(c, s) for c in CONDITIONS if c != 'O' for s in SEEDS]
        for position, key in enumerate(order):
            condition, seed = key
            if position == 3:
                write(output / 'compact' / 'BASELINE_COHORTS.json', {'O_only_frozen_before_candidate_collection': True,
                      'seeds': baselines, 'candidate_predictions_inspected': False})
            row = rows[key]
            cell = {'condition': condition, 'seed': seed, 'fit_status': row['status'], 'fit_output': row['fit_output'],
                    'fit_artifact_bindings': row.get('artifacts', {}), 'fit_cost': row.get('fit_cost'),
                    'fit_cost_limit': 'Original COMPLETE.seconds is inclusive fit wall time; absent resource/owner costs are unavailable'}
            ledger = {'condition': condition, 'seed': seed, 'attempted_member_calls': 0, 'returned_member_calls': 0}
            active_ledger = ledger
            cell_started, cell_initial = time.monotonic(), usage()
            try:
                if row['status'] != 'complete':
                    cell.update(status='fit_' + row['status'], reason=row.get('reason', 'No complete original endpoint'))
                else:
                    if device.type == 'cuda':
                        torch.cuda.reset_peak_memory_stats(device)
                    paths, config, complete = fit_metadata(row, pins, adoption, public, allocation)
                    cell['inclusive_original_fit_seconds'] = complete.get('seconds')
                    if row.get('fit_cost') is not None:
                        bound(PHASE, row['fit_cost'])
                    # Initial per-cell receipt precedes deserialization; there is no retry/resume path.
                    result, raw = serve(row, config, paths, output, public, allocation, adoption,
                                        data, train, valid, device, ledger)
                    cell.update(result)
                    if condition == 'O':
                        start = time.monotonic()
                        destination = output / 'raw' / ('O_cohorts_' + str(seed) + '.npz')
                        counts = freeze_baseline(raw, destination)
                        baselines[str(seed)] = {'available': True, 'raw': binding(PHASE, destination),
                                               'O_prediction': cell['raw'], 'counts': counts}
                        cell['O_cohort_freeze_seconds'] = time.monotonic() - start
                    del raw
            except Exception as failure:
                cell.update(status='collection_failed', error_type=type(failure).__name__, error=str(failure),
                            automatic_retry=False, attempted_member_calls=ledger['attempted_member_calls'],
                            returned_member_calls=ledger['returned_member_calls'])
            if cell['status'] == 'collection_failed':
                # Collect after the exception/traceback reference leaves its handler.
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            cell['inclusive_cell_with_custody_and_cohort_seconds'] = time.monotonic() - cell_started
            final = usage()
            cell.update(CPU_user_seconds=final['CPU_user_seconds'] - cell_initial['CPU_user_seconds'],
                        CPU_system_seconds=final['CPU_system_seconds'] - cell_initial['CPU_system_seconds'],
                        process_cumulative_peak_RSS_bytes=final['process_cumulative_peak_RSS_bytes'])
            if device.type == 'cuda' and row['status'] == 'complete':
                cell.update(CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                            CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            total_calls['attempted'] += ledger['attempted_member_calls']; total_calls['returned'] += ledger['returned_member_calls']
            if condition == 'O' and str(seed) not in baselines:
                baselines[str(seed)] = {'available': False, 'reason': cell.get('error', cell.get('reason'))}
            collection.append(cell)
            active_ledger = None
            write(output / 'compact' / 'COLLECTION.json', {'cells': collection, 'all18_accounted': len(collection) == 18})
        start = time.monotonic()
        analysis = analyze(PHASE, output, collection, baselines)
        stages['bounded_complete_pair_analysis_seconds'] = time.monotonic() - start
    except BaseException as failure:
        error = {'error_type': type(failure).__name__, 'error': str(failure), 'automatic_retry': False}
        write(output / 'compact' / 'FAILURE.json', error)
    finally:
        present = {(row['condition'], row['seed']) for row in collection}
        for row in closure['cells']:
            key = (row['condition'], row['seed'])
            if key not in present:
                missing = {'condition': key[0], 'seed': key[1], 'fit_status': row['status'],
                           'status': 'not_collected', 'reason': error or 'Collector did not reach this cell',
                           'fit_output': row['fit_output'], 'fit_artifact_bindings': row.get('artifacts', {})}
                if active_ledger is not None and key == (active_ledger['condition'], active_ledger['seed']):
                    missing.update(active_ledger)
                    total_calls['attempted'] += active_ledger['attempted_member_calls']
                    total_calls['returned'] += active_ledger['returned_member_calls']
                    if device is not None and device.type == 'cuda' and torch.cuda.is_initialized():
                        missing.update(CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                                       CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
                collection.append(missing)
        write(output / 'compact' / 'COLLECTION.json', {'cells': collection, 'all18_accounted': len(collection) == 18})
        final = usage()
        versions = {}
        for name in ('numpy', 'torch', 'torch_geometric', 'ogb', 'torch_sparse'):
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = None
        cost = {'schema': 'internal-be-molhiv-readout-cost-v1', 'event_id': cfg['event_id'],
                'final_receipt_written': True, 'complete18_readout': bool(analysis and analysis['complete18_readout']),
                'inclusive_wall_seconds': time.monotonic() - began,
                'CPU_user_seconds': final['CPU_user_seconds'] - initial['CPU_user_seconds'],
                'CPU_system_seconds': final['CPU_system_seconds'] - initial['CPU_system_seconds'],
                'process_cumulative_peak_RSS_bytes': final['process_cumulative_peak_RSS_bytes'],
                'authorized_work': cfg['authorized_work'], 'stage_seconds': stages, 'member_calls': total_calls,
                'max_complete_family_member_calls': 2079, 'raw_storage_bytes': sum(p.stat().st_size for p in (output / 'raw').rglob('*') if p.is_file()),
                'all_readout_storage_bytes_before_cost_receipt': sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
                'runtime': {'torch': str(torch.__version__) if torch else None,
                            'CPU_threads': torch.get_num_threads() if torch else None,
                            'installed_package_versions': versions, 'Python': sys.version},
                'CUDA_peak_allocated_bytes': max((row['CUDA_peak_allocated_bytes'] for row in collection if row.get('CUDA_peak_allocated_bytes') is not None), default=None),
                'CUDA_peak_reserved_bytes': max((row['CUDA_peak_reserved_bytes'] for row in collection if row.get('CUDA_peak_reserved_bytes') is not None), default=None),
                'failure': error, 'external_kill_final_cost_limitation': 'A hard kill can prevent this receipt; retain actual external terminal/cost custody'}
        write(output / 'compact' / 'COST.json', cost)
    require(error is None and analysis is not None and analysis['complete18_readout'],
            'Readout incomplete: every failed/unlaunched/invalid/collection-failed cell is retained in compact output')
    print('Complete separate MolHIV development readout: ' + str(output / 'compact'))


if __name__ == '__main__':
    main()
