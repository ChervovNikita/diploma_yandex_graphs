"""Disabled selected-state Wiki24 collector. No numerical/model import at module scope.

Consumes the separately released selected-metadata export. No training admission,
live process polling, checkpoint reselection, optimizer, or automatic retry.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import resource
import sys
import time

ROOT = Path(__file__).resolve().parent
PHASE = ROOT.parent
SUITE = 'learnable_internal_be_contrastive_multitask_suite_20261007_v4'
SUITE_SHA = '76de82781e7fd496a5a3382b3a71a5781ea023dfe3777469cc005a2b19afbfce'
READER = 'internal_BE_Wiki24_closed_family_reader_source_20261007_v3'
READER_SHA = '476a59b56bd3377c6c93d0048a979f2fe6f08795e77b68e1452ebc874ec96042'
ARMS = ['single', 'single_contrastive', 'independent4', 'independent4_contrastive',
        'be_unit', 'be_init', 'be_unit_contrastive', 'be_init_contrastive']
SEEDS = [6101, 6203, 6307]
MAX_FORWARDS = 78


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temp, path)


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(PHASE.resolve())), sha256=sha(path), bytes=path.stat().st_size)


def bound(row):
    path = (PHASE / row['path']).resolve(strict=True)
    if not path.is_relative_to(PHASE.resolve()) or not path.is_file() or sha(path) != row['sha256']:
        raise ValueError('Exact bound phase file changed: ' + row['path'])
    if 'bytes' in row and path.stat().st_size != row['bytes']:
        raise ValueError('Exact bound file size changed')
    return path


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def consume(release_path, release_sha):
    if sha(release_path) != release_sha:
        raise ValueError('Exact root release digest required')
    cfg = read(release_path)
    for key in ('stage_enabled', 'root_execution_authorized', 'source_review_approved',
                'owner_and_children_terminal', 'trusted_checkpoint_deserialization_authorized',
                'prediction_collection_and_analysis_cost_charged'):
        if cfg.get(key) is not True:
            raise ValueError('Disabled pending root release: ' + key)
    if cfg.get('schema') != 'internal-be-Wiki24-selected-prediction-release-v1' or cfg.get('TEST_access') is not False or cfg.get('automatic_retry') is not False or cfg.get('maximum_member_forwards') != MAX_FORWARDS:
        raise ValueError('Fixed read-only selected prediction scope required')
    if cfg['collector_manifest_sha256'] != sha(ROOT / 'MANIFEST.json'):
        raise ValueError('Reviewed collector source changed')
    for row in read(ROOT / 'MANIFEST.json')['files']:
        path = ROOT / row['path']
        if sha(path) != row['sha256'] or path.stat().st_size != row['bytes']:
            raise ValueError('Collector source bytes changed: ' + row['path'])
    if sha(PHASE / READER / 'MANIFEST.json') != READER_SHA:
        raise ValueError('Exact sealed metadata reader required')
    export_path = bound(cfg['metadata_export']); export = read(export_path); gate = export['gate']
    if export.get('schema') != 'internal-be-Wiki24-selected-metadata-export-v1' or export.get('whole24_accounted') is not True or export.get('TEST_access') is not False or export.get('reselected') is not False or gate.get('passed') is not True or gate.get('family_closed') is not True:
        raise ValueError('Separately released authoritative whole24 metadata export required')
    if gate['reader_manifest_sha256'] != READER_SHA or gate['source_manifest_sha256'] != SUITE_SHA:
        raise ValueError('Exact reader and original selected-family source authority')
    roster = [(arm, seed, arm + '_' + str(seed)) for seed in SEEDS for arm in ARMS]
    for records in (gate['cells'], export['cells']):
        if [(row['arm'], row['seed'], row['cell']) for row in records] != roster:
            raise ValueError('Exact24 saved cells required; no survivor-only roster')
    if [row['status'] for row in gate['cells']] != [row['status'] for row in export['cells']]:
        raise ValueError('Metadata export must retain the original closure statuses')
    extraction_path = bound(cfg['metadata_extraction_cost']); extraction = read(extraction_path)
    if extraction.get('status') != 'complete' or extraction.get('metadata_export') != cfg['metadata_export']:
        raise ValueError('Separate completed extraction cost must bind this exact export')
    terminal_path = bound(cfg['terminal_evidence']); terminal = read(terminal_path)
    if terminal.get('owner_and_children_terminal') is not True or terminal.get('closure') != gate['closure'] or terminal.get('owner') != gate['owner']:
        raise ValueError('Root terminal evidence for this exact owner and closure required')
    # Saved closure/reap receipts are the reader's authority. No reimplementation
    # of its gate and no /proc polling or live-training supervisor admission.
    output = Path(cfg['output_directory']).resolve()
    if not output.is_relative_to(PHASE.resolve()) or output.exists() or not output.parent.is_dir():
        raise ValueError('Fresh server phase output required; no retry/resume')
    return cfg, export, extraction, output


def historical(meta):
    return {key: meta.get(key) for key in ('selected_VALID', 'member_VALID', 'selection',
        'selected_epochs', 'member_global', 'member_VALID_authority', 'own_selected_member_diagnostics')}


def record_for(state, meta):
    return dict(arm=state['arm'], seed=state['seed'], cell=state['cell'], family_status=state['status'],
        collection_status='pending' if state['status'] == 'complete' else 'excluded_family_failure_or_unlaunched',
        members=1 if state['arm'].startswith('single') else 4,
        historical_selected_metadata=historical(meta),
        unavailable_reason=None if state['status'] == 'complete' else state['closure_row'])


def restore(torch, model, saved, state, meta, config):
    if type(saved) is not dict or saved.get('job') != state['job'] or saved.get('config') != config:
        raise ValueError('Original selected snapshot job/config changed')
    model.load_state_dict(saved['model'], strict=True)
    if state['arm'] == 'independent4':
        if saved.get('evaluation_only') is not True or saved.get('candidate') != 'individual_best_bank_only':
            raise ValueError('Original independently own-selected evaluation-only bank required')
        modes = saved['body_global']
        if type(modes) is not list or len(modes) != 4 or not all(type(mode) is bool for mode in modes):
            raise ValueError('Original four own-selected body modes required')
        for body, mode in zip(model.models, modes):
            body.set_global(mode)
        streams = None
    else:
        if saved.get('evaluation_only') is True or type(saved.get('global')) is not bool:
            raise ValueError('Original coherent joint selected snapshot required')
        model.set_global(saved['global']); streams = saved['streams']
        if len(streams) != model.members:
            raise ValueError('Original selected member RNG stream bank')
    body_modes = [body.body._global for body in model.models]
    member_modes = [body_modes[m if model.independent else 0] for m in range(model.members)]
    if member_modes != meta['member_global']:
        raise ValueError('Structural selected serving-mode metadata changed')
    model.eval()
    return body_modes, member_modes, streams


def collect_cell(torch, np, originals, sources, batch, truth, state, meta, config, output, cost, record):
    start = time.monotonic(); model = None; saved = None; completed_logits = []
    phase = 'construction'; phase_started = start
    partial = output / 'raw' / (state['cell'] + '_partial.npz')
    persistence_seconds = 0.0
    record['evaluated_full_graph_nodes'] = int(batch['x'].shape[0])
    record['scored_merged_development_nodes'] = int(truth.numel())
    torch.cuda.reset_peak_memory_stats(0)
    try:
        seed = state['seed']
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        construction = time.monotonic()
        model = originals['models'].Ensemble('wikics', state['arm'], seed, config['model'], sources).to('cuda:0')
        model.set_global(False)
        torch.cuda.synchronize(0)
        record['construction_seconds'] = time.monotonic() - construction
        checkpoint = state['selected_checkpoint']; checkpoint_path = bound(checkpoint)
        record['selected_checkpoint'] = checkpoint
        load_start = time.monotonic()
        phase = 'checkpoint_load_and_mode_restore'; phase_started = load_start
        cost['checkpoint_deserialization_attempts'] += 1
        cost['checkpoint_file_bytes_submitted_to_deserializer'] += checkpoint['bytes']
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        body_modes, member_modes, streams = restore(torch, model, saved, state, meta, config)
        bound(checkpoint)
        record['selected_body_modes'] = body_modes; record['selected_member_modes'] = member_modes
        record['stored_joint_dropout_streams_used'] = streams is not None
        torch.cuda.synchronize(0)
        record['checkpoint_load_and_mode_restore_seconds'] = time.monotonic() - load_start
        # No optimizer construction or restoration. Keep only saved stream values
        # for original forward RNG context; ordinary own bank has no such bank.
        del saved; saved = None
        forward_start = time.monotonic(); logits_rows = []
        phase = 'forward_and_CPU_transfer'; phase_started = forward_start
        with torch.no_grad():
            for member in range(model.members):
                if cost['attempted_member_forwards'] >= MAX_FORWARDS:
                    raise ValueError('Fixed maximum78 member calls exhausted')
                cost['attempted_member_forwards'] += 1
                record['attempted_member_forwards'] += 1
                persistence_start = time.monotonic()
                write(output / 'compact' / 'PROGRESS.json', dict(cell=state['cell'], member_index0=member,
                    stage='member_call_about_to_start', attempted_member_forwards=cost['attempted_member_forwards'],
                    completed_member_forwards=cost['completed_member_forwards'], automatic_retry=False))
                persistence_seconds += time.monotonic() - persistence_start
                context = torch.random.fork_rng(devices=[0]) if streams is not None else nullcontext()
                with context:
                    if streams is not None:
                        torch.set_rng_state(streams[member]['cpu']); torch.cuda.set_rng_state(streams[member]['cuda'])
                    logits, representation = model.member_forward(batch, member)
                del representation
                torch.cuda.synchronize(0)
                cost['completed_member_forwards'] += 1
                record['completed_member_forwards'] += 1
                completed_logits.append(logits.detach().cpu().numpy().copy())
                logits_rows.append(logits)
                persistence_start = time.monotonic()
                np.savez(partial, member_logits=np.stack(completed_logits), valid_ids=batch['ids'].cpu().numpy(), truth=truth.numpy())
                record['partial_prediction_archive'] = binding(partial)
                write(output / 'compact' / 'PROGRESS.json', dict(cell=state['cell'], member_index0=member,
                    stage='member_returned_and_partial_logits_saved', attempted_member_forwards=cost['attempted_member_forwards'],
                    completed_member_forwards=cost['completed_member_forwards'], partial_prediction_archive=record['partial_prediction_archive'],
                    automatic_retry=False))
                persistence_seconds += time.monotonic() - persistence_start
            logits = torch.stack(logits_rows)
            member_probability = logits.softmax(-1)
            pooled = member_probability.mean(0)
            originals['selection'].finite_predictions(logits, pooled)
            truth_gpu = truth.to('cuda:0')
            member_logp = logits.log_softmax(-1).gather(-1, truth_gpu[None, :, None].expand(model.members, -1, 1)).squeeze(-1)
            member_nll = -member_logp
            pool_nll = -(torch.logsumexp(member_logp, dim=0) - math.log(model.members))
            logits_cpu = logits.cpu(); pooled_cpu = pooled.cpu()
            predictions = logits_cpu.argmax(-1); pool_prediction = pooled_cpu.argmax(-1)
            record['source_float32_reevaluation'] = dict(selected_VALID=float((pool_prediction == truth).float().mean()),
                member_VALID=[float((pred == truth).float().mean()) for pred in predictions])
            arrays = dict(valid_ids=batch['ids'].cpu().numpy().copy(), truth=truth.numpy().copy(),
                member_logits=logits_cpu.numpy().copy(), member_probability=member_probability.cpu().numpy().copy(),
                pool_probability=pooled_cpu.numpy().copy(), member_prediction=predictions.numpy().copy(),
                pool_prediction=pool_prediction.numpy().copy(), member_nll=member_nll.cpu().numpy().copy(),
                pool_nll=pool_nll.cpu().numpy().copy())
        record['forward_and_CPU_transfer_seconds'] = time.monotonic() - forward_start - persistence_seconds
        originals['analysis'].validate(np, arrays, model.members)
        serial_start = time.monotonic()
        phase = 'prediction_serialization'; phase_started = serial_start
        archive_path = output / 'raw' / (state['cell'] + '.npz')
        np.savez(archive_path, **arrays)
        record['raw_prediction_archive'] = binding(archive_path)
        if partial.exists():
            partial.unlink()
            record.pop('partial_prediction_archive', None)
        record['prediction_serialization_seconds'] = time.monotonic() - serial_start
        record['collection_status'] = 'complete'
        # A baseline is frozen immediately, before collection of ANY candidate.
        if state['arm'] == 'be_init':
            cohort_start = time.monotonic()
            phase = 'baseline_cohort_derivation_and_serialization'; phase_started = cohort_start
            masks = originals['analysis'].baseline_cohorts(np, arrays)
            cohort_path = output / 'raw' / ('be_init_cohorts_' + str(seed) + '.npz')
            np.savez(cohort_path, **masks)
            record['baseline_cohort_seconds'] = time.monotonic() - cohort_start
            return dict(available=True, baseline_cell=state['cell'], baseline_prediction_archive=record['raw_prediction_archive'],
                cohort_archive=binding(cohort_path), counts={name: int(masks[name].sum()) for name in originals['analysis'].COHORTS},
                frozen_before_candidate_prediction_inspection=True)
        return None
    except Exception as error:
        record['collection_status'] = 'retained_collection_failure'
        record['unavailable_reason'] = dict(error_type=type(error).__name__, error=str(error), automatic_retry=False,
                                            failed_phase=phase, failed_phase_seconds=time.monotonic() - phase_started)
        record['failed_phase'] = phase; record['failed_phase_seconds'] = time.monotonic() - phase_started
        if completed_logits:
            np.savez(partial, member_logits=np.stack(completed_logits), valid_ids=batch['ids'].cpu().numpy(), truth=truth.numpy())
            record['partial_prediction_archive'] = binding(partial)
        return None
    finally:
        record['inclusive_cell_seconds'] = time.monotonic() - start
        record['intermediate_progress_and_partial_storage_seconds'] = persistence_seconds
        record['peak_CUDA_allocated_bytes'] = int(torch.cuda.max_memory_allocated(0))
        record['peak_CUDA_reserved_bytes'] = int(torch.cuda.max_memory_reserved(0))
        cost['peak_CUDA_allocated_bytes'] = max(cost['peak_CUDA_allocated_bytes'], record['peak_CUDA_allocated_bytes'])
        cost['peak_CUDA_reserved_bytes'] = max(cost['peak_CUDA_reserved_bytes'], record['peak_CUDA_reserved_bytes'])
        del model, saved
        gc.collect(); torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True, type=Path); parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args(); os.umask(0o077)
    started = time.monotonic(); usage_start = resource.getrusage(resource.RUSAGE_SELF)
    cfg, export, extraction, output = consume(args.release, args.release_sha256)
    output.mkdir(mode=0o700); (output / 'raw').mkdir(mode=0o700); (output / 'compact').mkdir(mode=0o700)
    gate = export['gate']; config = gate['config']
    records = [record_for(state, meta) for state, meta in zip(gate['cells'], export['cells'])]
    collection = dict(schema='internal-be-Wiki24-selected-prediction-collection-v1', cells=records,
        cohorts={str(seed): dict(available=False, unavailable_reason='No complete baseline prediction and cohort seal yet') for seed in SEEDS},
        whole24_accounted=True, TEST_access=False, automatic_retry=False,
        candidate_prediction_inspection_started=False, status='running')
    cost = dict(schema='internal-be-Wiki24-selected-prediction-cost-v1', status='failed', started_UTC=datetime.now(timezone.utc).isoformat(),
        release=binding(args.release), metadata_export=cfg['metadata_export'], metadata_extraction_cost=cfg['metadata_extraction_cost'],
        separate_metadata_extraction_measured_cost=extraction, terminal_evidence=cfg['terminal_evidence'],
        reader_manifest_sha256=READER_SHA, original_suite_manifest_sha256=SUITE_SHA,
        maximum_member_forwards=MAX_FORWARDS, expected_member_forwards=sum(row['members'] for row in records if row['family_status'] == 'complete'),
        attempted_member_forwards=0, completed_member_forwards=0, checkpoint_deserialization_attempts=0,
        checkpoint_file_bytes_submitted_to_deserializer=0, peak_CUDA_allocated_bytes=0, peak_CUDA_reserved_bytes=0,
        expected_full_feature_graph_nodes=11701, expected_merged_development_nodes_per_complete_cell=5274,
        TRAIN_updates=0, auxiliary_updates=0, backward_calls=0, optimizer_constructions=0, TEST_access=False,
        raw_arrays_server_only=True, no_retry_or_resume=True)
    write(output / 'compact' / 'COLLECTION.json', collection)
    try:
        os.environ['OMP_NUM_THREADS'] = '2'; os.environ['MKL_NUM_THREADS'] = '2'
        source_start = time.monotonic()
        if sha(PHASE / SUITE / 'runtime.py') != 'f6ba21d6a834c51fed4ebca99f328d727180f0d73c62c190f2f11df9f57eb38b':
            raise ValueError('Frozen original runtime source changed before import')
        runtime = module(PHASE / SUITE / 'runtime.py', 'wiki24_original_runtime')
        if runtime.verify_manifest() != SUITE_SHA:
            raise ValueError('Frozen original source manifest changed')
        runtime.allocation()
        cost['runtime_versions'] = runtime.runtime_versions()
        # First numerical/model imports are below root release, saved whole24
        # reader gate, terminal authority, and exact original source verification.
        import numpy as np
        import torch
        factors = module(PHASE / SUITE / 'factors.py', 'wiki24_original_factors')
        previous_factors = sys.modules.get('factors')
        try:
            sys.modules['factors'] = factors
            models = module(PHASE / SUITE / 'models.py', 'wiki24_original_models')
        finally:
            if previous_factors is None:
                sys.modules.pop('factors', None)
            else:
                sys.modules['factors'] = previous_factors
        data = module(PHASE / SUITE / 'data.py', 'wiki24_original_data')
        selection = module(PHASE / SUITE / 'selection.py', 'wiki24_original_selection')
        analysis = module(ROOT / 'analyse.py', 'wiki24_selected_analysis')
        originals = dict(models=models, data=data, selection=selection, analysis=analysis)
        dependencies = read(PHASE / SUITE / 'DEPENDENCIES.json')
        sources = models.native_sources(runtime.PHASE, dependencies)
        cost['frozen_native_sources'] = [binding(runtime.PHASE / row['path']) for row in (
            dependencies['polynormer'], dependencies['ncn']['utils'], dependencies['ncn']['model'])]
        cost['source_and_runtime_loading_seconds'] = time.monotonic() - source_start
        data_start = time.monotonic()
        train, valid, authority = data.load_projection(runtime.PHASE, gate['data_manifest_binding'], 'wikics', True)
        iterator = data.valid_batches('wikics', train, valid, config['training'], 'cuda:0')
        batch, truth = next(iterator)
        if next(iterator, None) is not None:
            raise ValueError('Original single full-graph VALID batch required')
        torch.cuda.synchronize(0)
        cost['data_loading_and_complete_batch_transfer_seconds'] = time.monotonic() - data_start
        cost['full_feature_graph_nodes'] = int(batch['x'].shape[0])
        cost['scored_merged_development_nodes_per_complete_cell'] = int(truth.numel())
        cost['peak_CUDA_allocated_bytes'] = int(torch.cuda.max_memory_allocated(0))
        cost['peak_CUDA_reserved_bytes'] = int(torch.cuda.max_memory_reserved(0))
        cost['data_manifest'] = gate['data_manifest_binding']; cost['data_export_review'] = gate['data_export_review_binding']
        cost['data_payloads'] = {role: binding(data.bound(runtime.PHASE, row)) for role, row in authority['payloads'].items()}
        cost['original_config'] = gate['config_binding']
        cost['original_architecture'] = config['model']; cost['original_initialization'] = config['initialization']
        cost['original_serving'] = config['pool']
        metadata = {(row['arm'], row['seed']): row for row in export['cells']}
        by_key = {(row['arm'], row['seed']): row for row in records}
        baseline_first = [state for state in gate['cells'] if state['arm'] == 'be_init']
        remaining = [state for state in gate['cells'] if state['arm'] != 'be_init']
        for stage, states in [('baseline', baseline_first), ('remaining', remaining)]:
            if stage == 'remaining':
                # Written once after all baseline outcomes; immutable membership
                # is already saved in raw archives. No candidate may set a mask.
                write(output / 'compact' / 'BASELINE_COHORTS.json', dict(schema='internal-be-Wiki24-baseline-cohort-custody-v1',
                    cohorts=collection['cohorts'], candidate_prediction_inspection_started=False,
                    derivation='be_init only; entire original merged development population; no node selection from candidate predictions'))
                collection['candidate_prediction_inspection_started'] = True
            for state in states:
                record = by_key[(state['arm'], state['seed'])]
                if state['status'] != 'complete':
                    if state['arm'] == 'be_init':
                        collection['cohorts'][str(state['seed'])] = dict(available=False, unavailable_reason=state['closure_row'])
                    continue
                record.update(attempted_member_forwards=0, completed_member_forwards=0)
                cohort = collect_cell(torch, np, originals, sources, batch, truth, state,
                    metadata[(state['arm'], state['seed'])], config, output, cost, record)
                if state['arm'] == 'be_init':
                    collection['cohorts'][str(state['seed'])] = cohort or dict(available=False, unavailable_reason=record['unavailable_reason'])
                write(output / 'compact' / 'COLLECTION.json', collection)
                write(output / 'compact' / 'COST.json', cost)
        collection['status'] = 'complete_with_retained_failures' if any(row['collection_status'] == 'retained_collection_failure' for row in records) else 'complete'
        write(output / 'compact' / 'COLLECTION.json', collection)
        analysis_start = time.monotonic(); analysis.run(output, collection)
        cost['analysis_seconds'] = time.monotonic() - analysis_start
        cost['status'] = collection['status']
    except Exception as error:
        collection['status'] = 'retained_pipeline_failure'
        failure = dict(error_type=type(error).__name__, error=str(error), automatic_retry=False, completed_work_preserved=True)
        collection['failure'] = failure; cost['failure'] = failure
        for record in records:
            if record['collection_status'] == 'pending':
                record['collection_status'] = 'not_attempted_pipeline_failure'; record['unavailable_reason'] = failure
        write(output / 'compact' / 'FAILURE.json', failure)
        write(output / 'compact' / 'COLLECTION.json', collection)
        raise
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        cost.update(inclusive_wall_seconds=time.monotonic() - started,
            process_CPU_user_seconds=usage.ru_utime - usage_start.ru_utime,
            process_CPU_system_seconds=usage.ru_stime - usage_start.ru_stime,
            process_peak_RSS_bytes=int(usage.ru_maxrss * (1024 if sys.platform.startswith('linux') else 1)),
            raw_storage_bytes=sum(path.stat().st_size for path in (output / 'raw').iterdir() if path.is_file()),
            compact_storage_bytes_before_final_cost_receipt=sum(path.stat().st_size for path in (output / 'compact').iterdir() if path.is_file()),
            finished_UTC=datetime.now(timezone.utc).isoformat())
        cost['per_cell_costs'] = [{key: record[key] for key in ('cell', 'family_status', 'collection_status',
            'attempted_member_forwards', 'completed_member_forwards', 'construction_seconds',
            'evaluated_full_graph_nodes', 'scored_merged_development_nodes',
            'checkpoint_load_and_mode_restore_seconds', 'forward_and_CPU_transfer_seconds',
            'prediction_serialization_seconds', 'baseline_cohort_seconds', 'inclusive_cell_seconds',
            'intermediate_progress_and_partial_storage_seconds', 'failed_phase', 'failed_phase_seconds',
            'peak_CUDA_allocated_bytes', 'peak_CUDA_reserved_bytes', 'selected_checkpoint',
            'raw_prediction_archive', 'partial_prediction_archive') if key in record} for record in records]
        write(output / 'compact' / 'COST.json', cost)
    print(str(output / 'compact'))


if __name__ == '__main__':
    main()
