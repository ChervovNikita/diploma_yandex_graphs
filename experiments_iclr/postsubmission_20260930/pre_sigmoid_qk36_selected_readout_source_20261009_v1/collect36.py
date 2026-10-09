"""Inactive postclosure Q/K36 callable/CLI, using the existing one-pass collector."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('_qk36_release_gate', HERE / 'gate36.py')
_gate = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_gate)
SEEDS, KINDS, OPERATORS, MAX_FORWARDS = _gate.SEEDS, _gate.KINDS, _gate.OPERATORS, _gate.MAX_FORWARDS


def run(release_path, release_sha256):
    os.umask(0o077); started = time.monotonic(); usage_start = resource.getrusage(resource.RUSAGE_SELF)
    cfg, pins, records, output, g, route, originals, supervision = _gate.consume(release_path, release_sha256)
    g.require(socket.gethostname() == route['hostname'] and Path.cwd().resolve() == Path(route['repository']).resolve()
        and _gate.PHASE == Path(route['phase']).resolve() and Path(sys.executable).resolve() == Path(route['python']).resolve()
        and ([] if not os.environ.get('PYTHONPATH') else os.environ['PYTHONPATH'].split(os.pathsep)) == route['PYTHONPATH']
        and os.environ.get('CUDA_VISIBLE_DEVICES') == route['GPU_uuid'], 'Exact explicitly admitted qualified serving route')
    free = int(subprocess.check_output(['nvidia-smi', '--id=' + route['GPU_uuid'], '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=5).strip()) * 1024**2
    g.require(free >= supervision['owned_GPU_bytes'], 'Fresh reader headroom; no other job signaling')
    output.mkdir(mode=0o700); (output / 'raw').mkdir(mode=0o700); (output / 'compact').mkdir(mode=0o700)
    collector = g.module(g.bound(pins['reuse']['variable_member_collect']), '_qk36_original_variable_member_collector')
    write = collector.write
    collection = dict(schema='qk36-selected-state-predictions-v1', cells=records,
        cohorts={kind: {str(seed): dict(available=False) for seed in SEEDS} for kind in KINDS},
        status='running', all36_complete_and_actual_route_custody_before_metrics=True,
        readout_route_id=cfg['readout_route_id'], original_training_routes_preserved=True,
        TEST_access=False, training=False, reselection=False, calibration=False, automatic_retry=False)
    cost = dict(schema='qk36-selected-state-readout-cost-v1', status='running', release=g.binding(release_path),
        started_UTC=datetime.now(timezone.utc).isoformat(), metadata_gate_runtime_and_headroom_seconds=time.monotonic() - started,
        maximum_member_forwards=108, attempted_member_forwards=0, completed_member_forwards=0,
        checkpoint_deserialization_attempts=0, checkpoint_file_bytes_submitted_to_deserializer=0,
        optimizer_construction_attempts=0, optimizer_constructions=0, optimizer_construction_seconds=0.,
        Session_factory_attempts=0, Session_factory_completions=0,
        TRAIN_updates=0, backward_calls=0, Adam_steps=0, optimizer_history_restores=0, training_RNG_history_restores=0,
        raw_archive_load_attempts=0, raw_archive_bytes_submitted_to_decoder=0,
        peak_CUDA_allocated_bytes=0, peak_CUDA_reserved_bytes=0,
        fullgraph_nodes_per_call=11701, development_objects_per_call=5274,
        actual_serving_route=route, training_and_serving_route_differences_preserved=True,
        raw_arrays_server_only=True, remote_transfer_calls_in_this_program=0, TEST_access=False,
        outer_owner_exit_cleanup_and_compact_mirroring_cost=None,
        outer_cost_custody='Root retains actual external owner exit/reap/cleanup and compact mirror costs separately; unavailable here, never zero.',
        fresh_Adam_scope='Both original Session and operator-replacement generations are observed; successful whole panel constructs144 fresh Adams and restores none.',
        timing_scope='Original inference and CPU transfer are measured together; inclusive parent/child and constructor/reconstruction timings overlap.')
    write(output / 'compact/ORIGINAL_COSTS.json', originals)
    write(output / 'compact/GATE.json', dict(passed=True, plan=pins['global_plan'], execution_source_commit=pins['plan']['execution_source_commit'],
        readout_manifest_sha256=cfg['readout_manifest_sha256'], route_absence_receipts=cfg['route_absence_receipts'],
        fresh_all_route_custody=cfg['fresh_all_route_custody'], all36_selected_and_own_state_hashes_verified=True,
        all36_closed_before_metrics=True, TEST_access=False))
    def persist():
        write(output / 'compact/COLLECTION.json', collection); write(output / 'compact/COST.json', cost)
    def storage():
        g.require(sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) <= supervision['maximum_output_bytes'], 'Owned readout output cap')
    persist()
    try:
        setup = time.monotonic()
        # All36 original endpoints and actual root custody passed before these imports.
        import numpy as np
        import torch
        torch.set_num_threads(2)
        if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(supervision['owned_GPU_bytes'] / torch.cuda.get_device_properties(0).total_memory, 0)
        versions = dict(torch=str(torch.__version__), numpy=np.__version__, **{
            name: importlib.metadata.version(name.replace('_', '-')) for name in ('torch_geometric', 'torch_scatter', 'torch_sparse', 'ogb')})
        g.require(versions == {name: route[name] for name in versions}, 'Declared matched scientific providers')
        routing = g.module(g.bound(pins['reuse']['routing_program']), '_qk36_exact_route_constructor')
        routing.sources(cfg['readout_route_id']); routing.verify_route(route); routing.verify_roles(route, pins['route_source_pins'])
        public = g.module(g.bound(pins['reuse']['public_portable']), '_qk36_original_public')
        prior_public = sys.modules.get('portable')
        g.require(prior_public is None or Path(prior_public.__file__).resolve() == Path(public.__file__).resolve(), 'Fresh readout process or exact public-module cache')
        sys.modules['portable'] = public
        data = g.module(g.bound(pins['reuse']['public_data']), '_qk36_original_data_interface')
        hooks = g.module(HERE / 'collection_hooks.py', '_qk36_selected_state_hooks')
        readout = g.module(HERE / 'readout36.py', '_qk36_frozen_pilot_readout')
        analysis, contract, arrays_for, analyse, adapted = readout.configure(g, pins)
        adapted['collect_cell_AST_sha256'] = hooks.configure(g, pins, routing, cfg['readout_route_id'], collector)
        g.require(adapted == pins['adapted_AST_sha256'], 'Reviewed small in-memory adaptations unchanged')
        original_load = contract.load
        def charged_load(np_arg, path, expected):
            cost['raw_archive_load_attempts'] += 1; cost['raw_archive_bytes_submitted_to_decoder'] += expected['bytes']
            return original_load(np_arg, path, expected)
        contract.load = charged_load
        native, native_origin = public.native_sources('wikics', g.bound(pins['route_source_pins']['native']))
        cost.update(source_and_runtime_loading_seconds=time.monotonic() - setup, runtime_versions=versions, native=native_origin)
        setup = time.monotonic(); roles = pins['route_source_pins']['roles']
        train, valid, origin = data.load_train_valid('wikics', g.bound(roles['train']), g.bound(roles['valid']))
        ordered = dict(x=train['x'], edge_index=train['edge_index'], train_ids=train['ids'], train_y=train['y'], valid_ids=valid['ids'], valid_y=valid['y'])
        for name, value in ordered.items():
            expected = pins['ordered_role_fingerprints'][name]
            g.require(str(value.dtype) == expected['dtype'] and list(value.shape) == expected['shape']
                and hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest() == expected['contiguous_raw_bytes_sha256'], 'Complete ordered role fingerprint: ' + name)
        iterator = data.validation_batches('wikics', train, valid, 'cuda:0'); batch, truth = next(iterator)
        g.require(next(iterator, None) is None and len(truth) == 5274 and tuple(batch['x'].shape) == (11701, 300)
            and tuple(batch['edge_index'].shape) == (2, 442907), 'Single full original graph/development batch')
        torch.cuda.synchronize()
        cost.update(data_loading_and_transfer_seconds=time.monotonic() - setup, data=origin,
            peak_CUDA_allocated_bytes=int(torch.cuda.max_memory_allocated()), peak_CUDA_reserved_bytes=int(torch.cuda.max_memory_reserved()))
        originals_for_collection = dict(selection=public._core()['selection'], analysis=contract)
        config = public.recipe('wikics')
        def collect(row):
            cohort = collector.collect_cell(torch, np, originals_for_collection, native, batch, truth, row, {}, config, output, cost, row)
            if row['collection_status'] == 'complete':
                row['raw_archive'] = row['raw_prediction_archive']
            else:
                row['failure'] = row.get('unavailable_reason')
            if row['operator'] == 'native_tied':
                collection['cohorts'][row['kind']][str(row['seed'])] = (dict(cohort, archive=cohort['cohort_archive'])
                    if cohort else dict(available=False, failure=row.get('failure')))
            persist(); storage()
        for kind in KINDS:
            for seed in SEEDS:
                collect(next(r for r in records if (r['kind'], r['operator'], r['seed']) == (kind, 'native_tied', seed)))
        frozen = all(c['available'] for seeds in collection['cohorts'].values() for c in seeds.values())
        collection['all9_native_tied_baselines_frozen_before_candidates'] = frozen
        write(output / 'compact/NATIVE_TIED_COHORTS.json', dict(cohorts=collection['cohorts'], all9_frozen=frozen,
            candidate_prediction_inspection_started=False, within_family_native_tied_only=True, overlapping_not_additive=True))
        if frozen:
            for operator in OPERATORS[1:]:
                for kind in KINDS:
                    for seed in SEEDS:
                        collect(next(r for r in records if (r['kind'], r['operator'], r['seed']) == (kind, operator, seed)))
        else:
            for seeds in collection['cohorts'].values():
                for cohort in seeds.values(): cohort.update(available=False, failure='Entire9 native_tied freeze required')
            for row in records:
                if row['collection_status'] == 'pending':
                    row.update(collection_status='skipped_before_candidate_forward', failure=dict(reason='Entire9 native_tied baseline freeze required', automatic_retry=False))
        g.require(0 <= cost['completed_member_forwards'] <= cost['attempted_member_forwards'] <= 108, 'Fixed108 attempted selected member inference budget')
        all_collected = all(r['collection_status'] == 'complete' for r in records)
        if all_collected:
            g.require(cost['attempted_member_forwards'] == cost['completed_member_forwards'] == 108
                and cost['Session_factory_completions'] == 36 and cost['optimizer_constructions'] == 144, 'Exact108 calls,36 fresh factories,144 actual Adam constructions, zero steps')
        collection['status'] = 'complete' if all_collected else 'complete_with_retained_collection_failures'; persist()
        setup = time.monotonic(); decision = analyse(np, g, pins, output, collection, analysis, contract)
        cost['analysis_seconds'] = time.monotonic() - setup
        if not collection['analysis_complete']: collection['status'] = 'complete_with_retained_failures'
        cost['status'] = collection['status']; persist(); storage()
        return dict(output=str(output / 'compact'), status=collection['status'], pilot_decision=decision)
    except Exception as error:
        collection['status'] = cost['status'] = 'retained_pipeline_failure'
        collection['failure'] = dict(error_type=type(error).__name__, error=str(error), automatic_retry=False)
        for row in records:
            if row['collection_status'] == 'pending': row.update(collection_status='skipped_pipeline_failure', failure=collection['failure'])
        write(output / 'compact/FAILURE.json', collection['failure']); persist(); raise
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        cost.update(inclusive_wall_seconds=time.monotonic() - started, CPU_user_seconds=usage.ru_utime - usage_start.ru_utime,
            CPU_system_seconds=usage.ru_stime - usage_start.ru_stime,
            peak_RSS_bytes=int(usage.ru_maxrss * (1024 if sys.platform.startswith('linux') else 1)),
            raw_server_only_storage_bytes=sum(p.stat().st_size for p in (output / 'raw').iterdir() if p.is_file()),
            compact_storage_bytes_before_final_receipt=sum(p.stat().st_size for p in (output / 'compact').iterdir() if p.is_file()),
            per_cell_costs=[{k:v for k,v in row.items() if k in ('cell','kind','operator','seed','collection_status','failure')
                or k.endswith('_seconds') or k.endswith('_member_forwards') or k.startswith('peak_') or 'construction' in k or 'factory' in k} for row in records],
            finished_UTC=datetime.now(timezone.utc).isoformat())
        write(output / 'compact/COST.json', cost)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True); args = parser.parse_args()
    print(run(args.release, args.release_sha256)['output'])


if __name__ == '__main__': main()
