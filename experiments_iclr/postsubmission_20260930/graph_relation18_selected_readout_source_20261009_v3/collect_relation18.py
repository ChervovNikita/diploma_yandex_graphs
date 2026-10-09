"""Inactive CLI/callable: 48 relation forwards plus six exact historical archives."""
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
_spec = importlib.util.spec_from_file_location('_relation18_release_gate', HERE / 'readout_gate.py')
_release_gate = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_release_gate)
consume = _release_gate.consume
SEEDS, POLICIES, MAX_FORWARDS = _release_gate.SEEDS, _release_gate.POLICIES, _release_gate.MAX_FORWARDS


def run(release_path, release_sha256):
    os.umask(0o077); started = time.monotonic(); usage_start = resource.getrusage(resource.RUSAGE_SELF)
    cfg, pins, records, output, gate, original_costs = consume(release_path, release_sha256)
    runtime = pins['runtime']; source = pins['data_source_pins']
    gate.require(socket.gethostname() == runtime['hostname'] and Path.cwd().resolve() == Path(runtime['repository']).resolve()
        and str(Path(sys.executable).absolute()) == runtime['python'] and os.environ.get('PYTHONPATH', '') == runtime['PYTHONPATH']
        and cfg['physical_gpu_uuid'] == runtime['physical_gpu_inventory'][0]
        and os.environ.get('CUDA_VISIBLE_DEVICES') == cfg['physical_gpu_uuid'], 'Exact normal77 readout runtime and physical GPU0')
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).splitlines()
    free = int(subprocess.check_output(['nvidia-smi', '--id=' + cfg['physical_gpu_uuid'], '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip()) * 1024**2
    gate.require(inventory == runtime['physical_gpu_inventory'] and free >= cfg['minimum_fresh_GPU_free_bytes'], 'Fresh source-reviewed reader headroom')
    output.mkdir(mode=0o700); (output / 'raw').mkdir(mode=0o700); (output / 'compact').mkdir(mode=0o700)
    legacy = gate.module(gate.bound(pins['reuse']['legacy_collect']), '_relation18_original_collector')
    write = legacy.write
    collection = dict(schema='relation18-selected-state-collection-v1', cells=records,
        cohorts={str(s): dict(available=False) for s in SEEDS}, status='running',
        whole_relation12_original12_union_historical24_closed_before_opening=True,
        historical_archive_choice='reuse_exact_six_archives', TEST_access=False,
        training=False, automatic_retry=False, reselection=False, calibration=False)
    cost = dict(schema='relation18-selected-state-readout-cost-v1', status='running',
        started_UTC=datetime.now(timezone.utc).isoformat(), release=gate.binding(release_path),
        metadata_gate_runtime_and_headroom_seconds=time.monotonic() - started,
        maximum_member_forwards=MAX_FORWARDS, attempted_member_forwards=0, completed_member_forwards=0,
        historical_control_member_forwards=0, checkpoint_deserialization_attempts=0,
        checkpoint_bytes_submitted_to_deserializer=0, optimizer_construction_attempts=0,
        optimizer_constructions=0, optimizer_construction_seconds=0., Session_construction_attempts=0,
        Session_constructions=0, Session_construction_seconds=0., TRAIN_updates=0, backward_calls=0, Adam_steps=0,
        optimizer_history_restores=0, training_RNG_history_restores=0,
        peak_CUDA_allocated_bytes=0, peak_CUDA_reserved_bytes=0,
        fullgraph_nodes_per_call=11701, development_objects_per_call=5274,
        historical_archive_open_attempts=0, historical_archive_bytes_opened=0,
        raw_archive_load_attempts=0, raw_archive_bytes_submitted_to_decoder=0,
        raw_arrays_server_only=True, TEST_access=False, remote_transfer_calls_in_this_program=0,
        external_exit_cleanup_transfer_cost=None,
        external_cost_authority='Root adds actual outer supervisor exit/reap/cleanup and compact mirror costs separately; unavailable here, never treated as zero.',
        transfer_timing='Data transfer and member inference/CPU transfer measured together by reused collector; no estimated separated timings.',
        original_cost_preservation='Full original successful/failed training, selection, owner, admission and resource histories in ORIGINAL_COSTS.json; unknown fields retained.')
    write(output / 'compact/ORIGINAL_COSTS.json', original_costs)
    write(output / 'compact/GATE.json', dict(passed=True, release=cost['release'], readout_manifest_sha256=cfg['readout_manifest_sha256'],
        relation_family_closure=cfg['relation_family_closure'], original12_union_closure=cfg['original12_union_closure'],
        historical_prediction_release=cfg['historical_prediction_release'], terminal_evidence=cfg['terminal_evidence'],
        source_selected_and_role_hashes_verified=True, all18_exact_slots=True, TEST_access=False))
    def persist():
        write(output / 'compact/COLLECTION.json', collection); write(output / 'compact/COST.json', cost)
    def skip_pending(reason):
        for row in records:
            if row['collection_status'] == 'pending':
                row.update(collection_status='skipped_before_opening', failure=dict(reason=reason, automatic_retry=False))
    def storage_gate():
        size = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        gate.require(size <= cfg['maximum_output_bytes'], 'Owned output storage cap; retain failure, no next opening')
    persist()
    try:
        setup = time.monotonic()
        # First numerical imports/deserialization follow the entire metadata gate.
        import numpy as np
        import torch
        torch.set_num_threads(2)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(cfg['owned_GPU_cap_bytes'] / torch.cuda.get_device_properties(0).total_memory, 0)
        versions = dict(torch=str(torch.__version__), numpy=np.__version__, **{
            name: importlib.metadata.version(name) for name in ('torch-geometric', 'torch-scatter', 'torch-sparse', 'ogb')})
        gate.require(versions == {k: runtime[k] for k in versions}, 'Original matched normal77 provider versions')
        public = gate.module(gate.bound(pins['relation_source_pins']['public_portable']), '_relation18_original_public')
        sys.modules['portable'] = public
        data = gate.module(gate.bound(pins['reuse']['public_data']), 'data_interface'); sys.modules['data_interface'] = data
        relation_root = gate.bound(pins['reuse']['relation_train']).parent
        for name in ('permissions', 'replay_adapter'):
            prior = sys.modules.get(name)
            gate.require(prior is None or Path(prior.__file__).resolve().parent == relation_root, 'Fresh relation dependency namespace')
        sys.path.insert(0, str(relation_root))
        relation = gate.module(gate.bound(pins['reuse']['relation_train']), '_relation18_original_train')
        hooks = gate.module(HERE / 'collection_hooks.py', '_relation18_reconstruction_hooks')
        readout = gate.module(HERE / 'readout.py', '_relation18_M1_M4_analysis')
        analysis, contract, pair_ast = readout.configure(gate, pins)
        collection_ast = hooks.bind_collector(gate, pins, relation, legacy)
        gate.require(collection_ast == pins['adapted_collect_cell_AST_sha256']
            and pair_ast == pins['pooled_pair_AST_sha256'], 'Reviewed in-memory adaptations unchanged')
        load_core = contract.load
        def charged_load(np_arg, path, expected):
            cost['raw_archive_load_attempts'] += 1
            cost['raw_archive_bytes_submitted_to_decoder'] += expected['bytes']
            return load_core(np_arg, path, expected)
        contract.load = charged_load
        native, native_origin = public.native_sources('wikics', gate.bound(source['polynormer']))
        cost.update(source_and_runtime_loading_seconds=time.monotonic() - setup, runtime_versions=versions, native=native_origin)
        setup = time.monotonic()
        train, valid, origin = data.load_train_valid('wikics', gate.bound(source['train']), gate.bound(source['development']))
        ordered = dict(x=train['x'], edge_index=train['edge_index'], train_ids=train['ids'], train_y=train['y'], valid_ids=valid['ids'], valid_y=valid['y'])
        for name, tensor in ordered.items():
            expected = source['ordered_raw_tensor_fingerprints'][name]
            gate.require(str(tensor.dtype) == expected['dtype'] and list(tensor.shape) == expected['shape']
                and hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest() == expected['contiguous_raw_bytes_sha256'], 'Original complete ordered role fingerprint: ' + name)
        iterator = data.validation_batches('wikics', train, valid, 'cuda:0'); batch, truth = next(iterator)
        gate.require(next(iterator, None) is None and len(truth) == 5274 and tuple(batch['x'].shape) == (11701, 300)
            and tuple(batch['edge_index'].shape) == (2, 442907), 'Complete original graph/development batch')
        torch.cuda.synchronize()
        cost.update(data_loading_and_transfer_seconds=time.monotonic() - setup, data=origin,
            peak_CUDA_allocated_bytes=int(torch.cuda.max_memory_allocated()), peak_CUDA_reserved_bytes=int(torch.cuda.max_memory_reserved()))
        def collect(condition):
            for seed in SEEDS:
                row = next(r for r in records if (r['seed'], r['condition']) == (seed, condition))
                cohort = legacy.collect_cell(torch, np, public, None, analysis, contract, native, batch, truth, row, source, output, cost)
                if condition == 'alphaF':
                    collection['cohorts'][str(seed)] = cohort or dict(available=False, failure=row.get('failure'))
                persist(); storage_gate()
        collect('alphaF')
        frozen = all(collection['cohorts'][str(seed)].get('available') is True for seed in SEEDS)
        collection['all_three_alphaF_cohorts_frozen_before_candidates'] = frozen
        write(output / 'compact/ALPHAF_COHORTS.json', dict(cohorts=collection['cohorts'], all_three_frozen=frozen,
            source_condition='alphaF', candidate_prediction_inspection_started=False,
            frozen_before_candidate_predictions=True, overlapping_not_additive=True))
        if frozen:
            for condition in POLICIES[1:]:
                collect(condition)
            for row in records:
                if not row['historical_reference']:
                    continue
                began = time.monotonic(); row['collection_status'] = 'opening_existing_exact_archive'
                cost['historical_archive_open_attempts'] += 1
                row['raw_archive'] = row['staged_archive']
                try:
                    arrays = readout.arrays_for(np, gate, row, contract)
                    gate.require(np.array_equal(arrays['valid_ids'], batch['ids'].cpu().numpy())
                        and np.array_equal(arrays['truth'], truth.numpy()), 'Historical complete IDs/truth equal original relation development population')
                    cost['historical_archive_bytes_opened'] += row['original_archive']['bytes']
                    row['collection_status'] = 'complete'; row['derived_Brier_only_from_original_probability'] = True
                    del arrays
                except Exception as error:
                    row.update(collection_status='retained_historical_archive_failure', failure=dict(error_type=type(error).__name__, error=str(error), automatic_retry=False))
                finally:
                    row['historical_archive_validation_and_Brier_seconds'] = time.monotonic() - began
                    persist()
        else:
            for cohort in collection['cohorts'].values():
                cohort.update(available=False, failure='All three original alphaF banks/cohorts required')
            skip_pending('Entire alphaF3 freeze required before any candidate prediction or historical array opening')
        gate.require(0 <= cost['completed_member_forwards'] <= cost['attempted_member_forwards'] <= MAX_FORWARDS, 'At most48 actual relation member calls, including failures')
        all_collected = all(r['collection_status'] == 'complete' for r in records)
        if all_collected:
            gate.require(cost['attempted_member_forwards'] == cost['completed_member_forwards'] == MAX_FORWARDS
                and cost['Session_constructions'] == cost['optimizer_constructions'] == 12, 'Exactly48 relation calls and12 actual fresh Session/Adam constructions')
        collection['status'] = 'complete' if all_collected else 'complete_with_retained_failures'; persist()
        began = time.monotonic(); screen = readout.run(np, gate, pins, output, collection, analysis, contract)
        cost['analysis_seconds'] = time.monotonic() - began
        if not collection['analysis_complete']:
            collection['status'] = 'complete_with_retained_failures'
        cost['status'] = collection['status']; persist(); storage_gate()
        return dict(output=str(output / 'compact'), status=collection['status'], unchanged_screen=screen)
    except Exception as error:
        collection['status'] = cost['status'] = 'retained_pipeline_failure'
        collection['failure'] = dict(error_type=type(error).__name__, error=str(error), automatic_retry=False)
        skip_pending('Retained pipeline failure; no automatic retry or partial scoring')
        write(output / 'compact/FAILURE.json', collection['failure']); persist()
        raise
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        cost.update(inclusive_wall_seconds=time.monotonic() - started, CPU_user_seconds=usage.ru_utime - usage_start.ru_utime,
            CPU_system_seconds=usage.ru_stime - usage_start.ru_stime,
            peak_RSS_bytes=int(usage.ru_maxrss * (1024 if sys.platform.startswith('linux') else 1)),
            raw_server_only_new_storage_bytes=sum(p.stat().st_size for p in (output / 'raw').iterdir() if p.is_file()),
            exact_historical_reused_storage_bytes=sum(r['original_archive']['bytes'] for r in records if r['historical_reference']),
            compact_storage_bytes_before_final_receipt=sum(p.stat().st_size for p in (output / 'compact').iterdir() if p.is_file()),
            per_cell_costs=[{k: v for k, v in r.items() if k in ('cell', 'seed', 'condition', 'collection_status', 'historical_reference', 'failure')
                or k.endswith('_seconds') or k.startswith('peak_') or k.endswith('_member_forwards')
                or 'construction' in k} for r in records], finished_UTC=datetime.now(timezone.utc).isoformat())
        write(output / 'compact/COST.json', cost)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True); parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    print(run(args.release, args.release_sha256)['output'])


if __name__ == '__main__':
    main()
