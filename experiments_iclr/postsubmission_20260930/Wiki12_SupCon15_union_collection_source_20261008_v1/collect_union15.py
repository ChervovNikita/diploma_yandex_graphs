"""Disabled whole-union entry; fifteen selected states and at most60 member calls."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import hashlib
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time
from union_gate import consume, CONDITIONS, MAX_FORWARDS, SEEDS

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args(); os.umask(0o077)
    started = time.monotonic(); usage_start = resource.getrusage(resource.RUSAGE_SELF)
    cfg, pins, source_pins, records, output, gate = consume(args.release, args.release_sha256)
    runtime = pins['runtime']; reuse = pins['reuse']
    gate.require(socket.gethostname() == runtime['hostname']
        and Path.cwd().resolve() == Path(runtime['repository']).resolve()
        and str(Path(sys.executable).absolute()) == runtime['python']
        and os.environ.get('PYTHONPATH', '') == runtime['PYTHONPATH']
        and cfg['physical_gpu_uuid'] == runtime['physical_gpu_inventory'][0]
        and os.environ.get('CUDA_VISIBLE_DEVICES') == cfg['physical_gpu_uuid'], 'Original normal77 reader runtime/GPU0')
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).splitlines()
    free = int(subprocess.check_output(['nvidia-smi', '--id=' + cfg['physical_gpu_uuid'], '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip()) * 1024**2
    gate.require(inventory == runtime['physical_gpu_inventory'] and free >= cfg['minimum_fresh_GPU_free_bytes'], 'Fresh reader headroom')
    output.mkdir(mode=0o700); (output / 'raw').mkdir(mode=0o700); (output / 'compact').mkdir(mode=0o700)
    # Reuse atomic persistence and collect_cell; no new forward implementation.
    gate.MAX_FORWARDS, gate.SEEDS, gate.CONDITIONS = MAX_FORWARDS, SEEDS, CONDITIONS
    collector = gate.module(gate.bound(reuse['legacy_collect']), '_union15_existing_collector')
    collector.MAX_FORWARDS = MAX_FORWARDS
    write = collector.write
    collection = dict(schema='Wiki12-SupCon15-union-selected-predictions-v1', cells=records,
        cohorts={str(seed): dict(available=False) for seed in SEEDS},
        alignment_cohorts={str(seed): dict(available=False) for seed in SEEDS}, status='running',
        whole_Wiki12_union_and_SupCon3_closed_before_opening=True, owners_and_all15_children_terminal_before_opening=True,
        original_failed_closure_immutable=True, Wiki12_union_closure=cfg['Wiki12_union_closure'],
        SupCon3_closure=cfg['SupCon3_closure'], old_terminal_custody=pins['old_terminal_custody'],
        TEST_access=False, training=False, reselection=False, calibration=False, automatic_retry=False)
    cost = dict(schema='Wiki12-SupCon15-union-collection-readout-cost-v1', status='running',
        started_UTC=datetime.now(timezone.utc).isoformat(), release=gate.binding(args.release),
        metadata_gate_seconds=time.monotonic() - started, maximum_member_forwards=MAX_FORWARDS,
        attempted_member_forwards=0, completed_member_forwards=0, checkpoint_deserialization_attempts=0,
        checkpoint_bytes_submitted_to_deserializer=0, peak_CUDA_allocated_bytes=0, peak_CUDA_reserved_bytes=0,
        fullgraph_nodes_per_call=11701, development_objects_per_call=5274, TRAIN_updates=0, backward_calls=0,
        optimizer_constructions=0, raw_logits_representations_and_labels_server_only=True, TEST_access=False,
        retained_original_failed_admission_cost='Bound old closure/preflight; this new cost record does not erase or replace it.')
    write(output / 'compact' / 'GATE.json', dict(passed=True, collection_manifest_sha256=cfg['collection_manifest_sha256'],
        Wiki12_union_closure=cfg['Wiki12_union_closure'], SupCon3_closure=cfg['SupCon3_closure'],
        old_terminal_custody=pins['old_terminal_custody'], terminal_evidence=cfg['terminal_evidence'],
        original_failed_closure_immutable=True, all15_original1100_endpoints_and_selected_hashes_verified=True,
        TEST_access=False, raw_arrays_server_only=True))
    def persist():
        write(output / 'compact' / 'COLLECTION.json', collection)
        write(output / 'compact' / 'COST.json', cost)
    def skip(conditions, reason):
        for record in records:
            if record['condition'] in conditions and record['collection_status'] == 'pending':
                record.update(collection_status='skipped_before_forward', failure=dict(reason=reason, automatic_retry=False))
    try:
        setup = time.monotonic()
        # First numerical imports occur only after the entire old10/new2/SupCon3 and terminal gate.
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
        gate.require(versions == {name: runtime[name] for name in versions}, 'Original matched provider versions')
        public = gate.module(gate.bound(reuse['public_adapter']), '_union15_original_public'); sys.modules['portable'] = public
        data = gate.module(gate.bound(reuse['public_data']), '_union15_original_data')
        recompute = gate.module(gate.bound(reuse['original_recompute']), '_union15_original_recompute'); sys.modules['recompute'] = recompute
        original = gate.module(gate.bound(reuse['original_adapter']), '_union15_original_v2_identity')
        supcon = gate.module(gate.bound(reuse['supcon_adapter']), '_union15_original_canonical_identity')
        supcon_recompute = gate.module(gate.bound(reuse['supcon_recompute']), '_union15_canonical_recompute_identity')
        hooks = gate.module(HERE / 'collection_hooks.py', '_union15_selected_restore_hooks')
        collector.restore = hooks.restore_factory(collector.restore)
        adapter = hooks.Adapter(original, supcon, supcon_recompute)
        readout = gate.module(HERE / 'readout.py', '_union15_paired_readout')
        analysis, contract = readout.configure(gate, pins, output, collection)
        restore_pins = dict(source_pins, supcon_wrapper_sha256=reuse['supcon_adapter']['sha256'],
            supcon_loss_sha256=reuse['supcon_loss']['sha256'], supcon_recompute_sha256=reuse['supcon_recompute']['sha256'])
        native, native_origin = public.native_sources('wikics', gate.bound(source_pins['polynormer']))
        cost.update(source_and_runtime_loading_seconds=time.monotonic() - setup, runtime_versions=versions, native=native_origin)
        setup = time.monotonic()
        train, valid, origin = data.load_train_valid('wikics', gate.bound(source_pins['train']), gate.bound(source_pins['development']))
        ordered = dict(x=train['x'], edge_index=train['edge_index'], train_ids=train['ids'], train_y=train['y'],
                       valid_ids=valid['ids'], valid_y=valid['y'])
        for name, tensor in ordered.items():
            expected = source_pins['ordered_raw_tensor_fingerprints'][name]
            gate.require(str(tensor.dtype) == expected['dtype'] and list(tensor.shape) == expected['shape']
                and hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest() == expected['contiguous_raw_bytes_sha256'],
                'Original ordered role fingerprint: ' + name)
        iterator = data.validation_batches('wikics', train, valid, 'cuda:0'); batch, truth = next(iterator)
        gate.require(next(iterator, None) is None and len(truth) == 5274
            and tuple(batch['x'].shape) == (11701, 300) and tuple(batch['edge_index'].shape) == (2, 442907), 'Complete original development graph')
        torch.cuda.synchronize(); cost.update(data_loading_and_transfer_seconds=time.monotonic() - setup, data=origin)
        def collect(condition):
            for seed in SEEDS:
                record = next(row for row in records if (row['seed'], row['condition']) == (seed, condition))
                cohort = collector.collect_cell(torch, np, public, adapter, analysis, contract, native,
                    batch, truth, record, restore_pins, output, cost)
                if condition == 'plain':
                    collection['cohorts'][str(seed)] = cohort or dict(available=False, failure=record.get('failure'))
                persist()
        collect('plain')
        plain_ready = all(collection['cohorts'][str(seed)].get('available') is True for seed in SEEDS)
        write(output / 'compact' / 'PLAIN_COHORTS.json', dict(cohorts=collection['cohorts'],
            all_three_frozen=plain_ready, source_condition='plain', frozen_before_candidate_prediction_inspection=True))
        if plain_ready:
            collect('alignment_only')
            alignment_ready = all(row['collection_status'] == 'complete' for row in records if row['condition'] == 'alignment_only')
            if alignment_ready:
                try:
                    for seed in SEEDS:
                        record = next(row for row in records if (row['seed'], row['condition']) == (seed, 'alignment_only'))
                        frozen_at = time.monotonic()
                        arrays = contract.load(np, output / 'raw' / (record['cell'] + '.npz'), record['raw_archive'])
                        analysis.validate(np, arrays, contract); masks = contract.baseline_cohorts(np, arrays)
                        path = output / 'raw' / ('alignment_cohorts_' + str(seed) + '.npz')
                        gate.require(not path.exists(), 'Original alignment cohorts freeze exactly once')
                        np.savez(path, **masks)
                        collection['alignment_cohorts'][str(seed)] = dict(available=True, alignment_cell=record['cell'],
                            alignment_archive=record['raw_archive'], archive=gate.binding(path), source_condition='alignment_only',
                            counts={name: int(masks[name].sum()) for name in contract.COHORTS},
                            frozen_before_SupCon_prediction_inspection=True)
                        record['alignment_cohort_freeze_seconds'] = time.monotonic() - frozen_at
                        del arrays, masks
                except Exception as error:
                    alignment_ready = False
                    collection['alignment_freeze_failure'] = dict(error_type=type(error).__name__, error=str(error), automatic_retry=False)
            if not alignment_ready:
                for cohort in collection['alignment_cohorts'].values():
                    cohort.update(available=False, failure='Entire original alignment3 collection/freeze required')
            write(output / 'compact' / 'ALIGNMENT_COHORTS.json', dict(cohorts=collection['alignment_cohorts'],
                all_three_frozen=alignment_ready, source_condition='alignment_only', frozen_before_SupCon_prediction_inspection=True))
            persist()
            collect('residual_only'); collect('combined')
            if alignment_ready:
                collect('supcon_eq2')
            else:
                skip(('supcon_eq2',), 'Entire original alignment3 collection/freeze required; no new candidate call')
        else:
            for cohort in collection['cohorts'].values():
                cohort.update(available=False, failure='Entire original plain3 cohort freeze required')
            skip(CONDITIONS[1:], 'Entire original plain3 cohort freeze required; no candidate call')
            write(output / 'compact' / 'PLAIN_COHORTS.json', dict(cohorts=collection['cohorts'], all_three_frozen=False,
                source_condition='plain', candidate_forwards=0, automatic_retry=False))
        gate.require(0 <= cost['completed_member_forwards'] <= cost['attempted_member_forwards'] <= MAX_FORWARDS,
                     'Maximum60 attempted member calls including retained failures')
        all_collected = all(row['collection_status'] == 'complete' for row in records)
        if all_collected:
            gate.require(cost['attempted_member_forwards'] == cost['completed_member_forwards'] == MAX_FORWARDS,
                         'Exactly four member inference calls per selected state')
        collection['status'] = 'complete' if all_collected else 'complete_with_retained_collection_failures'; persist()
        setup = time.monotonic(); readout.run(gate, pins, output, collection)
        cost.update(analysis_seconds=time.monotonic() - setup, status=collection['status']); persist()
    except Exception as error:
        collection['status'] = 'retained_pipeline_failure'
        collection['failure'] = dict(error_type=type(error).__name__, error=str(error), automatic_retry=False)
        cost['status'] = collection['status']; write(output / 'compact' / 'FAILURE.json', collection['failure']); persist()
        raise
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        cost.update(inclusive_wall_seconds=time.monotonic() - started, CPU_user_seconds=usage.ru_utime - usage_start.ru_utime,
            CPU_system_seconds=usage.ru_stime - usage_start.ru_stime,
            peak_RSS_bytes=int(usage.ru_maxrss * (1024 if sys.platform.startswith('linux') else 1)),
            raw_server_only_storage_bytes=sum(path.stat().st_size for path in (output / 'raw').iterdir() if path.is_file()),
            per_cell_costs=[{key: value for key, value in row.items() if key in ('cell', 'seed', 'condition', 'collection_status', 'custody_partition')
                or key.endswith('_seconds') or key.startswith('peak_') or key.endswith('member_forwards')} for row in records],
            finished_UTC=datetime.now(timezone.utc).isoformat())
        write(output / 'compact' / 'COST.json', cost)
    print(str(output / 'compact'))


if __name__ == '__main__':
    main()
