"""Disabled root-owned Context9 entry; reuse the sealed collection/readout APIs."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import shutil
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SEEDS = (8101, 8203, 8307)
CONDITIONS = ('shared_common', 'shared_route', 'shared_route_permuted')
MAX_FORWARDS = 36


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    def reject(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def inside(relative):
    value = Path(relative)
    require(not value.is_absolute() and '..' not in value.parts, 'Phase-relative custody required')
    path = (PHASE / value).resolve()
    require(path != PHASE and path.is_relative_to(PHASE), 'Path leaves authorized phase')
    return path


def binding(path):
    path = Path(path).resolve()
    require(path.is_relative_to(PHASE) and path.is_file(), 'Phase file custody')
    return dict(path=str(path.relative_to(PHASE)), sha256=sha(path), bytes=path.stat().st_size)


def bound(row):
    path = inside(row['path'])
    require(path.is_file() and sha(path) == row['sha256'], 'Bound file changed: ' + row['path'])
    require('bytes' not in row or path.stat().st_size == row['bytes'], 'Bound file size changed')
    return path


def verify_manifest(row):
    path = bound(row)
    for item in read(path)['files']:
        file = (path.parent / item['path']).resolve()
        require(file.is_relative_to(path.parent), 'Manifest file leaves its source directory')
        bound(dict(item, path=str(file.relative_to(PHASE))))
    return path.parent


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def identity(value):
    require(isinstance(value, dict) and type(value.get('pid')) is int and value['pid'] > 0
            and type(value.get('start_ticks')) is int and value['start_ticks'] > 0
            and isinstance(value.get('state'), str) and len(value['state']) == 1,
            'Original pid/start_ticks/state identity required')
    return {key: value[key] for key in ('pid', 'start_ticks', 'state')}


def consume(release_path, release_sha256):
    """Stdlib only: all nine endpoints and exact saved terminal custody first."""
    require(sha(release_path) == release_sha256, 'Exact separate root collection release')
    cfg = read(release_path)
    require(cfg.get('schema') == 'context9-root-whole-family-collection-release-v1', 'Exact release schema')
    for key in ('enabled', 'root_execution_authorized', 'source_review_approved', 'entire9_complete',
                'owner_and_children_terminal', 'trusted_checkpoint_deserialization_authorized',
                'runtime_resource_readiness_confirmed', 'collection_and_analysis_cost_charged',
                'external_owned_bound_confirmed'):
        require(cfg.get(key) is True, 'Disabled pending root release: ' + key)
    require(all(cfg.get(key) is False for key in ('TEST_access', 'training', 'reselection',
                'calibration', 'automatic_retry', 'stage2_admitted'))
            and type(cfg.get('maximum_member_forwards')) is int
            and cfg['maximum_member_forwards'] == MAX_FORWARDS, 'Fixed prediction-only scope')
    verify_manifest(dict(path=str((HERE / 'MANIFEST.json').relative_to(PHASE)),
                         sha256=cfg['collection_manifest_sha256']))
    pins = read(HERE / 'SOURCE_BINDINGS.json')
    require(pins['seeds'] == list(SEEDS) and pins['conditions'] == list(CONDITIONS)
            and pins['maximum_member_forwards'] == MAX_FORWARDS
            and pins['owner_identity'] == dict(pid=526200, start_ticks=6021896966), 'Original Context9 identity')
    sources = pins['source']
    for row in sources.values():
        bound(row)
    for key in ('integration_manifest', 'public_manifest', 'closed9_manifest', 'legacy_manifest'):
        verify_manifest(sources[key])
    verify_manifest(pins['audited_Wiki24_analysis_manifest'])
    bound(pins['audited_Wiki24_analysis'])
    integration = bound(sources['integration_manifest']).parent
    for row in read(integration / 'SOURCE_BINDINGS.json')['files']:
        bound(row)
    protocol = read(bound(sources['protocol']))
    schedule = read(bound(sources['schedule']))
    releases = pins['releases']
    roster = [row['cell_id'] for row in releases]
    require(len(roster) == len(set(roster)) == 9 and schedule['cells'] == [
        {key: row[key] for key in ('cell_id', 'method', 'seed', 'release_sha256')} for row in releases]
        and protocol['stage1_roster'] == [{key: row[key] for key in ('cell_id', 'method', 'seed')} for row in releases]
        and schedule['automatic_retry'] is False and schedule['TEST_access'] is False
        and schedule['stage2_admitted'] is False, 'Exact original prospective roster')
    activation = inside(pins['activation_directory'])
    family_path, parent_path = bound(cfg['family_closure']), bound(cfg['parent_owner'])
    require(family_path == activation / 'FAMILY_CLOSURE.json'
            and parent_path == activation / 'PARENT_OWNER.json'
            and not (activation / 'OWNER_FAILURE.json').exists(), 'Original successful owner custody')
    family, parent = read(family_path), read(parent_path)
    require(identity(parent) == parent and {key: parent[key] for key in ('pid', 'start_ticks')} == pins['owner_identity'],
            'Exact original parent 526200/6021896966')
    require(family.get('complete') is True and family.get('fixed_total') == 9
            and family.get('original_cells_only') is True and family.get('automatic_retry') is False
            and family.get('scores_read') is False and family.get('stage2_admitted') is False
            and [row['cell_id'] for row in family['completed']] == roster, 'Whole original9 successful closure')
    terminal = read(bound(cfg['terminal_evidence']))
    require(terminal.get('schema') == 'context9-root-saved-terminal-evidence-v1'
            and terminal.get('complete') is True and terminal.get('root_observed') is True
            and terminal.get('owner_and_children_terminal') is True
            and terminal.get('family_closure') == cfg['family_closure']
            and terminal.get('parent_owner') == cfg['parent_owner']
            and terminal.get('parent_identity') == parent
            and terminal.get('parent_absent') is True and terminal.get('parent_no_CUDA_rows') is True
            and terminal.get('scores_read') is False and terminal.get('TEST_access') is False,
            'Later root terminal snapshot for this exact original parent/closure required')
    observed, closed_at = datetime.fromisoformat(terminal['observed_UTC']), datetime.fromisoformat(family['UTC'])
    require(observed.utcoffset() is not None and closed_at.utcoffset() is not None and observed >= closed_at,
            'Root terminal observation must follow original whole9 closure')
    children = terminal['children']
    by_cell = {row['cell_id']: row for row in children}
    require(len(children) == len(by_cell) == 9 and set(by_cell) == set(roster), 'Whole9 terminal child roster')
    public_root = bound(sources['public_manifest']).parent
    core = {name: sha(public_root / 'core' / name) for name in ('factors.py', 'models.py', 'objectives.py', 'selection.py')}
    counts = dict(shadow_member_forwards=8800, replay_member_forwards=8800,
        output_cotangent_collections=1100, member_reverse_collections=8800,
        optimizer_bank_updates=1100, exact_member_RNG_endpoint_checks=1100)
    records = []
    for original, closed in zip(releases, family['completed']):
        cell, condition, seed = original['cell_id'], original['method'], original['seed']
        release = bound(original['binding']); cell_cfg = read(release)
        require(original['binding']['sha256'] == original['release_sha256']
                and cell == str(seed) + '_' + condition and seed in SEEDS and condition in CONDITIONS
                and cell_cfg['schema'] == 'context-target-cell-release-v1'
                and all(cell_cfg.get(key) is True for key in ('enabled', 'root_adopted', 'source_review_approved',
                    'fullgraph_targets_objective_and_selectors_qualified', 'external_supervision_confirmed',
                    'complete_staged_protocol_frozen'))
                and cell_cfg['method'] == condition and cell_cfg['seed'] == seed
                and cell_cfg['source_manifest_sha256'] == sources['integration_manifest']['sha256']
                and cell_cfg['inputs'] == pins['inputs'] and cell_cfg['TEST_access'] is False
                and cell_cfg['automatic_retry'] is False and cell_cfg['whole_stage1_gate_passed'] is False
                and cell_cfg['physical_gpu_uuid'] == pins['physical_gpu_uuid']
                and cell_cfg['frozen_protocol'] == {key: sources['protocol'][key] for key in ('path', 'sha256')},
                'Exact original admitted method/seed/source/roles/protocol')
        owner_path, exit_path = activation / 'logs' / (cell + '.OWNER.json'), activation / 'logs' / (cell + '.EXIT.json')
        owner, receipt = read(owner_path), read(exit_path)
        child = by_cell[cell]
        argv = [pins['runtime']['python'], '-B', str(bound(sources['scientific_wrapper'])), str(release)]
        require(identity(owner) == receipt['child'] and owner['argv'] == argv
                and owner['fresh_free_GPU_bytes'] >= cell_cfg['minimum_fresh_free_GPU_bytes']
                and receipt == closed['exit_receipt'] and receipt['cell_id'] == cell
                and receipt['release_sha256'] == original['release_sha256']
                and receipt['exit_code'] == 0 and receipt['exit_authority'] == 'subprocess.Popen.wait'
                and receipt['child_reaped'] is True and receipt['timeout'] is False
                and receipt['signals'] == [] and receipt['scores_read'] is False,
                'Original one-attempt successful wait/reap receipt')
        require(child['identity'] == receipt['child'] and child['owner_receipt'] == binding(owner_path)
                and child['exit_receipt'] == binding(exit_path) and child['child_absent'] is True
                and child['child_no_CUDA_rows'] is True and child['child_reaped'] is True
                and child['exit_authority'] == 'subprocess.Popen.wait', 'Root exact child terminal custody')
        output = inside(cell_cfg['output'])
        require(output == inside(pins['original_output_directory']) / cell
                and not (output / 'FAILURE.json').exists(), 'Original successful endpoint directory')
        complete_path, run_path = output / 'COMPLETE.json', output / 'RUN.json'
        require(sha(complete_path) == closed['completion_sha256'], 'Original COMPLETE custody')
        endpoint, run = read(complete_path), read(run_path)
        policy = dict(method=condition, underlying_arm='be_unit_contrastive', own_selected_four=False, objective_separable=True)
        objective = dict(method=condition, underlying_session_arm='be_unit_contrastive',
            mode='common' if condition == 'shared_common' else 'route',
            permuted=condition == 'shared_route_permuted', own_selected_four=False,
            fixed_target_objective_separable=True, no_inter_member_repulsion=True,
            original_stochastic_views=2, TEST_access=False, exploratory=True)
        require(endpoint['complete'] is True and endpoint['epochs'] == endpoint['steps'] == 1100
                and endpoint['alignment_source_calls'] == 1100 and endpoint['residual_source_calls'] == 0
                and endpoint['execution_accounting'] == counts
                and endpoint['checkpoint_policy'] == run['checkpoint_policy'] == policy
                and all(value['task'] == 'wikics' and value['arm'] == condition and value['seed'] == seed
                        and value['TEST_scoring'] is False for value in (endpoint, run))
                and run['underlying_session_arm'] == 'be_unit_contrastive' and run['device'] == 'cuda:0'
                and run['source_release_sha256'] == original['release_sha256']
                and run['driver_sha256'] == sources['context_driver']['sha256']
                and run['execution_mode'] == pins['original_execution_mode']
                and run['explicit_context_objective'] == objective
                and run['core'] == core and run['native']['polynormer_model_sha256'] == pins['inputs']['polynormer']['sha256']
                and run['data']['train_npz_sha256'] == pins['inputs']['train']['sha256']
                and run['data']['valid_npz_sha256'] == pins['inputs']['development']['sha256']
                and run['torch'] == pins['runtime']['versions']['torch']
                and run['numpy'] == pins['runtime']['versions']['numpy'], 'All original1100 execution identities')
        targets = run['verified_frozen_targets']
        require(targets['target_archive_sha256'] == pins['inputs']['target_archive']['sha256']
                and targets['target_preflight_receipt_sha256'] == pins['inputs']['target_preflight']['sha256']
                and targets['role_manifest_sha256'] == pins['inputs']['role_manifest']['sha256']
                and targets['train_npz_sha256'] == pins['inputs']['train']['sha256']
                and targets['original_panel_sha256'] == pins['original_panel_sha256']
                and targets['mask_source_sha256'] == pins['original_mask_source_sha256']
                and targets['permutation_seed'] == 991327
                and all(math.isfinite(targets[key]) and targets[key] >= .05
                        for key in ('route_common_mean_target_TV', 'route_permuted_mean_target_TV'))
                and targets['target_relations_regenerated'] is False
                and targets['exact_scored_positive_counts_preserved'] is True
                and targets['class_and_self_positives_preserved'] is True, 'Original frozen target custody')
        records.append(dict(cell=cell, seed=seed, condition=condition, method_identity=condition,
            release=binding(release), complete=binding(complete_path), run=binding(run_path),
            exit_receipt=binding(exit_path), family_status='complete', members=4,
            collection_status='pending', selected_checkpoint_path=output / 'selected.pt',
            selected_sha256=endpoint['selected_sha256']))
    require({(row['seed'], row['condition']) for row in records} == {(s, c) for s in SEEDS for c in CONDITIONS},
            'Entire fixed9 complete before numerical opening')
    # Hash byte streams only, after all endpoint/terminal metadata passed. No torch.load or NPZ load here.
    for row in records:
        row['selected_checkpoint'] = binding(row.pop('selected_checkpoint_path'))
        require(row['selected_checkpoint']['sha256'] == row.pop('selected_sha256'), 'Original selected hash; no reselection')
    for row in pins['inputs'].values():
        bound(row)
    output = inside(cfg['output'])
    frozen_directories = {pins['activation_directory'], pins['original_output_directory'], HERE.name}
    frozen_directories.update(Path(row['path']).parts[0] for row in (
        list(sources.values()) + list(pins['inputs'].values()) + [pins['audited_Wiki24_analysis']]))
    require(not output.exists() and output.parent.is_dir()
            and not any(output.is_relative_to(inside(directory)) for directory in frozen_directories),
            'Fresh server-only collection directory outside source/original families')
    for key in ('owned_GPU_memory_cap_bytes', 'minimum_fresh_free_GPU_bytes', 'minimum_fresh_disk_free_bytes',
                'external_active_seconds', 'external_cleanup_seconds', 'external_hard_seconds'):
        require(type(cfg.get(key)) is int and cfg[key] > 0, 'Finite positive root resource bound: ' + key)
    require(cfg['owned_GPU_memory_cap_bytes'] <= 20 * 1024**3
            and cfg['minimum_fresh_free_GPU_bytes'] >= max(24 * 1024**3, cfg['owned_GPU_memory_cap_bytes'] + 4 * 1024**3)
            and cfg['external_hard_seconds'] == cfg['external_active_seconds'] + cfg['external_cleanup_seconds'],
            'Root owned cap/headroom and external active-plus-cleanup envelope')
    return cfg, pins, records, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    os.umask(0o077)
    started, usage_start = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF)
    cfg, pins, records, output = consume(args.release, args.release_sha256)
    runtime, sources = pins['runtime'], pins['source']
    require(socket.gethostname() == runtime['hostname'] and Path.cwd().resolve() == Path(runtime['repository']).resolve()
            and str(Path(sys.executable).absolute()) == runtime['python']
            and os.environ.get('PYTHONPATH') == runtime['PYTHONPATH']
            and cfg['physical_gpu_uuid'] == pins['physical_gpu_uuid']
            and os.environ.get('CUDA_VISIBLE_DEVICES') == pins['physical_gpu_uuid'], 'Original allocation runtime/GPU')
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).splitlines()
    free = int(subprocess.check_output(['nvidia-smi', '--id=' + pins['physical_gpu_uuid'], '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip()) * 1024**2
    require(inventory == [pins['physical_gpu_uuid']] and free >= cfg['minimum_fresh_free_GPU_bytes']
            and shutil.disk_usage(output.parent).free >= cfg['minimum_fresh_disk_free_bytes'], 'Fresh runtime resource admission')
    output.mkdir(mode=0o700)
    (output / 'raw').mkdir(mode=0o700)
    (output / 'compact').mkdir(mode=0o700)
    collection = dict(schema='context9-root-selected-predictions-v1', cells=records,
        cohorts={str(seed): dict(available=False) for seed in SEEDS}, status='running',
        whole9_complete_before_opening=True, owner_and_children_terminal_before_opening=True,
        frozen_target_archive_sha256=pins['inputs']['target_archive']['sha256'],
        TEST_access=False, training=False, reselection=False, calibration=False, automatic_retry=False, stage2_admitted=False)
    cost = dict(schema='context9-root-collection-and-readout-cost-v1', status='running',
        started_UTC=datetime.now(timezone.utc).isoformat(), release=binding(args.release),
        metadata_gate_seconds=time.monotonic() - started, maximum_member_forwards=MAX_FORWARDS,
        attempted_member_forwards=0, completed_member_forwards=0, checkpoint_deserialization_attempts=0,
        checkpoint_bytes_submitted_to_deserializer=0, peak_CUDA_allocated_bytes=0, peak_CUDA_reserved_bytes=0,
        fullgraph_nodes_per_call=11701, development_objects_per_call=5274, TRAIN_updates=0, backward_calls=0,
        optimizer_constructions=0, raw_logits_representations_and_labels_server_only=True, TEST_access=False,
        fresh_free_GPU_bytes=free, resource_envelope={key: cfg[key] for key in (
            'owned_GPU_memory_cap_bytes', 'minimum_fresh_free_GPU_bytes', 'minimum_fresh_disk_free_bytes',
            'external_active_seconds', 'external_cleanup_seconds', 'external_hard_seconds')})
    write(output / 'compact' / 'GATE.json', dict(passed=True, source_manifest_sha256=cfg['collection_manifest_sha256'],
        family_closure=cfg['family_closure'], parent_owner=cfg['parent_owner'], terminal_evidence=cfg['terminal_evidence'],
        owner_identity=pins['owner_identity'], all9_original1100_endpoints_and_selected_hashes_validated=True,
        whole9_complete_before_opening=True, owner_and_children_terminal_before_opening=True, TEST_access=False))
    try:
        setup = time.monotonic()
        # First numerical imports: the entire original family and terminal snapshot already passed.
        import numpy as np
        import torch
        torch.set_num_threads(2)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(cfg['owned_GPU_memory_cap_bytes'] / torch.cuda.get_device_properties(0).total_memory, 0)
        versions = dict(torch=str(torch.__version__), numpy=np.__version__, **{
            name: importlib.metadata.version(name) for name in ('torch-geometric', 'torch-scatter', 'torch-sparse', 'ogb')})
        require(versions == runtime['versions'], 'Exact original provider versions')
        # Fresh modules change only callback/roster constants; original source files stay sealed.
        legacy_gate = module(bound(sources['legacy_gate']), 'gate')
        legacy_gate.SEEDS, legacy_gate.CONDITIONS, legacy_gate.MAX_FORWARDS = SEEDS, CONDITIONS, MAX_FORWARDS
        collector = module(bound(sources['legacy_collect']), '_context9_original_collector')
        hooks = module(bound(sources['collection_hooks']), '_context9_original_restore_hooks')
        collector.restore, collector.MAX_FORWARDS = hooks.restore, MAX_FORWARDS
        readout = module(bound(sources['closed9_readout']), '_context9_original_readout')
        public = module(bound(sources['public_adapter']), 'portable')
        data = module(bound(sources['public_data']), '_context9_original_data')
        dispatch = module(bound(sources['context_dispatch']), '_context9_original_dispatch')
        adapter = hooks.Adapter(dispatch)
        analysis, contract = readout.configure(pins)
        source_pins = dict(pins['inputs'], context_driver_sha256=sources['context_driver']['sha256'])
        native_sources, native_origin = public.native_sources('wikics', bound(source_pins['polynormer']))
        cost.update(source_and_runtime_loading_seconds=time.monotonic() - setup, runtime_versions=versions, native=native_origin)
        setup = time.monotonic()
        train, valid, origin = data.load_train_valid('wikics', bound(source_pins['train']), bound(source_pins['development']))
        ordered = dict(x=train['x'], edge_index=train['edge_index'], train_ids=train['ids'], train_y=train['y'],
                       valid_ids=valid['ids'], valid_y=valid['y'])
        for name, tensor in ordered.items():
            expected = pins['ordered_raw_tensor_fingerprints'][name]
            require(str(tensor.dtype) == expected['dtype'] and list(tensor.shape) == expected['shape']
                    and hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest() == expected['contiguous_raw_bytes_sha256'],
                    'Original ordered raw role fingerprint: ' + name)
        iterator = data.validation_batches('wikics', train, valid, 'cuda:0')
        batch, truth = next(iterator)
        require(next(iterator, None) is None and len(truth) == 5274 and tuple(batch['x'].shape) == (11701, 300)
                and tuple(batch['edge_index'].shape) == (2, 442907), 'Complete original fullgraph development batch')
        torch.cuda.synchronize()
        cost.update(data_loading_and_transfer_seconds=time.monotonic() - setup, data=origin)
        def persist():
            write(output / 'compact' / 'COLLECTION.json', collection)
            write(output / 'compact' / 'COST.json', cost)
        def collect(condition):
            for seed in SEEDS:
                record = next(row for row in records if (row['seed'], row['condition']) == (seed, condition))
                collector.collect_cell(torch, np, public, adapter, analysis, contract, native_sources,
                    batch, truth, record, source_pins, output, cost)
                persist()
        collect('shared_common')
        common = [row for row in records if row['condition'] == 'shared_common']
        common_ready = all(row['collection_status'] == 'complete' for row in common)
        if common_ready:
            try:
                for record in common:
                    freeze_started = time.monotonic()
                    arrays = contract.load(np, output / 'raw' / (record['cell'] + '.npz'), record['raw_archive'])
                    analysis.validate(np, arrays, contract)
                    masks = contract.baseline_cohorts(np, arrays)
                    path = output / 'raw' / ('plain_cohorts_' + str(record['seed']) + '.npz')
                    require(not path.exists(), 'COMMON cohorts freeze exactly once')
                    np.savez(path, **masks)
                    collection['cohorts'][str(record['seed'])] = dict(available=True,
                        common_cell=record['cell'], common_archive=record['raw_archive'], archive=binding(path),
                        counts={name: int(masks[name].sum()) for name in contract.COHORTS},
                        source_condition='shared_common', compatibility_filename_only='plain_cohorts_<seed>.npz',
                        frozen_before_candidate_prediction_inspection=True)
                    record['cohort_freeze_seconds'] = time.monotonic() - freeze_started
                    del arrays, masks
                # Durable whole3 cohort seal precedes every ROUTE/PERMUTED inference call.
                write(output / 'compact' / 'COMMON_COHORTS.json', dict(cohorts=collection['cohorts'],
                    all_three_common_collections_completed_first=True, all_three_frozen=True,
                    source_condition='shared_common', frozen_before_candidate_prediction_inspection=True))
                persist()
            except Exception as error:
                common_ready = False
                collection['cohort_freeze_failure'] = dict(error_type=type(error).__name__, error=str(error), automatic_retry=False)
        if common_ready:
            collect('shared_route')
            collect('shared_route_permuted')
        else:
            for cohort in collection['cohorts'].values():
                cohort.update(available=False, failure='Entire COMMON3 collection and cohort freeze required')
            for record in records:
                if record['condition'] != 'shared_common':
                    record.update(collection_status='skipped_before_forward', failure=dict(
                        reason='Entire COMMON3 collection and cohort freeze required', automatic_retry=False))
            write(output / 'compact' / 'COMMON_COHORTS.json', dict(cohorts=collection['cohorts'],
                all_three_frozen=False, source_condition='shared_common', candidate_forwards=0,
                failure=collection.get('cohort_freeze_failure'), automatic_retry=False))
        require(0 <= cost['completed_member_forwards'] <= cost['attempted_member_forwards'] <= MAX_FORWARDS,
                'Maximum36 attempted member inference calls, including failures')
        all_collected = all(row['collection_status'] == 'complete' for row in records)
        if all_collected:
            require(cost['attempted_member_forwards'] == cost['completed_member_forwards'] == MAX_FORWARDS,
                    'Whole9 collection is exactly four original member calls per cell')
        collection['status'] = 'complete' if all_collected else 'complete_with_retained_collection_failures'
        persist()
        setup = time.monotonic()
        fixed_gate = readout.run(output, collection, pins)
        cost.update(analysis_seconds=time.monotonic() - setup, status=collection['status'])
        collection['fixed_stage1_gate'] = fixed_gate
        persist()
    except Exception as error:
        collection['status'] = 'retained_pipeline_failure'
        collection['failure'] = dict(error_type=type(error).__name__, error=str(error), automatic_retry=False)
        write(output / 'compact' / 'FAILURE.json', collection['failure'])
        write(output / 'compact' / 'COLLECTION.json', collection)
        cost['status'] = collection['status']
        raise
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        cost.update(inclusive_wall_seconds=time.monotonic() - started,
            CPU_user_seconds=usage.ru_utime - usage_start.ru_utime, CPU_system_seconds=usage.ru_stime - usage_start.ru_stime,
            peak_RSS_bytes=int(usage.ru_maxrss * (1024 if sys.platform.startswith('linux') else 1)),
            raw_server_only_storage_bytes=sum(path.stat().st_size for path in (output / 'raw').iterdir() if path.is_file()),
            per_cell_costs=[{key: value for key, value in row.items() if key in ('cell', 'seed', 'condition', 'collection_status')
                or key.endswith('_seconds') or key.startswith('peak_') or key.endswith('member_forwards')} for row in records],
            finished_UTC=datetime.now(timezone.utc).isoformat())
        write(output / 'compact' / 'COST.json', cost)
    print(str(output / 'compact'))


if __name__ == '__main__':
    main()
