"""Stdlib whole36/source/custody gate. No metrics, tensors or remote calls."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import socket
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SEEDS = (6101, 6203, 6307)
OPERATORS = ('native_tied', 'active_reversible_exp', 'pre_sigmoid_split', 'full_qk')
KINDS = ('single', 'be_init', 'independent4')
MAX_FORWARDS = 108
COMMIT = 'a128c164bc33d7843a13c290b22ef1c311f4184a'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def helpers(pins):
    row = pins['reuse']['metadata_helpers']; path = PHASE / row['path']
    require(sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Exact stdlib helpers')
    spec = importlib.util.spec_from_file_location('gate', path)
    gate = importlib.util.module_from_spec(spec); sys.modules['gate'] = gate; spec.loader.exec_module(gate)
    return gate


def fresh_custody(g, pins, cfg, receipts):
    evidence = read(g.bound(cfg['fresh_all_route_custody']))
    require(evidence['schema'] == 'qk36-root-postunion-actual-custody-v1' and evidence['complete'] is True
        and evidence['root_observed'] is True and evidence['checked_after_all36_union'] is True
        and evidence['plan'] == pins['global_plan'] and evidence['route_absence_receipts'] == cfg['route_absence_receipts']
        and evidence['TEST_access'] is False and set(evidence['routes']) == set(receipts), 'Fresh root custody after complete36 union')
    fields = ('pid', 'start_ticks', 'group', 'session', 'boot_id')
    for route_id, receipt in receipts.items():
        observation = evidence['routes'][route_id]
        identities, groups = receipt['owned_identities'], receipt['owned_groups']
        require(observation['route_id'] == route_id and observation['physical_GPU_uuid'] == pins['plan']['GPU_assignment'][route_id]
            and observation['checked_after_all36_union'] is True and observation['actual_wait_and_reap_verified'] is True
            and observation['custody_program'] == pins['reuse']['custody_program']
            and [r['saved'] for r in observation['identity_observations']] == identities
            and [r['saved'] for r in observation['group_observations']] == groups, 'Actual original parent/children/groups on every route')
        for row in observation['identity_observations']:
            require(set(fields).issubset(row['saved']), 'Recorded boot-bound training process identity')
            current = row['current']
            require(current is None or all(k in current for k in fields)
                and any(current[k] != row['saved'][k] for k in fields), 'No original PID/birth/group/session/boot remains')
        require(all(r['current_members'] == [] for r in observation['group_observations'])
            and not ({r['pid'] for r in identities} & set(observation['current_CUDA_pids'])), 'Original groups and owned CUDA rows absent')
    return evidence


def source_lane(g, pins):
    common = g.module(g.bound(pins['reuse']['lane_common']), '_qk36_original_lane_common')
    prior = sys.modules.get('common')
    try:
        sys.modules['common'] = common
        lane = g.module(g.bound(pins['reuse']['lane_program']), '_qk36_original_work_receipt')
    finally:
        if prior is None:
            sys.modules.pop('common', None)
        else:
            sys.modules['common'] = prior
    return lane


def original_custody(g, pins, cfg, receipts):
    """All36 metadata proof; only this route's12 checkpoint bytes are read here."""
    row = read(g.bound(cfg['original_route_state_custody']))
    require(row['schema'] == 'qk36-original-resident-state-custody-v1'
        and row['complete'] is True and row['root_verified_after_all36_union'] is True
        and row['plan'] == pins['global_plan'] and row['route_absence_receipts'] == cfg['route_absence_receipts']
        and row['original_files_unchanged'] is True and row['checkpoint_copies_required'] is False
        and row['all36_original_endpoint_file_bytes_verified'] is True
        and set(row['routes']) == set(receipts), 'Complete36 original resident byte/cost custody, no central checkpoints')
    for route_id, receipt in receipts.items():
        route = row['routes'][route_id]
        require(route['cells'] == receipt['cells'] and route['physical_GPU_uuid'] == pins['routes'][route_id]['GPU_uuid']
            and route['original_phase'] == pins['routes'][route_id]['phase']
            and route['qualification_adoption'] == pins['plan']['qualification_adoptions'][route_id], 'All36 original endpoint hashes and actual route qualification retained')
        g.bound(route['original_training_and_qualification_cost_metadata'])
    g.bound(row['prior_central_checkpoint_transfer_attempt_cost_and_failure'])
    return row


def process_identity(value):
    fields = ('pid', 'start_ticks', 'group', 'session', 'boot_id')
    require(type(value) is dict and set(fields).issubset(value)
        and all(type(value[key]) is int and value[key] > 0 for key in fields[:-1])
        and type(value['boot_id']) is str and bool(value['boot_id']), 'Concrete boot-bound witnessed process identity')
    return value


def finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def same_process(left, right):
    process_identity(left); process_identity(right)
    return all(left[key] == right[key] for key in ('pid', 'start_ticks', 'group', 'session', 'boot_id'))


def phase_terminal(g, pins, row, route_id, phase, collection_binding, cost_binding, collection, cost, manifest_sha256):
    """Exact existing source/release/process/EXIT/cost custody; no new owner."""
    terminal = read(g.bound(row)); source_program = g.binding(HERE / 'collect_route.py')
    require(terminal['schema'] == 'qk36-original-route-reader-terminal-v2'
        and terminal['source_program'] == source_program and terminal['route_id'] == route_id
        and terminal['collection_phase'] == phase and terminal['collection'] == collection_binding
        and terminal['cost'] == cost_binding and terminal['actual_wait_and_reap'] is True
        and terminal['owned_group_absent'] is True and terminal['no_owned_CUDA'] is True,
        'Concrete exact phase collector/source/cost and direct child terminal custody')
    release_path = g.bound(terminal['release']); release = read(release_path)
    require(cost['release'] == terminal['release']
        and release['schema'] == 'qk36-original-route-two-phase-release-v1'
        and all(release[key] is True for key in ('enabled', 'root_execution_authorized', 'source_review_approved',
            'whole36_complete', 'actual_all_route_custody_closed', 'trusted_selected_state_deserialization_authorized',
            'original_local12_resident_states_confirmed', 'qualified_readout_route_confirmed',
            'finite_external_owner_confirmed', 'inclusive_cost_accounting_confirmed'))
        and release['readout_route_id'] == route_id and release['collection_phase'] == phase
        and release['readout_manifest_sha256'] == manifest_sha256 and release['execution_source_commit'] == COMMIT
        and release['maximum_member_forwards'] == (9 if phase == 'native_baselines' else 27)
        and release['maximum_route_member_forwards'] == 36
        and all(release[key] is False for key in ('training', 'TEST_access', 'automatic_retry', 'reselection', 'calibration'))
        and collection_binding['path'] == release['output_directory'] + '/compact/COLLECTION.json'
        and cost_binding['path'] == release['output_directory'] + '/compact/COST.json',
        'Enabled original root phase release from final COST, exact output/source/scope')
    parent, child = process_identity(terminal['parent_identity']), process_identity(terminal['child_identity'])
    require(child['pid'] == child['group'] == child['session'] and child['boot_id'] == parent['boot_id'],
        'Actual isolated collector child and its same-boot parent')
    witness = read(g.bound(terminal['actual_process_witness']))
    route = pins['routes'][route_id]
    expected_argv = [route['python'], '-B', str(Path(route['phase']) / source_program['path']),
        '--release', str(Path(route['phase']) / terminal['release']['path']), '--release-sha256', terminal['release']['sha256']]
    require(witness['schema'] == 'qk36-phase-collector-process-witness-v1' and witness['root_observed'] is True
        and witness['custody_program'] == pins['reuse']['custody_program']
        and witness['source_program'] == source_program and witness['release'] == terminal['release']
        and witness['route_id'] == route_id and witness['collection_phase'] == phase
        and same_process(witness['parent_identity'], parent) and same_process(witness['child_identity'], child)
        and witness['actual_child_argv'] == expected_argv, 'Witnessed exact source/release argv and parent/child relationship')
    actual = read(g.bound(terminal['actual_child_exit_receipt']))
    require(same_process(actual['child'], child) and actual['reaped'] is True
        and type(actual['exit_code']) is int and actual['exit_code'] == terminal['actual_exit_code']
        and type(terminal['actual_exit_code']) is int and actual['partial_work_and_costs_retained'] is True
        and actual['automatic_retry'] is False and actual['scores_read'] is False
        and any(same_process(child, item) for item in actual['witnessed_owned_members']), 'Actual direct waited/reaped child EXIT, including nonzero failures')
    known = [parent, *actual['witnessed_owned_members']]
    for identity in known:
        process_identity(identity)
    absence = read(g.bound(terminal['actual_owned_absence_receipt']))
    require(absence['schema'] == 'qk36-phase-collector-owned-absence-v1' and absence['root_observed'] is True
        and absence['source_program'] == source_program and absence['release'] == terminal['release']
        and absence['route_id'] == route_id and absence['collection_phase'] == phase
        and absence['custody_program'] == pins['reuse']['custody_program']
        and same_process(absence['parent_identity'], parent) and same_process(absence['child_identity'], child)
        and len(absence['identity_observations']) == len(known)
        and all(same_process(item['saved'], identity) for item, identity in zip(absence['identity_observations'], known))
        and len(absence['group_observations']) == 2
        and all(same_process(item['saved'], identity) for item, identity in zip(absence['group_observations'], (parent, child))),
        'Actual group/CUDA/identity observations belong to this exact phase owner and witnessed child tree')
    fields = ('pid', 'start_ticks', 'group', 'session', 'boot_id')
    require(all(item['current'] is None or all(key in item['current'] for key in fields)
        and any(item['current'][key] != item['saved'][key] for key in fields) for item in absence['identity_observations'])
        and all(item['current_members'] == [] for item in absence['group_observations'])
        and not ({identity['pid'] for identity in known} & set(absence['current_CUDA_pids'])),
        'Actual original phase parent/child groups and owned CUDA rows are absent')
    # Existing EXIT already contains measured combined active/cleanup cost,
    # signals/reason and retained failure fields. Bind it rather than inventing
    # a separated cleanup duration or implementing another owner.
    require(terminal['cleanup_cost_and_failures'] == terminal['actual_child_exit_receipt']
        and finite_nonnegative(actual['active_and_cleanup_seconds'])
        and finite_nonnegative(actual['resource_admission_seconds'])
        and finite_nonnegative(actual['max_sampled_owned_GPU_bytes'])
        and finite_nonnegative(actual['max_sampled_owned_RSS_bytes'])
        and type(actual['signals']) is list
        and terminal['separated_cleanup_seconds'] is None and terminal['separated_cleanup_duration_unavailable'] is True,
        'Concrete measured original EXIT cleanup/cost/failure receipt; separated cost honestly unavailable')
    parent_wait = terminal['parent_direct_wait_and_reap_observed']
    require(type(parent_wait) is bool, 'Actual parent OS-exit evidence kind')
    if parent_wait:
        parent_exit = read(g.bound(terminal['actual_parent_exit_receipt']))
        require(same_process(parent_exit['child'], parent) and parent_exit['reaped'] is True
            and type(parent_exit['exit_code']) is int and parent_exit['exit_code'] == terminal['parent_actual_exit_code'],
            'Actually observed direct parent exit/reap, never inferred from child completion')
    else:
        require(terminal['parent_actual_exit_code'] is None and terminal['actual_parent_exit_receipt'] is None,
            'Detached parent OS exit/direct wait remain unknown')
    supervision = read(g.bound(release['external_supervision']))
    limits = pins['plan']['root_resource_limits'][route_id]
    require(supervision['entry_program'] == source_program and supervision['enabled'] is True
        and supervision['finite_owned_bound_confirmed'] is True
        and supervision['custody_program'] == pins['reuse']['custody_program']
        and supervision['argv_prefix'] == expected_argv[:3]
        and supervision['release_argument_path'] == str(Path(route['phase']) / terminal['release']['path'])
        and supervision['route_id'] == route_id and supervision['physical_GPU_uuid'] == route['GPU_uuid']
        and 0 < supervision['active_seconds'] <= 7200 and 0 < supervision['cleanup_seconds'] <= 15
        and supervision['hard_seconds'] == supervision['active_seconds'] + supervision['cleanup_seconds']
        and 0 < supervision['owned_GPU_bytes'] <= limits['max_owned_GPU_bytes']
        and 0 < supervision['owned_RSS_bytes'] <= limits['max_owned_RSS_bytes'],
        'Original admitted phase supervision, no larger GPU envelope')
    successful = (actual['exit_code'] == 0 and actual['reason'] is None and actual['signals'] == []
        and actual['group_absent'] is True and actual['no_owned_CUDA'] is True
        and actual.get('error') is None and actual.get('cleanup_error') is None
        and actual['active_and_cleanup_seconds'] <= supervision['hard_seconds']
        and actual['max_sampled_owned_GPU_bytes'] <= supervision['owned_GPU_bytes']
        and actual['max_sampled_owned_RSS_bytes'] <= supervision['owned_RSS_bytes']
        and (not parent_wait or terminal['parent_actual_exit_code'] == 0))
    if phase == 'native_baselines':
        require(successful and collection['phase_complete'] is True and collection['status'] == cost['status'] == 'complete'
            and cost['attempted_member_forwards'] == cost['completed_member_forwards'] == 9
            and actual['active_and_cleanup_seconds'] <= supervision['hard_seconds']
            and actual['max_sampled_owned_GPU_bytes'] <= supervision['owned_GPU_bytes']
            and actual['max_sampled_owned_RSS_bytes'] <= supervision['owned_RSS_bytes'],
            'Phase1 barrier requires successful complete actual child exit0 and clean measured ownership')
    return dict(terminal=terminal, release=release, actual_exit=actual, actual_owned_absence=absence,
        actual_process_witness=witness, phase_owner_succeeded=successful,
        nonzero_child_exit_retained=actual['exit_code'] != 0,
        original_per_bank_availability_unchanged=True, survivor_only_promotion=False)


def global_baselines(g, pins, cfg, records):
    """Read only compact seals globally; verify local native raw bytes, no decode."""
    freeze = read(g.bound(cfg['global_native_baseline_freeze']))
    require(freeze['schema'] == 'qk36-all9-native-tied-global-freeze-v1'
        and freeze['complete'] is True and freeze['root_frozen_before_any_candidate_calls'] is True
        and freeze['phase1_owners_closed_on_all_routes'] is True
        and freeze['plan'] == pins['global_plan'] and freeze['readout_manifest_sha256'] == cfg['readout_manifest_sha256']
        and set(freeze['routes']) == set(pins['routes']), 'All9 original baseline cohorts globally frozen before any candidate calls')
    for route_id, seed in zip(pins['plan']['route_ids'], SEEDS):
        evidence = freeze['routes'][route_id]
        receipt = read(g.bound(evidence['baseline_receipt']))
        collection = read(g.bound(evidence['phase1_collection']))
        cost = read(g.bound(evidence['phase1_cost']))
        phase_terminal(g, pins, evidence['actual_phase1_owner_terminal_custody'], route_id, 'native_baselines',
            evidence['phase1_collection'], evidence['phase1_cost'], collection, cost, cfg['readout_manifest_sha256'])
        require(receipt['route_id'] == collection['readout_route_id'] == route_id and receipt['seed'] == seed
            and receipt['local3_frozen'] is True and receipt['candidate_prediction_inspection_started'] is False
            and receipt['phase1_collection'] == evidence['phase1_collection']
            and receipt['plan'] == pins['global_plan'] and receipt['readout_manifest_sha256'] == cfg['readout_manifest_sha256']
            and collection['collection_phase'] == 'native_baselines' and collection['phase_complete'] is True
            and collection['local3_native_tied_baselines_frozen'] is True and collection['candidate_prediction_inspection_started'] is False
            and cost['attempted_member_forwards'] == cost['completed_member_forwards'] == 9
            and cost['TRAIN_updates'] == cost['backward_calls'] == cost['Adam_steps'] == 0,
            'Actual original phase1 native calls, source and closed custody on every route')
        require(len(collection['cells']) == 12 and all(r['seed'] == seed and r['original_training_route'] == route_id for r in collection['cells'])
            and all(r['attempted_member_forwards'] == r['completed_member_forwards'] == 0 for r in collection['cells'] if r['operator'] != 'native_tied'), 'No candidates called in phase1')
        for kind in KINDS:
            cohort = receipt['cohorts'][kind][str(seed)]
            native = next(r for r in collection['cells'] if (r['kind'], r['operator']) == (kind, 'native_tied'))
            require(cohort == collection['cohorts'][kind][str(seed)] and cohort['available'] is True
                and cohort['frozen_before_candidate_prediction_inspection'] is True
                and native['collection_status'] == 'complete'
                and native['raw_prediction_archive'] == cohort['baseline_prediction_archive']
                and native['attempted_member_forwards'] == native['completed_member_forwards'] == native['members'], 'All9 exact complete bank/native cohort seals')
            if route_id == cfg['readout_route_id']:
                g.bound(cohort['archive']); g.bound(native['raw_prediction_archive'])
        if route_id == cfg['readout_route_id']:
            require(cfg['local_phase1_collection'] == evidence['phase1_collection'], 'Exact local phase1 before candidate release')
            for row in records:
                native = next(r for r in collection['cells'] if r['cell'] == row['cell'])
                for key in ('spec', 'selected_checkpoint', 'own_selected_checkpoints', 'original_completion', 'original_operator_binding'):
                    require(native[key] == row[key], 'Original selected bytes/cell unchanged between phases')
            local = dict(collection=collection, phase1_cost=evidence['phase1_cost'], baseline_receipt=evidence['baseline_receipt'])
    return local


def consume(release_path, release_sha256):
    require(sha(release_path) == release_sha256, 'Exact separate postclosure root opening release')
    cfg = read(release_path)
    require(cfg.get('schema') == 'qk36-original-route-two-phase-release-v1', 'Exact readout schema')
    for flag in ('enabled', 'root_execution_authorized', 'source_review_approved', 'whole36_complete',
        'actual_all_route_custody_closed', 'trusted_selected_state_deserialization_authorized',
        'original_local12_resident_states_confirmed', 'qualified_readout_route_confirmed',
        'finite_external_owner_confirmed', 'inclusive_cost_accounting_confirmed'):
        require(cfg.get(flag) is True, 'Inactive pending separate root opening: ' + flag)
    require(cfg['maximum_route_member_forwards'] == 36 and cfg['maximum_member_forwards'] == (9 if cfg['collection_phase'] == 'native_baselines' else 27) and cfg['collection_phase'] in ('native_baselines', 'candidates') and cfg['execution_source_commit'] == COMMIT
        and all(cfg[k] is False for k in ('training', 'TEST_access', 'automatic_retry', 'reselection', 'calibration')),
        'Fixed original36 calls per route in9+27 phases; full36 training anchor')
    pins = read(HERE / 'SOURCE_BINDINGS.json'); g = helpers(pins)
    require(cfg['readout_route_id'] in pins['routes'], 'Explicit original serving route before binary hash reads')
    actual_route = pins['routes'][cfg['readout_route_id']]
    require(socket.gethostname() == actual_route['hostname']
        and Path.cwd().resolve() == Path(actual_route['repository']).resolve()
        and PHASE == Path(actual_route['phase']).resolve()
        and Path(sys.executable).resolve() == Path(actual_route['python']).resolve()
        and ([] if not os.environ.get('PYTHONPATH') else os.environ['PYTHONPATH'].split(os.pathsep)) == actual_route['PYTHONPATH']
        and os.environ.get('CUDA_VISIBLE_DEVICES') == actual_route['GPU_uuid'], 'Original qualified runtime before local selected/role byte inspection')
    require(cfg['readout_manifest_sha256'] == sha(HERE / 'MANIFEST.json'), 'Reviewed readout source')
    g.verify(dict(path=str((HERE / 'MANIFEST.json').relative_to(PHASE)), sha256=cfg['readout_manifest_sha256']))
    for row in pins['source_files']:
        g.bound(row)
    for row in pins['source_manifests']:
        g.verify(row)
    plan = read(g.bound(pins['global_plan']))
    require(plan == pins['plan'] and plan['enabled'] is True and plan['execution_source_commit'] == COMMIT
        and plan['seeds'] == list(SEEDS) and plan['operators'] == list(OPERATORS) and plan['kinds'] == list(KINDS)
        and plan['required_group_endpoints'] == 36 and plan['automatic_retry'] is False
        and plan['lane_owner_manifest_sha256'] == pins['reuse']['lane_manifest']['sha256']
        and plan['route_adapter_manifest_sha256'] == pins['reuse']['routing_manifest']['sha256'], 'Immutable prospectively fixed Q/K36 panel')
    require(set(cfg['route_absence_receipts']) == set(plan['route_ids']), 'All three externally authenticated actual route receipts')
    receipts = {route: read(g.bound(row)) for route, row in cfg['route_absence_receipts'].items()}
    queue = g.module(g.bound(pins['reuse']['queue_program']), '_qk36_unchanged_union_gate')
    union = queue.verify_union(plan=plan, plan_sha256=pins['global_plan']['sha256'], receipts=receipts)
    require(union['all36_complete'] is True, 'Complete36 before any selected state or metric')
    require(cfg['readout_route_id'] in pins['routes'], 'Explicit qualified serving route before namespace mapping')
    namespace = g.module(HERE / 'namespace.py', '_qk36_verified_origin_namespaces')
    namespace_custody = read(g.bound(cfg['namespace_custody']))
    require(namespace_custody['schema'] == 'qk36-original-resident-namespace-custody-v1'
        and namespace_custody['complete'] is True and namespace_custody['root_verified_original_namespaces'] is True
        and namespace_custody['plan'] == pins['global_plan'] and namespace_custody['route_id'] == cfg['readout_route_id']
        and namespace_custody['original_phase'] == pins['routes'][cfg['readout_route_id']]['phase']
        and namespace_custody['original_receipt_bytes_preserved'] is True
        and namespace_custody['foreign_checkpoint_staging_required'] is False
        and namespace_custody['source_files_or_receipts_rewritten'] is False
        and namespace_custody['mounts_or_symlinks_created_by_mapping'] is False, 'Verified own original paths; foreign compact metadata only')
    fresh = fresh_custody(g, pins, cfg, receipts)
    lane = source_lane(g, pins); routes = pins['routes']; route_pins = pins['route_source_pins']
    expected_queues = queue.freeze_queues(plan['seeds'], plan['route_ids'])
    records = []; originals = {}
    for route_id, specs in expected_queues.items():
        if route_id != cfg['readout_route_id']:
            continue  # Foreign24 selected files remain on their original routes.
        root = g.inside(plan['fresh_output_relative']) / route_id
        absence_path = g.bound(cfg['route_absence_receipts'][route_id]); receipt = receipts[route_id]
        require(absence_path == root / 'ABSENCE_RECEIPT.json', 'Exact original route absence endpoint')
        close = read(root / 'CLOSURE.json'); owner = read(root / 'PARENT_OWNER.json')
        require(sha(root / 'CLOSURE.json') == receipt['closure_sha256']
            and all(receipt[k] == v for k, v in close.items() if k != 'actual_owned_absence_attested') and close['complete'] is True
            and close['clean_owned_terminal'] is True and close['error'] is None and close['cleanup_error'] is None
            and close['incomplete_child'] is None and close['lane_endpoint_count'] == 12
            and close['route_id'] == route_id and close['physical_GPU_uuid'] == plan['GPU_assignment'][route_id]
            and close['plan_sha256'] == pins['global_plan']['sha256'] and close['execution_source_commit'] == COMMIT
            and close['parent'] == owner['parent'] and close['release_sha256'] == owner['release_sha256']
            and close['scores_read'] is False and close['automatic_retry'] is False, 'Exact completed original lane/parent/work custody')
        require(receipt['owned_identities'] == [close['parent']] + [p for exit in close['exits'] for p in exit['witnessed_owned_members']]
            and receipt['owned_groups'] == [close['parent']] + [exit['child'] for exit in close['exits']], 'Original attester exact identities and groups')
        training_release_path = g.bound(pins['training_releases'][route_id]); training_release = read(training_release_path)
        require(close['release_sha256'] == pins['training_releases'][route_id]['sha256']
            and training_release['plan'] == pins['global_plan'] and training_release['route_id'] == route_id
            and training_release['resources'] == plan['root_resource_limits'][route_id]
            and training_release['qualification_adoption'] == plan['qualification_adoptions'][route_id]
            and training_release['execution_source_commit'] == COMMIT and training_release['enabled'] is True,
            'Original exact source/qualification/limits release')
        require(len(close['exits']) == 12, 'Twelve actual scientific exits per route')
        adoption = plan['qualification_adoptions'][route_id]; qualified = read(g.bound(adoption['report']))
        require(qualified['complete'] is True and qualified['scientific_fit'] is False and qualified['completed_updates'] == 24
            and qualified['completed_member_view_forwards'] == qualified['completed_member_view_backwards'] == 144
            and qualified['completed_Adam_steps'] == 48 and qualified['VALID_metrics'] == qualified['VALID_forwards'] == 0
            and qualified['TEST_access'] is False, 'Original completed unscored qualification work and cost custody')
        route_originals = dict(closure=close, owner=owner, absence_receipt=receipt, original_qualification=qualified,
            original_qualification_adoption=adoption, cells=[])
        for index, spec in enumerate(specs):
            output = root / spec['key']; row = receipt['cells'][spec['key']]
            require(not (output / 'FAILURE.json').exists() and sha(output / 'COMPLETE.json') == row['completion_sha256'], 'Original full endpoint hash')
            checked = lane.work_receipt(spec, output, training_release, routes[route_id], route_pins)
            require(checked == row, 'Unchanged original full1100/F/100/source/route validator')
            endpoint = read(output / 'COMPLETE.json'); run = read(output / 'RUN.json'); context = endpoint['owner_context']
            source = context['source']; operator_binding = endpoint['operator_binding']
            cell_program = next(r for r in route_pins['source_files'] if r['path'] == route_pins['cell_directory'] + '/cell.py')
            catalog = next(r for r in pins['source_files'] if r['path'] == Path(pins['reuse']['routing_program']['path']).parent.as_posix() + '/ROUTES.json')
            require(run['owner_context'] == context and context['cell'] == spec and context['release_sha256'] == close['release_sha256']
                and context['runtime'] == routes[route_id] and context['roles'] == route_pins['roles']
                and source['owner_manifest_sha256'] == plan['route_adapter_manifest_sha256']
                and source['cell_program_sha256'] == cell_program['sha256'] and source['route_id'] == route_id
                and source['route_catalog_sha256'] == catalog['sha256']
                and operator_binding['kind'] == spec['kind'] and operator_binding['seed'] == spec['seed']
                and operator_binding['adapter_program_sha256'] == route_pins['math_program_sha256']
                and operator_binding['adapter_manifest_sha256'] == route_pins['math_manifest_sha256']
                and operator_binding['native_sha256'] == route_pins['native']['sha256']
                and operator_binding['factor_sha256'] == pins['public_core_sha256']['factors.py']
                and operator_binding['mathematical_source_changed'] is False and operator_binding['bias_outside_scale'] is True
                and endpoint['TEST_scoring'] is False and run['TEST_scoring'] is False
                and run['core'] == pins['public_core_sha256'] and run['driver_sha256'] == pins['reuse']['public_driver']['sha256']
                and run['data']['train_npz_sha256'] == plan['role_hashes']['train']
                and run['data']['valid_npz_sha256'] == plan['role_hashes']['valid'], 'Exact source and complete official role identity')
            for filename, digest in row['bound_files'].items():
                require(Path(filename).name == filename and sha(output / filename) == digest, 'Original endpoint-bound state/trace bytes')
            selected = g.binding(output / 'selected.pt')
            require(selected['sha256'] == endpoint['selected_sha256'] == row['bound_files']['selected.pt'], 'Source selected bank bytes, no reselection')
            name = '%02d__%s__%s' % (index, spec['operator'], spec['kind'])
            handle_path = root / 'handles' / (name + '.json'); exit_path = root / 'handles' / (name + '.EXIT.json')
            handle, exit = read(handle_path), read(exit_path); child = exit['child']
            require(exit == close['exits'][index] and exit['exit_code'] == 0 and exit['reaped'] is True
                and exit['group_absent'] is True and exit['no_owned_CUDA'] is True and exit['reason'] is None
                and exit['signals'] == [] and exit['automatic_retry'] is False and exit['scores_read'] is False
                and handle['spec'] == spec and handle['child'] == child and handle['parent'] == close['parent']
                and child['pid'] == child['group'] == child['session'] and child.get('boot_id')
                and namespace.mapped(g, pins, route_id, handle['output'], plan['fresh_output_relative'] + '/' + route_id + '/' + spec['key']) == output
                and handle['release_sha256'] == close['release_sha256']
                and namespace.mapped(g, pins, route_id, handle['release_path'], pins['training_releases'][route_id]['path']) == training_release_path,
                'Actual cell handle, clean exit, boot-bound isolated group and direct reap')
            cost_path = root / 'costs' / (name + '.json'); cost = read(cost_path)
            require(namespace.mapped(g, pins, route_id, handle['cost'], plan['fresh_output_relative'] + '/' + route_id + '/costs/' + name + '.json') == cost_path
                and sha(cost_path) == exit['costs_sha256'] and cost['success'] is True
                and cost['spec'] == spec and cost['release_sha256'] == close['release_sha256']
                and cost['execution_source_commit'] == COMMIT, 'Original worker training/storage/resource costs')
            own = [g.binding(output / ('own_best_' + str(m) + '.pt')) for m in range(4)] if spec['kind'] == 'independent4' else []
            records.append(dict(cell=spec['key'].replace('/', '__'), scientific_key=spec['key'], spec=spec,
                seed=spec['seed'], arm=spec['kind'], kind=spec['kind'], operator=spec['operator'],
                condition=spec['kind'] + '/' + spec['operator'], members=1 if spec['kind'] == 'single' else 4,
                family_status='complete', collection_status='pending', historical_reference=False,
                selected_checkpoint=selected, own_selected_checkpoints=own,
                own_bank_metrics=g.binding(output / 'PRIVATE_OWN_BEST_BANK.json') if own else None,
                original_owner_context=context, original_operator_binding=endpoint['operator_binding'],
                original_training_route=route_id, original_completion=g.binding(output / 'COMPLETE.json'),
                original_cost=g.binding(cost_path), original_exit=g.binding(exit_path),
                attempted_member_forwards=0, completed_member_forwards=0))
            route_originals['cells'].append(dict(spec=spec, endpoint=endpoint, worker_cost=cost, handle=handle, exit=exit))
        originals[route_id] = route_originals
    require(len(records) == 12 and sum(r['members'] for r in records) == 36 and len({r['seed'] for r in records}) == 1, 'Original seed12 groups and exactly36 local member slots')
    route_id = cfg['readout_route_id']; require(route_id in routes, 'Explicit qualified serving route')
    route = routes[route_id]
    subprocess.run(['git', 'cat-file', '-e', COMMIT + '^{commit}'], cwd=route['repository'], check=True, timeout=10)
    subprocess.run(['git', 'merge-base', '--is-ancestor', COMMIT, 'HEAD'], cwd=route['repository'], check=True, timeout=10)
    for row in pins['training_source_manifests']:
        contents = subprocess.check_output(['git', 'show', COMMIT + ':' + pins['repository_phase_relative'] + '/' + row['path']],
            cwd=route['repository'], timeout=10)
        require(hashlib.sha256(contents).hexdigest() == row['sha256'], 'Frozen source manifests present in original execution anchor')
    for role in route_pins['roles'].values():
        g.bound(role)
    custody = original_custody(g, pins, cfg, receipts)
    phase1 = global_baselines(g, pins, cfg, records) if cfg['collection_phase'] == 'candidates' else None
    supervision = read(g.bound(cfg['external_supervision'])); entry = HERE / 'collect_route.py'
    require(supervision['enabled'] is True and supervision['finite_owned_bound_confirmed'] is True
        and supervision['entry_program'] == g.binding(entry) and supervision['custody_program'] == pins['reuse']['custody_program']
        and supervision['argv_prefix'] == [route['python'], '-B', str(entry)]
        and supervision['release_argument_path'] == str(Path(release_path).resolve())
        and supervision['route_id'] == route_id and supervision['physical_GPU_uuid'] == route['GPU_uuid']
        and 0 < supervision['active_seconds'] <= 7200 and 0 < supervision['cleanup_seconds'] <= 15
        and supervision['hard_seconds'] == supervision['active_seconds'] + supervision['cleanup_seconds']
        and 0 < supervision['owned_GPU_bytes'] <= plan['root_resource_limits'][route_id]['max_owned_GPU_bytes']
        and 0 < supervision['owned_RSS_bytes'] <= plan['root_resource_limits'][route_id]['max_owned_RSS_bytes']
        and 0 < supervision['maximum_output_bytes'] <= 4 * 1024**3
        and supervision['actual_exit_cleanup_and_mirroring_cost_required'] is True, 'Reuse existing finite owned resource/terminal boundary')
    runtime_evidence = read(g.bound(cfg['runtime_evidence']))
    version_keys = {'torch', 'numpy', 'torch_geometric', 'torch_scatter', 'torch_sparse', 'ogb'}
    require(runtime_evidence['schema'] == 'qk36-serving-runtime-evidence-v2' and runtime_evidence['complete'] is True
        and runtime_evidence['route_id'] == route_id and runtime_evidence['physical_GPU_uuid'] == route['GPU_uuid']
        and runtime_evidence['python'] == route['python'] and set(runtime_evidence['versions']) == version_keys
        and all(isinstance(v, str) and v for v in runtime_evidence['versions'].values())
        and all(runtime_evidence['versions'][key] == route[key] for key in version_keys if key in route),
        'Actual serving providers bound; undeclared allocation versions are supplied explicitly, never borrowed')
    cfg['expected_serving_provider_versions'] = runtime_evidence['versions']
    g.bound(cfg['resource_readiness'])
    output = g.inside(cfg['output_directory'])
    frozen_roots = {Path(r['path']).parts[0] for r in pins['source_files']} | {HERE.name, plan['fresh_output_relative']}
    require(not output.exists() and output.parent.is_dir()
        and not any(output.is_relative_to(g.inside(root)) for root in frozen_roots), 'Fresh server-only output outside all frozen/running sources')
    all_cells = [cell for lane_history in originals.values() for cell in lane_history['cells']]
    tracked_work = {key: sum(cell['endpoint']['operator_work'][key] for cell in all_cells)
        for key in ('update_attempts', 'completed_updates', 'member_view_forwards', 'member_view_backwards', 'Adam_steps')}
    selection_calls = sum((1100 + int(r['kind'] == 'independent4')) * r['members'] for r in records)
    return cfg, pins, records, output, g, route, dict(training_lanes=originals, fresh_root_custody=fresh,
        complete36_union=union, original_route_resident_state_custody=custody,
        local_phase1_reuse=phase1, global_baseline_freeze=cfg.get('global_native_baseline_freeze'),
        verified_original_to_serving_namespace_custody=namespace_custody, serving_runtime_evidence=runtime_evidence,
        original_tracked_F_work=tracked_work, local_original_training_native_bodies=24,
        original_selection_member_forwards_from_unchanged_complete_driver_schedule=selection_calls,
        qualification_adoptions_preserved=plan['qualification_adoptions'], missing_original_cost_slots=None), supervision
