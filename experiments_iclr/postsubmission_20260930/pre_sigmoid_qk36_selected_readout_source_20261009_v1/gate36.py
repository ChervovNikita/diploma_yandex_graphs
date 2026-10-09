"""Stdlib whole36/source/custody gate. No metrics, tensors or remote calls."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
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


def consume(release_path, release_sha256):
    require(sha(release_path) == release_sha256, 'Exact separate postclosure root opening release')
    cfg = read(release_path)
    require(cfg.get('schema') == 'qk36-selected-readout-release-v1', 'Exact readout schema')
    for flag in ('enabled', 'root_execution_authorized', 'source_review_approved', 'whole36_complete',
        'actual_all_route_custody_closed', 'trusted_selected_state_deserialization_authorized',
        'all36_server_only_states_staged_and_transfer_cost_bound', 'qualified_readout_route_confirmed',
        'finite_external_owner_confirmed', 'inclusive_cost_accounting_confirmed'):
        require(cfg.get(flag) is True, 'Inactive pending separate root opening: ' + flag)
    require(cfg['maximum_member_forwards'] == MAX_FORWARDS and cfg['execution_source_commit'] == COMMIT
        and all(cfg[k] is False for k in ('training', 'TEST_access', 'automatic_retry', 'reselection', 'calibration')),
        'Fixed108 serving-only calls; exact full36 training anchor')
    pins = read(HERE / 'SOURCE_BINDINGS.json'); g = helpers(pins)
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
    fresh = fresh_custody(g, pins, cfg, receipts)
    lane = source_lane(g, pins); routes = pins['routes']; route_pins = pins['route_source_pins']
    expected_queues = queue.freeze_queues(plan['seeds'], plan['route_ids'])
    records = []; originals = {}
    for route_id, specs in expected_queues.items():
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
                and Path(handle['output']).resolve() == output and handle['release_sha256'] == close['release_sha256']
                and Path(handle['release_path']).resolve() == training_release_path,
                'Actual cell handle, clean exit, boot-bound isolated group and direct reap')
            cost_path = root / 'costs' / (name + '.json'); cost = read(cost_path)
            require(Path(handle['cost']).resolve() == cost_path and sha(cost_path) == exit['costs_sha256'] and cost['success'] is True
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
    require(len(records) == 36 and sum(r['members'] for r in records) == MAX_FORWARDS, 'All36 groups and exactly108 selected member slots')
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
    transfer = read(g.bound(cfg['transfer_custody_and_cost']))
    require(transfer['complete'] is True and transfer['all36_after_closure'] is True and transfer['raw_states_server_only'] is True
        and transfer['destination_route_id'] == route_id and transfer['plan'] == pins['global_plan']
        and transfer['TEST_access'] is False, 'Actual server-only staging/cost custody after complete36')
    supervision = read(g.bound(cfg['external_supervision'])); entry = HERE / 'collect36.py'
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
    for field in ('runtime_evidence', 'resource_readiness'):
        g.bound(cfg[field])
    output = g.inside(cfg['output_directory'])
    frozen_roots = {Path(r['path']).parts[0] for r in pins['source_files']} | {HERE.name, plan['fresh_output_relative']}
    require(not output.exists() and output.parent.is_dir()
        and not any(output.is_relative_to(g.inside(root)) for root in frozen_roots), 'Fresh server-only output outside all frozen/running sources')
    all_cells = [cell for lane_history in originals.values() for cell in lane_history['cells']]
    tracked_work = {key: sum(cell['endpoint']['operator_work'][key] for cell in all_cells)
        for key in ('update_attempts', 'completed_updates', 'member_view_forwards', 'member_view_backwards', 'Adam_steps')}
    selection_calls = sum((1100 + int(r['kind'] == 'independent4')) * r['members'] for r in records)
    return cfg, pins, records, output, g, route, dict(training_lanes=originals, fresh_root_custody=fresh,
        complete36_union=union, actual_server_transfer_and_staging_cost=transfer,
        original_tracked_F_work=tracked_work, fresh_training_native_bodies=72,
        original_selection_member_forwards_from_unchanged_complete_driver_schedule=selection_calls,
        qualification_adoptions_preserved=plan['qualification_adoptions'], missing_original_cost_slots=None), supervision
