"""Stdlib-only whole-family custody gate; no live polling or payload decoding."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SEEDS = (6101, 6203, 6307)
POLICIES = ('alphaF', 'allJ', 'phiJ', 'relationJ')
CONDITIONS = POLICIES + ('single', 'independent4')
MAX_FORWARDS = 48
WORK = dict(shadow_member_forwards=8800, replay_member_forwards=8800,
    output_cotangent_collections=2200, member_reverse_collections=17600,
    optimizer_bank_updates=1100, exact_member_RNG_endpoint_checks=1100)


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def helpers(pins):
    row = pins['reuse']['legacy_gate']; path = PHASE / row['path']
    require(sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Exact stdlib custody helpers')
    spec = importlib.util.spec_from_file_location('gate', path)
    gate = importlib.util.module_from_spec(spec); sys.modules['gate'] = gate; spec.loader.exec_module(gate)
    return gate


def original_union(g, pins, cfg, terminal):
    """The original12 subset of the sealed union15 metadata checks."""
    p = pins['union_pins']; oldroot = g.inside(p['original_activation']); fresh = g.inside(p['unused2_output'])
    old = read(g.bound(p['old_terminal_custody']))
    require(old['old_owner_absent'] is True and old['quality_scores_read'] is False
        and old['checkpoint_loaded'] is False, 'Original old10 metadata custody')
    for path, row in old['files'].items():
        g.bound(dict(row, path=path))
    original_family = read(oldroot / 'FAMILY_CLOSURE.json'); original_lane_failure = read(oldroot / 'LANE_0_FAILURE.json')
    require(original_family['complete'] is False
        and original_lane_failure['error']
        == 'TimeoutError: Fixed fresh-GPU resource window exhausted before child launch', 'Preserve original failed closure')
    union_path = g.bound(cfg['original12_union_closure']); union = read(union_path)
    require(union_path == fresh / 'UNION_CLOSURE.json'
        and union['schema'] == 'Wiki12-old10-plus-never-started2-union-closure-v1'
        and union['complete'] is True and union['original_fixed_cells'] == 12 and union['original_completed_cells'] == 10
        and union['old_terminal_custody'] == p['old_terminal_custody'] and union['fatal_error'] is None
        and all(union[k] is True for k in ('original_complete_false_closure_immutable',
            'original_completed_cells_not_repeated', 'never_started_cells_only'))
        and all(union[k] is False for k in ('automatic_retry', 'quality_scores_read', 'TEST_access')), 'Exact complete original12 union')
    failed = union['preserved_original_failed_preflight']; preflight = read(g.bound(failed))
    require(g.bound(failed) == oldroot / 'logs/6307_residual_only.PREFLIGHT.json'
        and preflight == failed['observation'] and preflight['scientific_child_started'] is False
        and preflight['elapsed_seconds'] >= 1800, 'Retain failed original admission and its wait cost')
    old_rows = {r['cell_id']: r for r in old['completed']}; new_rows = {r['cell_id']: r for r in union['new_completed']}
    roster = p['roster']; expected = {r['cell'] for r in roster}
    require(len(old_rows) == 10 and len(new_rows) == 2 and set(new_rows) == {'6307_residual_only', '6307_combined'}
        and set(old_rows) == {r['cell'] for r in roster if r['custody_partition'] == 'old10'}
        and [r['release']['cell_id'] for r in old['unused']] == ['6307_residual_only', '6307_combined']
        and all(r['scientific_child_started'] is False and r['output_absent'] is True for r in old['unused']), 'Exact ten plus never-started two')
    children = {r['cell_id']: r for r in terminal['original12_children']}
    require(len(terminal['original12_children']) == len(children) == 12 and set(children) == expected, 'All original12 terminal slots')
    owners = {r['role']: r for r in terminal['original12_owners']}
    require(len(owners) == len(terminal['original12_owners']) == 2 and set(owners) == {'original_Wiki12', 'unused2'}, 'Both original owner histories')
    for role, root, filename in (('original_Wiki12', oldroot, 'PARENT_OWNER.json'), ('unused2', fresh, 'OWNER.json')):
        receipt = root / filename; saved = read(receipt); identity = saved if role == 'original_Wiki12' else saved['owner']
        fixed = p['owners'][role]; row = owners[role]
        require(identity['PID'] == fixed['pid'] and identity['start_ticks'] == fixed['start_ticks']
            and row['identity'] == identity and row['owner_receipt'] == g.binding(receipt)
            and row['absent'] is True and row['no_CUDA_rows'] is True, 'Exact original owner terminal identity')
        if role == 'unused2':
            require(saved['admission_sha256'] == p['reuse']['unused2_admission']['sha256']
                and saved['protocol'] == p['reuse']['unused2_protocol'], 'Original unused2 owner/admission')
    work = dict(WORK, output_cotangent_collections=1100, member_reverse_collections=8800); costs = []
    source = pins['data_source_pins']
    for item in roster:
        cell = item['cell']; closed = old_rows[cell] if item['custody_partition'] == 'old10' else new_rows[cell]
        release_path = g.bound(item['release']); release = read(release_path); output = g.inside(item['output'])
        require(closed['release'] == item['release'] and not (output / 'FAILURE.json').exists(), 'Unchanged original cell release')
        endpoint = read(output / 'COMPLETE.json'); run = read(output / 'RUN.json')
        require(release['condition'] == item['condition'] and release['seed'] == item['seed'] and release['output'] == item['output']
            and release['source_manifest_sha256'] == p['reuse']['original_manifest']['sha256']
            and release['train'] == source['train'] and release['development'] == source['development']
            and release['physical_gpu_uuid'] == source['GPU_per_seed'][str(item['seed'])], 'Original union cell source/roles/GPU')
        require(endpoint['complete'] is True and endpoint['epochs'] == endpoint['steps'] == 1100
            and endpoint['execution_accounting'] == work and all(r['task'] == 'wikics' and r['seed'] == item['seed']
                and r['arm'] == r['method_identity'] == item['method_identity'] and r['TEST_scoring'] is False
                and r['mechanism_ablation']['condition'] == item['condition']
                and r['ablation_adapter_sha256'] == p['reuse']['original_adapter']['sha256'] for r in (endpoint, run))
            and run['core'] == pins['public_core_sha256']
            and run['native']['polynormer_model_sha256'] == source['polynormer']['sha256']
            and run['data']['train_npz_sha256'] == source['train']['sha256']
            and run['data']['valid_npz_sha256'] == source['development']['sha256'], 'Original full1100 endpoint identity')
        exit_path = g.bound(closed['exit']) if item['custody_partition'] == 'old10' else fresh / 'logs' / (cell + '.EXIT.json')
        receipt = read(exit_path); identity = receipt['raw_identity_observation']
        argv = [source['python'], '-B', str(g.bound(p['reuse']['original_adapter'])), '--release', str(release_path), '--release-sha256', item['release']['sha256']]
        require(identity['argv'] == argv and identity['PID'] == identity['pgid'] == identity['sid']
            and receipt['exit_code'] == 0 and receipt['reason'] is None and receipt['signals_sent'] == []
            and receipt['terminal_wait_observed'] is True and receipt['job_sha256'] == item['release']['sha256'], 'Original union child actual exit/reap')
        if item['custody_partition'] == 'old10':
            require(g.bound(closed['completion']) == output / 'COMPLETE.json' and closed['exit']['pid'] == identity['PID']
                and closed['exit']['child_absent'] is True and closed['exit']['child_no_CUDA_rows'] is True, 'Immutable old10 completion/exit')
        else:
            require(not (oldroot / 'logs' / (cell + '.CHILD_STARTED.json')).exists()
                and sha(output / 'COMPLETE.json') == closed['completion_sha256'] and sha(exit_path) == closed['exit_sha256']
                and closed['exit_receipt'] == receipt and closed['complete'] is True and closed['child_absent'] is True
                and closed['child_no_CUDA_rows'] is True, 'Successful unchanged never-started2 custody')
        child = children[cell]
        require(child['identity'] == identity and child['source_receipt'] == g.binding(exit_path)
            and child['child_absent'] is True and child['child_no_CUDA_rows'] is True
            and child['wait_and_reap_observed'] is True, 'Root original12 child terminal custody')
        require(sha(output / 'selected.pt') == endpoint['selected_sha256'], 'Original union selected byte custody')
        costs.append(dict(cell=cell, endpoint=endpoint, owner_row=closed, exit_receipt=receipt))
    original_lane_history = {p.name: read(p) for p in oldroot.glob('LANE_*_*.json')
        if p.name.endswith(('_COMPLETE.json', '_FAILURE.json'))}
    return dict(union=union, original_failed_family_closure=original_family, original_lane_history=original_lane_history,
        old_terminal_custody=old, preserved_failed_preflight=preflight, cells=costs)


def historical(g, pins, cfg, terminal):
    """Exact saved metadata plus root original-route custody; no control fit."""
    collector = g.module(g.bound(pins['reuse']['historical_collect']), '_relation18_historical_custody')
    tree = ast.parse(Path(collector.__file__).read_text())
    fn = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'consume'))
    # Its sole fresh-output condition becomes an existing completed-output check.
    class CompletedOutput(ast.NodeTransformer):
        count = 0
        route_count = 0
        def visit_Assign(self, node):
            self.generic_visit(node)
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == 'output'):
                require(ast.unparse(node.value) == "Path(cfg['output_directory']).resolve()", 'Exact original completed-output route expression')
                node.value = ast.Name(id='completed_output', ctx=ast.Load())
                self.route_count += 1
            return node
        def visit_Call(self, node):
            self.generic_visit(node)
            if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == 'output' and node.func.attr == 'exists'):
                node.func.attr = 'is_dir'; self.count += 1
                return ast.UnaryOp(op=ast.Not(), operand=node)
            return node
    adapter = CompletedOutput(); fn = adapter.visit(fn)
    require(adapter.count == adapter.route_count == 1, 'Unique original historical completed-output predicate and route mapping')
    namespace = dict(vars(collector), completed_output=g.bound(pins['historical_metadata']['collection']).parent.parent)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
        collector.__file__ + ':completed-custody-only', 'exec'), namespace)
    release_path = g.bound(cfg['historical_prediction_release'])
    oldcfg, export, extraction, old_output = namespace['consume'](release_path, cfg['historical_prediction_release']['sha256'])
    # Original checkpoints remain on their original allocation. The root must
    # actually run the unmodified whole24 preflight there and bind its saved
    # result/custody here. Copied metadata alone cannot satisfy this branch.
    original_gate = export['gate']
    custody = read(g.bound(cfg['historical_original_route_custody']))
    original_phase = Path(pins['historical_collector_launch']['phase_directory'])
    require(custody['schema'] == 'relation18-original-route-historical-custody-attestation-v1'
        and custody['root_observed'] is True and custody['original_reader_gate_passed'] is True
        and custody['collector_child_direct_wait_reap'] is True
        and all(custody[k] is False for k in ('watcher_direct_OS_exit_or_wait_observed', 'original_scores_changed', 'raw_payloads_decoded', 'TEST_access'))
        and custody['new_scientific_fits'] == 0 and custody['hostname'] == 'anogena-2-0'
        and Path(custody['repository']) / 'experiments_iclr/postsubmission_20260930' == original_phase
        and custody['physical_GPU_uuid'] == 'GPU-44039938-fd82-41d2-fefd-de71514e2fac',
        'Actual root original-route whole24 custody, values remain closed')
    checked = read(g.bound(custody['original_gate']))
    require(checked['passed'] is True and checked['family_closed'] is True
        and all(checked[key] == original_gate[key] for key in ('closure', 'owner', 'activation', 'source_manifest_sha256', 'reader_manifest_sha256'))
        and [(row['cell'], row['status'], row.get('selected_checkpoint')) for row in checked['cells']]
        == [(row['cell'], row['status'], row.get('selected_checkpoint')) for row in original_gate['cells']]
        and custody['checkpoint_hashes_reverified'] == [dict(cell=row['cell'], selected_checkpoint=row['selected_checkpoint'])
            for row in checked['cells'] if row.get('selected_checkpoint')]
        and custody['historical_metadata_export'] == oldcfg['metadata_export']
        and custody['historical_extraction_cost'] == oldcfg['metadata_extraction_cost']
        and custody['historical_collector_release'] == cfg['historical_prediction_release']
        and custody['historical_terminal_evidence'] == oldcfg['terminal_evidence']
        and custody['historical_collection'] == pins['historical_metadata']['collection']
        and custody['historical_cost'] == pins['historical_metadata']['cost']
        and custody['historical_archives'] == [dict(cell_id=row['cell_id'], original_archive=row['raw_prediction_archive_inherited_binding'])
            for row in pins['reference_pointers']['records']]
        and oldcfg['output_directory'] == str(original_phase / Path(pins['historical_metadata']['collection']['path']).parent.parent),
        'Same original whole24 gate, all retained statuses and exact selected bindings verified on original route')
    historical_terminal = read(g.bound(oldcfg['terminal_evidence']))
    child_terminal = read(g.bound(custody['historical_child_terminal']))
    expected_handles = [dict(PID=row['PID'], expected_start_ticks=row.get('start_ticks'), present=False)
        for row in historical_terminal['current_observed_handles']]
    expected_handles.append(dict(PID=child_terminal['identity']['PID'], expected_start_ticks=child_terminal['identity']['start_ticks'], present=False))
    require(custody['actual_observations'] == expected_handles and custody['own_CUDA_rows'] == [],
        'Actual original-route observation covers every original retained handle and the collector with no own CUDA rows')
    stage = read(g.bound(cfg['historical_archive_staging']))
    require(stage['schema'] == 'relation18-exact-six-server-only-archive-staging-v1'
        and all(stage[k] is True for k in ('complete', 'root_observed', 'fresh_scope_created', 'server_only',
            'byte_preserved', 'original_paths_unchanged', 'all_original_and_staged_hashes_verified'))
        and stage['original_route_custody'] == cfg['historical_original_route_custody']
        and stage['arrays_decoded'] is False and stage['TEST_access'] is False,
        'Exact root byte-preserved six archive staging, without decoding')
    require(stage['directory'] == pins['historical_archive_stage_directory'], 'Fixed root supplied fresh historical staging scope')
    stage_directory = g.inside(stage['directory'])
    g.bound(stage['actual_transfer_receipt'])
    frozen_roots = {HERE.name, 'graph_relation18_selected_readout_source_20261009_v1',
        *(Path(row['path']).parts[0] for row in pins['source_files']),
        *(Path(row['raw_prediction_archive_inherited_binding']['path']).parts[0] for row in pins['reference_pointers']['records']),
        *(Path(row['selected_checkpoint_inherited_binding']['path']).parts[0] for row in pins['reference_pointers']['records'])}
    require(stage_directory.is_dir() and not any(stage_directory.is_relative_to(g.inside(path)) for path in frozen_roots),
        'Fresh independent server-only archive scope outside all original histories/sources')
    staged = {row['cell_id']: row for row in stage['records']}
    require(len(stage['records']) == len(staged) == 6
        and set(staged) == {row['cell_id'] for row in pins['reference_pointers']['records']}, 'Exactly six fixed staged historical slots')
    collection_path = g.bound(pins['historical_metadata']['collection']); collection = read(collection_path)
    cost_path = g.bound(pins['historical_metadata']['cost']); cost = read(cost_path)
    require(collection_path == old_output / 'compact/COLLECTION.json' and cost_path == old_output / 'compact/COST.json'
        and collection['schema'] == 'internal-be-Wiki24-selected-prediction-collection-v1'
        and cost['schema'] == 'internal-be-Wiki24-selected-prediction-cost-v1'
        and collection['status'] == cost['status'] == 'complete' and collection['TEST_access'] is False
        and cost['release'] == g.binding(release_path), 'Exact completed historical prediction collection and charged cost')
    process = terminal['historical_prediction_collector']; identity = process['identity']
    child_owner_path = g.bound(process['child_owner_receipt']); child_owner = read(child_owner_path)
    child_exit_path = g.bound(process['child_terminal_receipt']); child_exit = read(child_exit_path)
    original_execution = Path(pins['historical_metadata']['collection']['path']).parents[2]
    expected_command = [pins['historical_collector_launch']['python'], '-B',
        str(original_phase / pins['reuse']['historical_collect']['path']), '--release',
        str(original_phase / cfg['historical_prediction_release']['path']), '--release-sha256', cfg['historical_prediction_release']['sha256']]
    require(process['release'] == g.binding(release_path) and process['collection'] == pins['historical_metadata']['collection']
        and process['cost'] == pins['historical_metadata']['cost'] and process['absent'] is True
        and process['no_CUDA_rows'] is True and process['wait_and_reap_observed'] is True
        and process['exit_code'] == 0 and process['evidence_kind'] == 'direct_child_wait_and_reap'
        and child_owner_path == g.inside(str(original_execution / 'CHILD_OWNER.json'))
        and child_exit_path == g.inside(str(original_execution / 'CHILD_TERMINAL.json'))
        and child_owner['identity'] == child_exit['identity'] == identity
        and type(identity['PID']) is int and type(identity['start_ticks']) is int
        and child_owner['command'] == expected_command and process['command'] == expected_command
        and child_exit['child_started'] is True and child_exit['reaped'] is True and child_exit['exit_code'] == 0
        and child_exit['timed_out'] is False and child_exit['cap_exceeded'] is False
        and child_exit['error'] is None and child_exit['cleanup_error'] is None
        and child_exit['automatic_retry'] is False and child_exit['other_processes_signalled'] is False
        and custody['historical_child_owner'] == process['child_owner_receipt']
        and custody['historical_child_terminal'] == process['child_terminal_receipt']
        and process['watcher_source'] == pins['historical_collector_launch']['watcher_source'],
        'Actual historical watcher waited/reaped this exact collector; command is separate from saved identity')
    states = {r['cell']: r for r in checked['cells']}; metas = {r['cell']: r for r in export['cells']}
    collected = {r['cell']: r for r in collection['cells']}; pointers = pins['reference_pointers']['records']; records = []
    authority = read(g.bound(pins['reuse']['reference_records']))
    references = {r['cell_id']: r for r in authority['reference_records']}
    require(len(collection['cells']) == len(collected) == 24 and set(collected) == set(states), 'Original all24 collection slots retained')
    for pointer in pointers:
        cell = pointer['cell_id']; state = states[cell]; meta = metas[cell]; row = collected[cell]; reference = references[cell]
        members = 1 if pointer['arm'] == 'single' else 4
        require(state['status'] == 'complete' and row['collection_status'] == 'complete' and row['members'] == members
            and row['selected_checkpoint'] == pointer['selected_checkpoint_inherited_binding'] == state['selected_checkpoint']
            and row['raw_prediction_archive'] == pointer['raw_prediction_archive_inherited_binding']
            and meta['member_global'] == row['selected_member_modes'] == pointer['selected_body_modes']
            and meta['selected_epochs'] == pointer['selected_epochs']
            and meta['selection'] == row['historical_selected_metadata']['selection'] == pointer['selector'],
            'Exact historical own-selected bank/epochs/modes/archive')
        staged_row = staged[cell]; inherited = pointer['raw_prediction_archive_inherited_binding']; staged_binding = staged_row['staged_archive']
        require(staged_row['original_archive'] == inherited
            and staged_binding['sha256'] == inherited['sha256'] and staged_binding['bytes'] == inherited['bytes'],
            'Staged archives preserve exact original bytes/hash and inherited path separately')
        staged_path = g.bound(staged_binding)
        require(staged_path.parent == stage_directory and staged_path.name == Path(inherited['path']).name,
            'Fixed original archive basename in root server-only staging directory')
        records.append(dict(cell=cell, seed=pointer['seed'], condition=pointer['arm'], members=members,
            method_identity=pointer['arm'], family_status='complete', collection_status='pending',
            selected_checkpoint=pointer['selected_checkpoint_inherited_binding'], original_archive=pointer['raw_prediction_archive_inherited_binding'],
            staged_archive=staged_binding,
            selected_metadata=dict(selected_epochs=pointer['selected_epochs'], selected_member_modes=pointer['selected_body_modes'],
                selector=pointer['selector'], own_selected_evaluation_only=pointer['ordinary_own_selected_ensemble'],
                original_selected_accuracy_authority=reference['original_selected_accuracy_authority'],
                original_member_accuracy_authority=reference['original_member_accuracy_authority']),
            historical_original_metadata=meta, historical_original_collection_row=row, historical_reference=True,
            attempted_member_forwards=0, completed_member_forwards=0))
    require({(r['seed'], r['condition']) for r in records} == {(s, c) for s in SEEDS for c in ('single', 'independent4')}, 'Exactly six plain historical controls')
    require({path.resolve() for path in stage_directory.iterdir()} == {g.bound(row['staged_archive']) for row in staged.values()},
        'Exactly six byte-verified archives and no additional staged payloads')
    roles = authority['closed_data_payload_bindings']
    require(roles['train']['sha256'] == pins['data_source_pins']['train']['sha256']
        and roles['valid']['sha256'] == pins['data_source_pins']['development']['sha256'], 'Historical complete roles match relation roles')
    return records, dict(original_family_gate=checked, original_family_closure=read(g.bound(checked['closure'])),
        metadata_extraction=extraction, selected_prediction_cost=cost,
        original_collection=collection, original_runtime_and_provider_preserved=True,
        original_route_custody=custody, original_route_custody_binding=cfg['historical_original_route_custody'],
        archive_staging=stage, archive_staging_binding=cfg['historical_archive_staging'],
        historical_checkpoints_not_required_on_reader_host=True, historical_collector_actual_child_exit=child_exit)


def relations(g, pins, cfg, family, terminal):
    p = pins['relation_controller_pins']; source = pins['data_source_pins']; root = g.bound(cfg['relation_family_closure']).parent
    parent_path = g.bound(cfg['relation_parent_owner']); parent = read(parent_path)
    require(parent_path == root / 'PARENT_OWNER.json' and parent['identity'] == family['parent_identity']
        and parent['release'] == family['root_release'] and parent['source_manifest'] == p['scientific_manifest']
        and parent['controller_manifest_sha256'] == p['controller_manifest']['sha256'], 'Exact saved relation parent/closure identity')
    owner = terminal['relation_owner']; identity = family['parent_identity']
    launch_path = g.bound(cfg['relation_parent_launch']); launch = read(launch_path)
    require(launch_path == g.inside(read(g.bound(family['root_release']))['activation_directory']) / 'LAUNCH.json',
        'Exact original relation activation launch receipt')
    expected_argv = [p['runtime']['python'], '-B',
        str(g.inside('graph_relation_private_credit_full12_normal_host_family_source_20261008_v2/controller.py')),
        '--release', str(g.bound(family['root_release'])), '--release-sha256', family['root_release']['sha256']]
    require(launch['launch_reserved'] is True and launch['detached'] is True
        and all(launch['owner_identity'][key] == identity[key] for key in ('PID', 'start_ticks', 'pgid', 'sid', 'argv'))
        and launch['owner_identity']['observation_complete'] is True and identity['observation_complete'] is True
        and launch['release'] == family['root_release']
        and launch['argv'] == identity['argv'] == expected_argv and launch['parent_poll'] is None
        and launch['automatic_retry'] is False and launch['scores_read'] is False,
        'Actual original detached relation parent launch/identity/root release')
    require(owner['identity'] == identity and owner['owner_receipt'] == cfg['relation_parent_owner']
        and owner['launch_receipt'] == cfg['relation_parent_launch'] and owner['evidence_kind'] == 'detached_parent_closure_and_observed_absence'
        and owner['absent'] is True and owner['process_group_absent'] is True and owner['no_CUDA_rows'] is True
        and owner['wait_and_reap_observed'] is False and owner['exit_code'] is None
        and identity['PID'] == identity['pgid'] == identity['sid'] and type(identity['start_ticks']) is int,
        'Root detached relation parent complete closure and observed absence; OS exit/reap remain unavailable')
    family_release = read(g.bound(family['root_release']))
    require(family_release['schema'] == 'graph-relation-full12-root-family-release-v1'
        and family_release['controller_manifest_sha256'] == p['controller_manifest']['sha256']
        and g.inside(family_release['owner_output_directory']) == root
        and all(family_release[k] is True for k in ('enabled', 'root_full12_launch_authorized', 'source_review_approved',
            'finite_family_resource_admission_confirmed', 'future_root_lane_admissions_authorized'))
        and all(family_release[k] is False for k in ('anchor_reuse_authorized', 'comparative_opening_authorized', 'TEST_access', 'automatic_retry')), 'Original training-only relation release')
    children = {r['cell_id']: r for r in terminal['relation_children']}
    expected = {str(s) + '_' + c for s in SEEDS for c in POLICIES}
    require(len(children) == len(terminal['relation_children']) == 12 and set(children) == expected, 'All relation12 terminal slots')
    records = []; costs = []; qualifications = []
    mutable = {'enabled', 'root_execution_authorized', 'source_review_approved', 'native_qualification_approved',
        'source_manifest_sha256', 'native_qualification', 'physical_gpu_uuid', 'GPU_assignment_requires_root_freeze'}
    for index, lane in enumerate(family['lane_results']):
        fixed = p['lanes'][str(index)]; path = g.bound(cfg['relation_lane_closures'][str(index)])
        require(path == root / ('LANE_' + str(index) + '_CLOSURE.json') and read(path) == lane
            and lane['complete'] is True and lane['lane'] == index and lane['physical_gpu_uuid'] == fixed['physical_gpu_uuid']
            and lane['failure'] is None and lane['completed'] == fixed['cells']
            and [r['cell_id'] for r in lane['rows']] == fixed['cells']
            and all(lane[k] is False for k in ('automatic_retry', 'scores_read', 'TEST_access')), 'Complete exact relation lane dictionary')
        admitted = lane['lane_admission']; admission = read(g.bound(admitted['binding']))
        require(admitted['value'] == admission and read(root / ('LANE_' + str(index) + '_ADMISSION.json')) == admitted
            and admission['schema'] == 'graph-relation-full12-root-lane-admission-v1'
            and all(admission[k] is True for k in ('enabled', 'root_fit_launch_authorized', 'source_review_approved', 'native_qualification_approved'))
            and admission['TEST_access'] is False and admission['automatic_retry'] is False and admission['lane'] == index
            and admission['physical_gpu_uuid'] == fixed['physical_gpu_uuid']
            and admission['controller_manifest_sha256'] == p['controller_manifest']['sha256']
            and admission['source_manifest_sha256'] == p['scientific_manifest']['sha256'], 'Exact relation lane admission')
        qualification = read(g.bound(admission['native_qualification']))
        require(qualification['complete'] is True and qualification['source_static_only'] is False
            and qualification['source_manifest_sha256'] == p['scientific_manifest']['sha256']
            and qualification['physical_gpu_uuid'] == fixed['physical_gpu_uuid']
            and qualification['policies'] == list(POLICIES) and qualification['real_complete_TRAIN_updates'] == 8
            and qualification['VALID_scores_read'] is False and qualification['TEST_access'] is False, 'Actual same-GPU original qualification')
        qualifications.append(dict(lane=index, binding=admission['native_qualification'], original_metadata=qualification))
        require([r['cell_id'] for r in admission['cells']] == fixed['cells'], 'Frozen admitted lane order')
        for item, closed in zip(admission['cells'], lane['rows']):
            cell = item['cell_id']; release_path = g.bound(item); release = read(release_path); preview = read(g.bound(p['previews'][cell]))
            require(all(release[k] == v for k, v in preview.items() if k not in mutable)
                and release['source_manifest_sha256'] == p['scientific_manifest']['sha256']
                and release['physical_gpu_uuid'] == fixed['physical_gpu_uuid']
                and release['native_qualification'] == admission['native_qualification']
                and all(release[k] is True for k in ('enabled', 'root_execution_authorized', 'source_review_approved', 'native_qualification_approved'))
                and release['GPU_assignment_requires_root_freeze'] is False, 'Unchanged original relation scientific cell')
            require(closed['release'] == g.binding(release_path) and closed['status'] == 'complete' and closed['complete'] is True
                and closed['source_manifest'] == p['scientific_manifest'] and closed['train_program'] == p['train_program']
                and closed['execution_accounting'] == WORK and closed['development_member_forwards'] == 4400
                and closed['child_absent'] is True and closed['child_no_CUDA_rows'] is True
                and closed['direct_wait_and_reap_observed'] is True, 'Original complete relation work/terminal row')
            receipt = read(g.bound(closed['exit_receipt'])); child_identity = closed['child_identity']
            argv = [source['python'], '-B', str(g.bound(p['train_program'])), '--release', str(release_path), '--release-sha256', item['sha256']]
            require(receipt == closed['actual_exit_receipt'] and receipt['raw_identity_observation'] == child_identity
                and child_identity['argv'] == argv and child_identity['PID'] == child_identity['pgid'] == child_identity['sid']
                and receipt['exit_code'] == 0 and receipt['reason'] is None and receipt['signals_sent'] == []
                and receipt['terminal_wait_observed'] is True and receipt['job_sha256'] == item['sha256'], 'Exact relation child actual exit receipt')
            child = children[cell]
            require(child['identity'] == child_identity and child['source_receipt'] == closed['exit_receipt']
                and child['child_absent'] is True and child['child_no_CUDA_rows'] is True
                and child['wait_and_reap_observed'] is True, 'Root exact relation child terminal custody')
            output = g.inside(release['output']); complete_path = g.bound(closed['completion'])
            require(complete_path == output / 'COMPLETE.json' and not (output / 'FAILURE.json').exists(), 'Original relation completion location')
            endpoint = read(complete_path); run = read(output / 'RUN.json'); descriptor = endpoint['graph_relation_credit']
            method = 'graph_relation_credit__' + release['policy']
            require(endpoint['complete'] is True and endpoint['epochs'] == endpoint['steps'] == 1100
                and all(r['task'] == 'wikics' and r['seed'] == release['seed'] and r['arm'] == r['method_identity'] == method
                    and r['TEST_scoring'] is False and r['source_manifest_sha256'] == p['scientific_manifest']['sha256'] for r in (endpoint, run))
                and endpoint['risk_beta'] == descriptor['risk_beta'] == .5 and descriptor['auxiliary_weight'] == 0.
                and descriptor['effective_model_contrastive_flag'] is False and descriptor['partition']['policy'] == release['policy']
                and descriptor['new_method_source_manifest_sha256'] == p['scientific_manifest']['sha256']
                and descriptor['train_program_sha256'] == p['train_program']['sha256']
                and descriptor['work'] == WORK and descriptor['validation_work']['evaluations'] == 1100
                and descriptor['validation_work']['member_forwards'] == 4400
                and run['core'] == pins['public_core_sha256'] and run['native']['polynormer_model_sha256'] == source['polynormer']['sha256']
                and run['data']['train_npz_sha256'] == source['train']['sha256']
                and run['data']['valid_npz_sha256'] == source['development']['sha256'], 'Complete relation endpoint/source/roles/counters')
            selected = g.bound(closed['selected_checkpoint'])
            require(selected == output / 'selected.pt' and sha(selected) == endpoint['selected_sha256'], 'Original relation selected bytes; no reselection')
            records.append(dict(cell=cell, seed=release['seed'], condition=release['policy'], members=4,
                method_identity=method, family_status='complete', collection_status='pending', selected_checkpoint=closed['selected_checkpoint'],
                complete=closed['completion'], run=g.binding(output / 'RUN.json'), exit_receipt=closed['exit_receipt'],
                original_physical_gpu_uuid=fixed['physical_gpu_uuid'], historical_reference=False))
            costs.append(dict(cell=cell, endpoint=endpoint, owner_row=closed, exit_receipt=receipt))
    require(len(records) == 12 and {r['cell'] for r in records} == expected, 'Entire relation12 roster before numerical opening')
    return records, dict(family=family, lanes=family['lane_results'], native_qualification_history=qualifications, original_cells=costs)


def consume(release_path, release_sha256):
    require(sha(release_path) == release_sha256, 'Exact separate root opening release')
    cfg = read(release_path)
    require(cfg.get('schema') == 'relation18-selected-readout-release-v2', 'Exact opening schema')
    for key in ('enabled', 'root_execution_authorized', 'source_review_approved', 'whole_relation12_complete',
        'exact_original12_union_complete', 'historical_whole24_and_collection_closed', 'all_owners_and_children_terminal',
        'trusted_checkpoint_deserialization_authorized', 'historical_prediction_archive_opening_authorized',
        'runtime_resource_readiness_confirmed', 'collection_and_analysis_cost_charged', 'external_owned_bound_confirmed'):
        require(cfg.get(key) is True, 'Disabled pending separate root opening release: ' + key)
    require(all(cfg.get(k) is False for k in ('TEST_access', 'training', 'reselection', 'calibration', 'automatic_retry'))
        and cfg['maximum_member_forwards'] == MAX_FORWARDS and cfg['historical_archive_choice'] == 'reuse_exact_six_archives', 'Fixed inactive prediction-only scope')
    pins = read(HERE / 'SOURCE_BINDINGS.json'); g = helpers(pins)
    require(cfg['readout_manifest_sha256'] == sha(HERE / 'MANIFEST.json'), 'Exact reviewed readout manifest')
    g.verify(dict(path=str((HERE / 'MANIFEST.json').relative_to(PHASE)), sha256=cfg['readout_manifest_sha256']))
    for row in pins['source_files']:
        g.bound(row)
    for row in pins['source_manifests']:
        g.verify(row)
    family = read(g.bound(cfg['relation_family_closure']))
    require(family['schema'] == 'graph-relation-full12-normal77-family-closure-v1' and family['complete'] is True
        and family['fixed_scientific_cells'] == 12 and len(family['lane_results']) == 2
        and all(isinstance(lane, dict) and lane.get('complete') is True for lane in family['lane_results'])
        and family['source_manifest'] == pins['reuse']['relation_manifest']
        and family['controller_manifest_sha256'] == pins['reuse']['relation_controller_manifest']['sha256']
        and family['family_hard_cap_exceeded'] is False
        and all(family[k] is False for k in ('automatic_retry', 'TEST_access', 'anchor_reuse_authorized', 'comparative_opening_authorized')), 'Whole relation12 closure required before any opening')
    terminal = read(g.bound(cfg['terminal_evidence']))
    require(terminal['schema'] == 'relation18-root-saved-terminal-evidence-v2'
        and all(terminal[k] is True for k in ('complete', 'root_observed', 'all_owners_and_children_terminal'))
        and terminal['TEST_access'] is False and terminal['relation_family_closure'] == cfg['relation_family_closure']
        and terminal['relation_parent_owner'] == cfg['relation_parent_owner']
        and terminal['relation_parent_launch'] == cfg['relation_parent_launch']
        and terminal['original12_union_closure'] == cfg['original12_union_closure'], 'Whole fixed family saved root terminal evidence')
    union_cost = original_union(g, pins, cfg, terminal)
    historical_records, historical_cost = historical(g, pins, cfg, terminal)
    relation_records, relation_cost = relations(g, pins, cfg, family, terminal)
    records = relation_records + historical_records
    require(len(records) == 18 and {(r['seed'], r['condition']) for r in records}
        == {(s, c) for s in SEEDS for c in CONDITIONS}, 'Exact fixed18 bank roster')
    for key in ('train', 'development', 'polynormer'):
        g.bound(pins['data_source_pins'][key])
    for key in ('runtime_evidence', 'resource_readiness', 'external_supervision'):
        g.bound(cfg[key])
    require(type(cfg['external_active_seconds']) is int and 0 < cfg['external_active_seconds'] <= 3600
        and cfg['external_cleanup_seconds'] == 10 and cfg['external_hard_seconds'] == cfg['external_active_seconds'] + 10
        and 0 < cfg['owned_GPU_cap_bytes'] <= 32 * 1024**3
        and cfg['minimum_fresh_GPU_free_bytes'] >= cfg['owned_GPU_cap_bytes'] + 2 * 1024**3
        and 0 < cfg['maximum_output_bytes'] <= 4 * 1024**3, 'Finite root-owned readout envelope')
    supervision = read(g.bound(cfg['external_supervision']))
    entry = HERE / 'collect_relation18.py'; owner_pins = pins['relation_controller_pins']
    require(supervision['schema'] == 'relation18-existing-finite-owner-binding-v1'
        and all(supervision[k] is True for k in ('enabled', 'root_execution_authorized', 'finite_owned_bound_confirmed',
            'root_retains_actual_exit_reap_cleanup_transfer_cost'))
        and supervision['owned_entry_program'] == g.binding(entry)
        and supervision['existing_run_fit_helper'] == owner_pins['run_fit_helper']
        and supervision['existing_ownership_helper'] == owner_pins['ownership_helper']
        and supervision['argv_prefix'] == [pins['runtime']['python'], '-B', str(entry)]
        and supervision['release_argument_path'] == str(Path(release_path).resolve())
        and all(supervision[k] == cfg[k] for k in ('physical_gpu_uuid', 'external_active_seconds', 'external_cleanup_seconds',
            'external_hard_seconds', 'owned_GPU_cap_bytes', 'maximum_output_bytes'))
        and 0 < supervision['owned_RSS_cap_bytes'] <= 32 * 1024**3
        and 0 < supervision['combined_child_log_cap_bytes'] <= 8 * 1024**2
        and supervision['automatic_retry'] is False and supervision['TEST_access'] is False, 'Exact reviewed existing finite owner/command and root cost custody')
    g.bound(supervision['existing_run_fit_helper']); g.bound(supervision['existing_ownership_helper'])
    output = g.inside(cfg['output_directory'])
    frozen = {Path(r['path']).parts[0] for r in pins['source_files']}
    frozen.update((HERE.name, pins['union_pins']['original_activation'], pins['union_pins']['unused2_output'],
        Path(cfg['relation_family_closure']['path']).parts[0]))
    frozen.update(Path(r['selected_checkpoint']['path']).parts[0] for r in records)
    frozen.update(Path(r['original_archive']['path']).parts[0] for r in historical_records)
    frozen.update(Path(r['staged_archive']['path']).parts[0] for r in historical_records)
    require(not output.exists() and output.parent.is_dir()
        and not any(output.is_relative_to(g.inside(root)) for root in frozen), 'Fresh server-only output outside all sources and histories')
    return cfg, pins, records, output, g, dict(relation_training_selection_owner=relation_cost,
        original12_training_selection_owner_and_failed_admission=union_cost, historical_training_selection_resource_collection=historical_cost)
