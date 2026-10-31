"""Finite final-serving metadata contracts; imports no model or data library."""
import copy
import hashlib
import json
from pathlib import Path

FAMILIES = {'node', 'ogb', 'link'}
MEAN_METHODS = {
    'native_single', 'gnnm_boundary_native', 'boundary_private_half_pool',
    'boundary_private_fixed_responsibility', 'boundary_own_within_route_supcon',
    'task_geometry_alignment', 'task_geometry_alignment_variance',
    'link_same_query_cosine', 'link_same_query_global', 'link_same_query_endpoint',
    'structural_class_quartile_private_v1', 'structural_class_quartile_permuted_private_v1',
    'link_endpoint_responsibility_v2', 'link_query_responsibility_v2',
    'context_affinity_common', 'context_affinity_split',
    'portable_input_residual_be_v1', 'portable_input_residual_only_v1',
    'gnnm_private_bn_state_v3'}
AGGREGATION_METHODS = {'aggregation_mean_globalJ_v1', 'aggregation_global_globalJ_v1',
                       'aggregation_conditional_globalJ_v1'}
INTERACTION_METHODS = {'interaction_geometry_input', 'interaction_responsibility_private_pool',
                       'interaction_input_meanJ', 'interaction_input_conditionalJ'}
RUNTIME_ALIASES = {
    'boundary_global_half_pool': 'aggregation_mean_globalJ_v1',
    'task_graph_class_alignment_v1': 'task_geometry_alignment',
    'task_node_class_alignment_v1': 'task_geometry_alignment',
    'task_node_class_alignment_variance_v1': 'task_geometry_alignment_variance',
    'link_endpoint_responsibility_private_v2': 'link_endpoint_responsibility_v2',
    'link_query_iid_responsibility_private_v2': 'link_query_responsibility_v2',
    'link-same-query-cosine-private': 'link_same_query_cosine',
    'link-endpoint-contrast-private': 'link_same_query_endpoint',
    'link-global-opposite-label-private': 'link_same_query_global',
    **{'context_affinity_' + task + '_' + mode + version: 'context_affinity_' + mode
       for task, version in (('BUDDY', '_v1'), ('link', '_v2'), ('molecular', '_v1'),
                             ('ncnc', '_v1'), ('node', '_v1')) for mode in ('common', 'split')}}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    require(path.is_relative_to(root), 'Final-serving path escapes the project')
    return path


def receipt(path, root):
    path = inside(path, root)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def verify(pin, root, *, json_metadata=False):
    require(set(pin) == {'path', 'bytes', 'sha256'}, 'Exact path/bytes/SHA receipt required')
    require(receipt(pin['path'], root) == pin, 'Final-serving authority bytes changed')
    return json.loads(Path(pin['path']).read_text()) if json_metadata else Path(pin['path'])


def readout_policy(procedure, family):
    require(family in FAMILIES, 'Unknown native family')
    if procedure == 'boundary_global_half_pool':
        return {'members': 4, 'kind': 'mean', 'gate_leaf': None}
    procedure = RUNTIME_ALIASES.get(procedure, procedure)
    if procedure in MEAN_METHODS:
        return {'members': 1 if procedure == 'native_single' else 4,
                'kind': 'native' if procedure == 'native_single' else 'mean', 'gate_leaf': None}
    if procedure in AGGREGATION_METHODS:
        return {'members': 4, 'kind': 'restored_gate',
                'gate_leaf': 'conditional_aggregation' if family != 'link' else '_expanded_{architecture}_aggregation'}
    if procedure in INTERACTION_METHODS:
        # Node/molecular interactions all register their exact readout, including
        # parameter-free means. Link interactions register a gate only for C4.
        if family == 'link' and procedure != 'interaction_input_conditionalJ':
            return {'members': 4, 'kind': 'mean', 'gate_leaf': None}
        return {'members': 4, 'kind': 'restored_gate',
                'gate_leaf': '_interaction_readout' if family == 'link' else 'interaction_readout'}
    raise ValueError('Unclassified serving method; never substitute a uniform mean: ' + procedure)


def validate_role_contract(value):
    require(value.get('schema') == 'native-final-TEST-role-contract-v2'
            and value.get('family') in FAMILIES and value.get('role') == 'TEST', 'Typed final TEST role required')
    require(value.get('no_TEST_driven_selection') is True and value.get('calibration') == 'none',
            'TEST cannot select, tune or calibrate')
    for name in ('rtol', 'atol'):
        require(type(value.get(name)) in (int, float) and 0 <= value[name] < 1,
                'Frozen finite VALID reconstruction tolerance required')
    populations = value.get('populations', {})
    require(set(populations) == ({str(i) for i in range(10)} if value['family'] == 'node' else {'0'}),
            'Keep each literal node split population separately; molecular/link use their official split0')
    for population in populations.values():
        require(type(population.get('records')) is int and population['records'] > 0
                and isinstance(population.get('ids_sha256'), str) and len(population['ids_sha256']) == 64
                and isinstance(population.get('provider_evidence'), dict), 'Complete explicit official TEST population required')
    if value['family'] == 'node':
        require(value.get('topology') == 'unchanged_native_transductive_graph'
                and value.get('index_source') == 'same_literal_split_test_mask', 'Native node TEST topology/mask required')
    elif value['family'] == 'ogb':
        require(value.get('topology') == 'official_scaffold_TEST_graphs'
                and value.get('batch_size') == 32 and 'test_indices' in value, 'Native molecular TEST roles/batching required')
    else:
        population = populations['0']
        architecture = value.get('architecture')
        require(architecture in ('NCN', 'NCNC', 'BUDDY'), 'Explicit native link architecture required')
        require(value.get('topology') == ('weighted_year2007_TRAIN_plus_weighted_VALID' if architecture == 'BUDDY'
                                        else 'unweighted_TRAIN_plus_VALID'), 'Source-specific TEST topology required')
        require(value.get('evaluator') == {'name': 'ogbl-collab', 'metric': 'hits@50', 'K': 50}, 'Native Hits@50 required')
        require(type(population.get('positive')) is int and population['positive'] > 0
                and type(population.get('negative')) is int and population['negative'] > 0
                and population['positive'] + population['negative'] == population['records']
                and set(value.get('data_files', {})) == {'graph', 'train', 'valid', 'test'},
                'Whole official link queries and role files required')
    return value


def selected_role_contract(value, identity):
    validate_role_contract(value)
    selected = copy.deepcopy(value)
    key = str(identity['split_index'])
    require(key in value['populations'], 'Selected fit has no frozen TEST split population')
    selected['population'] = selected.pop('populations')[key]
    selected['split_index'] = identity['split_index']
    require(selected['preset_id'] == identity['preset_id'] and selected['dataset'] == identity['dataset'],
            'Role dataset/preset differs from the selected fit')
    if selected['family'] == 'node':
        require(selected['population']['literal_split_seed'] == identity['literal_split_seed'],
                'TEST mask belongs to another literal source split')
    return selected


def compile_roster(registry_index_pin, profiles, role_contract_pins, role_contracts, output_root):
    """Pure metadata compilation; caller supplies already pinned/loaded inputs."""
    require(len(profiles) == len(role_contract_pins) == len(role_contracts) == 27,
            'Keep every adopted preset in final scoring')
    tasks, fit_ids = [], set()
    count = {'own_selected_fit_serving': 0, 'I4_TEST_aggregations': 0,
             'same_bank_mean_diagnostics': 0, 'expected_predictor_rows': 0,
             'additional_training_fits': 0, 'I4_VALID_exports_already_in_training_plan': 270}
    for preset_id, profile in sorted(profiles.items()):
        role = validate_role_contract(role_contracts[preset_id])
        require(role['preset_id'] == preset_id, 'TEST role belongs to another preset')
        for fit_id, spec in profile['fit_specs'].items():
            require(fit_id not in fit_ids and spec['identity']['preset_id'] == preset_id, 'Duplicate or foreign physical fit')
            fit_ids.add(fit_id)
            readout_policy(spec['identity']['runtime_procedure_id'], role['family'])
            tasks.append({'id': 'TEST/' + fit_id, 'kind': 'selected_fit_TEST',
                          'training_fit_id': fit_id, 'preset_id': preset_id,
                          'descriptor': copy.deepcopy(spec['descriptor']), 'runtime_entrypoint': copy.deepcopy(spec['runtime_entrypoint']),
                          'job_hash': spec['job_hash'], 'role_contract': role_contract_pins[preset_id],
                          'training_output_relative': spec['output_relative'],
                          'output_relative': 'selected/' + digest(fit_id),
                          'selection_rule': 'existing_own_VALID_selected_checkpoint_only',
                          'checkpoint_receipt': 'derived_from_exact_terminal_training_RESULT',
                          'dependencies': [fit_id], 'physical_training_fits': 0})
            count['own_selected_fit_serving'] += 1
            count['same_bank_mean_diagnostics'] += 'same_bank_mean_identity' in spec
        for task_id, spec in profile['I4_specs'].items():
            require(len(spec['component_fit_ids']) == 4 and spec['component_fit_ids'][0] in profile['fit_specs'],
                    'Four exact own states and body0/M1 required')
            tasks.append({'id': 'TEST/' + task_id, 'kind': 'I4_TEST_aggregate',
                          'training_I4_id': task_id, 'preset_id': preset_id,
                          'role_contract': role_contract_pins[preset_id],
                          'component_fit_ids': spec['component_fit_ids'],
                          'dependencies': ['TEST/' + x for x in spec['component_fit_ids']],
                          'output_relative': 'I4/' + digest(task_id),
                          'selection_rule': 'four_own_VALID_selected_states', 'physical_training_fits': 0})
            count['I4_TEST_aggregations'] += 1
        count['expected_predictor_rows'] += len(profile['registry']['expected_predictors'])
    require(count == {'own_selected_fit_serving': 6460, 'I4_TEST_aggregations': 270,
                      'same_bank_mean_diagnostics': 270, 'expected_predictor_rows': 6190,
                      'additional_training_fits': 0, 'I4_VALID_exports_already_in_training_plan': 270},
            'Final-scoring inventory differs from the declared training proposal')
    return {'schema': 'frozen-final-selected-scoring-roster-v2', 'registry_index': registry_index_pin,
            'output_root': str(output_root), 'counts': count, 'tasks': tasks,
            'phase_order': ['all_frozen_training_outcomes_terminal', 'immutable_selection_seal',
                            'complete_selected_VALID_reconstruction', 'frozen_TEST_serving'],
            'live_online_coordinator_required': False,
            'TEST_reselection': False, 'calibration': 'none', 'launch_permission': False}


def authorize(request, go, root, source_manifest_pin):
    """Bind to the same real campaign GO; no per-checkpoint human ceremony."""
    require(go.get('schema') == 'portable-human-GO-v2' and go.get('user_explicit_GO') is True,
            'The pending giant campaign GO has not authorized execution')
    require(request['worker'] in go['allowed_workers'], 'Campaign GO excludes this worker')
    verify(go['human_instruction_evidence'], root)
    authority = request['campaign_scoring_authority']
    require(go.get('final_scoring') == authority and authority['source_manifest'] == source_manifest_pin,
            'The same reviewed campaign GO must include this exact frozen final scoring source/roster')
    roster = verify(authority['roster'], root, json_metadata=True)
    require(roster['schema'] == 'frozen-final-selected-scoring-roster-v2', 'Wrong final-scoring roster')
    task = next(row for row in roster['tasks'] if row['id'] == request['task_id'])
    require(request['task'] == task and request['output'] == str(inside(Path(roster['output_root']) / task['output_relative'], root)),
            'Runtime request differs from its predetermined campaign task')
    require(go['host_releases'][request['worker'].split('/')[0]] == authority['training_release'],
            'Scoring and training release authorities differ')
    seal = verify(request['training_seal'], root, json_metadata=True)
    require(seal['schema'] == 'all-campaign-outcomes-and-selections-seal-v2'
            and seal['all_6730_training_tasks_terminal'] is True
            and seal['training_release'] == authority['training_release']
            and seal['scoring_roster'] == authority['roster']
            and seal['TEST_accessed'] is False and seal['retry_or_arm_changes_after_seal'] is False,
            'Every frozen campaign outcome/selection must be sealed before final TEST scoring')
    return task
