"""Reuse actual training authority, then admit final-role successor metadata.

Call only after the original host guard. No model/data execution occurs here.
Prepared templates cannot pass these production checks without actual GO.
"""
from pathlib import Path
from .common import EXPANSION, HERE, digest, inside, require, verify, verify_source
from .projection import frozen_task, project_ownership


def check_host_decision(go, global_decision, contract, host, instruction_pin, template):
    require(global_decision.get('schema') == 'portable-global-human-decision-v1'
            and global_decision.get('user_explicit_GO') is True
            and global_decision.get('human_instruction_evidence') == instruction_pin,
            'One actual human campaign decision required; preparation is not authorization')
    require(go.get('decision_view_schema') == 'deterministic-host-campaign-decision-view-v1'
            and go.get('parent_global_decision_sha256') == digest(global_decision)
            and go.get('decision_host') == host
            and go.get('human_instruction_evidence') == instruction_pin
            and go.get('new_user_action_claimed') is False
            and global_decision.get('global_final_contract_sha256') == digest(contract),
            'Host view must derive from the same exact campaign decision/instruction')
    require(template.get('schema') == 'prepared-host-campaign-decision-template-v1'
            and template.get('host') == host, 'Exact host execution template required')
    normalized = dict(go, user_explicit_GO=False, human_instruction_evidence=None)
    normalized.pop('parent_global_decision_sha256')
    normalized['final_execution'] = dict(normalized['final_execution'])
    normalized['final_execution'].pop('global_contract')  # parent contract authenticates template, avoiding a hash cycle
    require(template['execution_view'] == normalized,
            'Actual host GO differs from its globally frozen execution template')


def validate_inference_resources(resources, cost, qualification, root, worker, gpu_uuid):
    """Use existing reservation dimensions without asserting an optimizer exists."""
    import math
    required = {'vram_bytes', 'ram_bytes', 'disk_bytes', 'measured', 'pair_qualified', 'profile',
                'qualified_peer_profiles', 'max_concurrent_owned_children', 'evidence',
                'cpu_slots', 'thread_count', 'cpu_evidence'}
    require(set(resources) == required, 'Explicit final inference resource envelope required; no mature-optimizer fiction')
    for field in ('vram_bytes', 'ram_bytes', 'disk_bytes', 'cpu_slots', 'thread_count'):
        require(type(resources[field]) is int and resources[field] > 0, 'Positive final resource dimension required')
    require(resources['thread_count'] <= resources['cpu_slots'] and resources['measured'] is True
            and resources['max_concurrent_owned_children'] in (1, 2) and type(resources['max_concurrent_owned_children']) is int,
            'Measured one/two-slot inference envelope required')
    require(isinstance(resources['profile'], str) and resources['profile']
            and isinstance(resources['qualified_peer_profiles'], list)
            and all(isinstance(value, str) and value for value in resources['qualified_peer_profiles'])
            and type(resources['pair_qualified']) is bool
            and (resources['max_concurrent_owned_children'] == 1 or
                 (resources['pair_qualified'] is True and resources['qualified_peer_profiles'])), 'Explicit actual compatible inference peers required')
    verify(resources['evidence'], root, json_metadata=True)
    cpu = verify(resources['cpu_evidence'], root, json_metadata=True)
    require(cpu['status'] == 'PASS' and cpu['worker'] == worker and cpu['cpu_slots'] == resources['cpu_slots']
            and cpu['thread_count'] == resources['thread_count'], 'Actual worker inference CPU/thread envelope differs')
    require(set(cost) == {'measured', 'observed_wall_seconds', 'observed_optimizer_calls', 'scope', 'evidence'}
            and cost['measured'] is True and type(cost['observed_wall_seconds']) in (int, float)
            and math.isfinite(cost['observed_wall_seconds']) and cost['observed_wall_seconds'] > 0
            and cost['observed_optimizer_calls'] == 0
            and cost['scope'] == 'VALID_inference_measurement_plus_frozen_TEST_shape_bound', 'Scoped measured inference cost required')
    verify(cost['evidence'], root, json_metadata=True)
    require(qualification['worker'] == worker and qualification['GPU_UUID'] == gpu_uuid
            and qualification['measurement_scope'] == cost['scope']
            and qualification['resources_evidence'] == resources['evidence']
            and qualification['cost_evidence'] == cost['evidence'], 'Final resource measurement identity/scope differs')
    reuse = qualification['representative_basis']
    require(reuse['kind'] in ('same_class_measurement', 'justified_conservative_class_bound')
            and isinstance(reuse['resource_class_key'], str) and reuse['resource_class_key']
            and isinstance(reuse['reuse_justification'], str) and reuse['reuse_justification']
            and reuse['seed_id_alone_is_not_a_resource_class'] is True
            and reuse['measured_cost_is_reference_not_current_task_cost'] is True,
            'Separate representative measurements from derived task bounds; seeds may share a resource class')
    measured = verify(reuse['measurement'], root, json_metadata=True)
    require(measured['status'] == 'PASS' and measured['worker'] == worker and measured['GPU_UUID'] == gpu_uuid
            and measured['TEST_accessed'] is False and measured['optimizer_updates'] == 0,
            'Representative inference measurement must be real, same-worker and VALID-only')
    for field in ('vram_bytes', 'ram_bytes', 'disk_bytes'):
        require(type(measured['measured_peak_envelope'][field]) is int
                and 0 < measured['measured_peak_envelope'][field] <= resources[field], 'Derived resource bound understates its representative measurement')
    require(reuse['frozen_TEST_shape_bound_covers_target'] is True,
            'A VALID measurement alone cannot certify a larger final role without a conservative shape bound')


def authority(spec, worker, training_contracts, config):
    """Returns original custody objects and a strict final-task recheck callback."""
    root = Path(config['project_root'])
    require(spec['schema'] == 'portable-final-reviewed-run-v1' and spec['worker'] == worker, 'Exact final run spec required')
    training_spec = verify(spec['training_run_spec'], root, json_metadata=True)
    train, train_pin, training_owner, budget, training_recheck = training_contracts.authority(training_spec, worker)
    go = verify(training_spec['go'], root, json_metadata=True)
    release = verify(spec['release'], root, json_metadata=True)
    source = verify_source(root, release['integration_source_manifest'])
    require(release['schema'] == 'portable-final-host-release-v1' and release['host'] == worker.split('/')[0]
            and release['training_release'] == train_pin and release['additional_training_fits'] == 0
            and release['automatic_retries'] == 0, 'Final release changed the original training authority')
    final_output, training_output = Path(release['output_root']), Path(train['output_root'])
    require(final_output == final_output.resolve() and final_output.is_relative_to(root)
            and not final_output.is_relative_to(training_output) and not training_output.is_relative_to(final_output),
            'Final and TRAIN output/control namespaces must be disjoint canonical project paths')
    require(go.get('final_execution') == {'source_manifest': release['integration_source_manifest'], 'release': spec['release'],
                                        'readiness': spec['readiness'], 'global_contract': spec['global_contract']},
            'Same host training+final GO must bind complete final execution closure')
    contract = verify(spec['global_contract'], root, json_metadata=True)
    decision = verify(spec['global_decision'], root, json_metadata=True)
    template = verify(contract['hosts'][release['host']]['GO_view_template'], root, json_metadata=True)
    check_host_decision(go, decision, contract, release['host'], go['human_instruction_evidence'], template)
    verify(go['human_instruction_evidence'], root)
    roster = verify(release['roster'], root, json_metadata=True)
    require(go['final_scoring']['roster'] == release['roster']
            and go['final_scoring']['training_release'] == train_pin
            and go['final_scoring']['source_manifest']['path'] == str(EXPANSION / 'final_selected_scoring_v4/MANIFEST.json'), 'Frozen v4 scorer authority differs; existing v2 role/GO/roster/seal schemas stay intact')
    ownership = project_ownership(training_owner, roster)
    require(release['final_ownership_sha256'] == digest(ownership)
            and release['training_ownership_sha256'] == digest(training_owner), 'Final permanent owner projection changed')
    expected = {row['id']: row for row in roster['tasks']}
    train_index = {row['id']: row for row in train['tasks']}
    require({row['id'] for row in release['tasks']} == {'TEST/' + key for key in train_index}, 'Final host task inventory incomplete')
    owners = {row['id']: row for row in ownership['assignments']}
    ready = verify(spec['readiness'], root, json_metadata=True)
    require(ready['schema'] == 'portable-final-readiness-v1' and ready['final_release'] == spec['release']
            and ready['final_ownership_sha256'] == digest(ownership), 'Exact final-role readiness required')
    worker_ready = ready['workers'][worker]
    require(worker_ready['status'] == 'PASS' and worker_ready['budget'] == budget, 'Final and training share exact host resource budget')
    for gate in ('phase_handoff_and_STOP', 'terminal_role_export_custody', 'qualified_python', 'final_inference_resource', 'offline_global_seal'):
        record = verify(worker_ready['gates'][gate], root, json_metadata=True)
        require(record['status'] == 'PASS' and record['worker'] == worker, 'Unqualified final execution gate: ' + gate)
    seal = verify(spec['training_seal'], root, json_metadata=True)
    require(seal['schema'] == 'all-campaign-outcomes-and-selections-seal-v2'
            and seal['training_release'] == train_pin and seal['scoring_roster'] == release['roster']
            and seal['all_6730_training_tasks_terminal'] is True and len(seal['outcomes']) == 6730
            and seal['TEST_accessed'] is False and seal['retry_or_arm_changes_after_seal'] is False,
            'Complete same-decision terminal training seal required')
    require(seal.get('offline_merge', {}).get('global_contract_hash') == digest(contract), 'Seal does not authenticate the global host partition')
    from .offline_seal import merge_host_shards
    reconstructed, _ = merge_host_shards(seal['offline_merge']['shards'], training_owner, contract, root, release['host'])
    require(reconstructed == seal, 'Derived global seal differs from authenticated host shards')
    parent_files = {row['path']: row['sha256'] for row in train['source_files']}

    def recheck(task):
        training_contracts.guard(worker)
        for key in ('release', 'training_run_spec', 'readiness', 'global_contract', 'global_decision', 'training_seal'):
            verify(spec[key], root)
        verify_source(root, release['integration_source_manifest'])
        require(task in release['tasks'] and task['worker'] == worker
                and frozen_task(task) == expected[task['id']]
                and task['runtime_job_hash'] == digest(expected[task['id']])
                and task['science_contract_hash'] == owners[task['id']]['science_contract_hash']
                and task['dependencies'] == owners[task['id']]['dependencies'], 'Changed final task/owner/science')
        original_id = task.get('training_fit_id', task.get('training_I4_id'))
        training_recheck(train_index[original_id])
        # Do not rehash checkpoints here: seal JSON and exact selected receipt
        # metadata are checked; the frozen scorer verifies tensor bytes itself.
        outcome = seal['outcomes'][original_id]
        for value in (outcome['state'], outcome['terminal'], outcome['training_result']):
            if value:
                verify(value, root, json_metadata=True)
        binding = task['binding']
        require(set(binding) == {'execution', 'resources', 'measured_cost', 'qualification'}, 'Typed final binding required')
        execution = binding['execution']
        require(execution['runtime_entrypoint']['path'] == str(HERE / 'runtime_entry.py'), 'Only fixed final observer entry allowed')
        verify(execution['runtime_entrypoint'], root)
        require(execution['python_executable'] == train_index[original_id]['binding']['execution']['python_executable']
                and execution['python_environment'] == train_index[original_id]['binding']['execution']['python_environment'], 'Final worker must use original qualified lexical environment')
        files = {**parent_files, execution['runtime_entrypoint']['path']: execution['runtime_entrypoint']['sha256']}
        training_contracts.validate_python_environment(execution, files)
        qualification = verify(binding['qualification'], root, json_metadata=True)
        require(qualification['schema'] == 'final-role-resource-qualification-v1' and qualification['status'] == 'PASS'
                and qualification['task_id'] == task['id'] and qualification['worker'] == worker
                and qualification['GPU_UUID'] == config['physical_gpu_uuid']
                and qualification['resources_sha256'] == digest(binding['resources'])
                and qualification['role_contract'] == task['role_contract']
                and qualification['no_TEST_score_or_hyperparameter_selection'] is True,
                'Actual final-role resource/compatible-pair qualification required')
        validate_inference_resources(binding['resources'], binding['measured_cost'], qualification, root, worker, config['physical_gpu_uuid'])
    for task in release['tasks']:
        if task['worker'] == worker:
            recheck(task)
    return release, ownership, budget, recheck, (train, training_owner, training_spec, seal, go)
