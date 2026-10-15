"""Finite metadata projection; external training dependencies remain sealed.

No task is added and no source fit is repeated. The scheduler sees zero/four
local final-task dependencies; the frozen request retains the external TRAIN
dependency unchanged. Resource bindings must come from separate admission.
"""
import copy
from .common import digest, require

WORKERS = {'allocation/gpu0', 'server77/gpu0', 'server77/gpu1'}


def project_ownership(training_owner, roster):
    require(roster['schema'] == 'frozen-final-selected-scoring-roster-v2', 'Frozen v2 roster required')
    old = {row['id']: row for row in training_owner['assignments']}
    tasks = {row['id']: row for row in roster['tasks']}
    require(len(old) == len(tasks) == 6730 and set(tasks) == {'TEST/' + key for key in old}, 'Complete unchanged global inventory required')
    rows = []
    for task_id, task in sorted(tasks.items()):
        origin = task.get('training_fit_id', task.get('training_I4_id'))
        parent = old[origin]
        deps = [] if task['kind'] == 'selected_fit_TEST' else task['dependencies']
        require(parent['worker'] in WORKERS and deps == (['TEST/' + x for x in parent['dependencies']] if parent['dependencies'] else []), 'Final I4 owner/dependencies changed')
        rows.append({'id': task_id, 'worker': parent['worker'], 'training_task_id': origin,
                     'science_contract_hash': digest({'stage': 'frozen_own_selection_TEST', 'training_science_contract_hash': parent['science_contract_hash'], 'selection_rule': task['selection_rule']}),
                     'dependencies': deps, 'output_relative': task['output_relative'], 'cohort': parent['cohort']})
    return {'schema': 'portable-final-global-ownership-v1', 'campaign': training_owner['campaign'] + '/final_selected_v2',
            'training_ownership_sha256': digest(training_owner), 'hosts': copy.deepcopy(training_owner['hosts']),
            'assignments': rows, 'physical_training_fits': 0, 'reassignment': False}


def project_release(roster_pin, roster, training_release_pin, training_release, training_owner, bindings, *, source_manifest):
    """Build a host release only after exact per-task scoring bindings exist.

    A prepared release is not GO. Binding validation, actual host admission,
    same host GO view, complete global seal and final readiness are mandatory.
    """
    ownership = project_ownership(training_owner, roster)
    owners = {row['id']: row for row in ownership['assignments']}
    host = training_release['host']
    own = {row['id']: row for row in training_release['tasks']}
    required = {'TEST/' + key for key in own}
    require(set(bindings) == required, 'Do not omit resource-unqualified scoring intentions')
    tasks = []
    for original in roster['tasks']:
        if original['id'] not in required:
            continue
        assignment = owners[original['id']]
        require(assignment['worker'].split('/')[0] == host, 'Foreign host task')
        task = copy.deepcopy(original)
        task['frozen_external_dependencies'] = task.pop('dependencies')
        task.update(worker=assignment['worker'], dependencies=assignment['dependencies'],
                    science_contract_hash=assignment['science_contract_hash'],
                    runtime_job_hash=digest(original), binding=copy.deepcopy(bindings[task['id']]))
        tasks.append(task)
    require(len(tasks) == len(required), 'Host final projection incomplete')
    return {'schema': 'portable-final-host-release-v1', 'host': host, 'tasks': tasks,
            'training_release': training_release_pin, 'training_ownership_sha256': digest(training_owner),
            'final_ownership_sha256': digest(ownership), 'roster': roster_pin,
            'output_root': roster['output_root'], 'integration_source_manifest': source_manifest,
            'ordinary_failure_policy': 'retain_failure_continue_other_admissible_tasks',
            'additional_training_fits': 0, 'automatic_retries': 0, 'launch_permission': False}


def frozen_task(projected):
    result = copy.deepcopy(projected)
    result['dependencies'] = result.pop('frozen_external_dependencies')
    for key in ('worker', 'science_contract_hash', 'runtime_job_hash', 'binding'):
        result.pop(key)
    return result


def build_request(task, release, profile_pin, training_seal_pin, seal, campaign_scoring_authority):
    """Deterministic post-seal request; this never performs selection or scoring."""
    original = frozen_task(task)
    origin = original.get('training_fit_id', original.get('training_I4_id'))
    request = {'schema': 'reservation-owned-final-request-v1', 'task_id': task['id'], 'worker': task['worker'],
               'task': original, 'profile': profile_pin, 'training_seal': training_seal_pin,
               'campaign_scoring_authority': campaign_scoring_authority,
               'output': str(__import__('pathlib').Path(release['output_root']) / task['output_relative'])}
    require(origin in seal['outcomes'], 'Training outcome absent from all-terminal seal')
    if task['kind'] == 'selected_fit_TEST':
        request['training_result'] = seal['outcomes'][origin]['training_result']
    return request
