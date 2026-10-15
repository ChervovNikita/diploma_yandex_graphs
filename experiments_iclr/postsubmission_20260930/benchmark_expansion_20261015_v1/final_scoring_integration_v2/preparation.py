"""One supervisor validation receipt; bounded per-task child authority.

The reservation-owning backend attaches the capsule receipt to its actual
startup row while holding HOST.lock. The committed child validates only that
capsule, its own selected scalar chain and at most four final I4 dependencies.
"""
import copy
from pathlib import Path
from .common import digest, exclusive, pin, require, verify
from .projection import build_request


def prepare_worker(spec_pin, spec, release, training, ownership, config_pin, root):
    value = {'schema': 'fully-validated-final-supervisor-preparation-v1', 'worker': spec['worker'],
             'run_spec': spec_pin, 'release': spec['release'], 'training_release': release['training_release'],
             'training_seal': spec['training_seal'], 'GO_view': spec['training_go'],
             'global_decision': spec['global_decision'], 'global_contract': spec['global_contract'],
             'source_manifest': release['integration_source_manifest'], 'host_config': config_pin,
             'final_ownership_sha256': digest(ownership), 'all_campaign_checks_performed_by_supervisor': True,
             'new_user_action_claimed': False, 'new_training_fits': 0}
    path = Path(release['output_root']) / 'control/preparations' / digest(spec['worker']) / 'PREPARATION.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(verify(pin(path, root), root, json_metadata=True) == value, 'Preserve changed supervisor preparation')
    else:
        exclusive(path, value)
    return pin(path, root)


def make_capsule(task, output_root, final_index, training_index, seal, profile_pin, preparation_pin):
    origin = task.get('training_fit_id', task.get('training_I4_id'))
    return {'schema': 'reservation-owned-final-task-capsule-v1', 'preparation': preparation_pin,
            'task': copy.deepcopy(task), 'training_task': copy.deepcopy(training_index[origin]),
            'training_outcome': copy.deepcopy(seal['outcomes'][origin]), 'profile': profile_pin,
            'output_root': output_root, 'dependency_tasks': [copy.deepcopy(final_index[key]) for key in task['dependencies']]}


def validate_capsule(preparation, capsule, request, go):
    """Pure bounded metadata check; never iterates campaign tasks/outcomes."""
    task, training = capsule['task'], capsule['training_task']
    require(preparation['schema'] == 'fully-validated-final-supervisor-preparation-v1'
            and preparation['all_campaign_checks_performed_by_supervisor'] is True
            and capsule['schema'] == 'reservation-owned-final-task-capsule-v1'
            and preparation['worker'] == task['worker'] == request['worker']
            and capsule['preparation']['sha256'] is not None, 'Wrong supervisor/capsule worker authority')
    origin = task.get('training_fit_id', task.get('training_I4_id'))
    require(training['id'] == origin and training['worker'] == task['worker'], 'Final task moved from its training physical owner')
    require(len(capsule['dependency_tasks']) == len(task['dependencies']) in (0, 4)
            and [row['id'] for row in capsule['dependency_tasks']] == task['dependencies']
            and all(row['worker'] == task['worker'] for row in capsule['dependency_tasks']), 'I4 dependencies moved/duplicated')
    release_view = {'output_root': capsule['output_root']}
    seal_view = {'outcomes': {origin: capsule['training_outcome']}}
    expected = build_request(task, release_view, capsule['profile'], preparation['training_seal'], seal_view, go['final_scoring'])
    if task['kind'] == 'I4_TEST_aggregate':
        require(len(request['component_results']) == 4, 'I4 needs its exact four result authorities')
        expected['component_results'] = request['component_results']
        for row, result in zip(capsule['dependency_tasks'], request['component_results']):
            require(result['path'] == str(Path(capsule['output_root']) / row['output_relative'] / 'RESULT.json'), 'I4 result moved from own final task')
    require(request == expected, 'Child request differs from the exact reservation capsule')
    return {'training_tasks_checked': 1, 'final_dependency_tasks_checked': len(task['dependencies']),
            'campaign_inventory_iterations': 0}


def verify_own_scalar_chain(capsule, root, verifier=verify):
    """No checkpoint/logit reads; own outcome metadata only."""
    count = 0
    for field in ('state', 'terminal', 'training_result'):
        value = capsule['training_outcome'][field]
        if value:
            verifier(value, root, json_metadata=True)
            count += 1
    return count
