"""Authenticate a local shard and independently ensure its ready receipt.

Runs once after local worker acknowledgments, never inside a polling loop.
Existing shard bytes are reused; recovery never rebuilds a committed shard.
Only JSON snapshots/current scalar receipt files are examined, no tensors.
"""
from pathlib import Path
from final_scoring_integration_v1.common import digest, pin, require, verify
from final_scoring_integration_v1.offline_seal import TERMINAL, role_semantics, unpack
from .stages import same_or_create


def validate_existing_shard(shard, expected, root):
    require(shard['schema'] == 'offline-terminal-host-shard-v1'
            and shard['host'] == expected['host']
            and shard['training_ownership_sha256'] == expected['training_ownership_sha256']
            and shard['training_release']['origin'] == expected['training_release']
            and shard['scoring_roster']['origin'] == expected['scoring_roster']
            and shard['all_local_tasks_terminal'] is True and shard['TEST_accessed'] is False
            and shard['retry_or_arm_changes_after_seal'] is False, 'Existing local shard changed campaign identity or terminal scope')
    release, roster = unpack(shard['training_release']), unpack(shard['scoring_roster'])
    verify(expected['training_release'], root, json_metadata=True)
    verify(expected['scoring_roster'], root, json_metadata=True)
    tasks = {task['id']: task for task in release['tasks']}
    require(set(tasks) == set(shard['outcomes']) == set(expected['state_pins']), 'Existing local shard omitted/added a terminal outcome')
    roles = {}
    for task in roster['tasks']:
        if task['preset_id'] not in roles:
            roles[task['preset_id']] = role_semantics(verify(task['role_contract'], root, json_metadata=True))
    require(roles == shard['role_semantics'], 'Existing local shard changed frozen TEST role semantics')
    metadata = shard['metadata_by_sha256']
    def body(value):
        snapshot = metadata[value['sha256']]
        require(snapshot['origin'] == value, 'Existing shard copied receipt differs')
        copied = unpack(snapshot)
        require(copied == verify(value, root, json_metadata=True), 'Existing shard scalar evidence changed after terminal acknowledgment')
        return copied
    for task_id, task in tasks.items():
        outcome = shard['outcomes'][task_id]
        require(outcome['state'] == expected['state_pins'][task_id]
                and outcome['status'] in TERMINAL and outcome['scientific_settings_reselected'] is False,
                'Existing shard no longer matches exact local terminal acknowledgments')
        require(outcome['state']['path'] == str(Path(release['output_root']) / 'control/tasks' / digest(task_id) / 'STATE.json'), 'Existing shard state is outside exact original task')
        state, terminal = body(outcome['state']), body(outcome['terminal'])
        require(state['attempt_terminal'] == outcome['terminal'] and terminal['id'] == task_id
                and state['status'] == terminal['status'] == outcome['status']
                and terminal.get('live_native_result_receipt') == outcome['training_result'], 'Existing shard terminal chain differs')
        raw = body(outcome['training_result']) if outcome['training_result'] else None
        if raw:
            require(outcome['training_result']['path'] == str(Path(release['output_root']) / task['output_relative'] / 'RESULT.json')
                    and raw['TEST_accessed'] is False, 'Existing shard TRAIN result role/namespace differs')
        if outcome['status'] == 'complete':
            require(raw and raw['status'] == 'complete', 'Existing complete outcome lacks a complete scalar result')
            if task['kind'] == 'fit':
                require(raw['job_id'] == task_id and raw['job_hash'] == task['runtime_job_hash']
                        and outcome['selected_state'] == {'checkpoint': raw['selected_checkpoint'], 'epoch': raw['selected_epoch'],
                            'VALID_export': raw['selected_VALID_export'], 'runtime_binding': raw['runtime_binding']},
                        'Existing shard changed the original selected state')
    return True


def ensure_local_ready(shard_path, ready_path, expected, acknowledgments, global_contract_pin, root, build_shard):
    shard_path, ready_path = Path(shard_path), Path(ready_path)
    reused = shard_path.exists()
    if reused:
        shard = verify(pin(shard_path, root), root, json_metadata=True)
    else:
        shard = build_shard()
    validate_existing_shard(shard, expected, root)
    shard_pin = same_or_create(shard_path, shard, root)
    # This is deliberately outside the shard creation branch. A crash after
    # shard commit is recoverable without creating another shard or a new GO.
    ready = {'schema': 'combined-host-ready-for-offline-exchange-v1', 'host': expected['host'],
             'host_shard': shard_pin, 'worker_terminal_acks': acknowledgments,
             'global_contract': global_contract_pin, 'not_global_terminal_permission': True}
    ready_pin = same_or_create(ready_path, ready, root)
    return {'host_shard': shard_pin, 'local_ready': ready_pin, 'existing_shard_reused': reused}
