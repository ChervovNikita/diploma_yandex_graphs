"""Offline host shards and host-local v2 seal views; JSON metadata only.

Embedded source text authenticates exact original receipt bytes after an
offline copy. Foreign filesystem paths are origin metadata, never opened.
Checkpoints, logits and datasets are neither read nor embedded here.
"""
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
from .common import digest, inside, require, verify

TERMINAL = {'complete', 'no_selected_state', 'failed', 'launch_failed',
            'blocked_dependency_no_selected_state', 'blocked_dependency_failed'}


def snapshot(value, root):
    verify(value, root, json_metadata=True)
    return {'origin': copy.deepcopy(value), 'exact_json_text': Path(value['path']).read_bytes().decode('utf-8')}


def unpack(value):
    raw = value['exact_json_text'].encode()
    origin = value['origin']
    require(len(raw) == origin['bytes'] and hashlib.sha256(raw).hexdigest() == origin['sha256'], 'Copied shard metadata differs from original exact bytes')
    require(PurePosixPath(origin['path']).is_absolute() and origin['path'].endswith('.json'), 'Only absolute JSON origin receipts allowed')
    return json.loads(raw)


def role_semantics(role):
    """Path-independent role facts; paths remain bound separately by receipts."""
    fields = ('schema', 'family', 'role', 'preset_id', 'dataset', 'architecture', 'topology',
              'index_source', 'batch_size', 'evaluator', 'rtol', 'atol', 'calibration', 'no_TEST_driven_selection')
    result = {key: copy.deepcopy(role[key]) for key in fields if key in role}
    result['populations'] = {key: {name: copy.deepcopy(value[name]) for name in
        ('records', 'ids_sha256', 'literal_split_seed', 'positive', 'negative') if name in value}
        for key, value in role['populations'].items()}
    return result


def build_host_shard(release_pin, roster_pin, state_pins, ownership, root):
    release = verify(release_pin, root, json_metadata=True)
    roster = verify(roster_pin, root, json_metadata=True)
    host = release['host']
    owners = {row['id']: row for row in ownership['assignments']}
    own_ids = {key for key, row in owners.items() if row['worker'].split('/')[0] == host}
    tasks = {row['id']: row for row in release['tasks']}
    require(set(tasks) == set(state_pins) == own_ids and len(owners) == 6730, 'Seal every and only host-owned task')
    require({row.get('training_fit_id', row.get('training_I4_id')) for row in roster['tasks']} == set(owners), 'Frozen roster omitted a global outcome')
    metadata, outcomes, roles = {}, {}, {}
    for task_id, task in tasks.items():
        state_path = inside(Path(release['output_root']) / 'control/tasks' / digest(task_id) / 'STATE.json', root)
        require(state_pins[task_id]['path'] == str(state_path), 'State outside exact training namespace')
        state = verify(state_pins[task_id], root, json_metadata=True)
        require(state.get('status') in TERMINAL, 'All training must be terminal; pending/interrupted is not missing')
        terminal = verify(state['attempt_terminal'], root, json_metadata=True)
        require(terminal['id'] == task_id and terminal['status'] == state['status'], 'Terminal/status identity changed')
        raw_pin = terminal.get('live_native_result_receipt')
        raw = verify(raw_pin, root, json_metadata=True) if raw_pin else None
        if raw_pin:
            require(raw_pin['path'] == str(inside(Path(release['output_root']) / task['output_relative'] / 'RESULT.json', root)), 'Result outside original output')
            require(raw.get('TEST_accessed') is False, 'TRAIN metadata accessed TEST')
        selected = None
        if state['status'] == 'complete':
            require(raw is not None and raw['status'] == 'complete', 'Complete terminal has no exact result')
            if task['kind'] == 'fit':
                require(raw['job_id'] == task_id and raw['job_hash'] == task['runtime_job_hash'], 'Selected fit identity differs')
                selected = {'checkpoint': raw['selected_checkpoint'], 'epoch': raw['selected_epoch'],
                            'VALID_export': raw['selected_VALID_export'], 'runtime_binding': raw['runtime_binding']}
        outcomes[task_id] = {'status': state['status'], 'state': state_pins[task_id], 'terminal': state['attempt_terminal'],
                             'training_result': raw_pin, 'selected_state': selected, 'scientific_settings_reselected': False}
        for value in (state_pins[task_id], state['attempt_terminal'], raw_pin):
            if value:
                metadata[value['sha256']] = snapshot(value, root)
    for task in roster['tasks']:
        value = task['role_contract']
        role = verify(value, root, json_metadata=True)
        semantic = role_semantics(role)
        require(roles.setdefault(task['preset_id'], semantic) == semantic, 'Preset TEST roles differ')
    return {'schema': 'offline-terminal-host-shard-v1', 'host': host, 'training_ownership_sha256': digest(ownership),
            'training_release': snapshot(release_pin, root), 'scoring_roster': snapshot(roster_pin, root),
            'role_semantics': roles, 'outcomes': outcomes, 'metadata_by_sha256': metadata,
            'all_local_tasks_terminal': True, 'TEST_accessed': False, 'checkpoint_or_tensor_reads': 0,
            'retry_or_arm_changes_after_seal': False}


def merge_host_shards(shard_pins, ownership, global_contract, root, local_host):
    """Return a frozen-v2-compatible local seal plus global merge evidence.

    global_contract is the prepared/then-GO-bound partition of origin release,
    roster and same-decision host views. It is not a second user instruction.
    """
    require(global_contract['schema'] == 'offline-final-global-contract-v1'
            and global_contract['training_ownership_sha256'] == digest(ownership), 'Exact global partition contract required')
    owners = {row['id']: row for row in ownership['assignments']}
    required_hosts = {row['worker'].split('/')[0] for row in owners.values()}
    require(set(shard_pins) == set(global_contract['hosts']) == required_hosts, 'Every declared nonempty host shard required')
    all_outcomes, seen, local = {}, set(), None
    for host in sorted(required_hosts):
        shard = verify(shard_pins[host], root, json_metadata=True)
        require(shard['schema'] == 'offline-terminal-host-shard-v1' and shard['host'] == host
                and shard['training_ownership_sha256'] == digest(ownership)
                and shard['all_local_tasks_terminal'] is True and shard['TEST_accessed'] is False
                and shard['retry_or_arm_changes_after_seal'] is False, 'Wrong terminal host shard')
        declaration = global_contract['hosts'][host]
        release, roster = unpack(shard['training_release']), unpack(shard['scoring_roster'])
        require(shard['training_release']['origin'] == declaration['training_release']
                and shard['scoring_roster']['origin'] == declaration['scoring_roster'], 'Shard changed a globally frozen release/roster')
        own = {key for key, row in owners.items() if row['worker'].split('/')[0] == host}
        tasks = {row['id']: row for row in release['tasks']}
        require(set(tasks) == set(shard['outcomes']) == own and not (seen & own), 'Duplicated, missing or moved host outcome')
        require(shard['role_semantics'] == global_contract['role_semantics'], 'Role/provider semantics differ between hosts')
        require(release['host'] == host and release['output_root'].startswith(ownership['hosts'][host]['project_root'] + '/'), 'Foreign release root changed')
        require({row.get('training_fit_id', row.get('training_I4_id')) for row in roster['tasks']} == set(owners), 'Global roster omitted intentions')
        metadata = shard['metadata_by_sha256']
        for task_id, outcome in shard['outcomes'].items():
            require(outcome['status'] in TERMINAL and outcome['scientific_settings_reselected'] is False, 'Nonterminal/reselected outcome')
            def body(value):
                receipt = metadata[value['sha256']]
                require(receipt['origin'] == value, 'Copied receipt path/authority differs')
                return unpack(receipt)
            state, terminal = body(outcome['state']), body(outcome['terminal'])
            require(outcome['state']['path'] == str(PurePosixPath(release['output_root']) / 'control/tasks' / digest(task_id) / 'STATE.json')
                    and state['attempt_terminal'] == outcome['terminal']
                    and state['status'] == terminal['status'] == outcome['status'] and terminal['id'] == task_id, 'Copied STATE/terminal chain differs')
            require(terminal.get('live_native_result_receipt') == outcome['training_result'], 'Copied live result chain differs')
            raw = body(outcome['training_result']) if outcome['training_result'] else None
            if raw:
                require(outcome['training_result']['path'] == str(PurePosixPath(release['output_root']) / tasks[task_id]['output_relative'] / 'RESULT.json')
                        and raw.get('TEST_accessed') is False, 'Copied result role/namespace differs')
            if outcome['status'] == 'complete':
                require(raw and raw['status'] == 'complete', 'Copied complete result missing')
                if tasks[task_id]['kind'] == 'fit':
                    require(raw['job_id'] == task_id and raw['job_hash'] == tasks[task_id]['runtime_job_hash']
                            and outcome['selected_state'] == {'checkpoint': raw['selected_checkpoint'], 'epoch': raw['selected_epoch'],
                                'VALID_export': raw['selected_VALID_export'], 'runtime_binding': raw['runtime_binding']}, 'Copied selected state differs')
            all_outcomes[task_id] = copy.deepcopy(outcome)
        seen |= own
        if host == local_host:
            local = declaration
    require(seen == set(owners) and len(seen) == 6730 and local is not None, 'All 6730 terminal outcomes and local host required')
    evidence = {'schema': 'offline-global-training-seal-merge-v1', 'shards': shard_pins,
                'global_contract_hash': digest(global_contract), 'complete_partition_tasks': 6730,
                'role_semantics_hash': digest(global_contract['role_semantics']), 'live_coordinator_used': False,
                'foreign_receipts_verified_from_exact_copied_JSON': True, 'foreign_files_opened': False}
    seal = {'schema': 'all-campaign-outcomes-and-selections-seal-v2', 'training_release': local['training_release'],
            'scoring_roster': local['scoring_roster'], 'all_6730_training_tasks_terminal': True,
            'outcomes': all_outcomes, 'TEST_accessed': False, 'human_action_claimed': False,
            'selection_rule': 'each_original_own_VALID_selector_only', 'retry_or_arm_changes_after_seal': False,
            'distribution': 'authenticated offline host shards; selected state stays with immutable physical owner',
            'offline_merge': evidence}
    return seal, evidence
