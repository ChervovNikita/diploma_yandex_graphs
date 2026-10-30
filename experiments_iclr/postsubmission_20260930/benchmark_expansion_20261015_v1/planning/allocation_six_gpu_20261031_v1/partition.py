"""Repartition untouched task groups after all prior owners acknowledge STOP."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
E = HERE.parent.parent
sys.path.insert(0, str(E / 'campaign_execution_20261016_v7'))
import common as c
from host_extension import SHARDS, ROOT, guard, pin

PLAN = E / 'campaign_execution_20261016_v7/PLAN-allocation_gpu0.json'
PLAN_SHA = '19fa1a8b96bcf3383965c266c4fa900e0ac09c62ca040553e7669acb659ffaad'
TERMINAL = {'complete', 'no_selected_state', 'blocked_dependency_no_selected_state'}
PREVIOUS = HERE.parent / 'allocation_gpu1_gpu6_20261030_v1'
PREVIOUS_SHARDS = ('allocation/gpu0', 'anogena/gpu1', 'anogena/gpu6')
PREVIOUS_PARTITION_SHA = '33e12649977306e2c3c92f0155770ead8373874e8a3ed9ff7ccdca070c051b39'
PREVIOUS_SOURCE_SHA = 'a2661fc1b9e983343218f5de44c759da97dd278a053acfa6b714a12a92c95148'


def source(expected=None):
    receipt = pin(HERE / 'MANIFEST.json')
    if expected:
        c.require(receipt['sha256'] == expected, 'Additive source manifest changed')
    for row in c.read(HERE / 'MANIFEST.json')['files']:
        path = (HERE / row['path']).resolve()
        c.require(path.is_relative_to(HERE), 'Additive source escapes package')
        c.require(all(pin(path)[key] == row[key] for key in ('bytes', 'sha256')), 'Additive source file changed')
    return receipt


def original_plan():
    c.require(pin(PLAN)['sha256'] == PLAN_SHA, 'Original allocation plan changed')
    plan = c.read(PLAN)
    c.require(plan['worker'] == 'allocation/gpu0' and plan['project_root'] == str(ROOT), 'Wrong original assignment')
    c.require(len(plan['tasks']) == len({x['id'] for x in plan['tasks']}) == 2244, 'Original task IDs differ')
    return plan


def live(identity):
    try:
        if not c.same_process(identity):
            return False
        raw = (Path('/proc') / str(identity['pid']) / 'stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        c.require(len(fields) >= 20, 'Malformed original owner process state')
        return int(fields[19]) == identity['start_ticks'] and fields[0] not in ('Z', 'X')
    except (FileNotFoundError, ProcessLookupError):
        return False


def form_shards(tasks, states, costs, prior_owners):
    """Greedy cost balance; each four-member bank and its members stay together."""
    index = {row['id']: row for row in tasks}
    groups = []
    claimed = set()
    for task in tasks:
        if task['kind'] == 'aggregate_I4_exports':
            group = [task['id'], *task['dependencies']]
            c.require(len(group) == 5 and not (set(group) & claimed), 'Overlapping/malformed I4 group')
            c.require(set(group) <= set(index), 'Missing I4 dependency')
            groups.append(group); claimed.update(group)
    groups.extend([[task['id']] for task in tasks if task['id'] not in claimed])
    def weight(task_id):
        if states[task_id]['status'] in TERMINAL:
            return 0.0
        task = index[task_id]
        return 1.0 if task['kind'] != 'fit' else costs.get(task['cell_id'], 3600.0)
    groups.sort(key=lambda group: (-sum(weight(key) for key in group), min(group)))
    assignment = {shard: [] for shard in SHARDS}
    load = {shard: 0.0 for shard in SHARDS}
    # Custody of any previously attempted or live group stays on its original
    # host. Only entirely untouched groups are eligible for the new GPUs.
    fixed = [group for group in groups if any(states[key]['status'] != 'pending' for key in group)]
    movable = [group for group in groups if group not in fixed]
    for group in fixed + movable:
        if group in fixed:
            owners = {prior_owners[key] for key in group}
            c.require(len(owners) == 1, 'Prior I4 group had split ownership')
            shard = next(iter(owners))
        else:
            shard = min(SHARDS, key=lambda key: (load[key], len(assignment[key]), SHARDS.index(key)))
        assignment[shard].extend(group)
        load[shard] += sum(weight(key) for key in group)
    order = {row['id']: i for i, row in enumerate(tasks)}
    assignment = {key: sorted(value, key=order.__getitem__) for key, value in assignment.items()}
    flattened = [key for value in assignment.values() for key in value]
    c.require(len(flattened) == len(set(flattened)) == len(tasks) and set(flattened) == set(index),
              'Partition must be an exact disjoint union')
    for ids in assignment.values():
        c.require(all(set(index[key]['dependencies']) <= set(ids) for key in ids), 'Cross-shard dependency')
    return assignment, load


def validate_partition(value, plan):
    c.require(value['original_plan'] == pin(PLAN), 'Partition original plan differs')
    previous, owners = previous_partition()
    c.require(value['previous_partition'] == pin(PREVIOUS / 'PARTITION.json') and value['prior_owners'] == owners, 'Prior custody differs')
    assignment = value['assignment']
    c.require(set(assignment) == set(SHARDS), 'Wrong shard set')
    index = {row['id']: row for row in plan['tasks']}
    flattened = [key for ids in assignment.values() for key in ids]
    c.require(len(flattened) == len(set(flattened)) == len(index) and set(flattened) == set(index),
              'Partition is not an exact disjoint union of the original assignment')
    c.require(set(value['original_task_states']) == set(index), 'Partition task snapshot differs')
    for shard, ids in assignment.items():
        own = set(ids)
        c.require(all(set(index[key]['dependencies']) <= own for key in ids), 'Partition splits an I4 dependency')
        for key in ids:
            if value['original_task_states'][key]['status'] != 'pending':
                c.require(shard == value['prior_owners'][key], 'Attempted task moved from its prior physical shard')
    return value


def publish_once(value):
    temporary = HERE / ('partition-publication-' + str(os.getpid()) + '.json')
    c.require(not temporary.exists(), 'Unexpected prior partition publication')
    c.write(temporary, value)
    try:
        os.link(temporary, HERE / 'PARTITION.json')
    finally:
        temporary.unlink()
    fd = os.open(HERE, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def previous_partition():
    c.require(pin(PREVIOUS / 'PARTITION.json')['sha256'] == PREVIOUS_PARTITION_SHA, 'Prior partition changed')
    c.require(pin(PREVIOUS / 'MANIFEST.json')['sha256'] == PREVIOUS_SOURCE_SHA, 'Prior scheduler source changed')
    for row in c.read(PREVIOUS / 'MANIFEST.json')['files']:
        c.require(all(pin(PREVIOUS / row['path'])[key] == row[key] for key in ('bytes', 'sha256')), 'Prior scheduler file changed')
    previous = c.read(PREVIOUS / 'PARTITION.json')
    owners = {key: shard for shard, ids in previous['assignment'].items() for key in ids}
    c.require(set(previous['assignment']) == set(PREVIOUS_SHARDS) and len(owners) == 2244, 'Prior shard inventory differs')
    return previous, owners


def stop_pins():
    previous, _ = previous_partition()
    plan = original_plan()
    original = pin(Path(plan['state_root']) / 'STOP')
    c.require(original == previous['original_stop'], 'Original 16 October scheduler STOP must remain immutable')
    result = {'original_execution7': original}
    for shard in PREVIOUS_SHARDS:
        path = PREVIOUS / 'runs' / shard.replace('/', '_') / 'STOP'
        ack_path = HERE / ('ACK-' + shard.replace('/', '_') + '.json')
        c.require(ack_path.exists(), 'Prior owner acknowledgement is not ready: ' + shard)
        ack = c.read(ack_path)
        c.require(ack['worker'] == shard and ack['status'] == 'PASS' and ack['stop'] == pin(path),
                  'Prior owner acknowledgement or permanent STOP differs')
        c.require(ack['prior_source_sha256'] == PREVIOUS_SOURCE_SHA and ack['source'] == source(),
                  'Acknowledgement source differs')
        result[shard] = pin(path)
    return result


def freeze():
    """All acknowledgements are generated on their actual local PID namespace."""
    guard()
    if not all((HERE / ('ACK-' + key.replace('/', '_') + '.json')).exists() for key in PREVIOUS_SHARDS):
        return None
    plan = original_plan(); previous, owners = previous_partition()
    handles = []
    try:
        # Both historical admission ledgers are locked while custody is frozen.
        # Native children keep training; their terminal transitions remain valid.
        paths = [HERE / 'PARTITION_PUBLICATION.lock', Path(plan['host_ledger']) / 'LEDGER.lock',
                 PREVIOUS / 'shared_disk_and_host_ledgers/LEDGER.lock']
        for path in paths:
            handle = c.lock(path)
            if handle is None:
                return None
            handles.append(handle)
        if (HERE / 'PARTITION.json').exists():
            return validate_partition(c.read(HERE / 'PARTITION.json'), plan)
        stops = stop_pins()
        root = Path(plan['state_root']); states = {}; durations = {}
        for task in plan['tasks']:
            folder = root / 'full' / c.token(task['id'])
            state = c.read(folder / 'STATE.json') if (folder / 'STATE.json').exists() else {'status': 'pending', 'attempt': 0}
            c.require(state['status'] in TERMINAL | {'pending', 'running'}, 'Resolve a failed task before repartition')
            if state['status'] == 'pending':
                c.require(not list((folder / 'attempts').glob('*')), 'Unaccounted prior attempt')
            elif state.get('attempt_result'):
                c.verify(state['attempt_result'], ROOT)
                if task['kind'] == 'fit' and state.get('terminal'):
                    terminal = c.read(c.verify(state['terminal'], ROOT))
                    if terminal.get('wall_seconds', 0) > 0:
                        durations.setdefault(task['cell_id'], []).append(terminal['wall_seconds'])
            states[task['id']] = state
        costs = {key: statistics.median(value) for key, value in durations.items()}
        assignment, estimated = form_shards(plan['tasks'], states, costs, owners)
        summary = {shard: {'total': len(ids),
                    'completed': sum(states[key]['status'] in TERMINAL for key in ids),
                    'pending': sum(states[key]['status'] == 'pending' for key in ids),
                    'running_preserved': sum(states[key]['status'] == 'running' for key in ids),
                    'estimated_serial_seconds': estimated[shard]} for shard, ids in assignment.items()}
        result = {'schema': 'allocation-six-device-redistribution-v1', 'created': time.time(),
                  'original_plan': pin(PLAN), 'previous_partition': pin(PREVIOUS / 'PARTITION.json'),
                  'source': source(), 'assignment': assignment, 'summary': summary,
                  'prior_owners': owners, 'cost_seconds_by_cell': costs, 'cost_fallback_seconds': 3600,
                  'balance_is_estimate': True, 'permanent_stops': stops, 'original_task_states': states,
                  'acknowledgements': [pin(HERE / ('ACK-' + key.replace('/', '_') + '.json')) for key in PREVIOUS_SHARDS],
                  'scientific_descriptors_changed': False, 'completed_tasks_rerun': False,
                  'server51_assignment_changed': False, 'historical_excluded_evidence_reused': False}
        validate_partition(result, plan)
        publish_once(result)
        return result
    finally:
        for handle in reversed(handles):
            handle.close()
