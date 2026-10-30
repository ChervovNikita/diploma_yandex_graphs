"""Freeze disjoint task custody only after the original allocation fully drains."""
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


def form_shards(tasks, states, costs):
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
        shard = SHARDS[0] if group in fixed else min(
            SHARDS, key=lambda key: (load[key], len(assignment[key]), SHARDS.index(key)))
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
        if shard != SHARDS[0]:
            c.require(all(value['original_task_states'][key]['status'] == 'pending' for key in ids),
                      'Previously attempted task reassigned to a new device')
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


def admission_acknowledgement(plan, stop):
    """Prove the unchanged supervisor cannot admit another old-queue task.

    Caller owns its LEDGER.lock. Every start(), including its 0.2s probe sleep,
    holds that lock. With phase=full, its only sleep outside start is the loop's
    final ten-second sleep. An observed hrtimer_nanosleep therefore follows the
    completed admission scan, and its next scan reads the preserved STOP file.
    No signal, debugger injection, source edit or scientific interruption occurs.
    """
    root = Path(plan['state_root'])
    state = c.read(root / 'STATUS.json')
    if not live(stop['owner']):
        if state['phase'] == 'stopped' and not state['active_fit_ids']:
            return {'mode': 'fully_drained', 'status': pin(root / 'STATUS.json')}
        return None
    c.require(state['owner'] == stop['owner'] and state['phase'] == 'full', 'Unexpected original owner/phase')
    try:
        wait_channel = (Path('/proc') / str(stop['owner']['pid']) / 'wchan').read_text().strip()
    except (FileNotFoundError, ProcessLookupError):
        return None
    if wait_channel != 'hrtimer_nanosleep':
        return None
    if not live(stop['owner']):
        return None
    c.require(c.read(root / 'STOP') == stop, 'Stop marker changed during admission acknowledgement')
    return {'mode': 'original_supervisor_asleep_outside_ledger_protected_start',
            'owner': stop['owner'], 'wait_channel': wait_channel,
            'ledger_lock_held': str(Path(plan['host_ledger']) / 'LEDGER.lock'),
            'observed': time.time(), 'status': pin(root / 'STATUS.json')}


def freeze():
    """Freeze after acknowledged STOP while preserving original live groups."""
    guard()
    c.require(__import__('socket').gethostname() == 'anogena-2-0', 'Freeze must observe original host-local owner')
    if (HERE / 'PARTITION.json').exists():
        return c.read(HERE / 'PARTITION.json')
    plan = original_plan(); root = Path(plan['state_root'])
    stop = c.read(root / 'STOP')
    c.require(stop == c.read(HERE / 'STOP_REQUEST.json'), 'Unrelated stop request')
    handles = []
    try:
        handle = c.lock(Path(plan['host_ledger']) / 'LEDGER.lock')
        if handle is None:
            return None
        handles.append(handle)
        if (HERE / 'PARTITION.json').exists():
            return validate_partition(c.read(HERE / 'PARTITION.json'), plan)
        ack = admission_acknowledgement(plan, stop)
        if ack is None:
            return None
        states = {}; durations = {}
        for task in plan['tasks']:
            folder = root / 'full' / c.token(task['id'])
            state = c.read(folder / 'STATE.json') if (folder / 'STATE.json').exists() else {'status': 'pending', 'attempt': 0}
            c.require(state['status'] in TERMINAL | {'pending', 'running'}, 'Resolve current failure before repartition')
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
        assignment, estimated = form_shards(plan['tasks'], states, costs)
        summary = {}
        for shard, ids in assignment.items():
            summary[shard] = {'total': len(ids), 'completed': sum(states[key]['status'] in TERMINAL for key in ids),
                              'pending': sum(states[key]['status'] == 'pending' for key in ids),
                              'running_preserved': sum(states[key]['status'] == 'running' for key in ids),
                              'estimated_serial_seconds': estimated[shard]}
        result = {'schema': 'allocation-remaining-task-redistribution-v1', 'created': time.time(),
                  'original_plan': pin(PLAN), 'source': source(), 'assignment': assignment,
                  'summary': summary, 'cost_seconds_by_cell': costs,
                  'cost_fallback_seconds': 3600, 'balance_is_estimate': True,
                  'old_owner': stop['owner'], 'admission_stop_acknowledgement': ack,
                  'original_stop': pin(root / 'STOP'), 'original_task_states': states,
                  'scientific_descriptors_changed': False, 'completed_tasks_rerun': False,
                  'server51_assignment_changed': False, 'historical_excluded_evidence_reused': False}
        validate_partition(result, plan)
        publish_once(result)
        return result
    finally:
        for handle in reversed(handles):
            handle.close()
