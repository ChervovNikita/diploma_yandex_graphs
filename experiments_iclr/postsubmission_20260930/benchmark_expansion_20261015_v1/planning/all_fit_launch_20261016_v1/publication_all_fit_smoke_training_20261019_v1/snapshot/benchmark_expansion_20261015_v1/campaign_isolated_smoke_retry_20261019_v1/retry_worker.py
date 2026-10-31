"""One explicit isolated CUDA OOM retry; execution7 owns smoke/barrier/full flow."""
import argparse
import copy
import hashlib
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'campaign_execution_20261016_v7'
PARENT_SHA = '52a996e6effb77748339c57a716159a80a06272d88914d37beaa2dd680e7e890'
sys.path.insert(0, str(PARENT))
for name in ('common', 'capacity', 'worker'):
    if name in sys.modules:
        assert Path(sys.modules[name].__file__).resolve().parent == PARENT, 'Fresh interpreter required'
import common as c
import worker as parent

AUTHORIZED = {
    'schema': 'campaign-isolated-smoke-retry-authorization-v1',
    'worker': 'server51/gpu0',
    'id': 'polyformer-coauthor-physics-mono/long_convergence/split-9/seed-9/wrapper-structural_class_quartile_permuted_private_v1',
    'phase': 'smoke',
    'failed_attempt': 1,
    'permitted_attempt': 2,
    'expected_fit_count': 2153,
    'plan_sha256': 'b23301b8674faf5dfad2000efeeb067d0c1a2fb97524277c2185d174a307a345',
    'original_attempt_result_sha256': '583674b4175dc5d277117b96428a64b4c35456e8fcd5c109bd220c8d8a01881e',
    'original_native_result_sha256': 'e134cdbee9ed295880ebb2eb817f4bf57f65d7d5e1970223dc5d0120a6f64359',
    'parent_manifest_sha256': PARENT_SHA,
    'inventory_sha256': '2acb56aec9905b44bf317a43efc2484bbe59286ede41c669a0e713a1333c3e9c',
    'native_port_manifest_sha256': 'b64f793b9c097497395110e6a2f15ca4ad4ed654388ffc618a53690b9b8a6e24',
    'smoke_manifest_sha256': 'c911fef19b379a130c18718dfe1272fff146ac63fd16041d1715104f7077e4a4',
    'reason': 'User-authorized technical repair: retry retained CUDA OOM once after all other assigned smokes PASS',
    'scientific_policy_changed': False,
    'TEST_accessed': False,
}


def own_source(expected):
    source = c.pin(HERE / 'MANIFEST.json')
    c.require(source['sha256'] == expected, 'Exact retry wrapper source required')
    for row in c.read(HERE / 'MANIFEST.json')['files']:
        path = (HERE / row['path']).resolve()
        c.require(path.is_relative_to(HERE), 'Retry source escapes package')
        actual = c.pin(path)
        c.require(all(actual[key] == row[key] for key in ('bytes', 'sha256')), 'Retry source changed')
    c.require(c.source_manifest()['sha256'] == PARENT_SHA, 'Frozen execution7 helper changed')
    return source


def authorization(path, expected):
    receipt = c.pin(path)
    c.require(receipt['sha256'] == expected, 'Retry authorization hash differs')
    c.require(c.read(path) == AUTHORIZED, 'Retry authorization scope differs')
    return receipt


def preserve_owner(path):
    """Called only by the parent OWNER write after acquiring SUPERVISOR.lock."""
    if not path.exists():
        return None
    previous = path.read_bytes()
    history = path.with_name('OWNER-before-isolated-smoke-retry-' + hashlib.sha256(previous).hexdigest() + '.json')
    if history.exists():
        c.require(history.read_bytes() == previous, 'Prior owner history differs')
    else:
        with history.open('xb') as stream:
            stream.write(previous); stream.flush(); os.fsync(stream.fileno())
        fd = os.open(history.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return c.pin(history)


class IsolatedRetryWorker(parent.Worker):
    def __init__(self, args, source):
        auth = authorization(args.authorization, args.authorization_sha256)
        c.require(args.worker == AUTHORIZED['worker'], 'Retry is restricted to server51/gpu0')
        plan_pin = c.pin(args.plan)
        c.require(args.plan_sha256 == plan_pin['sha256'] == AUTHORIZED['plan_sha256'], 'Retry plan differs')
        self.scheduler_source = {
            'schema': 'campaign-isolated-smoke-retry-producer-v1',
            'actual_outer_source_manifest': source,
            'unchanged_parent_source_manifest': c.source_manifest(),
            'plan': plan_pin, 'explicit_retry_authorization': auth,
            'authorized_fit_id': AUTHORIZED['id'], 'permitted_smoke_attempt': 2,
            'global_all_6460_PASS_required': True, 'full_training_supported': True,
            'native_source_or_scientific_policy_changed': False,
        }
        c.require(self.scheduler_source['unchanged_parent_source_manifest']['sha256'] == PARENT_SHA,
                  'Frozen execution7 helper changed')
        self.retry_disposition = 'not_loaded'
        owner_path = Path(c.read(args.plan)['state_root']) / 'OWNER.json'
        original_write = parent.write

        def owner_write(path, value):
            if Path(path) == owner_path:
                c.require(self.supervisor_lease is not None and not self.supervisor_lease.closed,
                          'Previous OWNER requires supervisor lease custody')
                previous = preserve_owner(owner_path)
                value = {**value, 'actual_scheduler_producer': self.scheduler_source,
                         'previous_owner': previous}
            return original_write(path, value)

        parent.write = owner_write
        try:
            super().__init__(args)
        finally:
            parent.write = original_write

    def original_failure(self, task):
        directory = self.folder(task, 'smoke') / 'attempts' / '0001'
        attempt_path = directory / 'ATTEMPT_RESULT.json'
        native_path = directory / 'NATIVE_RESULT.json'
        c.require(c.pin(attempt_path)['sha256'] == AUTHORIZED['original_attempt_result_sha256'],
                  'Authorized original attempt receipt changed')
        c.require(c.pin(native_path)['sha256'] == AUTHORIZED['original_native_result_sha256'],
                  'Authorized original native receipt changed')
        original = c.read(attempt_path)
        native = c.read(native_path)
        for key, expected in (('schema', 'all-fit-attempt-result-v1'), ('id', AUTHORIZED['id']),
                              ('worker', AUTHORIZED['worker']), ('phase', 'smoke'),
                              ('attempt', 1), ('status', 'failed'), ('exit_code', 1)):
            c.require(original.get(key) == expected, 'Original failure identity differs: ' + key)
        c.require(c.verify(original['native_result']) == native_path.resolve(), 'Original native custody differs')
        c.require(c.verify(original['terminal']) == (directory / 'TERMINAL.json').resolve(),
                  'Original terminal custody differs')
        c.require(Path(original['output']).resolve() == (directory / 'output').resolve(),
                  'Original output custody differs')
        for key in ('id', 'worker', 'runtime_job_hash', 'canonical_runtime_job_hash', 'science_contract_hash'):
            c.require(native.get(key) == self.smokes[task['id']][key], 'Original smoke identity differs: ' + key)
        c.require(native.get('schema') == 'all-fit-two-update-smoke-result-v1' and
                  native.get('status') == 'failed' and native.get('TEST_accessed') is False,
                  'Original failed smoke scope differs')
        c.require(native.get('failure', {}).get('type') == 'OutOfMemoryError' and
                  'CUDA out of memory' in native['failure'].get('message', ''), 'Original failure was not CUDA OOM')
        c.require(native.get('TRAIN_updates') == 0 and native.get('optimizer_calls') == {'attempts': [0], 'completed': [0]},
                  'Original OOM update count differs')
        c.require(native.get('descriptor') == task['descriptor'] and native.get('inventory') == self.plan['inventory'],
                  'Original descriptor/inventory differs')
        for receipt, expected in ((self.plan['inventory'], AUTHORIZED['inventory_sha256']),
                                  (native['producer_manifest'], AUTHORIZED['native_port_manifest_sha256']),
                                  (native['source_manifest'], AUTHORIZED['smoke_manifest_sha256'])):
            c.require(receipt['sha256'] == expected, 'Frozen native source/inventory differs')
        c.require(native['source_manifest'] == self.plan['smoke_source_manifest'], 'Frozen smoke source differs')
        return original

    def load_states(self, tasks, phase):
        super().load_states(tasks, phase)
        if phase != 'smoke':
            return
        task_by_id = {task['id']: task for task in tasks}
        target = AUTHORIZED['id']
        c.require(len(tasks) == len(task_by_id) == AUTHORIZED['expected_fit_count'] and target in task_by_id,
                  'Isolated retry requires exact assigned fit coverage')
        self.original_failure(task_by_id[target])
        c.require(all(state['status'] == 'PASS' for task_id, state in self.states.items() if task_id != target),
                  'Isolated retry requires every other assigned smoke PASS')
        state = self.states[target]
        c.require(state.get('id') == target and state.get('worker') == AUTHORIZED['worker'] and
                  state.get('phase') == 'smoke' and state.get('attempt') in (1, 2), 'Latest retry identity/attempt differs')
        if state['status'] == 'PASS':
            c.require(state['attempt'] == 2, 'Original authorized failure cannot become PASS')
            self.retry_disposition = 'already_PASS'
            return
        c.require(state['status'] == 'failed', 'Unexpected retained retry state')
        if state['attempt'] == 2:
            self.retry_disposition = 'attempt_0002_failed_no_further_retry'
            return
        c.require(state['attempt_result']['sha256'] == AUTHORIZED['original_attempt_result_sha256'],
                  'Latest failure is not the authorized original')
        # Parent load_states has retained failed STATE and validated prior PASSes.
        # Only this in-memory copy becomes pending. Parent start then creates 0002
        # and changes disk STATE to running after launching the unchanged guardian.
        pending = copy.deepcopy(state)
        pending['status'] = 'pending'
        self.states[target] = pending
        self.retry_disposition = 'authorized_0002_pending_in_memory'

    def report(self, **extra):
        if time.time() - self.last_report < 10 and not extra:
            return
        return super().report(actual_scheduler_producer=self.scheduler_source,
                              isolated_retry_disposition=self.retry_disposition, **extra)

    def invocation(self, task, phase, directory):
        value = {**super().invocation(task, phase, directory),
                 'actual_scheduler_producer': self.scheduler_source}
        if phase == 'smoke':
            c.require(task['id'] == AUTHORIZED['id'] and directory.name == '0002' and
                      self.states[task['id']]['attempt'] == 1, 'Only authorized smoke attempt 0002 may start')
            value['explicit_smoke_retry'] = {
                'authorization': self.scheduler_source['explicit_retry_authorization'],
                'previous_failed_attempt': c.pin(self.folder(task, phase) / 'attempts/0001/ATTEMPT_RESULT.json'),
                'previous_native_result': c.pin(self.folder(task, phase) / 'attempts/0001/NATIVE_RESULT.json'),
                'original_failure_preserved': True, 'maximum_smoke_attempt': 2,
            }
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=c.WORKERS, required=True)
    parser.add_argument('--plan', required=True); parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--authorization', required=True); parser.add_argument('--authorization-sha256', required=True)
    parser.add_argument('--retry-failed', action='store_true', help='Original full-phase behavior only; never repeats this smoke retry')
    parser.add_argument('--poll-seconds', type=float, default=10)
    parser.add_argument('--smoke-wall-seconds', type=int, default=1800)
    args = parser.parse_args()
    c.require(args.poll_seconds >= 1, 'Poll interval must be at least one second')
    c.require(args.smoke_wall_seconds == 1800, 'Keep original smoke wall policy')
    instance = None
    try:
        source = own_source(args.source_sha256)
        instance = IsolatedRetryWorker(args, source)
        return instance.run()
    except BaseException as exc:
        if instance is not None:
            instance.phase = 'supervisor_error'
            instance.report(error=type(exc).__name__ + ': ' + str(exc), children_retain_inherited_leases=True)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
