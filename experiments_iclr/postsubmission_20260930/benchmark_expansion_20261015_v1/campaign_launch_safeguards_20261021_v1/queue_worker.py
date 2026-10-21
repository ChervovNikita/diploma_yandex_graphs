"""Execution7 queue with verified I4 exports and a 20 GiB launch disk floor."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
MEMORY = HERE.parent / 'campaign_memory_admission_20261019_v1'
MEMORY_SHA = '6905ecec8b840925d2c3e03fb9dd43f06083372a05077579082f468a93aa4c31'
sys.path.insert(0, str(MEMORY))
if 'admission_worker' in sys.modules:
    assert Path(sys.modules['admission_worker'].__file__).resolve().parent == MEMORY, 'Fresh interpreter required'
import admission_worker as memory

c = memory.c
parent = memory.parent
DISK_FLOOR_BYTES = 20 * c.GiB


def own_source(expected):
    source = c.pin(HERE / 'MANIFEST.json')
    c.require(source['sha256'] == expected, 'Exact launch safeguards source required')
    for row in c.read(HERE / 'MANIFEST.json')['files']:
        path = (HERE / row['path']).resolve()
        c.require(path.is_relative_to(HERE), 'Launch safeguards source escapes package')
        actual = c.pin(path)
        c.require(all(actual[key] == row[key] for key in ('bytes', 'sha256')), 'Launch safeguards source changed')
    memory.own_source(MEMORY_SHA)
    return source


def capacity_decision(candidate, active, live, can_pair):
    """Retain existing admission checks and strengthen only its launch disk floor."""
    decision = memory.capacity.capacity_decision(candidate, active, live, can_pair)
    required = DISK_FLOOR_BYTES + candidate['reservation']['disk_bytes'] + decision['disk_reserved']
    reasons = list(decision['reasons'])
    if live['free_disk'] < required and 'disk capacity' not in reasons:
        reasons.append('disk capacity')
    return {**decision, 'admitted': not reasons, 'reasons': reasons,
            'disk_floor_bytes': DISK_FLOOR_BYTES, 'disk_required_free_bytes': required,
            'disk_policy': '20-GiB-floor-plus-active-and-candidate-reservations',
            'disk_policy_scope': 'new-task admission only',
            'whole_campaign_storage_reserved': False,
            'checkpoint_boundary_checks': False}


def preserve_owner(path):
    """Fsync exact previous OWNER bytes before its replacement under the lease."""
    if not path.exists():
        return None
    previous = path.read_bytes()
    history = path.with_name('OWNER-before-launch-safeguards-' + hashlib.sha256(previous).hexdigest() + '.json')
    if history.exists():
        c.require(history.read_bytes() == previous, 'Prior owner history differs')
    else:
        with history.open('xb') as stream:
            stream.write(previous)
            stream.flush()
            os.fsync(stream.fileno())
        fd = os.open(history.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return c.pin(history)


class LaunchSafeguardsWorker(parent.Worker):
    def __init__(self, args, source):
        c.require(not args.retry_failed, 'Failed fits require separate explicit retry handling')
        self.scheduler_source = {
            'schema': 'campaign-launch-safeguards-producer-v1',
            'actual_outer_source_manifest': source,
            'unchanged_execution7_source_manifest': c.source_manifest(),
            'unchanged_memory_admission_source_manifest': memory.own_source(MEMORY_SHA),
            'plan': c.pin(args.plan),
            'changes': ['I4 bank receipt confinement and actual bytes/SHA256 validation',
                        '20 GiB launch disk floor plus active and candidate reservations'],
            'memory_admission_policy': 'clean-active-inactive-file-cache-v1',
            'automatic_retry': False,
            'global_all_6460_PASS_required': True,
            'full_training_supported': True,
            'whole_campaign_storage_reserved': False,
            'checkpoint_boundary_checks': False}
        owner_path = Path(c.read(args.plan)['state_root']) / 'OWNER.json'
        original_write = parent.write

        def owner_write(path, value):
            if Path(path) == owner_path:
                # Execution7 holds SUPERVISOR.lock before this OWNER write.
                previous = preserve_owner(owner_path)
                value = {**value, 'actual_scheduler_producer': self.scheduler_source,
                         'previous_owner': previous}
            return original_write(path, value)

        parent.write = owner_write
        try:
            super().__init__(args)
        finally:
            parent.write = original_write
        # The inherited start method calls these under the original host ledger
        # lock. No native argv, scientific source or fit setting is replaced.
        parent.probe = memory.probe
        parent.capacity_decision = capacity_decision
        c.write(self.root / ('LAUNCH_SAFEGUARDS_OWNER-' + str(time.time_ns()) + '.json'),
                {'identity': c.identity(), 'producer': self.scheduler_source,
                 'previous_owner': c.read(owner_path)['previous_owner'], 'TEST_accessed': False})

    def validate_result(self, task, phase, value, output):
        status, selected = super().validate_result(task, phase, value, output)
        if phase != 'smoke' and task['kind'] == 'aggregate_I4_exports' and status == 'complete':
            bank = value.get('bank_receipt')
            c.require(isinstance(bank, dict) and all(key in bank for key in ('path', 'bytes', 'sha256')),
                      'I4 bank receipt requires path, bytes and SHA256')
            c.require(type(bank['bytes']) is int and bank['bytes'] >= 0,
                      'I4 bank receipt bytes must be a nonnegative integer')
            expected = Path(output).resolve() / 'I4_selected_VALID.pt'
            c.require(Path(bank['path']).resolve() == expected, 'I4 bank receipt escapes own output')
            # verify checks confinement and reads actual bytes/hash; a filename
            # alone, missing receipt fields or an escaping symlink is insufficient.
            c.verify(bank, self.plan['project_root'])
        return status, selected

    def report(self, **extra):
        if time.time() - self.last_report < 10 and not extra:
            return
        return super().report(actual_scheduler_producer=self.scheduler_source, **extra)

    def invocation(self, task, phase, directory):
        return {**super().invocation(task, phase, directory),
                'actual_scheduler_producer': self.scheduler_source}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=c.WORKERS, required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--poll-seconds', type=float, default=10)
    parser.add_argument('--smoke-wall-seconds', type=int, default=1800)
    args = parser.parse_args()
    c.require(args.poll_seconds >= 1 and args.smoke_wall_seconds == 1800, 'Keep original polling/wall policy')
    args.retry_failed = False
    instance = None
    try:
        source = own_source(args.source_sha256)
        instance = LaunchSafeguardsWorker(args, source)
        return instance.run()
    except BaseException as exc:
        if instance is not None:
            instance.phase = 'supervisor_error'
            instance.report(error=type(exc).__name__ + ': ' + str(exc), children_retain_inherited_leases=True)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
