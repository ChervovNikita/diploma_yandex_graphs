"""Execution7 with clean file-cache RAM admission; unchanged smoke/barrier/full flow."""
import argparse
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
import capacity
import worker as parent


def own_source(expected):
    source = c.pin(HERE / 'MANIFEST.json')
    c.require(source['sha256'] == expected, 'Exact admission wrapper source required')
    for row in c.read(HERE / 'MANIFEST.json')['files']:
        path = (HERE / row['path']).resolve()
        c.require(path.is_relative_to(HERE), 'Admission source escapes package')
        actual = c.pin(path)
        c.require(all(actual[key] == row[key] for key in ('bytes', 'sha256')), 'Admission source changed')
    c.require(c.source_manifest()['sha256'] == PARENT_SHA, 'Frozen execution7 helper changed')
    return source


def memory_headroom(maximum, current, stats):
    """Credit only bounded clean file cache, including pages on the active LRU.

    file and the two LRU counts overlap: never add file to the LRU counts.
    Some excluded pages are already absent from file LRUs, and dirty/writeback
    may overlap. Subtracting them again deliberately underestimates reclaim.
    Slab, anonymous memory and swap never contribute reclaimable credit.
    """
    keys = ('file', 'active_file', 'inactive_file', 'shmem', 'file_dirty',
            'file_writeback', 'unevictable')
    values = {key: int(stats[key]) for key in keys}
    current = int(current)
    c.require(current >= 0 and all(value >= 0 for value in values.values()), 'Negative memory accounting')
    reclaimable = max(0, min(current, values['file'] - values['shmem'],
                            values['active_file'] + values['inactive_file'])
                      - values['file_dirty'] - values['file_writeback'] - values['unevictable'])
    if maximum == 'max':
        headroom = None
    else:
        limit = int(maximum)
        c.require(limit >= 0, 'Negative memory limit')
        # Preserve an existing over-limit charge; clamping limit-current first
        # would incorrectly grant the full reclaim credit despite that deficit.
        headroom = max(0, min(limit, limit - current + reclaimable))
    return {'maximum': str(maximum), 'current': current, 'stats': values,
            'reclaimable': reclaimable, 'headroom': headroom}


def memory_available(host_available, groups):
    c.require(host_available >= 0, 'Negative host MemAvailable')
    available = host_available
    accounting = []
    for group in groups:
        maximum = group / 'memory.max'
        if not maximum.exists():
            continue
        stats = dict(line.split() for line in (group / 'memory.stat').read_text().splitlines())
        row = memory_headroom(maximum.read_text().strip(),
                              (group / 'memory.current').read_text(), stats)
        row['path'] = str(group)
        accounting.append(row)
        if row['headroom'] is not None:
            available = min(available, row['headroom'])
    return available, accounting


def probe(config, active):
    # Reuse the original GPU identities, CPU sampling, owned usage and disk
    # measurements. Only replace its RAM estimate, using fresh kernel counters.
    live = capacity.probe(config, active)
    host_available = next(int(line.split()[1]) * 1024
                          for line in Path('/proc/meminfo').read_text().splitlines()
                          if line.startswith('MemAvailable:'))
    available, accounting = memory_available(host_available, capacity.effective_cgroups())
    return {**live, 'available_ram': available, 'memory_accounting': accounting,
            'host_mem_available': host_available,
            'memory_admission_policy': 'clean-active-inactive-file-cache-v1',
            'memory_sampled_at': time.time()}


def preserve_owner(path):
    """Durably preserve exact previous bytes before the locked parent replaces them."""
    if not path.exists():
        return None
    previous = path.read_bytes()
    history = path.with_name('OWNER-before-memory-admission-' + hashlib.sha256(previous).hexdigest() + '.json')
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


class AdmissionWorker(parent.Worker):
    def __init__(self, args, source):
        c.require(not args.retry_failed, 'Unchanged failed fits may not be retried')
        self.scheduler_source = {'schema': 'campaign-memory-admission-producer-v1',
            'actual_outer_source_manifest': source,
            'unchanged_parent_source_manifest': c.source_manifest(),
            'plan': c.pin(args.plan), 'automatic_retry': False,
            'changes': 'RAM probe credits conservatively bounded clean active and inactive file cache',
            'global_all_6460_PASS_required': True, 'full_training_supported': True}
        c.require(self.scheduler_source['unchanged_parent_source_manifest']['sha256'] == PARENT_SHA,
                  'Frozen execution7 helper changed')
        owner_path = Path(c.read(args.plan)['state_root']) / 'OWNER.json'
        original_write = parent.write

        def owner_write(path, value):
            if Path(path) == owner_path:
                # The parent already holds SUPERVISOR.lock when it writes OWNER.
                previous = preserve_owner(owner_path)
                value = {**value, 'actual_scheduler_producer': self.scheduler_source,
                         'previous_owner': previous}
            return original_write(path, value)

        parent.write = owner_write
        try:
            super().__init__(args)
        finally:
            parent.write = original_write
        parent.probe = probe
        c.write(self.root / ('MEMORY_ADMISSION_OWNER-' + str(time.time_ns()) + '.json'),
                {'identity': c.identity(), 'producer': self.scheduler_source,
                 'previous_owner': c.read(owner_path)['previous_owner'], 'TEST_accessed': False})

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
    parser.add_argument('--plan', required=True); parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--poll-seconds', type=float, default=10)
    parser.add_argument('--smoke-wall-seconds', type=int, default=1800)
    args = parser.parse_args()
    c.require(args.poll_seconds >= 1 and args.smoke_wall_seconds == 1800, 'Keep original polling/wall policy')
    args.retry_failed = False
    instance = None
    try:
        source = own_source(args.source_sha256)
        instance = AdmissionWorker(args, source)
        return instance.run()
    except BaseException as exc:
        if instance is not None:
            instance.phase = 'supervisor_error'
            instance.report(error=type(exc).__name__ + ': ' + str(exc), children_retain_inherited_leases=True)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
