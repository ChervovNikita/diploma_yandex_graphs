"""Actual fixed-command backend; hardware probes run only after the host guard."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
from custody import require
from host_ledger import Ledger


def identity(pid):
    folder = Path('/proc') / str(pid)
    raw = (folder / 'stat').read_text()
    fields = raw[raw.rfind(')') + 2:].split()
    return {'pid': pid, 'UID': folder.stat().st_uid, 'linux_start_ticks': fields[19],
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def identity_status(value):
    if value.get('identity_unavailable'):
        return 'uncertain'
    try:
        return 'same' if identity(value['pid']) == value else 'different'
    except FileNotFoundError:
        return 'missing'
    except (OSError, ValueError, KeyError):
        return 'uncertain'


def process_status(pid):
    fields = {}
    for row in Path(f'/proc/{pid}/status').read_text().splitlines():
        if ':' in row:
            key, value = row.split(':', 1)
            fields[key] = value.split()
    return {'uid': int(fields['Uid'][0]), 'rss_bytes': int(fields.get('VmRSS', [0])[0]) * 1024}


def memory_available(host_available, cgroups):
    """Accepted conservative cache convention; all effective finite limits apply."""
    available = host_available
    accounting = []
    for group in cgroups:
        maximum, current, stats = group['maximum'], group['current'], group['stats']
        require(type(current) is int and current >= 0, 'Invalid cgroup memory.current')
        reclaimable = max(int(stats.get('inactive_file', 0)) - int(stats.get('shmem', 0)), 0)
        hard = host_available if maximum == 'max' else max(int(maximum) - current, 0)
        capacity = host_available if maximum == 'max' else min(int(maximum), hard + reclaimable)
        available = min(available, capacity)
        accounting.append({'path': group['path'], 'maximum': maximum, 'current': current,
                           'hard_headroom_bytes': hard, 'inactive_file_minus_shmem_bytes': reclaimable,
                           'cache_aware_headroom_bytes': capacity})
    return {'host_MemAvailable_bytes': host_available, 'available_ram_bytes': available,
            'cgroups': accounting, 'cache_reclamation_forced': False}


def probe(config, active):
    """Observed other-user consumption reduces free capacity; their PIDs are never controlled."""
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True, timeout=10).splitlines()
    free = {}
    for row in rows:
        gpu, mib = (s.strip() for s in row.split(','))
        free[gpu] = int(mib) * 1024**2
    require(list(free) == config['expected_gpu_uuids_by_index'], 'GPU identity table changed')
    known = {r['child_identity']['pid'] for r in active if r.get('child_identity')}
    used, unknown = {}, []
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'], text=True, timeout=10).splitlines()
    for row in rows:
        if not row.strip():
            continue
        gpu, pid, mib = (s.strip() for s in row.split(','))
        require(gpu in free, 'Unexpected GPU process device')
        pid = int(pid)
        try:
            status = process_status(pid)
        except FileNotFoundError:
            continue
        if pid not in known and status['uid'] == os.getuid():
            unknown.append(pid)
        used[pid] = used.get(pid, 0) + int(mib) * 1024**2
    memory = {}
    for row in Path('/proc/meminfo').read_text().splitlines():
        key, value = row.split(':', 1)
        memory[key] = int(value.split()[0]) * 1024
    paths = config.get('cgroup_v2_memory_paths')
    require(isinstance(paths, list) and paths and len(set(paths)) == len(paths), 'Qualified effective cgroup paths required')
    groups = []
    for value in paths:
        path = Path(value)
        require(path.is_absolute() and path.is_relative_to('/sys/fs/cgroup') and '..' not in path.parts, 'Confined cgroup-v2 memory path required')
        groups.append({'path': str(path), 'maximum': (path / 'memory.max').read_text().strip(),
                       'current': int((path / 'memory.current').read_text()),
                       'stats': {k: int(v) for k, v in (row.split() for row in (path / 'memory.stat').read_text().splitlines())}})
    ram = memory_available(memory['MemAvailable'], groups)
    rss = {}
    for pid in known:
        try:
            rss[pid] = process_status(pid)['rss_bytes']
        except FileNotFoundError:
            rss[pid] = 0
    return {'free_vram_by_uuid': free, 'available_ram_bytes': ram['available_ram_bytes'], 'memory_accounting': ram,
            'free_disk_bytes': shutil.disk_usage(config['project_root']).free,
            'allocated_vram_by_pid': used, 'rss_by_pid': rss, 'untracked_own_gpu_pids': unknown}


class PortableBackend:
    def __init__(self, config, ownership, budget, recheck, verify, reporting_transform=None, reporting_entry=None):
        self.config, self.ownership, self.budget = config, ownership, budget
        self.recheck, self.verify = recheck, verify
        self.reporting_transform, self.reporting_entry = reporting_transform, reporting_entry
        self.root = Path(config['project_root'])
        self.here = self.root / config['phase_relative'] / 'benchmark_expansion_20261015_v1/server51_campaign_port_v5'
        self.ledger = Ledger(self.root / config['phase_relative'] / '.server51_campaign_host_v1', ownership, config['worker'].split('/')[0])
        self.ledger.establish()
        self.assignments = {r['id']: r for r in ownership['assignments']}
        self.tokens, self.identities = {}, {}
        self.baseline_stops = set()
        self.supervisor_identity = identity(os.getpid())

    def admit(self, task, active):
        # `active` is only this GPU's children; the ledger supplies both GPUs.
        return self.ledger.reserve(self.assignments[task['id']], task.get('runtime_job_hash'),
            self.output_root / task['output_relative'], self.config['physical_gpu_uuid'], task['binding']['resources'],
            self.budget, lambda rows: probe(self.config, rows), identity_status, self.supervisor_identity, self.baseline_stops)

    def start(self, task, output, checkpoint, log, states):
        ticket = self.ledger.ticket
        require(ticket is not None, 'Actual Popen requires locked startup reservation')
        token = ticket['token']
        try:
            execution = task['binding']['execution']
            entry, environment = execution['runtime_entrypoint'], execution['python_environment']
            self.verify(entry)
            self.verify(environment)
            argv = [execution['python_executable'], '-B', '-u', str(self.here / 'qualified_python_entry.py'),
                    '--environment', environment['path'], '--environment-sha256', environment['sha256'],
                    '--entrypoint', entry['path'], '--entrypoint-sha256', entry['sha256'], '--', '--project-root', str(self.root)]
            if task['kind'] == 'fit':
                descriptor = execution['job_descriptor']
                self.verify(descriptor)
                argv += ['--job', descriptor['path'], '--output', str(output)]
                if checkpoint:
                    self.verify(checkpoint)
                    argv += ['--resume', checkpoint['path']]
            else:
                members = [states[x]['selected_VALID_export'] for x in task['dependencies']]
                require(len(members) == 4, 'I4 requires its four own selected exports')
                for value in members:
                    self.verify(value)
                argv += ['--members', *[v['path'] for v in members], '--output', str(output)]
            if self.reporting_transform is not None:
                observed = self.reporting_transform(task, output, checkpoint, states, argv)
                self.verify(self.reporting_entry)
                require(observed[:4] == [execution['python_executable'], '-B', '-u', self.reporting_entry['path']], 'Unexpected reporting argv transform')
                # Keep the qualified lexical interpreter guard around the observer.
                prefix = argv[:argv.index('--')]
                prefix[prefix.index('--entrypoint') + 1] = self.reporting_entry['path']
                prefix[prefix.index('--entrypoint-sha256') + 1] = self.reporting_entry['sha256']
                argv = prefix + ['--'] + observed[4:]
            threads = str(task['binding']['resources']['thread_count'])
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.config['physical_gpu_uuid'],
                       OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
                       PYTHONDONTWRITEBYTECODE='1')
            child = subprocess.Popen(argv, cwd=self.root, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException as exc:
            self.ledger.cancel_unspawned(str(exc))
            raise
        # From this line onward every error returns the child to Engine for drain.
        self.tokens[task['id']] = token
        ticket['child_created'] = True
        try:
            observed = identity(child.pid)
        except Exception as exc:
            observed = {'pid': child.pid, 'identity_unavailable': str(exc)}
        self.identities[child.pid] = observed
        try:
            self.ledger.commit(observed)
        except BaseException as exc:
            try:
                self.ledger.stop({'reason': 'child_identity_commit_failed', 'error': str(exc), 'pid': child.pid})
            except OSError:
                # Engine still gets the actual child and halts via this flag.
                pass
            self.control_error = str(exc)
        return child

    def identity(self, child):
        return self.identities[child.pid]

    def alive(self, recorded):
        return identity_status(recorded) != 'missing'

    @staticmethod
    def poll(child):
        return child.poll()

    @staticmethod
    def wait(child):
        return child.wait()

    def finished(self, task, terminal):
        self.ledger.reaped(self.tokens[task['id']], terminal)
        del self.tokens[task['id']]

    def cancel_pending(self, reason):
        self.ledger.cancel_unspawned(reason)

    def peer_activity(self):
        return self.ledger.has_peer_activity(self.config['worker'])
