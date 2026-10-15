"""Host-wide reservation admission using actual free capacity and remaining growth."""
import shutil
import re
import subprocess
from common import *


def cgroup_paths(membership_text, mountinfo_text, is_directory=None):
    """Resolve the actual unified membership through complete accessible views.

    Bind aliases of one cgroup filesystem are not extra resource controllers.
    Prefer the widest accessible view per device, and the canonical mount when
    equally wide. All ancestors in that view still contribute effective limits.
    """
    is_directory = (lambda path: path.is_dir()) if is_directory is None else is_directory
    entries = [line.split(':', 2)[2] for line in membership_text.splitlines() if line.startswith('0::')]
    require(len(entries) == 1, 'Exactly one cgroup-v2 membership required')
    membership = Path(entries[0])
    require(membership.is_absolute() and '..' not in membership.parts, 'Normalized cgroup membership required')
    candidates = {}
    def unescape(value):
        return re.sub(r'\\([0-7]{3})', lambda match: chr(int(match.group(1), 8)), value)
    for line in mountinfo_text.splitlines():
        left, right = line.split(' - ', 1)
        if right.split()[0] != 'cgroup2':
            continue
        fields = left.split()
        device = fields[2]; mount_root = Path(unescape(fields[3])); mount = Path(unescape(fields[4]))
        require(mount_root.is_absolute() and mount.is_absolute() and '..' not in mount_root.parts
                and '..' not in mount.parts, 'Normalized cgroup mount view required')
        if not membership.is_relative_to(mount_root):
            continue
        current = mount / membership.relative_to(mount_root)
        if not is_directory(current):
            continue
        rank = (len(mount_root.parts), 0 if mount == Path('/sys/fs/cgroup') else 1, str(mount))
        previous = candidates.get(device)
        if previous is None or rank < previous[0]:
            candidates[device] = (rank, current, mount)
    require(candidates, 'No accessible cgroup-v2 view maps the actual membership')
    result = []
    for _, current, mount in sorted(candidates.values(), key=lambda row: row[0]):
        while True:
            if current not in result:
                result.append(current)
            if current == mount:
                break
            current = current.parent
    return result


def effective_cgroups():
    return cgroup_paths(Path('/proc/self/cgroup').read_text(), Path('/proc/self/mountinfo').read_text())


def process_snapshot(active):
    processes = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            raw = (path / 'stat').read_text(); fields = raw[raw.rfind(')') + 2:].split()
            processes[int(path.name)] = {'ppid': int(fields[1]), 'ticks': int(fields[11]) + int(fields[12]),
                'rss': int(fields[21]) * os.sysconf('SC_PAGE_SIZE')}
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    result = {}
    for record in active:
        roots = set()
        if record.get('external_identity') and same_process(record['external_identity']):
            roots.add(record['external_identity']['pid'])
        for name in ('GUARDIAN.json', 'CHILD.json'):
            path = Path(record['directory']) / name
            if path.exists():
                ident = read(path)['identity']
                if same_process(ident):
                    roots.add(ident['pid'])
        selected = set(roots)
        while True:
            expanded = selected | {pid for pid, proc in processes.items() if proc['ppid'] in selected}
            if expanded == selected:
                break
            selected = expanded
        result[record['directory']] = {'pids': sorted(selected),
            'ticks': sum(processes.get(pid, {}).get('ticks', 0) for pid in selected),
            'rss': sum(processes.get(pid, {}).get('rss', 0) for pid in selected)}
    return result


def cpu_snapshot(groups):
    affinity = os.sched_getaffinity(0)
    total = busy = 0
    for line in Path('/proc/stat').read_text().splitlines():
        name, *values = line.split()
        if name.startswith('cpu') and name[3:].isdigit() and int(name[3:]) in affinity:
            ticks = list(map(int, values))[:8]
            total += sum(ticks); busy += sum(ticks) - ticks[3] - ticks[4]
    cgroups = {}
    for group in groups:
        maximum = group / 'cpu.max'
        if maximum.exists():
            quota, period = maximum.read_text().split()
            if quota != 'max':
                stat = dict(line.split() for line in (group / 'cpu.stat').read_text().splitlines())
                cgroups[str(group)] = {'cores': int(quota) / int(period), 'usage_usec': int(stat['usage_usec'])}
    return {'total': total, 'busy': busy, 'cores': len(affinity), 'cgroups': cgroups}


def probe(config, active):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True, timeout=10).splitlines()
    free = {gpu.strip(): int(mib.strip()) * 1024**2 for gpu, mib in (row.split(',') for row in rows)}
    require(list(free) == config['expected_gpu_uuids_by_index'], 'Physical GPU identities changed')
    gpu_usage = {}
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'], text=True, timeout=10).splitlines()
    for row in rows:
        if row.strip():
            gpu, pid, mib = (value.strip() for value in row.split(','))
            require(gpu in free, 'Unexpected physical GPU')
            gpu_usage[(gpu, int(pid))] = int(mib) * 1024**2
    mem = {line.split(':')[0]: int(line.split(':')[1].split()[0]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    ram = mem['MemAvailable']; groups = effective_cgroups(); memory_accounting = []
    for group in groups:
        if not (group / 'memory.max').exists():
            continue
        maximum = (group / 'memory.max').read_text().strip()
        current = int((group / 'memory.current').read_text())
        stats = dict(line.split() for line in (group / 'memory.stat').read_text().splitlines())
        reclaimable = max(0, int(stats.get('inactive_file', 0)) - int(stats.get('shmem', 0)))
        available = ram if maximum == 'max' else min(int(maximum), max(0, int(maximum) - current) + reclaimable)
        ram = min(ram, available)
        memory_accounting.append({'path': str(group), 'maximum': maximum, 'current': current, 'reclaimable': reclaimable, 'headroom': available})
    first = process_snapshot(active); cpu_first = cpu_snapshot(groups); started = time.monotonic()
    time.sleep(0.2)
    second = process_snapshot(active); cpu_second = cpu_snapshot(groups); elapsed = time.monotonic() - started
    own_busy = max(0, sum(row['ticks'] for row in second.values()) - sum(row['ticks'] for row in first.values())) / os.sysconf('SC_CLK_TCK') / elapsed
    total_delta = cpu_second['total'] - cpu_first['total']
    busy = cpu_second['cores'] * (cpu_second['busy'] - cpu_first['busy']) / max(1, total_delta)
    cpu_free = max(0, cpu_second['cores'] - max(0, busy - own_busy))
    for path, row in cpu_second['cgroups'].items():
        group_busy = max(0, row['usage_usec'] - cpu_first['cgroups'][path]['usage_usec']) / 1e6 / elapsed
        cpu_free = min(cpu_free, max(0, row['cores'] - max(0, group_busy - own_busy)))
    for record in active:
        measured = second[record['directory']]
        measured['vram'] = sum(gpu_usage.get((record['gpu_uuid'], pid), 0) for pid in measured['pids'])
    return {'free_vram': free, 'available_ram': ram, 'free_disk': shutil.disk_usage(config['project_root']).free,
        'available_cpu_before_owned_reservations': cpu_free, 'owned_usage': second, 'memory_accounting': memory_accounting,
        'sampled_at': time.time(), 'external_processes_signalled': False}


def capacity_decision(candidate, active, live, can_pair):
    reservation = candidate['reservation']; uuid = candidate['gpu_uuid']
    same_gpu = [row for row in active if row['gpu_uuid'] == uuid]
    reasons = []
    if len(same_gpu) >= 2:
        reasons.append('two GPU slots occupied')
    def compatible(row):
        if row.get('external_identity'):
            budget = row['reservation']
            return (reservation['vram_bytes'] + budget['vram_bytes'] <= 64 * GiB and
                    reservation['ram_bytes'] + budget['ram_bytes'] <= 48 * GiB)
        return can_pair(reservation, row['reservation'])
    if any(not compatible(row) for row in same_gpu):
        reasons.append('incompatible peak pair budgets')
    remaining_vram = remaining_ram = disk_reserved = cpu_reserved = 0
    for row in active:
        budget = row['reservation']; usage = live['owned_usage'].get(row['directory'], {})
        if row['gpu_uuid'] == uuid:
            remaining_vram += max(0, budget['vram_bytes'] - usage.get('vram', 0))
        remaining_ram += max(0, budget['ram_bytes'] - usage.get('rss', 0))
        disk_reserved += budget['disk_bytes']; cpu_reserved += budget['cpu_slots']
    if live['free_vram'][uuid] < reservation['vram_bytes'] + remaining_vram + GiB:
        reasons.append('GPU free capacity/growth')
    if live['available_ram'] < reservation['ram_bytes'] + remaining_ram + 8 * GiB:
        reasons.append('host RAM free capacity/growth')
    if live['free_disk'] < reservation['disk_bytes'] + disk_reserved + 8 * GiB:
        reasons.append('disk capacity')
    if live['available_cpu_before_owned_reservations'] + 0.05 < reservation['cpu_slots'] + cpu_reserved:
        reasons.append('CPU capacity')
    return {'admitted': not reasons, 'reasons': reasons, 'remaining_vram': remaining_vram,
        'remaining_ram': remaining_ram, 'disk_reserved': disk_reserved, 'cpu_reserved': cpu_reserved}
