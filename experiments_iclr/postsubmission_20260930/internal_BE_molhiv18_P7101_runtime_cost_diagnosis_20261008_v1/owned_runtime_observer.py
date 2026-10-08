"""Bounded read-only observer: exact authorized PIDs, metadata and aggregate GPU only."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

PHASE = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930').resolve()
EXPECTED = {523400: 6019318952, 526095: 6021762957}
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
FIELDS = ['index', 'uuid', 'name', 'utilization.gpu', 'utilization.memory', 'memory.total', 'memory.used', 'memory.free', 'temperature.gpu', 'power.draw']
HZ = os.sysconf('SC_CLK_TCK')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utc(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def inside(value):
    path = (PHASE / value).resolve()
    require(path.is_relative_to(PHASE) and path != PHASE, 'Receipt path outside authorized phase')
    return path


def read_json(path):
    return json.loads(path.read_text())


def gpu():
    output = subprocess.check_output(['/usr/bin/nvidia-smi', '--query-gpu=' + ','.join(FIELDS), '--format=csv,noheader,nounits'], text=True, timeout=5)
    rows = [dict(zip(FIELDS, [value.strip() for value in line.split(',')])) for line in output.splitlines()]
    require(len(rows) == 1 and rows[0]['uuid'] == GPU, 'Sole authorized GPU identity mismatch')
    return rows[0]


def proc(pid):
    base = Path('/proc') / str(pid)
    fields = (base / 'stat').read_text().rsplit(') ', 1)[1].split()
    require(int(fields[19]) == EXPECTED[pid], 'Authorized PID start identity mismatch')
    uptime = float(Path('/proc/uptime').read_text().split()[0])
    now = time.time()
    status = {}
    for line in (base / 'status').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in {'State', 'Threads', 'VmRSS', 'VmHWM', 'Cpus_allowed_list', 'voluntary_ctxt_switches', 'nonvoluntary_ctxt_switches'}:
            status[key] = value.strip()
    row = dict(pid=pid, start_ticks=int(fields[19]), state=fields[0], ppid=int(fields[1]), group=int(fields[2]), session=int(fields[3]),
               user_cpu_seconds=int(fields[11]) / HZ, system_cpu_seconds=int(fields[12]) / HZ,
               threads=int(fields[17]), elapsed_seconds=uptime-int(fields[19])/HZ,
               start_utc_estimate=utc(now-uptime+int(fields[19])/HZ), status=status)
    row['cmdline'] = [item.decode() for item in (base / 'cmdline').read_bytes().split(b'\0') if item]
    row['schedstat'] = [int(value) for value in (base / 'schedstat').read_text().split()]
    row['io'] = {key: int(value.strip()) for key, value in (line.split(':', 1) for line in (base / 'io').read_text().splitlines())}
    return row


require(socket.gethostname() == 'anogena-2-0', 'Authorized host mismatch')
first_gpu = gpu()
parent = proc(523400)
worker = proc(526095)
args = parent['cmdline']
release_path = inside(args[args.index('--release') + 1])
release_bytes = release_path.read_bytes()
require(hashlib.sha256(release_bytes).hexdigest() == args[args.index('--release-sha256') + 1], 'Parent release seal mismatch')
release = json.loads(release_bytes)
execution = inside(release['execution_directory'])
job_path = execution / 'receipts/P_7101_JOB.json'
owner_path = execution / 'receipts/P_7101_OWNER.json'
job = read_json(job_path)
owner = read_json(owner_path)
require(job['condition'] == 'P' and job['seed'] == 7101 and job['cell'] == 'P_7101', 'Owned cell mismatch')
require(owner['pid'] == 526095 and owner['start_ticks'] == EXPECTED[526095], 'Worker receipt identity mismatch')
parent_owner = read_json(execution / 'PARENT_OWNER.json')
require(parent_owner['pid'] == 523400 and parent_owner['start_ticks'] == EXPECTED[523400], 'Parent receipt identity mismatch')
require((job['active_seconds'], job['cleanup_seconds'], job['hard_seconds']) == (32390, 10, 32400), 'Frozen caps mismatch')
fit_output = inside(job['fit_output'])


def progress_metadata():
    # Only these engineering scalars leave the interpreter. No objective, trace,
    # selector, checkpoint, data, weight, or endpoint payload is output or retained.
    progress_path = fit_output / 'PROGRESS.json'
    value = read_json(progress_path)
    return dict(values={key: value[key] for key in ('epoch', 'complete_epochs', 'steps', 'complete') if key in value},
                receipt_mtime_utc=utc(progress_path.stat().st_mtime))


before = dict(observed_utc=utc(time.time()), monotonic=time.monotonic(), parent=parent, worker=worker,
              gpu_aggregate=first_gpu, progress=progress_metadata())
time.sleep(10)
after = dict(observed_utc=utc(time.time()), monotonic=time.monotonic(), parent=proc(523400), worker=proc(526095),
             gpu_aggregate=gpu(), progress=progress_metadata())
now = time.time()
job_mtime = job_path.stat().st_mtime
result = dict(schema='molhiv18-P7101-owned-runtime-cost-observation-v1', host=socket.gethostname(), clock_ticks_per_second=HZ,
              route_gpu_uuid=GPU, release=dict(path=str(release_path), sha256=hashlib.sha256(release_bytes).hexdigest()),
              execution_directory=str(execution), job_receipt=dict(path=str(job_path), condition=job['condition'], seed=job['seed'],
                  active_seconds=job['active_seconds'], cleanup_seconds=job['cleanup_seconds'], hard_seconds=job['hard_seconds'],
                  receipt_mtime_utc=utc(job_mtime), owner=owner),
              bounds=dict(active_cutoff_utc_upper_estimate=utc(job_mtime+job['active_seconds']),
                  active_remaining_seconds_upper_estimate=job_mtime+job['active_seconds']-now,
                  hard_cutoff_utc_upper_estimate=utc(job_mtime+job['hard_seconds']),
                  family_hard_remaining_seconds=parent_owner['family_hard_deadline_monotonic']-time.monotonic(),
                  cell_begin_monotonic_not_persisted=True),
              before=before, after=after,
              scope=dict(only_authorized_process_metadata=True, aggregate_GPU_only=True, other_processes_inspected=False,
                         quality_values_emitted=False, traces_datasets_weights_opened=False, remote_mutations=False))
print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
