"""Read current campaign owners, their own children, progress and GPU capacity."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--host', choices=('allocation', 'server51'), required=True)
args = parser.parse_args()
allocation = args.host == 'allocation'
expected_host = 'anogena-2-0' if allocation else 'olig'
uuids = ['GPU-44039938-fd82-41d2-fefd-de71514e2fac'] if allocation else [
    'GPU-9c6c20fa-a620-479d-f72f-163ac74deef2', 'GPU-30283973-140d-9e7b-e780-0fcc487dec54']
assert socket.gethostname() == expected_host
assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == uuids
project = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs' if allocation else
               '/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
execution = project / 'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/campaign_execution_20261016_v7'


def read(path):
    assert path.resolve().is_relative_to(project)
    return json.loads(path.read_text())


def process(identity):
    pid = int(identity['pid'])
    proc = Path('/proc') / str(pid)
    try:
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        matches = (int(fields[19]) == identity['start_ticks'] and proc.stat().st_uid == identity['uid'] and
                   Path('/proc/sys/kernel/random/boot_id').read_text().strip() == identity['boot_id'])
        return {'pid': pid, 'same_process': matches,
                'title': (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode().strip() if matches else None}
    except (FileNotFoundError, ProcessLookupError):
        return {'pid': pid, 'same_process': False, 'title': None}


out = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'host': args.host,
       'disk_free_bytes': shutil.disk_usage(project).free,
       'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
                                       '--format=csv,noheader'], text=True), 'workers': {}}
for gpu in ([0] if allocation else [0, 1]):
    worker = args.host + '/gpu' + str(gpu)
    state = execution / 'runs' / worker.replace('/', '_')
    status = read(state / 'STATUS.json')
    value = {key: status.get(key) for key in ('phase', 'counts', 'active_fit_ids', 'updated', 'error', 'failed_fit_ids')}
    value['owner'] = process(read(state / 'OWNER.json')['identity'])
    value['global_marker'] = (execution / 'GLOBAL_SMOKE_PASS.json').exists()
    value['training_accepted'] = (state / 'TRAINING_BARRIER_ACCEPTED.json').exists()
    value['tmux_alive'] = subprocess.run(['tmux', 'has-session', '-t', 'gnnm-allfit-' + worker.replace('/', '-') + '-20261016'],
                                         capture_output=True).returncode == 0
    value['children'] = []
    for fit_id in status.get('active_fit_ids', []):
        folder = state / 'full' / hashlib.sha256(fit_id.encode()).hexdigest()
        fit = read(folder / 'STATE.json')
        directory, output = Path(fit['directory']), Path(fit['output'])
        child = {'fit_id': fit_id, 'output': str(output)}
        if (directory / 'CHILD.json').exists():
            child.update(process(read(directory / 'CHILD.json')['identity']))
        for name in ('PROGRESS.json', 'RESULT.json'):
            if (output / name).exists():
                child[name] = read(output / name)
        log = directory / 'native.log'
        if log.exists():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 2048))
                child['log_tail'] = stream.read().decode(errors='replace')
        value['children'].append(child)
    if (state / 'CAPACITY_WAIT.json').exists():
        value['last_capacity_wait'] = read(state / 'CAPACITY_WAIT.json')
    out['workers'][worker] = value
print(json.dumps(out, sort_keys=True))
