"""One bounded ordinary installer; owns only its process group, never a science job."""
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
OUT = Path(__file__).resolve().parent
PREFIX = REPO/'.gnnm_runtime/private_transfer_cp311_cu118_20261005_v1'
CONDA = Path('/disk/10tb/home/shmelev/miniconda3/bin/conda')
assert Path.cwd() == REPO and socket.gethostname() == 'peptide'
assert hashlib.sha256(CONDA.read_bytes()).hexdigest() == 'd42fd7d2a0ec468c7767b62ec863698054ec26c8ea7bdba7fc6383477544ce85'
assert hashlib.sha256((OUT/'SETUP_COMMANDS.sh.txt').read_bytes()).hexdigest() == 'b451e13a687e56adaffa14b48fa9326d60d8942041582f9f5a5624189a07aff4'
assert not PREFIX.exists()
assert not (OUT/'INSTALLER_START.json').exists()
assert not (OUT/'INSTALLER_TERMINAL.json').exists()

def now():
    return datetime.now(timezone.utc).isoformat()

def ticks(pid):
    return int(Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[19])

def same_owner():
    try:
        return ticks(child.pid) == terminal['installer_start_ticks']
    except FileNotFoundError:
        return False

def write(name, data):
    temp = OUT/(name+'.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')
    temp.replace(OUT/name)

def file_receipt(path):
    return {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

env = dict(os.environ)
env.pop('PYTHONPATH', None)
env['PYTHONNOUSERSITE'] = '1'
start = time.monotonic()
terminal = {'UTC_start': now(), 'hard_seconds': 3600, 'single_attempt': True,
            'setup_script_sha256': file_receipt(OUT/'SETUP_COMMANDS.sh.txt')['sha256'],
            'runner_pid': os.getpid(), 'runner_start_ticks': ticks(os.getpid()),
            'models_imported': False, 'data_reads': False, 'fits_executed': False,
            'base_RAPIDS_packages_modified': False, 'timeout': False}
stdout = open(OUT/'INSTALLER.stdout.log', 'xb', buffering=0)
stderr = open(OUT/'INSTALLER.stderr.log', 'xb', buffering=0)
child = subprocess.Popen(['/bin/bash', str(OUT/'SETUP_COMMANDS.sh.txt')], cwd=REPO,
                         env=env, stdout=stdout, stderr=stderr, start_new_session=True)
terminal.update(installer_pid=child.pid, installer_start_ticks=ticks(child.pid), installer_pgid=child.pid)
write('INSTALLER_START.json', terminal)
try:
    terminal['installer_exit_code'] = child.wait(timeout=3600)
except subprocess.TimeoutExpired:
    terminal['timeout'] = True
    if same_owner():
        os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if same_owner():
            os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=10)
    terminal['installer_exit_code'] = child.returncode
stdout.close()
stderr.close()
terminal['installer_wall_seconds'] = time.monotonic()-start
terminal['status'] = 'INSTALL_FAILED'
if terminal['installer_exit_code'] == 0 and not terminal['timeout']:
    try:
        with open(OUT/'PROVIDERS.stdout.log', 'xb') as out, open(OUT/'PROVIDERS.stderr.log', 'xb') as err:
            provider = subprocess.run([str(PREFIX/'bin/python'), '-I', '-B', str(OUT/'collect_providers.py')],
                                      cwd=REPO, env=env, stdout=out, stderr=err, timeout=180)
        terminal['provider_exit_code'] = provider.returncode
        terminal['status'] = 'PASS' if provider.returncode == 0 else 'PROVIDER_FAILED'
    except Exception:
        terminal['provider_exception'] = traceback.format_exc()
        terminal['status'] = 'PROVIDER_FAILED'
    for name in ('TORCH_INSTALL_REPORT.json','CORE_INSTALL_REPORT.json','SPARSE_SCATTER_INSTALL_REPORT.json',
                 'PIP_FREEZE.txt','CONDA_EXPLICIT.txt'):
        source = PREFIX/name
        (OUT/name).write_bytes(source.read_bytes())
terminal['UTC_terminal'] = now()
terminal['total_wall_seconds'] = time.monotonic()-start
terminal['artifacts'] = {p.name: file_receipt(p) for p in sorted(OUT.iterdir()) if p.is_file()
                         and p.name not in ('INSTALLER_TERMINAL.json','INSTALLER_LAUNCH.json','RUNNER.stdout.log','RUNNER.stderr.log')}
write('INSTALLER_TERMINAL.json', terminal)
print(json.dumps({'status': terminal['status'], 'installer_exit_code': terminal['installer_exit_code']}), flush=True)
