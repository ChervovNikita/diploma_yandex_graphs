"""Explicit readable operations for the authorized six-device allocation queue."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent / 'allocation_six_gpu_20261031_v1'
sys.path.insert(0, str(PACKAGE))
from host_extension import SHARDS, ROOT, config

REMOTE = ROOT / 'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/planning/allocation_six_gpu_20261031_v1'
KEY = '/Users/alex/.ssh/mlspace__private_key_anogena.txt'


def run(worker, operation, hold=None):
    host = config(worker)
    login = ('anogena.' if worker.startswith('anogena/') else 'anogena-2.') + 'ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
    ssh = ['ssh', '-tt', '-e', 'none', '-i', KEY, '-p', '2222', '-o', 'BatchMode=yes',
           '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=yes', '-o', 'UpdateHostKeys=no', login]
    source_sha = hashlib.sha256((PACKAGE / 'MANIFEST.json').read_bytes()).hexdigest()
    guard = ['set -eu', 'test "$(hostname)" = ' + shlex.quote(host['expected_hostname']),
             'test "$(nvidia-smi --query-gpu=uuid --format=csv,noheader)" = ' + shlex.quote('\n'.join(host['expected_gpu_uuids_by_index'])),
             'stty raw -echo', 'cd ' + shlex.quote(str(ROOT))]
    script = {'preflight': 'preflight.py', 'handover': 'handover.py', 'launch': 'launch.py',
              'status': 'status.py', 'lock-probe': 'lock_probe.py', 'tests': 'test_extension.py',
              'verify': 'verify_live_v2.py'}[operation]
    script_path = (REMOTE.parent / 'allocation_six_gpu_operations_20261031_v1' / script) if operation == 'verify' else REMOTE / script
    command = [str(ROOT / '.venv/bin/python'), '-B', '-u', str(script_path)]
    if operation != 'tests':
        command += ['--worker', worker]
    if operation in ('preflight', 'handover', 'launch'):
        command += ['--source-sha256', source_sha]
    if operation == 'lock-probe' and hold is not None:
        command += ['--hold', str(hold)]
    argv = [*ssh, '\n'.join([*guard, shlex.join(command)])]
    tag = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + worker.replace('/', '_') + '-' + operation + '-' + uuid.uuid4().hex[:8]
    (HERE / (tag + '-COMMAND.json')).write_text(json.dumps({'argv': argv}, indent=2) + '\n')
    result = subprocess.run(argv, capture_output=True, timeout=300)
    (HERE / (tag + '-STDOUT.txt')).write_bytes(result.stdout)
    (HERE / (tag + '-STDERR.txt')).write_bytes(result.stderr)
    receipt = {'operation': operation, 'worker': worker, 'returncode': result.returncode,
               'stdout': str(HERE / (tag + '-STDOUT.txt')), 'source_sha256': source_sha}
    (HERE / (tag + '-RESULT.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)
    print(result.stdout.decode(), flush=True)
    if result.returncode:
        print(result.stderr.decode(), file=sys.stderr, flush=True)
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=SHARDS, required=True)
    parser.add_argument('--operation', choices=('preflight', 'handover', 'launch', 'status', 'lock-probe', 'tests', 'verify'), required=True)
    parser.add_argument('--hold', type=int)
    args = parser.parse_args()
    run(args.worker, args.operation, args.hold)


if __name__ == '__main__':
    main()
