"""Fail closed on the corrected one-GPU allocation, then supervise research."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
PHASE = Path(__file__).resolve().parents[1]
REPO = PHASE.parents[1]
EXPECTED_LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
FORBIDDEN_LOGIN = 'anogena.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--route', type=Path, default=PHASE / 'protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json')
    p.add_argument('--require-idle-gpu', action='store_true')
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    route_path, output = args.route.resolve(), args.output.resolve()
    if not route_path.is_relative_to(PHASE) or not output.is_relative_to(PHASE) or output.exists():
        raise ValueError('Require confined route evidence and new supervision output')
    route = json.loads(route_path.read_text())
    if route['ssh_destination'] != EXPECTED_LOGIN or route['forbidden_destination'] != FORBIDDEN_LOGIN:
        raise ValueError('Wrong allocation route')
    if route['repository'] != str(REPO) or route['visible_gpu_count'] != 1 or not route['repository_pwd_matches']:
        raise ValueError('Route evidence does not authorize this repository/allocation')
    if os.environ.get('GNNM_SSH_DESTINATION') != EXPECTED_LOGIN:
        raise ValueError('Calling SSH route must be explicitly supplied by the coordinator')
    if Path.cwd().resolve() != REPO:
        raise ValueError('Require exact repository cwd')
    query = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu',
                            '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True)
    rows = [line for line in query.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError('Expected exactly one GPU; seven-GPU allocation is forbidden')
    row = [part.strip() for part in rows[0].split(',')]
    if len(row) != 6 or int(row[0]) != 0 or int(row[4]) != 81920 or row[2] != 'NVIDIA A100-SXM4-80GB':
        raise RuntimeError('GPU identity differs from verified authorized allocation')
    if args.require_idle_gpu and (int(row[3]) > 100 or int(row[5]) != 0):
        raise RuntimeError('Authorized GPU is occupied; no child started')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if len(command) < 2:
        raise ValueError('Require repository interpreter and phase script')
    start = datetime.now(timezone.utc).isoformat()
    argv = [sys.executable, str(PHASE / 'protocols/run_logged.py'), '--output', str(output), '--', *command]
    child = subprocess.run(argv, check=False, env=os.environ.copy())
    if not output.is_dir():
        raise RuntimeError('Supervisor did not create its confined output')
    record = {'schema': 'gnnm-authorized-allocation-supervision-v2', 'start_UTC': start,
              'completion_UTC': datetime.now(timezone.utc).isoformat(), 'ssh_destination': EXPECTED_LOGIN,
              'route_record_sha256': sha(route_path), 'allocation_guard_sha256': sha(Path(__file__)),
              'visible_gpu_count': 1, 'gpu_uuid': row[1], 'gpu_name': row[2],
              'initial_memory_MiB': int(row[3]), 'initial_utilization_percent': int(row[5]),
              'idle_required': args.require_idle_gpu, 'child_exit_code': child.returncode,
              'inner_completion_sha256': sha(output / 'completion.json') if (output / 'completion.json').exists() else None,
              'scope': 'Corrected supplied SSH route plus one-GPU identity; inner supervisor checks declared cache/temp paths.'}
    with (output / 'authorization.json').open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'authorized_ssh_destination': EXPECTED_LOGIN, 'visible_gpu_count': 1,
                      'authorization_sha256': sha(output / 'authorization.json'), 'child_exit_code': child.returncode}), flush=True)
    sys.exit(child.returncode)


if __name__ == '__main__':
    main()
