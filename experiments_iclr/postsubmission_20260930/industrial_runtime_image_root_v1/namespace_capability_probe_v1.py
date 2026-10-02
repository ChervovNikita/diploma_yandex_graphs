"""No-sudo OS capability diagnosis; no dataset, library, model or GPU access."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / 'protocols'))
from fetch_authorized_evidence_v2 import SSH, REMOTE
here = Path(__file__).resolve().parent
report = here / 'NAMESPACE_CAPABILITY_LAUNCH_v1.json'
if report.exists():
    raise RuntimeError('Preserve prior attempt')
code = r'''
import datetime, json, os, pathlib, subprocess, sys
path = pathlib.Path(REMOTE_PHASE) / 'industrial_runtime_image_root_v1/NAMESPACE_CAPABILITY_v1.json'
if path.exists():
    raise RuntimeError('Preserve prior diagnosis')
record = {'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'scientific_execution': False, 'data_access': False, 'GPU_access': False,
          'host_mount_namespace': os.readlink('/proc/self/ns/mnt'), 'tests': []}
commands = [
    ('user_namespace', ['unshare', '--user', '--map-root-user', sys.executable, '-c',
                       'import os; print(os.getuid(), os.readlink("/proc/self/ns/user"))']),
    ('mount_namespace_without_automatic_propagation', ['unshare', '--user', '--map-root-user',
        '--mount', '--propagation', 'unchanged', sys.executable, '-c',
        'import os; print(os.getuid(), os.readlink("/proc/self/ns/mnt"))']),
    ('manual_private_propagation_in_new_namespace', ['unshare', '--user', '--map-root-user',
        '--mount', '--propagation', 'unchanged', sys.executable, '-c',
        'import os,subprocess,sys; assert os.readlink("/proc/self/ns/mnt") != sys.argv[1]; '
        'r=subprocess.run(["mount","--make-rprivate","/"],capture_output=True,text=True); '
        'print(r.stdout); print(r.stderr,file=sys.stderr); raise SystemExit(r.returncode)',
        record['host_mount_namespace']]),
    ('all_namespaces_without_automatic_propagation', ['unshare', '--user', '--map-root-user',
        '--mount', '--propagation', 'unchanged', '--pid', '--fork', '--net', '--ipc', '--uts',
        sys.executable, '-c', 'import os; print(os.getpid(),os.getuid(),os.readlink("/proc/self/ns/mnt"),os.readlink("/proc/self/ns/net"))'])]
for name, argv in commands:
    result = subprocess.run(argv, capture_output=True, text=True, timeout=20,
                            stdin=subprocess.DEVNULL, close_fds=True)
    record['tests'].append({'name': name, 'argv': argv, 'exit_code': result.returncode,
                            'stdout': result.stdout, 'stderr': result.stderr})
path.write_text(json.dumps(record, indent=2) + '\n')
'''
code = 'REMOTE_PHASE = ' + repr(REMOTE) + '\n' + code
command = 'cd ' + shlex.quote(REMOTE) + ' && . ' + shlex.quote(REMOTE + '/protocols/repo_env.sh') + ' && python3 -'
start = datetime.now(timezone.utc).isoformat()
(here / 'NAMESPACE_CAPABILITY_REMOTE_CODE_v1.txt').write_text(code)
result = subprocess.run(SSH + [command], input=code, capture_output=True, text=True)
report.write_text(json.dumps({'start_UTC': start, 'terminal_UTC': datetime.now(timezone.utc).isoformat(),
                             'ssh_destination': SSH[-1], 'exit_code': result.returncode,
                             'stdout': result.stdout, 'stderr': result.stderr}, indent=2) + '\n')
print(json.dumps({'exit_code': result.returncode, 'report': str(report)}))
raise SystemExit(result.returncode)
