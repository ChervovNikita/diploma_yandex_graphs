#!/usr/bin/env python3
"""Complete failed prelaunch staging; never re-run a started probe."""
from datetime import datetime, timezone
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
previous = (HERE / 'REMOTE_LAUNCH_SOURCE.py').read_text()
assert previous.count('assert not OUTPUT.exists()') == 1
assert previous.count('OUTPUT.mkdir()') == 1
failure = dict(UTC=datetime.now(timezone.utc).isoformat(),
               status='FAILED_PRELAUNCH_STAGING_MISSING_PLAN_DIRECTORY',
               remote_execution_directory_verified_empty=True,
               no_probe_or_supervisor_launched=True, probe_attempt_count=0,
               source_harness_sha256=hashlib.sha256(previous.encode()).hexdigest(),
               transport_receipt=json.loads((HERE / 'LAUNCH_TRANSPORT.json').read_text()))
(HERE / 'PRELAUNCH_STAGING_FAILURE.json').write_text(json.dumps(failure, indent=2) + '\n')
revised = previous.replace('assert not OUTPUT.exists()',
                           'assert OUTPUT.is_dir() and not list(OUTPUT.iterdir())')
revised = revised.replace('OUTPUT.mkdir()',
        "PLAN.mkdir()\n(OUTPUT/'PRELAUNCH_STAGING_FAILURE.json').write_text(" +
        'json.dumps(' + repr(failure) + ",indent=2,sort_keys=True)+'\\n')")
ast.parse(revised)
wrapped = 'import traceback,json\ntry:\n' + ''.join(' ' + line + '\n' for line in revised.splitlines())
wrapped += "except BaseException as error:\n print(json.dumps(dict(status='FAIL_PRELAUNCH_STAGING_PRESERVED',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())))\n raise\n"
ast.parse(wrapped)
(HERE / 'REMOTE_STAGING_RESUMPTION_SOURCE.py').write_text(wrapped)
command = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
           '-o', 'ConnectTimeout=20', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
           '-p', '2222', 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
           'cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs && '
           'exec /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/'
           'experiments_iclr/postsubmission_20260930/native_ncn_runtime_20261005_v1/.venv/bin/python -']
result = subprocess.run(command, input=wrapped, capture_output=True, text=True, timeout=90)
transport = dict(UTC=datetime.now(timezone.utc).isoformat(), command=command,
                exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr,
                staging_only_resumption=True, prior_probe_attempt_count=0,
                source_sha256=hashlib.sha256(wrapped.encode()).hexdigest())
(HERE / 'STAGING_RESUMPTION_TRANSPORT.json').write_text(json.dumps(transport, indent=2) + '\n')
if result.returncode == 0:
    data = json.loads(result.stdout)
    for key, name in (('launch', 'LAUNCH_RECEIPT.json'), ('release', 'ROOT_RESOURCE_RELEASE.json'),
                      ('authorization', 'AUTHORIZATION.json')):
        (HERE / name).write_text(json.dumps(data[key], indent=2, sort_keys=True) + '\n')
    print(json.dumps(data['launch'], sort_keys=True))
else:
    print(json.dumps(transport, sort_keys=True))
    sys.exit(result.returncode)
