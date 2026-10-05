#!/usr/bin/env python3
"""Read only this owned resource execution and copy its small receipts."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
script = r'''
from pathlib import Path
import hashlib,json,socket
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
p=repo/'experiments_iclr/postsubmission_20260930/pencil_citeseer_zero_update_resource_execution_20261005_v1'
result=dict(host=socket.gethostname(),files={})
for name in ['SUPERVISOR_RECEIPT.json','TERMINAL_RECEIPT.json','run01/PROGRESS.json','run01/RESULT.json','run01/FAILURE.json']:
 path=p/name
 if path.exists():
  assert path.stat().st_size<1048576
  result['files'][name]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,utf8=path.read_text())
for name in ['supervisor_stdout_stderr.log','probe_stdout_stderr.log']:
 path=p/name
 if path.exists():
  with path.open('rb') as stream:
   stream.seek(max(0,path.stat().st_size-16384))
   result[name+'_tail']=stream.read().decode(errors='replace')
print(json.dumps(result,sort_keys=True))
'''
command = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
           '-o', 'ConnectTimeout=20', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
           '-p', '2222', 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
           'cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs && '
           'exec /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/'
           'experiments_iclr/postsubmission_20260930/native_ncn_runtime_20261005_v1/.venv/bin/python -']
result = subprocess.run(command, input=script, capture_output=True, text=True, timeout=50)
utc = datetime.now(timezone.utc).isoformat()
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
transport = dict(UTC=utc, command=command, exit_code=result.returncode,
                 stdout=result.stdout, stderr=result.stderr, readonly_owned_execution=True)
(HERE / ('MONITOR_TRANSPORT_' + stamp + '.json')).write_text(json.dumps(transport, indent=2) + '\n')
if result.returncode:
    print(json.dumps(dict(UTC=utc, exit_code=result.returncode, stderr=result.stderr)))
else:
    received = json.loads(result.stdout)
    for name, row in received['files'].items():
        path = HERE / name
        path.parent.mkdir(exist_ok=True)
        assert hashlib.sha256(row['utf8'].encode()).hexdigest() == row['sha256']
        path.write_text(row['utf8'])
    receipt = json.loads(received['files']['SUPERVISOR_RECEIPT.json']['utf8'])
    compact = dict(UTC=utc, status=receipt['status'],
                   immediate_gpu_gate=receipt.get('immediate_gpu_gate'),
                   child_identity=receipt.get('child_identity'),
                   elapsed_child_seconds=receipt.get('elapsed_child_seconds'),
                   max_process_tree_RSS_bytes=receipt.get('max_parent_plus_loader_descendants_RSS_bytes'),
                   probe_progress=receipt.get('latest_probe_progress'), error=receipt.get('error'),
                   terminal='TERMINAL_RECEIPT.json' in received['files'])
    if compact['terminal']:
        compact['probe_log_tail'] = received.get('probe_stdout_stderr.log_tail')
        compact['supervisor_log_tail'] = received.get('supervisor_stdout_stderr.log_tail')
    print(json.dumps(compact, sort_keys=True))
