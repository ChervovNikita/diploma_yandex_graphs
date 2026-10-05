#!/usr/bin/env python3
"""Copy only coverage, resource, ownership and first-Adam metadata."""
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
phase=repo/'experiments_iclr/postsubmission_20260930'
output=phase/'pencil_citeseer_native300_train_valid_execution_20261005_v1'
launch=phase/'pencil_citeseer_native300_launch_receipts_20261005_v1'
r=dict(host=socket.gethostname(),output_directory_exists=output.exists(),files={})
def read(path,key):
 if path.exists():
  assert path.stat().st_size<1048576
  raw=path.read_text()
  r['files'][key]=dict(bytes=len(raw.encode()),sha256=hashlib.sha256(raw.encode()).hexdigest(),utf8=raw)
for name in ['COHORT_PROGRESS.json','COHORT_FREEZE.json','COHORT_FAILURE.json','TERMINAL_RECEIPT.json']:
 read(output/name,name)
for seed in [0,1,2]:
 for name in ['DISPATCH_GATE.json','SUPERVISOR_FIT_RECEIPT.json','run01/PROGRESS.json','run01/FIRST_ADAM_UPDATE.json','run01/FAILURE.json']:
  read(output/('seed_'+str(seed))/name,'seed_'+str(seed)+'/'+name)
# Scientific scalar histories, selections, FIT results, logits and checkpoints remain closed.
if (output/'COHORT_FAILURE.json').exists() or not output.exists():
 path=launch/'supervisor_stdout_stderr.log'
 if path.exists():
  with path.open('rb') as f:
   f.seek(max(0,path.stat().st_size-8192));r['supervisor_failure_log_tail']=f.read().decode(errors='replace')
print(json.dumps(r,sort_keys=True))
'''
command = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=20',
        '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt', '-p', '2222',
        'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
        'cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs && '
        'exec /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/'
        'experiments_iclr/postsubmission_20260930/native_ncn_runtime_20261005_v1/.venv/bin/python -']
result = subprocess.run(command, input=script, capture_output=True, text=True, timeout=50)
utc = datetime.now(timezone.utc).isoformat()
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
snapshot = HERE / ('metadata_' + stamp)
snapshot.mkdir()
transport = dict(UTC=utc, exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr,
                 scientific_outcomes_opened=False, TEST_access=False, readonly_metadata=True)
(snapshot / 'TRANSPORT.json').write_text(json.dumps(transport, indent=2) + '\n')
if result.returncode:
    print(json.dumps(dict(UTC=utc, exit_code=result.returncode, stderr=result.stderr)))
else:
    received = json.loads(result.stdout)
    compact = dict(UTC=utc, host=received['host'], output_directory_exists=received['output_directory_exists'], seeds=[])
    for name, row in received['files'].items():
        assert hashlib.sha256(row['utf8'].encode()).hexdigest() == row['sha256']
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(row['utf8'])
    for name in ['COHORT_PROGRESS.json','COHORT_FREEZE.json','COHORT_FAILURE.json','TERMINAL_RECEIPT.json']:
        if name in received['files']:
            data = json.loads(received['files'][name]['utf8'])
            compact[name] = {k: data[k] for k in ['status','complete','active_seed','error','completed_fits','signals_sent','inclusive_cohort_seconds'] if k in data}
    for seed in [0,1,2]:
        prefix = 'seed_' + str(seed) + '/'
        row = dict(seed=seed)
        for name in ['DISPATCH_GATE.json','run01/PROGRESS.json','run01/FIRST_ADAM_UPDATE.json']:
            key = prefix + name
            if key in received['files']:
                row[name] = json.loads(received['files'][key]['utf8'])
        receipt_key = prefix + 'SUPERVISOR_FIT_RECEIPT.json'
        if receipt_key in received['files']:
            fit = json.loads(received['files'][receipt_key]['utf8'])
            row['supervisor'] = {k: fit[k] for k in ['status','child_identity','elapsed_child_seconds','max_owned_RSS_bytes','max_output_including_log_bytes','signals_sent','child_exit_code'] if k in fit}
        if len(row)>1:
            compact['seeds'].append(row)
    if 'supervisor_failure_log_tail' in received:
        compact['supervisor_failure_log_tail'] = received['supervisor_failure_log_tail']
    (snapshot / 'COMPACT_METADATA.json').write_text(json.dumps(compact, indent=2, sort_keys=True) + '\n')
    print(json.dumps(compact, sort_keys=True))
