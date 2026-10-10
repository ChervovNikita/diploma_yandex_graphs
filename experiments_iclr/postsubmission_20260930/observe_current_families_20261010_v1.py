"""Read existing owned progress without scores, signals, launches or retries."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
STAMP = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT = HERE / ('family_observation_' + STAMP)
OUT.mkdir(exist_ok=False)

ALLOCATION = r'''
import hashlib,json,socket,subprocess
from pathlib import Path
from datetime import datetime,timezone
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
P=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
Q=P/'masked_context_pubmed_stage1_root_execution_20261010_v1'
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=='24c315a7-3c08-471f-b550-b9a3e1faf75d'
def identity(pid):
 try:
  s=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
  return dict(PID=pid,start_ticks=int(s[19]),state=s[0],parent=int(s[1]))
 except FileNotFoundError:return None
owner=identity(588121)
assert owner is None or owner['start_ticks']==6040573132
files={}
for name in ('FAMILY_PROGRESS.json','FAMILY_FAILURE.json','FAMILY_COMPLETE.json'):
 q=Q/name
 if q.is_file():
  data=q.read_bytes();files[name]=dict(sha256=hashlib.sha256(data).hexdigest(),content=json.loads(data))
progress=files.get('FAMILY_PROGRESS.json',{}).get('content',{})
child=progress.get('active_child')
live=identity(child['PID']) if child else None
assert live is None or live['start_ticks']==child['start_ticks']
pub=P/'publication/masked_context9_launch_and_typed_context_source_20261010_v1/PUSH_RECEIPT.json'
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),owner=owner,current_child=live,files=files,push_receipt=json.loads(pub.read_text()) if pub.is_file() else None,quality_fields_opened=False,signals_sent=[],restarts=0)))
'''
argv = ['ssh', '-tt', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
        '-o', 'ConnectTimeout=15', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
        'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
        shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', ALLOCATION])]
try:
    result = subprocess.run(argv, capture_output=True, text=True, timeout=45)
    receipt = dict(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr,
                   command_sha256=hashlib.sha256(ALLOCATION.encode()).hexdigest())
    (OUT/'ALLOCATION_RECEIPT.json').write_text(json.dumps(receipt, indent=2)+'\n')
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    if result.returncode != 0 or len(rows) != 1:
        print(json.dumps(dict(allocation_observation_failed=True,exit_code=result.returncode,receipt=str(OUT/'ALLOCATION_RECEIPT.json'))))
    else:
        row=rows[0]
        (OUT/'ALLOCATION.json').write_text(json.dumps(row, indent=2)+'\n')
        progress=row['files'].get('FAMILY_PROGRESS.json',{}).get('content',{})
        print(json.dumps(dict(allocation_owner=row['owner'],current_record=progress.get('current_record'),epoch=progress.get('epoch'),completed=progress.get('completed'),failure=row['files'].get('FAMILY_FAILURE.json'),complete=row['files'].get('FAMILY_COMPLETE.json'),push_verified=(row['push_receipt'] or {}).get('verified'),quality_fields_opened=False)))
except subprocess.TimeoutExpired:
    (OUT/'ALLOCATION_OBSERVATION_TIMEOUT.json').write_text(json.dumps(dict(timeout=45,job_failure_inferred=False))+'\n')
    print(json.dumps(dict(allocation_observation_timeout=True,job_failure_inferred=False)))

wrapper=HERE/'gpu77_connection_recovery_v1/run_gpu77_v3.py'
command=HERE/'gpu77_connection_recovery_v1/commands/INDEPENDENT_SCORER_FIXED6_COMBINATION_CURRENT_OWNED_EPOCH_METADATA_20261010_v2/COMMAND.txt'
command_id='fixed6_quality_free_'+STAMP
try:
    result=subprocess.run(['/usr/bin/python3','-B',str(wrapper),'--id',command_id,'--command-file',str(command)],capture_output=True,text=True,timeout=155)
    (OUT/'GPU77_WRAPPER_TERMINAL.json').write_text(json.dumps(dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr),indent=2)+'\n')
    receipt=HERE/'gpu77_connection_recovery_v1/commands'/command_id/'RECEIPT.json'
    if result.returncode==0 and receipt.is_file():
        transport=json.loads(receipt.read_text())
        rows=[json.loads(line) for line in transport['stdout'].splitlines() if line.startswith('{') and 'fixed6-owned-process-and-epoch-observation-v1' in line]
        if len(rows)!=1:raise ValueError('No unique complete owned observation')
        row=rows[0]
        summary=dict(UTC=row['UTC'],quality_fields_opened=False,errors=row['transient_monitoring_errors'],lanes=[dict(lane=x['lane'],owner_alive=x['current_owner_identity'] is not None,failure=x['owned_owner_failure'],closed=x['lane_closure_present'],cells=[dict(cell_id=c['cell_id'],launched=c['launched'],alive=c['current_child_identity'] is not None,progress=c['epoch_progress']) for c in x['cells']]) for x in row['lanes']])
        (OUT/'GPU77_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary))
    else:print(json.dumps(dict(gpu77_observation_failed=True,exit_code=result.returncode,job_failure_inferred=False,receipt=str(receipt))))
except subprocess.TimeoutExpired:
    (OUT/'GPU77_OBSERVATION_TIMEOUT.json').write_text(json.dumps(dict(timeout=155,job_failure_inferred=False))+'\n')
    print(json.dumps(dict(gpu77_observation_timeout=True,job_failure_inferred=False)))
