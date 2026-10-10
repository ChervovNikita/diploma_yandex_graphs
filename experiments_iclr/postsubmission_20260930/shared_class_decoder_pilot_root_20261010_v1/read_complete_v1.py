"""Read the full fixed family after closure, retaining all compact partitions."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import zlib

HERE = Path(__file__).resolve().parent
LOCAL = HERE.parent / 'shared_class_decoder_complete_reader_20261010_v1'
REMOTE = r'''
import base64,hashlib,json,os,socket,subprocess,time,zlib
from datetime import datetime,timezone
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
u='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[u]
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930';H=P/'shared_class_decoder_pilot_root_20261010_v1'
end=H/'OWNER_END.json'
if not end.exists():
    start=json.loads((H/'OWNER_START.json').read_text())
    assert Path('/proc',str(start['child']['PID']),'stat').exists()
    print('READOUT_JSON='+json.dumps(dict(running=True,progress=json.loads((H/'actual_family_v1/PROGRESS.json').read_text()))))
    raise SystemExit(0)
owner=json.loads(end.read_text())
assert owner['scientific_success'] and owner['complete_family'] and owner['direct_child_wait'] and owner['child_pid_absent'] and owner['owned_cuda_pid_absent']
base=H/'complete_readout_v1';report=base/'COMPLETE_ANALYSIS.json';receipt=H/'READOUT_RECEIPT_V1.json'
if not receipt.exists():
    assert not report.exists()
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    start=time.monotonic();utc=datetime.now(timezone.utc).isoformat()
    proc=subprocess.run([str(R/'.venv/bin/python'),'-B',str(P/'shared_class_decoder_complete_reader_20261010_v1/analysis.py'),'--report',str(report)],cwd=R,env=env,text=True,capture_output=True,timeout=120)
    row=dict(start_UTC=utc,terminal_UTC=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-start,
             exit_code=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,new_training=False,new_model_forward=False,TEST_access=False)
    receipt.write_text(json.dumps(row,indent=2)+'\n')
row=json.loads(receipt.read_text())
if row['exit_code']!=0:
    print('READOUT_JSON='+json.dumps(dict(running=False,failed=True,receipt=row)))
    raise SystemExit(0)
summary=json.loads((base/'COMPLETE_ANALYSIS_SUMMARY.json').read_text())
assert summary['complete'] and summary['new_optimizer_fits']==30 and summary['new_native_bodies_fitted']==39 and summary['banks']==36 and summary['TEST_access'] is False
rows=[]
for item in summary['partitions']:
    q=Path(item['path']).resolve();assert q.is_relative_to(base.resolve())
    data=q.read_bytes();assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
    rows.append(dict(name=q.name,sha256=item['sha256'],data=base64.b64encode(data).decode()))
for name,q in [('COMPLETE_ANALYSIS_SUMMARY.json',base/'COMPLETE_ANALYSIS_SUMMARY.json'),('OWNER_END.json',end),('READOUT_RECEIPT_V1.json',receipt)]:
    data=q.read_bytes();rows.append(dict(name=name,sha256=hashlib.sha256(data).hexdigest(),data=base64.b64encode(data).decode()))
payload=base64.b64encode(zlib.compress(json.dumps(rows).encode())).decode()
print('READOUT_JSON='+json.dumps(dict(running=False,failed=False,payload=payload,complete30=True)))
'''
args=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
      '-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes',
      '-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
      'python3 -c '+shlex.quote(REMOTE)]
proc=subprocess.run(args,text=True,capture_output=True)
assert proc.returncode==0,proc.stderr[-2000:]
lines=[s for s in proc.stdout.splitlines() if s.startswith('READOUT_JSON=')]
assert len(lines)==1
value=json.loads(lines[0].split('=',1)[1])
if value.get('running') or value.get('failed'):
    print(json.dumps(value,indent=2))
    if value.get('failed'):
        (HERE/'READOUT_FAILURE_V1.json').write_text(json.dumps(value,indent=2)+'\n')
    raise SystemExit(0)
rows=json.loads(zlib.decompress(base64.b64decode(value['payload'])))
receipts=[]
for row in rows:
    name=row['name'];assert Path(name).name==name and name.endswith('.json')
    data=base64.b64decode(row['data']);assert hashlib.sha256(data).hexdigest()==row['sha256']
    target=(HERE if name=='OWNER_END.json' else LOCAL)/name
    if target.exists():
        assert target.read_bytes()==data
    else:
        target.write_bytes(data)
    receipts.append(dict(name=name,bytes=len(data),sha256=row['sha256']))
(LOCAL/'COMPLETE_FETCH_V1.json').write_text(json.dumps(dict(complete=True,files=receipts,raw_arrays_fetched=False,raw_checkpoints_fetched=False),indent=2)+'\n')
summary=json.loads((LOCAL/'COMPLETE_ANALYSIS_SUMMARY.json').read_text())
print(json.dumps(dict(complete=True,files=len(receipts),new_fits=summary['new_optimizer_fits'],
                     frozen_decision=summary['frozen_decision'],all_arm_aggregates=summary['all_arm_aggregates']),indent=2))
