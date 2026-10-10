"""Start the one-shot complete-reader watcher on the authorized route."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
body = (HERE / 'reader_after_closure_v1.py').read_bytes()
REMOTE = r'''
import base64,hashlib,json,socket,subprocess,time
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930'
H=P/'SAGE_GNCL_SupCon_2x2_pilot_root_20261010_v1'
receipt=json.loads((P/'publication/SAGE_factorial_actual_start_and_complete_reader_20261010_v1/PUSH_RECEIPT.json').read_text())
assert receipt['verified']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==receipt['remote_commit']
data=base64.b64decode(ENCODED);assert hashlib.sha256(data).hexdigest()==DIGEST
target=H/'reader_after_closure_v1.py'
if target.exists():assert target.read_bytes()==data
else:target.write_bytes(data)
if not (H/'READER_JOB_START_V1.json').exists():
 assert not (H/'READER_WATCH_LAUNCH_V1.json').exists()
 with (H/'reader_watch.stdout.log').open('x') as out,(H/'reader_watch.stderr.log').open('x') as err:
  proc=subprocess.Popen([str(R/'.venv/bin/python'),'-B',str(target)],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 for _ in range(50):
  if (H/'READER_JOB_START_V1.json').exists() or proc.poll() is not None:break
  time.sleep(.1)
 assert (H/'READER_JOB_START_V1.json').exists(),(H/'reader_watch.stderr.log').read_text()[-1800:]
 result={'existing':False,'watcher_PID':proc.pid,'start':json.loads((H/'READER_JOB_START_V1.json').read_text()),'watcher_source_sha256':DIGEST,'push_receipt':receipt}
 (H/'READER_WATCH_LAUNCH_V1.json').write_text(json.dumps(result,indent=2)+'\n')
else:result=json.loads((H/'READER_WATCH_LAUNCH_V1.json').read_text())
print('GNNM_READER_WATCH='+json.dumps(result))
'''.replace('ENCODED', repr(base64.b64encode(body).decode())).replace('DIGEST', repr(hashlib.sha256(body).hexdigest()))
command = ['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
           '-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes',
           '-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
           'python3 -c ' + shlex.quote(REMOTE)]
proc = subprocess.run(command, text=True, capture_output=True, timeout=55)
(HERE / 'READER_WATCH_TRANSPORT_V1.json').write_text(json.dumps(dict(exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr), indent=2)+'\n')
assert proc.returncode == 0, proc.stderr[-1800:]
lines = [s.split('=',1)[1] for s in proc.stdout.splitlines() if s.startswith('GNNM_READER_WATCH=')]
assert len(lines) == 1
result = json.loads(lines[0])
(HERE / 'ACTUAL_READER_WATCH_LAUNCH_V1.json').write_text(json.dumps(result, indent=2)+'\n')
publish = HERE.parent / 'publication/SAGE_factorial_actual_start_and_complete_reader_20261010_v1'
(publish / 'PUSH_RECEIPT.json').write_text(json.dumps(result['push_receipt'], indent=2)+'\n')
print(json.dumps(result, indent=2))
