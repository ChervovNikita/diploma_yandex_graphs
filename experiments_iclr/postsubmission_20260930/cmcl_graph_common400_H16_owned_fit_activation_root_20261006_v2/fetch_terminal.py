"""Fetch the existing CMCL attempt's small terminal metadata; never launch jobs."""
from pathlib import Path
from datetime import datetime, timezone
import base64
import hashlib
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
REMOTE = r'''
from pathlib import Path
import base64, hashlib, json, socket
assert socket.gethostname() == 'anogena-2-0'
p = Path('experiments_iclr/postsubmission_20260930/cmcl_graph_common400_H16_owned_fit_execution_root_20261006_v2')
if not (p/'TERMINAL.json').is_file():
 f=p/'fit/RESULT.json'; x=json.loads(f.read_bytes()) if f.is_file() else {}
 print(json.dumps({'terminal':False,'status':x.get('status'),'counts':x.get('counts')}))
else:
 rows={}
 for name in ('TERMINAL.json','SUPERVISOR_STATUS.json','fit/RESULT.json','fit/TRACE.jsonl','fit/CONTEXT.json','supervisor.stdout.log','supervisor.stderr.log'):
  f=p/name
  if f.is_file():
   assert not f.is_symlink() and f.resolve().is_relative_to(p.resolve())
   b=f.read_bytes(); assert len(b)<2000000
   rows[name]={'data':base64.b64encode(b).decode(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
 print(json.dumps({'terminal':True,'files':rows}))
'''

command = 'cd ' + shlex.quote(REPO) + ' && ' + shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE])
r = subprocess.run(['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
    '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=yes',
    'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',command],
    capture_output=True,text=True,timeout=30)
if r.returncode:
    raise RuntimeError('Observation failed; do not restart the existing fit: '+r.stderr)
x = json.loads(r.stdout)
if not x['terminal']:
    print(json.dumps(x))
else:
    output=HERE/'RESULTS'; output.mkdir(exist_ok=False); rows=[]
    for name,item in x['files'].items():
        data=base64.b64decode(item['data']); digest=hashlib.sha256(data).hexdigest()
        assert len(data)==item['bytes'] and digest==item['sha256']
        target=output/name; assert target.resolve().is_relative_to(output)
        target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data); target.chmod(0o444)
        rows.append({'path':name,'bytes':len(data),'sha256':digest})
    receipt={'UTC':datetime.now(timezone.utc).isoformat(),'files':rows,
             'server_checkpoints_fetched':False,'training_or_scoring':False}
    with (HERE/'FETCH_RECEIPT.json').open('x') as f:
        json.dump(receipt,f,indent=2); f.write('\n')
    worker=json.loads((output/'fit/RESULT.json').read_bytes())
    terminal=json.loads((output/'TERMINAL.json').read_bytes())
    print(json.dumps({'terminal':True,'worker_status':worker.get('status'),
        'counts':worker.get('counts'),'worker_errors':{k:worker[k] for k in worker if 'error' in k},
        'terminal_status':terminal.get('status'),
        'terminal_errors':{k:terminal[k] for k in terminal if 'error' in k},
        'children':terminal.get('children'),'files':rows}))
