"""Launch the frozen SAGE learning sparse feature-kernel once on the authorized scientific route."""
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REMOTE = r'''
import hashlib,json,os,socket,subprocess,time
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==[UUID]
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930'
H=P/'SAGE_sparse_feature_kernel_pilot_root_20261010_v1'
receipt=json.loads((P/'publication/SAGE_factorial_closed_kernel_qualified_frozen_20261010_v1/PUSH_RECEIPT.json').read_text())
assert receipt['verified']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==receipt['remote_commit']
start=H/'OWNER_START.json'
if start.exists():
    result=dict(existing=True,owner_start=json.loads(start.read_text()),push_receipt=receipt)
else:
    assert not (H/'actual_family_v1').exists()
    assert not (H/'LAUNCH_V1.json').exists()
    with (H/'owner.stdout.log').open('x') as out,(H/'owner.stderr.log').open('x') as err:
        proc=subprocess.Popen([str(R/'.venv/bin/python'),'-B',str(H/'owner.py')],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
    for _ in range(50):
        if start.exists() or proc.poll() is not None:
            break
        time.sleep(.1)
    assert start.exists(),dict(returncode=proc.poll(),stderr=(H/'owner.stderr.log').read_text()[-3000:])
    result=dict(existing=False,owner_start=json.loads(start.read_text()),push_receipt=receipt,
                owner_launcher_PID=proc.pid,freeze_sha256=hashlib.sha256((H/'FREEZE.json').read_bytes()).hexdigest())
    (H/'LAUNCH_V1.json').write_text(json.dumps(result,indent=2)+'\n')
result['owner_end']=json.loads((H/'OWNER_END.json').read_text()) if (H/'OWNER_END.json').exists() else None
result['worker_stderr_tail']=(H/'worker.stderr.log').read_text()[-2000:] if (H/'worker.stderr.log').exists() else None
print('GNNM_LAUNCH_JSON='+json.dumps(result))
'''
args=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
      '-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes',
      '-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
      'python3 -c '+shlex.quote(REMOTE)]
proc=subprocess.run(args,text=True,capture_output=True)
(HERE/'LAUNCH_TRANSPORT_V1.json').write_text(json.dumps(dict(exit_code=proc.returncode,stdout=proc.stdout,stderr=proc.stderr),indent=2)+'\n')
assert proc.returncode==0,proc.stderr[-2000:]
lines=[x for x in proc.stdout.splitlines() if x.startswith('GNNM_LAUNCH_JSON=')]
assert len(lines)==1
result=json.loads(lines[0].split('=',1)[1])
(HERE/'ACTUAL_LAUNCH_V1.json').write_text(json.dumps(result,indent=2)+'\n')
push_root=HERE.parent/'publication/SAGE_factorial_closed_kernel_qualified_frozen_20261010_v1'
(push_root/'PUSH_RECEIPT.json').write_text(json.dumps(result['push_receipt'],indent=2)+'\n')
print(json.dumps(result,indent=2))
