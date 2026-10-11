"""Launch the committed TRAIN-gradient diagnostic exactly once on the authorized route."""
import base64
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shlex
import zlib

H = Path(__file__).resolve().parent
P = H.parent
REMOTE = r'''
import base64,zlib,json,sys,socket,subprocess,pathlib,hashlib,datetime
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
R=pathlib.Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
H=P/'WikiCS_E_TRAIN_shared_gradient_cancellation_diagnostic_root_20261011_v1'
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
assert commit==payload['commit']
for item in payload['files']:
    path=P/item['relative'];assert path.resolve().is_relative_to(P)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
    assert subprocess.check_output(['git','show',commit+':experiments_iclr/postsubmission_20260930/'+item['relative']],cwd=R)==path.read_bytes()
assert not (H/'SCIENTIFIC_LAUNCH_V1.json').exists() and not (H/'OWNER_START_V1.json').exists() and not (H/'complete_v1').exists()
with (H/'owner_outer_v1.log').open('x') as log:
    owner=subprocess.Popen(['/usr/bin/python3','-I','-S','-B',str(H/'owner_v1.py')],cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
value=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner_pid=owner.pid,commit=commit,status='ADMITTED_TRAIN_GRADIENT_DIAGNOSTIC_LAUNCHED',banks=15,selected_states=24,TRAIN_backwards=51,model_fits=0,TEST_access=False,current_Physics_Penn_access=False,paper_review_started=False)
(H/'SCIENTIFIC_LAUNCH_V1.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value))
'''

def main():
    assert not (H / 'SCIENTIFIC_LAUNCH_V1.json').exists()
    commit = json.loads((H / 'COMMITTED_SOURCE_BINDING_V1.json').read_text())['commit']
    names = ['WikiCS_E_TRAIN_shared_gradient_cancellation_diagnostic_source_20261011_v1/diagnose.py', 'WikiCS_E_TRAIN_shared_gradient_cancellation_diagnostic_source_20261011_v1/PLAN.json', 'WikiCS_E_TRAIN_shared_gradient_cancellation_diagnostic_root_20261011_v1/ADMISSION_V1.json', 'WikiCS_E_TRAIN_shared_gradient_cancellation_diagnostic_root_20261011_v1/owner_v1.py']
    import hashlib
    payload = dict(commit=commit, files=[dict(relative=n, sha256=hashlib.sha256((P/n).read_bytes()).hexdigest()) for n in names])
    spec=importlib.util.spec_from_file_location('head_transport',P/'publication/publish_exact_inventory_v19.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    argv=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,'fixed-source-TRAIN-gradient-diagnostic','literal-route','exact-committed-source'])]
    result=module.terminal_transport(argv,input=base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode(),capture_output=True,text=True,timeout=60)
    (H/'LAUNCH_TRANSPORT_V1.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr),indent=2)+'\n')
    assert result.returncode==0,result.stderr[-2000:]+result.stdout[-2000:]
    value=json.loads(result.stdout);(H/'SCIENTIFIC_LAUNCH_V1.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))

if __name__=='__main__':main()
