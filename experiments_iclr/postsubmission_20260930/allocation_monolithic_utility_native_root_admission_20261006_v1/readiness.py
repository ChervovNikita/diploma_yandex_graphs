"""Verify the authorized allocation and read only known qualification metadata."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shlex, subprocess

D = Path(__file__).resolve().parent
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
REMOTE = r'''
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,socket,subprocess
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd()==repo and socket.gethostname()=='anogena-2-0'
rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15).strip().splitlines()
assert len(rows)==1
a=[x.strip() for x in rows[0].split(',')]
assert a[0]=='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
binary=phase/'native_ncn_runtime_20261005_v1/.venv/bin/python'
assert binary.resolve().is_relative_to(repo)
b=binary.read_bytes()
assert hashlib.sha256(b).hexdigest()=='6ff97f602038740073dca96714310a30e303332326268e0f1bb2767edc820944'
observations=[]
for name in ('allocation_streamed_utility_native_joint_oracle_execution_root_20261006_v1/owned_run01/TERMINAL.json','allocation_native16_first_order_execution_root_20261006_v1/supervisor/RESULT.json'):
 q=phase/name
 assert q.is_file() and not q.is_symlink() and q.resolve().is_relative_to(phase) and q.stat().st_size<200000
 raw=q.read_bytes();v=json.loads(raw)
 observations.append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'status':v.get('status'),'exit_code':v.get('exit_code')})
print(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'hostname':socket.gethostname(),'repo':str(repo),'GPU':{'UUID':a[0],'name':a[1],'total_MiB':int(a[2]),'free_MiB':int(a[3]),'utilization_percent':int(a[4])},'python_executable':str(binary),'python_resolved':str(binary.resolve()),'python_binary_bytes':len(b),'python_binary_sha256':hashlib.sha256(b).hexdigest(),'known_terminal_observations':observations,'no_payload_runtime_import_or_setting_changes':True}))
'''

def main():
    target=D/'READINESS_TRANSPORT_RECEIPT.json'
    assert not target.exists()
    command=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','cd '+shlex.quote(REPO)+' && /usr/bin/python3 -I -S -B -c '+shlex.quote(REMOTE)]
    r=subprocess.run(command,capture_output=True,text=True,timeout=45)
    receipt={'UTC':datetime.now(timezone.utc).isoformat(),'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'remote_source_sha256':hashlib.sha256(REMOTE.encode()).hexdigest(),'local_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    target.write_text(json.dumps(receipt,indent=2)+'\n')
    if r.returncode:
        print(json.dumps(receipt));raise SystemExit(r.returncode)
    value=json.loads(r.stdout)
    out=D/'READINESS.json';assert not out.exists()
    out.write_text(json.dumps(value,indent=2)+'\n');out.chmod(0o444)
    print(json.dumps(value))

if __name__=='__main__':main()
