"""Launch one frozen equal-control study after reviewed-source publication."""
from pathlib import Path
from datetime import datetime,timezone
import json,shlex,subprocess

REMOTE=r'''
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,socket,subprocess,sys
assert socket.gethostname()=='anogena-2-0'
uuid='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[uuid]
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';H=P/'common_wrapper_graph_reliability_root_20261010_v2';os.chdir(R)
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==sys.argv[1]
frozen=json.loads((H/'FREEZE.json').read_text())
for row in frozen['bound_files']:assert hashlib.sha256((P/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
assert not (H/'LAUNCH.json').exists() and not (H/'actual_export_v1').exists() and not (H/'actual_study_v1').exists()
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>=8192,'Defer without touching other jobs'
with (H/'owner.stdout.log').open('x') as out,(H/'owner.stderr.log').open('x') as err:
 child=subprocess.Popen(['/usr/bin/python3','-I','-S','-B',str(H/'owner.py')],cwd=R,stdout=out,stderr=err,start_new_session=True)
f=Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()
v=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=child.pid,start_ticks=int(f[19]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),hostname=socket.gethostname(),gpu_uuid=uuid,source_commit=sys.argv[1],freeze_sha256=hashlib.sha256((H/'FREEZE.json').read_bytes()).hexdigest(),finite_seconds=21600,detached=True,TEST_access=False,base_training_fits=0,selected_export_calls=99,member_trajectories=126,fusion_fit_calls=855)
(H/'LAUNCH.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
'''

def main():
 h=Path(__file__).resolve().parent;p=h.parent
 head=json.loads((p/'publication/graph_fusion_descriptor_repair_and_closed_history_20261010_v1/COMMIT_RECEIPT.json').read_text())['commit']
 argv=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,head])]
 r=subprocess.run(argv,capture_output=True,text=True,timeout=45)
 value=dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)
 if r.returncode==0:
  value['result']=json.loads(r.stdout);(h/'ACTUAL_LAUNCH.json').write_text(json.dumps(value['result'],indent=2)+'\n')
 (h/'ACTUAL_ACTIVATION_TRANSPORT.json').write_text(json.dumps(value,indent=2)+'\n')
 print(json.dumps(value.get('result',value)));return r.returncode

if __name__=='__main__':raise SystemExit(main())
