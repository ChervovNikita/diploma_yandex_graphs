"""Launch the three frozen native families with the existing owners."""
from pathlib import Path
from datetime import datetime,timezone
import json,shlex,subprocess
REMOTE=r"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,socket,subprocess,sys
assert socket.gethostname()=='anogena-2-0'
uuid='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[uuid]
r=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');p=r/'experiments_iclr/postsubmission_20260930';h=p/'nonlocal_label_retrieval_pilot_root_20261010_v1';os.chdir(r)
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==sys.argv[1]
assert not (h/'ALL_LAUNCH.json').exists()
qualified=json.loads((h/'ACTUAL_QUALIFICATION_V2.json').read_text());assert qualified['qualified'] and len(qualified['cases'])==9 and not qualified['scientific_quality_evidence']
for b in ('SAGE','GCN','GAT'):
 d=h/b;assert not(d/'LAUNCH.json').exists() and not(d/'actual_family_v1').exists()
 for row in json.loads((d/'FREEZE.json').read_text())['bound_files']:assert hashlib.sha256((r/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())>=24576,'Defer without stopping other jobs'
records=[]
for b in ('SAGE','GCN','GAT'):
 d=h/b
 with(d/'owner.stdout.log').open('x')as out,(d/'owner.stderr.log').open('x')as err:
  process=subprocess.Popen(['/usr/bin/python3','-I','-S','-B',str(d/'owner.py')],cwd=r,stdout=out,stderr=err,start_new_session=True)
 f=Path('/proc',str(process.pid),'stat').read_text().rsplit(')',1)[1].split()
 v=dict(UTC=datetime.now(timezone.utc).isoformat(),backbone=b,PID=process.pid,start_ticks=int(f[19]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),hostname=socket.gethostname(),gpu_uuid=uuid,source_commit=sys.argv[1],freeze_sha256=hashlib.sha256((d/'FREEZE.json').read_bytes()).hexdigest(),finite_seconds=21600,fit_units=21,groups=12,detached=True,TEST_access=False)
 (d/'LAUNCH.json').write_text(json.dumps(v,indent=2)+'\n');records.append(v)
value=dict(UTC=datetime.now(timezone.utc).isoformat(),families=records,new_fit_units=63,new_banks=36,all_complete_before_comparison=True,TEST_access=False,concurrent=True)
(h/'ALL_LAUNCH.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
"""
def main():
 h=Path(__file__).resolve().parent;p=h.parent
 head=json.loads((p/'publication/nonlocal_label_retrieval_actual_qualification_and_family_freezes_20261010_v1/COMMIT_RECEIPT.json').read_text())['commit']
 argv=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,head])]
 r=subprocess.run(argv,capture_output=True,text=True,timeout=45)
 v=dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)
 if r.returncode==0:v['result']=json.loads(r.stdout);(h/'ACTUAL_ALL_LAUNCH.json').write_text(json.dumps(v['result'],indent=2)+'\n')
 (h/'ACTIVATION_TRANSPORT.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v.get('result',v)));return r.returncode
if __name__=='__main__':raise SystemExit(main())
