"""Use existing finite owner after one operator integration on the authorized allocation."""
import base64,json,shlex,subprocess
from datetime import datetime,timezone
from pathlib import Path

REMOTE=r'''
import base64,hashlib,json,os,socket,subprocess,signal,sys,time
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
uuid='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[uuid]
r=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');os.chdir(r)
h=r/'experiments_iclr/postsubmission_20260930/private_feature_rotation_pilot_decision_20261010_v1'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==sys.argv[1]
for backbone in ('GAT','SAGE'):
 frozen=json.loads((h/backbone/'FREEZE.json').read_text())
 for row in frozen['bound_files']:assert hashlib.sha256((r/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
assert not (h/'ACTUAL_QUALIFICATION.json').exists()
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>=32768,'Less than32GiB free: defer without touching other jobs'
env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
start=time.monotonic()
with (h/'qualification.stdout.log').open('x') as out,(h/'qualification.stderr.log').open('x') as err:
 child=subprocess.Popen([str(r/'.venv/bin/python'),'-B',str(h/'qualify.py')],cwd=r,env=env,stdout=out,stderr=err,start_new_session=True)
 timed_out=False
 try:code=child.wait(timeout=300)
 except subprocess.TimeoutExpired:
  timed_out=True;os.killpg(child.pid,signal.SIGTERM)
  try:code=child.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=15)
report_path=h/'actual_qualification_v1/REPORT.json'
value=json.loads(report_path.read_text()) if code==0 and report_path.exists() else {'passed':False}
value.update(exit_code=code,timed_out=timed_out,direct_child_wait=True,child_pid=child.pid,child_pid_absent=not Path('/proc',str(child.pid)).exists(),owned_cuda_pid_absent=str(child.pid) not in [x.strip() for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()],total_seconds=time.monotonic()-start,source_commit=sys.argv[1],hostname=socket.gethostname(),gpu_uuid=uuid,TEST_access=False)
value['passed']=bool(value['passed'] and code==0 and value['child_pid_absent'] and value['owned_cuda_pid_absent'])
(h/'ACTUAL_QUALIFICATION.json').write_text(json.dumps(value,indent=2)+'\n')
if not value['passed']:
 print(json.dumps({'qualification':value,'error_tail':(h/'qualification.stderr.log').read_text()[-6000:],'scientific_launches':[]}));raise SystemExit(1)
launches=[]
for backbone in ('GAT','SAGE'):
 q=h/backbone;assert not (q/'LAUNCH.json').exists()
 with (q/'owner.stdout.log').open('x') as out,(q/'owner.stderr.log').open('x') as err:
  owner=subprocess.Popen(['/usr/bin/python3','-I','-S','-B',str(q/'owner.py')],cwd=r,stdout=out,stderr=err,start_new_session=True)
  fields=Path('/proc',str(owner.pid),'stat').read_text().rsplit(')',1)[1].split()
 launch={'UTC':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'backbone':backbone,'PID':owner.pid,'start_ticks':int(fields[19]),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'hostname':socket.gethostname(),'gpu_uuid':uuid,'source_commit':sys.argv[1],'freeze_sha256':hashlib.sha256((q/'FREEZE.json').read_bytes()).hexdigest(),'TEST_access':False,'expected_new_fits':12,'detached':True}
 (q/'LAUNCH.json').write_text(json.dumps(launch,indent=2)+'\n');launches.append(launch)
print(json.dumps({'qualification':value,'scientific_launches':launches}))
'''

def main():
    here=Path(__file__).resolve().parent;phase=here.parent
    head=json.loads((phase/'publication/private_feature_rotation_pilot_and_companion_results_20261010_v1/COMMIT_RECEIPT.json').read_text())['commit']
    argv=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,head])]
    result=subprocess.run(argv,capture_output=True,text=True,timeout=360)
    transport={'UTC':datetime.now(timezone.utc).isoformat(),'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
    (here/'ACTUAL_ACTIVATION_TRANSPORT.json').write_text(json.dumps(transport,indent=2)+'\n')
    if result.returncode==0:
        value=json.loads(result.stdout);(here/'ACTUAL_QUALIFICATION.json').write_text(json.dumps(value['qualification'],indent=2)+'\n')
        for row in value['scientific_launches']:(here/row['backbone']/'ACTUAL_LAUNCH.json').write_text(json.dumps(row,indent=2)+'\n')
        print(json.dumps({'qualification_passed':True,'cases':len(value['qualification']['cases']),'scientific_launches':value['scientific_launches']}))
    else:print(json.dumps(transport))
    return result.returncode

if __name__=='__main__':raise SystemExit(main())
