"""Slim direct-child owner: 600-second child, 660-second inclusive envelope."""
import hashlib,json,os,signal,socket,subprocess,time
from pathlib import Path
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
H=Path(__file__).resolve().parent
def identity(pid):
 p=Path('/proc')/str(pid)/'stat'
 if not p.exists():return None
 a=p.read_text().rsplit(')',1)[1].split()
 return dict(pid=pid,start_ticks=int(a[19]),pgid=int(a[2]))
def write(p,v):
 with p.open('x') as f:f.write(json.dumps(v,indent=2)+'\n')
assert socket.gethostname()=='anogena-2-0' and Path.cwd()==R
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==[UUID]
spec_raw=(H/'SPEC.json').read_bytes()
env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
started=time.monotonic();deadline=started+660
argv=[str(R/'.venv/bin/python'),'-B','-u',str(H/'worker.py')]
result=dict(status='RUNNING',external_wall_seconds=660,child_wall_seconds=600,direct_child_wait=True,
 spec_sha256=hashlib.sha256(spec_raw).hexdigest(),retry=False,full_scientific_acquisition=False)
with (H/'child.log').open('xb') as log:
 child=subprocess.Popen(argv,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
 birth=identity(child.pid);assert birth and birth['pgid']==child.pid
 result.update(owner=identity(os.getpid()),child=birth,command=argv)
 write(H/'OWNER_START.json',result)
 print(json.dumps(dict(status='OWNER_STARTED',child_pid=child.pid,owner_pid=os.getpid(),external_wall_seconds=660)),flush=True)
 timed_out=False
 try:code=child.wait(timeout=max(.1,deadline-time.monotonic()-30))
 except subprocess.TimeoutExpired:
  timed_out=True
  assert identity(child.pid)==birth
  os.killpg(child.pid,signal.SIGTERM)
  try:code=child.wait(timeout=max(.1,min(15,deadline-time.monotonic())))
  except subprocess.TimeoutExpired:
   assert identity(child.pid)==birth
   os.killpg(child.pid,signal.SIGKILL)
   code=child.wait(timeout=max(.1,deadline-time.monotonic()))
 cuda_pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=15).splitlines()
 result.update(status='COMPLETE_EXIT_0' if code==0 and not timed_out else 'FAILED_RETAINED',
  exit_code=code,timed_out=timed_out,seconds=time.monotonic()-started,
  child_pid_absent=identity(child.pid) is None,owned_cuda_pid_absent=str(child.pid) not in [s.strip() for s in cuda_pids])
 write(H/'OWNER_END.json',result)
print(json.dumps(dict(owner_end=result)),flush=True)
readiness=H/'readiness_v1/FULL_INPUT_READINESS_COMPLETE.json'
failure=H/'readiness_v1/FAILURE.json'
if readiness.exists():print(json.dumps(dict(readiness=json.loads(readiness.read_text()))),flush=True)
if failure.exists():print(json.dumps(dict(failure=json.loads(failure.read_text()))),flush=True)
if code!=0:print((H/'child.log').read_text()[-14000:],flush=True)
raise SystemExit(0 if result['status']=='COMPLETE_EXIT_0' and result['child_pid_absent'] and result['owned_cuda_pid_absent'] else 1)
