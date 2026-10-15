"""Finite owner for one already frozen nine-fit scientific development study."""
import hashlib,json,os,signal,socket,subprocess,sys,time
from pathlib import Path
ROOT=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
EXP=ROOT/'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SPEC_SHA='eacd22be19a1cc108d816cd062f87dde902a4ad67aab6fe31301730c663c2d6b'
def main():
 assert socket.gethostname()=='anogena-2-0'
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[UUID]
 source=EXP/'pilot_20261015_bn_geometry_correction_v1';admission=EXP/'qualification/BN_GEOMETRY_ROOT_ADMISSION_v1.json'
 a=json.loads(admission.read_text());assert a['SPEC_sha256']==SPEC_SHA and a['approved']is True and a['full_campaign_GO']is False
 assert hashlib.sha256((source/'SPEC.json').read_bytes()).hexdigest()==SPEC_SHA
 owner=EXP/'results/bn_geometry_followup_v1_owner';out=EXP/'results/bn_geometry_followup_v1';owner.mkdir(exist_ok=False)
 assert not out.exists()
 command=[str(ROOT/'.venv/bin/python'),'-B','-u',str(source/'study.py'),'--spec',str(source/'SPEC.json'),'--spec-sha256',SPEC_SHA,'--mode','launch','--output',str(out),'--launch-authority',str(admission)]
 r={'status':'starting','owner_pid':os.getpid(),'SPEC_sha256':SPEC_SHA,'authority':a,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'maximum_wall_seconds':43200,'command':command,'TEST_accessed':False,'full_campaign_GO':False,'automatic_retries':0}
 def save():
  p=owner/'OWNER.json.tmp';p.write_text(json.dumps(r,indent=2)+'\n');p.replace(owner/'OWNER.json')
 start=time.monotonic();save();child=None
 try:
  with(owner/'study.log').open('xb')as log:
   child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1'))
  r.update(status='running',child_pid=child.pid,child_pgid=child.pid);save()
  try:child.wait(timeout=43200)
  except subprocess.TimeoutExpired:
   r['timeout']=True;os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait();r['forced_KILL']=True
  r.update(exit_code=child.returncode,status='finished'if child.returncode==0 else'failed_or_timeout')
  p=out/'END.json';r['study_terminal']=json.loads(p.read_text())if p.exists()else None
  if child.returncode==0 and(not r['study_terminal']or r['study_terminal'].get('status')!='complete'):r['status']='failed_missing_complete_terminal'
 finally:
  if child is not None and child.poll()is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  r['wall_seconds']=time.monotonic()-start;r['raw_outputs_preserved']=True
  r['incomplete_work_cost']='Read every existing fit RESULT/PROGRESS/CURVE. Missing terminal work is unknown, never zero.'
  save()
 return 0 if r['status']=='finished'else 1
if __name__=='__main__':sys.exit(main())
