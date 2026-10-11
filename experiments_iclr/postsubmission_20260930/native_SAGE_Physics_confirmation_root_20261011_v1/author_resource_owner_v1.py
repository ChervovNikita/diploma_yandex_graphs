"""Finite own-child custodian; no model or score operations."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930';H=Path(__file__).resolve().parent
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
def write(p,r):p.write_text(json.dumps(r,indent=2)+'\n')
def identity(pid):
    q=Path('/proc')/str(pid)/'stat'
    if not q.exists():return None
    a=q.read_text().rsplit(')',1)[1].split();return dict(PID=pid,state=a[0],start_ticks=int(a[19]))
def main():
    assert socket.gethostname()=='anogena-2-0' and Path.cwd()==R and H.is_relative_to(R)
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[UUID]
    side=H/'author_resource_owner_v1';side.mkdir(exist_ok=False)
    cmd=[str(R/'.venv/bin/python'),'-B',str(H/'run_author_resources_v1.py')]
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    start=time.monotonic();timed_out=False
    with (side/'stdout.log').open('x') as out,(side/'stderr.log').open('x') as err:
        child=subprocess.Popen(cmd,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True);ident=identity(child.pid)
        write(side/'OWNER_START.json',dict(UTC=datetime.now(timezone.utc).isoformat(),child=ident,owner=identity(os.getpid()),finite_seconds=600,scientific_fits=0))
        try:code=child.wait(timeout=600)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=15)
    q=H/'author_resources_qualification_v1/AUTHOR_MEANSAGE_RESOURCE_QUALIFICATION.json';result=json.loads(q.read_text()) if q.exists() else {}
    success=code==0 and result.get('status')=='PASS' and identity(child.pid) is None
    write(side/'OWNER_END.json',dict(UTC=datetime.now(timezone.utc).isoformat(),status='COMPLETE_EXIT_0' if success else 'FAILED_RETAINED',exit_code=code,timed_out=timed_out,
        child=ident,direct_child_wait=True,child_pid_absent=identity(child.pid) is None,seconds=time.monotonic()-start,scientific_fits=0,TEST_scoring=False))
    return 0 if success else 1
if __name__=='__main__':raise SystemExit(main())
