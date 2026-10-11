"""Finite direct-child owner; only the bounded engine/resource checks."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
H=Path(__file__).resolve().parent
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'

def stamp():return datetime.now(timezone.utc).isoformat()
def ident(pid):
    p=Path('/proc')/str(pid)/'stat'
    if not p.exists():return None
    a=p.read_text().rsplit(')',1)[1].split();return dict(PID=pid,state=a[0],start_ticks=int(a[19]))
def write(p,r):p.write_text(json.dumps(r,indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['engine','resources'],required=True);a=ap.parse_args()
    assert socket.gethostname()=='anogena-2-0' and Path.cwd()==R and H.is_relative_to(R)
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[UUID]
    cfgpath=H/(a.stage.upper()+'_CONFIG_V3.json');cfg=json.loads(cfgpath.read_text());timeout=cfg['family_limits']['wall_seconds']
    side=H/(a.stage+'_owner_v3');side.mkdir(exist_ok=False)
    cmd=[str(R/'.venv/bin/python'),'-B',str(H/'run_qualification_v3.py'),'--stage',a.stage]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',DGLBACKEND='pytorch')
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    start=time.monotonic();timed_out=False
    with (side/'stdout.log').open('x') as out,(side/'stderr.log').open('x') as err:
        child=subprocess.Popen(cmd,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
        identity=ident(child.pid)
        write(side/'OWNER_START.json',dict(UTC=stamp(),owner=ident(os.getpid()),child=identity,finite_seconds=timeout,stage=a.stage,command=cmd))
        try:code=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=15)
    cuda_pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()
    output=H/(a.stage+'_qualification_v3')/('ENGINE_EQUIVALENCE.json' if a.stage=='engine' else 'ARXIV_RESOURCE_QUALIFICATION.json')
    result=json.loads(output.read_text()) if output.exists() else {}
    success=code==0 and result.get('status')=='PASS' and ident(child.pid) is None and str(child.pid) not in [s.strip() for s in cuda_pids]
    write(side/'OWNER_END.json',dict(UTC=stamp(),status='COMPLETE_EXIT_0' if success else 'FAILED_RETAINED',exit_code=code,timed_out=timed_out,
        child=identity,direct_child_wait=True,child_pid_absent=ident(child.pid) is None,owned_cuda_pid_absent=str(child.pid) not in [s.strip() for s in cuda_pids],
        seconds=time.monotonic()-start,stage=a.stage,scientific_fits=0,TEST_scoring=False,
        qualification_sha256=hashlib.sha256(output.read_bytes()).hexdigest() if output.exists() else None))
    return 0 if success else 1

if __name__=='__main__':raise SystemExit(main())
