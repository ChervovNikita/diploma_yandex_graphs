"""One finite wait followed by the already fixed three-bank PE acquisition."""
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
P=R/'experiments_iclr/postsubmission_20260930'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
def now():return datetime.now(timezone.utc).isoformat()
def item(p):
    p=Path(p).resolve();assert p.is_relative_to(R);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,v):assert not p.exists();p.write_text(json.dumps(v,indent=2)+'\n')

def main():
    assert socket.gethostname()=='anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[UUID]
    H=Path(__file__).resolve().parent;assert H.is_relative_to(P)
    E=P/'native_SAGE_endpoint_value_execution_root_20261011_v1'
    q=json.loads((H/'qualify_owner_v1/OWNER_END.json').read_text());assert q['success']
    assert json.loads((H/'qualify_v1/QUALIFICATION.json').read_text())['status']=='PASS'
    fixed=[H/'owner.py',H/'PROSPECTIVE_COMPARISON.json']
    bindings=[item(p) for p in fixed]
    write(H/'FIT_WATCH_START_V1.json',dict(UTC=now(),PID=os.getpid(),qualification_owner=item(H/'qualify_owner_v1/OWNER_END.json'),bound_sources=bindings,finite_wait_seconds=87000,fit_cap_seconds=7200,partial_predictive_metrics_read=False,TEST_access=False))
    start=time.monotonic();endpath=E/'fit_owner_v1/OWNER_END.json'
    while not endpath.exists():
        if time.monotonic()-start>87000:raise TimeoutError('Existing endpoint native owner did not close; no PE fit started')
        time.sleep(30)
    assert [item(p) for p in fixed]==bindings
    e=json.loads(endpath.read_text())
    if not e['scientific_success']:
        write(H/'FIT_WATCH_END_V1.json',dict(UTC=now(),status='E_FAILURE_RETAINED_NO_PE_FIT',endpoint_owner=item(endpath)));return 1
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    command=[str(R/'.venv/bin/python'),'-B',str(H/'owner.py'),'--mode','fit']
    with (H/'fit_launch_stdout.log').open('x') as out,(H/'fit_launch_stderr.log').open('x') as err:
        child=subprocess.Popen(command,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
        write(H/'FIT_WATCH_LAUNCH_V1.json',dict(UTC=now(),owner_PID=child.pid,command=command,full_endpoint_native_closure=True,partial_predictive_metrics_read=False))
        timed_out=False
        try:code=child.wait(timeout=7260)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=15)
    ownerend=H/'fit_owner_v1/OWNER_END.json'
    write(H/'FIT_WATCH_END_V1.json',dict(UTC=now(),exit_code=code,timed_out=timed_out,owner_end=item(ownerend) if ownerend.exists() else None,TEST_access=False))
    return code

if __name__=='__main__':raise SystemExit(main())
