"""Finite direct-child owner for qualification or the one complete VALUE family."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def stamp():return datetime.now(timezone.utc).isoformat()
def identity(pid):
    p=Path('/proc',str(pid),'stat')
    if not p.exists():return None
    a=p.read_text().rsplit(')',1)[1].split()
    return dict(PID=pid,start_ticks=int(a[19]),state=a[0])
def write(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['qualify','fit'],required=True);a=ap.parse_args()
    assert socket.gethostname()=='anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[UUID]
    H=Path(__file__).resolve().parent;S=H.parent/'native_SAGE_endpoint_value_transport_source_20261011_v1'
    assert H.is_relative_to(R) and Path.cwd()==R
    config=H/('QUALIFICATION_CONFIG_V1.json' if a.stage=='qualify' else 'SCIENTIFIC_CONFIG_V1.json')
    cfg=json.loads(config.read_text())
    for row in cfg['source_receipts'].values():assert digest(Path(row['path']))==row['sha256']
    side=H/(a.stage+'_owner_v1');side.mkdir(exist_ok=False)
    output=H/('qualify_v1' if a.stage=='qualify' else 'actual_family_v1');assert not output.exists()
    timeout=300 if a.stage=='qualify' else cfg['family_limits']['wall_seconds']
    cmd=[str(R/'.venv/bin/python'),'-B']
    cmd+=([str(H/'run_qualification_v1.py')] if a.stage=='qualify' else [str(S/'run_family.py'),'--config',str(config),'--output',str(output)])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',DGLBACKEND='pytorch')
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    started=time.monotonic()
    with (side/'stdout.log').open('x') as out,(side/'stderr.log').open('x') as err:
        child=subprocess.Popen(cmd,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
        ident=identity(child.pid)
        write(side/'OWNER_START.json',dict(UTC=stamp(),owner=identity(os.getpid()),child=ident,stage=a.stage,config_sha256=digest(config),finite_seconds=timeout,command=cmd,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
        timed_out=False
        try:code=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=15)
    cuda_pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()
    closed=identity(child.pid) is None and str(child.pid) not in [x.strip() for x in cuda_pids]
    fullpath=output/('QUALIFICATION.json' if a.stage=='qualify' else 'COMPLETE_FAMILY.json')
    full=json.loads(fullpath.read_text()) if fullpath.exists() else {}
    complete=(full.get('status')=='PASS' if a.stage=='qualify' else full.get('complete') is True and full.get('groups')==15 and full.get('fit_units')==24 and full.get('native_body_fit_records')==24)
    success=code==0 and complete and closed
    result=dict(UTC=stamp(),status='COMPLETE_EXIT_0' if success else 'FAILED_RETAINED',stage=a.stage,exit_code=code,timed_out=timed_out,
        direct_child_wait=True,child=ident,child_pid_absent=identity(child.pid) is None,owned_cuda_pid_absent=str(child.pid) not in [x.strip() for x in cuda_pids],
        seconds=time.monotonic()-started,study_kind='native_SAGE_endpoint_VALUE',config_sha256=digest(config),training_source_sha256=cfg['source_receipts']['training_family']['sha256'],
        runtime_fingerprint_sha256=json.loads(Path(cfg['root_admission']['path']).read_text())['runtime_fingerprint_sha256'],hostname=socket.gethostname(),gpu_uuid=UUID,
        complete_family_sha256=digest(fullpath) if fullpath.exists() else None,scientific_success=success,TEST_access=False,
        overlap='Normal sole allocation GPU, concurrent Penn94 acquisition; observed costs are not isolated speed evidence.')
    if a.stage=='fit':result.update(banks=full.get('groups'),optimizer_bundles=full.get('optimizer_acquisition_bundles'),native_body_records=full.get('native_body_fit_records'))
    write(side/'OWNER_END.json',result)
    return 0 if success else 1

if __name__=='__main__':raise SystemExit(main())
