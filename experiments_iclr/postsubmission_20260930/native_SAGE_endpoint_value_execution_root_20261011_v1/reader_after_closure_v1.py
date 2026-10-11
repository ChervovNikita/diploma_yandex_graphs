"""Run complete fixed readout once after this exact fit owner closes."""
from datetime import datetime, timezone
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
H=Path(__file__).resolve().parent
S=P/'native_SAGE_endpoint_value_transport_source_20261011_v2'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
def now():return datetime.now(timezone.utc).isoformat()
def item(p):
    p=Path(p).resolve();assert p.is_relative_to(R)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,v):
    assert not p.exists();p.write_text(json.dumps(v,indent=2)+'\n')

def main():
    assert socket.gethostname()=='anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[UUID]
    start=json.loads((H/'fit_owner_v1/OWNER_START.json').read_text())
    assert start['owner']['PID']==626845 and start['child']['PID']==626848
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==start['boot_id']
    cfg=H/'SCIENTIFIC_CONFIG_V1.json'
    assert item(cfg)['sha256']==start['config_sha256']
    reader=S/'analysis.py';math=P/'SAGE_matched_joint_reference_77_complete_reader_source_20261010_v1/analysis.py'
    assert item(reader)['sha256']=='f17dc5ac8e0f85add18e920decd867c650a050d522bc549f71be20cd7ce93c1f'
    assert item(math)['sha256']=='b2d4deb6e7a1b9c6da5597e851a42caf11ece728f13b95ebd4a3871b01941ef0'
    write(H/'READER_WATCH_START_V1.json',dict(UTC=now(),PID=os.getpid(),fit_owner=start,
        reader=item(reader),math_reader=item(math),wait_cap_seconds=87000,analysis_cap_seconds=3600,
        partial_predictive_metrics_read=False,TEST_access=False))
    began=time.monotonic();endpath=H/'fit_owner_v1/OWNER_END.json'
    while not endpath.exists():
        if time.monotonic()-began>87000:raise TimeoutError('Finite owner-closure wait expired; no readout attempted')
        time.sleep(30)
    end=json.loads(endpath.read_text())
    if not end.get('scientific_success') or end.get('status')!='COMPLETE_EXIT_0':
        write(H/'READER_WATCH_END_V1.json',dict(UTC=now(),status='FIT_FAILURE_RETAINED_NO_READOUT',owner_end=item(endpath),TEST_access=False));return 1
    assert end['child']['PID']==626848 and end['child']['start_ticks']==start['child']['start_ticks']
    assert end['child_pid_absent'] and end['owned_cuda_pid_absent']
    complete=H/'actual_family_v1/COMPLETE_FAMILY.json'
    assert item(complete)['sha256']==end['complete_family_sha256']
    spec=dict(schema_version=1,study_kind='native_SAGE_endpoint_VALUE',study_id='native_SAGE_endpoint_VALUE_20261011_v1',
        family_root=str(H/'actual_family_v1'),reader_source=item(reader),protocol=item(S/'PROTOCOL.json'),config=item(cfg),
        complete_family=item(complete),owner_end=item(endpath),math_reader=item(math))
    write(H/'COMPLETE_READER_INPUT_SPEC_V1.json',spec)
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    command=[str(R/'.venv/bin/python'),'-B',str(reader),'--input-spec',str(H/'COMPLETE_READER_INPUT_SPEC_V1.json'),'--output',str(H/'complete_analysis_v1')]
    with (H/'reader_stdout.log').open('x') as out,(H/'reader_stderr.log').open('x') as err:
        child=subprocess.Popen(command,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
        write(H/'READER_EXECUTION_START_V1.json',dict(UTC=now(),child_PID=child.pid,command=command,full_fit_closure_before_analysis=True,TEST_access=False))
        timed_out=False
        try:code=child.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=15)
    summary=H/'complete_analysis_v1/COMPLETE_ANALYSIS_SUMMARY.json'
    write(H/'READER_WATCH_END_V1.json',dict(UTC=now(),exit_code=code,timed_out=timed_out,
        complete_analysis=code==0 and summary.exists(),summary=item(summary) if summary.exists() else None,
        native_fits_added=0,native_forward_calls=0,TEST_access=False))
    return code

if __name__=='__main__':raise SystemExit(main())
