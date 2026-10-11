"""Advance the exact Penn94 acquisition through fixed CAL and one CONFIRM release.

No partial native metric, calibration comparison or CONFIRM target is opened by
this watcher. Existing sealed stage gates own all native/role/readout operations.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930'
H=Path(__file__).resolve().parent
S=P/'native_SAGE_Penn94_confirmation_integrated_source_20261011_v1'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
CONFIG_SHA='1626a42bd1df0a0654d672e5a5a59fe76bfd53fc5dc8fd2cbfe2cfd339b303b9'
SEAL_SHA='6efb07fff02e86d06c705d90e13f8da87d75b7bebc3e5c468f91a52e02eba9a0'

def now():return datetime.now(timezone.utc).isoformat()
def describe(p):
    p=Path(p).resolve();assert p.is_relative_to(R)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,v):
    assert not p.exists();p.write_text(json.dumps(v,indent=2)+'\n')

def run_stage(stage,admission,output):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    command=[str(R/'.venv/bin/python'),'-B',str(S/'owner.py'),'--config',str(H/'QUALIFICATION_CONFIG_V2.json'),
        '--admission',str(admission),'--output',str(output),'--stage',stage]
    with (H/(stage+'_pipeline_stdout.log')).open('x') as out,(H/(stage+'_pipeline_stderr.log')).open('x') as err:
        child=subprocess.Popen(command,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
        write(H/(stage.upper()+'_PIPELINE_LAUNCH_V1.json'),dict(UTC=now(),owner_PID=child.pid,command=command,
            admission=describe(admission),CONFIRM_truth_access=stage=='confirm'))
        try:code=child.wait(timeout=1860)
        except subprocess.TimeoutExpired:
            # The sealed owner separately bounds its child to1800 seconds.
            # Preserve a missing terminal owner rather than start another stage.
            write(H/(stage.upper()+'_PIPELINE_OWNER_TIMEOUT_V1.json'),dict(UTC=now(),owner_PID=child.pid,
                next_stage_admitted=False,retry=False,CONFIRM_truth_access=stage=='confirm'))
            raise
    assert code==0,('Failed sealed stage owner; retain its receipts without retry',stage,code)
    return output.parent/(output.name+'_owner/OWNER_END.json')

def main():
    assert socket.gethostname()=='anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[UUID]
    assert H.is_relative_to(P) and Path.cwd()==R
    config=H/'QUALIFICATION_CONFIG_V2.json'
    assert describe(config)['sha256']==CONFIG_SHA and describe(S/'SEAL.json')['sha256']==SEAL_SHA
    acquire_owner=H/'native_acquisition_v1_owner'
    start=json.loads((acquire_owner/'OWNER_START.json').read_text())
    assert start['owner']['PID']==626530 and start['child']['PID']==626532
    assert start['custody']['config_sha256']==CONFIG_SHA
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==start['boot_id']
    assert not (H/'confirm_v1').exists() and not (H/'CONFIRM_ADMISSION_V1.json').exists()
    write(H/'PIPELINE_WATCH_START_V1.json',dict(UTC=now(),PID=os.getpid(),acquisition_owner=start,
        config_sha256=CONFIG_SHA,source_seal_sha256=SEAL_SHA,wait_cap_seconds=87000,
        policy='All70banks/100body terminal closure, then90fixed CAL endpoints, then one complete CONFIRM release',
        partial_predictive_metrics_read=False,no_NEW_fits_or_forwards=True))
    began=time.monotonic();end=acquire_owner/'OWNER_END.json'
    while not end.exists():
        if time.monotonic()-began>87000:raise TimeoutError('Finite acquisition closure wait expired; no CAL or CONFIRM release')
        time.sleep(30)
    sys.path.insert(0,str(S))
    spec=importlib.util.spec_from_file_location('penn_pipeline_sealed_contracts',S/'contracts.py')
    contracts=importlib.util.module_from_spec(spec);spec.loader.exec_module(contracts)
    owner=contracts.closed_owner(describe(end))
    assert owner['child']['PID']==626532 and owner['child']['start_ticks']==start['child']['start_ticks']
    native_path=H/'native_acquisition_v1/COMPLETE_NATIVE.json'
    native=contracts.close_native(describe(native_path))
    assert native['custody']['config_sha256']==CONFIG_SHA and native['placement_native_initial_bytes_equal']
    base=dict(admitted=True,config_sha256=CONFIG_SHA,integrated_source_seal_sha256=SEAL_SHA,
        source_and_provider_custody_complete=True,native_terminal_receipt=describe(native_path),
        native_owner_receipt=describe(end),UTC=now(),partial_predictive_metrics_read=False,
        scope='Prospectively fixed complete Penn94 procedure; native failures retained, no seed or recipe rescue')
    cal_admission=H/'CALIBRATION_ADMISSION_V1.json'
    write(cal_admission,dict(base,stage='calibrate',CONFIRM_truth_access=False))
    cal_output=H/'calibrate_v1'
    cal_owner=run_stage('calibrate',cal_admission,cal_output)
    contracts.closed_owner(describe(cal_owner))
    cal=json.loads((cal_output/'COMPLETE_CAL.json').read_text())
    assert cal['terminal'] and cal['scalar_endpoints']==90 and len(cal['records'])==90
    assert cal['native_terminal_receipt']==base['native_terminal_receipt'] and not cal['CONFIRM_truth_access']
    assert cal['custody']['config_sha256']==CONFIG_SHA
    confirm_admission=H/'CONFIRM_ADMISSION_V1.json'
    write(confirm_admission,dict(base,stage='confirm',calibration_terminal_receipt=describe(cal_output/'COMPLETE_CAL.json'),
        calibration_owner_receipt=describe(cal_owner),one_complete_CONFIRM_release=True,no_prior_CONFIRM_score_in_scope=True,
        CONFIRM_truth_access=True,all90_fixed_endpoints_terminal_before_release=True))
    confirm_output=H/'confirm_v1'
    confirm_owner=run_stage('confirm',confirm_admission,confirm_output)
    contracts.closed_owner(describe(confirm_owner))
    complete=confirm_output/'COMPLETE_CONFIRM.json'
    result=json.loads(complete.read_text())
    assert result['terminal'] and result['complete_CONFIRM_release'] and result['CONFIRM_truth_access']
    write(H/'PIPELINE_WATCH_END_V1.json',dict(UTC=now(),complete=True,native=describe(native_path),
        calibration=describe(cal_output/'COMPLETE_CAL.json'),confirmation=describe(complete),
        calibration_owner=describe(cal_owner),confirmation_owner=describe(confirm_owner),
        extra_native_fits=0,extra_native_forwards=0,manuscript_acceptance_verdict=None))

if __name__=='__main__':main()
