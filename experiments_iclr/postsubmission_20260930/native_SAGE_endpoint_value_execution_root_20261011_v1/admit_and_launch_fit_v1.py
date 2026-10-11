"""Admit one measured-cap complete family; no predictive outcome is opened."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
from datetime import datetime, timezone

R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930'
H=Path(__file__).resolve().parent
OLD=P/'native_SAGE_endpoint_value_transport_source_20261011_v1'
S=P/'native_SAGE_endpoint_value_transport_source_20261011_v2'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'

def item(p):
    p=Path(p).resolve();assert p.is_relative_to(R)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,v):
    assert not p.exists();p.write_text(json.dumps(v,indent=2)+'\n')

def main():
    assert socket.gethostname()=='anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[UUID]
    owner=json.loads((H/'qualify_owner_v1/OWNER_END.json').read_text())
    q=json.loads((H/'qualify_v1/QUALIFICATION.json').read_text())
    assert owner['scientific_success'] and owner['child_pid_absent'] and owner['owned_cuda_pid_absent'] and q['status']=='PASS'
    assert q['scientific_fits']==0 and q['VALID_quality_calls']==0 and not q['TEST_access']
    assert item(H/'qualify_v1/QUALIFICATION.json')['sha256']==owner['complete_family_sha256']
    for row in json.loads((S/'SEAL.json').read_text())['files']:assert item(S/row['path'])['sha256']==row['sha256']
    for name in ('run_family.py','value_installer.py','value_transport.py','qualify_full_train.py','analysis.py','PROTOCOL.json'):
        assert (OLD/name).read_bytes()==(S/name).read_bytes(),name
    text=(OLD/'contracts.py').read_text()
    assert text.count('limits["wall_seconds"] > 7200')==1
    assert text.replace('limits["wall_seconds"] > 7200','limits["wall_seconds"] > 86400')==(S/'contracts.py').read_text()
    cfg=json.loads((H/'QUALIFICATION_CONFIG_V1.json').read_text())
    assert owner['config_sha256']==item(H/'QUALIFICATION_CONFIG_V1.json')['sha256']
    for row in cfg['source_receipts'].values():assert item(row['path'])==row
    for row in cfg['safe_roles'].values():assert item(row['path'])==row
    cfg['family_limits']['wall_seconds']=86400
    for key,name in [('training_family','run_family.py'),('value_installer','value_installer.py'),('value_helper','value_transport.py'),('contracts','contracts.py')]:cfg['source_receipts'][key]=item(S/name)
    science_hash=hashlib.sha256(json.dumps({k:v for k,v in cfg.items() if k!='root_admission'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    admission=dict(status='HOST_GPU_SOURCE_DATA_ADMITTED',action='fixed_family_fit',hostname=socket.gethostname(),gpu_uuid=UUID,
        runtime_fingerprint_sha256=item(H/'RUNTIME_FINGERPRINT_V1.json')['sha256'],
        source_sha256={str(Path(v['path']).resolve()):v['sha256'] for v in cfg['source_receipts'].values()},
        safe_role_sha256={k:v['sha256'] for k,v in cfg['safe_roles'].items()},science_config_sha256=science_hash,
        qualification=item(H/'qualify_v1/QUALIFICATION.json'),qualification_owner=item(H/'qualify_owner_v1/OWNER_END.json'),
        source_amendment='Finite wall budget allowance only; numerical modules and scientific protocol byte-identical to actual qualified V1.',
        source_seal=item(S/'SEAL.json'),UTC=datetime.now(timezone.utc).isoformat(),banks=15,optimizer_bundles=24,
        native_body_fit_records=24,partial_predictive_metrics_read=False,TEST_access=False,
        schedule='Concurrent normal allocation execution with Penn94; qualification CUDA peak16.107GB plus Penn94 observed5.349GB fits80GB.24hour hard cap is not an ETA.')
    write(H/'SCIENTIFIC_ADMISSION_V1.json',admission)
    cfg['root_admission']=item(H/'SCIENTIFIC_ADMISSION_V1.json')
    write(H/'SCIENTIFIC_CONFIG_V1.json',cfg)
    write(H/'RESOURCE_CAP_AMENDMENT_V1.json',dict(UTC=admission['UTC'],old_seconds=7200,new_seconds=86400,
        qualification_seconds=q['elapsed_seconds'],qualification_cuda_peak_bytes=q['cuda_peak_allocated_bytes'],
        qualification_process_RSS_bytes=q['process_peak_RSS_bytes'],metadata_limit_only=True,scientific_parameters_changed=False,
        no_second_disposable_qualification=True,no_scientific_result_opened=True))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',DGLBACKEND='pytorch')
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    cmd=[str(R/'.venv/bin/python'),'-B',str(H/'owner_v2.py'),'--stage','fit']
    with (H/'fit_owner_stdout.log').open('x') as out,(H/'fit_owner_stderr.log').open('x') as err:
        child=subprocess.Popen(cmd,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
    result=dict(UTC=datetime.now(timezone.utc).isoformat(),owner_PID=child.pid,command=cmd,config=item(H/'SCIENTIFIC_CONFIG_V1.json'),
        admission=item(H/'SCIENTIFIC_ADMISSION_V1.json'),source_seal=item(S/'SEAL.json'),output=str(H/'actual_family_v1'),
        banks=15,optimizer_bundles=24,all75_fixed_calibration_endpoints_before_interpretation=True,TEST_access=False,
        gpu_observation=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines())
    write(H/'SCIENTIFIC_LAUNCH_V1.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
