"""One root-released resource scope; payload is injected by local transport."""
import base64,datetime,hashlib,json,os,socket,subprocess
from pathlib import Path

assert socket.gethostname()=='anogena-2-0'
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
def gpu():
    rows=subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    assert len(rows)==1 and rows[0].split(',')[0].strip()==GPU
    return int(rows[0].split(',')[1].strip())*1024**2
gpu()
REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
NAME='internal_BE_molhiv_be_init_contrastive6101_resource_execution_20261007_v1'
ROOT=PHASE/NAME
PYTHON=PHASE/'native_ncn_runtime_20261005_v1/.venv/bin/python'
WIKI=PHASE/'learnable_internal_be_WikiCS_scientific_family_execution_root_20261007_v1'
def sha(data):return hashlib.sha256(data).hexdigest()
def stat(pid):return Path('/proc/'+str(pid)+'/stat').read_text().rsplit(') ',1)[1].split()
def json_write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');os.chmod(path,0o600)
def gate():
    free=gpu()
    owner=json.loads((WIKI/'OWNER.json').read_text())
    assert owner['PID']==510850 and owner['start_ticks']==6015502511
    assert int(stat(owner['PID'])[19])==owner['start_ticks'] and stat(owner['PID'])[0] not in ('Z','X')
    running=json.loads((WIKI/'RUNNING_CELL.json').read_text())
    assert running['cell']=='independent4_6101' and running['arm']=='independent4' and running['seed']==6101
    progress=json.loads((WIKI/'fits/outputs/independent4_6101/PROGRESS.json').read_text())
    assert progress['scores_closed'] is True and progress['complete_epochs']==1100
    assert type(progress['epoch']) is int and 0<progress['epoch']<600
    live=json.loads((WIKI/'fits/receipts/independent4_6101_LIVE.json').read_text())
    assert live['supervisor_pid']==511830 and live['supervisor_start_ticks']==6015684939
    supervisor_stat=stat(511830)
    assert int(supervisor_stat[19])==6015684939 and supervisor_stat[0] not in ('Z','X')
    worker_stat=stat(511832)
    assert int(worker_stat[19])==6015684955 and int(worker_stat[1])==511830 and worker_stat[0] not in ('Z','X')
    args=Path('/proc/511832/cmdline').read_bytes().decode().strip('\0').split('\0')
    assert args==[str(PYTHON),'-B',str(PHASE/'learnable_internal_be_contrastive_multitask_suite_20261007_v4/run.py'),'--job',str(WIKI/'fit_jobs/independent4_6101.json')]
    assert not (WIKI/'fits/receipts/independent4_6101_TERMINAL.json').exists()
    assert free>=8589934592
    return {'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'WikiCS_cell':running['cell'],'WikiCS_epoch':progress['epoch'],'WikiCS_complete_epochs':1100,'WikiCS_worker_PID':511832,'WikiCS_worker_start_ticks':6015684955,'WikiCS_supervisor_PID':511830,'WikiCS_supervisor_start_ticks':6015684939,'free_GPU_bytes':free,'minimum_free_GPU_bytes':8589934592,'hostname':socket.gethostname(),'physical_GPU_UUID':GPU,'scores_read':False,'other_jobs_modified':False}
def admission_failure(error,staged=False):
    record={'launched':False,'staged':staged,'admission_error_type':type(error).__name__,'automatic_retry':False,'scores_read':False,'other_jobs_modified':False,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if staged:json_write(ROOT/'LAUNCH.json',record)
    print(json.dumps(record));raise SystemExit(0)
try:pre=gate()
except Exception as error:admission_failure(error)
payload={name:base64.b64decode(value) for name,value in PAYLOAD.items()}
assert set(payload)=={'MANIFEST.json','ROOT_RELEASE.json','job.json','run_once.py','STAGE_AND_LAUNCH.py'}
assert sha(payload['MANIFEST.json'])==EXPECTED_MANIFEST
manifest=json.loads(payload['MANIFEST.json'])
for row in manifest['files']:
    assert row['path'] in payload and len(payload[row['path']])==row['bytes'] and sha(payload[row['path']])==row['sha256']
job=json.loads(payload['job.json']);release=json.loads(payload['ROOT_RELEASE.json'])
assert release['ROOT_NEW_TASK_authorization'] is True and release['job_sha256']==sha(payload['job.json'])
assert job['task']=='molhiv' and job['arm']=='be_init_contrastive' and job['seed']==6101
assert (job['hard_seconds'],job['active_compute_seconds'],job['cleanup_grace_seconds'])==(120,110,10)
assert job['output_directory']==str(ROOT/'outputs/be_init_contrastive_6101')
assert job['supervisor_receipt_path']==NAME+'/receipts/be_init_contrastive_6101_LIVE.json'
for directory,expected in [('learnable_internal_be_contrastive_multitask_suite_20261007_v4',job['source_manifest_sha256']),('learnable_internal_be_resource_qualifier_source_20261007_v1',job['qualifier_manifest_sha256'])]:
    source=PHASE/directory
    assert sha((source/'MANIFEST.json').read_bytes())==expected
    for row in json.loads((source/'MANIFEST.json').read_text())['files']:
        file=source/row['path'];assert file.is_relative_to(source)
        assert file.stat().st_size==row['bytes'] and sha(file.read_bytes())==row['sha256']
os.umask(0o077)
assert not ROOT.exists()
ROOT.mkdir(mode=0o700)
for name,data in payload.items():
    with (ROOT/name).open('xb') as destination:destination.write(data)
    os.chmod(ROOT/name,0o600)
    assert sha((ROOT/name).read_bytes())==sha(data)
(ROOT/'outputs').mkdir(mode=0o700);(ROOT/'receipts').mkdir(mode=0o700)
try:admission=gate()
except Exception as error:admission_failure(error,True)
json_write(ROOT/'ADMISSION.json',admission)
env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,PYTHONPATH=str(PHASE/'native_ncn_dependency_overlay_20261005_v1')+':'+str(REPO/'.venv/lib/python3.11/site-packages'));env.pop('PYTHONHOME',None)
with (ROOT/'outer.log').open('x') as log:
    child=subprocess.Popen([str(PYTHON),'-B',str(ROOT/'run_once.py')],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
launch={'launched':True,'staged':True,'outer_PID':child.pid,'outer_start_ticks':int(stat(child.pid)[19]),'activation_source_manifest_sha256':EXPECTED_MANIFEST,'job_sha256':release['job_sha256'],'task':'molhiv','arm':'be_init_contrastive','seed':6101,'hard_seconds':120,'active_compute_seconds':110,'cleanup_grace_seconds':10,'scientific_fit_authorized':False,'automatic_retry':False,'resource_weights_used_as_fit_start':False,'admission':admission}
json_write(ROOT/'LAUNCH.json',launch)
print(json.dumps(launch))
