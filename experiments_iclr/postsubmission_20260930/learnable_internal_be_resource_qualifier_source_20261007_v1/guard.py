"""Resource bootstrap authority, distinct from sealed v4 fit admission."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SUITE = ROOT.parent/'learnable_internal_be_contrastive_multitask_suite_20261007_v4'
SUITE_MANIFEST = '76de82781e7fd496a5a3382b3a71a5781ea023dfe3777469cc005a2b19afbfce'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for part in iter(lambda:source.read(1048576),b''):h.update(part)
    return h.hexdigest()


def json_write(path, value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    os.chmod(temp,0o600);os.replace(temp,path)


def verify_sources():
    if sha(SUITE/'MANIFEST.json') != SUITE_MANIFEST:
        raise ValueError('Exact reviewed v4 manifest required')
    for root in (ROOT,SUITE):
        manifest=json.loads((root/'MANIFEST.json').read_text())
        for row in manifest['files']:
            path=root/row['path']
            if not path.is_relative_to(root) or sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:
                raise ValueError('Source seal differs: '+row['path'])
    return sha(ROOT/'MANIFEST.json')


def suite_modules():
    # These two stdlib-only modules are loaded only after their complete seal.
    verify_sources()
    old=sys.modules.get('runtime')
    if old is not None and Path(old.__file__).resolve()!=SUITE/'runtime.py':
        raise ValueError('Conflicting runtime module')
    if old is None:
        spec=importlib.util.spec_from_file_location('runtime',SUITE/'runtime.py')
        old=importlib.util.module_from_spec(spec);sys.modules['runtime']=old
        spec.loader.exec_module(old)
    spec=importlib.util.spec_from_file_location('qualifier_v4_supervisor',SUITE/'supervise.py')
    supervisor=importlib.util.module_from_spec(spec);spec.loader.exec_module(supervisor)
    return old,supervisor


def identity(job, runtime):
    return {'cell_identity':runtime.cell_identity(job),'seed':job['seed'],
        'qualifier_manifest_sha256':job['qualifier_manifest_sha256'],
        'suite_manifest_sha256':SUITE_MANIFEST,'scope':'committed_representative_update_complete_VALID',
        'data_export_review':job['data_export_review']}


def admit(job_path, worker=False):
    job_path=Path(job_path).resolve();job=json.loads(job_path.read_text())
    # No fit_admit(), circular qualification receipt, framework or data loading.
    for key in ('root_resource_execution_authorized','qualifier_source_review_approved','resource_scope_adopted'):
        if job.get(key) is not True:raise ValueError('Disabled resource authority: '+key)
    for key in ('TEST_access','automatic_retry','fit_authorized','export_authorized'):
        if job.get(key) is not False:raise ValueError('Resource-only boundary: '+key)
    if job.get('schema')!='internal-be-resource-bootstrap-job-v1':raise ValueError('Exact bootstrap schema')
    source_hash=verify_sources()
    if job.get('qualifier_manifest_sha256')!=source_hash or job.get('source_manifest_sha256')!=SUITE_MANIFEST:
        raise ValueError('Exact qualifier and suite source custody')
    runtime,supervisor=suite_modules();runtime.allocation()
    if not ROOT.is_relative_to(runtime.PHASE):raise ValueError('Phase-owned qualifier source')
    if not job_path.is_relative_to(runtime.PHASE):raise ValueError('Phase-owned exact resource job')
    config=json.loads(runtime.bound(runtime.PHASE,job['config']).read_text())
    if job['config']['path']!=SUITE.name+'/configs/'+job['task']+'.json' or config['task']!=job['task']:
        raise ValueError('Exact unchanged v4 task config')
    if job['arm'] not in config['arms'] or type(job['seed']) is not int or job['seed'] not in config['pilot_seeds']:
        raise ValueError('Exact registered arm and measured pilot seed')
    for row in job['suite_source_reviews']:
        review=json.loads(runtime.bound(runtime.PHASE,row).read_text())
        if review.get('approved') is not True or review.get('source_manifest_sha256')!=SUITE_MANIFEST:
            raise ValueError('Exact independent v4 source approval')
    if not job['suite_source_reviews']:raise ValueError('Suite review missing')
    review=json.loads(runtime.bound(runtime.PHASE,job['qualifier_source_review']).read_text())
    if review.get('approved') is not True or review.get('qualifier_manifest_sha256')!=source_hash:
        raise ValueError('Independent qualifier source approval')
    review=json.loads(runtime.bound(runtime.PHASE,job['data_export_review']).read_text())
    if review.get('approved') is not True or review.get('source_manifest_sha256')!=SUITE_MANIFEST or review.get('data_manifest')!=job['data_manifest']:
        raise ValueError('Independent exact official TRAIN/VALID export approval')
    # Actual role manifest hash, only metadata here; loader checks values later.
    runtime.bound(runtime.PHASE,job['data_manifest'])
    if any(type(job.get(k)) is not int for k in ('hard_seconds','active_compute_seconds','cleanup_grace_seconds')):
        raise ValueError('Genuine integer resource caps')
    if not 30<=job['hard_seconds']<=3600 or job['cleanup_grace_seconds']!=10 or job['active_compute_seconds']+10!=job['hard_seconds']:
        raise ValueError('Finite resource-only bound including ten-second cleanup')
    if job.get('external_hard_bound_confirmed') is not True:raise ValueError('Actual parent hard bound required')
    output=Path(job['output_directory']).resolve()
    live=(runtime.PHASE/job['supervisor_receipt_path']).resolve()
    if not output.is_relative_to(runtime.PHASE) or not live.is_relative_to(runtime.PHASE) or not output.parent.is_dir() or not live.parent.is_dir():
        raise ValueError('Phase-owned existing output/receipt parents')
    if output.exists():raise ValueError('Fresh qualifier output; no overwrite/resume')
    if worker:
        if not live.is_file() or os.environ.get('INTERNAL_BE_RESOURCE_SUPERVISOR_RECEIPT')!=str(live) or os.environ.get('INTERNAL_BE_RESOURCE_SUPERVISOR_SHA256')!=sha(live):
            raise ValueError('Exact live parent-created receipt')
        record=json.loads(live.read_text())
        expected=identity(job,runtime)
        if record.get('qualifier_identity')!=expected or record.get('job_sha256')!=sha(job_path) or record.get('supervisor_source_sha256')!=sha(ROOT/'supervise.py'):
            raise ValueError('Exact parent identity/source/job')
        if record.get('supervisor_pid')!=os.getppid() or record.get('supervisor_start_ticks')!=runtime.proc_start(os.getppid()):
            raise ValueError('Actual live parent, not a detached/fabricated receipt')
        for key in ('hard_seconds','active_compute_seconds','cleanup_grace_seconds'):
            if record.get(key)!=job[key]:raise ValueError('Parent/worker caps differ')
    elif live.exists():raise ValueError('Fresh live receipt')
    return job,config,output,live,runtime,supervisor
