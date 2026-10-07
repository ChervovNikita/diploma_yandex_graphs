"""One serial, fixed 24-cell WikiCS resource/fit family; release disabled.

Only stdlib and the sealed stdlib v4 runtime are loaded by this controller.
Learning, validation and finite checks remain in the unchanged native workers.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
SUITE=ROOT.parent/'learnable_internal_be_contrastive_multitask_suite_20261007_v4'
QUALIFIER=ROOT.parent/'learnable_internal_be_resource_qualifier_source_20261007_v1'
SOURCE_SHA='76de82781e7fd496a5a3382b3a71a5781ea023dfe3777469cc005a2b19afbfce'
QUALIFIER_SHA='3668747a8e27ab7eaa3754f56744697d6a71cb1e2e3a05284175362e64833f50'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda:handle.read(1048576),b''):digest.update(part)
    return digest.hexdigest()


def write(path,value,immutable=False):
    path=Path(path)
    if immutable:
        with path.open('x') as handle:json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False);handle.write('\n')
        os.chmod(path,0o444)
    else:
        temporary=path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
        os.chmod(temporary,0o600);os.replace(temporary,path)


def seal(root,pin=None):
    if pin is not None and sha(root/'MANIFEST.json')!=pin:raise ValueError('Exact source manifest')
    for row in json.loads((root/'MANIFEST.json').read_text())['files']:
        path=(root/row['path']).resolve(strict=True)
        if not path.is_relative_to(root) or sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:
            raise ValueError('Sealed source changed: '+row['path'])
    return sha(root/'MANIFEST.json')


def runtime_module():
    spec=importlib.util.spec_from_file_location('family_v4_runtime',SUITE/'runtime.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def bound(runtime,row):return runtime.bound(runtime.PHASE,row)


def record(runtime,row):return json.loads(bound(runtime,row).read_text())


def alive(runtime,pid,ticks):
    try:return runtime.proc_start(pid)==ticks
    except FileNotFoundError:return False


def no_resource_children(runtime,origins):
    # Inspect actual recorded supervisors and actual worker/supervisor commands,
    # not a queued owner's existence or a fictitious cooperative lock.
    for origin in origins:
        folder=runtime.PHASE/origin['execution_directory']/'receipts'
        if folder.is_dir():
            for path in folder.glob('*_LIVE.json'):
                live=json.loads(path.read_text())
                pid,ticks=live.get('supervisor_pid'),live.get('supervisor_start_ticks')
                if type(pid) is int and type(ticks) is int and alive(runtime,pid,ticks):
                    raise ValueError('Actual resource supervisor remains live')
    scripts={str(QUALIFIER/'supervise.py'),str(QUALIFIER/'worker.py')}
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():continue
        try:arguments=(path/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        decoded=[x.decode(errors='replace') for x in arguments]
        active=any(x in scripts for x in decoded)
        active=active or any(x.startswith(str(runtime.PHASE)+'/') and 'learnable_internal_be_' in x and 'resource_qualifier_source_' in x and Path(x).name in ('supervise.py','worker.py') for x in decoded)
        if active:
            raise ValueError('Actual full resource supervisor/worker remains live')


def admit(release_path):
    release_path=Path(release_path).resolve();release=json.loads(release_path.read_text())
    for key in ('root_scientific_fit_authorized','root_resource_execution_authorized','fixed_family_adopted','root_driver_source_approved'):
        if release.get(key) is not True:raise ValueError('Disabled family release: '+key)
    if release.get('TEST_access') is not False or release.get('automatic_retry') is not False or release.get('predictive_opening_authorized') is not False:
        raise ValueError('Closed fixed family scope')
    driver_sha=seal(ROOT)
    if release.get('driver_manifest_sha256')!=driver_sha:raise ValueError('Exact root-inspected driver source')
    seal(SUITE,SOURCE_SHA);seal(QUALIFIER,QUALIFIER_SHA)
    runtime=runtime_module();runtime.allocation()
    if not ROOT.is_relative_to(runtime.PHASE) or not release_path.is_relative_to(runtime.PHASE):raise ValueError('Exact phase-owned source/release')
    adoption=record(runtime,release['adoption'])
    if release['adoption']['path']!=ROOT.name+'/ADOPTION_PROSPECTIVE.json':raise ValueError('Fixed adopted record only')
    inspection=record(runtime,release['root_driver_inspection'])
    if inspection.get('approved') is not True or inspection.get('driver_manifest_sha256')!=driver_sha:raise ValueError('Root exact-source inspection')
    config=record(runtime,adoption['config'])
    roster=[{'arm':arm,'seed':seed,'cell':arm+'_'+str(seed)} for seed in config['pilot_seeds'] for arm in config['arms']]
    if adoption['task']!='wikics' or adoption['cells']!=roster or len(roster)!=24 or config['training']['epochs']!=1100 or config['training']['local_epochs']!=100:
        raise ValueError('Unchanged full 8-arm by 3-seed WikiCS family')
    if adoption['source_manifest_sha256']!=SOURCE_SHA or adoption['qualifier_manifest_sha256']!=QUALIFIER_SHA:
        raise ValueError('Exact suite/qualifier identity')
    for row in adoption['source_review_evidence']:
        approval=record(runtime,row)
        if approval.get('approved') is not True or approval.get('source_manifest_sha256')!=SOURCE_SHA:raise ValueError('Exact existing suite approval')
    approval=record(runtime,adoption['qualifier_source_review'])
    if approval.get('approved') is not True or approval.get('qualifier_manifest_sha256')!=QUALIFIER_SHA:raise ValueError('Exact existing qualifier approval')
    approval=record(runtime,adoption['data_export_review'])
    if approval.get('approved') is not True or approval.get('source_manifest_sha256')!=SOURCE_SHA or approval.get('data_manifest')!=adoption['data_manifest']:
        raise ValueError('Exact existing official WikiCS data approval')
    bound(runtime,adoption['data_manifest']) # Metadata only; numeric values remain in native loaders.
    supersession=record(runtime,release['resource_queue_supersession'])
    expected=adoption['superseded_resource_owner']
    for key in ('PID','start_ticks','config_sha256','source_driver_sha256'):
        if supersession.get(key)!=expected[key]:raise ValueError('Exact old resource-owner supersession')
    if supersession.get('superseded') is not True or supersession.get('idle_boundary_verified') is not True or supersession.get('no_active_resource_child') is not True or supersession.get('scientific_jobs_signalled') is not False:
        raise ValueError('Actual root-recorded idle/no-child supersession only')
    if alive(runtime,expected['PID'],expected['start_ticks']):raise ValueError('Superseded resource owner still live')
    no_resource_children(runtime,adoption['prior_resource_origins'])
    execution=(runtime.PHASE/release['execution_directory']).resolve()
    if not execution.is_relative_to(runtime.PHASE) or execution.exists() or not execution.parent.is_dir():raise ValueError('Fresh fixed family output')
    return release,adoption,config,execution,runtime


def base_job(adoption,arm,seed):
    return {'task':'wikics','arm':arm,'seed':seed,'config':adoption['config'],
        'data_manifest':adoption['data_manifest'],'data_export_review':adoption['data_export_review'],
        'source_manifest_sha256':SOURCE_SHA,'TEST_access':False,'automatic_retry':False}


def finite_number(value):return type(value) in (int,float) and math.isfinite(value)


def verify_resource(runtime,adoption,arm,seed,origin):
    cell=arm+'_'+str(seed)
    folder=runtime.PHASE/origin['execution_directory']/'receipts'
    path=folder/(cell+'_LIVE_RESOURCE.json')
    if not path.is_file():return None
    receipt=json.loads(path.read_text())
    if receipt.get('passed') is not True:return None # Preserve failures; never promote a worker candidate.
    job_path=runtime.PHASE/origin['job_directory']/(cell+'.json')
    job=json.loads(job_path.read_text());expected=runtime.cell_identity(base_job(adoption,arm,seed))
    if receipt.get('schema')!='internal-be-resource-qualification-v2' or receipt.get('status')!='complete' or receipt.get('exit_code')!=0 or receipt.get('cell_identity')!=expected or receipt.get('seed')!=seed:
        raise ValueError('Exact supervisor-issued same-arm/seed resource pass')
    if receipt.get('qualifier_manifest_sha256')!=QUALIFIER_SHA or receipt.get('job_sha256')!=sha(job_path) or job.get('seed')!=seed or job.get('arm')!=arm or runtime.cell_identity(job)!=expected or job.get('data_export_review')!=adoption['data_export_review']:
        raise ValueError('Exact resource source/job/seed custody')
    qidentity={'cell_identity':expected,'seed':seed,'qualifier_manifest_sha256':QUALIFIER_SHA,
        'suite_manifest_sha256':SOURCE_SHA,'scope':'committed_representative_update_complete_VALID','data_export_review':adoption['data_export_review']}
    if receipt.get('qualifier_identity')!=qidentity:raise ValueError('Exact qualified task/config/data/runtime/member scope')
    terminal_path=folder/(cell+'_LIVE_TERMINAL.json');live_path=folder/(cell+'_LIVE.json')
    terminal=json.loads(terminal_path.read_text());live=json.loads(live_path.read_text())
    if sha(terminal_path)!=receipt.get('terminal_receipt_sha256') or sha(live_path)!=receipt.get('live_receipt_sha256') or terminal.get('live_receipt_sha256')!=sha(live_path):
        raise ValueError('Actual resource live/terminal hash custody')
    if terminal.get('status')!='complete' or terminal.get('exit_code')!=0 or terminal.get('owned_worker_completed') is not True or terminal.get('reap_observed') is not True or terminal.get('cap_exceeded') is not False or terminal.get('qualifier_identity')!=qidentity:
        raise ValueError('Actual completed/reaped in-cap resource terminal')
    if terminal.get('schema')!='internal-be-resource-supervisor-terminal-v1' or live.get('schema')!='internal-be-resource-live-supervisor-v1':
        raise ValueError('Actual original resource live/terminal schemas')
    if live.get('job_sha256')!=sha(job_path) or live.get('qualifier_identity')!=qidentity or live.get('supervisor_source_sha256')!=sha(QUALIFIER/'supervise.py'):
        raise ValueError('Actual reviewed resource parent')
    if not finite_number(receipt['inclusive_seconds']) or receipt['inclusive_seconds']>job['hard_seconds'] or terminal.get('absolute_hard_seconds')!=job['hard_seconds']:
        raise ValueError('Genuine original resource cap')
    runtime.resource_measurements(receipt)
    candidate_path=Path(job['output_directory'])/'WORKER_RESOURCE.json'
    candidate=json.loads(candidate_path.read_text())
    if sha(candidate_path)!=receipt.get('worker_resource_sha256') or candidate.get('worker_completed') is not True or candidate.get('passed') is not False or candidate.get('qualifier_identity')!=qidentity or candidate.get('finite_parameters_gradients_optimizer_outputs') is not True:
        raise ValueError('Actual finite resource worker custody; no checkpoint reuse')
    if candidate.get('worker_pid')!=terminal.get('child_pid') or candidate.get('worker_start_ticks')!=terminal.get('child_start_ticks'):
        raise ValueError('Actual resource child identity')
    work=receipt['work']
    if work.get('modes')!=['local','global'] or work.get('members')!=expected['members'] or work.get('own_views')!=2:
        raise ValueError('Both complete source WikiCS phases required')
    for key in ('two_view_TRAIN_backward_Adam','complete_VALID_evaluation','checkpoint_serialization','predictive_scores_closed'):
        if work.get(key) is not True:raise ValueError('Complete real resource workload required')
    if work!=candidate['work']:raise ValueError('Worker/parent work custody differs')
    return {'path':str(path.relative_to(runtime.PHASE)),'sha256':sha(path)},receipt


def memory_wait(runtime,execution,cell,kind,minimum,adoption):
    started=time.monotonic();observed=None
    while True:
        no_resource_children(runtime,adoption['prior_resource_origins'])
        raw=subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=uuid,memory.free,memory.total',
            '--format=csv,noheader,nounits'],text=True,timeout=5).strip().split(',')
        if len(raw)!=3 or raw[0].strip()!=runtime.GPU:raise ValueError('Exact singleton memory observation')
        free,total=[int(x.strip())*1024**2 for x in raw[1:]]
        observed={'free_GPU_bytes':free,'total_GPU_bytes':total,'minimum_free_GPU_bytes':minimum,
            'wait_seconds':time.monotonic()-started,'kind':kind,'cell':cell}
        write(execution/'MEMORY_WAIT.json',observed)
        if minimum>total:raise MemoryError('Measured peak plus fixed headroom exceeds physical device')
        if free>=minimum:
            write(execution/'receipts'/(cell+'_'+kind+'_MEMORY.json'),observed,immutable=True);return observed
        if time.monotonic()-started>=adoption['memory_wait_hard_seconds']:
            write(execution/'receipts'/(cell+'_'+kind+'_MEMORY_FAILURE.json'),observed,immutable=True)
            raise TimeoutError('Retained finite memory wait; no unrelated jobs stopped')
        time.sleep(20)


def run_supervisor(runtime,source,job_path,log_path):
    started=time.monotonic();env=dict(os.environ,CUDA_VISIBLE_DEVICES=runtime.GPU);env.pop('PYTHONHOME',None)
    with log_path.open('x') as log:
        child=subprocess.Popen([str(runtime.PYTHON),'-B',str(source/'supervise.py'),'--job',str(job_path)],
            cwd=runtime.REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            owner={'pid':child.pid,'start_ticks':runtime.proc_start(child.pid),'job_sha256':sha(job_path),
                'source_supervisor_sha256':sha(source/'supervise.py')}
            write(log_path.with_suffix('.OWNER.json'),owner,immutable=True)
        finally:
            # Even a metadata failure cannot silently overlap a still running
            # owned source supervisor with the next cell or family closure.
            code=child.wait() # Unchanged supervisor owns the finite child hard bound.
    return {'exit_code':code,'inclusive_driver_seconds':time.monotonic()-started,
        'supervisor_owner':owner,'closed_parent_log_sha256':sha(log_path)}


def verify_fit(runtime,job,job_path,supervised):
    live_path=runtime.PHASE/job['supervisor_receipt_path']
    terminal_path=live_path.with_name(live_path.stem+'_TERMINAL.json')
    if not terminal_path.is_file():return False,{'failure':'supervisor_terminal_missing'}
    terminal=json.loads(terminal_path.read_text())
    custody={'terminal':{'path':str(terminal_path.relative_to(runtime.PHASE)),'sha256':sha(terminal_path)}}
    if supervised['exit_code']!=0 or terminal.get('status')!='complete' or terminal.get('exit_code')!=0 or terminal.get('admission_success') is not True or terminal.get('reap_observed') is not True or terminal.get('cap_exceeded') is not False:
        return False,custody
    live=json.loads(live_path.read_text());expected=runtime.cell_identity(job)
    if terminal.get('live_receipt_sha256')!=sha(live_path) or live.get('cell_identity')!=expected or terminal.get('cell_identity')!=expected or live.get('job_sha256')!=sha(job_path):
        raise ValueError('Actual immutable scientific job/live/terminal custody')
    if live.get('supervisor_pid')!=supervised['supervisor_owner']['pid'] or live.get('supervisor_start_ticks')!=supervised['supervisor_owner']['start_ticks'] or live.get('supervisor_source_sha256')!=sha(SUITE/'supervise.py'):
        raise ValueError('Actual unchanged v4 fit parent')
    if live.get('hard_seconds')!=job['hard_seconds'] or terminal.get('absolute_admission_cap_seconds')!=job['hard_seconds']:
        raise ValueError('Unchanged full fit hard-cap custody')
    for key in ('active_compute_seconds','cleanup_grace_seconds'):
        if live.get(key)!=job[key] or terminal.get(key)!=job[key]:raise ValueError('Unchanged full fit cap custody')
    if not finite_number(terminal['inclusive_seconds']) or terminal['inclusive_seconds']>job['hard_seconds']:raise ValueError('Actual in-cap full fit')
    freeze_path=Path(job['output_directory'])/'FREEZE.json'
    if not freeze_path.is_file():return False,custody
    freeze=json.loads(freeze_path.read_text()) # No predictive values in source FREEZE.
    if freeze.get('complete') is not True or freeze.get('epochs')!=1100 or freeze.get('steps')!=1100 or freeze.get('task')!='wikics' or freeze.get('arm')!=job['arm'] or freeze.get('seed')!=job['seed'] or freeze.get('source_manifest_sha256')!=SOURCE_SHA or freeze.get('TEST_access') is not False or freeze.get('scores_closed') is not True:
        raise ValueError('Actual complete unchanged 1100-epoch scientific endpoint')
    custody['freeze']={'path':str(freeze_path.relative_to(runtime.PHASE)),'sha256':sha(freeze_path)}
    custody['selected_sha256']=freeze['selected_sha256']
    return True,custody


def main():
    started=time.monotonic();os.umask(0o077)
    parser=argparse.ArgumentParser();parser.add_argument('--release',required=True);args=parser.parse_args()
    release,adoption,config,execution,runtime=admit(args.release)
    execution.mkdir(mode=0o700)
    for directory in ('receipts','resource_jobs','fit_jobs','resources','resources/outputs','resources/receipts','fits','fits/outputs','fits/receipts'):
        (execution/directory).mkdir(mode=0o700)
    owner={'PID':os.getpid(),'start_ticks':runtime.proc_start(os.getpid()),'release_sha256':sha(args.release),
        'driver_manifest_sha256':sha(ROOT/'MANIFEST.json'),'adoption':release['adoption']}
    write(execution/'OWNER.json',owner,immutable=True)
    origins=list(adoption['prior_resource_origins'])+[{'execution_directory':str((execution/'resources').relative_to(runtime.PHASE)),
        'job_directory':str((execution/'resource_jobs').relative_to(runtime.PHASE))}]
    rows=[];fatal=None
    try:
        for cell in adoption['cells']:
            arm,seed,name=cell['arm'],cell['seed'],cell['cell'];cell_started=time.monotonic()
            row={**cell,'status':'declared','predictive_values_opened':False}
            write(execution/'RUNNING_CELL.json',row)
            try:
                no_resource_children(runtime,origins)
                found=None
                for origin in origins:
                    value=verify_resource(runtime,adoption,arm,seed,origin)
                    if value is not None:found=value;break
                if found is None:
                    memory_wait(runtime,execution,name,'RESOURCE',adoption['unmeasured_resource_minimum_free_GPU_bytes'],adoption)
                    resource_job=base_job(adoption,arm,seed)
                    resource_job.update(schema='internal-be-resource-bootstrap-job-v1',root_resource_execution_authorized=True,
                        qualifier_source_review_approved=True,resource_scope_adopted=True,fit_authorized=False,export_authorized=False,
                        qualifier_manifest_sha256=QUALIFIER_SHA,qualifier_source_review=adoption['qualifier_source_review'],
                        suite_source_reviews=adoption['source_review_evidence'],external_hard_bound_confirmed=True,
                        hard_seconds=900,active_compute_seconds=890,cleanup_grace_seconds=10,
                        output_directory=str(execution/'resources/outputs'/name),
                        supervisor_receipt_path=str((execution/'resources/receipts'/(name+'_LIVE.json')).relative_to(runtime.PHASE)))
                    path=execution/'resource_jobs'/(name+'.json');write(path,resource_job,immutable=True)
                    row['new_resource_attempt']=run_supervisor(runtime,QUALIFIER,path,execution/'receipts'/(name+'_RESOURCE_PARENT.log'))
                    found=verify_resource(runtime,adoption,arm,seed,origins[-1])
                    if row['new_resource_attempt']['exit_code']!=0 or found is None:
                        row['status']='retained_resource_failure';continue
                evidence,measurement=found;row['resource_evidence']=evidence
                row['same_arm_and_seed_resource_binding']=True
                minimum=measurement['peak_GPU_bytes']+adoption['fit_headroom_GPU_bytes']
                row['fit_memory']=memory_wait(runtime,execution,name,'FIT',minimum,adoption)
                no_resource_children(runtime,origins)
                job=base_job(adoption,arm,seed)
                job.update(schema='internal-be-predictive-cell-v1',root_execution_authorized=True,source_review_approved=True,
                    fixed_protocol_adopted=True,source_review_evidence=adoption['source_review_evidence'],
                    resource_qualification_evidence=[evidence],external_hard_bound_confirmed=True,
                    soft_seconds=config['budget']['cell_soft_seconds'],hard_seconds=config['budget']['cell_hard_seconds'],
                    active_compute_seconds=config['budget']['cell_active_compute_seconds'],cleanup_grace_seconds=10,
                    output_directory=str(execution/'fits/outputs'/name),
                    supervisor_receipt_path=str((execution/'fits/receipts'/(name+'_LIVE.json')).relative_to(runtime.PHASE)),
                    family_adoption=release['adoption'],family_release_sha256=sha(args.release),
                    same_seed_resource_binding_enforced_by_driver=True,resource_checkpoints_reused=False)
                path=execution/'fit_jobs'/(name+'.json');write(path,job,immutable=True)
                row['immutable_fit_job']={'path':str(path.relative_to(runtime.PHASE)),'sha256':sha(path)}
                row['scientific_supervisor']=run_supervisor(runtime,SUITE,path,execution/'receipts'/(name+'_FIT_PARENT.log'))
                complete,custody=verify_fit(runtime,job,path,row['scientific_supervisor'])
                row['status']='complete' if complete else 'retained_fit_failure';row['fit_custody']=custody
            except (TimeoutError,MemoryError) as error:
                row['status']='retained_memory_admission_failure';row['error_type']=type(error).__name__
            except Exception as error:
                row['status']='retained_source_or_custody_failure';row['error_type']=type(error).__name__
                raise
            finally:
                row['inclusive_cell_driver_seconds']=time.monotonic()-cell_started
                rows.append(row);write(execution/'receipts'/(name+'_CELL.json'),row,immutable=True)
                write(execution/'LEDGER.json',rows)
    except Exception as error:
        fatal=type(error).__name__
        # Custody/source failure stops admission. Retain every unlaunched cell;
        # never reinterpret it as an endpoint, retry or shorten a fit.
        for cell in adoption['cells'][len(rows):]:rows.append({**cell,'status':'not_launched_driver_failure','error_type':fatal,'predictive_values_opened':False})
        write(execution/'DRIVER_FAILURE.json',{'error_type':fatal,'rows':rows,'automatic_retry':False})
    if len(rows)!=24 or [r['cell'] for r in rows]!=[r['cell'] for r in adoption['cells']]:raise ValueError('Exact complete family ledger')
    write(execution/'FAMILY_CLOSURE.json',{'schema':'internal-be-WikiCS-family-closure-v1','closed':True,'owner':owner,
        'adoption':release['adoption'],'cells':rows,'complete_cells':sum(r['status']=='complete' for r in rows),
        'failed_or_not_launched_cells':sum(r['status']!='complete' for r in rows),
        'predictive_values_opened':False,'TEST_access':False,'automatic_retry':False,
        'all_full_endpoints_or_retained_failure':True,'inclusive_family_driver_seconds':time.monotonic()-started},immutable=True)
    if fatal is not None:raise SystemExit(1)


if __name__=='__main__':main()
