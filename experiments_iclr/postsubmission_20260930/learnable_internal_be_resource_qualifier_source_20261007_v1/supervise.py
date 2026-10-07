"""Resource-only owned-child parent. Only this parent can issue passed evidence.

Reuses the reviewed v4 terminal classifier and /proc identity helper. Parent
starts the clock before admission; TERM/KILL/reap fit inside the same cap.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from guard import admit,identity,json_write,sha,ROOT


def rss(pid):
    values={}
    for line in Path('/proc/'+str(pid)+'/status').read_text().splitlines():
        if line.startswith(('VmRSS:','VmHWM:')):
            key,value,*_=line.split();values[key.rstrip(':')]=int(value)*1024
    return max(values.values(),default=0)


def driver_sample(runtime,pid,timeout):
    result=subprocess.run(['/usr/bin/nvidia-smi','-i',runtime.GPU,
        '--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],
        capture_output=True,text=True,check=True,timeout=timeout)
    found=[]
    for line in result.stdout.splitlines():
        fields=[x.strip() for x in line.split(',')]
        if len(fields)==2 and fields[0]==str(pid):
            found.append(int(fields[1])*1024**2)
    return max(found,default=0)


def owned_stop(child,child_start,runtime,hard_deadline):
    if child.poll() is not None:return
    if runtime.proc_start(child.pid)!=child_start or os.getpgid(child.pid)!=child.pid:
        raise ValueError('Only the actual owned child group may be stopped')
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=max(0.,min(5.,hard_deadline-time.monotonic())))
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL)
        try:child.wait(timeout=max(0.,hard_deadline-time.monotonic()))
        except subprocess.TimeoutExpired:pass


def validate_candidate(output,candidate,job,runtime,child,child_start,live):
    if candidate.get('schema')!='internal-be-resource-worker-candidate-v1' or candidate.get('worker_completed') is not True or candidate.get('passed') is not False:
        raise ValueError('Actual worker candidate; never a self-issued pass')
    if candidate.get('qualifier_identity')!=identity(job,runtime) or candidate.get('cell_identity')!=runtime.cell_identity(job) or candidate.get('seed')!=job['seed']:
        raise ValueError('Exact measured source/config/data/arm/seed identity')
    if candidate.get('worker_pid')!=child.pid or candidate.get('worker_start_ticks')!=child_start or candidate.get('supervisor_receipt_sha256')!=sha(live):
        raise ValueError('Actual owned worker custody')
    if candidate.get('finite_parameters_gradients_optimizer_outputs') is not True or candidate.get('TEST_access') is not False or candidate.get('automatic_retry') is not False:
        raise ValueError('Finite work and closed roles')
    work=candidate['work'];modes=['local','global'] if job['task']=='wikics' else ['native']
    for key in ('two_view_TRAIN_backward_Adam','complete_VALID_evaluation','checkpoint_serialization','predictive_scores_closed','resource_only_not_fit'):
        if work.get(key) is not True:raise ValueError('Missing real complete workload: '+key)
    if work.get('modes')!=modes or work.get('members')!=runtime.cell_identity(job)['members'] or work.get('own_views')!=2 or work.get('committed_TRAIN_updates')!=len(modes):
        raise ValueError('Exact complete work counts')
    train_count=580 if job['task']=='wikics' else (131072 if job['task']=='collab' else 128)
    valid_count=5274 if job['task']=='wikics' else (160084 if job['task']=='collab' else 4113)
    optimizers=work['members'] if job['arm'].removesuffix('_contrastive') in ('single','independent4') else 1
    for mode in modes:
        counts=work['counters'][mode]
        if counts['TRAIN']['calls']!=2 or counts['TRAIN']['member_forwards']!=2*work['members'] or counts['TRAIN']['objects_per_member']!=2*train_count:
            raise ValueError('Both full TRAIN views and every member required')
        if counts['VALID']['objects_per_member']!=valid_count or counts['VALID']['member_forwards']!=counts['VALID']['calls']*work['members'] or counts['backward_calls']!=1 or counts['optimizer_steps']!=optimizers:
            raise ValueError('Complete VALID and exact Adam grouping required')
    for key in ('peak_CUDA_allocated_bytes','peak_CUDA_reserved_bytes','peak_RSS_bytes','artifact_storage_bytes'):
        if type(candidate.get(key)) is not int or candidate[key]<=0:raise ValueError('Genuine measured byte counters')
    for row in candidate['closed_artifacts']:
        path=(output/row['path']).resolve(strict=True)
        if not path.is_relative_to(output) or not path.is_file() or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError('Complete private output custody')
    custody=json.loads((output/'WORKER_OUTPUT_CUSTODY.json').read_text())
    if custody.get('WORKER_RESOURCE_sha256')!=sha(output/'WORKER_RESOURCE.json'):
        raise ValueError('Worker candidate serialization custody')
    return work


def main():
    started=time.monotonic();os.umask(0o077)
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True)
    args=parser.parse_args();job_path=Path(args.job).resolve()
    job,config,output,live,runtime,original_supervisor=admit(job_path)
    hard_deadline=started+job['hard_seconds'];active_deadline=started+job['active_compute_seconds']
    record={'schema':'internal-be-resource-live-supervisor-v1','qualifier_identity':identity(job,runtime),
        'job_sha256':sha(job_path),'supervisor_pid':os.getpid(),'supervisor_start_ticks':runtime.proc_start(os.getpid()),
        'supervisor_source_sha256':sha(ROOT/'supervise.py'),
        **{k:job[k] for k in ('hard_seconds','active_compute_seconds','cleanup_grace_seconds')}}
    json_write(live,record)
    env=dict(os.environ,INTERNAL_BE_RESOURCE_SUPERVISOR_RECEIPT=str(live),
        INTERNAL_BE_RESOURCE_SUPERVISOR_SHA256=sha(live))
    child=None;child_start=None;exit_code=None;status='startup_failed';candidate=None
    peak_rss=0;peak_driver=0;monitor_errors=[];driver_samples=0;last_driver=0.;failure_type=None
    log_path=live.with_name(live.stem+'_CHILD.log')
    if log_path.exists():raise ValueError('No child log overwrite')
    try:
        with log_path.open('x') as log:
            child=subprocess.Popen([str(runtime.PYTHON),'-B',str(ROOT/'worker.py'),'--job',str(job_path)],
                cwd=str(runtime.REPO),env=env,start_new_session=True,stdout=log,stderr=log)
            child_start=runtime.proc_start(child.pid)
            while child.poll() is None:
                now=time.monotonic()
                if now>=active_deadline:
                    owned_stop(child,child_start,runtime,hard_deadline);status='hard_timeout';break
                try:peak_rss=max(peak_rss,rss(child.pid))
                except FileNotFoundError:pass
                if now-last_driver>=1. and active_deadline-now>2.:
                    last_driver=now
                    try:
                        peak_driver=max(peak_driver,driver_sample(runtime,child.pid,min(2.,active_deadline-now)))
                        driver_samples+=1
                    except Exception as error:monitor_errors.append(type(error).__name__)
                time.sleep(min(.25,max(0.,active_deadline-time.monotonic())))
            exit_code=child.poll()
            if status!='hard_timeout':status='complete' if exit_code==0 else 'worker_failed'
        if status=='complete':
            candidate=json.loads((output/'WORKER_RESOURCE.json').read_text())
            validate_candidate(output,candidate,job,runtime,child,child_start,live)
    except Exception as error:
        failure_type=type(error).__name__;status='supervisor_or_custody_failure'
        candidate=None  # Unvalidated candidate fields cannot bypass failure custody.
        if child is not None and child.poll() is None:
            try:owned_stop(child,child_start,runtime,hard_deadline)
            except Exception as cleanup_error:monitor_errors.append('cleanup_'+type(cleanup_error).__name__)
        exit_code=child.poll() if child else None
    failure_record=None
    if candidate is None and output.is_dir() and (output/'FAILURE.json').is_file():
        try:
            failure_record=json.loads((output/'FAILURE.json').read_text())
            if not isinstance(failure_record,dict) or failure_record.get('schema')!='internal-be-resource-worker-failure-v1' or failure_record.get('qualifier_identity')!=identity(job,runtime) or failure_record.get('worker_pid')!=(child.pid if child else None) or failure_record.get('worker_start_ticks')!=child_start:
                failure_record=None;status='failure_custody_invalid'
        except Exception as error:
            failure_record=None;status='failure_custody_invalid';failure_type=type(error).__name__
    # Candidate/failure/log serialization and hashing are part of the parent cap.
    elapsed=time.monotonic()-started
    status=original_supervisor.normalize_terminal(status,exit_code,elapsed,job['hard_seconds'],child is not None)
    success=status=='complete' and exit_code==0 and candidate is not None
    peak=max(peak_driver,candidate['peak_CUDA_allocated_bytes'] if candidate else 0,
        candidate['peak_CUDA_reserved_bytes'] if candidate else 0,
        failure_record.get('peak_CUDA_allocated_bytes',0) if failure_record else 0,
        failure_record.get('peak_CUDA_reserved_bytes',0) if failure_record else 0)
    receipt={'schema':'internal-be-resource-qualification-v2','passed':False,
        'cell_identity':runtime.cell_identity(job),'seed':job['seed'],'qualifier_identity':identity(job,runtime),
        'live_receipt_sha256':sha(live),'job_sha256':sha(job_path),
        'qualifier_manifest_sha256':job['qualifier_manifest_sha256'],
        'status':status,'exit_code':exit_code,'inclusive_seconds':elapsed,'peak_GPU_bytes':peak,
        'sampled_peak_driver_process_GPU_bytes':peak_driver,'driver_samples':driver_samples,'monitor_errors':monitor_errors,
        'parent_sampled_peak_child_RSS_bytes':peak_rss,'work':candidate['work'] if candidate else {},
        'worker_measurements':{k:candidate[k] for k in ('peak_CUDA_allocated_bytes','peak_CUDA_reserved_bytes','peak_RSS_bytes','worker_inclusive_seconds','artifact_storage_bytes')} if candidate else {},
        'source_scope_contract':candidate['source_scope_contract'] if candidate else {},
        'worker_resource_sha256':sha(output/'WORKER_RESOURCE.json') if candidate else None,
        'unaccepted_worker_candidate_sha256':sha(output/'WORKER_RESOURCE.json') if candidate is None and (output/'WORKER_RESOURCE.json').is_file() else None,
        'worker_failure_sha256':sha(output/'FAILURE.json') if failure_record else None,
        'worker_failure_measurements':failure_record,
        'cuda_device':candidate['cuda_device'] if candidate else None,
        'child_log_sha256':sha(log_path),'failure_type':failure_type,
        'predictive_scores_closed':True,'automatic_retry':False,'TEST_access':False,
        'scientific_fit_or_quality_admission':False,'full_fit_time_forecast_claimed':False,
        'seed_compatibility_enforced_by_v4_cell_identity':False}
    if success:
        try:runtime.resource_measurements(receipt)
        except Exception as error:
            success=False;receipt.update(status='resource_measurement_failure',failure_type=type(error).__name__)
    # Write nonpassing evidence first. A real terminal is closed before pass.
    result_path=live.with_name(live.stem+'_RESOURCE.json')
    terminal_path=live.with_name(live.stem+'_TERMINAL.json')
    if result_path.exists() or terminal_path.exists():raise ValueError('No resource/terminal overwrite')
    json_write(result_path,receipt)
    elapsed=time.monotonic()-started
    status=original_supervisor.normalize_terminal(receipt['status'],exit_code,elapsed,job['hard_seconds'],child is not None)
    success=success and status=='complete' and elapsed<=job['hard_seconds']
    terminal={'schema':'internal-be-resource-supervisor-terminal-v1','status':status,'exit_code':exit_code,
        'child_pid':child.pid if child else None,'child_start_ticks':child_start,'reap_observed':exit_code is not None,
        'inclusive_seconds':elapsed,'absolute_hard_seconds':job['hard_seconds'],'active_compute_seconds':job['active_compute_seconds'],
        'cleanup_grace_seconds':10,'cap_exceeded':elapsed>job['hard_seconds'],'owned_worker_completed':success,
        'live_receipt_sha256':sha(live),'qualifier_identity':identity(job,runtime),'predictive_scores_closed':True,
        'automatic_retry':False,'failure_type':receipt['failure_type']}
    json_write(terminal_path,terminal)
    # The final resource receipt is issued only after observing actual worker
    # completion and after terminal closure. Any observed output-write overrun
    # is demoted again, preserving all prior candidate/failure artifacts.
    receipt.update(passed=success,status=status,inclusive_seconds=time.monotonic()-started,
        terminal_receipt_sha256=sha(terminal_path),total_output_storage_bytes=sum(p.stat().st_size for p in output.iterdir() if p.is_file()) if output.exists() else 0)
    receipt['observed_log_live_terminal_storage_bytes']=sum(p.stat().st_size for p in (log_path,live,terminal_path))
    if receipt['inclusive_seconds']>job['hard_seconds']:
        receipt.update(passed=False,status='admission_cap_exceeded');success=False
    json_write(result_path,receipt)
    end_elapsed=time.monotonic()-started
    if end_elapsed>job['hard_seconds']:
        receipt.update(passed=False,status='admission_cap_exceeded',inclusive_seconds=end_elapsed)
        json_write(result_path,receipt);success=False
    if not success:raise SystemExit(1)
    print(json.dumps({'resource_worker_complete':True,'predictive_scores_closed':True,'receipt':str(result_path)}))


if __name__=='__main__':main()
