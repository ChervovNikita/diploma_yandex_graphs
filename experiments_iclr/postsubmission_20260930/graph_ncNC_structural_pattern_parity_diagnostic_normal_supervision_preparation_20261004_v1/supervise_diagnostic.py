"""Ordinary detached parent for one separately admitted diagnostic on GPU1."""
import argparse
from datetime import datetime,timezone
import hashlib,json,os,signal,socket,subprocess,sys,time
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
EXECUTION=PHASE/'graph_ncNC_structural_pattern_parity_diagnostic_execution_root_20261004_v1'
PYTHON=Path('/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12')
PYTHON_SHA='14776d98474f987919376922a9995a20733e13b51d7d122873b068bf2e47d1b2'
GPU_UUIDS={'GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998','GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'}
GPU_UUID='GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'
CAPS={'wall_seconds':1800,'host_RSS_bytes':32*1024**3,'cuda_peak_allocated_bytes':70*1024**3,'cuda_peak_reserved_bytes':75*1024**3}
DIAGNOSTIC=PHASE/'graph_ncNC_structural_pattern_parity_diagnostic_source_preparation_20261004_v1'
DIAGNOSTIC_SHA='b665affd187593875a2be81e30dbe7a3c8e3ce231f7e344d791d67a800331ebe'
DIAGNOSTIC_REVIEW=PHASE/'graph_ncNC_structural_pattern_parity_diagnostic_independent_source_review_20261004_v1'
DIAGNOSTIC_REVIEW_SHA='c09384131c4f378337591ba3f29655911598ce32a887160a58e180b818402bd2'
INVOCATION={'stage':'parity_diagnostic','unit':'pair','engineering_seed':20261003,'engineering_sign_seed':2026100301,'fresh_run_count':1,'automatic_retry':False}
MIN_GPU_FREE_MIB=79872
MIN_HOST_AVAILABLE_BYTES=40*1024**3


def require(value,message):
    if not value:raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def utc():return datetime.now(timezone.utc).isoformat()


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def verify_packet(root,pin):
    root=Path(root).resolve();require(root.is_relative_to(PHASE),'Source escaped authorized phase')
    manifest=root/'MANIFEST.json';require(sha(manifest)==pin,'Source manifest custody differs')
    for row in json.loads(manifest.read_text())['files']:
        path=(root/row['path']).resolve();require(path.is_relative_to(root),'Source path escaped packet')
        require(path.stat().st_size==row.get('bytes',row.get('size')) and sha(path)==row['sha256'],'Source payload custody differs: '+row['path'])


def physical(pid):
    base=Path('/proc')/str(pid)
    stat=(base/'stat').read_text();fields=stat[stat.rfind(')')+2:].split()
    namespaces={name:os.readlink(base/'ns'/name) for name in ('pid','mnt','user','net','uts','cgroup')}
    return {'PID':pid,'parent_PID':int(fields[1]),'process_group':int(fields[2]),'session':int(fields[3]),
            'start_time_ticks':int(fields[19]),'argv':(base/'cmdline').read_bytes().decode(errors='replace').split('\0')[:-1],
            'cwd':str((base/'cwd').resolve()),'exe':str((base/'exe').resolve()),'namespaces':namespaces}


def session_rss(session):
    total=0;identities=[]
    for base in Path('/proc').iterdir():
        if not base.name.isdecimal():continue
        try:
            stat=(base/'stat').read_text();fields=stat[stat.rfind(')')+2:].split()
            if int(fields[3])!=session:continue
            rss=0;hwm=0
            for line in (base/'status').read_text().splitlines():
                if line.startswith('VmRSS:'):rss=int(line.split()[1])*1024
                elif line.startswith('VmHWM:'):hwm=int(line.split()[1])*1024
            total+=rss
            identities.append({'PID':int(base.name),'start_time_ticks':int(fields[19]),'RSS_bytes':rss,'VmHWM_bytes':hwm})
        except (OSError,ValueError):continue
    return total,identities


def main():
    started=time.monotonic()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-admission',required=True,type=Path)
    parser.add_argument('--admission-sha256',required=True)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();admission_path=args.root_admission.resolve();output=args.output.resolve()
    require(Path.cwd().resolve()==REPO and os.environ.get('GNNM_SSH_DESTINATION')=='shmelev@192.168.18.77','Exact authorized route/repository required')
    require(admission_path==EXECUTION/'ROOT_DIAGNOSTIC_ADMISSION.json' and sha(admission_path)==args.admission_sha256 and output==EXECUTION/'supervision/run01','Exact fresh diagnostic admission/output required')
    require(not output.exists(),'Supervisor output already exists; no retry')
    admission=json.loads(admission_path.read_text())
    require(admission.get('schema')=='ncnc-pattern-parity-diagnostic-root-admission-v1' and admission.get('status')=='APPROVED' and admission.get('root_authorization_reference'),'Diagnostic not separately root admitted')
    require(admission['authorized_invocation']==INVOCATION and admission['authorized_optimizer_updates']==12 and admission['authorized_member_trajectory_updates']==48,'Only the fixed diagnostic work is admitted')
    require(admission['caps']==CAPS and admission['CUDA_VISIBLE_DEVICES']==GPU_UUID,'Unchanged caps and GPU1 required')
    require(admission['state_donor'] is False and admission['scientific_fit_admitted'] is False and admission['TEST_supported'] is False,'Diagnostic-only scope required')
    require(admission['diagnostic_output_directory']==str(EXECUTION/'diagnostic/run01') and admission['supervision_output_directory']==str(output) and admission['runner_path']==str(HERE/'diagnostic_run.py'),'Wrong exact diagnostic paths')
    require(not Path(admission['diagnostic_output_directory']).exists(),'Diagnostic output already exists; no resume')
    require(Path(sys.executable).resolve()==PYTHON.resolve() and sha(PYTHON)==PYTHON_SHA,'Qualified interpreter differs')
    verify_packet(HERE,admission['normal_supervision_manifest_sha256'])
    require(admission['diagnostic_manifest_sha256']==DIAGNOSTIC_SHA and admission['diagnostic_review_manifest_sha256']==DIAGNOSTIC_REVIEW_SHA,'Exact reviewed diagnostic source required')
    verify_packet(DIAGNOSTIC,DIAGNOSTIC_SHA);verify_packet(DIAGNOSTIC_REVIEW,DIAGNOSTIC_REVIEW_SHA)
    reviewed=json.loads((DIAGNOSTIC_REVIEW/'REVIEW.json').read_text())
    require(reviewed['verdict']=='PASS' and reviewed['candidate_manifest_sha256']==DIAGNOSTIC_SHA,'Independent exact diagnostic review differs')
    require(time.monotonic()-started<CAPS['wall_seconds'],'Setup exhausted inclusive wall cap')
    # Availability is checked fresh immediately before this child. GPU0 may be
    # running the independently admitted BUDDY evaluation; it is not signaled.
    query=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    gpu_rows=[[part.strip() for part in line.split(',')] for line in query.stdout.splitlines() if line.strip()]
    require(len(gpu_rows)==2 and {r[1] for r in gpu_rows}==GPU_UUIDS,'Wrong two-GPU physical route')
    require(all(r[2]=='NVIDIA A100 80GB PCIe' and int(r[3])==81920 for r in gpu_rows),'Physical GPU profile differs')
    selected=next(row for row in gpu_rows if row[1]==GPU_UUID)
    free_mib=int(selected[3])-int(selected[4])
    require(selected[0]=='1' and free_mib>=MIN_GPU_FREE_MIB,'GPU1 fresh availability below admitted headroom')
    mem_available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    require(mem_available>=MIN_HOST_AVAILABLE_BYTES,'Fresh host memory availability below admitted headroom')
    output.mkdir(parents=True,mode=0o700)
    supervisor=physical(os.getpid())
    write(output/'SUPERVISOR_STARTED.json',{'UTC':utc(),'physical_identity':supervisor,'hostname':socket.gethostname(),
        'admission_sha256':args.admission_sha256,'normal_supervision_manifest_sha256':admission['normal_supervision_manifest_sha256'],
        'diagnostic_manifest_sha256':DIAGNOSTIC_SHA,'GPU_rows_before_child':gpu_rows,'selected_GPU_free_MiB':free_mib,
        'host_MemAvailable_bytes':mem_available,'required_GPU_free_MiB':MIN_GPU_FREE_MIB,'required_host_MemAvailable_bytes':MIN_HOST_AVAILABLE_BYTES,'caps':CAPS,
        'ordinary_host_execution':True,'namespaces_created':False,'unrelated_jobs_mutated':False,'scientific_fits_authorized':False})
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPU_UUID,PYTHONPATH=str(REPO/'.gnnm_runtime/buddy_extra_v1/site'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    argv=[str(PYTHON),'-B',str(HERE/'diagnostic_child.py'),'--root-admission',str(admission_path),'--admission-sha256',args.admission_sha256,'--supervision-output',str(output)]
    child=None;identity=None;cap_violation=None;peak=0;elapsed_start=time.monotonic();usage=None;exit_code=None;signal_number=None
    with (output/'CHILD.stdout.log').open('x') as stdout,(output/'CHILD.stderr.log').open('x') as stderr:
        try:
            child=subprocess.Popen(argv,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
            # Popen has completed exec; /proc cmdline/exe are the actual child.
            identity=physical(child.pid)
            require(identity['parent_PID']==os.getpid() and identity['session']==child.pid and identity['process_group']==child.pid,'Unexpected child physical parent/session identity')
            require(identity['namespaces']==supervisor['namespaces'] and identity['exe']==str(PYTHON.resolve()) and identity['cwd']==str(REPO),'Child is not normal host process in exact repository')
            write(output/'CHILD_STARTED.json',{'UTC':utc(),'physical_identity':identity,'requested_argv':argv,'diagnostic_invocation':INVOCATION,'selected_physical_GPU_UUID':GPU_UUID})
            last_write=0
            while True:
                rss,members=session_rss(child.pid);peak=max(peak,rss)
                elapsed=time.monotonic()-started
                if cap_violation is None and (elapsed>CAPS['wall_seconds'] or rss>CAPS['host_RSS_bytes']):
                    cap_violation={'kind':'wall' if elapsed>CAPS['wall_seconds'] else 'host_RSS','elapsed_seconds':elapsed,'RSS_bytes':rss}
                    require(physical(child.pid)['start_time_ticks']==identity['start_time_ticks'],'Child PID was reused; refusing signal')
                    os.killpg(child.pid,signal.SIGTERM)
                    termination_started=time.monotonic()
                if cap_violation is not None and time.monotonic()-termination_started>5:
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                pid,status,ru=os.wait4(child.pid,os.WNOHANG)
                if pid:
                    usage=ru;exit_code=os.waitstatus_to_exitcode(status);child.returncode=exit_code
                    signal_number=-exit_code if exit_code<0 else None
                    break
                if elapsed-last_write>=1:
                    write(output/'SUPERVISOR_STATUS.json',{'UTC':utc(),'child_PID':child.pid,'child_start_time_ticks':identity['start_time_ticks'],'elapsed_seconds':elapsed,'session_RSS_bytes':rss,'peak_observed_session_RSS_bytes':peak,'session_members':members,'cap_violation':cap_violation})
                    last_write=elapsed
                time.sleep(.25)
        except BaseException as error:
            if child is not None and child.returncode is None:
                try:
                    if identity is None or physical(child.pid)['start_time_ticks']==identity['start_time_ticks']:os.killpg(child.pid,signal.SIGTERM)
                    child.wait(timeout=5)
                except (OSError,subprocess.TimeoutExpired):
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    child.wait()
            write(output/'SUPERVISOR_FAILURE.json',{'UTC':utc(),'exception_type':type(error).__name__,'condition':str(error),'child_PID':child.pid if child else None,'child_exit_code':child.returncode if child else None,'elapsed_seconds':time.monotonic()-started,'peak_observed_session_RSS_bytes':peak,'unrelated_jobs_mutated':False})
            raise
    cuda=json.loads((output/'CHILD_CUDA_PEAKS.json').read_text()) if (output/'CHILD_CUDA_PEAKS.json').exists() else None
    if cuda and cuda.get('cap_violation'):cap_violation=cuda['cap_violation']
    final_path=Path(admission['diagnostic_output_directory'])/'FINAL.json'
    peak_kernel=int(usage.ru_maxrss*1024) if usage else None
    if cap_violation is None and peak_kernel is not None and peak_kernel>CAPS['host_RSS_bytes']:
        cap_violation={'kind':'kernel_recorded_host_RSS_peak','kernel_child_peak_RSS_bytes':peak_kernel}
    result=json.loads(final_path.read_text()) if final_path.exists() else None
    result_ok=bool(result and result.get('schema')=='ncnc-pattern-normal-parity-diagnostic-run-terminal-v1' and result.get('status')=='DIAGNOSTIC_COMPLETE' and result.get('identity',{}).get('admission_sha256')==args.admission_sha256 and result.get('identity',{}).get('diagnostic_manifest_sha256')==DIAGNOSTIC_SHA and result.get('executed_optimizer_updates')==12 and result.get('executed_member_trajectory_updates')==48 and result.get('publisher_callbacks_returned_neutral')==29 and result.get('final_publication_neutrality_verified') is True and result.get('final_caller_RNG_runtime_profile_exact') is True and result.get('state_donor') is False and result.get('TEST_opened') is False and result.get('project_metric_computed') is False and result.get('qualification_or_fit_admission') is False)
    if result_ok:
        for name in ('diagnostic_receipt','publication_neutrality_receipt'):
            pin=result[name];target=Path(pin['path']).resolve()
            require(target.parent==EXECUTION/'diagnostic/run01' and target.stat().st_size==pin['bytes'] and sha(target)==pin['sha256'],'Final scalar receipt custody differs')
        diagnostic=json.loads(Path(result['diagnostic_receipt']['path']).read_text())
        result_ok=bool(diagnostic.get('status')=='DIAGNOSTIC_COMPLETE' and diagnostic.get('admission',{}).get('sha256')==args.admission_sha256 and diagnostic.get('executed_optimizer_updates')==12 and diagnostic.get('caller_rng_restored_exactly') and diagnostic.get('runtime_flags_unchanged') and diagnostic.get('qualification_or_fit_admission') is False)
    capture_ok=bool(cuda and cuda.get('driver_peak_capture_observed') and cuda.get('driver_capture_receipts',{}).get('FINAL.json',{}).get('sha256')==(sha(final_path) if final_path.exists() else None))
    if result_ok and capture_ok:
        capture_ok=all(type(cuda.get(name)) is int and cuda[name]>=result[name] for name in ('cuda_peak_allocated_bytes','cuda_peak_reserved_bytes'))
    if cap_violation is None and cuda:
        exceeded=[name for name in ('cuda_peak_allocated_bytes','cuda_peak_reserved_bytes') if type(cuda.get(name)) is int and cuda[name]>CAPS[name]]
        if exceeded:cap_violation={'kind':'terminal_recorded_CUDA_peak','fields':exceeded}
    success=bool(exit_code==0 and cap_violation is None and cuda and cuda.get('CUDA_observed') and not cuda.get('monitor_failure') and result_ok and capture_ok)
    terminal={'schema':'ncnc-pattern-normal-parity-diagnostic-supervision-terminal-v1','UTC':utc(),'status':'COMPLETE' if success else 'FAILED',
        'supervisor_physical_identity':supervisor,'child_physical_identity':identity,'child_exit_code':exit_code,'child_signal':signal_number,
        'supervisor_inclusive_wall_seconds':time.monotonic()-started,'child_wait_wall_seconds':time.monotonic()-elapsed_start,
        'peak_observed_session_RSS_bytes':peak,'kernel_child_peak_RSS_bytes':peak_kernel,
        'kernel_child_user_CPU_seconds':usage.ru_utime if usage else None,'kernel_child_system_CPU_seconds':usage.ru_stime if usage else None,
        'CUDA_peak_observation':cuda,'cap_violation':cap_violation,'admission_sha256':args.admission_sha256,
        'diagnostic_final_receipt':{'path':str(final_path),'bytes':final_path.stat().st_size,'sha256':sha(final_path)} if final_path.exists() else None,
        'diagnostic_completion_and_neutrality_checks_passed':result_ok,'complete_persistent_CUDA_capture_checks_passed':capture_ok,
        'ordinary_host_execution':True,'namespaces_created':False,'automatic_retry_or_restart':False,'other_jobs_signaled':False,
        'RSS_observation_poll_seconds':.25,'CUDA_caps_checked_from_live_persistent_counters_and_driver_captures':True,
        'terminal_write_tail_measured':False,'scientific_cap_verdict':None,'qualification_or_fit_admission':False,
        'work_cost_bucket':'new_parity_diagnostic_only','prior_failed_qualification_wall_seconds_not_added_here':22.072347790002823}
    write(output/'SUPERVISOR_TERMINAL.json',terminal)
    return 0 if success else 1


if __name__=='__main__':raise SystemExit(main())
