"""Ordinary detached parent for one fixed100-epoch arm/seed fit on18.77."""
import argparse
import importlib.util
from datetime import datetime,timezone
import hashlib,json,os,signal,socket,subprocess,sys,time
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
EXECUTION=PHASE/'exact_cb_support_bucket_paired_predictive_execution_root_20261004_v1'
PYTHON=Path('/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12')
PYTHON_SHA='14776d98474f987919376922a9995a20733e13b51d7d122873b068bf2e47d1b2'
GPU_UUIDS={'GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998','GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'}
CAPS={'wall_seconds':86400,'host_RSS_bytes':32*1024**3,'cuda_peak_allocated_bytes':70*1024**3,'cuda_peak_reserved_bytes':75*1024**3}
DRIVER_SHA=None


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
    global DRIVER_SHA
    DRIVER_SHA=sha(HERE/'MANIFEST.json')
    started=time.monotonic()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-release',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--arm',required=True,choices=('target_only','joint','separate'))
    parser.add_argument('--seed',required=True,type=int,choices=(0,1,2))
    args=parser.parse_args();release_path=args.root_release.resolve();output=args.output.resolve()
    require(Path.cwd().resolve()==REPO and os.environ.get('GNNM_SSH_DESTINATION')=='shmelev@192.168.18.77','Exact authorized route/repository required')
    require(release_path.is_relative_to(EXECUTION) and output==EXECUTION/('supervision/fit_'+args.arm+'_seed'+str(args.seed)+'/run01'),'Fresh external per-arm fit execution paths required')
    require(not output.exists(),'Supervisor output already exists')
    release=json.loads(release_path.read_text())
    require(release['schema']=='ncnc-structural-pattern-root-release-v1' and release.get('execution_enabled') is True and release.get('root_authorization_reference'),'Fit stage not root released')
    require(release['authorized_stages']==['fit'] and len(release['authorized_invocations'])==1,'Supervisor is one fresh fit only')
    invocation=release['authorized_invocations'][0]
    require(invocation=={'stage':'fit','unit':args.arm,'base_seed':args.seed,'output_directory':str(EXECUTION/('fit_'+args.arm+'_seed'+str(args.seed)+'/run01'))},'Wrong exact fixed fit invocation')
    require(not Path(invocation['output_directory']).exists(),'Fit output already exists')
    require(release['fit_caps']==CAPS,'Proposed per-fit caps differ')
    require(release['cuda_visible_devices'] in GPU_UUIDS,'Single bound physical GPU required')
    require(Path(sys.executable).resolve()==PYTHON.resolve() and sha(PYTHON)==PYTHON_SHA,'Qualified interpreter differs')
    verify_packet(HERE,release['fit_supervision_manifest_sha256'])
    driver=Path(release['fit_driver_path']).resolve()
    require(driver==HERE/'paired_run.py','Only exact reviewed paired driver may be released')
    require(release['driver_manifest_sha256']==DRIVER_SHA,'Exact reviewed predictive source required')
    verify_packet(driver.parent,DRIVER_SHA)
    spec=importlib.util.spec_from_file_location('v5_fit_profile_contract',driver.parent/'pilot_common.py')
    require(spec is not None and spec.loader is not None,'V5 profile contract loader unavailable')
    common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    require(Path(common.__file__).resolve()==driver.parent/'pilot_common.py'
            and release.get('runtime_profile_transition')==common.RUNTIME_PROFILE_TRANSITION,'Exact V5 deterministic profile release required')
    context=common.preflight(release_path,'fit')
    common.admit_invocation(context,'fit',args.arm,args.seed,invocation['output_directory'],False)
    require(set(release['qualification'])=={'numerical','full_graph'},'Both actual V5 qualifications required')
    common.qualification_required(context,'fit')
    qualification_roots={'numerical':PHASE/'graph_ncNC_structural_pattern_numerical_execution_root_20261004_v5',
                         'full_graph':PHASE/'graph_ncNC_structural_pattern_full_graph_execution_root_20261004_v2'}
    for stage,root in qualification_roots.items():
        prior=release['qualification'][stage];prior_path=Path(prior['path']).resolve()
        require(prior_path==root/(stage+'/run01/QUALIFICATION.json'),'Wrong '+stage+' prerequisite path')
        terminal_pin=release[stage+'_supervision_terminal'];terminal_path=Path(terminal_pin['path']).resolve()
        require(terminal_path==root/'supervision/run01/SUPERVISOR_TERMINAL.json'
                and sha(terminal_path)==terminal_pin['sha256'],'Exact '+stage+' physical supervision custody required')
        terminal=json.loads(terminal_path.read_text())
        require(terminal['status']=='COMPLETE' and terminal['child_exit_code']==0 and terminal['child_signal'] is None
                and terminal['cap_violation'] is None and terminal['ordinary_host_execution'] is True,
                stage+' did not complete under normal supervision')
        require(terminal['qualification_receipt']['sha256']==prior['sha256']
                and terminal['qualification_receipt']['path']==str(prior_path),stage+' supervision qualification custody differs')
    query=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    gpu_rows=[[part.strip() for part in line.split(',')] for line in query.stdout.splitlines() if line.strip()]
    require(len(gpu_rows)==2 and {r[1] for r in gpu_rows}==GPU_UUIDS,'Wrong two-GPU physical route')
    require(all(r[2]=='NVIDIA A100 80GB PCIe' and int(r[3])==81920 for r in gpu_rows),'Physical GPU profile differs')
    output.mkdir(parents=True,mode=0o700)
    supervisor=physical(os.getpid())
    write(output/'SUPERVISOR_STARTED.json',{'UTC':utc(),'physical_identity':supervisor,'hostname':socket.gethostname(),
        'root_release_sha256':sha(release_path),'supervision_manifest_sha256':release['fit_supervision_manifest_sha256'],
        'driver_manifest_sha256':release['driver_manifest_sha256'],'GPU_rows_before_child':gpu_rows,'caps':CAPS,
        'ordinary_host_execution':True,'namespaces_created':False,'unrelated_jobs_mutated':False,'scientific_fit_authorized':True,'arm':args.arm,'base_seed':args.seed,'automatic_retry_or_resume':False})
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=release['cuda_visible_devices'],CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH=str(REPO/'.gnnm_runtime/buddy_extra_v1/site'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    argv=[str(PYTHON),'-B',str(HERE/'fit_child.py'),'--root-release',str(release_path),'--supervision-output',str(output),'--arm',args.arm,'--seed',str(args.seed)]
    child=None;identity=None;cap_violation=None;peak=0;elapsed_start=time.monotonic();usage=None;exit_code=None;signal_number=None
    with (output/'CHILD.stdout.log').open('x') as stdout,(output/'CHILD.stderr.log').open('x') as stderr:
        try:
            child=subprocess.Popen(argv,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
            # Popen has completed exec; /proc cmdline/exe are the actual child.
            identity=physical(child.pid)
            require(identity['parent_PID']==os.getpid() and identity['session']==child.pid and identity['process_group']==child.pid,'Unexpected child physical parent/session identity')
            require(identity['namespaces']==supervisor['namespaces'] and identity['exe']==str(PYTHON.resolve()) and identity['cwd']==str(REPO),'Child is not normal host process in exact repository')
            write(output/'CHILD_STARTED.json',{'UTC':utc(),'physical_identity':identity,'requested_argv':argv,'fit_invocation':invocation,'selected_physical_GPU_UUID':release['cuda_visible_devices']})
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
    fit_output=Path(invocation['output_directory'])
    receipt=fit_output/'COMPLETE.json'
    peak_kernel=int(usage.ru_maxrss*1024) if usage else None
    if cap_violation is None and peak_kernel is not None and peak_kernel>CAPS['host_RSS_bytes']:
        cap_violation={'kind':'kernel_recorded_host_RSS_peak','kernel_child_peak_RSS_bytes':peak_kernel}
    result=json.loads(receipt.read_text()) if receipt.exists() else None
    fit_ok=bool(result and result.get('schema')=='ncnc-pattern-complete-fit-v1'
        and result.get('identity')==context['identity'] and result.get('arm')==args.arm and result.get('seed')==args.seed
        and result.get('epochs')==100 and result.get('optimizer_steps')==1700 and result.get('selection_candidates')==100
        and result.get('full_VALID_evaluations')==102 and result.get('extra_complete_VALID_replay_evaluations')==2
        and result.get('selected_roundtrip_and_full_served_replay') is True
        and result.get('resource_state_donor') is False and result.get('test_file_opened') is False
        and result.get('predictive_values_exposed') is False and len(result.get('epoch_streams',[]))==100)
    profile_failure=None
    if fit_ok:
        try:common.require_profile_receipt(result)
        except Exception as error:
            fit_ok=False
            profile_failure={'exception_type':type(error).__name__,'condition':str(error)}
    artifacts={};artifact_failure=None
    def bind(name,pin=None):
        if name not in artifacts:
            path=fit_output/name
            require(path.is_file() and not path.is_symlink() and path.resolve().parent==fit_output,'Owned fit artifact missing: '+name)
            artifacts[name]={'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}
        observed=artifacts[name]
        if pin is not None:
            require(pin['path']==name and pin['bytes']==observed['bytes'] and pin['sha256']==observed['sha256'],
                    'Own fit artifact custody differs: '+name)
    try:
        for name in ('ATTEMPTS.json','JOURNAL.json','STATE_SLOT_0.pt','STATE_SLOT_1.pt','SELECTED_'+args.arm+'.pt',
                     'PRIVATE_SELECTION_'+args.arm+'.json','RUNTIME_PROFILE_TRANSITION.json','EPOCHS.jsonl','VALID_LOGITS.pt','FAILED.json','COMPLETE.json'):
            if (fit_output/name).exists():bind(name)
        if fit_ok:
            bind('SELECTED_'+args.arm+'.pt',result['selected_checkpoint'])
            bind('PRIVATE_SELECTION_'+args.arm+'.json',result['private_selection'])
            bind('JOURNAL.json',result['journal'])
            bind('EPOCHS.jsonl',result['epoch_log'])
            bind('VALID_LOGITS.pt',result['selected_VALID_logits'])
            bind('RUNTIME_PROFILE_TRANSITION.json',result['runtime_profile_transition_receipt'])
            journal=json.loads((fit_output/'JOURNAL.json').read_text())
            require(journal['identity']==context['identity'] and journal['unit']==args.arm and journal['seed']==args.seed
                    and journal['epoch']==100 and journal['state_file']['path'] in ('STATE_SLOT_0.pt','STATE_SLOT_1.pt'),
                    'Complete own fit journal differs')
            bind(journal['state_file']['path'],journal['state_file'])
            accounting=json.loads((fit_output/'ATTEMPTS.json').read_text())
            require(accounting['identity']==context['identity'] and len(accounting['attempts'])==1
                    and accounting['attempts'][0]['stage']=='fit' and accounting['attempts'][0]['unit']==args.arm
                    and accounting['attempts'][0]['status']=='COMPLETE'
                    and accounting['attempts'][0]['root_release_sha256']==sha(release_path)
                    and result['inclusive_accounting']['attempts']==1
                    and result['inclusive_accounting']['failed_or_interrupted_attempts']==0
                    and result['inclusive_accounting']['total_cost_exact'] is True,'Fresh completed fit accounting differs')
    except Exception as error:
        fit_ok=False
        artifact_failure={'exception_type':type(error).__name__,'condition':str(error)}
    capture_ok=bool(cuda and cuda.get('driver_complete_capture_observed'))
    if fit_ok and capture_ok:
        capture_ok=bool(cuda.get('driver_capture_receipts',{}).get('COMPLETE.json',{}).get('sha256')==sha(receipt)
            and all(type(cuda.get(name)) is int and cuda[name]>=result[name]
                    for name in ('cuda_peak_allocated_bytes','cuda_peak_reserved_bytes')))
    if cap_violation is None and cuda:
        exceeded=[name for name in ('cuda_peak_allocated_bytes','cuda_peak_reserved_bytes')
                  if type(cuda.get(name)) is int and cuda[name]>CAPS[name]]
        if exceeded:cap_violation={'kind':'terminal_recorded_CUDA_peak','fields':exceeded}
    if cap_violation is None and time.monotonic()-started>CAPS['wall_seconds']:
        cap_violation={'kind':'inclusive_fit_wall','elapsed_seconds':time.monotonic()-started}
    success=bool(exit_code==0 and cap_violation is None and cuda and cuda.get('CUDA_observed')
        and not cuda.get('monitor_failure') and fit_ok and capture_ok)
    terminal={'schema':'ncnc-pattern-normal-fit-supervision-terminal-v1','UTC':utc(),'status':'COMPLETE' if success else 'FAILED',
        'arm':args.arm,'seed':args.seed,'supervisor_physical_identity':supervisor,'child_physical_identity':identity,
        'child_exit_code':exit_code,'child_signal':signal_number,
        'supervisor_inclusive_wall_seconds':time.monotonic()-started,'child_wait_wall_seconds':time.monotonic()-elapsed_start,
        'peak_observed_session_RSS_bytes':peak,'kernel_child_peak_RSS_bytes':peak_kernel,
        'kernel_child_user_CPU_seconds':usage.ru_utime if usage else None,'kernel_child_system_CPU_seconds':usage.ru_stime if usage else None,
        'CUDA_peak_observation':cuda,'cap_violation':cap_violation,'root_release_sha256':sha(release_path),
        'fit_receipt':artifacts.get('COMPLETE.json'),'owned_artifact_receipts':artifacts,
        'driver_inclusive_accounting':result.get('inclusive_accounting') if result else None,
        'runtime_profile_receipt_failure':profile_failure,'owned_artifact_custody_failure':artifact_failure,
        'complete_fit_work_identity_profile_and_artifacts_checks_passed':fit_ok,'complete_driver_peak_capture_checks_passed':capture_ok,
        'ordinary_host_execution':True,'namespaces_created':False,'automatic_retry_or_restart':False,'resume_requested':False,
        'other_jobs_signaled':False,'TEST_opened':False,'scientific_state_or_private_selection_payloads_deserialized_by_supervisor':False,
        'RSS_observation_poll_seconds':.25,'CUDA_caps_checked_from_live_persistent_counters_and_final_driver_peaks':True,
        'terminal_write_tail_measured':False}
    write(output/'SUPERVISOR_TERMINAL.json',terminal)
    return 0 if success else 1



if __name__=='__main__':raise SystemExit(main())
