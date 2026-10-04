"""Ordinary parent for one fixed V5 diagnostic pair or subsequent stdlib closure on18.77."""
import argparse
import importlib.util
from datetime import datetime,timezone
import hashlib,json,os,signal,socket,subprocess,sys,time
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
EXECUTION=PHASE/'graph_ncNC_structural_pattern_execution_root_20261004_v5'
PYTHON=Path('/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12')
PYTHON_SHA='14776d98474f987919376922a9995a20733e13b51d7d122873b068bf2e47d1b2'
GPU_UUIDS={'GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998','GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'}
CAPS={'wall_seconds':1800,'host_RSS_bytes':32*1024**3,'cuda_peak_allocated_bytes':70*1024**3,'cuda_peak_reserved_bytes':75*1024**3}
DRIVER_SHA='9fc539b8f92d7d4e883224c3b0aa85ae583b64c70f2a701a4a648fd818aa32a1'


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


def normal_input(context, common, pin, terminal_pin, root, kind, arm=None):
    path=Path(pin['path']).resolve();terminal_path=Path(terminal_pin['path']).resolve()
    expected=root/('COMPLETE.json' if kind in ('fit','diagnostics') else 'QUALIFICATION.json')
    require(path==expected and sha(path)==pin['sha256'],'Exact completed '+kind+' receipt required')
    expected_terminal=(EXECUTION/('supervision/fit_'+arm+'/run01/SUPERVISOR_TERMINAL.json')) if kind=='fit' else (EXECUTION/'supervision/diagnostics/run01/SUPERVISOR_TERMINAL.json') if kind=='diagnostics' else root.parent.parent/'supervision/run01/SUPERVISOR_TERMINAL.json'
    require(terminal_path==expected_terminal and sha(terminal_path)==terminal_pin['sha256'],'Exact normal '+kind+' terminal required')
    receipt=json.loads(path.read_text());terminal=json.loads(terminal_path.read_text())
    require(receipt['identity']==context['identity'],'Completed '+kind+' source/profile identity differs')
    common.require_profile_receipt(receipt)
    require(terminal['status']=='COMPLETE' and terminal['child_exit_code']==0 and terminal['child_signal'] is None
            and terminal['cap_violation'] is None and terminal['ordinary_host_execution'] is True,
            kind+' did not complete under normal supervision')
    custody=terminal['fit_receipt' if kind=='fit' else 'diagnostics_receipt' if kind=='diagnostics' else 'qualification_receipt']
    require(custody['path']==str(path) and custody['sha256']==pin['sha256'] and custody['bytes']==path.stat().st_size,
            kind+' normal terminal receipt custody differs')
    if kind=='fit':
        require(receipt['schema']=='ncnc-pattern-complete-fit-v1' and receipt['arm']==arm and receipt['seed']==0
                and receipt['epochs']==100 and receipt['optimizer_steps']==1700 and receipt['selection_candidates']==100
                and receipt['full_VALID_evaluations']==102 and receipt['extra_complete_VALID_replay_evaluations']==2
                and receipt['selected_roundtrip_and_full_served_replay'] is True
                and receipt['resource_state_donor'] is False and receipt['test_file_opened'] is False
                and receipt['predictive_values_exposed'] is False and len(receipt['epoch_streams'])==100,
                'Fixed complete '+arm+' fit coverage differs')
        require(terminal['arm']==arm and terminal['seed']==0
                and terminal['complete_fit_work_identity_profile_and_artifacts_checks_passed'] is True
                and terminal['complete_driver_peak_capture_checks_passed'] is True,'Incomplete normal '+arm+' fit terminal')
    if kind=='diagnostics':
        require(receipt['schema']=='ncnc-pattern-diagnostics-complete-v1' and receipt['arms']==['J','F']
                and receipt['status']=='COMPLETE' and receipt['full_selected_VALID_evaluations']==2
                and receipt['full_TRAIN_mask_epochs']==2 and receipt['matched_mask_supports'] is True
                and receipt['test_file_opened'] is False and receipt['predictive_values_exposed'] is False,
                'Fixed completed diagnostics coverage differs')
        require(terminal['stage']=='diagnostics' and terminal['complete_stage_custody_checks_passed'] is True
                and terminal['complete_driver_peak_capture_checks_passed'] is True,'Incomplete normal diagnostics terminal')
    return {'path':str(terminal_path),'bytes':terminal_path.stat().st_size,'sha256':terminal_pin['sha256'],
            'supervisor_inclusive_wall_seconds':terminal['supervisor_inclusive_wall_seconds'],
            'terminal_write_tail_measured':terminal.get('terminal_write_tail_measured',False)}


def main():
    started=time.monotonic()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-release',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--stage',required=True,choices=('diagnostics','close'))
    args=parser.parse_args();release_path=args.root_release.resolve();output=args.output.resolve()
    require(Path.cwd().resolve()==REPO and os.environ.get('GNNM_SSH_DESTINATION')=='shmelev@192.168.18.77','Exact authorized route/repository required')
    require(release_path.is_relative_to(EXECUTION) and output==EXECUTION/('supervision/'+args.stage+'/run01'),'Fresh external stage execution paths required')
    require(not output.exists(),'Supervisor output already exists')
    release=json.loads(release_path.read_text())
    require(release['schema']=='ncnc-structural-pattern-root-release-v1' and release.get('execution_enabled') is True and release.get('root_authorization_reference'),'Stage not root released')
    invocation={'stage':args.stage,'unit':'pair','base_seed':0,'output_directory':str(EXECUTION/(args.stage+'/run01'))}
    require(release['authorized_stages']==[args.stage] and release['authorized_invocations']==[invocation],'One fresh fixed pair stage only')
    require(not Path(invocation['output_directory']).exists(),'Stage output already exists')
    require(release['stage_caps']==CAPS,'Proposed stage caps differ')
    require(Path(sys.executable).resolve()==PYTHON.resolve() and sha(PYTHON)==PYTHON_SHA,'Qualified interpreter differs')
    verify_packet(HERE,release['stage_supervision_manifest_sha256'])
    driver=Path(release['stage_driver_path']).resolve()
    require(driver==PHASE/'graph_ncNC_structural_pattern_pilot_preparation_20261004_v5/pattern_run.py','Only separately reviewed V5 may be released')
    require(release['driver_manifest_sha256']==DRIVER_SHA,'Exact sealed V5 manifest required');verify_packet(driver.parent,DRIVER_SHA)
    spec=importlib.util.spec_from_file_location('v5_stage_profile_contract',driver.parent/'pilot_common.py')
    require(spec is not None and spec.loader is not None,'V5 profile contract loader unavailable')
    common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    require(Path(common.__file__).resolve()==driver.parent/'pilot_common.py' and release.get('runtime_profile_transition')==common.RUNTIME_PROFILE_TRANSITION,'Exact V5 deterministic profile release required')
    context=common.preflight(release_path,args.stage)
    common.admit_invocation(context,args.stage,'pair',0,invocation['output_directory'],False)
    common.qualification_required(context,args.stage)
    require(set(release['qualification'])=={'numerical','full_graph'} and set(release['fit_receipts'])=={'J','F'}
            and set(release['fit_supervision_terminals'])=={'J','F'},'Both completed source qualifications and fixed fits required')
    bound_normal={}
    qualification_roots={'numerical':PHASE/'graph_ncNC_structural_pattern_numerical_execution_root_20261004_v5/numerical/run01',
                         'full_graph':PHASE/'graph_ncNC_structural_pattern_full_graph_execution_root_20261004_v2/full_graph/run01'}
    for stage,root in qualification_roots.items():
        bound_normal[stage]=normal_input(context,common,release['qualification'][stage],release[stage+'_supervision_terminal'],root,stage)
    for arm in ('J','F'):
        bound_normal['fit_'+arm]=normal_input(context,common,release['fit_receipts'][arm],release['fit_supervision_terminals'][arm],EXECUTION/('fit_'+arm+'/run01'),'fit',arm)
    if args.stage=='close':
        bound_normal['diagnostics']=normal_input(context,common,release['diagnostics_receipt'],release['diagnostics_supervision_terminal'],EXECUTION/'diagnostics/run01','diagnostics')
    gpu_rows=[]
    if args.stage=='diagnostics':
        require(release['cuda_visible_devices'] in GPU_UUIDS,'Single bound physical GPU required')
        query=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        gpu_rows=[[part.strip() for part in line.split(',')] for line in query.stdout.splitlines() if line.strip()]
        require(len(gpu_rows)==2 and {r[1] for r in gpu_rows}==GPU_UUIDS,'Wrong two-GPU physical route')
        require(all(r[2]=='NVIDIA A100 80GB PCIe' and int(r[3])==81920 for r in gpu_rows),'Physical GPU profile differs')
    output.mkdir(parents=True,mode=0o700);supervisor=physical(os.getpid())
    write(output/'SUPERVISOR_STARTED.json',{'UTC':utc(),'physical_identity':supervisor,'hostname':socket.gethostname(),
        'root_release_sha256':sha(release_path),'supervision_manifest_sha256':release['stage_supervision_manifest_sha256'],
        'driver_manifest_sha256':DRIVER_SHA,'GPU_rows_before_child':gpu_rows,'caps':CAPS,'stage':args.stage,
        'ordinary_host_execution':True,'namespaces_created':False,'unrelated_jobs_mutated':False,'new_optimization_or_selection':False,
        'automatic_retry_or_resume':False,'bound_completed_normal_stage_terminals':bound_normal})
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=release['cuda_visible_devices'] if args.stage=='diagnostics' else '',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH=str(REPO/'.gnnm_runtime/buddy_extra_v1/site'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    argv=[str(PYTHON),'-B',str(HERE/'stage_child.py'),'--root-release',str(release_path),'--supervision-output',str(output),'--stage',args.stage]
    child=None;identity=None;cap_violation=None;peak=0;elapsed_start=time.monotonic();usage=None;exit_code=None;signal_number=None
    with (output/'CHILD.stdout.log').open('x') as stdout,(output/'CHILD.stderr.log').open('x') as stderr:
        try:
            child=subprocess.Popen(argv,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
            # Popen has completed exec; /proc cmdline/exe are the actual child.
            identity=physical(child.pid)
            require(identity['parent_PID']==os.getpid() and identity['session']==child.pid and identity['process_group']==child.pid,'Unexpected child physical parent/session identity')
            require(identity['namespaces']==supervisor['namespaces'] and identity['exe']==str(PYTHON.resolve()) and identity['cwd']==str(REPO),'Child is not normal host process in exact repository')
            write(output/'CHILD_STARTED.json',{'UTC':utc(),'physical_identity':identity,'requested_argv':argv,'stage_invocation':invocation,'selected_physical_GPU_UUID':release['cuda_visible_devices'] if args.stage=='diagnostics' else None})
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
    capture_path=output/'CHILD_STAGE_CAPTURE.json'
    capture=json.loads(capture_path.read_text()) if capture_path.exists() else None
    if capture and capture.get('cap_violation'):cap_violation=capture['cap_violation']
    stage_output=Path(invocation['output_directory']);receipt=stage_output/('COMPLETE.json' if args.stage=='diagnostics' else 'CLOSURE.json')
    result=json.loads(receipt.read_text()) if receipt.exists() else None
    peak_kernel=int(usage.ru_maxrss*1024) if usage else None
    if cap_violation is None and peak_kernel is not None and peak_kernel>CAPS['host_RSS_bytes']:
        cap_violation={'kind':'kernel_recorded_host_RSS_peak','kernel_child_peak_RSS_bytes':peak_kernel}
    artifacts={};custody_failure=None;stage_ok=False
    def bind(name,pin=None):
        path=stage_output/name
        require(path.is_file() and not path.is_symlink() and path.resolve().parent==stage_output,'Owned stage artifact missing: '+name)
        observed={'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)};artifacts[name]=observed
        if pin is not None:require(pin['path']==name and pin['bytes']==observed['bytes'] and pin['sha256']==observed['sha256'],'Owned stage artifact custody differs: '+name)
    try:
        for name in ('ATTEMPTS.json','FAILED.json','COMPLETE.json','CLOSURE.json','PRIVATE_DIAGNOSTICS.json','PAIR_RESULTS.json','RUNTIME_PROFILE_TRANSITION.json'):
            if (stage_output/name).exists():bind(name)
        require(result and result['identity']==context['identity'],'Complete stage identity missing/different')
        if args.stage=='diagnostics':
            common.require_profile_receipt(result)
            require(result['schema']=='ncnc-pattern-diagnostics-complete-v1' and result['arms']==['J','F'] and result['status']=='COMPLETE'
                    and result['full_selected_VALID_evaluations']==2 and result['full_TRAIN_mask_epochs']==2 and result['matched_mask_supports'] is True
                    and result['predictive_values_exposed'] is False and result['test_file_opened'] is False,'Fixed diagnostic work incomplete')
            bind('PRIVATE_DIAGNOSTICS.json',result['private_diagnostics']);bind('RUNTIME_PROFILE_TRANSITION.json',result['runtime_profile_transition_receipt'])
        else:
            require(result['schema']=='ncnc-pattern-pair-closure-v1' and result['status']=='CLOSED'
                    and result['fit_receipts']==release['fit_receipts'] and result['diagnostics_receipt']==release['diagnostics_receipt']
                    and result['unique_scientific_fits']==2 and result['scientific_optimizer_steps']==3400 and result['scientific_selector_candidates']==200
                    and result['complete_scientific_VALID_evaluations_including_replay_and_diagnostics']==206
                    and result['matched_pair_streams_RNG_and_supports'] is True and result['TEST_supported'] is False
                    and result['prior_failure_receipts']==release.get('prior_failure_receipts',[])
                    and result['cost_scope']=='bound_stage_receipts_and_their_own_ledgers_plus_root_listed_prior_failed_stage_receipts; external_unlisted_attempts_not_certified',
                    'Pair closure work/custody/cost scope differs')
            bind('PAIR_RESULTS.json',result['pair_results'])
        ledger=json.loads((stage_output/'ATTEMPTS.json').read_text())
        require(ledger['identity']==context['identity'] and len(ledger['attempts'])==1 and ledger['attempts'][0]['stage']==args.stage
                and ledger['attempts'][0]['unit']=='pair' and ledger['attempts'][0]['base_seed']==0 and ledger['attempts'][0]['status']=='COMPLETE'
                and ledger['attempts'][0]['root_release_sha256']==sha(release_path) and result['inclusive_accounting']['attempts']==1
                and result['inclusive_accounting']['failed_or_interrupted_attempts']==0 and result['inclusive_accounting']['total_cost_exact'] is True,
                'Fresh stage accounting differs')
        stage_ok=True
    except Exception as error:custody_failure={'type':type(error).__name__,'condition':str(error)}
    capture_ok=bool(capture and capture.get('driver_complete_capture_observed') and not capture.get('monitor_failure'))
    if stage_ok and capture_ok:
        capture_ok=bool(capture.get('driver_capture_receipts',{}).get(receipt.name,{}).get('sha256')==sha(receipt))
        if args.stage=='diagnostics':
            capture_ok=bool(capture_ok and capture.get('CUDA_observed') and all(type(capture.get(name)) is int and capture[name]>=result[name] for name in ('cuda_peak_allocated_bytes','cuda_peak_reserved_bytes')))
        else:capture_ok=bool(capture_ok and capture.get('stdlib_closure_scientific_imports_absent') is True and not capture.get('CUDA_observed'))
    if cap_violation is None and capture:
        exceeded=[name for name in ('cuda_peak_allocated_bytes','cuda_peak_reserved_bytes') if type(capture.get(name)) is int and capture[name]>CAPS[name]]
        if exceeded:cap_violation={'kind':'terminal_recorded_CUDA_peak','fields':exceeded}
    if cap_violation is None and time.monotonic()-started>CAPS['wall_seconds']:cap_violation={'kind':'inclusive_stage_wall','elapsed_seconds':time.monotonic()-started}
    success=bool(exit_code==0 and cap_violation is None and stage_ok and capture_ok)
    terminal={'schema':'ncnc-pattern-normal-diagnostics-closure-supervision-terminal-v1','UTC':utc(),'status':'COMPLETE' if success else 'FAILED',
        'stage':args.stage,'unit':'pair','seed':0,'supervisor_physical_identity':supervisor,'child_physical_identity':identity,
        'child_exit_code':exit_code,'child_signal':signal_number,'supervisor_inclusive_wall_seconds':time.monotonic()-started,
        'child_wait_wall_seconds':time.monotonic()-elapsed_start,'peak_observed_session_RSS_bytes':peak,'kernel_child_peak_RSS_bytes':peak_kernel,
        'kernel_child_user_CPU_seconds':usage.ru_utime if usage else None,'kernel_child_system_CPU_seconds':usage.ru_stime if usage else None,
        'child_stage_capture':capture,'cap_violation':cap_violation,'root_release_sha256':sha(release_path),
        ('diagnostics_receipt' if args.stage=='diagnostics' else 'closure_receipt'):artifacts.get(receipt.name),
        'owned_artifact_receipts':artifacts,'driver_inclusive_accounting':result.get('inclusive_accounting') if result else None,
        'complete_stage_custody_checks_passed':stage_ok,'complete_driver_peak_capture_checks_passed':capture_ok,'owned_artifact_custody_failure':custody_failure,
        'bound_completed_normal_stage_terminals':bound_normal,'bound_normal_stage_supervisor_wall_seconds':sum(row['supervisor_inclusive_wall_seconds'] for row in bound_normal.values()),
        'normal_supervision_and_nested_driver_wall_are_overlapping_not_additive':True,
        'preclosure_bound_stage_accounting':result.get('preclosure_bound_stage_accounting') if result and args.stage=='close' else None,
        'predecessor_or_unlisted_costs_certified':False,'ordinary_host_execution':True,'namespaces_created':False,
        'new_optimization_or_selection':False,'automatic_retry_or_restart':False,'resume_requested':False,'other_jobs_signaled':False,'TEST_opened':False,
        'private_payloads_deserialized_by_supervisor':False,'predictive_values_exposed':False,'RSS_observation_poll_seconds':.25,
        'terminal_write_tail_measured':False}
    write(output/'SUPERVISOR_TERMINAL.json',terminal)
    print('SUPERVISION_TERMINAL stage='+args.stage+' status='+terminal['status'],flush=True)
    return 0 if success else 1


if __name__=='__main__':raise SystemExit(main())
