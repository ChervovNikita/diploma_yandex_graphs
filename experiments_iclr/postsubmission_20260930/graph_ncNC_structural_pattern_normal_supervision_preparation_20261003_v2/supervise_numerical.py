"""Ordinary detached parent for one separately released numerical child on18.77."""
import argparse
from datetime import datetime,timezone
import hashlib,json,os,signal,socket,subprocess,sys,time
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
EXECUTION=PHASE/'graph_ncNC_structural_pattern_numerical_execution_root_20261003_v2'
PYTHON=Path('/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12')
PYTHON_SHA='14776d98474f987919376922a9995a20733e13b51d7d122873b068bf2e47d1b2'
GPU_UUIDS={'GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998','GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'}
CAPS={'wall_seconds':1800,'host_RSS_bytes':16*1024**3,'cuda_peak_allocated_bytes':8*1024**3,'cuda_peak_reserved_bytes':8*1024**3}


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
    parser.add_argument('--root-release',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();release_path=args.root_release.resolve();output=args.output.resolve()
    require(Path.cwd().resolve()==REPO and os.environ.get('GNNM_SSH_DESTINATION')=='shmelev@192.168.18.77','Exact authorized route/repository required')
    require(release_path.is_relative_to(EXECUTION) and output==EXECUTION/'supervision/run01','Fresh external numerical execution paths required')
    require(not output.exists(),'Supervisor output already exists')
    release=json.loads(release_path.read_text())
    require(release['schema']=='ncnc-structural-pattern-root-release-v1' and release.get('root_authorization_reference'),'Numerical stage not root released')
    require(release['authorized_stages']==['numerical'] and len(release['authorized_invocations'])==1,'Supervisor is numerical only')
    invocation=release['authorized_invocations'][0]
    require(invocation=={'stage':'numerical','unit':'pair','base_seed':0,'output_directory':str(EXECUTION/'numerical/run01')},'Wrong exact numerical invocation')
    require(not Path(invocation['output_directory']).exists(),'Numerical output already exists')
    require(release['numerical_caps']==CAPS,'Reviewed candidate numerical caps differ')
    require(release['cuda_visible_devices'] in GPU_UUIDS,'Single bound physical GPU required')
    require(Path(sys.executable).resolve()==PYTHON.resolve() and sha(PYTHON)==PYTHON_SHA,'Qualified interpreter differs')
    verify_packet(HERE,release['supervision_manifest_sha256'])
    driver=Path(release['numerical_driver_path']).resolve()
    require(driver==PHASE/'graph_ncNC_structural_pattern_pilot_preparation_20261003_v3/pattern_run.py','Only independently repaired V3 may be released')
    verify_packet(driver.parent,release['driver_manifest_sha256'])
    query=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    gpu_rows=[[part.strip() for part in line.split(',')] for line in query.stdout.splitlines() if line.strip()]
    require(len(gpu_rows)==2 and {r[1] for r in gpu_rows}==GPU_UUIDS,'Wrong two-GPU physical route')
    require(all(r[2]=='NVIDIA A100 80GB PCIe' and int(r[3])==81920 for r in gpu_rows),'Physical GPU profile differs')
    output.mkdir(parents=True,mode=0o700)
    supervisor=physical(os.getpid())
    write(output/'SUPERVISOR_STARTED.json',{'UTC':utc(),'physical_identity':supervisor,'hostname':socket.gethostname(),
        'root_release_sha256':sha(release_path),'supervision_manifest_sha256':release['supervision_manifest_sha256'],
        'driver_manifest_sha256':release['driver_manifest_sha256'],'GPU_rows_before_child':gpu_rows,'caps':CAPS,
        'ordinary_host_execution':True,'namespaces_created':False,'unrelated_jobs_mutated':False,'scientific_fits_authorized':False})
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=release['cuda_visible_devices'],PYTHONPATH=str(REPO/'.gnnm_runtime/buddy_extra_v1/site'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    argv=[str(PYTHON),'-B',str(HERE/'numerical_child.py'),'--root-release',str(release_path),'--supervision-output',str(output)]
    child=None;identity=None;cap_violation=None;peak=0;elapsed_start=time.monotonic();usage=None;exit_code=None;signal_number=None
    with (output/'CHILD.stdout.log').open('x') as stdout,(output/'CHILD.stderr.log').open('x') as stderr:
        try:
            child=subprocess.Popen(argv,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
            # Popen has completed exec; /proc cmdline/exe are the actual child.
            identity=physical(child.pid)
            require(identity['parent_PID']==os.getpid() and identity['session']==child.pid and identity['process_group']==child.pid,'Unexpected child physical parent/session identity')
            require(identity['namespaces']==supervisor['namespaces'] and identity['exe']==str(PYTHON.resolve()) and identity['cwd']==str(REPO),'Child is not normal host process in exact repository')
            write(output/'CHILD_STARTED.json',{'UTC':utc(),'physical_identity':identity,'requested_argv':argv,'numerical_invocation':invocation,'selected_physical_GPU_UUID':release['cuda_visible_devices']})
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
    receipt=Path(invocation['output_directory'])/'QUALIFICATION.json'
    peak_kernel=int(usage.ru_maxrss*1024) if usage else None
    if cap_violation is None and peak_kernel is not None and peak_kernel>CAPS['host_RSS_bytes']:
        cap_violation={'kind':'kernel_recorded_host_RSS_peak','kernel_child_peak_RSS_bytes':peak_kernel}
    success=bool(exit_code==0 and cap_violation is None and cuda and cuda.get('CUDA_observed') and receipt.exists())
    terminal={'schema':'ncnc-pattern-normal-numerical-supervision-terminal-v1','UTC':utc(),'status':'COMPLETE' if success else 'FAILED',
        'supervisor_physical_identity':supervisor,'child_physical_identity':identity,'child_exit_code':exit_code,'child_signal':signal_number,
        'supervisor_inclusive_wall_seconds':time.monotonic()-started,'child_wait_wall_seconds':time.monotonic()-elapsed_start,
        'peak_observed_session_RSS_bytes':peak,'kernel_child_peak_RSS_bytes':peak_kernel,
        'kernel_child_user_CPU_seconds':usage.ru_utime if usage else None,'kernel_child_system_CPU_seconds':usage.ru_stime if usage else None,
        'CUDA_peak_observation':cuda,'cap_violation':cap_violation,'root_release_sha256':sha(release_path),
        'qualification_receipt':{'path':str(receipt),'bytes':receipt.stat().st_size,'sha256':sha(receipt)} if receipt.exists() else None,
        'ordinary_host_execution':True,'namespaces_created':False,'automatic_retry_or_restart':False,'other_jobs_signaled':False,
        'RSS_observation_poll_seconds':.25,'CUDA_caps_checked_from_persistent_Torch_peak_counters':True,'terminal_write_tail_measured':False}
    write(output/'SUPERVISOR_TERMINAL.json',terminal)
    return 0 if success else 1


if __name__=='__main__':raise SystemExit(main())
