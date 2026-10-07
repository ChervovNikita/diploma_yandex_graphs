"""Allocation-only five-minute own-child envelope; resource qualifier untouched."""
import hashlib,json,os,resource,signal,socket,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;PHASE=HERE.parent;REPO=PHASE.parents[1]
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
PYTHON=PHASE/'native_ncn_runtime_20261005_v1/.venv/bin/python'
WIKI=PHASE/'learnable_internal_be_WikiCS_scientific_family_execution_root_20261007_v1'
def ticks(pid):return int(Path('/proc/'+str(pid)+'/stat').read_text().rsplit(') ',1)[1].split()[19])
def write(name,value):(HERE/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def gate():
    assert socket.gethostname()=='anogena-2-0'
    rows=subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=5).strip().splitlines()
    assert len(rows)==1 and rows[0].split(',')[0].strip()==GPU
    free=int(rows[0].split(',')[1].strip())*1024**2;assert free>=8*1024**3
    assert ticks(510850)==6015502511
    running=json.loads((WIKI/'RUNNING_CELL.json').read_text());assert running['cell']=='independent4_contrastive_6101'
    progress=json.loads((WIKI/'fits/outputs/independent4_contrastive_6101/PROGRESS.json').read_text());assert 0<progress['epoch']<600 and progress['complete_epochs']==1100
    live=json.loads((WIKI/'fits/receipts/independent4_contrastive_6101_LIVE.json').read_text());pid=live['supervisor_pid'];assert ticks(pid)==live['supervisor_start_ticks']
    children=Path('/proc/'+str(pid)+'/task/'+str(pid)+'/children').read_text().split();native=[]
    for child in children:
        try:args=Path('/proc/'+child+'/cmdline').read_bytes().decode().strip('\0').split('\0')
        except FileNotFoundError:continue
        if str(PHASE/'learnable_internal_be_contrastive_multitask_suite_20261007_v4/run.py') in args and str(WIKI/'fit_jobs/independent4_contrastive_6101.json') in args:native.append({'PID':int(child),'start_ticks':ticks(int(child))})
    assert len(native)==1 and not (WIKI/'fits/receipts/independent4_contrastive_6101_TERMINAL.json').exists()
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():continue
        try:args=(path/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        assert not any(argument.startswith(str(PHASE)+'/') and 'learnable_internal_be_' in argument and 'resource_qualifier_source_' in argument and Path(argument).name in ('worker.py','supervise.py') for argument in args)
    return {'WikiCS_cell':running['cell'],'WikiCS_epoch':progress['epoch'],'WikiCS_native_child':native[0],
            'WikiCS_owner_PID':510850,'WikiCS_owner_start_ticks':6015502511,'physical_GPU_uuid':GPU,
            'free_GPU_bytes':free,'minimum_free_GPU_bytes':8*1024**3,'no_next_cell_admission':True,'no_qualifier_process_overlap':True}

started=time.monotonic();os.umask(0o077);child=None;code=None;status='admission_pending';peak=0;samples=0;errors=[];before=resource.getrusage(resource.RUSAGE_CHILDREN)
assert not (HERE/'TERMINAL.json').exists()
try:
    admission=gate();write('ADMISSION.json',admission)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,PYTHONPATH=str(PHASE/'native_ncn_dependency_overlay_20261005_v1')+':'+str(REPO/'.venv/lib/python3.11/site-packages'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');env.pop('PYTHONHOME',None)
    cpu=PHASE/'internal_BE_portable_molhiv_cpu_check_20261007_v1'
    command=[str(PYTHON),'-B',str(HERE/'check.py'),'--source',str(PHASE/'portable_internal_be_public_interface_20261007_v2'),'--queue-source',str(PHASE/'portable_internal_be_family_queue_source_20261007_v1'),'--train',str(cpu/'roles/train.npz'),'--valid',str(cpu/'roles/valid.npz'),'--geometry',str(cpu/'TRAIN_BATCH_GEOMETRY.json'),'--output',str(HERE/'outputs'),'--device','cuda:0']
    with (HERE/'check.log').open('x') as log:
        child=subprocess.Popen(command,cwd=REPO,env=env,start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
        identity={'parent_PID':os.getpid(),'parent_start_ticks':ticks(os.getpid()),'worker_PID':child.pid,'worker_start_ticks':ticks(child.pid),'hard_seconds':300,'active_seconds':290,'cleanup_seconds':10}
        write('OWNER.json',identity);last=0
        while child.poll() is None:
            now=time.monotonic()
            if now-started>=290:
                assert ticks(child.pid)==identity['worker_start_ticks'] and os.getpgid(child.pid)==child.pid
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=max(0,min(5,300-(time.monotonic()-started))))
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    try:child.wait(timeout=max(0,300-(time.monotonic()-started)))
                    except subprocess.TimeoutExpired:pass
                status='active_timeout';break
            if now-last>=1 and 290-(now-started)>2:
                last=now
                try:
                    sample=subprocess.check_output(['/usr/bin/nvidia-smi','-i',GPU,'--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=2)
                    values=[int(line.split(',')[1].strip())*1024**2 for line in sample.splitlines() if line.split(',')[0].strip()==str(child.pid)]
                    peak=max([peak]+values);samples+=1
                except Exception as error:errors.append(type(error).__name__)
            time.sleep(.25)
        code=child.poll()
        if status!='active_timeout':status='complete' if code==0 else 'worker_failed'
except BaseException as error:
    write('PARENT_FAILURE.json',{'error_type':type(error).__name__,'error':str(error),'GPU_child_started':child is not None})
    if child is not None and child.poll() is None:
        assert ticks(child.pid)==identity['worker_start_ticks'] and os.getpgid(child.pid)==child.pid
        os.killpg(child.pid,signal.SIGKILL)
        try:child.wait(timeout=max(0,300-(time.monotonic()-started)))
        except subprocess.TimeoutExpired:pass
    code=child.poll() if child else None
    status='admission_pending' if child is None else 'parent_failure'
after=resource.getrusage(resource.RUSAGE_CHILDREN);elapsed=time.monotonic()-started
candidate_path=HERE/'outputs/CANDIDATE.json';candidate=json.loads(candidate_path.read_text()) if status=='complete' and candidate_path.is_file() else None
complete=status=='complete' and code==0 and elapsed<=300 and candidate is not None and candidate['complete'] and not errors
terminal={'complete':complete,'status':status if elapsed<=300 else 'cap_overrun','exit_code':code,'reaped':code is not None,'inclusive_parent_seconds':elapsed,
          'hard_seconds':300,'cap_overrun':elapsed>300,'CPU_user_seconds':after.ru_utime-before.ru_utime,'CPU_system_seconds':after.ru_stime-before.ru_stime,
          'cumulative_owned_process_peak_RSS_bytes':after.ru_maxrss*1024,'sampled_worker_GPU_peak_bytes':peak,'driver_samples':samples,'monitor_errors':errors,
          'GPU_child_started':child is not None,'other_jobs_modified':False,'TEST_scoring':False,'automatic_retry':False,'quality_values_closed':True}
write('TERMINAL.json',terminal)
if complete:
    receipt={**candidate,'schema':'portable-representative-work-v1','exit_code':0,'reaped':True,'peak_GPU_bytes':max(peak,candidate['peak_GPU_bytes']),
             'parent_terminal_sha256':hashlib.sha256((HERE/'TERMINAL.json').read_bytes()).hexdigest(),'identity':identity,'concurrency':admission,
             'parent_costs':terminal,'inclusive_parent_seconds':time.monotonic()-started}
    if receipt['inclusive_parent_seconds']>300:receipt['complete']=False;complete=False
    write('WORK_RECEIPT.json',receipt)
raise SystemExit(0 if complete else 1)
