"""Bounded own-child CPU envelope; no author resource-qualifier process."""
import hashlib,json,os,resource,signal,socket,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;PHASE=HERE.parent;REPO=PHASE.parents[1]
PYTHON=PHASE/'native_ncn_runtime_20261005_v1/.venv/bin/python'
def ticks(pid):return int(Path('/proc/'+str(pid)+'/stat').read_text().rsplit(') ',1)[1].split()[19])
def write(name,value):(HERE/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
started=time.monotonic();os.umask(0o077)
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=5).strip().splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
assert not (HERE/'TERMINAL.json').exists()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(PHASE/'native_ncn_dependency_overlay_20261005_v1')+':'+str(REPO/'.venv/lib/python3.11/site-packages'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2');env.pop('PYTHONHOME',None)
before=resource.getrusage(resource.RUSAGE_CHILDREN);status='worker_failed'
with (HERE/'check.log').open('x') as log:
    child=subprocess.Popen([str(PYTHON),'-B',str(HERE/'run_cpu_check.py')],cwd=REPO,env=env,start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
    identity={'parent_PID':os.getpid(),'parent_start_ticks':ticks(os.getpid()),'worker_PID':child.pid,'worker_start_ticks':ticks(child.pid),'hard_seconds':300,'active_seconds':290,'cleanup_seconds':10}
    write('OWNER.json',identity)
    try:
        code=child.wait(timeout=max(0,290-(time.monotonic()-started)));status='complete' if code==0 else 'worker_failed'
    except subprocess.TimeoutExpired:
        status='active_timeout'
        assert ticks(child.pid)==identity['worker_start_ticks'] and os.getpgid(child.pid)==child.pid
        os.killpg(child.pid,signal.SIGTERM)
        try:code=child.wait(timeout=max(0,min(5,300-(time.monotonic()-started))))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL)
            try:code=child.wait(timeout=max(0,300-(time.monotonic()-started)))
            except subprocess.TimeoutExpired:code=None
after=resource.getrusage(resource.RUSAGE_CHILDREN);elapsed=time.monotonic()-started
complete=status=='complete' and code==0 and elapsed<=300 and (HERE/'CPU_CHECK.json').is_file()
if complete:assert json.loads((HERE/'CPU_CHECK.json').read_text())['complete']
terminal={'complete':complete,'status':status if elapsed<=300 else 'cap_exceeded','exit_code':code,'reaped':code is not None,'inclusive_parent_seconds':elapsed,'cap_exceeded':elapsed>300,'identity':identity,'CPU_user_seconds':after.ru_utime-before.ru_utime,'CPU_system_seconds':after.ru_stime-before.ru_stime,'peak_owned_process_RSS_bytes':after.ru_maxrss*1024,'CPU_check_sha256':hashlib.sha256((HERE/'CPU_CHECK.json').read_bytes()).hexdigest() if complete else None,'TEST_scoring':False,'automatic_retry':False,'other_jobs_modified':False,'GPU_execution':False}
write('TERMINAL.json',terminal)
raise SystemExit(0 if complete else 1)
