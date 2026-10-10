"""Finite owner for the already selected nine stored PubMed prediction banks."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, signal, socket, subprocess, time

R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930'
H=Path(__file__).resolve().parent
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'

def write(q,v):
    with q.open('x') as f: json.dump(v,f,indent=2);f.write('\n')

def identity(pid):
    f=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=int(f[19]),pgid=int(f[2]),sid=int(f[3]),RSS_bytes=int(f[21])*os.sysconf('SC_PAGE_SIZE'))

def check(birth):
    now=identity(birth['pid'])
    assert all(now[k]==birth[k] for k in ('pid','start_ticks','pgid','sid'))
    return now

def size(q):
    return sum(f.stat().st_size for f in q.rglob('*') if f.is_file()) if q.exists() else 0

assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=5).splitlines()==[GPU]
assert H==P/'pubmed_complete9_error_diagnostic_root_20261010_v1'
spec=json.loads((H/'RELEASE.json').read_text());limits=spec['limits']
assert limits['external_active_seconds']==300 and limits['external_cleanup_seconds']==10
assert not (H/'TERMINAL.json').exists()
argv=[spec['runtime']['python']['path'],'-B',str(P/'combination_masked_context_pubmed_complete9_stored_prediction_diagnostic_source_20261010_v1/diagnose.py'),'--mode','diagnose','--release',str(H/'RELEASE.json'),'--release-sha256',hashlib.sha256((H/'RELEASE.json').read_bytes()).hexdigest()]
env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,PYTHONPATH=spec['runtime']['PYTHONPATH'],PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
started=time.monotonic();child=None;birth=None;reason=None;waited=False
peaks=dict(RSS_bytes=0,GPU_bytes=0,output_bytes=0)
try:
    with (H/'WORKER.log').open('xb') as log:
        child=subprocess.Popen(argv,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        birth=identity(child.pid)
        assert birth['pgid']==birth['sid']==child.pid
        write(H/'LAUNCH.json',dict(UTC=datetime.now(timezone.utc).isoformat(),child=birth,argv=argv,automatic_retry=False))
        while child.poll() is None:
            if time.monotonic()-started>=300: reason='active_time_cap';break
            try: now=check(birth)
            except FileNotFoundError:
                if child.poll() is not None: break
                raise
            raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=5)
            gpu=sum(int(row.split(',')[1].strip())*1024**2 for row in raw.splitlines() if row.split(',')[0].strip()==str(child.pid))
            peaks['RSS_bytes']=max(peaks['RSS_bytes'],now['RSS_bytes'])
            peaks['GPU_bytes']=max(peaks['GPU_bytes'],gpu)
            peaks['output_bytes']=max(peaks['output_bytes'],size(Path(spec['output'])))
            for key in peaks:
                if peaks[key]>limits[key]: reason=key+'_cap'
            if (H/'WORKER.log').stat().st_size>limits['log_bytes']: reason='log_cap'
            if reason: break
            time.sleep(.5)
except BaseException as e:
    reason=type(e).__name__+': '+str(e)
finally:
    if child is not None:
        if child.poll() is None:
            check(birth);os.killpg(birth['pgid'],signal.SIGTERM)
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                check(birth);os.killpg(birth['pgid'],signal.SIGKILL);child.wait(timeout=5)
        child.wait(timeout=1);waited=True
    absent=child is not None and not Path('/proc',str(child.pid)).exists()
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=5)
    cuda_absent=child is not None and str(child.pid) not in [r.strip() for r in raw.splitlines()]
    if Path(spec['output']).exists() and size(Path(spec['output']))>limits['output_bytes']: reason=reason or 'terminal_output_cap'
    ok=child is not None and child.returncode==0 and reason is None and waited and absent and cuda_absent
    write(H/'TERMINAL.json',dict(UTC=datetime.now(timezone.utc).isoformat(),complete=ok,child_exit_code=None if child is None else child.returncode,directly_waited=waited,process_absent=absent,CUDA_absent=cuda_absent,reason=reason,inclusive_seconds=time.monotonic()-started,peaks=peaks,automatic_retry=False))
print(json.dumps(dict(complete=ok,seconds=time.monotonic()-started,reason=reason)))
raise SystemExit(0 if ok else 1)
