"""Finite CPU owner for the sealed, resident IMDB confidence-count reader."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json,os,signal,socket,subprocess,sys,time

R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930'
H=Path(__file__).resolve().parent
S=P/'imdb_common_wrong_confidence_bins_source_20261010_v1'
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'

def sha(q):return hashlib.sha256(q.read_bytes()).hexdigest()
def write(q,v):
 with q.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def identity(pid):
 f=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
 return dict(PID=pid,start_ticks=int(f[19]),pgid=int(f[2]),sid=int(f[3]),RSS_bytes=int(f[21])*os.sysconf('SC_PAGE_SIZE'))
def same(birth):
 now=identity(birth['PID']);assert all(now[k]==birth[k] for k in ['PID','start_ticks','pgid','sid']);return now
def size(q):return sum(f.stat().st_size for f in q.rglob('*') if f.is_file()) if q.exists() else 0

def worker():
 spec=importlib.util.spec_from_file_location('_imdb_confidence_counts',S/'__init__.py',submodule_search_locations=[str(S)])
 package=importlib.util.module_from_spec(spec);sys.modules[spec.name]=package;spec.loader.exec_module(package)
 from _imdb_confidence_counts.common import Caps
 from _imdb_confidence_counts.reader import execute
 caps=Caps(source_bound=True,archive_read=True,runtime=True,root_review_sha256=sha(H/'ROOT_REVIEW.json'))
 out=execute(H/'RELEASE.json',caps)
 print(json.dumps(dict(complete=out['complete'],histogram_rows=len(out['rows']))))

def owner():
 release=json.loads((H/'RELEASE.json').read_text());runtime=json.loads((H/'RUNTIME.json').read_text())
 assert release['wall_budget_seconds']==300 and release['finite_external_owner_bound']
 assert not (H/'TERMINAL.json').exists() and not Path(release['output_directory']).exists()
 argv=[runtime['python']['path'],'-B','-P',str(H/'run_owned.py'),'--worker']
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=runtime['PYTHONPATH'],OMP_NUM_THREADS='1',
          OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 started=time.monotonic();child=None;birth=None;reason=None;waited=False
 peaks=dict(RSS_bytes=0,output_bytes=0,log_bytes=0)
 try:
  with (H/'WORKER.log').open('xb') as log:
   child=subprocess.Popen(argv,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
   birth=identity(child.pid);assert birth['pgid']==birth['sid']==child.pid
   write(H/'LAUNCH.json',dict(UTC=datetime.now(timezone.utc).isoformat(),child=birth,argv=argv,
    release_sha256=sha(H/'RELEASE.json'),owner_sha256=sha(H/'run_owned.py'),automatic_retry=False))
   while child.poll() is None:
    if time.monotonic()-started>=300:reason='active_time_cap';break
    try:now=same(birth)
    except FileNotFoundError:
     if child.poll() is not None:break
     raise
    peaks['RSS_bytes']=max(peaks['RSS_bytes'],now['RSS_bytes'])
    peaks['output_bytes']=max(peaks['output_bytes'],size(Path(release['output_directory'])))
    peaks['log_bytes']=max(peaks['log_bytes'],(H/'WORKER.log').stat().st_size)
    if peaks['RSS_bytes']>4*1024**3:reason='RSS_cap'
    if peaks['output_bytes']>16*1024**2:reason='output_cap'
    if peaks['log_bytes']>2*1024**2:reason='log_cap'
    if reason:break
    time.sleep(.25)
 except BaseException as e:reason=type(e).__name__+': '+str(e)
 finally:
  if child is not None:
   if child.poll() is None:
    same(birth);os.killpg(birth['pgid'],signal.SIGTERM)
    try:child.wait(timeout=5)
    except subprocess.TimeoutExpired:
     same(birth);os.killpg(birth['pgid'],signal.SIGKILL);child.wait(timeout=5)
   child.wait(timeout=1);waited=True
  absent=child is not None and not Path('/proc',str(child.pid)).exists()
  gpu_pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=5).splitlines()
  cuda_absent=child is not None and str(child.pid) not in [r.strip() for r in gpu_pids]
  if size(Path(release['output_directory']))>16*1024**2:reason=reason or 'terminal_output_cap'
  inner=Path(release['output_directory'])/'TERMINAL.json'
  result=json.loads(inner.read_text()) if inner.is_file() else None
  ok=child is not None and child.returncode==0 and reason is None and waited and absent and cuda_absent and result is not None and result['complete']
  value=dict(UTC=datetime.now(timezone.utc).isoformat(),complete=ok,child_exit_code=None if child is None else child.returncode,
   directly_waited=waited,process_absent=absent,CUDA_absent=cuda_absent,reason=reason,inclusive_seconds=time.monotonic()-started,
   peaks=peaks,automatic_retry=False,inner_terminal=result,source_manifest_sha256=release['source_manifest_sha256'])
  write(H/'TERMINAL.json',value)
 print(json.dumps(dict(complete=ok,seconds=value['inclusive_seconds'],reason=reason,inner_status=None if result is None else result['status'])))
 raise SystemExit(0 if ok else 1)

def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');args=p.parse_args()
 assert socket.gethostname()=='anogena-2-0'
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=5).splitlines()==[GPU]
 assert H==P/'imdb_confidence_bins_root_execution_20261010_v1' and Path.cwd().resolve()==R
 if args.worker:worker()
 else:owner()

if __name__=='__main__':main()
