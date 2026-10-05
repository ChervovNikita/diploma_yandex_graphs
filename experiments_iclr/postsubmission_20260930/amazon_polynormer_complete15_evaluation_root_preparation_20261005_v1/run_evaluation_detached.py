
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,time
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
root=phase/'amazon_polynormer_complete15_evaluation_root_preparation_20261005_v1'
launch=root/'detached_launch_v1';release=root/'ROOT_EVALUATION_RELEASE.json'
source=phase/'amazon_polynormer_paired_family_source_preparation_20261003_v6'
output=phase/'amazon_polynormer_paired_family_execution_root_20261003_v3/v6_complete15_evaluation_20261005_v1'
def identity(pid):
 p=Path('/proc')/str(pid);raw=(p/'stat').read_text();v=raw[raw.rfind(')')+2:].split()
 return dict(pid=pid,start_ticks=int(v[19]),state=v[0],argv=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x])
def write(p,v):
 with p.open('x') as h:json.dump(v,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
def desc(p):
 b=p.read_bytes();return dict(path=str(p.relative_to(phase)),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
started=time.perf_counter()
write(launch/'RUNNER_STARTED.json',dict(UTC=datetime.now(timezone.utc).isoformat(),identity=identity(os.getpid()),automatic_retry=False))
command=['/usr/bin/python3','-B',str(source/'supervise.py'),'--kind','evaluate','--release',str(release),'--output',str(output)]
with (launch/'SUPERVISOR_STDOUT.txt').open('x') as out,(launch/'SUPERVISOR_STDERR.txt').open('x') as err:
 child=subprocess.Popen(command,cwd=phase,stdin=subprocess.DEVNULL,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
 write(launch/'SUPERVISOR_LAUNCH.json',dict(UTC=datetime.now(timezone.utc).isoformat(),supervisor_identity=identity(child.pid),runner_identity=identity(os.getpid()),argv=command,cwd=str(phase),release=desc(release),ordinary_unchanged_supervisor=True,detached_runner_survives_SSH=True))
 code=child.wait()
write(launch/'SUPERVISOR_TERMINAL.json',dict(UTC=datetime.now(timezone.utc).isoformat(),physical_supervisor_exit_code=code,status='success' if code==0 else 'failed',whole_runner_wall_seconds=time.perf_counter()-started,release=desc(release),automatic_retry=False))
raise SystemExit(code)
