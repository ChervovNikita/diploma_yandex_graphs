"""Stage and launch the unchanged complete-family evaluator exactly once."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
LOGIN='anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
REPO='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
RUNNER=r'''
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
'''
REMOTE=r'''
from pathlib import Path
import base64,hashlib,json,os,socket,subprocess,sys
repo=Path(sys.argv[1]);phase=repo/'experiments_iclr/postsubmission_20260930'
root=phase/'amazon_polynormer_complete15_evaluation_root_preparation_20261005_v1'
source=phase/'amazon_polynormer_paired_family_source_preparation_20261003_v6'
assert repo.resolve()==repo and Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
p=json.load(sys.stdin);op=p['operation']
def verify(d):
 rel=Path(d['path']);assert not rel.is_absolute() and '..' not in rel.parts
 path=phase/rel;assert path.resolve().is_relative_to(phase) and not path.is_symlink()
 b=path.read_bytes();assert len(b)==d['bytes'] and hashlib.sha256(b).hexdigest()==d['sha256'];return path
def write(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
release=root/'ROOT_EVALUATION_RELEASE.json';launch=root/'detached_launch_v1'
if op=='prepare':
 assert not root.exists(),'Do not replace a prior preparation/dispatch'
 a=p['release'];assert a['execution_authorized'] and a['kind']=='evaluate' and a['test_labels_authorized'] is False
 assert a['self_path']==str(release.relative_to(phase))
 for d in [a['source']['manifest'],a['source']['seal'],a['source_review'],*a['custody_inputs']]:verify(d)
 review_row=p['evaluation_source_review'];assert review_row['path']=='amazon_polynormer_v6_complete15_evaluation_source_supplement_20261005_v1/REVIEW.json'
 review_bytes=base64.b64decode(p['evaluation_review_payload']);assert len(review_bytes)==review_row['bytes'] and hashlib.sha256(review_bytes).hexdigest()==review_row['sha256']
 review_path=phase/review_row['path'];assert not review_path.is_symlink() and review_path.resolve().is_relative_to(phase)
 if not review_path.exists():
  review_path.parent.mkdir(parents=True,exist_ok=True)
  with review_path.open('xb') as f:f.write(review_bytes)
 review=verify(review_row);assert json.loads(review.read_text())['status']=='PASS_SOURCE_ONLY_SUPPLEMENT'
 sys.path.insert(0,str(source));import common as c
 assert c.verify_sources()==a['source']
 freeze=json.loads(verify(a['closure_freeze']).read_text());assert freeze['status']=='success' and freeze['source']==a['source']
 root.mkdir();launch.mkdir();write(release,a);write(root/'ROOT_EVALUATION_DECISION.json',p['decision'])
 runner=base64.b64decode(p['runner']);assert hashlib.sha256(runner).hexdigest()==p['runner_sha256']
 with (root/'run_evaluation_detached.py').open('xb') as f:f.write(runner)
 print(json.dumps(dict(prepared=True,release_sha256=hashlib.sha256(release.read_bytes()).hexdigest(),runner_sha256=p['runner_sha256'],launched=False)))
else:
 assert op=='launch' and not (launch/'DISPATCH_INTENT.json').exists()
 a=json.loads(release.read_text());assert a==p['release']
 assert hashlib.sha256((root/'run_evaluation_detached.py').read_bytes()).hexdigest()==p['runner_sha256']
 sys.path.insert(0,str(source));import common as c
 output=c.confined(a['output']);assert not output.exists()
 c.gate(release,'evaluate',output)
 free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=15).splitlines()
 assert len(free)==1;free=int(free[0])*1024*1024
 assert free>=p['decision']['minimum_fresh_GPU_free_bytes'],'Leave registered work undisturbed; do not dispatch below the measured envelope'
 write(launch/'DISPATCH_INTENT.json',dict(release_sha256=hashlib.sha256(release.read_bytes()).hexdigest(),fresh_GPU_free_bytes=free,automatic_retry=False))
 command=['/usr/bin/python3','-I','-S','-B',str(root/'run_evaluation_detached.py')]
 with (launch/'RUNNER_STDOUT.txt').open('x') as out,(launch/'RUNNER_STDERR.txt').open('x') as err:
  child=subprocess.Popen(command,cwd=repo,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
  proc=Path('/proc')/str(child.pid);raw=(proc/'stat').read_text();v=raw[raw.rfind(')')+2:].split()
  identity=dict(pid=child.pid,start_ticks=int(v[19]),argv=command,cwd=str(repo))
 write(launch/'DETACHED_LAUNCH.json',identity)
 print(json.dumps(dict(launched=True,runner_identity=identity,fresh_GPU_free_bytes=free,new_scientific_fits=0,TEST_access=False,automatic_retry=False)))
'''

def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=('prepare','launch'));args=parser.parse_args()
    marker=HERE/(args.operation.upper()+'_DISPATCH_INTENT.json')
    with marker.open('x') as f:json.dump({'UTC':datetime.now(timezone.utc).isoformat(),'automatic_redispatch':False,'client_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},f)
    a=json.loads((HERE/'ROOT_EVALUATION_RELEASE.json').read_text())
    decision=json.loads((HERE/'ROOT_EVALUATION_DECISION.json').read_text())
    review=decision['evaluation_source_review']
    assert hashlib.sha256((PHASE/review['path']).read_bytes()).hexdigest()==review['sha256']
    payload=dict(operation=args.operation,release=a,decision=decision,evaluation_source_review=review,
                 evaluation_review_payload=base64.b64encode((PHASE/review['path']).read_bytes()).decode(),
                 runner=base64.b64encode(RUNNER.encode()).decode(),runner_sha256=hashlib.sha256(RUNNER.encode()).hexdigest())
    command=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
             '-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no',
             '-o','ConnectTimeout=20',LOGIN,'cd '+shlex.quote(REPO)+' && exec '+shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,REPO])]
    started=datetime.now(timezone.utc).isoformat()
    try:
        r=subprocess.run(command,input=json.dumps(payload),capture_output=True,text=True,timeout=90)
        receipt=dict(start_UTC=started,terminal_UTC=datetime.now(timezone.utc).isoformat(),exit_code=r.returncode,stderr=r.stderr,stdout=r.stdout,operation=args.operation,automatic_redispatch=False)
    except Exception as error:
        receipt=dict(start_UTC=started,terminal_UTC=datetime.now(timezone.utc).isoformat(),operation=args.operation,unknown_remote_outcome=True,error_type=type(error).__name__,error=str(error),automatic_redispatch=False)
    receipt['source_client_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    receipt['remote_source_sha256']=hashlib.sha256(REMOTE.encode()).hexdigest()
    with (HERE/(args.operation.upper()+'_TRANSPORT.json')).open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps(receipt))
    if receipt.get('exit_code')!=0:raise SystemExit(1)

if __name__=='__main__':main()
