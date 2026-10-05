"""Stage exact project files, run synthetic CPU qualification, or launch development."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import base64
import hashlib
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
P = HERE.parent
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
SOURCE = 'amazon_polynormer_logits_graph_moment_source_preparation_20261005_v1'
PROTOCOL = 'amazon_polynormer_logits_graph_moment_retrospective_protocol_20261005_v2'
QUALIFIER = 'amazon_polynormer_logits_graph_moment_root_cpu_qualification_preparation_20261005_v1'
OUTPUT = 'amazon_polynormer_logits_graph_moment_retrospective_cpu_execution_root_20261005_v1'

REMOTE = r'''
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,socket,subprocess,sys
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
packet=json.load(sys.stdin);mode=packet['mode']
def path(rel):
 p=Path(rel);assert not p.is_absolute() and '..' not in p.parts
 p=phase/p;assert p.resolve().is_relative_to(phase)
 assert not any(q.is_symlink() for q in (p,*p.parents) if q.is_relative_to(phase))
 return p
def desc(p):
 b=p.read_bytes();return dict(path=str(p.relative_to(phase)),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def verify(d):
 p=path(d['path']);assert desc(p)==d;return p
def write(p,value):
 with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
root=path(packet['root']);root.mkdir(exist_ok=True)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',BLIS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
python=repo/'.venv/bin/python'
if mode=='stage':
 rows=packet['files'];assert len({r['path'] for r in rows})==len(rows)
 for row in rows:
  data=base64.b64decode(row['data']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
  p=path(row['path'])
  if p.exists():assert p.read_bytes()==data,'Never overwrite differing original/staged project file'
  else:
   p.parent.mkdir(parents=True,exist_ok=True)
   with p.open('xb') as f:f.write(data)
 print(json.dumps(dict(staged_files=len(rows),new_training=0,scientific_payload_access=False)))
elif mode=='qualify':
 qualifier=verify(packet['qualifier']);source=path(packet['source'])
 for row in packet['source_descriptors']:verify(row)
 out=root/'NUMERICAL_QUALIFICATION';assert not out.exists()
 command=[str(python),'-B',str(qualifier),'--source',str(source),'--output',str(out)]
 with (root/'QUALIFICATION_STDOUT.txt').open('x') as stdout,(root/'QUALIFICATION_STDERR.txt').open('x') as stderr:
  result=subprocess.run(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,timeout=120)
 write(root/'QUALIFICATION_TERMINAL.json',dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=result.returncode,argv=command,source_descriptors=packet['source_descriptors'],no_scientific_payload_access=True))
 print(json.dumps(dict(exit_code=result.returncode,stdout=(root/'QUALIFICATION_STDOUT.txt').read_text(),stderr=(root/'QUALIFICATION_STDERR.txt').read_text())))
 sys.exit(result.returncode)
elif mode=='launch':
 release=packet['release'];assert release['development_authorized'] and not release['TEST_authorized'] and release['final_refits']==0
 for row in release['inputs']:verify(row)
 qualification=json.loads(verify(release['qualification']).read_text())
 assert qualification['status']=='PASS_SYNTHETIC_NUMERICAL_QUALIFICATION'
 review=json.loads(verify(release['source_review']).read_text())
 assert review['status']=='PASS_SOURCE_REVIEW'
 output=path(packet['output']);assert not output.exists()
 write(root/'ROOT_DEVELOPMENT_RELEASE.json',release)
 runner=base64.b64decode(packet['runner']);assert hashlib.sha256(runner).hexdigest()==packet['runner_sha256']
 runner_path=root/'run_detached.py'
 with runner_path.open('xb') as f:f.write(runner)
 command=['/usr/bin/python3','-B',str(runner_path)]
 with (root/'RUNNER_STDOUT.txt').open('x') as stdout,(root/'RUNNER_STDERR.txt').open('x') as stderr:
  child=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
  proc=Path('/proc')/str(child.pid);raw=(proc/'stat').read_text();values=raw[raw.rfind(')')+2:].split()
  identity=dict(pid=child.pid,start_ticks=int(values[19]),argv=command,cwd=str(repo))
 write(root/'DETACHED_LAUNCH.json',dict(UTC=datetime.now(timezone.utc).isoformat(),runner_identity=identity,automatic_retry=False,source=packet['source'],output=packet['output']))
 print(json.dumps(dict(launched=True,runner_identity=identity,new_backbone_fits=0,TEST_access=False,CPU_only=True)))
else:raise ValueError(mode)
'''

RUNNER = r'''
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,resource,subprocess,time
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
root=Path(__file__).resolve().parent
release=json.loads((root/'ROOT_DEVELOPMENT_RELEASE.json').read_text())
def identity(pid):
 p=Path('/proc')/str(pid);raw=(p/'stat').read_text();v=raw[raw.rfind(')')+2:].split()
 return dict(pid=pid,start_ticks=int(v[19]),argv=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x],cwd=str((p/'cwd').resolve()))
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
command=release['argv'];start=time.perf_counter()
with (root/'DEVELOPMENT_STDOUT.txt').open('x') as out,(root/'DEVELOPMENT_STDERR.txt').open('x') as err:
 child=subprocess.Popen(command,cwd=repo,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=out,stderr=err)
 write(root/'CHILD_IDENTITY.json',dict(UTC=datetime.now(timezone.utc).isoformat(),identity=identity(child.pid),release_sha256=hashlib.sha256((root/'ROOT_DEVELOPMENT_RELEASE.json').read_bytes()).hexdigest()))
 code=child.wait()
result=phase/release['output']/'RESULT.json'
value=dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=code,wall_seconds=time.perf_counter()-start,peak_child_RSS_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,automatic_retry=False,CPU_only=True,TEST_access=False)
if code==0:
 assert result.is_file()
 b=result.read_bytes();value['result']=dict(path=str(result.relative_to(phase)),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
write(root/'DEVELOPMENT_TERMINAL.json',value)
raise SystemExit(code)
'''

def descriptor(p):
    b=p.read_bytes()
    return dict(path=str(p.relative_to(P)),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('stage','qualify','launch'))
    parser.add_argument('--stage-list')
    parser.add_argument('--receipt',required=True)
    args=parser.parse_args()
    receipt=HERE/args.receipt
    assert not receipt.exists()
    packet=dict(mode=args.mode,root=HERE.name,source=SOURCE,output=OUTPUT)
    source_files=sorted((P/SOURCE).iterdir())
    if args.mode=='stage':
        names=json.loads((HERE/args.stage_list).read_text())
        packet['files']=[]
        for name in names:
            p=P/name
            assert p.resolve().is_relative_to(P) and p.is_file() and not p.is_symlink() and p.stat().st_size<2_000_000
            packet['files'].append(dict(descriptor(p),data=base64.b64encode(p.read_bytes()).decode()))
    elif args.mode=='qualify':
        packet['qualifier']=descriptor(P/QUALIFIER/'qualify_numerical.py')
        packet['source_descriptors']=[descriptor(p) for p in source_files if p.is_file()]
    else:
        packet['release']=json.loads((HERE/'ROOT_DEVELOPMENT_RELEASE.json').read_text())
        packet['runner']=base64.b64encode(RUNNER.encode()).decode()
        packet['runner_sha256']=hashlib.sha256(RUNNER.encode()).hexdigest()
    command=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=20',LOGIN,
             'cd '+shlex.quote(REPO)+' && exec /usr/bin/python3 -I -S -B -c '+shlex.quote(REMOTE)]
    before=datetime.now(timezone.utc).isoformat()
    r=subprocess.run(command,input=json.dumps(packet),capture_output=True,text=True,timeout=160)
    value=dict(start_UTC=before,terminal_UTC=datetime.now(timezone.utc).isoformat(),mode=args.mode,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,remote_source_sha256=hashlib.sha256(REMOTE.encode()).hexdigest(),local_client_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),automatic_retry=False)
    with receipt.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(value))
    raise SystemExit(r.returncode)

if __name__=='__main__':main()
