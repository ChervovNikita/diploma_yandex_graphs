"""Root-released native resource qualification; bounded CPU, authorized SSH route."""
import argparse
import base64
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
REPO='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
LOGIN='anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
REMOTE=r'''
import base64,hashlib,json,os,pathlib,resource,subprocess,sys,time
repo=pathlib.Path(sys.argv[1]);request=json.load(sys.stdin)
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):
 digest=hashlib.sha256()
 with path.open('rb') as stream:
  for data in iter(lambda:stream.read(1<<20),b''):digest.update(data)
 return digest.hexdigest()
def verify(row):
 path=pathlib.Path(row['path']);require(sha(path)==row['sha256'] and path.stat().st_size==row['bytes'],'Source/input changed: '+str(path));return path
def write(path,value):
 with path.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')
require(subprocess.run(['git','rev-parse','--show-toplevel'],cwd=repo,capture_output=True,text=True,check=True).stdout.strip()==str(repo),'Wrong canonical repository')
head=subprocess.run(['git','rev-parse','HEAD'],cwd=repo,capture_output=True,text=True,check=True).stdout.strip()
require(subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac'],'Wrong authorized one-GPU account inventory')
phase=repo/'experiments_iclr/postsubmission_20260930';base=phase/'graph_heterogeneous_dblp_native_resource_qualification_20261003_v1';base.mkdir(exist_ok=True)
for row in request['files']:
 relative=pathlib.PurePosixPath(row['path']);require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe custody path')
 path=phase/relative;require(path.resolve().is_relative_to(phase),'Custody path outside canonical phase')
 data=base64.b64decode(row['data'],validate=True);require(hashlib.sha256(data).hexdigest()==row['sha256'] and len(data)==row['bytes'],'Transport bytes differ')
 if path.exists():require(path.read_bytes()==data,'Immutable existing source/input differs')
 else:
  path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as stream:stream.write(data)
manifest_sha=sha(base/'MANIFEST.json');require(manifest_sha==request['manifest_sha256'] and json.loads((base/'SEAL.json').read_text())['manifest_sha256']==manifest_sha,'Qualification seal differs')
binding=json.loads((base/'BINDINGS.json').read_text());manifest=json.loads((base/'MANIFEST.json').read_text())
for row in manifest['payload']:verify(dict(row,path=str(base/row['path'])))
for row in binding['source_records']+[binding['freeze'],binding['paired_HGT_freeze']]:verify(row)
data=base64.b64decode(request['release_data'],validate=True);release=json.loads(data)
require(hashlib.sha256(data).hexdigest()==request['release_sha256'] and release['qualification_authorized'] is True
 and release['execution_authorized'] is False,'Separate root resource-only release required')
run_name=release['run_name'];require(pathlib.Path(run_name).name==run_name and run_name not in ('','.','..'),'Fresh simple qualification run name')
admissions=base/'admissions';admissions.mkdir(exist_ok=True);release_path=admissions/(run_name+'.json')
if release_path.exists():require(release_path.read_bytes()==data,'Existing root qualification release differs')
else:
 with release_path.open('xb') as stream:stream.write(data)
out=base/'transport'/run_name;out.mkdir(parents=True,exist_ok=False);limits=binding['limits']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
# Root admission reads only source/freeze/CPU receipt metadata; it precedes Torch/data execution.
guard='import importlib.util,json,sys; s=importlib.util.spec_from_file_location("native_resource_guard",sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); b=json.load(open(sys.argv[2])); m.admission(b,json.load(open(b["freeze"]["path"])),json.load(open(sys.argv[3])),sys.argv[4],sys.argv[5])'
subprocess.run(['/usr/bin/python3','-I','-S','-B','-c',guard,str(base/'qualify_native.py'),str(base/'BINDINGS.json'),str(release_path),manifest_sha,run_name],cwd=repo,env=env,check=True)
memory={}
for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
 if line.startswith('MemAvailable:'):memory['host_available_bytes']=int(line.split()[1])*1024
try:
 maximum=pathlib.Path('/sys/fs/cgroup/memory.max').read_text().strip()
 if maximum!='max':memory['cgroup_available_bytes']=int(maximum)-int(pathlib.Path('/sys/fs/cgroup/memory.current').read_text())
except (OSError,ValueError):pass
free=min(memory.values());affinity=len(os.sched_getaffinity(0));load=os.getloadavg()[0]
preflight=dict(memory,free_bytes=free,affinity_CPUs=affinity,one_minute_load=load,CUDA_VISIBLE_DEVICES='',threads=1,
 preimport_RLIMIT_AS_bytes=limits['address_space_limit_bytes'],RSS_limit_bytes=limits['RSS_limit_bytes'],
 RLIMIT_CPU_soft_seconds=limits['CPU_soft_seconds'],RLIMIT_CPU_hard_seconds=limits['CPU_hard_seconds'],wall_budget_seconds=limits['wall_budget_seconds'])
write(out/'PREFLIGHT.json',preflight)
receipt=dict(status='resource_deferred',exit_code=1,current_HEAD=head,preflight=preflight,qualification_only=True,
 training_driver_main_called=False,GPU_computation=False,root_release_sha256=request['release_sha256'],qualification_manifest_sha256=manifest_sha)
if free>=limits['required_free_host_bytes'] and affinity>=1 and load<=.75*affinity:
 def bounded():
  resource.setrlimit(resource.RLIMIT_AS,(limits['address_space_limit_bytes'],)*2)
  resource.setrlimit(resource.RLIMIT_CPU,(limits['CPU_soft_seconds'],limits['CPU_hard_seconds']))
 argv=[str(repo/'.venv/bin/python'),'-B',str(base/'qualify_native.py'),'--admission',str(release_path),'--run-name',run_name]
 start=time.monotonic();status='running';peak=0;last_report=0
 with (out/'stdout.log').open('x') as stdout,(out/'stderr.log').open('x') as stderr:
  child=subprocess.Popen(argv,cwd=repo,env=env,stdout=stdout,stderr=stderr,preexec_fn=bounded)
  print(json.dumps(dict(status='native_CPU_resource_qualification_started',pid=child.pid,preflight=preflight)),flush=True)
  try:
   while child.poll() is None:
    elapsed=time.monotonic()-start
    try:
     for line in pathlib.Path('/proc',str(child.pid),'status').read_text().splitlines():
      if line.startswith('VmRSS:'):peak=max(peak,int(line.split()[1])*1024)
    except OSError:pass
    if elapsed>limits['wall_budget_seconds'] or peak>limits['RSS_limit_bytes']:
     status='wall_budget_exhausted' if elapsed>limits['wall_budget_seconds'] else 'RSS_budget_exhausted';child.terminate();break
    if elapsed-last_report>=15:
     print(json.dumps(dict(status='native_CPU_resource_qualification_running',wall_seconds=elapsed,peak_observed_RSS_bytes=peak)),flush=True);last_report=elapsed
    time.sleep(.5)
  finally:
   if child.poll() is None:
    child.terminate()
    try:child.wait(timeout=5)
    except subprocess.TimeoutExpired:child.kill();child.wait()
 receipt.update(status='completed' if status=='running' else status,exit_code=child.returncode,argv=argv,
  wall_seconds=time.monotonic()-start,peak_observed_RSS_bytes=peak,stdout=(out/'stdout.log').read_text(),stderr=(out/'stderr.log').read_text())
 result=base/'runs'/run_name/'RESOURCE_QUALIFICATION.json'
 if result.exists():receipt['qualification_result']=json.loads(result.read_text())
 if receipt['status']!='completed' or receipt['exit_code']!=0:receipt['uncompleted_terminal_policy']='all missing frozen arms retained as resource-deferred; no fitted subset comparison or implicit cap relaxation'
else:receipt['reason']='insufficient current host/cgroup headroom or CPU availability; native dataset/model execution not started'
arms=('native_GAT','native_Simple_HGN','native_SeHGNN');journal=base/'runs'/run_name/'ARM_TERMINALS.jsonl';observed={}
if journal.exists():
 for line in journal.read_text().splitlines():
  try:row=json.loads(line)
  except json.JSONDecodeError:continue
  if isinstance(row,dict) and row.get('arm') in arms:observed[row['arm']]=row
receipt['all_three_qualification_terminals']=[observed.get(arm,dict(arm=arm,seed=131,status='resource_deferred',
 attempted=(base/'runs'/run_name/arm).exists(),reason='qualification_child_or_resource_screen_not_completed',
 validation_or_test_scored=False,model_selection=False)) for arm in arms]
for row in binding['source_records']+[binding['freeze'],binding['paired_HGT_freeze']]:verify(row)
receipt['original_sources_and_freezes_preserved']=True
write(out/'REMOTE_RECEIPT.json',receipt);print(json.dumps(receipt,sort_keys=True),flush=True);sys.exit(receipt['exit_code'])
'''


def require(ok,message):
    if not ok:raise ValueError(message)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission',required=True,type=Path);parser.add_argument('--receipt',required=True,type=Path)
    args=parser.parse_args();require(not args.receipt.exists(),'Fresh local qualification transport receipt required')
    manifest_bytes=(HERE/'MANIFEST.json').read_bytes();manifest=json.loads(manifest_bytes);manifest_sha=hashlib.sha256(manifest_bytes).hexdigest()
    require(json.loads((HERE/'SEAL.json').read_text())['manifest_sha256']==manifest_sha,'Local qualification seal differs')
    for row in manifest['payload']:
        relative=Path(row['path']);require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe local manifest path')
        data=(HERE/relative).read_bytes();require(hashlib.sha256(data).hexdigest()==row['sha256'] and len(data)==row['bytes'],'Sealed qualification payload changed')
    release_bytes=args.admission.read_bytes();release=json.loads(release_bytes);binding=json.loads((HERE/'BINDINGS.json').read_text())
    require(release['qualification_authorized'] is True and release['execution_authorized'] is False
        and release['qualification_manifest_sha256']==manifest_sha and release['device']=='cpu'
        and release['mode']=='native_full_graph_resource_only','Root CPU resource-only qualification release required')
    files=[]
    paths=[HERE/r['path'] for r in manifest['payload']]+[HERE/'MANIFEST.json',HERE/'SEAL.json']
    # Immutable root metadata may not have been deployed with the synthetic packet.
    paths += [PHASE/Path(binding[key]['path']).relative_to(binding['canonical_phase']) for key in ('freeze','paired_HGT_freeze','CPU_original_receipt')]
    for path in dict.fromkeys(paths):
        data=path.read_bytes();files.append(dict(path=str(path.relative_to(PHASE)),data=base64.b64encode(data).decode(),
            sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
    ssh=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes',
        '-o','BatchMode=yes','-o','UpdateHostKeys=no','-o','StrictHostKeyChecking=yes',LOGIN]
    command=shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,REPO])
    request=dict(files=files,manifest_sha256=manifest_sha,release_data=base64.b64encode(release_bytes).decode(),
        release_sha256=hashlib.sha256(release_bytes).hexdigest())
    run=subprocess.Popen([*ssh,command],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    run.stdin.write(json.dumps(request));run.stdin.close();lines=[]
    for line in run.stdout:lines.append(line);print(line,end='',flush=True)
    stderr=run.stderr.read();code=run.wait()
    receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),destination=LOGIN,exit_code=code,stdout=''.join(lines),stderr=stderr,
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification_manifest_sha256=manifest_sha,root_release_sha256=request['release_sha256'])
    for line in reversed(lines):
        try:parsed=json.loads(line)
        except json.JSONDecodeError:continue
        if 'preflight' in parsed and 'exit_code' in parsed:receipt['remote_receipt']=parsed;break
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    with args.receipt.open('x') as stream:json.dump(receipt,stream,indent=2,sort_keys=True);stream.write('\n')
    if stderr:print(stderr,file=sys.stderr,end='')
    return code


if __name__=='__main__':raise SystemExit(main())
