"""Stage sealed CPU scheduler; launch only with an exact root execution release."""
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
REPO='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
LOGIN='anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
REMOTE=r'''
import base64,hashlib,json,os,pathlib,subprocess,sys
repo=pathlib.Path(sys.argv[1]);request=json.load(sys.stdin)
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):
 digest=hashlib.sha256()
 with path.open('rb') as stream:
  for data in iter(lambda:stream.read(1<<20),b''):digest.update(data)
 return digest.hexdigest()
def verify(row):
 path=pathlib.Path(row['path']);require(sha(path)==row['sha256'],'Source/input hash changed: '+str(path))
 if 'bytes' in row:require(path.stat().st_size==row['bytes'],'Source/input length changed: '+str(path))
 return path
def write(path,value):
 with path.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')
require(subprocess.run(['git','rev-parse','--show-toplevel'],cwd=repo,capture_output=True,text=True,check=True).stdout.strip()==str(repo),'Wrong canonical repository')
head=subprocess.run(['git','rev-parse','HEAD'],cwd=repo,capture_output=True,text=True,check=True).stdout.strip()
require(subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac'],'Wrong authorized one-GPU account inventory')
phase=repo/'experiments_iclr/postsubmission_20260930';base=phase/'graph_heterogeneous_dblp_parallel_cpu_preparation_20261003_v1'
base.mkdir(exist_ok=True)
for row in request['files']:
 relative=pathlib.PurePosixPath(row['path']);require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe payload path')
 data=base64.b64decode(row['data'],validate=True);require(hashlib.sha256(data).hexdigest()==row['sha256'] and len(data)==row['bytes'],'Transport payload differs')
 target=base/relative
 if target.exists():require(target.read_bytes()==data,'Existing immutable packet bytes differ')
 else:
  target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as stream:stream.write(data)
manifest=json.loads((base/'MANIFEST.json').read_text());manifest_sha=sha(base/'MANIFEST.json')
require(manifest_sha==request['manifest_sha256'] and json.loads((base/'SEAL.json').read_text())['manifest_sha256']==manifest_sha,'Sealed scheduler manifest differs')
require({r['path'] for r in request['files']}=={r['path'] for r in manifest['payload']}|{'MANIFEST.json','SEAL.json'},'Exact sealed inventory required')
for row in manifest['payload']:verify(dict(row,path=str(base/row['path'])))
binding=json.loads((base/'BINDINGS.json').read_text())
for row in binding['source_records']+[binding['freeze'],binding['resource_qualification']]:verify(row)
receipt=dict(status='sealed_CPU_scheduler_staged',repo=str(repo),current_HEAD=head,packet=str(base),
 scheduler_manifest_sha256=manifest_sha,inventory_files=len(request['files']),GPU_computation=False,
 original_sources_preserved=True,study_training_started=False)
if not request['stage_only']:
 data=base64.b64decode(request['release_data'],validate=True);release=json.loads(data)
 require(hashlib.sha256(data).hexdigest()==request['release_sha256'] and release['execution_authorized'] is True,'Exact root execution release required')
 run_name=release['run_name'];require(pathlib.Path(run_name).name==run_name and run_name not in ('','.','..'),'Simple fresh released run name required')
 require(release['scheduler_manifest_sha256']==manifest_sha and release['device']=='cpu'
  and release['mode']=='parallel_CPU_five_seeds','CPU scheduler release differs')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
  OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
 admissions=base/'admissions';admissions.mkdir(exist_ok=True);release_path=admissions/(run_name+'.json')
 if release_path.exists():require(release_path.read_bytes()==data,'Previously saved root admission differs')
 else:
  with release_path.open('xb') as stream:stream.write(data)
 # Validate the actual sealed v2 guard and qualified geometry before detaching.
 guard_code='import importlib.util,sys,json; p=sys.argv[1]; s=importlib.util.spec_from_file_location("released_CPU_scheduler",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); b,f,d,h=m.packet_guard(); m.admission_guard(b,f,d,json.load(open(sys.argv[2])),h,sys.argv[3])'
 subprocess.run([str(repo/'.venv/bin/python'),'-B','-c',guard_code,str(base/'parallel_cpu.py'),str(release_path),run_name],cwd=repo,env=env,check=True)
 frozen=json.loads(verify(binding['freeze']).read_text())
 for row in [frozen['archive'],frozen['development_labels']]+[r['descriptor'] for r in frozen['splits']]:verify(row)
 require(not list((base/'runs').glob('*/STUDY_STARTED.json')),'Parallel study already started')
 require(not list((phase/'graph_heterogeneous_dblp_training_preparation_20261003_v2/runs').glob('*/STUDY_STARTED.json')),'Original serial study already started')
 transport=base/'transport'/run_name;transport.mkdir(parents=True,exist_ok=False)
 argv=[str(repo/'.venv/bin/python'),'-B',str(base/'parallel_cpu.py'),'--admission',str(release_path),'--run-name',run_name]
 with (transport/'controller.stdout.log').open('x') as stdout,(transport/'controller.stderr.log').open('x') as stderr:
  child=subprocess.Popen(argv,cwd=repo,env=env,stdout=stdout,stderr=stderr,start_new_session=True,stdin=subprocess.DEVNULL)
 receipt.update(status='released_CPU_controller_dispatched',controller_pid=child.pid,argv=argv,
  root_release_sha256=request['release_sha256'],run_name=run_name,transport=str(transport),
  terminal_receipt=str(base/'runs'/run_name/'PARALLEL_STUDY.json'),study_training_started='controller decides after resource preflight')
 write(transport/'DISPATCH_RECEIPT.json',receipt)
print(json.dumps(receipt,sort_keys=True),flush=True)
'''


def require(ok,message):
    if not ok:raise ValueError(message)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--stage-only',action='store_true');mode.add_argument('--admission',type=Path)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    require(not args.receipt.exists(),'Fresh local transport receipt required')
    manifest_bytes=(HERE/'MANIFEST.json').read_bytes();manifest=json.loads(manifest_bytes)
    manifest_sha=hashlib.sha256(manifest_bytes).hexdigest()
    require(json.loads((HERE/'SEAL.json').read_text())['manifest_sha256']==manifest_sha,'Local scheduler seal differs')
    files=[]
    for row in manifest['payload']:
        relative=Path(row['path']);require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe local manifest path')
        data=(HERE/relative).read_bytes();require(hashlib.sha256(data).hexdigest()==row['sha256'] and len(data)==row['bytes'],'Local payload changed')
    for name in [r['path'] for r in manifest['payload']]+['MANIFEST.json','SEAL.json']:
        data=(HERE/name).read_bytes();files.append(dict(path=name,data=base64.b64encode(data).decode(),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
    request=dict(files=files,manifest_sha256=manifest_sha,stage_only=args.stage_only)
    if args.admission:
        data=args.admission.read_bytes();release=json.loads(data)
        require(release['execution_authorized'] is True and release['scheduler_manifest_sha256']==manifest_sha
            and release['mode']=='parallel_CPU_five_seeds' and release['device']=='cpu','Exact root CPU execution release required')
        request.update(release_data=base64.b64encode(data).decode(),release_sha256=hashlib.sha256(data).hexdigest())
    ssh=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes',
        '-o','BatchMode=yes','-o','UpdateHostKeys=no','-o','StrictHostKeyChecking=yes',LOGIN]
    command=shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,REPO])
    result=subprocess.run([*ssh,command],input=json.dumps(request),capture_output=True,text=True)
    receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),destination=LOGIN,exit_code=result.returncode,
        stdout=result.stdout,stderr=result.stderr,helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scheduler_manifest_sha256=manifest_sha,stage_only=args.stage_only)
    if result.returncode==0:receipt['remote_receipt']=json.loads(result.stdout.strip().splitlines()[-1])
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    with args.receipt.open('x') as stream:json.dump(receipt,stream,indent=2,sort_keys=True);stream.write('\n')
    if result.stdout:print(result.stdout,end='')
    if result.stderr:print(result.stderr,file=sys.stderr,end='')
    return result.returncode


if __name__=='__main__':raise SystemExit(main())
