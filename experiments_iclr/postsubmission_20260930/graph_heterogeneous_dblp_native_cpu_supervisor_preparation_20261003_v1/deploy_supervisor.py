"""Deploy immutable native CPU supervisor and detach only under root release."""
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
def write(path,value):
 with path.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')
require(subprocess.run(['git','rev-parse','--show-toplevel'],cwd=repo,capture_output=True,text=True,check=True).stdout.strip()==str(repo),'Wrong canonical repository')
require(subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac'],'Wrong authorized one-GPU account inventory')
head=subprocess.run(['git','rev-parse','HEAD'],cwd=repo,capture_output=True,text=True,check=True).stdout.strip()
base=repo/'experiments_iclr/postsubmission_20260930/graph_heterogeneous_dblp_native_cpu_supervisor_preparation_20261003_v1';base.mkdir(exist_ok=True)
for row in request['files']:
 relative=pathlib.PurePosixPath(row['path']);require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe own payload path')
 target=base/relative;require(target.resolve().is_relative_to(base),'Own payload escapes packet')
 data=base64.b64decode(row['data'],validate=True);require(hashlib.sha256(data).hexdigest()==row['sha256'] and len(data)==row['bytes'],'Transport payload differs')
 if target.exists():require(target.read_bytes()==data,'Existing immutable supervisor bytes differ')
 else:
  target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as stream:stream.write(data)
require(hashlib.sha256((base/'MANIFEST.json').read_bytes()).hexdigest()==request['manifest_sha256'],'Supervisor manifest differs')
data=base64.b64decode(request['release_data'],validate=True);release=json.loads(data)
require(hashlib.sha256(data).hexdigest()==request['release_sha256'] and release['execution_authorized'] is True
 and release['supervisor_manifest_sha256']==request['manifest_sha256'],'Exact root execution release required')
run_name=release['run_name'];require(pathlib.Path(run_name).name==run_name and run_name not in ('','.','..'),'Fresh simple released run name')
admissions=base/'admissions';admissions.mkdir(exist_ok=True);release_path=admissions/(run_name+'.json')
if release_path.exists():require(release_path.read_bytes()==data,'Existing immutable root release differs')
else:
 with release_path.open('xb') as stream:stream.write(data)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
guard='import importlib.util,json,sys; s=importlib.util.spec_from_file_location("native_supervisor_guard",sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); b=json.load(open(sys.argv[2])); r=dict(path=sys.argv[3],sha256=sys.argv[4],bytes=__import__("pathlib").Path(sys.argv[3]).stat().st_size); m.preservation(b,sys.argv[5],r); m.admission(b,json.load(open(b["freeze"]["path"])),json.load(open(sys.argv[3])),sys.argv[5],sys.argv[6])'
subprocess.run([str(repo/'.venv/bin/python'),'-B','-c',guard,str(base/'cpu_supervisor.py'),str(base/'BINDINGS.json'),str(release_path),request['release_sha256'],request['manifest_sha256'],run_name],cwd=repo,env=env,check=True)
out=base/'transport'/run_name;out.mkdir(parents=True,exist_ok=False)
argv=[str(repo/'.venv/bin/python'),'-B',str(base/'cpu_supervisor.py'),'--admission',str(release_path),'--run-name',run_name]
with (out/'supervisor.stdout.log').open('x') as stdout,(out/'supervisor.stderr.log').open('x') as stderr:
 child=subprocess.Popen(argv,cwd=repo,env=env,stdout=stdout,stderr=stderr,start_new_session=True,stdin=subprocess.DEVNULL)
receipt=dict(status='root_released_native_CPU_supervisor_dispatched',supervisor_pid=child.pid,argv=argv,current_HEAD=head,
 supervisor_manifest_sha256=request['manifest_sha256'],root_release_sha256=request['release_sha256'],
 root_owns_native_output_validation=True,supervisor_receipt=str(base/'runs'/run_name/'SUPERVISOR_RECEIPT.json'),GPU_computation=False)
write(out/'DISPATCH_RECEIPT.json',receipt);print(json.dumps(receipt,sort_keys=True))
'''


def require(ok,message):
    if not ok:raise ValueError(message)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--admission',required=True,type=Path);parser.add_argument('--receipt',required=True,type=Path)
    args=parser.parse_args();require(not args.receipt.exists(),'Fresh local dispatch receipt required')
    manifest_bytes=(HERE/'MANIFEST.json').read_bytes();manifest=json.loads(manifest_bytes);manifest_sha=hashlib.sha256(manifest_bytes).hexdigest()
    require(json.loads((HERE/'SEAL.json').read_text())['manifest_sha256']==manifest_sha,'Local supervisor seal differs')
    files=[]
    for row in manifest['payload']:
        relative=Path(row['path']);require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe own manifest path')
        data=(HERE/relative).read_bytes();require(hashlib.sha256(data).hexdigest()==row['sha256'] and len(data)==row['bytes'],'Local own payload differs')
    for name in [r['path'] for r in manifest['payload']]+['MANIFEST.json','SEAL.json']:
        data=(HERE/name).read_bytes();files.append(dict(path=name,data=base64.b64encode(data).decode(),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
    release_bytes=args.admission.read_bytes();release=json.loads(release_bytes)
    require(release['execution_authorized'] is True and release['device']=='cpu' and release['mode']=='native_v2_serial_CPU_supervised'
        and release['supervisor_manifest_sha256']==manifest_sha,'Exact root native CPU supervisor execution release required')
    request=dict(files=files,manifest_sha256=manifest_sha,release_data=base64.b64encode(release_bytes).decode(),release_sha256=hashlib.sha256(release_bytes).hexdigest())
    ssh=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes',
        '-o','UpdateHostKeys=no','-o','StrictHostKeyChecking=yes',LOGIN]
    command=shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,REPO])
    result=subprocess.run([*ssh,command],input=json.dumps(request),capture_output=True,text=True)
    receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),destination=LOGIN,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),supervisor_manifest_sha256=manifest_sha)
    if result.returncode==0:receipt['dispatch']=json.loads(result.stdout.strip().splitlines()[-1])
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    with args.receipt.open('x') as stream:json.dump(receipt,stream,indent=2,sort_keys=True);stream.write('\n')
    if result.stdout:print(result.stdout,end='')
    if result.stderr:print(result.stderr,file=sys.stderr,end='')
    return result.returncode


if __name__=='__main__':raise SystemExit(main())
