"""Run the seven sealed BUDDY CPU checks without data or CUDA; fits remain gated."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'buddy_shared_cache_execution_v3'
MANIFEST_SHA = 'bc88d12c41516591d678f1e677a7cc4173a6ebaf6f1f5767b66350b728b9bd68'
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
REMOTE = '''
from datetime import datetime,timezone
import base64,hashlib,json,os,pathlib,subprocess,sys,time
repo=pathlib.Path(sys.argv[1]);login=sys.argv[2];uuid=sys.argv[3]
if repo.resolve()!=repo: raise RuntimeError('Actual fixed project root required')
gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True)
if gpu.stdout.strip().splitlines()!=[uuid]: raise RuntimeError('Authorized GPU route differs')
git=subprocess.run(['git','rev-parse','--show-toplevel'],cwd=repo,capture_output=True,text=True,check=True)
if pathlib.Path(git.stdout.strip()).resolve()!=repo: raise RuntimeError('Project Git root differs')
phase=repo/'experiments_iclr/postsubmission_20260930'
source=phase/'buddy_shared_cache_execution_v3'
payload=json.load(sys.stdin)
for row in payload['files']:
 rel=pathlib.PurePosixPath(row['path'])
 if rel.is_absolute() or '..' in rel.parts: raise RuntimeError('Invalid source path')
 path=source.joinpath(*rel.parts)
 if path.is_symlink() or not path.resolve().is_relative_to(source): raise RuntimeError('Escaping source')
 data=base64.b64decode(row['data'])
 if hashlib.sha256(data).hexdigest()!=row['sha256']: raise RuntimeError('Source byte identity differs')
 if path.exists():
  if path.read_bytes()!=data: raise RuntimeError('Existing source differs')
 else:
  path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as stream: stream.write(data)
run=phase/'buddy_cpu_numerical_qualification_v3/root_run_v1'
if not run.resolve().is_relative_to(phase): raise RuntimeError('Unconfined CPU receipt path')
run.mkdir(parents=True,exist_ok=False)
for name in ('tmp','cache'): (run/name).mkdir()
extra=phase/'buddy_cpu_runtime_preparation_v1/root_setup_v2/site'
setup=json.loads((extra.parent/'SETUP_RECEIPT.json').read_text())
if setup.get('status')!='DEPENDENCIES_INSTALLED_UNQUALIFIED' or setup.get('exit_code')!=0:
 raise RuntimeError('Dependency setup has not succeeded')
for name,digest in setup['installed_record_sha256'].items():
 if hashlib.sha256((extra/name).read_bytes()).hexdigest()!=digest: raise RuntimeError('Installed dependency record differs')
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(extra),PYTHONDONTWRITEBYTECODE='1',
 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 TMPDIR=str(run/'tmp'),XDG_CACHE_HOME=str(run/'cache'),TORCH_HOME=str(run/'cache/torch'))
certificate=run/'CPU_QUALIFICATION.json'; started=time.monotonic()
result=subprocess.run([str(repo/'.venv/bin/python'),'-B',str(source/'test_cpu.py'),'--output',str(certificate)],
 cwd=source,env=env,capture_output=True,text=True,timeout=180)
(run/'stdout.txt').write_text(result.stdout);(run/'stderr.txt').write_text(result.stderr)
value=json.loads(certificate.read_text()) if certificate.exists() else None
success=result.returncode==0 and value is not None and value['status']=='synthetic_cpu_pass' and value['test_count']==7
record=dict(UTC=datetime.now(timezone.utc).isoformat(),schema='root-buddy-seven-CPU-checks-v3',
 ssh_destination=login,GPU_uuid=uuid,GPU_compute=False,dataset_access=False,
 source_manifest_sha256=payload['source_manifest_sha256'],dependency_setup_sha256=hashlib.sha256((extra.parent/'SETUP_RECEIPT.json').read_bytes()).hexdigest(),
 git_head=subprocess.run(['git','rev-parse','HEAD'],cwd=repo,capture_output=True,text=True,check=True).stdout.strip(),
 exit_code=result.returncode,status='SEVEN_CPU_CHECKS_PASSED' if success else 'NUMERICAL_QUALIFICATION_FAILED',
 seconds=time.monotonic()-started,certificate=value,stdout=result.stdout,stderr=result.stderr,
 independent_source_review_required_before_cache_or_fitting=True)
(run/'CPU_RUN_RECEIPT.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record,sort_keys=True))
raise SystemExit(0 if success else 1)
'''


def main():
    manifest_path = SOURCE / 'SOURCE_MANIFEST.json'
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != MANIFEST_SHA:
        raise RuntimeError('Sealed source manifest differs')
    manifest = json.loads(manifest_path.read_text())
    rows = []
    for row in manifest['files'] + [dict(path='SOURCE_MANIFEST.json'), dict(path='SEAL.json')]:
        path = SOURCE / row['path']
        if path.is_symlink() or not path.resolve().is_relative_to(SOURCE):
            raise RuntimeError('Unconfined source')
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if 'sha256' in row and digest != row['sha256']:
            raise RuntimeError('Manifest source byte differs')
        rows.append(dict(path=row['path'], sha256=digest, data=base64.b64encode(data).decode()))
    receipt = HERE / 'ROOT_LAUNCH.json'
    if receipt.exists():
        raise RuntimeError('Single-use qualification already attempted')
    ssh = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
           '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
           '-o', 'StrictHostKeyChecking=yes', LOGIN]
    command = shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', REMOTE, REPO, LOGIN, GPU])
    result = subprocess.run([*ssh, command], input=json.dumps(dict(files=rows, source_manifest_sha256=MANIFEST_SHA)),
                            capture_output=True, text=True, timeout=240)
    launch = dict(UTC=datetime.now(timezone.utc).isoformat(), ssh_destination=LOGIN,
                  source_wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_manifest_sha256=MANIFEST_SHA, exit_code=result.returncode,
                  stdout=result.stdout, stderr=result.stderr, dataset_or_GPU_execution=False)
    with receipt.open('x') as stream:
        json.dump(launch, stream, indent=2)
        stream.write('\n')
    try:
        remote = json.loads(result.stdout)
        (HERE / 'REMOTE_CPU_RUN_RECEIPT.json').write_text(json.dumps(remote, indent=2) + '\n')
        print(json.dumps({k: remote[k] for k in ('status', 'seconds', 'exit_code', 'stderr')}, sort_keys=True))
    except ValueError:
        print(json.dumps(launch, sort_keys=True))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
