"""Deploy immutable HGT sources and run synthetic CPU family/checkpoint fixtures."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
IMPL = PHASE / 'graph_heterogeneous_dblp_training_preparation_20261003_v1'
SOURCE = PHASE / 'graph_heterogeneous_private_modulation_source_20261003_v1'
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
REMOTE = r'''
import base64,hashlib,json,os,pathlib,subprocess,sys,time
repo=pathlib.Path(sys.argv[1]);request=json.load(sys.stdin)
assert subprocess.run(['git','rev-parse','--show-toplevel'],cwd=repo,capture_output=True,text=True,check=True).stdout.strip()==str(repo)
assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
phase=repo/'experiments_iclr/postsubmission_20260930'
for row in request['files']:
 rel=pathlib.PurePosixPath(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 path=phase/rel;body=base64.b64decode(row['data']);assert hashlib.sha256(body).hexdigest()==row['sha256']
 if path.exists():assert path.read_bytes()==body,str(rel)
 else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(body)
impl=phase/'graph_heterogeneous_dblp_training_preparation_20261003_v1'
assert hashlib.sha256((impl/'MANIFEST.json').read_bytes()).hexdigest()=='6bb754f75ff6512900b49e85b91fde11a88ae2a63f68969448d7f9c3d1d7d02c'
out=phase/'graph_heterogeneous_dblp_CPU_root_v1/run01';out.mkdir(parents=True,exist_ok=False)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',DGLBACKEND='pytorch',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(repo/'.gnnm_runtime/hgb_dgl_cpu_v1/site'))
py=str(repo/'.venv/bin/python')
verify=subprocess.run([py,'-I','-S','-B',str(impl/'verify_source.py')],cwd=repo,capture_output=True,text=True,timeout=20)
assert verify.returncode==0,verify.stderr
started=time.monotonic()
run=subprocess.run([py,'-B',str(impl/'cpu_training_fixtures.py')],cwd=repo,env=env,capture_output=True,text=True,timeout=50)
receipt=dict(exit_code=run.returncode,wall_seconds=time.monotonic()-started,stdout=run.stdout,stderr=run.stderr,source_verification=json.loads(verify.stdout),uploaded_files=len(request['files']),manifest_sha256=request['manifest_sha256'],scope=dict(remote_CPU_execution=True,synthetic_labels_autograd=True,original_dataset_or_label_payloads=False,GPU_model_execution=False,optimizer_or_training_driver_execution=False))
(out/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt));sys.exit(run.returncode)
'''


def main():
    attempt = HERE / 'run01'
    attempt.mkdir(exist_ok=False)
    provenance = json.loads((IMPL/'PROVENANCE.json').read_text())
    paths = set(IMPL.iterdir())
    paths |= {SOURCE/line.split('  ',1)[1] for line in (SOURCE/'MANIFEST.sha256').read_text().splitlines()}
    paths |= {SOURCE/'MANIFEST.sha256', SOURCE/'SEAL.json'}
    paths |= {PHASE/row['path'] for row in provenance['inputs']}
    rows = []
    for path in sorted(paths):
        assert path.is_file() and path.is_relative_to(PHASE), path
        body = path.read_bytes()
        rows.append(dict(path=str(path.relative_to(PHASE)),sha256=hashlib.sha256(body).hexdigest(),data=base64.b64encode(body).decode()))
    manifest_sha = hashlib.sha256((IMPL/'MANIFEST.json').read_bytes()).hexdigest()
    assert manifest_sha=='6bb754f75ff6512900b49e85b91fde11a88ae2a63f68969448d7f9c3d1d7d02c'
    (attempt/'REMOTE_CODE.txt').write_text(REMOTE)
    (attempt/'DEPLOYMENT_INVENTORY.json').write_text(json.dumps([{k:v for k,v in row.items() if k!='data'} for row in rows],indent=2)+'\n')
    ssh = ['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','UpdateHostKeys=no','-o','StrictHostKeyChecking=yes',LOGIN]
    command = shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,REPO])
    run = subprocess.run([*ssh,command],input=json.dumps(dict(files=rows,manifest_sha256=manifest_sha)),capture_output=True,text=True,timeout=65)
    receipt = dict(UTC=datetime.now(timezone.utc).isoformat(),destination=LOGIN,exit_code=run.returncode,stdout=run.stdout,stderr=run.stderr,helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if run.stdout:
        try: receipt['original_remote_receipt']=json.loads(run.stdout)
        except json.JSONDecodeError: pass
    (attempt/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt.get('original_remote_receipt',receipt)))
    raise SystemExit(run.returncode)


if __name__=='__main__':
    main()
