"""Independent root execution of prepared native-challenger CPU fixtures."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
PACKET = PHASE / 'graph_heterogeneous_dblp_native_challengers_preparation_20261003_v2'
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
 path=phase/rel;assert path.resolve().is_relative_to(phase)
 data=base64.b64decode(row['data']);assert hashlib.sha256(data).hexdigest()==row['sha256']
 if path.exists():assert path.read_bytes()==data,str(rel)
 else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
packet=phase/'graph_heterogeneous_dblp_native_challengers_preparation_20261003_v2'
assert hashlib.sha256((packet/'MANIFEST.json').read_bytes()).hexdigest()=='5956ed69b1c9646aedda0ce67d950081b23ca6ca77796ac7f98c4b900544af43'
out=phase/'graph_heterogeneous_dblp_native_challengers_CPU_root_v1/run01';out.mkdir(parents=True,exist_ok=False)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',DGLBACKEND='pytorch',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(repo/'.gnnm_runtime/hgb_dgl_cpu_v1/site'),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
py=str(repo/'.venv/bin/python')
verify=subprocess.run([py,'-I','-S','-B',str(packet/'verify_source.py')],cwd=repo,capture_output=True,text=True,timeout=30)
start=time.monotonic();run=None
if verify.returncode==0:
 run=subprocess.run([py,'-B',str(packet/'cpu_fixtures.py')],cwd=packet,env=env,capture_output=True,text=True,timeout=60)
receipt=dict(exit_code=verify.returncode if run is None else run.returncode,wall_seconds=time.monotonic()-start,
 source_verification=dict(exit_code=verify.returncode,stdout=verify.stdout,stderr=verify.stderr),
 stdout='' if run is None else run.stdout,stderr='' if run is None else run.stderr,
 prepared_manifest_sha256=request['manifest_sha256'],uploaded_files=len(request['files']),
 scope=dict(remote_CPU_execution=True,synthetic_optimizer_autograd=True,real_dataset_labels_opened=False,GPU_model_execution=False,training_driver_main_called=False),
 original_source_hashes_preserved=all(hashlib.sha256((phase/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in request['files']))
(out/'ORIGINAL_REMOTE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt));sys.exit(receipt['exit_code'])
'''


def main():
    out = HERE / 'run01'
    out.mkdir(exist_ok=False)
    paths = {f for f in PACKET.rglob('*') if f.is_file() and '__pycache__' not in f.parts}
    provenance = json.loads((PACKET / 'PROVENANCE.json').read_text())
    paths.update(PHASE / r['path'] for r in provenance['inputs'])
    manifest = PHASE / 'graph_heterogeneous_private_modulation_source_20261003_v1/MANIFEST.sha256'
    paths.update(manifest.parent / line.split('  ', 1)[1] for line in manifest.read_text().splitlines())
    paths.update(f for f in (PHASE / 'graph_heterogeneous_dblp_native_challengers_preparation_20261003_v1').rglob('*')
                 if f.is_file() and '__pycache__' not in f.parts)
    rows = []
    for path in sorted(paths):
        assert path.resolve().is_relative_to(PHASE) and path.is_file(), path
        body = path.read_bytes()
        rows.append(dict(path=str(path.relative_to(PHASE)), sha256=hashlib.sha256(body).hexdigest(),
                         bytes=len(body), data=base64.b64encode(body).decode()))
    manifest_sha = hashlib.sha256((PACKET / 'MANIFEST.json').read_bytes()).hexdigest()
    assert manifest_sha == '5956ed69b1c9646aedda0ce67d950081b23ca6ca77796ac7f98c4b900544af43'
    (out / 'REMOTE_CODE.txt').write_text(REMOTE)
    (out / 'DEPLOYMENT_INVENTORY.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='data'} for r in rows], indent=2)+'\n')
    ssh = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
           '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
           '-o', 'StrictHostKeyChecking=yes', LOGIN]
    command = shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', REMOTE, REPO])
    result = subprocess.run([*ssh, command], input=json.dumps(dict(files=rows, manifest_sha256=manifest_sha)),
                            capture_output=True, text=True, timeout=110)
    record = dict(UTC=datetime.now(timezone.utc).isoformat(), destination=LOGIN, exit_code=result.returncode,
                  stdout=result.stdout, stderr=result.stderr,
                  helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if result.stdout:
        try:
            record['original_remote_receipt'] = json.loads(result.stdout)
        except json.JSONDecodeError:
            pass
    (out / 'RECEIPT.json').write_text(json.dumps(record, indent=2)+'\n')
    value = record.get('original_remote_receipt', record)
    print(json.dumps(value))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
