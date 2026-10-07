"""Stage reviewed P/G source and start its bounded CPU work on the authorized allocation."""
import base64
import hashlib
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = HERE.parent
NAME = 'allocation_controls_molhiv_CPU_representative_source_20261007_v1'
R = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
REMOTE_P = R + '/experiments_iclr/postsubmission_20260930'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'ConnectTimeout=15',
       '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt', '-p', '2222', LOGIN]

REMOTE = r'''
import base64,hashlib,json,os,socket,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930';os.chdir(R)
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(sys.stdin.read());D=P/payload['source_name']
assert not D.exists();D.mkdir()
for row in payload['files']:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 data=base64.b64decode(row['data']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
 path=D/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
assert hashlib.sha256((D/'MANIFEST.json').read_bytes()).hexdigest()==payload['manifest_sha256']
for row in json.loads((D/'MANIFEST.json').read_text())['files']:
 path=D/row['path'];assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
A=P/'allocation_controls_molhiv_CPU_representative_activation_root_20261007_v1'
A.mkdir(exist_ok=False)
py=P/'native_ncn_runtime_20261005_v1/.venv/bin/python'
out=P/'allocation_controls_molhiv_CPU_representative_execution_root_20261007_v1/run01'
assert py.is_file() and not out.exists()
cmd=[str(py),'-B',str(D/'watch_cpu_once.py'),'--python',str(py),
 '--dependency-path',str(P/'native_ncn_dependency_overlay_20261005_v1'),
 '--dependency-path',str(R/'.venv/lib/python3.11/site-packages')]
for key,value in [('public-root','portable_internal_be_public_interface_20261007_v2'),
 ('adapter-root','public_internal_be_private_steering_adapter_20261007_v1'),
 ('interface-root','public_internal_be_private_steering_complete_interface_20261007_v1'),
 ('allocation-root','public_internal_be_allocation_controls_20261007_v1'),
 ('train','internal_BE_portable_molhiv_cpu_check_20261007_v1/roles/train.npz'),
 ('valid','internal_BE_portable_molhiv_cpu_check_20261007_v1/roles/valid.npz'),
 ('geometry','internal_BE_portable_molhiv_cpu_check_20261007_v1/TRAIN_BATCH_GEOMETRY.json')]:
 path=P/value;assert path.exists();cmd.extend(['--'+key,str(path)])
cmd.extend(['--output',str(out)])
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
with (A/'PARENT.log').open('x') as log:
 child=subprocess.Popen(cmd,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 ticks=int(Path('/proc/'+str(child.pid)+'/stat').read_text().rsplit(') ',1)[1].split()[19])
value={'UTC':datetime.now(timezone.utc).isoformat(),'parent_PID':child.pid,'parent_start_ticks':ticks,
 'source_manifest_sha256':payload['manifest_sha256'],'argv':cmd,'output':str(out),
 'CPU_only':True,'scientific_fit':False,'quality_values_closed':True,'new_77_contact':False}
(A/'LAUNCH.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value))
'''


def main():
    source = P / NAME
    manifest_path = source / 'MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    for row in manifest['files']:
        path = source / row['path']
        assert path.stat().st_size == row['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
    payload = dict(source_name=NAME, manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(), files=[])
    for path in sorted(source.iterdir()):
        assert path.is_file() and not path.is_symlink() and path.stat().st_size < 500_000
        data = path.read_bytes()
        payload['files'].append(dict(path=path.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                                    data=base64.b64encode(data).decode()))
    assert not (HERE / 'LAUNCH_TRANSPORT.json').exists()
    remote = shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', REMOTE])
    result = subprocess.run([*SSH, remote], input=json.dumps(payload), text=True, capture_output=True, timeout=40)
    receipt = dict(UTC=datetime.now(timezone.utc).isoformat(), returncode=result.returncode,
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   source_manifest_sha256=payload['manifest_sha256'], stdout=result.stdout, stderr=result.stderr)
    (HERE / 'LAUNCH_TRANSPORT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if result.returncode == 0:
        launch = json.loads(result.stdout)
        (HERE / 'LAUNCH.json').write_text(json.dumps(launch, indent=2) + '\n')
        print(json.dumps({k: launch[k] for k in ('UTC', 'parent_PID', 'parent_start_ticks', 'CPU_only', 'scientific_fit')}))
    else:
        print(json.dumps(dict(returncode=result.returncode, stderr=result.stderr)))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
