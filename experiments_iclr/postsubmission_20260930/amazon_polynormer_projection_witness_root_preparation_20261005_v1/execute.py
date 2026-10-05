"""Run one authenticated numerical witness diagnostic on the authorized allocation."""
from pathlib import Path
from datetime import datetime, timezone
import base64
import hashlib
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SOURCE = 'amazon_polynormer_logits_graph_moment_projection_failure_diagnostic_preparation_20261005_v1'
OUTPUT = 'amazon_polynormer_projection_failure_witness_execution_root_20261005_v1'
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
REMOTE = r'''
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,socket,subprocess,sys,time
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
packet=json.load(sys.stdin)
def path(rel):
 p=Path(rel);assert not p.is_absolute() and '..' not in p.parts
 p=phase/p;assert p.resolve().is_relative_to(phase)
 assert not any(q.is_symlink() for q in (p,*p.parents) if q.is_relative_to(phase))
 return p
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
for row in packet['files']:
 p=path(row['path']);b=base64.b64decode(row['data'])
 assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
 if p.exists():assert p.read_bytes()==b
 else:
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(b)
root=path(packet['root']);root.mkdir(exist_ok=False)
out=path(packet['output']);assert not out.exists()
write(root/'RELEASE.json',packet['release'])
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',BLIS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
argv=packet['release']['argv'];start=time.perf_counter()
with (root/'STDOUT.txt').open('x') as stdout,(root/'STDERR.txt').open('x') as stderr:
 r=subprocess.run(argv,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,timeout=120)
write(root/'TERMINAL.json',dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=r.returncode,wall_seconds=time.perf_counter()-start,automatic_retry=False))
rows=[]
for p in [root/'RELEASE.json',root/'TERMINAL.json',root/'STDOUT.txt',root/'STDERR.txt',out/'BOUND_CONTEXT.json',out/'WITNESS.json',out/'DIAGNOSTIC_RESULT.json']:
 if p.is_file():
  assert p.stat().st_size<2_000_000
  b=p.read_bytes();rows.append(dict(path=str(p.relative_to(phase)),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),data=base64.b64encode(b).decode()))
print(json.dumps(dict(exit_code=r.returncode,files=rows)))
'''

def main():
    rows=[]
    for name in ['extract_projection_witness.py','MANIFEST.json','SEAL.json','README.md','SOURCE_BINDINGS.json','STATIC_CHECK.json']:
        path=PHASE/SOURCE/name
        data=path.read_bytes()
        rows.append(dict(path=str(path.relative_to(PHASE)),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),data=base64.b64encode(data).decode()))
    by_name={Path(row['path']).name:row for row in rows}
    assert by_name['MANIFEST.json']['sha256']=='0587f3cbe9fea72176b79a19c1f0096675c0498d5c18cc8845b6302a9897e291'
    assert by_name['SEAL.json']['sha256']=='9ae7420da3270721397996774ff8e1f95c28675f36d995d50a1119a0ec0a2f3b'
    remote_phase=REPO+'/experiments_iclr/postsubmission_20260930'
    argv=[REPO+'/.venv/bin/python','-B',remote_phase+'/'+SOURCE+'/extract_projection_witness.py',
          '--source',remote_phase+'/amazon_polynormer_logits_graph_moment_source_preparation_20261005_v2',
          '--manifest-sha256','af15ac11349ac7d9dbc8608b4362d461195409ca6348fdfc67e7ac7b4a273b21',
          '--seal-sha256','2ed8fef0d08e420e140ddcddfa2929e00849797715d16e1e0be626e9cb2ef900',
          '--phase',remote_phase,'--closure-freeze','amazon_polynormer_paired_family_execution_root_20261003_v3/v6_closure_v1/FREEZE.json',
          '--closure-sha256','37145c431daaf74dbd622f287285ec4947f7b4dbdb6b88a19c723376d1ca29fc','--output',remote_phase+'/'+OUTPUT]
    release=dict(UTC=datetime.now(timezone.utc).isoformat(),root_authorized=True,purpose='Reproduce only failed projection arithmetic',
                 numerical_witness_only=True,scored_outcomes=0,head_fits=0,optimizer_updates=0,backbone_forwards=0,TEST_access=False,
                 original_source_unchanged=True,automatic_retry=False,argv=argv,
                 source_descriptors=[{k:r[k] for k in ('path','sha256','bytes')} for r in rows])
    with (HERE/'RELEASE.json').open('x') as f:json.dump(release,f,indent=2);f.write('\n')
    packet=dict(files=rows,root=HERE.name,output=OUTPUT,release=release)
    command=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','IdentitiesOnly=yes',
             '-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=20',
             'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','cd '+REPO+' && exec /usr/bin/python3 -I -S -B -c '+shlex.quote(REMOTE)]
    r=subprocess.run(command,input=json.dumps(packet),capture_output=True,text=True,timeout=155)
    with (HERE/'TRANSPORT.json').open('x') as f:json.dump(dict(exit_code=r.returncode,stderr=r.stderr,remote_source_sha256=hashlib.sha256(REMOTE.encode()).hexdigest()),f,indent=2);f.write('\n')
    assert r.returncode==0,r.stderr
    received=json.loads(r.stdout)
    observation=HERE/'observation';observation.mkdir()
    for row in received['files']:
        rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
        data=base64.b64decode(row['data']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
        path=observation/rel;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(data)
    result=json.loads((observation/OUTPUT/'DIAGNOSTIC_RESULT.json').read_text())
    print(json.dumps(dict(exit_code=received['exit_code'],status=result['status'],node=result.get('native_node_id'),feasible_faces=result.get('old_face_feasible_count'),cost=result['cost'],local_observation=str(observation))))

if __name__=='__main__':main()
