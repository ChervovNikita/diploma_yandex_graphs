"""One bounded, discarded FP32 qualification on the authorized singleton GPU.

Requires root-admitted exact JOB.json and closed TRAIN-only input roles. No fit
or score is launched. Stops only the newly owned child on a declared bound.
"""
import ast
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

HERE=Path(__file__).resolve().parent
REMOTE=r'''
import base64,hashlib,json,os,signal,socket,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
payload=json.load(sys.stdin)
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=20).strip().splitlines()
assert len(gpu)==1 and gpu[0].split(',')[0].strip()=='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert int(gpu[0].split(',')[1].strip())*1024**2>=12*1024**3
root=phase/'shared_private_transfer_row0_fp32_execution_root_20261005_v1'
assert not root.exists() and str(root)==payload['remote_execution_root']
allowed={payload['training_source_name'],'endpoint_episode_geometry_preparation_20261005_v1','shared_backbone_private_transfer_source_independent_review_20261005_v1','shared_backbone_private_transfer_source_independent_review_20261005_v2','shared_backbone_private_transfer_source_independent_review_20261005_v3','shared_private_transfer_row0_companion_source_review_20261005_v1',root.name}
decoded=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for row in payload['files']:
 rel=Path(row['path']); assert not rel.is_absolute() and '..' not in rel.parts and rel.parts[0] in allowed
 path=phase/rel; assert path.resolve().is_relative_to(phase.resolve())
 content=base64.b64decode(row['base64'],validate=True)
 assert len(content)==row['bytes'] and hashlib.sha256(content).hexdigest()==row['sha256']
 if path.exists():assert path.is_file() and not path.is_symlink() and sha(path)==row['sha256']
 decoded.append((path,content))
root.mkdir()
for path,content in decoded:
 if not path.exists():
  path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as f:f.write(content)
source=phase/payload['training_source_name']
job=json.loads((root/'JOB.json').read_text())
assert job['source_review_approved'] is True and job['purpose']=='fp32_discarded_training_step_qualification'
assert job['fits_authorized'] is False and job['VALID_values_access'] is False and job['TEST_access'] is False
assert job['external_hard_bound_confirmed'] is True and job['soft_seconds']==600
assert job['source_manifest_sha256']==sha(source/'SOURCE_MANIFEST.json') and job['program_sha256']==sha(source/'qualify_training_step.py')
assert job['output_directory']==str(root/'result') and job['retry'] is False
for r in json.loads((source/'SOURCE_MANIFEST.json').read_text())['files']:assert sha(source/r['path'])==r['sha256']
stage=dict(UTC=datetime.now(timezone.utc).isoformat(),host=socket.gethostname(),GPU_metadata=gpu,job_sha256=sha(root/'JOB.json'),source_manifest_sha256=sha(source/'SOURCE_MANIFEST.json'),files=[{k:r[k] for k in ('path','bytes','sha256')} for r in payload['files']],fits=0,VALID_TEST_values_access=False)
(root/'STAGING_RECEIPT.json').write_text(json.dumps(stage,indent=2)+'\n')
interpreter=phase/'native_ncn_runtime_20261005_v1/.venv/bin/python'
env=dict(os.environ)
env.update(CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONPATH=str(phase/'native_ncn_dependency_overlay_20261005_v1')+':'+str(repo/'.venv/lib/python3.11/site-packages'))
env.pop('PYTHONHOME',None)
argv=[str(interpreter),str(source/'qualify_training_step.py'),'--job',str(root/'JOB.json'),'--output',str(root/'result')]
started=time.monotonic();reason=None;signals=[];max_rss=0;max_cuda=0
def identity(pid):
 p=Path('/proc')/str(pid)
 try:
  raw=(p/'stat').read_text();f=raw[raw.rfind(')')+2:].split()
  return dict(PID=pid,start_ticks=int(f[19]),state=f[0],argv=[s.decode() for s in (p/'cmdline').read_bytes().split(bytes([0])) if s],pgid=int(f[2]),sid=int(f[3]))
 except FileNotFoundError:return None
with (root/'child.stdout.log').open('xb') as stdout,(root/'child.stderr.log').open('xb') as stderr:
 process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,cwd=repo,env=env,start_new_session=True)
 owner=identity(process.pid);assert owner is not None and owner['argv']==argv and owner['pgid']==owner['sid']==process.pid
 (root/'CHILD_STARTED.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),identity=owner,argv=argv,hard_seconds=720,RSS_cap_bytes=64*1024**3,own_CUDA_cap_bytes=10*1024**3),indent=2)+'\n')
 while process.poll() is None:
  live=identity(process.pid)
  if live is not None:
   assert live['start_ticks']==owner['start_ticks'] and live['pgid']==owner['pgid'] and live['sid']==owner['sid']
   if live['state']!='Z' and live['argv']:assert live['argv']==argv
  status=Path('/proc')/str(process.pid)/'status'
  try:
   for line in status.read_text().splitlines():
    if line.startswith('VmRSS:'):max_rss=max(max_rss,int(line.split()[1])*1024)
  except FileNotFoundError:pass
  q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=20)
  assert q.returncode==0
  for line in q.stdout.splitlines():
   parts=[s.strip() for s in line.split(',')]
   if len(parts)==2 and parts[0]==str(process.pid) and parts[1].isdigit():max_cuda=max(max_cuda,int(parts[1])*1024**2)
  if time.monotonic()-started>720:reason='hard_wall_bound'
  elif max_rss>64*1024**3:reason='own_RSS_cap'
  elif max_cuda>10*1024**3:reason='own_CUDA_cap'
  elif sum((root/name).stat().st_size for name in ('child.stdout.log','child.stderr.log'))>8*1024**2:reason='own_log_cap'
  if reason:
   live=identity(process.pid)
   if live is not None and live['state']!='Z':
    assert live['start_ticks']==owner['start_ticks'] and live['argv']==argv and live['pgid']==live['sid']==process.pid
    os.killpg(process.pid,signal.SIGKILL);signals.append(dict(PID=process.pid,signal='SIGKILL',reason=reason))
   process.wait();break
  time.sleep(1)
 process.wait()
diagnostics={}
for name in ('START.json','RESULT.json','FAILURE.json'):
 p=root/'result'/name
 if p.exists():
  assert p.stat().st_size<2*1024**2
  diagnostics[name]=dict(utf8=p.read_text(),sha256=sha(p),bytes=p.stat().st_size)
receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=process.returncode,reason=reason,signals_sent=signals,inclusive_seconds=time.monotonic()-started,max_owned_RSS_bytes=max_rss,max_owned_CUDA_bytes=max_cuda,attempts=1,numerical_launches=1,fits=0,VALID_TEST_values_access=False,source_bytes_unchanged=all(sha(p)==hashlib.sha256(c).hexdigest() for p,c in decoded),source_manifest_sha256=sha(source/'SOURCE_MANIFEST.json'),job_sha256=sha(root/'JOB.json'),child_identity=owner)
(root/'EXECUTION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
extra={name:dict(utf8=(root/name).read_text(),sha256=sha(root/name)) for name in ('child.stdout.log','child.stderr.log')}
print(json.dumps(dict(staging=stage,execution=receipt,diagnostics=diagnostics,logs=extra)))
'''

def main():
    ast.parse(REMOTE)
    command=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=20','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
      'cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs && exec python3 -I -c '+shlex.quote(REMOTE)]
    with (HERE/'STAGING_PAYLOAD.json').open('rb') as payload:
        result=subprocess.run(command,stdin=payload,capture_output=True,text=True,timeout=780)
    with (HERE/'TRANSPORT.json').open('x') as f:
        json.dump(dict(exit_code=result.returncode,stdout_sha256=hashlib.sha256(result.stdout.encode()).hexdigest(),stdout_bytes=len(result.stdout.encode()),stderr=result.stderr,remote_program_sha256=hashlib.sha256(REMOTE.encode()).hexdigest()),f,indent=2);f.write('\n')
    if result.returncode:raise RuntimeError(result.stderr)
    received=json.loads(result.stdout)
    for key,name in [('staging','STAGING_RECEIPT.json'),('execution','EXECUTION_RECEIPT.json')]:
        with (HERE/name).open('x') as f:json.dump(received[key],f,indent=2);f.write('\n')
    for name,row in {**received['diagnostics'],**received['logs']}.items():
        assert hashlib.sha256(row['utf8'].encode()).hexdigest()==row['sha256']
        with (HERE/name).open('x') as f:f.write(row['utf8'])
    print(json.dumps(dict(execution=received['execution'],status='PASS' if 'RESULT.json' in received['diagnostics'] else 'FAILURE',diagnostic_files=list(received['diagnostics']))))

if __name__=='__main__':main()
