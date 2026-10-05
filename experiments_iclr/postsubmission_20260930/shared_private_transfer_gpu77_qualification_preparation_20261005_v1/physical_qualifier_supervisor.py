"""Root-released one-shot host port of the unchanged bounded FP32 supervisor.

Takes root-reviewed STAGING_PAYLOAD JSON on stdin; disabled payloads fail.
No fit or metric work; source and numerical recipe remain unchanged.
"""
import base64,hashlib,json,os,signal,socket,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
payload=json.load(sys.stdin)
assert payload.get('qualification_execution_enabled') is True
assert payload.get('fits_authorized') is False and payload.get('retry') is False
repo=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd().resolve()==repo and socket.gethostname()=='peptide'
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=20).strip().splitlines()
inventory=['GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998','GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced']
assert [row.split(',')[0].strip() for row in gpu]==inventory
selected=payload['physical_gpu_uuid'];assert selected in inventory
assert int(gpu[inventory.index(selected)].split(',')[1].strip())*1024**2>=12*1024**3
root=phase/'shared_private_transfer_gpu77_fp32_execution_root_20261005_v1'
assert not root.exists() and str(root)==payload['remote_execution_root']
allowed={payload['training_source_name'],'endpoint_episode_geometry_preparation_20261005_v1','shared_backbone_private_transfer_source_independent_review_20261005_v1','shared_backbone_private_transfer_source_independent_review_20261005_v2','shared_backbone_private_transfer_source_independent_review_20261005_v3','shared_private_transfer_gpu77_qualification_preparation_20261005_v1','shared_private_transfer_gpu77_environment_execution_20261005_v1',root.name}
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
assert job['physical_gpu_uuid']==selected and job['expected_hostname']=='peptide'
assert payload['training_source_name']=='shared_backbone_private_transfer_training_source_gpu77_20261005_v1'
assert job['fits_authorized'] is False and job['VALID_values_access'] is False and job['TEST_access'] is False
assert job['external_hard_bound_confirmed'] is True and job['soft_seconds']==600
assert job['source_manifest_sha256']==sha(source/'SOURCE_MANIFEST.json') and job['program_sha256']==sha(source/'qualify_training_step.py')
assert job['output_directory']==str(root/'result') and job['retry'] is False
for r in json.loads((source/'SOURCE_MANIFEST.json').read_text())['files']:assert sha(source/r['path'])==r['sha256']
stage=dict(UTC=datetime.now(timezone.utc).isoformat(),host=socket.gethostname(),GPU_metadata=gpu,job_sha256=sha(root/'JOB.json'),source_manifest_sha256=sha(source/'SOURCE_MANIFEST.json'),files=[{k:r[k] for k in ('path','bytes','sha256')} for r in payload['files']],fits=0,VALID_TEST_values_access=False)
(root/'STAGING_RECEIPT.json').write_text(json.dumps(stage,indent=2)+'\n')
interpreter=repo/'.gnnm_runtime/private_transfer_cp311_cu118_20261005_v1/bin/python'
env=dict(os.environ)
env.update(CUDA_VISIBLE_DEVICES=selected,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONPATH='')
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
  if live is not None:assert live['start_ticks']==owner['start_ticks'] and live['argv']==argv
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
terminal_owner=identity(process.pid)
assert terminal_owner is None
terminal_gpu=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=20)
assert terminal_gpu.returncode==0
assert not any(len(parts)>=2 and parts[1].strip()==str(process.pid) for parts in (row.split(',') for row in terminal_gpu.stdout.splitlines()))
(root/'PHYSICAL_TERMINAL.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),owned_PID_absent=True,owned_PID_no_CUDA_rows=True,physical_gpu_uuid=selected,terminal_wait_observed=True,exit_code_authority='subprocess.Popen.wait/poll',child_PID=process.pid,child_start_ticks=owner['start_ticks'],compute_rows=terminal_gpu.stdout),indent=2)+'\n')
diagnostics={}
for name in ('START.json','RESULT.json','FAILURE.json','PARTIAL_RESULT.json'):
 p=root/'result'/name
 if p.exists():
  assert p.stat().st_size<2*1024**2
  diagnostics[name]=dict(utf8=p.read_text(),sha256=sha(p),bytes=p.stat().st_size)
receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),terminal_wait_observed=True,exit_code_authority='subprocess.Popen.wait/poll',physical_gpu_uuid=selected,exit_code=process.returncode,reason=reason,signals_sent=signals,inclusive_seconds=time.monotonic()-started,max_owned_RSS_bytes=max_rss,max_owned_CUDA_bytes=max_cuda,attempts=1,numerical_launches=1,fits=0,VALID_TEST_values_access=False,source_bytes_unchanged=all(sha(p)==hashlib.sha256(c).hexdigest() for p,c in decoded),source_manifest_sha256=sha(source/'SOURCE_MANIFEST.json'),job_sha256=sha(root/'JOB.json'),child_identity=owner)
(root/'EXECUTION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
extra={name:dict(utf8=(root/name).read_text(),sha256=sha(root/name)) for name in ('child.stdout.log','child.stderr.log')}
print(json.dumps(dict(staging=stage,execution=receipt,diagnostics=diagnostics,logs=extra)))
