"""Root-released one-shot host port of the unchanged bounded FP32 supervisor.

Takes root-reviewed STAGING_PAYLOAD JSON on stdin; disabled payloads fail.
No fit or metric work; source and numerical recipe remain unchanged.
"""
import base64,hashlib,importlib.util,json,os,signal,socket,subprocess,sys,time
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
root=phase/'shared_private_transfer_gpu77_fp32_execution_root_20261005_v2'
assert not root.exists() and str(root)==payload['remote_execution_root']
allowed={payload['training_source_name'],'endpoint_episode_geometry_preparation_20261005_v1','shared_backbone_private_transfer_source_independent_review_20261005_v1','shared_backbone_private_transfer_source_independent_review_20261005_v2','shared_backbone_private_transfer_source_independent_review_20261005_v3','shared_private_transfer_gpu77_qualification_preparation_20261005_v2','shared_private_transfer_gpu77_environment_execution_20261005_v1',root.name}
decoded=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
prep=Path(__file__).resolve().parent
assert sha(prep/'MANIFEST.json')==payload['wrapper_manifest_sha256']
assert sha(__file__)==payload['supervisor_program_sha256']
for row in json.loads((prep/'MANIFEST.json').read_text())['files']:
 assert sha(prep/row['path'])==row['sha256'] and (prep/row['path']).stat().st_size==row['bytes']
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
limits={'external_hard_seconds_per_cell':720,'owned_tree_RSS_cap_bytes':64*1024**3,
        'owned_tree_GPU_memory_cap_bytes':10*1024**3,'combined_child_log_cap_bytes':8*1024**2,
        'telemetry_timeout_seconds':20,'poll_interval_seconds':1}
helper_path=Path(__file__).resolve().parent/'ownership_helpers.py'
spec=importlib.util.spec_from_file_location('qualified_v3_ownership_helpers',helper_path)
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);h.GPU=selected
started=time.monotonic();reason=None;signals=[];max_rss=0;max_cuda=0;max_logs=0
owner=None;raw_owner=None;observed_cwd=None;signal_refusal=None;terminal_wait_observed=False
incomplete_exit_samples=0;last_incomplete_exit=None
with (root/'child.stdout.log').open('xb') as stdout,(root/'child.stderr.log').open('xb') as stderr:
 process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,cwd=repo,env=env,start_new_session=True)
 try:
  raw_owner=h.identity(process.pid)
  observed_cwd=(Path('/proc')/str(process.pid)/'cwd').resolve(strict=True)
  if raw_owner is None or raw_owner['argv']!=argv or raw_owner['pgid']!=process.pid or raw_owner['sid']!=process.pid or observed_cwd!=repo:
   reason='Initial fresh owned child/session/argv/cwd could not be verified; no signal authorized'
  else:owner=raw_owner
 except Exception as error:
  reason='Initial identity observation: '+type(error).__name__+': '+str(error)
 (root/'CHILD_STARTED.json').write_text(json.dumps(dict(UTC=h.now(),identity=owner,raw_identity_observation=raw_owner,identity_admitted_for_signals=owner is not None,observed_cwd=str(observed_cwd) if observed_cwd is not None else None,argv=argv,hard_seconds=720,RSS_cap_bytes=64*1024**3,own_CUDA_cap_bytes=10*1024**3),indent=2)+'\n')
 try:
  while process.poll() is None:
   if reason is not None:break
   remaining=720-(time.monotonic()-started)
   if remaining<=0:reason='hard_wall_bound'
   else:
    rows=h.owned_tree(owner)
    incomplete=[row for row in rows if not row.get('observation_complete',True)]
    if incomplete:
     incomplete_exit_samples+=1;last_incomplete_exit=dict(UTC=h.now(),rows=incomplete)
     if process.poll() is not None:break
    rss,cuda=h.resources(rows,limits,remaining)
    max_rss=max(max_rss,rss);max_cuda=max(max_cuda,cuda)
    max_logs=max(max_logs,sum((root/name).stat().st_size for name in ('child.stdout.log','child.stderr.log')))
    if max_rss>64*1024**3:reason='own_RSS_cap'
    elif max_cuda>10*1024**3:reason='own_CUDA_cap'
    elif max_logs>8*1024**2:reason='own_log_cap'
    elif time.monotonic()-started>=720:reason='hard_wall_bound'
   if reason:
    try:
     sent=h.kill_owned(process,owner,reason)
     if sent:signals.append(sent)
    except h.IncompleteExitObservation as error:signal_refusal=type(error).__name__+': '+str(error)
    break
   time.sleep(min(1,max(.01,remaining)))
 except (Exception,KeyboardInterrupt) as error:
  reason='required_bounds_telemetry_failure: '+type(error).__name__+': '+str(error)
  try:
   sent=h.kill_owned(process,owner,reason) if owner is not None else None
   if sent:signals.append(sent)
  except Exception as error:signal_refusal=type(error).__name__+': '+str(error)
 finally:
  reason,signal_refusal,terminal_wait_observed=h.observe_terminal(process,owner,started,limits,signals,reason,signal_refusal)
max_logs=max(max_logs,sum((root/name).stat().st_size for name in ('child.stdout.log','child.stderr.log')))
if reason is None and max_logs>8*1024**2:reason='own_log_cap_at_exit'
terminal_owner=None;identity_closure_observed=False
terminal_gpu_rows=None;closure_error=None;owned_no_cuda=None
try:
 terminal_owner=h.identity(process.pid);identity_closure_observed=True
 terminal_gpu_rows=h.query(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],20)
 owned_no_cuda=not any(len(parts)>=2 and parts[1].strip()==str(process.pid) for parts in (row.split(',') for row in terminal_gpu_rows))
except Exception as error:
 closure_error=type(error).__name__+': '+str(error)
if reason is None and (not terminal_wait_observed or not identity_closure_observed or terminal_owner is not None or owned_no_cuda is not True):
 reason='physical_terminal_closure_unconfirmed'
physical=dict(UTC=h.now(),owned_PID_absent=identity_closure_observed and terminal_owner is None,owned_PID_no_CUDA_rows=owned_no_cuda,physical_gpu_uuid=selected,terminal_wait_observed=terminal_wait_observed,exit_code_authority='subprocess.Popen.wait/poll' if terminal_wait_observed else None,child_PID=process.pid,child_start_ticks=raw_owner['start_ticks'] if raw_owner is not None else None,compute_rows=terminal_gpu_rows,closure_error=closure_error)
(root/'PHYSICAL_TERMINAL.json').write_text(json.dumps(physical,indent=2)+'\n')
receipt=dict(UTC=h.now(),terminal_wait_observed=terminal_wait_observed,exit_code_authority='subprocess.Popen.wait/poll' if terminal_wait_observed else None,physical_gpu_uuid=selected,exit_code=process.returncode if terminal_wait_observed else None,reason=reason,signals_sent=signals,signal_refusal=signal_refusal,incomplete_exit_observation_samples=incomplete_exit_samples,last_incomplete_exit_observation=last_incomplete_exit,inclusive_seconds=time.monotonic()-started,max_owned_RSS_bytes=max_rss,max_owned_CUDA_bytes=max_cuda,max_owned_log_bytes=max_logs,attempts=1,numerical_launches=1,fits=0,VALID_TEST_values_access=False,source_bytes_unchanged=all(sha(p)==hashlib.sha256(c).hexdigest() for p,c in decoded),source_manifest_sha256=sha(source/'SOURCE_MANIFEST.json'),job_sha256=sha(root/'JOB.json'),child_identity=owner,raw_identity_observation=raw_owner,identity_admitted_for_signals=owner is not None,observed_cwd=str(observed_cwd) if observed_cwd is not None else None,owned_helper_sha256=sha(helper_path),physical_terminal_sha256=sha(root/'PHYSICAL_TERMINAL.json'))
(root/'EXECUTION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
diagnostics={}
for name in ('START.json','RESULT.json','FAILURE.json','PARTIAL_RESULT.json'):
 p=root/'result'/name
 if p.exists():
  row=dict(sha256=sha(p),bytes=p.stat().st_size,path=str(p))
  if p.stat().st_size<2*1024**2:row['utf8']=p.read_text()
  else:row['retained_on_disk_not_inlined']=True
  diagnostics[name]=row
extra={name:dict(utf8=(root/name).read_text(),sha256=sha(root/name)) for name in ('child.stdout.log','child.stderr.log')}
print(json.dumps(dict(staging=stage,execution=receipt,physical_terminal=physical,diagnostics=diagnostics,logs=extra)))
