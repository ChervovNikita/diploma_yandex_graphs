"""One root-authorized full-input readiness action; no scientific acquisition."""
import hashlib, importlib.util, json, os, socket, subprocess, sys, time
from pathlib import Path
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
H=Path(__file__).resolve().parent
def receipt(p):
 p=Path(p).resolve();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(r):
 assert receipt(r['path'])==r,r['path']
 return Path(r['path'])
def identity(pid):
 p=Path('/proc')/str(pid)/'stat'
 if not p.exists():return None
 a=p.read_text().rsplit(')',1)[1].split()
 return dict(pid=pid,start_ticks=int(a[19]),pgid=int(a[2]))
assert socket.gethostname()=='anogena-2-0' and Path.cwd()==R
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==[UUID]
raw=(H/'SPEC.json').read_bytes();spec=json.loads(raw)
decision=json.loads(checked(spec['decision']).read_text())
assert decision['action']=='Physics_original_boundary_readiness' and decision['full_scientific_acquisition_admitted'] is False
assert decision['limits']==dict(wall_seconds=600,cuda_bytes=48*1024**3,rss_bytes=64*1024**3)
source=Path(decision['source_directory'])
deadline=time.monotonic()+5
while not (H/'OWNER_START.json').exists():
 if time.monotonic()>=deadline:raise RuntimeError('Actual direct owner start receipt absent')
 time.sleep(.01)
def root_guard(actual,*,action,source_paths):
 assert actual==spec and action=='Physics_original_boundary_readiness'
 own=json.loads((H/'OWNER_START.json').read_text())
 assert own['child']==identity(os.getpid()) and own['owner']==identity(os.getppid())
 assert own['spec_sha256']==hashlib.sha256(raw).hexdigest() and own['external_wall_seconds']==660
 assert receipt(H/'owner.py')==decision['owner_source']
 hashes={p:receipt(p)['sha256'] for p in source_paths}
 assert hashes==decision['source_sha256']
 fixed=json.loads((source/'SOURCE_BINDINGS.json').read_text())
 assert actual['roles']==fixed['Physics_roles']
 for name,r in actual['source_receipts'].items():
  checked(r);f=fixed['runtime_sources'][name]
  assert r['bytes']==f['bytes'] and r['sha256']==f['sha256']
 checked(actual['roles']['8101']['train'])
 metadata=decision['existing_Physics_metadata_binding']
 checked(metadata)
 assert decision['runtime_executable']==str(R/'.venv/bin/python') and Path(sys.executable)==R/'.venv/bin/python'
 return dict(admitted=True,action=action,source_sha256=hashes,
  native_runtime_and_full_inputs_verified=True,finite_external_owner_bound=True,
  external_owner_wall_seconds=660,limits=actual['limits'],decision_receipt=actual['decision'],
  CAL_TEST_access=False,actual_owner_start=receipt(H/'OWNER_START.json'),
  checked_TRAIN_receipt=actual['roles']['8101']['train'],
  existing_Physics_metadata_binding=metadata,actual_runtime_executable=sys.executable)
module_spec=importlib.util.spec_from_file_location('root_original_Physics_readiness',source/'acquire.py')
module=importlib.util.module_from_spec(module_spec);sys.modules[module_spec.name]=module;module_spec.loader.exec_module(module)
result=module.full_input_readiness(spec,root_guard)
print(json.dumps(dict(status='READINESS_RETURNED',result_path=str(H/'readiness_v1/FULL_INPUT_READINESS_COMPLETE.json'),result=result),allow_nan=False),flush=True)
