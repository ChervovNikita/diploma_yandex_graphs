from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,socket,subprocess,sys,time,traceback
REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
OUTPUT=PHASE/'pencil_citeseer_cpu_import_qualification_20261005_v1'
DEPS=PHASE/'pencil_citeseer_missing_ancillary_resolver_execution_20261005_v1'
OVERLAY=PHASE/'pencil_one_gpu_dependency_overlay_20261005_v1'
PYTHON=PHASE/'native_ncn_runtime_20261005_v1/.venv/bin/python'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert Path.cwd().resolve()==REPO and socket.gethostname()=='anogena-2-0'
assert sha(DEPS/'INSTALL_EXECUTION_RECEIPT.json')=='ae7147efcd438c1f710fa8a51cb7233a92321dd88c1de016f6266fbd88001335'
install=json.loads((DEPS/'INSTALL_EXECUTION_RECEIPT.json').read_text())
assert install['GPU_UUID']=='GPU-44039938-fd82-41d2-fefd-de71514e2fac' and install['hostname']=='anogena-2-0'
assert install['original61providers_and_metadata_unchanged'] is True
assert not OUTPUT.exists()
OUTPUT.mkdir()
started=time.monotonic()
receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),host=socket.gethostname(),cwd=str(Path.cwd()),bound_verified_singleton_GPU_UUID=install['GPU_UUID'],new_GPU_queries_or_operators=False,staged=[],no_source_overwrites=True,packages_installed=False,model_constructors=False,dataset_constructors=False,TEST_access=False,automatic_retry=False)
try:
 payload=json.loads(sys.stdin.read())
 assert hashlib.sha256(payload['source_bindings'].encode()).hexdigest()=='fc8bade93efc3ca540ddf1c6f7d392cda749a217f4a326f51f681bad1863432c'
 bindings=json.loads(payload['source_bindings'])
 assert len(bindings['files'])==23 and bindings['native_file_count']==20
 rows={r['phase_relative']:r for r in bindings['files']}
 for item in payload['files']:
  relative=Path(item['phase_relative']);assert not relative.is_absolute() and '..' not in relative.parts
  target=PHASE/relative;assert target.resolve().is_relative_to(PHASE)
  data=base64.b64decode(item['base64']);expected=rows[str(relative)]
  assert len(data)==expected['bytes'] and hashlib.sha256(data).hexdigest()==expected['sha256']
  if target.exists():
   assert target.is_file() and sha(target)==expected['sha256']
   action='REUSED_EXACT_EXISTING_BYTES'
  else:
   target.parent.mkdir(parents=True,exist_ok=True)
   with target.open('xb') as f:f.write(data)
   action='STAGED_MISSING_EXACT_BYTES'
  receipt['staged'].append(dict(path=str(target),sha256=expected['sha256'],action=action))
 assert len(receipt['staged'])==23
 (OUTPUT/'SOURCE_BINDINGS.json').write_text(payload['source_bindings'])
 worker=payload['worker_source']
 assert hashlib.sha256(worker.encode()).hexdigest()==payload['worker_sha256']
 (OUTPUT/'check_cpu_imports.py').write_text(worker)
 (OUTPUT/'STAGING_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
 env=dict(os.environ)
 env.update(PYTHONPATH=str(OVERLAY)+':'+str(PHASE/'native_ncn_dependency_overlay_20261005_v1')+':'+str(REPO/'.venv/lib/python3.11/site-packages'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',OUTDATED_IGNORE='1')
 command=[str(PYTHON),str(OUTPUT/'check_cpu_imports.py')]
 child=subprocess.run(command,cwd=REPO,env=env,capture_output=True,text=True)
 receipt.update(command=command,worker_sha256=payload['worker_sha256'],child_environment={k:env[k] for k in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE','CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','WANDB_MODE','OUTDATED_IGNORE')},child_exit_code=child.returncode,stdout=child.stdout,stderr=child.stderr,inclusive_seconds=time.monotonic()-started)
 if child.returncode:raise RuntimeError('CPU import worker failed before its final result')
 result=json.loads(child.stdout)
 receipt['status']=result['status']
 (OUTPUT/'EXECUTION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
except BaseException as error:
 receipt.update(status='FAIL_STAGING_OR_CPU_IMPORT_PRESERVED',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc(),inclusive_seconds=time.monotonic()-started)
 (OUTPUT/'EXECUTION_FAILURE.json').write_text(json.dumps(receipt,indent=2)+'\n')
small={}
for name in ('STAGING_RECEIPT.json','RESULT.json','FAILURE.json','EXECUTION_RECEIPT.json','EXECUTION_FAILURE.json'):
 path=OUTPUT/name
 if path.exists():
  row=dict(bytes=path.stat().st_size,sha256=sha(path))
  if path.stat().st_size<=1048576:row['utf8']=path.read_text()
  small[name]=row
print(json.dumps(dict(receipt=receipt,small_receipts=small),sort_keys=True))
