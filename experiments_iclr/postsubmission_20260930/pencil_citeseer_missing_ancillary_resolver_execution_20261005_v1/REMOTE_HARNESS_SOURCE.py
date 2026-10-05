from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,socket,subprocess,sys,time
REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
PACKET=PHASE/'pencil_citeseer_missing_ancillary_install_plan_20261005_v1'
assert Path.cwd().resolve()==REPO and socket.gethostname()=='anogena-2-0'
devices=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=30)
assert devices.stdout.split()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
assert sha(PACKET/'MANIFEST.json')=='d0a647774f59a6b1206b75338894220be7257b02b294d5dcc871bdaa7112457a'
for row in json.loads((PACKET/'MANIFEST.json').read_text())['files']:
 assert sha(PACKET/row['path'])==row['sha256']
plan=json.loads((PACKET/'PLAN.json').read_text())
output=Path(plan['fresh_resolver_directory'])
assert not output.exists() and output.is_relative_to(PHASE)
env=dict(os.environ);env.update(plan['environment_before'])
started=time.monotonic()
receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),hostname=socket.gethostname(),cwd=str(Path.cwd()),verified_GPU_UUID=devices.stdout.strip(),plan_sha256=sha(PACKET/'PLAN.json'),source_manifest_sha256=sha(PACKET/'MANIFEST.json'),commands=[],packages_installed=False,numerical_modules_imported=False,model_imports=False,TEST_access=False)
for index,stage in enumerate(plan['sequential_commands'][:3]):
 cmd=stage['args'];t=time.monotonic()
 child=subprocess.run(cmd,cwd=REPO,env=env,capture_output=True,text=True)
 row=dict(stage=stage['stage'],command=cmd,PYTHONPATH=env['PYTHONPATH'],exit_code=child.returncode,stdout=child.stdout,stderr=child.stderr,elapsed_seconds=time.monotonic()-t)
 receipt['commands'].append(row)
 if output.exists():
  (output/f'STAGE_{index+1:02d}_RECEIPT.json').write_text(json.dumps(row,indent=2)+'\n')
 if child.returncode:
  receipt['status']='FAILED_PRESERVED_NO_RETRY'
  break
else:
 receipt['status']='COMPLETE_RESOLVER_AND_LOCK_ONLY_NO_INSTALL'
receipt['inclusive_seconds']=time.monotonic()-started
if output.exists():
 (output/'REMOTE_EXECUTION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
small={}
for name in ('BEFORE_METADATA.json','RESOLVER_CONSTRAINTS.txt','PIP_DRY_RUN_REPORT.json','MISSING_WHEELS.lock','RESOLVED_MISSING_WHEELS.json','REMOTE_EXECUTION_RECEIPT.json'):
 f=output/name
 if f.exists():
  row=dict(bytes=f.stat().st_size,sha256=sha(f))
  if f.stat().st_size<=1048576:row['utf8']=f.read_text()
  else:row['not_retrieved']='larger_than_small_receipt_bound'
  small[name]=row
print(json.dumps(dict(receipt=receipt,small_receipts=small),sort_keys=True))
