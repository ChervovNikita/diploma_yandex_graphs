from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,socket,subprocess,time
REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
PACKET=PHASE/'pencil_citeseer_missing_ancillary_install_plan_20261005_v1'
OUTPUT=PHASE/'pencil_citeseer_missing_ancillary_resolver_execution_20261005_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert Path.cwd().resolve()==REPO and socket.gethostname()=='anogena-2-0'
devices=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=30)
assert devices.stdout.split()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
assert sha(PACKET/'MANIFEST.json')=='d0a647774f59a6b1206b75338894220be7257b02b294d5dcc871bdaa7112457a'
for row in json.loads((PACKET/'MANIFEST.json').read_text())['files']:assert sha(PACKET/row['path'])==row['sha256']
assert sha(OUTPUT/'MISSING_WHEELS.lock')=='09694097759d457115ad983e38ff1cc4ae637aef5581e1f85fd3edda8d2bf922'
assert sha(OUTPUT/'BEFORE_METADATA.json')=='fc4698d328558fd48a5ddb5f3d4bd3a2e894b05dfce39c19832e0b6cac1d88d6'
plan=json.loads((PACKET/'PLAN.json').read_text())
overlay=Path(plan['fresh_dependency_overlay'])
assert not overlay.exists() and overlay.is_relative_to(PHASE)
assert not (OUTPUT/'INSTALL_EXECUTION_RECEIPT.json').exists() and not (OUTPUT/'INSTALL_FAILURE.json').exists()
resolved=json.loads((OUTPUT/'RESOLVED_MISSING_WHEELS.json').read_text())
assert resolved['package_count']==22 and resolved['core_or_existing_replacements'] is False
original=json.loads((OUTPUT/'BEFORE_METADATA.json').read_text())['installed']
metadata_source="""from importlib import metadata
import hashlib,json,re
canon=lambda x:re.sub(r'[-_.]+','-',x).lower()
names={canon(d.metadata['Name']) for d in metadata.distributions() if d.metadata.get('Name')}
rows={}
for name in sorted(names):
 d=metadata.distribution(name)
 rows[name]=dict(version=d.version,root=str(d.locate_file('')),METADATA_sha256=hashlib.sha256((d.read_text('METADATA') or '').encode()).hexdigest())
print(json.dumps(rows,sort_keys=True))
"""
oldenv=dict(os.environ);oldenv.update(plan['environment_before'])
newenv=dict(os.environ);newenv.update(plan['environment_after'])
started=time.monotonic()
receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),hostname=socket.gethostname(),cwd=str(Path.cwd()),GPU_UUID=devices.stdout.strip(),packages_installed=False,model_or_numerical_imports=False,TEST_access=False,lock_sha256=sha(OUTPUT/'MISSING_WHEELS.lock'),command=plan['sequential_commands'][3]['args'],PYTHONPATH_install=oldenv['PYTHONPATH'],automatic_retry=False)
try:
 oldread=subprocess.run([plan['interpreter'],'-c',metadata_source],cwd=REPO,env=oldenv,capture_output=True,text=True,check=True)
 before=json.loads(oldread.stdout)
 assert len(before)==len(original)==61
 assert all(before[k]['version']==v['version'] and before[k]['root']==v['root'] for k,v in original.items())
 (OUTPUT/'PREINSTALL_SELECTED_METADATA.json').write_text(json.dumps(before,indent=2)+'\n')
 receipt['before_selected_metadata_sha256']=sha(OUTPUT/'PREINSTALL_SELECTED_METADATA.json')
 (OUTPUT/'INSTALL_START_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
 result=subprocess.run(receipt['command'],cwd=REPO,env=oldenv,capture_output=True,text=True)
 receipt.update(install_exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,inclusive_seconds=time.monotonic()-started)
 if result.returncode:raise RuntimeError('Exact ordinary pip install failed; no retry')
 receipt['packages_installed']=True
 installed_report=json.loads((OUTPUT/'PIP_INSTALL_REPORT.json').read_text())
 actual={x['metadata']['name'].lower().replace('_','-'):x['metadata']['version'] for x in installed_report['install']}
 expect={x['name']:x['version'] for x in resolved['packages']}
 assert actual==expect and len(actual)==22 and not(set(actual)&set(original))
 inventory=[]
 for path in sorted(overlay.rglob('*')):
  if path.is_file():
   assert not path.is_symlink() and path.resolve().is_relative_to(overlay)
   inventory.append(dict(path=str(path.relative_to(overlay)),bytes=path.stat().st_size,sha256=sha(path),mode=path.stat().st_mode))
 assert inventory
 (OUTPUT/'INSTALLED_FILE_INVENTORY.json').write_text(json.dumps(dict(root=str(overlay),files=inventory),indent=2)+'\n')
 newread=subprocess.run([plan['interpreter'],'-c',metadata_source],cwd=REPO,env=newenv,capture_output=True,text=True,check=True)
 after=json.loads(newread.stdout)
 assert all(after.get(k)==v for k,v in before.items())
 assert {k:after[k]['version'] for k in expect}==expect
 assert all(Path(after[k]['root']).resolve()==overlay for k in expect)
 (OUTPUT/'POSTINSTALL_SELECTED_METADATA.json').write_text(json.dumps(after,indent=2)+'\n')
 receipt.update(status='COMPLETE_EXACT_22_ANCILLARY_OVERLAY_ONLY',installed_package_count=22,installed_file_count=len(inventory),installed_file_bytes=sum(x['bytes'] for x in inventory),original61providers_and_metadata_unchanged=True,core_versions={k:after[k]['version'] for k in ('torch','numpy','torch-geometric','torch-sparse','torch-scatter')},overlay=str(overlay),inventory_sha256=sha(OUTPUT/'INSTALLED_FILE_INVENTORY.json'),postinstall_selected_metadata_sha256=sha(OUTPUT/'POSTINSTALL_SELECTED_METADATA.json'),PIP_INSTALL_REPORT_sha256=sha(OUTPUT/'PIP_INSTALL_REPORT.json'),inclusive_seconds=time.monotonic()-started,import_or_operator_readiness=False)
 (OUTPUT/'INSTALL_EXECUTION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
except BaseException as error:
 receipt.update(status='FAILED_INSTALL_OR_AUDIT_PRESERVED_NO_RETRY',exception=type(error).__name__+': '+str(error),inclusive_seconds=time.monotonic()-started)
 (OUTPUT/'INSTALL_FAILURE.json').write_text(json.dumps(receipt,indent=2)+'\n')
small={}
for name in ('PREINSTALL_SELECTED_METADATA.json','PIP_INSTALL_REPORT.json','INSTALLED_FILE_INVENTORY.json','POSTINSTALL_SELECTED_METADATA.json','INSTALL_EXECUTION_RECEIPT.json','INSTALL_FAILURE.json'):
 path=OUTPUT/name
 if path.exists():
  row=dict(bytes=path.stat().st_size,sha256=sha(path))
  if path.stat().st_size<=1048576:row['utf8']=path.read_text()
  else:row['not_retrieved']='larger_than_small_receipt_bound'
  small[name]=row
print(json.dumps(dict(receipt=receipt,small_receipts=small),sort_keys=True))
