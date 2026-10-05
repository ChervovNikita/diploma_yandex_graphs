"""Exactly released metadata freeze; no accuracy child starts in this command."""
from pathlib import Path
import base64,hashlib,importlib.util,json,zlib
P=Path('/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930')
HERE=P/'shared_private_transfer_gpu77_block_launch_execution_root_20261005_v1';HERE.mkdir()
PREP=P/'shared_private_transfer_gpu77_block_launch_preparation_20261005_v1'
RELEASE=P/'shared_private_transfer_gpu77_block_root_release_20261005_v1'
spec=importlib.util.spec_from_file_location('blocklaunch77_transport',P/'shared_private_transfer_gpu77_environment_execution_20261005_v1/remote_transport.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t);t.HERE=HERE
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):
 with (HERE/name).open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
expected={'ROOT_REVIEW.md':'f2efc428ba7fbdbff17d4fd96e30729c047b8301e61aff2221ac8347c2378b52','ROOT_RELEASE_B1.json':'46e13b58fff191541737bc347b1e69446857365bc9133d30c4e6bcfabaf91c98','ROOT_RELEASE_B2.json':'22eb56abec6ce8fe74db2d2cb16ce65ba42e953da693c76598be3320a8c71c72'}
for name,digest in expected.items():assert sha(RELEASE/name)==digest
assert sha(PREP/'SOURCE_MANIFEST.json')=='028f6f5d07e95f7edc58a61d3f6a7f39d1685403d46ab6bba431bb6072cf26c0'
metadata=json.loads((PREP/'METADATA_STAGE_PLAN.json').read_text())
paths=[RELEASE/name for name in expected]+[P/row['path'] for row in metadata['files']]
rows=[dict(path=str(path.relative_to(P)),bytes=path.stat().st_size,sha256=sha(path),base64=base64.b64encode(path.read_bytes()).decode()) for path in paths]
save('ROOT_TRANSFER_INVENTORY.json',[{k:v for k,v in r.items() if k!='base64'} for r in rows])
encoded=base64.b64encode(zlib.compress(json.dumps(rows).encode(),9)).decode()
code=f'''
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,importlib.util,json,os,socket,subprocess,zlib
repo=Path({str(t.REPO)!r});phase=Path({str(t.REMOTE_PHASE)!r});receiptroot=phase/{HERE.name!r};prep=phase/{PREP.name!r};release_root=phase/{RELEASE.name!r}
assert Path.cwd()==repo and socket.gethostname()=='peptide' and not receiptroot.exists()
receiptroot.mkdir()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write(chr(10))
try:
 rows=json.loads(zlib.decompress(base64.b64decode({encoded!r},validate=True)));decoded=[]
 for row in rows:
  rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
  target=phase/rel;assert target.resolve().is_relative_to(phase)
  for parent in [target,*target.parents]:
   if parent==phase.parent:break
   assert not parent.is_symlink()
  raw=base64.b64decode(row['base64'],validate=True);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
  if target.exists():assert target.is_file() and target.read_bytes()==raw
  decoded.append((target,raw))
 for target,raw in decoded:
  if not target.exists():
   target.parent.mkdir(parents=True,exist_ok=True)
   with target.open('xb') as f:f.write(raw)
 spec=importlib.util.spec_from_file_location('block77_common',prep/'pilot_common.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
 c.verify_packet('028f6f5d07e95f7edc58a61d3f6a7f39d1685403d46ab6bba431bb6072cf26c0');c.verify_training()
 blocks={{}};artifacts={{}}
 for block in ('b1','b2'):
  c.bind_block([block]);c.physical_host()
  release_path=release_root/('ROOT_RELEASE_'+block.upper()+'.json');release=c.read(release_path)
  assert sha(release_path)=={expected!r}['ROOT_RELEASE_'+block.upper()+'.json']
  assert release['root_provider_block_approved'] is True and release['execution_blocks']==[block]
  assert release['environment_overrides']['CUDA_VISIBLE_DEVICES']==c.GPU_UUID
  execution=phase/release['execution_directory_relative'];assert not execution.exists()
  env=c.check_environment(release['environment_overrides'])
  argv=[release['python_executable'],'-B',str(prep/'freeze_queue.py'),'--release',str(release_path)]
  frozen=subprocess.run(argv,cwd=repo,env=env,capture_output=True,text=True,timeout=120)
  freezer=dict(UTC=datetime.now(timezone.utc).isoformat(),argv=argv,exit_code=frozen.returncode,stdout=frozen.stdout,stderr=frozen.stderr,fits_started=0,attempts=1,retry=False)
  save(receiptroot/(block+'.FREEZE_RECEIPT.json'),freezer)
  assert frozen.returncode==0,frozen.stderr
  q=c.read(execution/'QUEUE.json');plan=c.read(execution/'COHORT_PLAN.json');adm=c.read(execution/'PROVIDER_ADMISSION.json');reg=c.read(execution/'DONOR_REGISTRATION.json')
  expected_order=[r for r in plan['execution_order'] if r.startswith(block+'_')]
  assert q['execution_blocks']==[block] and len(q['entries'])==10 and [r['cell_id'] for r in q['entries']]==expected_order
  assert sha(execution/'COHORT_PLAN.json')==c.CANONICAL_PLAN_SHA and (execution/'EXTERNAL_ANCHORS.json').read_bytes()==(prep/'EXTERNAL_ANCHORS.json').read_bytes()
  assert q['root_release_sha256']==sha(execution/'ROOT_RELEASE.json')==sha(release_path)
  assert reg['queue_sha256']==sha(execution/'QUEUE.json') and reg['provider_admission']['sha256']==sha(execution/'PROVIDER_ADMISSION.json')
  assert reg['registered_before_provider_first_fit'] is True and adm['admitted_before_provider_first_fit'] is True and adm['approved'] is True and adm['GPU_UUID']==c.GPU_UUID
  assert adm['provider_source_manifest']['sha256']==c.SOURCE_SHA and adm['python_executable']==release['python_executable']
  for entry in q['entries']:
   path=phase/entry['job_relative'];job=c.read(path);cell=next(r for r in plan['cells'] if r['cell_id']==entry['cell_id'])
   assert sha(path)==entry['job_sha256'] and all(job[k]==cell[k] for k in ('cell_id','cell','paired_seed_block','seed','factor_seed','arm','rule','geometry','outer_size','inner_size','schedule'))
   assert job['fits_authorized'] is True and job['VALID_values_access'] is True and job['TEST_access'] is False and job['retry'] is False
   assert job['source_manifest_sha256']==c.SOURCE_SHA and job['physical_gpu_uuid']==c.GPU_UUID and job['training_step_gate']['sha256']==c.GATE_SHA
   assert job['soft_seconds']==plan['fit_bounds_by_cell'][job['cell']]['soft_seconds'] and entry['hard_seconds']==plan['fit_bounds_by_cell'][job['cell']]['hard_seconds']
  assert not (execution/'QUEUE_START.json').exists() and not (execution/'CURRENT_PROCESS.json').exists()
  blocks[block]=dict(execution_directory_relative=execution.name,queue_sha256=sha(execution/'QUEUE.json'),root_release_sha256=sha(execution/'ROOT_RELEASE.json'),provider_admission_sha256=sha(execution/'PROVIDER_ADMISSION.json'),donor_registration_sha256=sha(execution/'DONOR_REGISTRATION.json'),physical_GPU_UUID=c.GPU_UUID,ordered_cell_ids=expected_order,physical_fits=10,scientific_children_started=0)
  for path in [execution/n for n in ('QUEUE.json','ROOT_RELEASE.json','PROVIDER_ADMISSION.json','DONOR_REGISTRATION.json','COHORT_PLAN.json','EXTERNAL_ANCHORS.json')]+[phase/r['job_relative'] for r in q['entries']]+[receiptroot/(block+'.FREEZE_RECEIPT.json')]:
   raw=path.read_bytes();rel=str(path.relative_to(phase));artifacts[rel]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),zlib_base64=base64.b64encode(zlib.compress(raw,9)).decode())
 result=dict(UTC=datetime.now(timezone.utc).isoformat(),status='BOTH_EXACT_COMPLETE_BLOCKS_FROZEN_BEFORE_FITS',source_manifest_sha256=c.SOURCE_SHA,pilot_source_manifest_sha256=sha(prep/'SOURCE_MANIFEST.json'),cohort_plan_sha256=c.CANONICAL_PLAN_SHA,qualifier_sha256=c.GATE_SHA,blocks=blocks,scientific_children_started=0,fits=0,scores_read=False,artifacts=artifacts)
 save(receiptroot/'FROZEN_AUTHENTICATION.json',{{k:v for k,v in result.items() if k!='artifacts'}})
except Exception as e:
 result=dict(UTC=datetime.now(timezone.utc).isoformat(),status='OPERATIONAL_FREEZE_FAILURE',error=type(e).__name__+': '+str(e),partial_artifacts_preserved=True,scientific_children_started=0,fits=0,retry=False)
 save(receiptroot/'FREEZE_CONTROL_FAILURE.json',result)
print(json.dumps(result))
'''
r=t.run('private_transfer77_both_blocks_root_stage_freeze_20261005_v1',code)
save('FREEZE_TRANSPORT_RESULT.json',r)
if r['status']!='BOTH_EXACT_COMPLETE_BLOCKS_FROZEN_BEFORE_FITS':raise RuntimeError(r)
for rel,row in r['artifacts'].items():
 raw=zlib.decompress(base64.b64decode(row['zlib_base64'],validate=True));assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 path=HERE/'authenticated_remote'/rel;path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as f:f.write(raw)
save('FROZEN_AUTHENTICATION.json',{k:v for k,v in r.items() if k!='artifacts'})
print(json.dumps({k:v for k,v in r.items() if k!='artifacts'}))
