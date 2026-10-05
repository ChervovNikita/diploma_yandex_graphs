"""One normal detached queue per separately released complete block."""
from pathlib import Path
import hashlib,importlib.util,json
P=Path('/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930')
HERE=P/'shared_private_transfer_gpu77_block_launch_execution_root_20261005_v1'
PREP=P/'shared_private_transfer_gpu77_block_launch_preparation_20261005_v1'
RELEASE=P/'shared_private_transfer_gpu77_block_root_release_20261005_v1'
spec=importlib.util.spec_from_file_location('blocklaunch77_transport',P/'shared_private_transfer_gpu77_environment_execution_20261005_v1/remote_transport.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t);t.HERE=HERE
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):
 with (HERE/name).open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
freeze=json.loads((HERE/'FROZEN_AUTHENTICATION.json').read_text());assert freeze['status']=='BOTH_EXACT_COMPLETE_BLOCKS_FROZEN_BEFORE_FITS'
for block in ('b1','b2'):
 expected=freeze['blocks'][block]
 release=json.loads((RELEASE/('ROOT_RELEASE_'+block.upper()+'.json')).read_text())
 assert sha(RELEASE/('ROOT_RELEASE_'+block.upper()+'.json'))==expected['root_release_sha256']
 code=f'''
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,os,socket,subprocess,time
repo=Path({str(t.REPO)!r});phase=Path({str(t.REMOTE_PHASE)!r});prep=phase/{PREP.name!r};receiptroot=phase/{HERE.name!r};block={block!r};expected={expected!r}
assert Path.cwd()==repo and socket.gethostname()=='peptide'
control=receiptroot/(block+'.LAUNCH_RECEIPT.json');assert not control.exists()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
 with p.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write(chr(10))
process=None;owner=None;raw_owner=None;observed_cwd=None
try:
 spec=importlib.util.spec_from_file_location('block77_common',prep/'pilot_common.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
 c.bind_block([block]);c.physical_host();c.verify_packet('028f6f5d07e95f7edc58a61d3f6a7f39d1685403d46ab6bba431bb6072cf26c0');c.verify_training();s=c.supervisor()
 execution=phase/expected['execution_directory_relative'];queue=execution/'QUEUE.json';q=c.read(queue);release=c.read(execution/'ROOT_RELEASE.json')
 assert sha(queue)==expected['queue_sha256'] and sha(execution/'ROOT_RELEASE.json')==expected['root_release_sha256']
 assert sha(execution/'PROVIDER_ADMISSION.json')==expected['provider_admission_sha256'] and sha(execution/'DONOR_REGISTRATION.json')==expected['donor_registration_sha256']
 assert sha(phase/{str((RELEASE/'ROOT_REVIEW.md').relative_to(P))!r})=='f2efc428ba7fbdbff17d4fd96e30729c047b8301e61aff2221ac8347c2378b52'
 assert release['root_provider_block_approved'] is True and release['execution_blocks']==q['execution_blocks']==[block]
 assert c.GPU_UUID==expected['physical_GPU_UUID'] and q['environment_overrides']['CUDA_VISIBLE_DEVICES']==c.GPU_UUID
 assert not (execution/'QUEUE_START.json').exists() and not (execution/'QUEUE_FAILURE.json').exists() and not (execution/'CURRENT_PROCESS.json').exists()
 assert sha(execution/'COHORT_PLAN.json')==c.CANONICAL_PLAN_SHA
 for entry in q['entries']:assert sha(phase/entry['job_relative'])==entry['job_sha256']
 assert [entry['cell_id'] for entry in q['entries']]==expected['ordered_cell_ids'] and len(q['entries'])==10
 env=c.check_environment(q['environment_overrides']);assert Path(q['python_executable']).is_file()
 gpu=s.query(['--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],q['resource_limits']['telemetry_timeout_seconds'])
 assert tuple(r.split(',')[0].strip() for r in gpu)==c.GPU_UUIDS
 selected=next(r for r in gpu if r.split(',')[0].strip()==c.GPU_UUID);free=int(selected.split(',')[1].strip())*1024**2
 assert free>=q['resource_limits']['minimum_fresh_GPU_free_bytes']
 preflight=dict(UTC=s.now(),hostname=socket.gethostname(),block=block,GPU_UUID=c.GPU_UUID,GPU_free_bytes=free,minimum_fresh_GPU_free_bytes=q['resource_limits']['minimum_fresh_GPU_free_bytes'],GPU_metadata=gpu,exact_source_and_raw_gate_verified=True,queue_and_release_and_jobs_verified=True,provider_admission_and_registration_verified=True,queue_start_absent=True,scientific_children_started=0)
 save(receiptroot/(block+'.LAUNCH_PREFLIGHT.json'),preflight)
 argv=[q['python_executable'],'-B',str(prep/'run_queue.py'),'--queue',str(queue)]
 with (receiptroot/(block+'.queue.stdout.log')).open('xb') as out,(receiptroot/(block+'.queue.stderr.log')).open('xb') as err:
  process=subprocess.Popen(argv,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
  save(receiptroot/(block+'.POPEN_STARTED.json'),dict(UTC=s.now(),PID=process.pid,argv=argv,cwd=str(repo),block=block,queue_launches=1,retry=False))
 time.sleep(.2)
 raw_owner=s.identity(process.pid);observed_cwd=(Path('/proc')/str(process.pid)/'cwd').resolve(strict=True)
 assert raw_owner is not None and raw_owner['argv']==argv and raw_owner['pgid']==process.pid and raw_owner['sid']==process.pid and observed_cwd==repo.resolve()
 owner=raw_owner
 receipt=dict(UTC=s.now(),status='ONE_COMPLETE_ROOT_RELEASED_BLOCK_QUEUE_LAUNCHED',hostname=socket.gethostname(),block=block,physical_GPU_UUID=c.GPU_UUID,queue_identity=owner,raw_identity_observation=raw_owner,identity_admitted_for_signals=True,observed_cwd=str(observed_cwd),queue_sha256=sha(queue),root_release_sha256=sha(execution/'ROOT_RELEASE.json'),provider_admission_sha256=sha(execution/'PROVIDER_ADMISSION.json'),donor_registration_sha256=sha(execution/'DONOR_REGISTRATION.json'),pilot_source_manifest_sha256=sha(prep/'SOURCE_MANIFEST.json'),training_source_manifest_sha256=c.SOURCE_SHA,qualifier_sha256=c.GATE_SHA,cohort_plan_sha256=c.CANONICAL_PLAN_SHA,physical_fits=10,full_family_fits=30,queue_hard_seconds=q['queue_hard_seconds'],resource_limits=q['resource_limits'],GPU_free_bytes_at_preflight=free,detached_launches=1,attempts=1,retry=False,scores_read=False,TEST_access=False)
except Exception as e:
 receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),status='OPERATIONAL_BLOCK_LAUNCH_FAILURE',block=block,error=type(e).__name__+': '+str(e),queue_Popen_started=process is not None,queue_PID=process.pid if process is not None else None,queue_identity=owner,raw_identity_observation=raw_owner,observed_cwd=str(observed_cwd) if observed_cwd is not None else None,identity_admitted_for_signals=owner is not None,exit_code=process.poll() if process is not None else None,exit_code_authority='subprocess.Popen.poll' if process is not None and process.returncode is not None else None,partial_artifacts_preserved=True,retry=False,scores_read=False)
save(control,receipt);print(json.dumps(receipt))
'''
 r=t.run(f'private_transfer77_{block}_root_block_launch_once_20261005_v1',code)
 save(block+'.LAUNCH_RECEIPT.json',r);print(json.dumps(r))
 if r['status']!='ONE_COMPLETE_ROOT_RELEASED_BLOCK_QUEUE_LAUNCHED':raise RuntimeError(r)
