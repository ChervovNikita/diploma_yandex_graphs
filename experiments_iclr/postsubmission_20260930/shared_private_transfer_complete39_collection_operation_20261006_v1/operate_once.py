"""Execute the exact root-authorized collection once and gather only its 42 metadata files."""
from pathlib import Path
from datetime import datetime, timezone
import base64
import hashlib
import json
import shlex
import subprocess
import time
import zlib

HERE=Path(__file__).resolve().parent
P=HERE.parent
REQUEST=P/'shared_private_transfer_complete39_collection_root_request_preparation_20261006_v1'
AUTH=P/'shared_private_transfer_complete39_collection_root_authorization_20261006_v1/AUTHORIZATION.json'
COLLECTOR=P/'shared_private_transfer_allocation_replication_preparation_20261005_v2'
REPO='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'

def sha(b):return hashlib.sha256(b).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(name,value):
 p=HERE/name
 with p.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
 p.chmod(0o444)
def binding(p):
 b=p.read_bytes();return {'path':str(p.relative_to(P)),'bytes':len(b),'sha256':sha(b)}

auth=json.loads(AUTH.read_text())
assert auth['one_collection_execution_authorized'] is True and auth['freeze42_emitted_metadata_files_0444_authorized'] is True
assert auth['byte_custody_only'] is True and auth['history_score_semantics_tensor_held_D2_fit_GPU_job_actions_authorized'] is False
assert auth['retry_resume_donor_or_selector_change'] is False and auth['stop_after_actual_collection_freeze'] is True
request_bytes=(REQUEST/'ROOT_APPROVAL_REQUEST.json').read_bytes();assert sha(request_bytes)==auth['root_reviewed_request_sha256']
request=json.loads(request_bytes);stage_plan=json.loads((REQUEST/'STAGING_INVENTORY.json').read_text())
release=REQUEST/'ROOT_COLLECTION_RELEASE.json';body=release.read_bytes()
assert len(body)==59954 and sha(body)==auth['approved_release_sha256']=='b9375929fcc458d21a3309757b4ff515d26eaf37d6edc3ff32a70159a03b065e'
assert release.stat().st_mode&0o222==0 and json.loads(body)['root_collection_approved'] is True
assert not (HERE/'COLLECTOR_LAUNCH_INTENT.json').exists()
payload=base64.b64encode(zlib.compress(json.dumps({'release_base64':base64.b64encode(body).decode()},sort_keys=True).encode(),9))+b'\n'
assert len(payload)==stage_plan['expected_stdin_bytes'] and sha(payload)==stage_plan['expected_stdin_sha256']
with (HERE/'STAGING_STDIN_COMPRESSED_BASE64.txt').open('xb') as f:f.write(payload)
(HERE/'STAGING_STDIN_COMPRESSED_BASE64.txt').chmod(0o444)
write('STAGING_COMMAND.json',{'schema':'complete39_collection_exact_staging_command_v1','UTC':now(),'authorization':binding(AUTH),
 'argv':stage_plan['argv'],'stdin_payload':binding(HERE/'STAGING_STDIN_COMPRESSED_BASE64.txt'),'remote_source':stage_plan['remote_source'],'staged_file_count':1})
t=time.monotonic();s=subprocess.run(stage_plan['argv'],input=payload.decode(),capture_output=True,text=True,timeout=55)
stage={'schema':'complete39_collection_staging_transport_receipt_v1','UTC':now(),'exit_code':s.returncode,'elapsed_seconds':time.monotonic()-t,'stderr':s.stderr}
if s.returncode==0:stage['result']=json.loads(s.stdout)
else:stage['stdout']=s.stdout
write('STAGING_RECEIPT.json',stage)
assert s.returncode==0,'Staging failed; no collector launched. Preserve evidence without automatic retry.'
write('COLLECTOR_LAUNCH_INTENT.json',{'schema':'complete39_exact_once_only_collection_launch_intent_v1','UTC':now(),'authorization':binding(AUTH),
 'release':binding(release),'request':binding(REQUEST/'ROOT_APPROVAL_REQUEST.json'),'argv':request['remote_invocation_argv'],
 'collector_argv':request['collector_argv'],'invocation_count_limit':1,'retry_or_resume':False,'GPU_jobs_launched_or_stopped':0})
t=time.monotonic();started=now();proc=subprocess.Popen(request['remote_invocation_argv'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
write('LOCAL_SSH_PROCESS_STARTED.json',{'schema':'complete39_once_only_collection_local_ssh_process_v1','UTC':started,'PID':proc.pid,
 'argv':request['remote_invocation_argv'],'monotonic_started_seconds':t})
print(json.dumps({'stage_exit':0,'collector_invoked_once':True,'local_ssh_PID':proc.pid}),flush=True)
# Await this exact foreground SSH process without timeout, signals, retry or resumption.
stdout,stderr=proc.communicate()
exit_record={'schema':'complete39_once_only_collection_exit_receipt_v1','UTC':now(),'started_UTC':started,'local_ssh_PID':proc.pid,
 'exit_code':proc.returncode,'terminal_wait_observed':True,'elapsed_seconds':time.monotonic()-t,'stdout':stdout,'stderr':stderr,
 'collector_invocations':1,'retry_or_resume':False,'external_timeout_or_signals_sent':False,
 'history_semantics_tensor_scores_held_D2_model_fit_or_GPU_job_actions':False}
write('COLLECTOR_EXIT.json',exit_record)
print(json.dumps({'collector_exit':proc.returncode,'elapsed_seconds':exit_record['elapsed_seconds']}),flush=True)
assert proc.returncode==0,'Actual collector failed; preserve fresh output and inspect same state. Do not rerun.'
amendment_bytes=(COLLECTOR/'PROSPECTIVE_AMENDMENT.json').read_bytes();assert sha(amendment_bytes)=='d965a699bc9d150470ea2b4b0fc0c07a4ffd90139230db065e0473b0311d7e2d'
cell_ids=json.loads(amendment_bytes)['selected39_order'];assert len(cell_ids)==len(set(cell_ids))==39
names=['COLLECTION_RELEASE.json','ATTEMPT_HISTORY.json','COLLECTION_FREEZE.json']+['provider_custody_'+x+'.json' for x in cell_ids]
assert len(names)==42
GATHER=r'''
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,socket,subprocess
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).split();assert uuids==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
root=phase/OUTPUT_TEXT;names=json.loads(NAMES_TEXT);order=json.loads(ORDER_TEXT)
assert root.resolve(strict=True).is_relative_to(phase) and not root.is_symlink() and len(names)==42
assert {p.name for p in root.iterdir()}==set(names)
freeze=root/'COLLECTION_FREEZE.json';registry=json.loads(freeze.read_text())
assert registry['schema']=='authenticated_complete_prospective_allocation_replication39_v1' and registry['complete'] is True and registry['full39_terminal_source_artifact_custody_passed'] is True
assert registry['selected_logical_scientific_fits']==39 and registry['selected_physical_fits']==39 and registry['new_physical_fits']==26
assert [r['cell_id'] for r in registry['completed']]==order
for key in ('comparative_scoring_performed','TEST_access','fits_authorized','history_bytes_observed','history_JSON_parsed','FREEZE_JSON_parsed','CONFIG_JSON_parsed','selected_prediction_payloads_deserialized','quality_fields_accessed_or_emitted','automatic_retry'):assert registry[key] is False
assert registry['collection_release_sha256']=='b9375929fcc458d21a3309757b4ff515d26eaf37d6edc3ff32a70159a03b065e'
assert registry['collector_manifest_sha256']=='c29ccad053cfcaa4c11499854fdaca962ade48a910efa82435241d7d2f3e2ec6'
assert hashlib.sha256((root/'COLLECTION_RELEASE.json').read_bytes()).hexdigest()==registry['collection_release_sha256']
original_history=phase/'shared_private_transfer_allocation_replication_preparation_20261005_v2/ATTEMPT_HISTORY.json'
assert hashlib.sha256(original_history.read_bytes()).hexdigest()==hashlib.sha256((root/'ATTEMPT_HISTORY.json').read_bytes()).hexdigest()=='e3c70e8b22f71075078f376a82ec2534cb89d07cd508c4e2a2361dc2c0484970'
for name in names:
 p=root/name;assert p.suffix=='.json' and p.name not in ('FREEZE.json','CONFIG.json','VALID_HISTORY.jsonl') and p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root) and p.stat().st_size<1_000_000
files=[]
for name in names:
 p=root/name;before=hashlib.sha256(p.read_bytes()).hexdigest();mode_before=oct(p.stat().st_mode&0o777)
 p.chmod(0o444)
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==before and p.stat().st_mode&0o777==0o444
 files.append({'path':str(p.relative_to(phase)),'bytes':len(b),'sha256':before,'mode_before':mode_before,'mode':'0444','utf8':b.decode()})
print(json.dumps({'schema':'actual_complete39_collection_metadata_freeze_and_gather_v1','UTC':datetime.now(timezone.utc).isoformat(),'hostname':socket.gethostname(),'GPU_UUIDs':uuids,
 'repository':str(repo),'files':files,'JSON_metadata_file_count':42,'complete39_registry':{'path':str(freeze.relative_to(phase)),'sha256':hashlib.sha256(freeze.read_bytes()).hexdigest()},
 'original_UNKNOWN_history_preserved':True,'scientific_artifact_or_directory_chmod':False,'history_or_payload_semantics_tensor_score_held_D2_model_or_fit_execution':False,'GPU_jobs_launched_or_stopped':0}))
'''.replace('OUTPUT_TEXT',repr(request['output_phase_relative'])).replace('NAMES_TEXT',repr(json.dumps(names))).replace('ORDER_TEXT',repr(json.dumps(cell_ids)))
with (HERE/'METADATA_FREEZE_GATHER_REMOTE_SOURCE.py').open('x') as f:f.write(GATHER)
(HERE/'METADATA_FREEZE_GATHER_REMOTE_SOURCE.py').chmod(0o444)
gather_payload=base64.b64encode(zlib.compress(GATHER.encode(),9)).decode()+'\n'
with (HERE/'METADATA_GATHER_STDIN_COMPRESSED_BASE64.txt').open('x') as f:f.write(gather_payload)
(HERE/'METADATA_GATHER_STDIN_COMPRESSED_BASE64.txt').chmod(0o444)
bootstrap="import sys,zlib,base64;exec(compile(zlib.decompress(base64.b64decode(sys.stdin.buffer.read())), '<complete39-metadata-freeze-gather>', 'exec'))"
gather_argv=request['remote_invocation_argv'][:-1]+['cd '+shlex.quote(REPO)+' && exec /usr/bin/python3 -I -S -B -c '+shlex.quote(bootstrap)]
t=time.monotonic();g=subprocess.run(gather_argv,input=gather_payload,capture_output=True,text=True,timeout=55)
gather={'schema':'complete39_collection_metadata_freeze_gather_transport_v1','UTC':now(),'exit_code':g.returncode,'elapsed_seconds':time.monotonic()-t,'stderr':g.stderr,
 'argv':gather_argv,'remote_source_sha256':sha(GATHER.encode()),'stdin_payload':binding(HERE/'METADATA_GATHER_STDIN_COMPRESSED_BASE64.txt'),'authorization':binding(AUTH)}
if g.returncode==0:gather['result']=json.loads(g.stdout)
else:gather['stdout']=g.stdout
write('METADATA_FREEZE_GATHER_RECEIPT.json',gather)
assert g.returncode==0,'Actual collection succeeded; metadata freeze/gather failed. Preserve state, never rerun collector.'
observed={}
for row in gather['result']['files']:
 b=row['utf8'].encode();assert len(b)==row['bytes'] and sha(b)==row['sha256']
 name=Path(row['path']).name;observed[name]=json.loads(b)
 local=HERE/'emitted_metadata'/name;local.parent.mkdir(exist_ok=True)
 with local.open('xb') as f:f.write(b)
 local.chmod(0o444)
registry=observed['COLLECTION_FREEZE.json'];collection=observed['COLLECTION_RELEASE.json']
assert collection['root_collection_approved'] is True and registry['complete'] is True
assert len(registry['completed'])==39 and len(observed)==42
assert sha((HERE/'emitted_metadata/COLLECTION_RELEASE.json').read_bytes())==sha(body)
assert sha((HERE/'emitted_metadata/ATTEMPT_HISTORY.json').read_bytes())=='e3c70e8b22f71075078f376a82ec2534cb89d07cd508c4e2a2361dc2c0484970'
candidate=json.loads((REQUEST/'COLLECTION_RELEASE_DISABLED.json').read_text());candidate['root_collection_approved']=True;assert collection==candidate
for row in registry['completed']:
 descriptor=observed['provider_custody_'+row['cell_id']+'.json'];assert descriptor['provider']=='authorized_one_GPU_allocation'
 assert descriptor['actual_donor']['donor_directory_relative']==row['original_donor_directory_relative']
write('COMPLETION.json',{'schema':'actual_complete39_once_only_collection_completion_v1','UTC':now(),'status':'SUCCESS_STOPPED_AFTER_ACTUAL_COLLECTION_FREEZE',
 'authorization':binding(AUTH),'collector_exit':binding(HERE/'COLLECTOR_EXIT.json'),'staging_receipt':binding(HERE/'STAGING_RECEIPT.json'),
 'metadata_freeze_gather_receipt':binding(HERE/'METADATA_FREEZE_GATHER_RECEIPT.json'),'complete39_registry':gather['result']['complete39_registry'],
 'local_registry':binding(HERE/'emitted_metadata/COLLECTION_FREEZE.json'),'emitted_JSON_metadata_files':42,'all_emitted_metadata_0444':True,
 'collector_invocations':1,'exit_code':0,'terminal_wait_observed':True,'retry_or_resume':False,'selected_logical_and_physical_fits':39,'new_physical_fits':26,
 'scientific_artifact_BYTE_custody_only':True,'scientific_artifact_copies':0,'history_or_tensor_semantics_scores_held_TEST_D2_model_fit_or_GPU_job_actions':False,
 'original_UNKNOWN_history_preserved':True,'excluded_old20_GPU77_donors_unchanged':True,'source_and_numerical_selectors_unchanged':True,
 'cost_seconds':{'staging':stage['elapsed_seconds'],'collector':exit_record['elapsed_seconds'],'metadata_freeze_gather':gather['elapsed_seconds']},
 'next_gate':'Separate root authorization for operativeV2 all39 history BYTE inventory; no history or D2 release enabled here.'})
print(json.dumps({'status':'SUCCESS_STOPPED_AFTER_ACTUAL_COLLECTION_FREEZE','complete39_registry':gather['result']['complete39_registry'],
 'registry_bytes':(HERE/'emitted_metadata/COLLECTION_FREEZE.json').stat().st_size,'metadata_files':42,'collector_elapsed_seconds':exit_record['elapsed_seconds']}),flush=True)
